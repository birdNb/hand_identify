#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 5 跟手: 位置误差 > 阈值 → ±CMD_MAG; 手静止不发指令; 脖子跟手检测抬头。"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Int32

# ----- ROS -----
CMD_VEL_TOPIC = "/cmd_vel"
JOY_TOPIC = "/joy"
FSM_STATE_TOPIC = "/fsm_state"
JOY_ACTIVE_THRESH = 0.15
JOY_IDLE_SEC = 3.0
# Xbox LT/RT 轴：松开≈+1.0，按下≈-1.0；不能按 |axis|>阈值 判为手柄占用
JOY_TRIGGER_AXIS_IDS = (2, 5)
JOY_TRIGGER_REST = 1.0
JOY_TRIGGER_ACTIVE_MARGIN = 0.35
FSM_EXEC_DEFAULT = 5
PUBLISH_RATE_HZ = 20

# ----- 跟手目标 -----
TARGET_DISTANCE_M = 1.0
POS_THRESH_M = 0.3
CMD_MAG = 0.3
LATERAL_ROTATE_THRESH_M = 0.5
HAND_MOVE_THRESH_M = 0.04

# ----- 脖子 (与 locate_face 一致: pitch 负=抬头) -----
ABSOLUTE_TOPIC = "/pi_plus_absolute"
HEAD_YAW_JOINT = "head_yaw_joint"
HEAD_PITCH_JOINT = "head_pitch_joint"
HAND_FEEDBACK_PITCH_DEG = -10.0    # 反馈抬头 10° (pitch 负=抬头)
HAND_FEEDBACK_RAMP_SEC = 1.0       # 1 秒内抬到目标后停止
NECK_RETURN_RAMP_DEG_PER_SEC = 8.0 # 无手后回中角速度
NECK_PUBLISH_RATE_HZ = 50

CMD_VALID_SEC = 0.5
GESTURE_FOLLOW = 5


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bang_cmd(
    error_m: float,
    thresh_m: float = POS_THRESH_M,
    magnitude: float = CMD_MAG,
) -> float:
    """误差超过阈值 → ±magnitude, 否则 0。"""
    if abs(error_m) <= thresh_m:
        return 0.0
    return magnitude if error_m > 0 else -magnitude


class HandMotionGate:
    """手不动不发指令: 帧间掌心位移超过阈值才允许底盘控制。"""

    def __init__(self, move_thresh_m: float = HAND_MOVE_THRESH_M):
        self.move_thresh_m = move_thresh_m
        self._prev = None
        self.is_moving = False

    def reset(self):
        self._prev = None
        self.is_moving = False

    def update(self, palm_pos: Optional[Tuple[float, float, float]]) -> bool:
        if palm_pos is None:
            self.reset()
            return False
        if self._prev is None:
            self._prev = palm_pos
            self.is_moving = False
            return False
        dx = palm_pos[0] - self._prev[0]
        dy = palm_pos[1] - self._prev[1]
        dz = palm_pos[2] - self._prev[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        self._prev = palm_pos
        self.is_moving = dist >= self.move_thresh_m
        return self.is_moving


@dataclass
class HandControlInput:
    gesture: int = -1
    distance_m: float = 0.0
    palm_x_m: float = 0.0
    palm_y_m: float = 0.0
    active: bool = False


@dataclass
class HandControlOutput:
    """控制指令 X/Y/Z ∈ {-1, 0, +1} → cmd_vel linear.x / linear.y / angular.z"""
    cmd_x: float = 0.0
    cmd_y: float = 0.0
    cmd_z: float = 0.0
    mode: str = "idle"


class HandFollowController:
    def __init__(
        self,
        target_distance_m=TARGET_DISTANCE_M,
        pos_thresh_m=POS_THRESH_M,
        rotate_thresh_m=LATERAL_ROTATE_THRESH_M,
        cmd_mag: float = CMD_MAG,
    ):
        self.target_distance_m = target_distance_m
        self.pos_thresh_m = pos_thresh_m
        self.rotate_thresh_m = rotate_thresh_m
        self.cmd_mag = cmd_mag
        self.last_out = HandControlOutput()

    def reset(self):
        self.last_out = HandControlOutput()

    def compute(self, inp: HandControlInput) -> HandControlOutput:
        if not inp.active or inp.gesture != GESTURE_FOLLOW:
            self.reset()
            return self.last_out

        # X: 深度 Z 与目标距离误差 → 前后 (linear.x)
        e_depth = inp.distance_m - self.target_distance_m
        mag = self.cmd_mag
        cmd_x = bang_cmd(e_depth, self.pos_thresh_m, mag)

        e_x = inp.palm_x_m
        e_y = inp.palm_y_m

        if abs(e_x) > self.rotate_thresh_m:
            cmd_y = 0.0
            cmd_z = bang_cmd(-e_x, self.pos_thresh_m, mag)
            mode = "rotate"
        else:
            cmd_y = bang_cmd(e_x, self.pos_thresh_m, mag)
            cmd_z = bang_cmd(-e_y, self.pos_thresh_m, mag)
            mode = "track"

        out = HandControlOutput(cmd_x=cmd_x, cmd_y=cmd_y, cmd_z=cmd_z, mode=mode)
        self.last_out = out
        return out


class JoyMonitor:
    def __init__(
        self,
        topic: str = JOY_TOPIC,
        active_thresh: float = JOY_ACTIVE_THRESH,
        idle_sec: float = JOY_IDLE_SEC,
    ):
        self._lock = threading.Lock()
        self._active_thresh = active_thresh
        self._idle_sec = idle_sec
        self._last_active_t = 0.0
        self._sub = rospy.Subscriber(topic, Joy, self._cb, queue_size=10)

    def _axis_active(self, idx: int, val: float) -> bool:
        v = float(val)
        if idx in JOY_TRIGGER_AXIS_IDS:
            # 仅扳机按下(明显低于松开位 1.0) 才算手柄输入
            return v < (JOY_TRIGGER_REST - JOY_TRIGGER_ACTIVE_MARGIN)
        return abs(v) > self._active_thresh

    def _axes_buttons_active(self, msg: Joy) -> bool:
        for i, ax in enumerate(msg.axes):
            if self._axis_active(i, ax):
                return True
        for btn in msg.buttons:
            if int(btn) != 0:
                return True
        return False

    def _cb(self, msg: Joy):
        if self._axes_buttons_active(msg):
            with self._lock:
                self._last_active_t = time.time()

    def allow_program_cmd(self) -> bool:
        with self._lock:
            if self._last_active_t <= 0:
                return True
            return (time.time() - self._last_active_t) >= self._idle_sec

    def idle_remaining(self) -> float:
        with self._lock:
            if self._last_active_t <= 0:
                return 0.0
            return max(0.0, self._idle_sec - (time.time() - self._last_active_t))


class VelCommand:
    def __init__(self):
        self._lock = threading.Lock()
        self._cmd_x = 0.0
        self._cmd_y = 0.0
        self._cmd_z = 0.0
        self._t = time.time()
        self._stale_after = 0.0

    def set(self, cmd_x: float, cmd_y: float, cmd_z: float,
            valid_for_sec: float = CMD_VALID_SEC):
        with self._lock:
            self._cmd_x = cmd_x
            self._cmd_y = cmd_y
            self._cmd_z = cmd_z
            self._t = time.time()
            self._stale_after = valid_for_sec

    def get(self):
        with self._lock:
            if self._stale_after > 0 and time.time() - self._t > self._stale_after:
                return 0.0, 0.0, 0.0, True
            return self._cmd_x, self._cmd_y, self._cmd_z, False

    def stop(self):
        self.set(0.0, 0.0, 0.0, valid_for_sec=0.0)


class FsmStateMonitor:
    _NAME_MAP = {
        0: "INIT", 1: "ERROR",
        2: "CANDIDATE_DEFAULT", 3: "CANDIDATE_CUSTOM",
        4: "CANDIDATE_REMOTE",
        5: "EXEC_DEFAULT", 6: "EXEC_CUSTOM", 7: "EXEC_REMOTE",
        8: "PROTECTION_SHUTDOWN",
        9: "CANDIDATE_CALIBRATION", 10: "EXEC_CALIBRATING",
        11: "EXEC_CALIB_OK", 12: "EXEC_CALIB_FAILED",
        13: "CANDIDATE_TEACHING", 14: "EXEC_TEACHING",
        15: "CANDIDATE_DEVELOP", 16: "EXEC_DEVELOP",
    }

    def __init__(self, topic: str = FSM_STATE_TOPIC):
        self._lock = threading.Lock()
        self._state = None
        self._sub = rospy.Subscriber(topic, Int32, self._cb, queue_size=10)

    def _cb(self, msg):
        with self._lock:
            self._state = int(msg.data)

    @property
    def state(self):
        with self._lock:
            return self._state

    @classmethod
    def state_name(cls, v) -> str:
        return cls._NAME_MAP.get(v, f"UNKNOWN({v})")

    def is_exec_default(self) -> bool:
        return self.state == FSM_EXEC_DEFAULT

    def wait_for_exec_default(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while not rospy.is_shutdown():
            if self.state == FSM_EXEC_DEFAULT:
                return True
            if timeout > 0 and time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return False


class NeckTarget:
    def __init__(self):
        self._lock = threading.Lock()
        self._yaw = 0.0
        self._pitch = 0.0

    def set(self, yaw_rad: float, pitch_rad: float):
        with self._lock:
            self._yaw = yaw_rad
            self._pitch = pitch_rad

    def get(self):
        with self._lock:
            return self._yaw, self._pitch


def _step_toward(cur: float, goal: float, max_step: float) -> float:
    if abs(goal - cur) <= max_step:
        return goal
    return cur + max_step if goal > cur else cur - max_step


class NeckController(threading.Thread):
    """
    识别到手：1 秒内抬头 10° 后保持（反馈），手离开再回中。
  与 locate_face 同时运行时勿调用（由 hand_follow 在脸跟踪开启时禁用）。
    """

    def __init__(
        self,
        fsm: Optional[FsmStateMonitor],
        dry_run: bool,
        pitch_up_deg: float = HAND_FEEDBACK_PITCH_DEG,
        feedback_ramp_sec: float = HAND_FEEDBACK_RAMP_SEC,
        return_ramp_deg_s: float = NECK_RETURN_RAMP_DEG_PER_SEC,
    ):
        super().__init__(daemon=True)
        self._fsm = fsm
        self._dry_run = dry_run
        self._pitch_up_rad = math.radians(pitch_up_deg)
        self._feedback_ramp_sec = max(0.1, float(feedback_ramp_sec))
        self._feedback_ramp_rad_s = abs(self._pitch_up_rad) / self._feedback_ramp_sec
        self._return_ramp_rad_s = math.radians(return_ramp_deg_s)
        self._lock = threading.Lock()
        self._phase = "idle"
        self._hand_prev = False
        self._feedback_deadline = 0.0
        self._goal_yaw = 0.0
        self._goal_pitch = 0.0
        self._cur_yaw = 0.0
        self._cur_pitch = 0.0
        self._stop_evt = threading.Event()
        self._pub = rospy.Publisher(ABSOLUTE_TOPIC, JointState, queue_size=10)
        self._rate = rospy.Rate(NECK_PUBLISH_RATE_HZ)

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def update_hand(self, detected: bool):
        """手出现上升沿触发一次抬头反馈；手离开则回中。"""
        with self._lock:
            rising = detected and not self._hand_prev
            falling = not detected and self._hand_prev
            self._hand_prev = detected
            self._goal_yaw = 0.0

            if rising:
                self._phase = "feedback"
                self._feedback_deadline = (
                    time.time() + self._feedback_ramp_sec
                )
                self._goal_pitch = self._pitch_up_rad
            elif falling and self._phase in ("feedback", "hold"):
                self._phase = "return"
                self._goal_pitch = 0.0
            elif self._phase == "hold":
                self._goal_pitch = self._pitch_up_rad
            elif self._phase == "return":
                self._goal_pitch = 0.0
            elif self._phase == "idle":
                self._goal_pitch = 0.0

    def set_hand_detected(self, detected: bool):
        """兼容旧接口。"""
        self.update_hand(detected)

    def stop(self):
        self._stop_evt.set()

    def publish_center_blocking(self, duration: float = 0.5):
        with self._lock:
            self._goal_yaw = 0.0
            self._goal_pitch = 0.0
            self._cur_yaw = 0.0
            self._cur_pitch = 0.0
        msg = JointState()
        msg.name = [HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
        msg.position = [0.0, 0.0]
        msg.velocity = []
        msg.effort = []
        end_t = time.time() + duration
        while time.time() < end_t:
            if not self._dry_run:
                msg.header.stamp = rospy.Time.now()
                self._pub.publish(msg)
            time.sleep(1.0 / max(NECK_PUBLISH_RATE_HZ, 1))

    def run(self):
        msg = JointState()
        msg.name = [HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
        msg.velocity = []
        msg.effort = []
        dt = 1.0 / max(NECK_PUBLISH_RATE_HZ, 1)
        pitch_eps = math.radians(0.5)
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            if self._fsm is not None and not self._fsm.is_exec_default():
                self._rate.sleep()
                continue
            with self._lock:
                phase = self._phase
                goal_pitch = self._goal_pitch
                if phase == "feedback":
                    ramp_step = self._feedback_ramp_rad_s * dt
                elif phase == "return":
                    ramp_step = self._return_ramp_rad_s * dt
                else:
                    ramp_step = self._return_ramp_rad_s * dt

                self._cur_yaw = _step_toward(
                    self._cur_yaw, self._goal_yaw, ramp_step,
                )
                self._cur_pitch = _step_toward(
                    self._cur_pitch, goal_pitch, ramp_step,
                )
                if phase == "feedback":
                    reached = abs(self._cur_pitch - goal_pitch) <= pitch_eps
                    if reached or time.time() >= self._feedback_deadline:
                        self._phase = "hold"
                        self._cur_pitch = goal_pitch
                elif phase == "return":
                    if abs(self._cur_pitch) <= pitch_eps and abs(goal_pitch) < 1e-6:
                        self._phase = "idle"
                        self._cur_yaw = 0.0
                        self._cur_pitch = 0.0
                yaw, pitch = self._cur_yaw, self._cur_pitch
            msg.position = [yaw, pitch]
            msg.header.stamp = rospy.Time.now()
            if not self._dry_run:
                self._pub.publish(msg)
            self._rate.sleep()


class CmdVelPublisher(threading.Thread):
    def __init__(self, vel: VelCommand, fsm: Optional[FsmStateMonitor], dry_run: bool):
        super().__init__(daemon=True)
        self._vel = vel
        self._fsm = fsm
        self._dry_run = dry_run
        self._pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
        if not self._dry_run:
            t0 = time.time()
            while (
                self._pub.get_num_connections() == 0
                and not rospy.is_shutdown()
                and time.time() - t0 < 5.0
            ):
                time.sleep(0.05)
            if self._pub.get_num_connections() == 0:
                rospy.logwarn(
                    "[cmd_vel] 尚无 /cmd_vel 订阅者, 跟手可能无效",
                )
        self._rate = rospy.Rate(PUBLISH_RATE_HZ)
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def publish_stop_blocking(self, duration: float = 0.5):
        msg = Twist()
        end_t = time.time() + duration
        while time.time() < end_t:
            if not self._dry_run:
                self._pub.publish(msg)
            time.sleep(1.0 / max(PUBLISH_RATE_HZ, 1))

    def run(self):
        msg = Twist()
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            cx, cy, cz, stale = self._vel.get()
            fsm_ok = (self._fsm is None) or self._fsm.is_exec_default()
            if not fsm_ok or stale:
                cx, cy, cz = 0.0, 0.0, 0.0
            msg.linear.x = cx
            msg.linear.y = cy
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = cz
            if not self._dry_run:
                self._pub.publish(msg)
            self._rate.sleep()
