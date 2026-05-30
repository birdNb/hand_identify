#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FSM / 手柄仲裁。手势动作用 JOY_IDLE_SEC；手部跟踪用 HAND_TRACKING_JOY_IDLE_SEC。"""

import math
import threading
import time
from typing import Optional, Tuple

import rospy
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Int32

# ----- ROS -----
JOY_TOPIC = "/joy"
FSM_STATE_TOPIC = "/fsm_state"
JOY_ACTIVE_THRESH = 0.15
JOY_IDLE_SEC = 3.0
HAND_TRACKING_JOY_IDLE_SEC = 5.0
# Xbox LT/RT 轴：松开≈+1.0，按下≈-1.0；不能按 |axis|>阈值 判为手柄占用
JOY_TRIGGER_AXIS_IDS = (2, 5)
JOY_TRIGGER_REST = 1.0
JOY_TRIGGER_ACTIVE_MARGIN = 0.35
FSM_EXEC_DEFAULT = 5
# ----- 脖子 (与 locate_face 一致: pitch 负=抬头) -----
ABSOLUTE_TOPIC = "/pi_plus_absolute"
HEAD_YAW_JOINT = "head_yaw_joint"
HEAD_PITCH_JOINT = "head_pitch_joint"
GESTURE_NOD_PITCH_DEG = -20.0      # NeckController 备用 (pitch 负=抬头)
GESTURE_NOD_RAMP_SEC = 1.0         # 1 秒内抬到目标
NECK_RETURN_RAMP_DEG_PER_SEC = 8.0 # 复原回中角速度
NECK_PUBLISH_RATE_HZ = 50

class JoyMonitor:
    """
    监听物理手柄 /joy。有输入时 block 手势动作库/点头；脸跟踪不受影响。
  手柄松开后需空闲 idle_sec 才恢复手势控制。
    """

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
        self._was_blocking = False
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

    def is_active_now(self) -> bool:
        """当前是否检测到手柄输入（未等空闲计时）。"""
        with self._lock:
            if self._last_active_t <= 0:
                return False
            return (time.time() - self._last_active_t) < self._idle_sec

    def blocks_hand_tracking(self) -> bool:
        """手部跟踪应停止发布 /cmd_vel（与 blocks_gesture_control 相同逻辑）。"""
        return not self.allow_program_cmd()

    def blocks_gesture_control(self) -> bool:
        """为 True 时手势动作库应让路给手柄。"""
        return not self.allow_program_cmd()

    def allow_program_cmd(self) -> bool:
        with self._lock:
            if self._last_active_t <= 0:
                return True
            return (time.time() - self._last_active_t) >= self._idle_sec

    def poll_takeover_edge(self) -> bool:
        """手柄刚接管控制时返回 True（上升沿，每轮循环调一次）。"""
        blocking = self.blocks_gesture_control()
        edge = blocking and not self._was_blocking
        self._was_blocking = blocking
        return edge

    def idle_remaining(self) -> float:
        with self._lock:
            if self._last_active_t <= 0:
                return 0.0
            return max(0.0, self._idle_sec - (time.time() - self._last_active_t))


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

    def wait_for_exec_default(
        self,
        timeout: float = 30.0,
        *,
        should_stop=None,
    ) -> bool:
        t0 = time.time()
        while not rospy.is_shutdown():
            if should_stop is not None and should_stop():
                return False
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
    """脖子 pitch 抬头后复原（勿用于手势1，与脸跟踪冲突）。"""

    def __init__(
        self,
        fsm: Optional[FsmStateMonitor],
        dry_run: bool,
        pitch_up_deg: float = GESTURE_NOD_PITCH_DEG,
        nod_ramp_sec: float = GESTURE_NOD_RAMP_SEC,
        return_ramp_deg_s: float = NECK_RETURN_RAMP_DEG_PER_SEC,
    ):
        super().__init__(daemon=True)
        self._fsm = fsm
        self._dry_run = dry_run
        self._pitch_up_rad = math.radians(pitch_up_deg)
        self._nod_ramp_sec = max(0.1, float(nod_ramp_sec))
        self._nod_ramp_rad_s = abs(self._pitch_up_rad) / self._nod_ramp_sec
        self._return_ramp_rad_s = math.radians(return_ramp_deg_s)
        self._lock = threading.Lock()
        self._phase = "idle"
        self._nod_deadline = 0.0
        self._hold_yaw = 0.0
        self._goal_pitch = 0.0
        self._cur_pitch = 0.0
        self._stop_evt = threading.Event()
        self._pub = rospy.Publisher(ABSOLUTE_TOPIC, JointState, queue_size=10)
        self._rate = rospy.Rate(NECK_PUBLISH_RATE_HZ)

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._phase in ("feedback", "return")

    def abort_gesture_nod(self) -> bool:
        """手柄接管等场景下中止手势1点头。"""
        with self._lock:
            if self._phase == "idle":
                return False
            self._phase = "idle"
            self._goal_pitch = 0.0
            return True

    def trigger_gesture_nod(
        self,
        hold_yaw_rad: float = 0.0,
        start_pitch_rad: float = 0.0,
    ) -> bool:
        """手势 1：仅 pitch 抬头 20° 再回中；yaw 锁定不变。忙时返回 False。"""
        with self._lock:
            if self._phase != "idle":
                return False
            self._phase = "feedback"
            self._nod_deadline = time.time() + self._nod_ramp_sec
            self._hold_yaw = float(hold_yaw_rad)
            self._cur_pitch = float(start_pitch_rad)
            self._goal_pitch = self._pitch_up_rad
            return True

    def stop(self):
        self._stop_evt.set()

    def publish_center_blocking(self, duration: float = 0.5):
        with self._lock:
            hold_yaw = self._hold_yaw
            self._goal_pitch = 0.0
            self._cur_pitch = 0.0
            self._phase = "idle"
        msg = JointState()
        msg.name = [HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
        msg.position = [hold_yaw, 0.0]
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
                    ramp_step = self._nod_ramp_rad_s * dt
                elif phase == "return":
                    ramp_step = self._return_ramp_rad_s * dt
                else:
                    ramp_step = 0.0

                if phase != "idle":
                    self._cur_pitch = _step_toward(
                        self._cur_pitch, goal_pitch, ramp_step,
                    )
                if phase == "feedback":
                    reached = abs(self._cur_pitch - goal_pitch) <= pitch_eps
                    if reached or time.time() >= self._nod_deadline:
                        self._phase = "return"
                        self._goal_pitch = 0.0
                elif phase == "return":
                    if abs(self._cur_pitch) <= pitch_eps and abs(goal_pitch) < 1e-6:
                        self._phase = "idle"
                yaw = self._hold_yaw
                pitch = self._cur_pitch
            if phase != "idle" and not self._dry_run:
                msg.position = [yaw, pitch]
                msg.header.stamp = rospy.Time.now()
                self._pub.publish(msg)
            self._rate.sleep()
