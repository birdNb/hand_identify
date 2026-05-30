#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五指手势(手势5)底盘跟手 —— 默认不启用，备份于此。

全轴跟手备份模块（linear.x/y + angular.z），当前未接入启动脚本。
模块名以数字开头，需用 importlib 加载::

    import importlib.util, os
    _p = os.path.join(os.path.dirname(__file__), "locomotion.py")
    _spec = importlib.util.spec_from_file_location("finger_locomotion", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FingerLocomotionStack = _mod.FingerLocomotionStack

原逻辑：手势5 + 有效距离 + 手在动 → 发布 /cmd_vel (linear.x/y, angular.z)
"""

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(tracking=True)

import rospy
from geometry_msgs.msg import Twist

from ros_control import FsmStateMonitor

# ----- 跟手参数 -----
CMD_VEL_TOPIC = "/cmd_vel"
TARGET_DISTANCE_M = 1.0
POS_THRESH_M = 0.3
CMD_MAG = 0.3
LATERAL_ROTATE_THRESH_M = 0.5
HAND_MOVE_THRESH_M = 0.04
CMD_VALID_SEC = 0.5
GESTURE_FOLLOW = 5
LOCOMOTION_PUBLISH_RATE_HZ = 20


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bang_cmd(
    error_m: float,
    thresh_m: float = POS_THRESH_M,
    magnitude: float = CMD_MAG,
) -> float:
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


class VelCommand:
    def __init__(self):
        self._lock = threading.Lock()
        self._cmd_x = 0.0
        self._cmd_y = 0.0
        self._cmd_z = 0.0
        self._t = time.time()
        self._stale_after = 0.0

    def set(
        self,
        cmd_x: float,
        cmd_y: float,
        cmd_z: float,
        valid_for_sec: float = CMD_VALID_SEC,
    ):
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


class CmdVelPublisher(threading.Thread):
    def __init__(
        self,
        vel: VelCommand,
        fsm: Optional[FsmStateMonitor],
        dry_run: bool,
    ):
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
                    "[5_finger_locomotion] 尚无 /cmd_vel 订阅者",
                )
        self._rate = rospy.Rate(LOCOMOTION_PUBLISH_RATE_HZ)
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def publish_stop_blocking(self, duration: float = 0.5):
        msg = Twist()
        end_t = time.time() + duration
        while time.time() < end_t:
            if not self._dry_run:
                self._pub.publish(msg)
            time.sleep(1.0 / max(LOCOMOTION_PUBLISH_RATE_HZ, 1))

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


@dataclass
class LocomotionFrameResult:
    cmd_x: float = 0.0
    cmd_y: float = 0.0
    cmd_z: float = 0.0
    mode: str = "idle"
    hand_moving: bool = False


class FingerLocomotionStack:
    """
    五指跟手栈：在自定义主循环中每帧调用 update()。
    默认不在主程序中实例化。
    """

    def __init__(
        self,
        fsm: Optional[FsmStateMonitor],
        *,
        dry_run: bool = False,
        target_distance_m: float = TARGET_DISTANCE_M,
        pos_thresh_m: float = POS_THRESH_M,
        cmd_mag: float = CMD_MAG,
        move_thresh_m: float = HAND_MOVE_THRESH_M,
    ):
        self.follow_ctrl = HandFollowController(
            target_distance_m=target_distance_m,
            pos_thresh_m=pos_thresh_m,
            cmd_mag=cmd_mag,
        )
        self.motion_gate = HandMotionGate(move_thresh_m=move_thresh_m)
        self.vel_cmd = VelCommand()
        self.pub_thread = CmdVelPublisher(self.vel_cmd, fsm, dry_run=dry_run)
        self.pub_thread.start()

    def update(
        self,
        *,
        gesture: int,
        distance_m: float,
        palm_x_m: float,
        palm_y_m: float,
        palm_pos: Optional[Tuple[float, float, float]],
        active: bool,
        zero_estop: bool,
        joy_blocking: bool,
        action_fired: bool,
        action_busy: bool,
    ) -> LocomotionFrameResult:
        hand_moving = self.motion_gate.update(palm_pos)
        ctrl_inp = HandControlInput(
            gesture=gesture,
            distance_m=distance_m,
            palm_x_m=palm_x_m,
            palm_y_m=palm_y_m,
            active=active,
        )

        if zero_estop or joy_blocking or action_fired or action_busy:
            self.follow_ctrl.reset()
            out = LocomotionFrameResult(hand_moving=hand_moving)
            if zero_estop:
                out.mode = "estop"
            elif joy_blocking:
                out.mode = "joy"
            else:
                out.mode = "action"
        elif not ctrl_inp.active or not hand_moving:
            self.follow_ctrl.reset()
            out = LocomotionFrameResult(hand_moving=hand_moving)
            out.mode = "idle" if not ctrl_inp.active else "hold"
        else:
            computed = self.follow_ctrl.compute(ctrl_inp)
            out = LocomotionFrameResult(
                cmd_x=computed.cmd_x,
                cmd_y=computed.cmd_y,
                cmd_z=computed.cmd_z,
                mode=computed.mode,
                hand_moving=hand_moving,
            )

        self.vel_cmd.set(out.cmd_x, out.cmd_y, out.cmd_z)
        return out

    def reset_on_estop(self):
        self.follow_ctrl.reset()
        self.motion_gate.reset()
        self.vel_cmd.stop()

    def shutdown(self):
        self.vel_cmd.stop()
        self.pub_thread.publish_stop_blocking(0.5)
        self.pub_thread.stop()
