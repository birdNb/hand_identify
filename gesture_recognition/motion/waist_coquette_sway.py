#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手势 1 撒娇扭腰：waist_yaw ±45° 来回 2 次，并触发 cheer(挥双手)。

仅发布 waist_yaw_joint，不控 head，避免与脸部跟踪冲突。
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Optional, Set

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(motion=True)

from ros_setup import require_sim2real_msg

require_sim2real_msg()

import rospy
from sensor_msgs.msg import JointState
from sim2real_msg.msg import Joy

from ros_control import ABSOLUTE_TOPIC, FsmStateMonitor

# ----- 时序（与 waist_coquette_player 中 COQUETTE_BUSY_SEC 对齐） -----
SWAY_AMPLITUDE_DEG = 45.0
SWAY_CYCLES = 2
# 匀速角速度 (deg/s)，全程线性插值、端点不停留
SWAY_ANGULAR_VEL_DEG_PER_SEC = 60.0
ACTION_DURATION_SEC = 5.0
ARM_RESET_WAIT_SEC = 0.5
TRIGGER_PULSE_SEC = 0.5
JOY_MSG_TOPIC = "/joy_msg"
JOY_PUBLISH_HZ = 20
PUBLISH_HZ = 50

WAIST_YAW_JOINT = "waist_yaw_joint"
CHEER_KEYS = {"rt", "a"}


def _sway_waypoints_rad(amplitude_rad: float) -> list[float]:
    """0 → (+,-) 交替 cycles 次 → 回 0。"""
    pts = [0.0]
    for i in range(SWAY_CYCLES * 2):
        pts.append(amplitude_rad if (i % 2 == 0) else -amplitude_rad)
    if abs(pts[-1]) > 1e-9:
        pts.append(0.0)
    return pts


def _sway_motion_duration_sec(waypoints_rad: list[float]) -> float:
    speed = math.radians(max(SWAY_ANGULAR_VEL_DEG_PER_SEC, 1e-3))
    total = 0.0
    for i in range(1, len(waypoints_rad)):
        total += abs(waypoints_rad[i] - waypoints_rad[i - 1]) / speed
    return total


_SWAY_WAYPOINTS = _sway_waypoints_rad(math.radians(SWAY_AMPLITUDE_DEG))
SWAY_MOTION_SEC = _sway_motion_duration_sec(_SWAY_WAYPOINTS)
ACTION_TOTAL_SEC = (
    TRIGGER_PULSE_SEC
    + SWAY_MOTION_SEC
    + ACTION_DURATION_SEC
    + ARM_RESET_WAIT_SEC
    + TRIGGER_PULSE_SEC
)


def _parse_keys(combo: str) -> Set[str]:
    return {p.strip().lower() for p in combo.split("+") if p.strip()}


def _joy_from_keys(keys: Set[str], pressed: bool) -> Joy:
    msg = Joy()
    field_map = {
        "a": "a", "b": "b", "x": "x", "y": "y",
        "lb": "lb", "rb": "rb", "back": "back", "start": "start",
        "lt": "lt", "rt": "rt",
        "l": "L", "r": "R", "center": "center",
    }
    press_val = 1.0 if pressed else 0.0
    trig_press, trig_release = -1.0, 1.0
    for key in keys:
        attr = field_map.get(key)
        if attr is None:
            continue
        val = (
            (trig_press if pressed else trig_release)
            if key in ("lt", "rt")
            else press_val
        )
        setattr(msg, attr, val)
    return msg


def _pulse_joy(
    pub: rospy.Publisher,
    keys: Set[str],
    *,
    duration_sec: float,
    dry_run: bool,
    abort_evt: Optional[threading.Event],
) -> None:
    if dry_run or not keys:
        time.sleep(min(duration_sec, 0.1))
        return
    press = _joy_from_keys(keys, True)
    release = _joy_from_keys(keys, False)
    interval = 1.0 / max(JOY_PUBLISH_HZ, 1)
    end_t = time.time() + max(0.05, duration_sec)
    while time.time() < end_t and not rospy.is_shutdown():
        if abort_evt is not None and abort_evt.is_set():
            return
        pub.publish(press)
        time.sleep(interval)
    for _ in range(3):
        if rospy.is_shutdown() or (
            abort_evt is not None and abort_evt.is_set()
        ):
            return
        pub.publish(release)
        time.sleep(interval)


def _interp_waypoints(
    waypoints_rad: list[float],
    elapsed: float,
    seg_starts: list[float],
) -> float:
    if elapsed >= seg_starts[-1]:
        return waypoints_rad[-1]
    for i in range(1, len(seg_starts)):
        if elapsed <= seg_starts[i]:
            t0, t1 = seg_starts[i - 1], seg_starts[i]
            alpha = (elapsed - t0) / max(t1 - t0, 1e-9)
            p0, p1 = waypoints_rad[i - 1], waypoints_rad[i]
            return p0 + (p1 - p0) * alpha
    return waypoints_rad[-1]


def _run_uniform_sway(
    pub: rospy.Publisher,
    waypoints_rad: list[float],
    *,
    dry_run: bool,
    abort_evt,
) -> float:
    """
    按固定角速度在路径点间线性插值，中途不停、段间速度连续。
    返回结束时腰 yaw (rad)。
    """
    if len(waypoints_rad) < 2:
        return waypoints_rad[0] if waypoints_rad else 0.0

    speed = math.radians(max(SWAY_ANGULAR_VEL_DEG_PER_SEC, 1e-3))
    seg_starts = [0.0]
    for i in range(1, len(waypoints_rad)):
        dt = abs(waypoints_rad[i] - waypoints_rad[i - 1]) / speed
        seg_starts.append(seg_starts[-1] + dt)
    total = seg_starts[-1]

    msg = JointState()
    msg.name = [WAIST_YAW_JOINT]
    msg.velocity = []
    msg.effort = []
    interval = 1.0 / max(PUBLISH_HZ, 1)
    t0 = time.time()
    pos = waypoints_rad[0]
    while not rospy.is_shutdown():
        if abort_evt is not None and abort_evt.is_set():
            return pos
        elapsed = time.time() - t0
        pos = _interp_waypoints(waypoints_rad, elapsed, seg_starts)
        if not dry_run:
            msg.header.stamp = rospy.Time.now()
            msg.position = [pos]
            pub.publish(msg)
        if elapsed >= total:
            break
        time.sleep(interval)
    return pos


def _wait_fsm(skip: bool) -> None:
    if skip:
        return
    fsm = FsmStateMonitor()
    fsm.wait_for_exec_default(timeout=30.0)


def run_coquette_action(
    *,
    dry_run: bool = False,
    abort_evt=None,
    skip_fsm_wait: bool = False,
) -> None:
    """执行撒娇：挥双手(cheer) + 腰部 ±45° 来回 2 次 + 回中。"""
    _wait_fsm(skip_fsm_wait)
    if abort_evt is not None and abort_evt.is_set():
        return

    waypoints = _sway_waypoints_rad(math.radians(SWAY_AMPLITUDE_DEG))

    waist_pub = rospy.Publisher(ABSOLUTE_TOPIC, JointState, queue_size=10)
    joy_pub = rospy.Publisher(JOY_MSG_TOPIC, Joy, queue_size=1)
    if not dry_run:
        t0 = time.time()
        while (
            waist_pub.get_num_connections() == 0
            and not rospy.is_shutdown()
            and time.time() - t0 < 3.0
        ):
            time.sleep(0.05)

    cheer_keys = _parse_keys("rt+a")
    _pulse_joy(
        joy_pub, cheer_keys,
        duration_sec=TRIGGER_PULSE_SEC,
        dry_run=dry_run,
        abort_evt=abort_evt,
    )

    sway_t0 = time.time()
    cur = _run_uniform_sway(
        waist_pub,
        waypoints,
        dry_run=dry_run,
        abort_evt=abort_evt,
    )

    cheer_remain = max(
        0.0,
        ACTION_DURATION_SEC - (time.time() - sway_t0),
    )
    end_cheer = time.time() + cheer_remain
    while time.time() < end_cheer and not rospy.is_shutdown():
        if abort_evt is not None and abort_evt.is_set():
            break
        time.sleep(0.05)

    if abort_evt is None or not abort_evt.is_set():
        _pulse_joy(
            joy_pub, cheer_keys,
            duration_sec=TRIGGER_PULSE_SEC,
            dry_run=dry_run,
            abort_evt=abort_evt,
        )

    if ARM_RESET_WAIT_SEC > 0:
        time.sleep(ARM_RESET_WAIT_SEC)
