#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过 /joy_msg 触发 sim2real 动作库（与 custom_action.yaml 手柄组合键一致）。"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(motion=True)

from ros_setup import require_sim2real_msg

require_sim2real_msg()

import rospy
from sim2real_msg.msg import Joy

from gesture_actions import (
    GESTURE_ACTION_LABELS,
    GESTURE_ACTION_SPECS,
    format_action_trigger_line,
)

JOY_MSG_TOPIC = "/joy_msg"
JOY_PUBLISH_HZ = 20

# 每个手势动作最长执行时长；到时再发一次相同组合键以停止
ACTION_DURATION_SEC = 5.0
TRIGGER_PULSE_SEC = 0.5
ACTION_COOLDOWN_SEC = ACTION_DURATION_SEC + 1.0
BUTTON_PRESS = 1.0
BUTTON_RELEASE = 0.0
# 与 joy_teleop / Xbox 一致：RT/LT 松开=+1.0，按下=-1.0
TRIGGER_PRESS = -1.0
TRIGGER_RELEASE = 1.0


def _parse_key_combo(combo: str) -> Set[str]:
    return {p.strip().lower() for p in combo.split("+") if p.strip()}


def _joy_key_value(key: str, pressed: bool) -> float:
    if key in ("lt", "rt"):
        return TRIGGER_PRESS if pressed else TRIGGER_RELEASE
    return BUTTON_PRESS if pressed else BUTTON_RELEASE


def _joy_from_keys(keys: Set[str], pressed: bool) -> Joy:
    """将组合键映射为 sim2real_msg/Joy（与手柄 joy_teleop 语义一致）。"""
    msg = Joy()
    field_map = {
        "a": "a", "b": "b", "x": "x", "y": "y",
        "lb": "lb", "rb": "rb", "back": "back", "start": "start",
        "lt": "lt", "rt": "rt",
        "l": "L", "r": "R", "center": "center",
    }
    for key in keys:
        attr = field_map.get(key)
        if attr is None:
            rospy.logwarn("[gesture_action] 未知按键: %s", key)
            continue
        setattr(msg, attr, _joy_key_value(key, pressed))
    return msg


def _pulse_keys(
    pub: rospy.Publisher,
    keys: Set[str],
    *,
    duration_sec: float,
    dry_run: bool,
    abort_evt: threading.Event,
) -> None:
    """短时按下组合键（触发或再次触发停止）。"""
    if dry_run or not keys:
        return
    press = _joy_from_keys(keys, pressed=True)
    release = _joy_from_keys(keys, pressed=False)
    interval = 1.0 / max(JOY_PUBLISH_HZ, 1)
    end_t = time.time() + max(0.05, duration_sec)
    while time.time() < end_t and not rospy.is_shutdown():
        if abort_evt.is_set():
            break
        pub.publish(press)
        time.sleep(interval)
    for _ in range(3):
        if rospy.is_shutdown() or abort_evt.is_set():
            break
        pub.publish(release)
        time.sleep(interval)


@dataclass
class GestureActionSpec:
    gesture: int
    action_name: str
    label: str
    key_combo: str
    keepalive_sec: float

    @classmethod
    def from_gesture(cls, gesture: int) -> Optional["GestureActionSpec"]:
        row = GESTURE_ACTION_SPECS.get(gesture)
        if row is None:
            return None
        name, combo, keepalive = row
        return cls(
            gesture=gesture,
            action_name=name,
            label=GESTURE_ACTION_LABELS.get(gesture, name),
            key_combo=combo,
            keepalive_sec=keepalive,
        )


class GestureActionPlayer:
    """手势边沿触发动作库；播放期间 is_busy 为 True。"""

    def __init__(
        self,
        dry_run: bool = True,
        cooldown_sec: float = ACTION_COOLDOWN_SEC,
    ):
        self._dry_run = dry_run
        self._cooldown_sec = cooldown_sec
        self._pub = rospy.Publisher(JOY_MSG_TOPIC, Joy, queue_size=1)
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
                    "[gesture_action] 尚无 /joy_msg 订阅者, 动作可能无效",
                )
        self._lock = threading.Lock()
        self._last_gesture = -1
        self._last_fire_t = 0.0
        self._busy_until = 0.0
        self._last_label = ""
        self._worker: Optional[threading.Thread] = None
        self._abort_evt = threading.Event()
        self._active_keys: Set[str] = set()

    @property
    def is_busy(self) -> bool:
        return time.time() < self._busy_until

    @property
    def last_label(self) -> str:
        return self._last_label

    def abort(self, *, fast: bool = False):
        """中止当前动作：再次发送动作组合键并松开。fast=True 用于 Ctrl+C 强制退出。"""
        keys = set(self._active_keys)
        label = self._last_label
        self._abort_evt.set()
        self._busy_until = 0.0
        self._last_label = ""
        if not fast and not self._dry_run and keys:
            _pulse_keys(
                self._pub, keys,
                duration_sec=TRIGGER_PULSE_SEC,
                dry_run=False,
                abort_evt=threading.Event(),
            )
        self._active_keys.clear()
        if not fast and not self._dry_run:
            release = Joy()
            for _ in range(3):
                if self._abort_evt.is_set():
                    break
                self._pub.publish(release)
                time.sleep(1.0 / max(JOY_PUBLISH_HZ, 1))
        if label and not fast:
            stop_line = f">>> 动作中止: {label}"
            rospy.logwarn("[gesture_action] %s", stop_line)
            print(stop_line, flush=True)
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.15 if fast else 0.8)
        with self._lock:
            self._worker = None
        self._abort_evt.clear()

    def update(
        self,
        gesture: int,
        *,
        has_hand: bool,
        in_range: bool,
        joy_blocking: bool = False,
        fsm_ok: bool = True,
        allow_retry: bool = False,
    ) -> bool:
        """
        检测手势上升沿并触发动作。返回本帧是否新触发。
        allow_retry: 稳定确认后若上次因 busy/fsm 未触发，可重试。
        """
        if gesture not in GESTURE_ACTION_SPECS:
            self._last_gesture = -1
            return False

        prev = self._last_gesture
        self._last_gesture = gesture

        if joy_blocking or not fsm_ok or self.is_busy:
            return False
        if not has_hand or not in_range:
            return False
        if gesture == 0:
            return False
        if gesture == prev and not allow_retry:
            return False
        if time.time() - self._last_fire_t < self._cooldown_sec:
            return False

        spec = GestureActionSpec.from_gesture(gesture)
        if spec is None:
            return False

        self._last_fire_t = time.time()
        self._busy_until = (
            time.time() + ACTION_DURATION_SEC + TRIGGER_PULSE_SEC * 2 + 0.5
        )
        self._last_label = spec.label
        self._start_play(spec)
        return True

    def _start_play(self, spec: GestureActionSpec):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                rospy.logwarn(
                    "[gesture_action] 上一动作未完成，跳过 %s", spec.label,
                )
                return
            self._worker = threading.Thread(
                target=self._play_blocking,
                args=(spec,),
                daemon=True,
            )
            self._worker.start()

    def _play_blocking(self, spec: GestureActionSpec):
        keys = _parse_key_combo(spec.key_combo)
        self._active_keys = set(keys)
        line = format_action_trigger_line(
            spec.gesture, dry_run=self._dry_run,
        )
        rospy.loginfo("[gesture_action] %s", line)
        print(line, flush=True)
        if self._dry_run:
            deadline = time.time() + ACTION_DURATION_SEC
            while time.time() < deadline and not self._abort_evt.is_set():
                time.sleep(0.05)
            self._active_keys.clear()
            if not self._abort_evt.is_set():
                stop_line = (
                    f">>> 动作停止: {spec.label} "
                    f"({ACTION_DURATION_SEC:.0f}s 后再次发送指令)"
                )
                print(stop_line, flush=True)
            return

        try:
            _pulse_keys(
                self._pub, keys,
                duration_sec=TRIGGER_PULSE_SEC,
                dry_run=False,
                abort_evt=self._abort_evt,
            )
            start_t = time.time()
            while (
                time.time() - start_t < ACTION_DURATION_SEC
                and not rospy.is_shutdown()
                and not self._abort_evt.is_set()
            ):
                time.sleep(0.05)

            if not self._abort_evt.is_set():
                stop_line = (
                    f">>> 动作停止: {spec.label} "
                    f"({ACTION_DURATION_SEC:.0f}s 后再次发送 {spec.key_combo})"
                )
                rospy.loginfo("[gesture_action] %s", stop_line)
                print(stop_line, flush=True)
                _pulse_keys(
                    self._pub, keys,
                    duration_sec=TRIGGER_PULSE_SEC,
                    dry_run=False,
                    abort_evt=self._abort_evt,
                )
        finally:
            self._active_keys.clear()
            if not self._abort_evt.is_set():
                release = _joy_from_keys(keys, pressed=False) if keys else Joy()
                interval = 1.0 / max(JOY_PUBLISH_HZ, 1)
                for _ in range(3):
                    if rospy.is_shutdown():
                        break
                    self._pub.publish(release)
                    time.sleep(interval)
