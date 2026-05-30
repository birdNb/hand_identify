#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 0~4 的 ROS 动作执行（供 zed_gesture_recognition 等调用）。"""

import os
import sys
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(gesture_recognition=True, motion=True)

from ros_setup import require_sim2real_msg

require_sim2real_msg()

import rospy

from gesture_actions import (
    ConfirmedActionGate,
    GESTURE_ACTION_SPECS,
    GESTURE_HEAD_NOD,
    GESTURE_STOP,
    log_gesture_zero_estop,
)
from hand_action_library import GestureActionPlayer
from ros_control import FsmStateMonitor, JoyMonitor, JOY_IDLE_SEC
from waist_coquette_player import WaistCoquettePlayer


class GestureMotionController:
    """手势稳定确认后触发撒娇扭腰(1)与动作库(2~4)。不控脖子，避免与脸跟踪冲突。"""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        no_actions: bool = False,
        no_coquette: bool = False,
        no_joy: bool = False,
        no_fsm: bool = False,
    ):
        self._dry_run = dry_run
        self._fsm = None if no_fsm else FsmStateMonitor()
        self._joy = None if no_joy else JoyMonitor()
        self._action_player: Optional[GestureActionPlayer] = None
        self._coquette_player: Optional[WaistCoquettePlayer] = None
        self._fire_gate = ConfirmedActionGate()

        if not no_actions:
            self._action_player = GestureActionPlayer(dry_run=dry_run)
        if not no_actions and not no_coquette:
            self._coquette_player = WaistCoquettePlayer(dry_run=dry_run)

        mode = "DRY-RUN" if dry_run else "EXEC"
        rospy.loginfo(
            "[gesture_motion] 模式=%s | 撒娇扭腰=%s | 动作库=%s | FSM=%s | 手柄=%s",
            mode,
            "关" if no_coquette or no_actions else "开",
            "关" if no_actions else "2~4",
            "跳过" if no_fsm else "等待5",
            "无" if no_joy else f"空闲{JOY_IDLE_SEC:.0f}s",
        )

    @property
    def fsm(self) -> Optional[FsmStateMonitor]:
        return self._fsm

    def wait_fsm(
        self,
        timeout: float = 30.0,
        *,
        should_stop=None,
    ) -> bool:
        if self._fsm is None:
            return True
        rospy.loginfo("[gesture_motion] 等待 FSM EXEC_DEFAULT(5)...")
        ok = self._fsm.wait_for_exec_default(
            timeout=timeout, should_stop=should_stop,
        )
        if ok:
            rospy.loginfo("[gesture_motion] FSM OK")
        else:
            rospy.logwarn("[gesture_motion] FSM 超时，动作可能不执行")
        return ok

    @property
    def joy_blocking(self) -> bool:
        return self._joy is not None and self._joy.blocks_gesture_control()

    @property
    def fsm_ok(self) -> bool:
        return self._fsm is None or self._fsm.is_exec_default()

    def poll_joy_takeover(self) -> bool:
        if self._joy is None:
            return False
        return self._joy.poll_takeover_edge()

    def on_zero_estop(self, *, hold_sec: float) -> None:
        if hold_sec < 0.15:
            log_gesture_zero_estop(dry_run=self._dry_run)
        if self._action_player is not None:
            self._action_player.abort()
        if self._coquette_player is not None:
            self._coquette_player.abort()

    def on_confirmed(self, gesture: int, *, has_hand: bool, in_range: bool) -> bool:
        """已稳定 hold 的手势触发动作；未成功则下帧重试。返回本帧是否新触发。"""
        if gesture == GESTURE_STOP:
            return False
        if not self._fire_gate.need_fire(gesture):
            return False

        joy_blocking = self.joy_blocking
        fsm_ok = self.fsm_ok
        allow_retry = True

        if self.poll_joy_takeover():
            busy = (
                (self._action_player is not None and self._action_player.is_busy)
                or (
                    self._coquette_player is not None
                    and self._coquette_player.is_busy
                )
            )
            if busy:
                self.on_zero_estop(hold_sec=1.0)
            rospy.loginfo("[gesture_motion] 手柄接管，已停手势动作")

        if joy_blocking:
            return False

        action_busy = (
            self._action_player is not None and self._action_player.is_busy
        )
        coquette_busy = (
            self._coquette_player is not None and self._coquette_player.is_busy
        )

        fired = False
        if gesture == GESTURE_HEAD_NOD and self._coquette_player is not None:
            fired = self._coquette_player.update(
                gesture,
                has_hand=has_hand,
                in_range=in_range,
                joy_blocking=joy_blocking,
                fsm_ok=fsm_ok,
                other_busy=action_busy,
                allow_retry=allow_retry,
            )
        elif (
            self._action_player is not None
            and gesture in GESTURE_ACTION_SPECS
            and not coquette_busy
        ):
            fired = self._action_player.update(
                gesture,
                has_hand=has_hand,
                in_range=in_range,
                joy_blocking=joy_blocking,
                fsm_ok=fsm_ok,
                allow_retry=allow_retry,
            )
        if fired:
            self._fire_gate.mark_fired(gesture)
        return fired

    def clear_pending_fire(self) -> None:
        self._fire_gate.clear()

    def shutdown(self, *, fast: bool = False) -> None:
        self._fire_gate.clear()
        if self._action_player is not None:
            self._action_player.abort(fast=fast)
        if self._coquette_player is not None:
            self._coquette_player.abort(fast=fast)
        if fast:
            try:
                import rospy
                if rospy.core.is_initialized() and not rospy.is_shutdown():
                    rospy.signal_shutdown("user exit")
            except Exception:
                pass
