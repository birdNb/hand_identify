#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 1：触发 wing_wist 腰部撒娇固定动作。"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import rospy

_GESTURE = 1
_WING_WIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "wing_wist"),
)
if _WING_WIST not in sys.path:
    sys.path.insert(0, _WING_WIST)

from waist_coquette_sway import (  # noqa: E402
    ACTION_TOTAL_SEC,
    ARM_RESET_WAIT_SEC,
    RAMP_SEC,
    run_coquette_action,
)

COQUETTE_LABEL = "腰部撒娇"
COQUETTE_BUSY_SEC = ACTION_TOTAL_SEC + ARM_RESET_WAIT_SEC + RAMP_SEC + 1.5
COQUETTE_COOLDOWN_SEC = COQUETTE_BUSY_SEC + 0.5


class WaistCoquettePlayer:
    """手势 1 上升沿触发撒娇动作（后台线程）。"""

    def __init__(self, *, dry_run: bool = False):
        self._dry_run = dry_run
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._abort_evt = threading.Event()
        self._last_gesture = -1
        self._last_fire_t = 0.0
        self._busy_until = 0.0
        self._last_label = COQUETTE_LABEL

    @property
    def is_busy(self) -> bool:
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            return True
        return time.time() < self._busy_until

    @property
    def last_label(self) -> str:
        return self._last_label

    def abort(self) -> None:
        self._abort_evt.set()
        self._busy_until = 0.0
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        with self._lock:
            self._worker = None
        self._abort_evt.clear()
        line = f">>> 动作中止: {COQUETTE_LABEL}"
        rospy.logwarn("[coquette_player] %s", line)
        print(line, flush=True)

    def update(
        self,
        gesture: int,
        *,
        has_hand: bool,
        in_range: bool,
        joy_blocking: bool = False,
        fsm_ok: bool = True,
        other_busy: bool = False,
    ) -> bool:
        if gesture != _GESTURE:
            self._last_gesture = -1
            return False

        prev = self._last_gesture
        self._last_gesture = gesture

        if joy_blocking or not fsm_ok or self.is_busy or other_busy:
            return False
        if not has_hand or not in_range:
            return False
        if gesture == prev:
            return False
        if time.time() - self._last_fire_t < COQUETTE_COOLDOWN_SEC:
            return False

        self._last_fire_t = time.time()
        self._busy_until = time.time() + COQUETTE_BUSY_SEC
        self._start_worker()
        return True

    def _start_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                rospy.logwarn("[coquette_player] 上一段撒娇未完成，跳过")
                return
            self._abort_evt.clear()
            self._worker = threading.Thread(
                target=self._run_blocking,
                daemon=True,
            )
            self._worker.start()

    def _run_blocking(self) -> None:
        mode = "DRY-RUN" if self._dry_run else "EXEC"
        line = f">>> 触发动作: {COQUETTE_LABEL} [{mode}]"
        rospy.loginfo("[coquette_player] %s", line)
        print(line, flush=True)
        try:
            run_coquette_action(
                dry_run=self._dry_run,
                abort_evt=self._abort_evt,
                skip_fsm_wait=True,
            )
        except Exception as exc:
            rospy.logerr("[coquette_player] 执行失败: %s", exc)
        finally:
            with self._lock:
                self._worker = None
            if not self._abort_evt.is_set():
                print(f">>> 动作完成: {COQUETTE_LABEL}", flush=True)
