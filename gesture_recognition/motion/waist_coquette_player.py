#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 1：触发撒娇扭腰（waist_yaw 摆动 + cheer 挥双手）。"""

from __future__ import annotations

import threading
import time
from typing import Optional

import rospy

from waist_coquette_sway import ACTION_TOTAL_SEC, run_coquette_action

COQUETTE_LABEL = "撒娇扭腰"
COQUETTE_BUSY_SEC = ACTION_TOTAL_SEC + 1.5
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

    def abort(self, *, fast: bool = False) -> None:
        self._abort_evt.set()
        self._busy_until = 0.0
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.15 if fast else 1.0)
        with self._lock:
            self._worker = None
        self._abort_evt.clear()
        if not fast:
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
        allow_retry: bool = False,
    ) -> bool:
        if gesture != 1:
            self._last_gesture = -1
            return False

        prev = self._last_gesture
        self._last_gesture = gesture

        if joy_blocking or not fsm_ok or self.is_busy or other_busy:
            return False
        if not has_hand or not in_range:
            return False
        if gesture == prev and not allow_retry:
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
