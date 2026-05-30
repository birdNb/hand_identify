#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 0~5 定义与动作库映射（手势识别 / 手部跟踪共用）。"""

import time
from typing import Dict, Optional, Tuple

GESTURE_STOP = 0
GESTURE_HEAD_NOD = 1
GESTURE_FACE_TRACK = GESTURE_HEAD_NOD  # 兼容旧名
GESTURE_FOLLOW = 5
GESTURE_FOLLOW_LABEL = "五指跟手(切换手部跟踪)"
# 手势5 进入/退出跟手模式的稳定时长（秒）
GESTURE_FOLLOW_HOLD_SEC = 8.0
GESTURE_FOLLOW_LOST_SEC = 8.0
GESTURE_ZERO_EXIT_SEC = 5.0
GESTURE_ZERO_LABEL = "急停/退出"
GESTURE_HEAD_NOD_LABEL = "撒娇扭腰"
GESTURE_FACE_TRACK_LABEL = "脸部跟踪(常开)"

# 需稳定 hold 后才触发的手势（含 1 撒娇、2~4 动作库）
GESTURE_HOLD_GESTURES = frozenset({1, 2, 3, 4})

# gesture -> (action_name, joy_key_combo, keepalive_sec 已废弃)
GESTURE_ACTION_SPECS: Dict[int, Tuple[str, str, float]] = {
    2: ("hello", "rt+x", 5.0),
    3: ("cheer", "rt+a", 5.0),
    4: ("byd_small_kick", "x", 5.0),
}

GESTURE_ACTION_LABELS: Dict[int, str] = {
    2: "抬手",
    3: "挥动双手",
    4: "踢球",
}

GESTURE_ACTION_HOLD_SEC = 2.0
# 短暂丢手/出距时不立刻清零稳定计时（减轻卡顿导致的确认失败）
HAND_LOST_GRACE_SEC = 0.45

TERM_LINE_WIDTH = 96


def emit_status_line(text: str, *, width: int = TERM_LINE_WIDTH) -> None:
    """固定宽度 \\r 刷新，避免上一行残留字符。"""
    plain = text
    try:
        from colorama import Fore, Style

        for token in (
            Fore.CYAN, Fore.YELLOW, Fore.WHITE, Fore.GREEN, Fore.RED,
            Fore.BLUE, Fore.MAGENTA, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX,
            Style.RESET_ALL,
        ):
            plain = plain.replace(token, "")
    except ImportError:
        pass
    pad = max(0, width - len(plain))
    print(f"\r{text}{' ' * pad}", end="", flush=True)


def action_hint_for_gesture(gesture: int, *, face_track_on: bool = False) -> str:
    """状态行后缀：当前手势对应的动作说明。"""
    if gesture == GESTURE_STOP:
        return GESTURE_ZERO_LABEL
    if gesture == GESTURE_HEAD_NOD:
        return GESTURE_HEAD_NOD_LABEL
    if gesture in GESTURE_ACTION_LABELS:
        spec = GESTURE_ACTION_SPECS[gesture]
        return f"动作:{GESTURE_ACTION_LABELS[gesture]}({spec[0]})"
    if gesture == GESTURE_FOLLOW:
        return GESTURE_FOLLOW_LABEL
    return ""


def format_action_trigger_line(
    gesture: int,
    *,
    dry_run: bool = False,
    skipped_reason: Optional[str] = None,
) -> str:
    """上升沿触发时单独打印一整行（不覆盖状态行）。"""
    if gesture not in GESTURE_ACTION_LABELS:
        return ""
    name, keys, _ = GESTURE_ACTION_SPECS[gesture]
    label = GESTURE_ACTION_LABELS[gesture]
    if skipped_reason:
        return (
            f">>> 动作未执行: {label} ({name}, {keys}) "
            f"[跳过: {skipped_reason}]"
        )
    mode = "DRY-RUN" if dry_run else "EXEC"
    return f">>> 触发动作: {label} ({name}, {keys}) [{mode}]"


def log_gesture_coquette(*, dry_run: bool = False) -> None:
    mode = "DRY-RUN" if dry_run else "EXEC"
    line = f">>> 手势1: {GESTURE_HEAD_NOD_LABEL} [{mode}]"
    try:
        from colorama import Fore

        print(
            f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}{line}",
            flush=True,
        )
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def log_gesture_head_nod(*, dry_run: bool = False) -> None:
    """兼容旧名。"""
    log_gesture_coquette(dry_run=dry_run)


def log_face_track_toggle(
    *,
    enabled: bool,
    dry_run: bool = False,
    reason: str = "",
) -> None:
    state = "开启" if enabled else "关闭"
    mode = "DRY-RUN" if dry_run else "EXEC"
    extra = f" ({reason})" if reason else ""
    line = f">>> 脸部跟踪{state}{extra} [{mode}]"
    try:
        from colorama import Fore

        color = Fore.GREEN if enabled else Fore.YELLOW
        print(
            f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {color}{line}",
            flush=True,
        )
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def log_gesture_action_edge(
    gesture: int,
    prev_gesture: int,
    *,
    in_range: bool,
    has_hand: bool,
    preview_only: bool = True,
    face_track_on: bool = False,
) -> None:
    """手势切换到 2~4 时单独打日志。"""
    import time

    if gesture == prev_gesture:
        return
    if gesture not in GESTURE_ACTION_LABELS:
        return
    if not has_hand or not in_range:
        line = format_action_trigger_line(
            gesture,
            skipped_reason="无手或超出识别距离",
        )
    else:
        line = format_action_trigger_line(
            gesture, dry_run=preview_only,
        )
    try:
        from colorama import Fore

        prefix = f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}"
    except ImportError:
        prefix = f"\n[{time.strftime('%H:%M:%S')}] "
    print(f"{prefix}{line}", flush=True)


class ConfirmedActionGate:
    """稳定确认后允许重试触发，直到动作真正启动。"""

    def __init__(self):
        self._episode = -1
        self._fired = False

    def clear(self) -> None:
        self._episode = -1
        self._fired = False

    def need_fire(self, gesture: int) -> bool:
        if gesture < 0:
            self.clear()
            return False
        if gesture != self._episode:
            self._episode = gesture
            self._fired = False
        return not self._fired

    def mark_fired(self, gesture: int) -> None:
        if gesture == self._episode:
            self._fired = True


class GestureActionHold:
    """指定手势需连续稳定 hold_sec 后才输出（默认 1~4 动作，可仅 5 跟手切换）。"""

    def __init__(
        self,
        hold_sec: float = GESTURE_ACTION_HOLD_SEC,
        lost_grace_sec: float = HAND_LOST_GRACE_SEC,
        allowed_gestures: Optional[frozenset] = None,
    ):
        self.hold_sec = max(0.1, float(hold_sec))
        self._lost_grace = max(0.0, float(lost_grace_sec))
        self._allowed = (
            allowed_gestures
            if allowed_gestures is not None
            else GESTURE_HOLD_GESTURES
        )
        self._candidate = -1
        self._since = 0.0
        self._lost_since: Optional[float] = None

    def reset(self):
        self._candidate = -1
        self._since = 0.0
        self._lost_since = None

    @property
    def pending_gesture(self) -> int:
        return self._candidate

    @property
    def progress(self) -> float:
        if self._candidate < 0 or self._since <= 0:
            return 0.0
        return min(1.0, (time.time() - self._since) / self.hold_sec)

    @property
    def hold_remaining(self) -> float:
        if self._candidate < 0:
            return 0.0
        return max(0.0, self.hold_sec - (time.time() - self._since))

    def _tracking_lost(self, has_hand: bool, in_range: bool, now: float) -> bool:
        if has_hand and in_range:
            self._lost_since = None
            return False
        if self._lost_grace <= 0:
            return True
        if self._lost_since is None:
            self._lost_since = now
        return (now - self._lost_since) >= self._lost_grace

    def update(
        self,
        gesture: int,
        *,
        has_hand: bool,
        in_range: bool,
    ) -> int:
        """
        返回已确认的手势(1~4)，未满足稳定时长则返回 -1。
        确认后持续返回同一手势，直至手势变化或丢手超过 grace。
        """
        now = time.time()
        if self._tracking_lost(has_hand, in_range, now):
            self.reset()
            return -1
        if gesture not in self._allowed:
            self.reset()
            return -1

        if gesture != self._candidate:
            self._candidate = gesture
            self._since = now
            return -1
        if now - self._since >= self.hold_sec:
            return gesture
        return -1


class GestureZeroHandler:
    """手势 0：立即急停/中止动作；持续按住超过 exit_hold_sec 后请求退出程序。"""

    def __init__(self, exit_hold_sec: float = GESTURE_ZERO_EXIT_SEC):
        self.exit_hold_sec = max(0.1, float(exit_hold_sec))
        self._hold_start: Optional[float] = None
        self._estop_announced = False

    def reset(self):
        self._hold_start = None
        self._estop_announced = False

    def update(
        self,
        gesture: int,
        *,
        has_hand: bool,
        in_range: bool,
    ) -> Tuple[bool, bool, float]:
        """
        Returns:
            need_estop: 本帧应执行急停（进入手势 0 的首帧为 True，持续按住也为 True）
            should_exit: 按住已达 exit_hold_sec，应退出视觉程序
            hold_sec: 当前已连续按住手势 0 的秒数（未按住时为 0）
        """
        active = has_hand and in_range and gesture == GESTURE_STOP
        if not active:
            self.reset()
            return False, False, 0.0

        now = time.time()
        if self._hold_start is None:
            self._hold_start = now
            self._estop_announced = False

        hold_sec = now - self._hold_start
        need_estop = True
        if not self._estop_announced:
            self._estop_announced = True

        should_exit = hold_sec >= self.exit_hold_sec
        return need_estop, should_exit, hold_sec

    def hold_remaining(self, hold_sec: float) -> float:
        return max(0.0, self.exit_hold_sec - hold_sec)


def log_gesture_zero_estop(*, dry_run: bool = False) -> None:
    mode = "DRY-RUN" if dry_run else "EXEC"
    line = f">>> 手势0急停: 停止动作与运动 [{mode}]"
    try:
        from colorama import Fore

        print(f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.RED}{line}", flush=True)
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)


class GestureFiveLostWatch:
    """手部跟踪中：连续无手势5 超过 lost_sec 则请求退回手势识别。"""

    def __init__(self, lost_sec: float = GESTURE_FOLLOW_LOST_SEC):
        self.lost_sec = max(0.1, float(lost_sec))
        self._lost_since: Optional[float] = None

    def reset(self) -> None:
        self._lost_since = None

    def should_return_to_gesture(
        self,
        *,
        engaged: bool,
        is_gesture_five: bool,
    ) -> bool:
        if not engaged:
            self.reset()
            return False
        now = time.time()
        if is_gesture_five:
            self._lost_since = None
            return False
        if self._lost_since is None:
            self._lost_since = now
        return (now - self._lost_since) >= self.lost_sec

    def lost_elapsed(self) -> float:
        if self._lost_since is None:
            return 0.0
        return time.time() - self._lost_since


def log_gesture_zero_exit(hold_sec: float) -> None:
    line = f">>> 手势0保持 {hold_sec:.1f}s，退出视觉识别"
    try:
        from colorama import Fore

        print(f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}{line}", flush=True)
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)
