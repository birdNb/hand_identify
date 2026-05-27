#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 0~5 与动作库映射（zed 预览与 hand_follow 共用）。"""

import time
from typing import Dict, Optional, Tuple

GESTURE_STOP = 0
GESTURE_HEAD_NOD = 1
GESTURE_FACE_TRACK = GESTURE_HEAD_NOD  # 兼容旧名
GESTURE_FOLLOW = 5
GESTURE_ZERO_EXIT_SEC = 5.0
GESTURE_ZERO_LABEL = "急停/退出"
GESTURE_HEAD_NOD_LABEL = "抬头20°"
GESTURE_FACE_TRACK_LABEL = "脸部跟踪(常开)"

# gesture -> (action_name, joy_key_combo, keepalive_sec 已废弃，实际时长见 hand_action_library.ACTION_DURATION_SEC)
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
        return "跟手"
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


def log_gesture_head_nod(*, dry_run: bool = False) -> None:
    mode = "DRY-RUN" if dry_run else "EXEC"
    line = f">>> 手势1: {GESTURE_HEAD_NOD_LABEL} 仅pitch (yaw保持) [{mode}]"
    try:
        from colorama import Fore

        print(
            f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}{line}",
            flush=True,
        )
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)


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


def log_gesture_zero_exit(hold_sec: float) -> None:
    line = f">>> 手势0保持 {hold_sec:.1f}s，退出视觉识别"
    try:
        from colorama import Fore

        print(f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}{line}", flush=True)
    except ImportError:
        print(f"\n[{time.strftime('%H:%M:%S')}] {line}", flush=True)
