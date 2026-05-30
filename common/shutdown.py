#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""进程级退出请求（Ctrl+C / SIGTERM）。"""

import signal
import sys
from typing import Callable, Optional

_requested = False
_handlers_installed = False


def is_requested() -> bool:
    return _requested


def request(msg: str = "用户中断") -> None:
    global _requested
    _requested = True
    if msg:
        print(f"\n{msg}", flush=True)


def install_handlers(
    on_stop: Optional[Callable[[], None]] = None,
) -> None:
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True

    def _handler(signum, _frame):
        request("收到退出信号 (Ctrl+C)，正在关闭...")
        if on_stop is not None:
            try:
                on_stop()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _handler)


def rospy_shutdown_if_init(reason: str = "exit") -> None:
    try:
        import rospy
        if rospy.core.is_initialized() and not rospy.is_shutdown():
            rospy.signal_shutdown(reason)
    except Exception:
        pass


def exit_ros_if_init(code: int = 0) -> None:
    rospy_shutdown_if_init()
    sys.exit(code)
