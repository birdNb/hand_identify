#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手势 5 确认后释放 ZED，切换到 hand_tracking（无 GUI）。"""

from __future__ import annotations

import os
import sys


def hand_identify_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def exec_gesture_recognition(extra_args: list[str] | None = None) -> None:
    """手部跟踪超时无五指后，切回手势识别（替换当前进程）。"""
    root = hand_identify_root()
    script = os.path.join(root, "start_gesture_recognition.sh")
    if not os.path.isfile(script):
        raise FileNotFoundError(script)
    argv = ["bash", script]
    if extra_args:
        argv.extend(extra_args)
    os.chdir(root)
    os.execv("/bin/bash", argv)


def exec_hand_tracking(extra_args: list[str] | None = None) -> None:
    """
    用 start_hand_tracking.sh 替换当前进程（默认无 GUI，见 distance_hold.py）。
    调用前须已关闭 ZED 与 ROS 节点，避免相机占用。
    """
    root = hand_identify_root()
    script = os.path.join(root, "start_hand_tracking.sh")
    if not os.path.isfile(script):
        raise FileNotFoundError(script)
    argv = ["bash", script]
    if extra_args:
        argv.extend(extra_args)
    os.chdir(root)
    os.execv("/bin/bash", argv)


def release_before_handoff(
    *,
    face_track=None,
    motion=None,
    zed=None,
    hands=None,
    no_gui: bool = True,
) -> None:
    """关闭脸跟踪、手势动作与 ZED，便于 hand_tracking 独占相机。"""
    if face_track is not None:
        try:
            face_track.shutdown()
        except Exception:
            pass
    if motion is not None:
        try:
            motion.shutdown(fast=True)
        except Exception:
            pass
    if zed is not None:
        try:
            zed.close()
        except Exception:
            pass
    if hands is not None:
        try:
            hands.close()
        except Exception:
            pass
    if not no_gui:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
    try:
        from shutdown import rospy_shutdown_if_init

        rospy_shutdown_if_init()
    except Exception:
        pass


def release_before_gesture_return(
    *,
    tracker=None,
    pub=None,
    last_pub_active: bool = False,
    dry_run: bool = False,
    no_gui: bool = True,
) -> None:
    """关闭手部跟踪与 ZED，便于手势识别重新打开相机。"""
    if not dry_run and pub is not None and last_pub_active:
        try:
            from geometry_msgs.msg import Twist

            pub.publish(Twist())
        except Exception:
            pass
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    if not no_gui:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
    try:
        from shutdown import rospy_shutdown_if_init

        rospy_shutdown_if_init()
    except Exception:
        pass


def log_follow_handoff(*, preview: bool = False) -> None:
    from gesture_actions import GESTURE_FOLLOW_HOLD_SEC

    mode = "预览(不切换)" if preview else "切换"
    line = (
        f">>> 手势5稳定{GESTURE_FOLLOW_HOLD_SEC:.0f}s: "
        "切换 → start_hand_tracking.sh（无 GUI，停脸跟踪）"
    )
    try:
        import time
        from colorama import Fore

        print(
            f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.GREEN}{line}",
            flush=True,
        )
    except ImportError:
        print(f"\n{line}", flush=True)


def log_return_to_gesture(*, lost_sec: float) -> None:
    line = f">>> 跟手无五指({lost_sec:.0f}s): 切回 start_gesture_recognition.sh"
    try:
        import time
        from colorama import Fore

        print(
            f"\n{Fore.CYAN}[{time.strftime('%H:%M:%S')}] {Fore.YELLOW}{line}",
            flush=True,
        )
    except ImportError:
        print(f"\n{line}", flush=True)
