#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 sim2real / ROS 工作空间的 Python 包加入 sys.path（免手动 source）。"""

import os
import sys

_SIM2REAL_PY_PATHS = (
    "~/sim2real/devel/lib/python3/dist-packages",
    "~/sim2real/install/lib/python3/dist-packages",
)


def ensure_ros_python_path() -> bool:
    """返回是否找到 sim2real_msg 所在目录。"""
    added = False
    for rel in _SIM2REAL_PY_PATHS:
        p = os.path.expanduser(rel)
        if not os.path.isdir(p):
            continue
        if p not in sys.path:
            sys.path.insert(0, p)
            added = True
    try:
        import sim2real_msg  # noqa: F401
        return True
    except ImportError:
        return added


def require_sim2real_msg():
    if ensure_ros_python_path():
        return
    raise SystemExit(
        "找不到 sim2real_msg。请先:\n"
        "  source /opt/ros/noetic/setup.bash\n"
        "  source ~/sim2real/devel/setup.bash   # 或 install/setup.bash\n"
        "再运行本程序。"
    )
