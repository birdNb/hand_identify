#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容层：脸部跟踪已内嵌至 face_tracker.py（共用 ZED RGB）。"""

from face_tracker import (
    IntegratedFaceTracker,
    FaceTrackingLauncher,
)

DEFAULT_LOCATE_FACE_SCRIPT = ""  # 已弃用子进程，保留常量避免旧代码 import 报错

__all__ = [
    "IntegratedFaceTracker",
    "FaceTrackingLauncher",
    "DEFAULT_LOCATE_FACE_SCRIPT",
]
