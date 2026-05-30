#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 hand_identify 各子模块的 sys.path。"""

import os
import sys


def hand_identify_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_paths(
    *,
    motion: bool = False,
    tracking: bool = False,
    gesture_recognition: bool = False,
) -> str:
    root = hand_identify_root()
    entries = [
        os.path.join(root, "common"),
        root,
    ]
    if gesture_recognition or motion:
        entries.append(os.path.join(root, "gesture_recognition"))
    if motion:
        entries.append(os.path.join(root, "gesture_recognition", "motion"))
    if tracking:
        entries.append(os.path.join(root, "hand_tracking"))
        entries.append(os.path.join(root, "gesture_recognition"))
    for p in entries:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
