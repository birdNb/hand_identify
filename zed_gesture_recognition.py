#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================
# zed_gesture_recognition.py  ——  ZED Mini 手势数字(0-5) + 手掌3D位置 + 移动方向
#
# 架构:
#   ZED SDK 取左眼 RGB + 深度/点云 → MediaPipe Hands → 手势 / 3D位置 / 方向
#   画面叠加 + 终端彩色日志 (colorama)
#
# 运行:
#   python3 zed_gesture_recognition.py
#   python3 zed_gesture_recognition.py --no-gui
# ==============================================================

import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyzed.sl as sl
except ImportError as exc:
    raise SystemExit(
        "缺少 pyzed, 请先安装 ZED SDK 并: pip install pyzed"
    ) from exc

from colorama import Fore, Style, init

from gesture_actions import (
    GESTURE_HEAD_NOD,
    GESTURE_ZERO_EXIT_SEC,
    GESTURE_ZERO_LABEL,
    GestureZeroHandler,
    action_hint_for_gesture,
    emit_status_line,
    log_gesture_action_edge,
    log_gesture_head_nod,
    log_gesture_zero_estop,
    log_gesture_zero_exit,
)

init(autoreset=True)

# ----- MediaPipe -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----- 手势画面颜色 BGR -----
GESTURE_COLORS_BGR = [
    (0, 0, 255),      # 0 红
    (0, 165, 255),    # 1 橙
    (0, 255, 255),    # 2 黄
    (0, 255, 0),      # 3 绿
    (255, 0, 0),      # 4 蓝
    (255, 0, 255),    # 5 紫
]

# ----- 终端手势颜色 (colorama 无 ORANGE, 用 LIGHTYELLOW 代替) -----
GESTURE_TERM_COLORS = [
    Fore.RED,
    Fore.LIGHTRED_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.GREEN,
    Fore.BLUE,
    Fore.MAGENTA,
]

MOVEMENT_THRESHOLD_M = 0.02   # 2cm 移动阈值
DIST_MIN_M = 0.2              # 有效识别最近距离(米)
DIST_MAX_M = 2.0              # 有效识别最远距离(米)
FULLSCREEN = True
WINDOW_NAME = "ZED Mini Gesture"
GESTURE_SMOOTH_FRAMES = 5     # 手势结果滑动窗口, 抑制抖动与误检
THUMB_EXTEND_MIN = 0.04       # 拇指水平张开阈值(归一化)
# 指尖到手腕距离 > 指关节到手腕 * 比例 → 判定伸直(张开五指更稳)
FINGER_WRIST_RATIO = 1.02     # 食/中/无名
PINKY_WRIST_RATIO = 1.01      # 小指略放宽, 避免 5 判成 3/4
THUMB_WRIST_RATIO = 1.02

# ----- 相机配置 (对齐 locate_face.py) -----
# ZED Mini 双目 V4L2 可选: 4416x1242@15 / 3840x1080@30 / 2560x720@60 / 1344x376@100
# ZED SDK 单目: HD1080=1920x1080@30 (比 HD720 更清晰, 与 1280x720 左眼同量级且更高)
TARGET_FPS = 30
PROC_MAX_W = 960                # MediaPipe 输入最大宽度(全分辨率显示+深度对齐)
USE_HD1080 = True               # True=HD1080, False=HD720


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def is_distance_in_range(z_m, dist_min=DIST_MIN_M, dist_max=DIST_MAX_M):
    return dist_min <= z_m <= dist_max


def distance_range_hint(z_m, dist_min=DIST_MIN_M, dist_max=DIST_MAX_M):
    if z_m < dist_min:
        return f"过近(<{dist_min:.1f}m)"
    if z_m > dist_max:
        return f"过远(>{dist_max:.1f}m)"
    return ""


def _dist2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _is_right_hand(lm, handedness_label):
    """MediaPipe handedness 优先; 否则用掌心 MCP 横向关系推断。"""
    if handedness_label in ("Left", "Right"):
        return handedness_label == "Right"
    return lm[5].x < lm[17].x


def _finger_extended(lm, tip_id, pip_id, wrist_ratio):
    """食指~小指: 指尖比 PIP 更远离手腕则伸直(五指张开时比 y 坐标链更稳)。"""
    wrist = lm[0]
    tip, pip = lm[tip_id], lm[pip_id]
    tip_d = _dist2(tip, wrist)
    pip_d = _dist2(pip, wrist)
    if tip_d > pip_d * wrist_ratio:
        return True
    # 侧向举手时备用: tip/pip/mcp 仍呈伸展链
    mcp_id = pip_id - 1
    mcp = lm[mcp_id]
    return tip.y < pip.y and pip.y < mcp.y and tip_d > pip_d * 0.98


def _thumb_extended(lm, is_right):
    """拇指: 先看到手腕距离, 再用左右手水平规则, 兼顾 1 不误检与 5 不漏检。"""
    wrist = lm[0]
    tip, ip, mcp = lm[4], lm[3], lm[2]
    if _dist2(tip, wrist) > _dist2(ip, wrist) * THUMB_WRIST_RATIO:
        return True
    spread = (tip.x - ip.x) if is_right else (ip.x - tip.x)
    if spread < THUMB_EXTEND_MIN:
        return False
    if is_right:
        return tip.x > ip.x and tip.x > mcp.x
    return tip.x < ip.x and tip.x < mcp.x


def recognize_gesture(hand_landmarks, handedness_label=None):
    """根据 21 关键点识别手势数字 0-5 (伸直手指数)。"""
    lm = hand_landmarks.landmark
    is_right = _is_right_hand(lm, handedness_label)

    # 食指~小指: (tip, pip) + 手腕距离比例
    chains = [
        (8, 6, FINGER_WRIST_RATIO),
        (12, 10, FINGER_WRIST_RATIO),
        (16, 14, FINGER_WRIST_RATIO),
        (20, 18, PINKY_WRIST_RATIO),
    ]
    fingers_up = [_thumb_extended(lm, is_right)]
    for tip_id, pip_id, ratio in chains:
        fingers_up.append(_finger_extended(lm, tip_id, pip_id, ratio))

    return sum(fingers_up), fingers_up


class GestureSmoother:
    """滑动窗口众数, 减少 1→2、2→3 这类瞬时多计一根指。"""

    def __init__(self, window=GESTURE_SMOOTH_FRAMES):
        self._hist = []
        self._window = max(1, window)

    def reset(self):
        self._hist.clear()

    def update(self, raw_gesture):
        if raw_gesture < 0:
            self.reset()
            return -1
        self._hist.append(raw_gesture)
        if len(self._hist) > self._window:
            self._hist.pop(0)
        return max(set(self._hist), key=self._hist.count)


def calculate_palm_position(hand_landmarks, img_w, img_h, point_cloud):
    """手掌中心 3D 坐标 (米), 像素中心点。优先用 XYZ 点云。"""
    wrist = hand_landmarks.landmark[0]
    mid_base = hand_landmarks.landmark[9]
    cx_n = (wrist.x + mid_base.x) / 2.0
    cy_n = (wrist.y + mid_base.y) / 2.0
    px = int(clamp(cx_n * img_w, 0, img_w - 1))
    py = int(clamp(cy_n * img_h, 0, img_h - 1))

    err, pt = point_cloud.get_value(px, py)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, (px, py)

    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    if np.isnan(x) or np.isnan(y) or np.isnan(z):
        return None, (px, py)
    return (x, y, z), (px, py)


class MovementTracker:
    """帧间手掌位移 → 方向文字。"""

    def __init__(self, threshold_m=MOVEMENT_THRESHOLD_M):
        self._prev = None
        self._threshold = threshold_m

    def reset(self):
        self._prev = None

    def update(self, pos_3d):
        if pos_3d is None:
            self._prev = None
            return "无深度"

        if self._prev is None:
            self._prev = pos_3d
            return "静止"

        dx = pos_3d[0] - self._prev[0]
        dy = pos_3d[1] - self._prev[1]
        dz = pos_3d[2] - self._prev[2]
        self._prev = pos_3d

        parts = []
        if abs(dx) > self._threshold:
            parts.append("右" if dx > 0 else "左")
        if abs(dy) > self._threshold:
            parts.append("下" if dy > 0 else "上")
        if abs(dz) > self._threshold:
            parts.append("后" if dz > 0 else "前")

        return "、".join(parts) if parts else "静止"


def draw_overlay_log(frame, text, position, color=(0, 255, 0),
                      font_scale=0.6, thickness=2):
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA,
    )


def print_terminal_log(
    gesture, distance, direction,
    in_range=True, has_hand=False, face_track_on=False,
):
    ts = time.strftime("%H:%M:%S")
    if gesture < 0:
        if has_hand and distance > 0:
            emit_status_line(
                f"{Fore.CYAN}[{ts}] {Fore.YELLOW}距离 {distance:.2f}m "
                f"{Fore.WHITE}{direction}",
            )
        else:
            emit_status_line(
                f"{Fore.CYAN}[{ts}] {Fore.WHITE}未检测到手",
            )
        return

    gcol = GESTURE_TERM_COLORS[min(gesture, 5)]
    dcol = Fore.GREEN if direction == "静止" else Fore.YELLOW
    hint = action_hint_for_gesture(gesture, face_track_on=face_track_on)
    act_part = ""
    if hint:
        act_part = f" {Fore.MAGENTA}| {hint}{Style.RESET_ALL}"
    emit_status_line(
        f"{Fore.CYAN}[{ts}] "
        f"{gcol}手势:{gesture}{Style.RESET_ALL} "
        f"{Fore.WHITE}距离:{distance:.2f}m "
        f"{dcol}方向:{direction}{act_part}",
    )


def compute_proc_size(src_w, src_h, max_w):
    """与 locate_face 相同: 等比缩小 MediaPipe 输入, 归一化坐标仍映射回原图。"""
    if src_w <= max_w:
        return src_w, src_h
    scale = max_w / src_w
    return int(src_w * scale), int(src_h * scale)


def open_zed_camera(use_hd1080=True, dist_min=DIST_MIN_M, dist_max=DIST_MAX_M):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = (
        sl.RESOLUTION.HD1080 if use_hd1080 else sl.RESOLUTION.HD720
    )
    init_params.camera_fps = TARGET_FPS
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = dist_min
    init_params.depth_maximum_distance = dist_max

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS and use_hd1080:
        print(Fore.YELLOW + "HD1080 打开失败, 回退 HD720...")
        zed.close()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED 相机打开失败: {err}")

    cam_info = zed.get_camera_information()
    res = cam_info.camera_configuration.resolution
    print(
        Fore.GREEN
        + f"ZED 已打开: {res.width}x{res.height}@{TARGET_FPS}fps  "
        f"识别距离 {dist_min:.1f}~{dist_max:.1f}m  "
        f"MediaPipe proc<={PROC_MAX_W}px"
    )
    return zed


def main():
    parser = argparse.ArgumentParser(description="ZED Mini 手势数字 + 3D 跟踪")
    parser.add_argument("--no-gui", action="store_true", help="不显示窗口")
    parser.add_argument(
        "--move-threshold", type=float, default=MOVEMENT_THRESHOLD_M,
        help="移动判定阈值(米), 默认 0.02",
    )
    parser.add_argument(
        "--hd720", action="store_true",
        help="使用 HD720 (默认 HD1080 更清晰)",
    )
    parser.add_argument(
        "--proc-max-w", type=int, default=PROC_MAX_W,
        help=f"MediaPipe 最大输入宽度 (默认 {PROC_MAX_W})",
    )
    parser.add_argument(
        "--dist-min", type=float, default=DIST_MIN_M,
        help=f"最近识别距离/米 (默认 {DIST_MIN_M})",
    )
    parser.add_argument(
        "--dist-max", type=float, default=DIST_MAX_M,
        help=f"最远识别距离/米 (默认 {DIST_MAX_M})",
    )
    parser.add_argument(
        "--zero-exit-sec", type=float, default=GESTURE_ZERO_EXIT_SEC,
        help="手势0持续按住多少秒后退出 (默认 5)",
    )
    args = parser.parse_args()
    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")
    dist_min, dist_max = args.dist_min, args.dist_max

    if not args.no_gui and not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    zed = open_zed_camera(
        use_hd1080=not args.hd720, dist_min=dist_min, dist_max=dist_max,
    )
    proc_max_w = max(320, args.proc_max_w)
    image = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50

    tracker = MovementTracker(threshold_m=args.move_threshold)
    gesture_smoother = GestureSmoother()

    is_fullscreen = FULLSCREEN
    if not args.no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if FULLSCREEN:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
            )

    print(
        Fore.GREEN
        + f"系统启动成功！请将手放在相机前 {dist_min:.1f}~{dist_max:.1f} 米范围内。"
    )
    print(Fore.YELLOW + "按 ESC 退出, 'f' 切换全屏。")
    print(
        Fore.GREEN
        + "手势动作: 0急停/按住"
        f"{args.zero_exit_sec:.0f}s退出 "
        + "1抬头20° 2抬手 3挥双手 4踢球; "
        + "切换手势打印动作日志",
    )

    last_logged_gesture = -1
    zero_handler = GestureZeroHandler(exit_hold_sec=args.zero_exit_sec)

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

            frame = image.get_data()
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            img_w = image.get_width()
            img_h = image.get_height()

            proc_w, proc_h = compute_proc_size(img_w, img_h, proc_max_w)
            if (proc_w, proc_h) != (img_w, img_h):
                proc_bgr = cv2.resize(frame, (proc_w, proc_h))
            else:
                proc_bgr = frame
            rgb_mp = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            rgb_mp.flags.writeable = False
            results = hands.process(rgb_mp)

            gesture = -1
            raw_gesture = -1
            distance = 0.0
            direction = "无手"
            in_range = True
            palm_center_px = None
            palm_pos = None

            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness or []
                for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(121, 22, 76), thickness=2, circle_radius=4,
                        ),
                        mp_drawing.DrawingSpec(
                            color=(250, 44, 250), thickness=2, circle_radius=2,
                        ),
                    )

                    palm_pos, palm_center_px = calculate_palm_position(
                        hand_lm, img_w, img_h, point_cloud,
                    )

                    if palm_pos is not None:
                        distance = palm_pos[2]
                        in_range = is_distance_in_range(
                            distance, dist_min, dist_max,
                        )
                        if palm_center_px is not None:
                            dot_col = (0, 255, 0) if in_range else (0, 165, 255)
                            cv2.circle(
                                frame, palm_center_px, 8, dot_col, -1,
                            )
                        draw_overlay_log(
                            frame, f"X: {palm_pos[0]:+.2f}m",
                            (10, 30), (255, 0, 0),
                        )
                        draw_overlay_log(
                            frame, f"Y: {palm_pos[1]:+.2f}m",
                            (10, 60), (0, 255, 0),
                        )
                        z_col = (0, 0, 255) if in_range else (0, 165, 255)
                        draw_overlay_log(
                            frame, f"Z: {palm_pos[2]:+.2f}m",
                            (10, 90), z_col,
                        )
                        if not in_range:
                            hint = distance_range_hint(
                                distance, dist_min, dist_max,
                            )
                            direction = f"超出范围({hint})"
                            gesture_smoother.reset()
                            tracker.reset()
                        else:
                            h_label = None
                            if idx < len(handedness_list):
                                h_label = (
                                    handedness_list[idx]
                                    .classification[0].label
                                )
                            raw_gesture, _ = recognize_gesture(
                                hand_lm, handedness_label=h_label,
                            )
                            gesture = gesture_smoother.update(raw_gesture)
                            direction = tracker.update(palm_pos)
                    else:
                        in_range = False
                        direction = tracker.update(None)
            else:
                tracker.reset()
                gesture_smoother.reset()
                direction = "无手"

            draw_overlay_log(
                frame,
                f"Range: {dist_min:.1f}~{dist_max:.1f}m",
                (10, img_h - 20), (200, 200, 200), font_scale=0.5, thickness=1,
            )

            if zero_estop:
                remain = zero_handler.hold_remaining(zero_hold)
                draw_overlay_log(
                    frame,
                    f"G0 {GESTURE_ZERO_LABEL} exit {remain:.1f}s",
                    (10, 130), (0, 0, 255), font_scale=0.9, thickness=2,
                )
            elif gesture >= 0:
                col = GESTURE_COLORS_BGR[min(gesture, 5)]
                gtext = f"Gesture: {gesture}"
                if raw_gesture >= 0 and raw_gesture != gesture:
                    gtext += f" (raw {raw_gesture})"
                draw_overlay_log(
                    frame, gtext, (10, 130),
                    col, font_scale=1.0, thickness=3,
                )
                dcol = (0, 255, 0) if direction == "静止" else (0, 255, 255)
                draw_overlay_log(
                    frame, f"Dir: {direction}", (10, 170), dcol,
                )
            elif results.multi_hand_landmarks and not in_range:
                draw_overlay_log(
                    frame, direction, (10, 130), (0, 165, 255),
                    font_scale=0.8, thickness=2,
                )
            else:
                draw_overlay_log(
                    frame, "No hand", (10, 130), (128, 128, 128),
                )

            has_hand = bool(results.multi_hand_landmarks)
            zero_estop, zero_exit, zero_hold = zero_handler.update(
                gesture, has_hand=has_hand, in_range=in_range,
            )
            if zero_estop and zero_hold < 0.15:
                log_gesture_zero_estop(dry_run=True)
            if zero_exit:
                log_gesture_zero_exit(zero_hold)
                break

            if gesture != 0 and gesture != last_logged_gesture:
                if gesture == GESTURE_HEAD_NOD:
                    if has_hand and in_range:
                        log_gesture_head_nod(dry_run=True)
                else:
                    log_gesture_action_edge(
                        gesture,
                        last_logged_gesture,
                        in_range=in_range,
                        has_hand=has_hand,
                        preview_only=True,
                    )
            last_logged_gesture = gesture

            print_terminal_log(
                gesture, distance, direction,
                in_range=in_range, has_hand=has_hand,
            )

            if not args.no_gui:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("f"):
                    is_fullscreen = not is_fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_fullscreen
                        else cv2.WINDOW_NORMAL,
                    )
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n用户中断")

    finally:
        zed.close()
        hands.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print(Fore.GREEN + "\n程序已退出")


if __name__ == "__main__":
    main()
