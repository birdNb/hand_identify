#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZED + MediaPipe 手部感知: 手势 0~5、掌心 3D、画面横向偏差。

库用法:
    from hand_perception import ZedHandTracker, HandObservation

终端预览:
    python3 hand_perception.py
    python3 hand_perception.py --no-gui
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(tracking=True)

import cv2
import mediapipe as mp
import numpy as np

from gesture_actions import (
    GESTURE_ZERO_EXIT_SEC,
    GESTURE_ZERO_LABEL,
    GestureActionHold,
    GestureZeroHandler,
    GESTURE_ACTION_HOLD_SEC,
    action_hint_for_gesture,
    emit_status_line,
    log_gesture_action_edge,
    log_gesture_zero_estop,
    log_gesture_zero_exit,
)

try:
    import pyzed.sl as sl
except ImportError as exc:
    raise SystemExit(
        "缺少 pyzed, 请先安装 ZED SDK 并: pip install pyzed"
    ) from exc

# ----- 默认参数 -----
TARGET_FPS = 30
PROC_MAX_W = 640
USE_HD1080 = False
LITE_DISPLAY = True
DIST_MIN_M = 0.2
DIST_MAX_M = 2.0
GESTURE_SMOOTH_FRAMES = 5
THUMB_EXTEND_MIN = 0.04
FINGER_WRIST_RATIO = 1.02
PINKY_WRIST_RATIO = 1.01
THUMB_WRIST_RATIO = 1.02
MOVEMENT_THRESHOLD_M = 0.02

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURE_COLORS_BGR = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]

WINDOW_NAME = "Hand Perception"
FULLSCREEN = True

try:
    from colorama import Fore, Style, init as _colorama_init

    _colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

    class _Plain:
        """无 colorama 时退化为无色输出。"""

        CYAN = YELLOW = GREEN = WHITE = RED = MAGENTA = BLUE = ""
        LIGHTYELLOW_EX = LIGHTRED_EX = ""
        RESET_ALL = ""

    Fore = Style = _Plain()  # type: ignore


GESTURE_TERM_COLORS = [
    Fore.RED,
    getattr(Fore, "LIGHTRED_EX", Fore.RED),
    getattr(Fore, "LIGHTYELLOW_EX", Fore.YELLOW),
    Fore.GREEN,
    Fore.BLUE,
    Fore.MAGENTA,
]


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


def compute_proc_size(src_w, src_h, max_w):
    if src_w <= max_w:
        return src_w, src_h
    scale = max_w / src_w
    return int(src_w * scale), int(src_h * scale)


def _dist2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _is_right_hand(lm, handedness_label):
    if handedness_label in ("Left", "Right"):
        return handedness_label == "Right"
    return lm[5].x < lm[17].x


def _finger_extended(lm, tip_id, pip_id, wrist_ratio):
    wrist = lm[0]
    tip, pip = lm[tip_id], lm[pip_id]
    tip_d = _dist2(tip, wrist)
    pip_d = _dist2(pip, wrist)
    if tip_d > pip_d * wrist_ratio:
        return True
    mcp_id = pip_id - 1
    mcp = lm[mcp_id]
    return tip.y < pip.y and pip.y < mcp.y and tip_d > pip_d * 0.98


def _thumb_extended(lm, is_right):
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
    lm = hand_landmarks.landmark
    is_right = _is_right_hand(lm, handedness_label)
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


@dataclass
class HandObservation:
    """单帧手部感知结果, 供显示或底盘控制使用。"""
    gesture: int = -1
    raw_gesture: int = -1
    distance_m: float = 0.0
    dx_norm: float = 0.0
    palm_pos: Optional[Tuple[float, float, float]] = None
    palm_px: Optional[Tuple[int, int]] = None
    in_range: bool = False
    has_hand: bool = False
    direction: str = "无手"

    @property
    def valid_for_control(self) -> bool:
        """五指跟手已禁用，见 hand_tracking/locomotion.py。"""
        return False


def open_zed_camera(use_hd1080=True, dist_min=DIST_MIN_M, dist_max=DIST_MAX_M):
    init_params = sl.InitParameters()
    init_params.camera_resolution = (
        sl.RESOLUTION.HD1080 if use_hd1080 else sl.RESOLUTION.HD720
    )
    init_params.camera_fps = TARGET_FPS
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = dist_min
    init_params.depth_maximum_distance = dist_max

    # 某些情况下(上次进程异常退出/USB带宽抖动)，zed.open 可能返回 STREAM FAILED。
    # 这里做多次重试并逐步降级分辨率，尽量自动恢复。
    last_err = None
    for attempt in range(1, 6):
        zed = sl.Camera()
        err = zed.open(init_params)
        if err == sl.ERROR_CODE.SUCCESS:
            break
        last_err = err
        try:
            zed.close()
        except Exception:
            pass
        if attempt == 1 and use_hd1080:
            # 首次失败且原本是 1080 时，降级到 720 再继续重试
            init_params.camera_resolution = sl.RESOLUTION.HD720
        time.sleep(0.3 * attempt)
    else:
        raise RuntimeError(f"ZED 相机打开失败: {last_err}")

    res = zed.get_camera_information().camera_configuration.resolution
    print(
        f"{Fore.GREEN}ZED 已打开: {res.width}x{res.height}@{TARGET_FPS}fps  "
        f"识别距离 {dist_min:.1f}~{dist_max:.1f}m  "
        f"MediaPipe proc<={PROC_MAX_W}px",
        flush=True,
    )
    return zed


def print_terminal_log(obs: HandObservation, *, face_track_on: bool = False):
    """单行刷新终端日志（与 zed_gesture_recognition 风格一致）。"""
    ts = time.strftime("%H:%M:%S")
    if not obs.has_hand:
        emit_status_line(f"{Fore.CYAN}[{ts}] {Fore.WHITE}未检测到手")
        return

    if obs.gesture < 0:
        if obs.distance_m > 0:
            emit_status_line(
                f"{Fore.CYAN}[{ts}] {Fore.YELLOW}距离 {obs.distance_m:.2f}m "
                f"{Fore.WHITE}{obs.direction}",
            )
        else:
            emit_status_line(
                f"{Fore.CYAN}[{ts}] {Fore.WHITE}有手(无深度) {obs.direction}",
            )
        return

    gcol = GESTURE_TERM_COLORS[min(obs.gesture, 5)]
    dcol = Fore.GREEN if obs.direction == "静止" else Fore.YELLOW
    raw = ""
    if obs.raw_gesture >= 0 and obs.raw_gesture != obs.gesture:
        raw = f" raw={obs.raw_gesture}"
    palm = ""
    if obs.palm_pos is not None:
        x, y, _z = obs.palm_pos
        palm = f" palm X:{x:+.2f} Y:{y:+.2f}"
    hint = action_hint_for_gesture(obs.gesture, face_track_on=face_track_on)
    act_part = ""
    if hint:
        act_part = f" {Fore.MAGENTA}| {hint}{getattr(Style, 'RESET_ALL', '')}"
    ctrl = f" {Fore.MAGENTA}[跟手可用]" if obs.valid_for_control else ""
    emit_status_line(
        f"{Fore.CYAN}[{ts}] "
        f"{gcol}手势:{obs.gesture}{raw}{getattr(Style, 'RESET_ALL', '')} "
        f"{Fore.WHITE}Z:{obs.distance_m:.2f}m dx:{obs.dx_norm:+.2f} "
        f"{dcol}动:{obs.direction}{palm}{act_part}{ctrl}",
    )


class ZedHandTracker:
    """ZED 抓帧 + 手势/深度/横向偏差。"""

    def __init__(
        self,
        dist_min=DIST_MIN_M,
        dist_max=DIST_MAX_M,
        use_hd1080=USE_HD1080,
        proc_max_w=PROC_MAX_W,
        move_threshold=MOVEMENT_THRESHOLD_M,
        lite_display=LITE_DISPLAY,
    ):
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.proc_max_w = max(320, proc_max_w)
        self.lite_display = lite_display
        self.zed = open_zed_camera(
            use_hd1080=use_hd1080,
            dist_min=dist_min,
            dist_max=dist_max,
        )
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._image = sl.Mat()
        self._point_cloud = sl.Mat()
        self._point_cloud_ready = False
        self._runtime = sl.RuntimeParameters()
        self._runtime.confidence_threshold = 50
        self._smoother = GestureSmoother()
        self._movement = MovementTracker(threshold_m=move_threshold)

        info = self.zed.get_camera_information()
        res = info.camera_configuration.resolution
        self.img_w = res.width
        self.img_h = res.height

    def close(self):
        self._hands.close()
        self.zed.close()

    def process_frame(
        self,
        draw_landmarks=True,
        face_tracker=None,
    ) -> Tuple[np.ndarray, HandObservation]:
        obs = HandObservation()
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8), obs

        self.zed.retrieve_image(self._image, sl.VIEW.LEFT)
        self._point_cloud_ready = False

        frame = self._image.get_data()
        if frame is None:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8), obs
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img_w = self._image.get_width()
        img_h = self._image.get_height()
        cx_img = img_w / 2.0

        proc_w, proc_h = compute_proc_size(img_w, img_h, self.proc_max_w)
        use_lite = (
            self.lite_display
            and (proc_w, proc_h) != (img_w, img_h)
        )
        if (proc_w, proc_h) != (img_w, img_h):
            proc_bgr = cv2.resize(
                frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA,
            )
        else:
            proc_bgr = frame
        display_bgr = proc_bgr if use_lite else frame
        disp_sx = proc_w / max(img_w, 1) if use_lite else 1.0
        disp_sy = proc_h / max(img_h, 1) if use_lite else 1.0

        # 手部 + 脸部 MediaPipe 共用同一份降采样 RGB，避免重复 cvtColor/上传
        rgb_mp = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        if not rgb_mp.flags["C_CONTIGUOUS"]:
            rgb_mp = np.ascontiguousarray(rgb_mp)
        rgb_mp.flags.writeable = False

        if face_tracker is not None and face_tracker.is_active:
            face_tracker.process_shared_rgb(
                rgb_mp, display_bgr, proc_w, proc_h,
            )
            if draw_landmarks:
                face_tracker.draw_overlay(display_bgr)

        results = self._hands.process(rgb_mp)

        if not results.multi_hand_landmarks:
            self._smoother.reset()
            self._movement.reset()
            return display_bgr, obs

        if not self._point_cloud_ready:
            self.zed.retrieve_measure(self._point_cloud, sl.MEASURE.XYZ)
            self._point_cloud_ready = True

        obs.has_hand = True
        handedness_list = results.multi_handedness or []
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            if draw_landmarks:
                mp_drawing.draw_landmarks(
                    display_bgr, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4,
                    ),
                    mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2,
                    ),
                )

            palm_pos, palm_px = calculate_palm_position(
                hand_lm, img_w, img_h, self._point_cloud,
            )
            obs.palm_pos = palm_pos
            obs.palm_px = palm_px

            if palm_pos is not None:
                obs.distance_m = palm_pos[2]
                obs.in_range = is_distance_in_range(
                    obs.distance_m, self.dist_min, self.dist_max,
                )
                if palm_px is not None:
                    obs.dx_norm = clamp(
                        (palm_px[0] - cx_img) / (img_w / 2.0), -1.0, 1.0,
                    )
                    if draw_landmarks and use_lite:
                        px_d = (
                            int(palm_px[0] * disp_sx),
                            int(palm_px[1] * disp_sy),
                        )
                        col = (0, 255, 0) if obs.in_range else (0, 165, 255)
                        cv2.circle(display_bgr, px_d, 6, col, -1)
                if not obs.in_range:
                    hint = distance_range_hint(
                        obs.distance_m, self.dist_min, self.dist_max,
                    )
                    obs.direction = f"超出范围({hint})"
                    self._smoother.reset()
                    self._movement.reset()
                else:
                    h_label = None
                    if idx < len(handedness_list):
                        h_label = handedness_list[idx].classification[0].label
                    obs.raw_gesture, _ = recognize_gesture(
                        hand_lm, handedness_label=h_label,
                    )
                    obs.gesture = self._smoother.update(obs.raw_gesture)
                    obs.direction = self._movement.update(palm_pos)
            else:
                obs.direction = self._movement.update(None)
            break

        return display_bgr, obs


def main():
    parser = argparse.ArgumentParser(
        description="ZED 手部感知预览（手势 0~5 + 掌心 3D）",
    )
    parser.add_argument("--no-gui", action="store_true", help="不显示 OpenCV 窗口")
    parser.add_argument(
        "--hd1080", action="store_true", help="使用 HD1080（默认 HD720）",
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
        "--move-threshold", type=float, default=MOVEMENT_THRESHOLD_M,
        help="移动判定阈值/米",
    )
    parser.add_argument(
        "--zero-exit-sec", type=float, default=GESTURE_ZERO_EXIT_SEC,
        help="手势0持续按住多少秒后退出 (默认 5)",
    )
    args = parser.parse_args()

    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")

    if not args.no_gui and not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    print(f"{Fore.GREEN}正在打开 ZED 与 MediaPipe...", flush=True)
    tracker = ZedHandTracker(
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        use_hd1080=args.hd1080,
        proc_max_w=args.proc_max_w,
        move_threshold=args.move_threshold,
    )

    is_fullscreen = FULLSCREEN
    if not args.no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if FULLSCREEN:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
            )

    print(
        f"{Fore.GREEN}系统就绪。请将手置于 {args.dist_min:.1f}~{args.dist_max:.1f}m 内。"
        f"{Fore.YELLOW} ESC 退出, f 切换全屏。",
        flush=True,
    )
    print(
        f"{Fore.GREEN}手势动作: 0急停/按住{args.zero_exit_sec:.0f}s退出 "
        f"1撒娇扭腰 2抬手 3挥双手 4踢球",
        flush=True,
    )

    last_logged_gesture = -1
    last_logged_confirmed = -1
    zero_handler = GestureZeroHandler(exit_hold_sec=args.zero_exit_sec)
    action_hold = GestureActionHold(hold_sec=GESTURE_ACTION_HOLD_SEC)
    try:
        while True:
            frame, obs = tracker.process_frame(draw_landmarks=not args.no_gui)
            zero_estop, zero_exit, zero_hold = zero_handler.update(
                obs.gesture,
                has_hand=obs.has_hand,
                in_range=obs.in_range,
            )
            if zero_estop and zero_hold < 0.15:
                log_gesture_zero_estop(dry_run=True)
            if zero_exit:
                log_gesture_zero_exit(zero_hold)
                break
            confirmed = action_hold.update(
                obs.gesture,
                has_hand=obs.has_hand,
                in_range=obs.in_range,
            )
            if (
                confirmed >= 0
                and confirmed != last_logged_confirmed
            ):
                log_gesture_action_edge(
                    confirmed,
                    last_logged_confirmed,
                    in_range=obs.in_range,
                    has_hand=obs.has_hand,
                    preview_only=True,
                )
            last_logged_gesture = obs.gesture
            last_logged_confirmed = confirmed if confirmed >= 0 else -1
            print_terminal_log(obs)

            if not args.no_gui:
                if zero_estop:
                    remain = zero_handler.hold_remaining(zero_hold)
                    cv2.putText(
                        frame,
                        f"G0 {GESTURE_ZERO_LABEL} exit {remain:.1f}s",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA,
                    )
                elif obs.gesture >= 0:
                    col = GESTURE_COLORS_BGR[min(obs.gesture, 5)]
                    cv2.putText(
                        frame, f"Gesture: {obs.gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3, cv2.LINE_AA,
                    )
                elif obs.has_hand:
                    cv2.putText(
                        frame, obs.direction, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                        cv2.LINE_AA,
                    )
                if obs.palm_px is not None:
                    dot_col = (0, 255, 0) if obs.in_range else (0, 165, 255)
                    cv2.circle(frame, obs.palm_px, 8, dot_col, -1)
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
        print(f"\n{Fore.YELLOW}用户中断", flush=True)
    finally:
        tracker.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print(f"{Fore.GREEN}已退出", flush=True)


if __name__ == "__main__":
    main()
