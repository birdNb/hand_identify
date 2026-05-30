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
#   python3 zed_gesture_recognition.py          # 默认执行手势动作(需 ROS)
#   python3 zed_gesture_recognition.py --preview # 仅识别与日志，不发指令
#   python3 zed_gesture_recognition.py --no-gui
# ==============================================================

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(gesture_recognition=True, motion=True)

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
    GESTURE_ACTION_HOLD_SEC,
    GESTURE_FOLLOW,
    GESTURE_FOLLOW_HOLD_SEC,
    GESTURE_STOP,
    GESTURE_ZERO_EXIT_SEC,
    GESTURE_ZERO_LABEL,
    GestureActionHold,
    GestureZeroHandler,
    action_hint_for_gesture,
    emit_status_line,
    log_gesture_action_edge,
    log_gesture_zero_estop,
    log_gesture_zero_exit,
)
from handoff import exec_hand_tracking, log_follow_handoff, release_before_handoff
from shutdown import install_handlers, is_requested, request, rospy_shutdown_if_init

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
PROC_MAX_W = 560                # MediaPipe 输入宽度（手势优先）
PROC_MAX_W_FULL = 960           # --full-res-gui 时用
USE_HD1080 = False              # 默认 HD720 更流畅
LITE_DISPLAY = True             # 显示用降采样画面
DRAW_LANDMARKS = False          # 关闭骨架绘制以减负
DEPTH_EVERY_N = 4               # 无手时降频取点云
FACE_EVERY_N = 2                # 隔帧做人脸推理
LOG_INTERVAL_SEC = 0.25         # 终端日志刷新间隔


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


def _zed_busy_hint() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["pgrep", "-af", "zed_gesture_recognition"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return (
                "\n可能仍有进程占用 ZED（勿 Ctrl+Z 挂起）：\n"
                f"{out}\n"
                "处理: pkill -f zed_gesture_recognition.py  或 fg 后 Esc/Ctrl+C 正常退出"
            )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return (
        "\n可尝试: 拔掉 ZED USB 等 3s 再插；"
        "确认无 locate_face / 其它 ZED 程序；"
        "pkill -f zed_gesture_recognition.py"
    )


def open_zed_camera(use_hd1080=True, dist_min=DIST_MIN_M, dist_max=DIST_MAX_M):
    """打开 ZED；失败时降分辨率/帧率重试。"""
    attempts = []
    if use_hd1080:
        attempts.append((sl.RESOLUTION.HD1080, TARGET_FPS))
    attempts.extend([
        (sl.RESOLUTION.HD720, TARGET_FPS),
        (sl.RESOLUTION.HD720, 15),
        (sl.RESOLUTION.VGA, 15),
    ])

    last_err = sl.ERROR_CODE.CAMERA_NOT_DETECTED
    zed = sl.Camera()
    for res, fps in attempts:
        init_params = sl.InitParameters()
        init_params.camera_resolution = res
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = dist_min
        init_params.depth_maximum_distance = dist_max

        for try_i in range(3):
            err = zed.open(init_params)
            if err == sl.ERROR_CODE.SUCCESS:
                cam_info = zed.get_camera_information()
                r = cam_info.camera_configuration.resolution
                print(
                    Fore.GREEN
                    + f"ZED 已打开: {r.width}x{r.height}@{fps}fps  "
                    f"识别距离 {dist_min:.1f}~{dist_max:.1f}m  "
                    f"MediaPipe proc<={PROC_MAX_W}px"
                )
                return zed
            last_err = err
            zed.close()
            if try_i < 2:
                time.sleep(0.8)
        label = f"{res}@{fps}fps"
        print(Fore.YELLOW + f"ZED 打开失败 ({label}): {err}")

    raise RuntimeError(
        f"ZED 相机打开失败: {last_err}{_zed_busy_hint()}",
    )

def main():
    parser = argparse.ArgumentParser(description="ZED Mini 手势数字 + 3D 跟踪")
    parser.add_argument("--no-gui", action="store_true", help="不显示窗口")
    parser.add_argument(
        "--move-threshold", type=float, default=MOVEMENT_THRESHOLD_M,
        help="移动判定阈值(米), 默认 0.02",
    )
    parser.add_argument(
        "--hd1080", action="store_true",
        help="使用 HD1080 (默认 HD720 更流畅)",
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
    parser.add_argument(
        "--preview", action="store_true",
        help="仅预览：不初始化 ROS，不执行机器人动作",
    )
    parser.add_argument(
        "--no-actions", action="store_true",
        help="禁用动作库(2~4)与手势1点头",
    )
    parser.add_argument(
        "--no-coquette", action="store_true",
        help="禁用手势1撒娇扭腰",
    )
    parser.add_argument(
        "--no-joy", action="store_true",
        help="不监听手柄仲裁",
    )
    parser.add_argument(
        "--no-fsm", action="store_true",
        help="不等待 FSM=EXEC_DEFAULT",
    )
    parser.add_argument(
        "--gesture-hold-sec", type=float, default=GESTURE_ACTION_HOLD_SEC,
        help="手势1~4稳定多少秒后触发 (默认 2)",
    )
    parser.add_argument(
        "--no-face-track", action="store_true",
        help="禁用内嵌脸部跟踪(默认开启, 与 locate_face 同控制律)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="性能模式: proc 480 + 脸 mesh 更降频",
    )
    parser.add_argument(
        "--full-res-gui", action="store_true",
        help="全分辨率 1080p 显示(更清晰但更卡)",
    )
    args = parser.parse_args()
    if args.full_res_gui:
        args.proc_max_w = max(args.proc_max_w, PROC_MAX_W_FULL)
    elif args.fast:
        args.proc_max_w = min(args.proc_max_w, 480)
    elif not args.no_face_track and args.proc_max_w > PROC_MAX_W:
        args.proc_max_w = PROC_MAX_W
    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")
    dist_min, dist_max = args.dist_min, args.dist_max

    install_handlers()
    motion = None
    face_track = None

    if not args.no_gui and not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    zed = open_zed_camera(
        use_hd1080=args.hd1080, dist_min=dist_min, dist_max=dist_max,
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
    print(Fore.YELLOW + "按 ESC 退出, 'f' 切换全屏, Ctrl+C 强制退出。")
    need_ros = (not args.preview and not args.no_actions) or not args.no_face_track
    if need_ros:
        import rospy
        from gesture_motion import GestureMotionController
        from face_tracker import IntegratedFaceTracker
        from ros_control import FsmStateMonitor

        if not rospy.core.is_initialized():
            rospy.init_node("zed_gesture_recognition", anonymous=True)

    if args.preview:
        print(
            Fore.YELLOW
            + "预览模式: 仅识别与日志，不发送机器人指令 (--preview)",
        )
    elif args.no_actions:
        print(Fore.YELLOW + "动作执行已禁用 (--no-actions)")

    if need_ros and not args.preview and not args.no_actions:
        motion = GestureMotionController(
            dry_run=False,
            no_actions=False,
            no_coquette=args.no_coquette,
            no_joy=args.no_joy,
            no_fsm=args.no_fsm,
        )
        motion.wait_fsm(should_stop=is_requested)
        if is_requested():
            return

    if need_ros and not args.no_face_track:
        fsm = None
        if motion is not None and motion.fsm is not None:
            fsm = motion.fsm
        elif not args.no_fsm:
            fsm = FsmStateMonitor()
        face_dry = args.preview or args.no_actions
        mesh_iv = 5 if args.fast else 3
        roi_iv = 18 if args.fast else 12
        face_track = IntegratedFaceTracker(
            fsm,
            dry_run=face_dry,
            mesh_interval=mesh_iv,
            roi_interval=roi_iv,
        )
        face_track.start()

    use_lite_display = (
        LITE_DISPLAY
        and not args.full_res_gui
        and not args.no_gui
    )
    face_hint = "关" if args.no_face_track else "常开(共用ZED)"
    action_desc = (
        "关"
        if args.no_actions
        else "1撒娇扭腰 2抬手 3挥双手 4踢球 5→跟手"
    )
    mode_desc = "预览" if args.preview or args.no_actions else "执行"
    disp_hint = "640p轻量" if use_lite_display else "1080p全分辨率"
    print(
        Fore.GREEN
        + f"显示: {disp_hint} | 脸部跟踪: {face_hint} | 手势动作[{mode_desc}]: 0急停/按住"
        f"{args.zero_exit_sec:.0f}s退出 "
        + f"{action_desc}; 1~4稳{args.gesture_hold_sec:.0f}s 5稳{GESTURE_FOLLOW_HOLD_SEC:.0f}s→跟手",
    )

    last_logged_gesture = -1
    last_logged_confirmed = -1
    last_log_t = 0.0
    zero_handler = GestureZeroHandler(exit_hold_sec=args.zero_exit_sec)
    action_hold = GestureActionHold(hold_sec=args.gesture_hold_sec)
    follow_hold = GestureActionHold(
        hold_sec=GESTURE_FOLLOW_HOLD_SEC,
        allowed_gestures=frozenset({GESTURE_FOLLOW}),
    )
    handoff_requested = False
    frame_idx = 0
    had_hand_prev = False
    draw_landmarks = DRAW_LANDMARKS and not args.no_gui

    try:
        while not is_requested():
            confirmed = -1
            follow_confirmed = -1
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                if is_requested():
                    break
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)

            frame = image.get_data()
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            img_w = image.get_width()
            img_h = image.get_height()

            need_depth = had_hand_prev or (frame_idx % DEPTH_EVERY_N == 0)
            if need_depth:
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

            proc_w, proc_h = compute_proc_size(img_w, img_h, proc_max_w)
            if (proc_w, proc_h) != (img_w, img_h):
                proc_bgr = cv2.resize(
                    frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA,
                )
            else:
                proc_bgr = frame
            display_bgr = proc_bgr if use_lite_display else frame
            rgb_mp = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            if not rgb_mp.flags["C_CONTIGUOUS"]:
                rgb_mp = np.ascontiguousarray(rgb_mp)
            rgb_mp.flags.writeable = False

            results = hands.process(rgb_mp)

            gesture = -1
            raw_gesture = -1
            distance = 0.0
            direction = "无手"
            in_range = True
            palm_center_px = None
            palm_pos = None

            had_hand_prev = bool(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
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

                    if need_depth:
                        palm_pos, palm_center_px = calculate_palm_position(
                            hand_lm, img_w, img_h, point_cloud,
                        )
                    else:
                        palm_pos, palm_center_px = None, None

                    if palm_pos is not None:
                        distance = palm_pos[2]
                        in_range = is_distance_in_range(
                            distance, dist_min, dist_max,
                        )
                        if palm_center_px is not None:
                            dot_col = (0, 255, 0) if in_range else (0, 165, 255)
                            if use_lite_display:
                                sx = proc_w / max(img_w, 1)
                                sy = proc_h / max(img_h, 1)
                                dot_px = (
                                    int(palm_center_px[0] * sx),
                                    int(palm_center_px[1] * sy),
                                )
                            else:
                                dot_px = palm_center_px
                            cv2.circle(
                                display_bgr, dot_px, 8, dot_col, -1,
                            )
                        draw_overlay_log(
                            display_bgr, f"X: {palm_pos[0]:+.2f}m",
                            (10, 30), (255, 0, 0),
                        )
                        draw_overlay_log(
                            display_bgr, f"Y: {palm_pos[1]:+.2f}m",
                            (10, 60), (0, 255, 0),
                        )
                        z_col = (0, 0, 255) if in_range else (0, 165, 255)
                        draw_overlay_log(
                            display_bgr, f"Z: {palm_pos[2]:+.2f}m",
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

            has_hand = bool(results.multi_hand_landmarks)

            face_paused_for_follow = (
                gesture == GESTURE_FOLLOW
                or follow_hold.pending_gesture == GESTURE_FOLLOW
            )
            run_face = (
                face_track is not None
                and face_track.is_active
                and not face_paused_for_follow
                and (frame_idx % FACE_EVERY_N == 0)
            )
            if run_face:
                face_track.process_shared_rgb(
                    rgb_mp, display_bgr, proc_w, proc_h,
                )

            zero_estop, zero_exit, zero_hold = zero_handler.update(
                gesture, has_hand=has_hand, in_range=in_range,
            )
            if zero_estop:
                if motion is not None:
                    motion.on_zero_estop(hold_sec=zero_hold)
                elif zero_hold < 0.15:
                    log_gesture_zero_estop(dry_run=True)
            if zero_exit:
                log_gesture_zero_exit(zero_hold)
                break

            disp_h = display_bgr.shape[0]
            draw_overlay_log(
                display_bgr,
                f"Range: {dist_min:.1f}~{dist_max:.1f}m",
                (10, disp_h - 20), (200, 200, 200), font_scale=0.5, thickness=1,
            )

            if zero_estop:
                remain = zero_handler.hold_remaining(zero_hold)
                draw_overlay_log(
                    display_bgr,
                    f"G0 {GESTURE_ZERO_LABEL} exit {remain:.1f}s",
                    (10, 130), (0, 0, 255), font_scale=0.9, thickness=2,
                )
            elif face_paused_for_follow:
                draw_overlay_log(
                    display_bgr,
                    "FACE off (G5跟手)",
                    (10, 130),
                    (0, 165, 255),
                    font_scale=0.75,
                    thickness=2,
                )
            elif face_track is not None and face_track.is_active:
                ov = face_track.overlay
                if ov is not None and ov.has_face:
                    yaw_d, pitch_d = face_track.get_neck_target_deg()
                    face_txt = f"FACE yaw{yaw_d:+.0f} pitch{pitch_d:+.0f}"
                    face_col = (0, 255, 128)
                else:
                    face_txt = "FACE search..."
                    face_col = (0, 200, 255)
                draw_overlay_log(
                    display_bgr, face_txt, (10, 130), face_col,
                    font_scale=0.75, thickness=2,
                )
                if run_face and ov is not None and ov.has_face:
                    face_track.draw_overlay(display_bgr)
            elif gesture >= 0:
                col = GESTURE_COLORS_BGR[min(gesture, 5)]
                gtext = f"Gesture: {gesture}"
                if raw_gesture >= 0 and raw_gesture != gesture:
                    gtext += f" (raw {raw_gesture})"
                draw_overlay_log(
                    display_bgr, gtext, (10, 130),
                    col, font_scale=1.0, thickness=3,
                )
                dcol = (0, 255, 0) if direction == "静止" else (0, 255, 255)
                draw_overlay_log(
                    display_bgr, f"Dir: {direction}", (10, 170), dcol,
                )
            elif results.multi_hand_landmarks and not in_range:
                draw_overlay_log(
                    display_bgr, direction, (10, 130), (0, 165, 255),
                    font_scale=0.8, thickness=2,
                )
            else:
                draw_overlay_log(
                    display_bgr, "No hand", (10, 130), (128, 128, 128),
                )

            if gesture == GESTURE_STOP:
                action_hold.reset()
                follow_hold.reset()
                if motion is not None:
                    motion.clear_pending_fire()
                confirmed = -1
                follow_confirmed = -1
            else:
                confirmed = action_hold.update(
                    gesture, has_hand=has_hand, in_range=in_range,
                )
                follow_confirmed = follow_hold.update(
                    gesture, has_hand=has_hand, in_range=in_range,
                )

            if confirmed < 0 and motion is not None:
                motion.clear_pending_fire()

            if follow_confirmed == GESTURE_FOLLOW:
                log_follow_handoff(preview=args.preview or args.no_actions)
                if not args.preview and not args.no_actions:
                    handoff_requested = True
                break

            if confirmed >= 0:
                if motion is not None:
                    motion.on_confirmed(
                        confirmed,
                        has_hand=has_hand,
                        in_range=in_range,
                    )
                elif confirmed != last_logged_confirmed:
                    log_gesture_action_edge(
                        confirmed,
                        last_logged_confirmed,
                        in_range=in_range,
                        has_hand=has_hand,
                        preview_only=True,
                    )

            last_logged_gesture = gesture
            last_logged_confirmed = confirmed if confirmed >= 0 else -1

            face_track_on = (
                face_track is not None and face_track.is_active
            )
            now_log = time.time()
            if now_log - last_log_t >= LOG_INTERVAL_SEC:
                last_log_t = now_log
                hold_pct = ""
                if action_hold.pending_gesture >= 0:
                    hold_pct = f" 稳{action_hold.progress * 100:.0f}%"
                print_terminal_log(
                    gesture, distance, direction,
                    in_range=in_range, has_hand=has_hand,
                    face_track_on=face_track_on,
                )
                if hold_pct and gesture in (1, 2, 3, 4):
                    emit_status_line(
                        f"{Fore.CYAN}[{time.strftime('%H:%M:%S')}] "
                        f"{Fore.WHITE}手势{gesture}确认中{hold_pct}",
                    )
                elif (
                    follow_hold.pending_gesture == GESTURE_FOLLOW
                    and follow_hold.progress > 0
                ):
                    fp = follow_hold.progress * 100
                    emit_status_line(
                        f"{Fore.CYAN}[{time.strftime('%H:%M:%S')}] "
                        f"{Fore.WHITE}手势5→跟手 稳{fp:.0f}%",
                    )

            frame_idx += 1

            if not args.no_gui:
                cv2.imshow(WINDOW_NAME, display_bgr)
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
        request("用户中断")

    finally:
        fast_exit = is_requested()
        if handoff_requested and not args.preview and not args.no_actions:
            release_before_handoff(
                face_track=face_track,
                motion=motion,
                zed=zed,
                hands=hands,
                no_gui=args.no_gui,
            )
            exec_hand_tracking()
        if face_track is not None:
            try:
                face_track.shutdown()
            except Exception:
                pass
        if motion is not None:
            motion.clear_pending_fire()
            motion.shutdown(fast=fast_exit)
        rospy_shutdown_if_init()
        try:
            zed.close()
        except Exception:
            pass
        try:
            hands.close()
        except Exception:
            pass
        if not args.no_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(Fore.GREEN + "\n程序已退出")


if __name__ == "__main__":
    main()
