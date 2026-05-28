#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hand_follow_robot.py — 脸跟踪常开 + 手势 1~4 动作
#
# 手势 0: 急停+中止动作；按住 5s 退出程序
# 手势 1: 腰部撒娇（wing_wist，±45° 来回 2 次 + 挥双手）
# 手势 2~4: 抬手 / 挥动双手 / 踢球（/joy_msg 动作库）
# 手柄 /joy 有输入时：停手势动作库，脸跟踪保持
# 手势 5 五指跟手: 已禁用，备份见 5_finger_locomotion.py

import argparse
import os
import sys

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ros_setup import require_sim2real_msg

require_sim2real_msg()

import cv2
import rospy

from hand_perception import (
    DIST_MAX_M,
    DIST_MIN_M,
    GESTURE_COLORS_BGR,
    ZedHandTracker,
)
from gesture_actions import (
    GESTURE_ACTION_HOLD_SEC,
    GESTURE_ACTION_SPECS,
    GESTURE_FACE_TRACK_LABEL,
    GESTURE_HEAD_NOD,
    GESTURE_HOLD_GESTURES,
    GESTURE_STOP,
    GESTURE_ZERO_EXIT_SEC,
    GestureActionHold,
    GestureZeroHandler,
    log_gesture_zero_estop,
    log_gesture_zero_exit,
)
from hand_action_library import GestureActionPlayer
from face_tracker import IntegratedFaceTracker
from hand_follow_control import FsmStateMonitor, JoyMonitor, JOY_IDLE_SEC
from waist_coquette_player import WaistCoquettePlayer

FULLSCREEN = True
WINDOW_NAME = "Hand Follow"

# OpenCV 默认字体无法显示中文，GUI 仅用 ASCII
GESTURE_UI_EN = {1: "coquette", 2: "hello", 3: "cheer", 4: "kick"}


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.6, thick=2):
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="手势识别 + 脸跟踪(常开) + 手势1撒娇/2~4动作库",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="预览模式：不发 /joy_msg，不启脸跟踪控头",
    )
    parser.add_argument(
        "--enable-motion", action="store_true",
        help="(已默认开启) 与不加 --preview 相同，保留兼容",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--no-fsm", action="store_true")
    parser.add_argument("--no-joy", action="store_true")
    parser.add_argument(
        "--no-actions", action="store_true",
        help="禁用手势 1 撒娇与 2~4 动作库",
    )
    parser.add_argument(
        "--no-coquette", action="store_true",
        help="禁用手势 1 腰部撒娇",
    )
    parser.add_argument(
        "--no-face-track", action="store_true",
        help="禁用内嵌脸部跟踪 (默认开启，共用 ZED RGB)",
    )
    parser.add_argument(
        "--hd1080", action="store_true",
        help="使用 HD1080 (默认 HD720，更流畅)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="性能模式: 640px + 脸检测降频",
    )
    parser.add_argument("--dist-min", type=float, default=DIST_MIN_M)
    parser.add_argument("--dist-max", type=float, default=DIST_MAX_M)
    parser.add_argument("--proc-max-w", type=int, default=640)
    parser.add_argument(
        "--zero-exit-sec", type=float, default=GESTURE_ZERO_EXIT_SEC,
        help="手势0持续按住多少秒后退出 (默认 5)",
    )
    parser.add_argument(
        "--gesture-hold-sec", type=float, default=GESTURE_ACTION_HOLD_SEC,
        help="手势1~4连续稳定多少秒后触发动作 (默认 2)",
    )
    args = parser.parse_args()
    if args.fast:
        args.proc_max_w = min(args.proc_max_w, 640)

    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")

    rospy.init_node("hand_follow_robot", anonymous=False)
    dry_run = args.preview and not args.enable_motion

    tracker = ZedHandTracker(
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        use_hd1080=args.hd1080,
        proc_max_w=args.proc_max_w,
    )
    fsm = None if args.no_fsm else FsmStateMonitor()
    joy = None if args.no_joy else JoyMonitor()

    action_player = None
    coquette_player = None
    if not args.no_actions:
        action_player = GestureActionPlayer(dry_run=dry_run)
    if not args.no_actions and not args.no_coquette:
        coquette_player = WaistCoquettePlayer(dry_run=dry_run)

    face_track = None
    if not args.no_face_track:
        mesh_iv = 1
        roi_iv = 10 if args.fast else 8
        face_track = IntegratedFaceTracker(
            fsm,
            dry_run=dry_run,
            mesh_interval=mesh_iv,
            roi_interval=roi_iv,
        )

    joy_hint = (
        "无手柄仲裁" if joy is None
        else f"手柄优先, 空闲 {JOY_IDLE_SEC:.0f}s"
    )
    face_hint = "关" if args.no_face_track else "常开"
    action_hint = (
        "关" if args.no_actions
        else (
            "2抬手 3挥双手 4踢球"
            if args.no_coquette
            else "1撒娇 2抬手 3挥双手 4踢球"
        )
    )
    zero_handler = GestureZeroHandler(exit_hold_sec=args.zero_exit_sec)
    action_hold = GestureActionHold(hold_sec=args.gesture_hold_sec)
    if dry_run:
        rospy.logwarn("[hand_follow] 预览模式 (--preview)")
    rospy.loginfo(
        "[hand_follow] 脸跟踪=%s | 动作=%s(稳定%.1fs) | "
        "手势0按住%.0fs退出 | %s | %s",
        face_hint,
        action_hint,
        args.gesture_hold_sec,
        args.zero_exit_sec,
        "DRY-RUN" if dry_run else "MOTION",
        joy_hint,
    )

    if fsm is not None:
        rospy.loginfo("[FSM] 等待 EXEC_DEFAULT(5)...")
        if fsm.wait_for_exec_default(timeout=30.0):
            rospy.loginfo("[FSM] OK")
        else:
            rospy.logwarn("[FSM] 超时")

    if face_track is not None:
        face_track.start()
        rospy.loginfo("[hand_follow] %s 已启动", GESTURE_FACE_TRACK_LABEL)

    is_fullscreen = FULLSCREEN
    if not args.no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if FULLSCREEN:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
            )

    def _release_gesture_control_to_joy():
        if action_player is not None and action_player.is_busy:
            action_player.abort()
        if coquette_player is not None and coquette_player.is_busy:
            coquette_player.abort()

    try:
        while not rospy.is_shutdown():
            frame, obs = tracker.process_frame(
                draw_landmarks=True,
                face_tracker=face_track,
            )
            h = frame.shape[0]

            palm_pos = obs.palm_pos
            palm_x, palm_y = 0.0, 0.0
            if palm_pos is not None:
                palm_x, palm_y = palm_pos[0], palm_pos[1]

            zero_estop, zero_exit, zero_hold = zero_handler.update(
                obs.gesture,
                has_hand=obs.has_hand,
                in_range=obs.in_range,
            )
            if zero_estop:
                if zero_hold < 0.15:
                    log_gesture_zero_estop(dry_run=dry_run)
                    rospy.logwarn("[hand_follow] 手势0急停")
                if action_player is not None:
                    action_player.abort()
                if coquette_player is not None:
                    coquette_player.abort()
                action_hold.reset()
            if zero_exit:
                log_gesture_zero_exit(zero_hold)
                rospy.loginfo(
                    "[hand_follow] 手势0保持 %.1fs，退出视觉识别",
                    zero_hold,
                )
                break

            fsm_ok = fsm is None or fsm.is_exec_default()
            joy_blocking = (
                joy is not None and joy.blocks_gesture_control()
            )
            if joy is not None:
                need_release = (
                    (action_player is not None and action_player.is_busy)
                    or (coquette_player is not None and coquette_player.is_busy)
                )
                if joy.poll_takeover_edge():
                    if need_release:
                        _release_gesture_control_to_joy()
                    rospy.loginfo(
                        "[hand_follow] 手柄接管: 已停手势动作, 脸跟踪保持",
                    )
                elif joy_blocking and need_release:
                    _release_gesture_control_to_joy()

            face_track_on = face_track is not None and face_track.is_active

            confirmed_action_g = -1
            action_fired = False
            coquette_fired = False
            if obs.gesture != GESTURE_STOP:
                confirmed_action_g = action_hold.update(
                    obs.gesture,
                    has_hand=obs.has_hand,
                    in_range=obs.in_range,
                )
            if confirmed_action_g == GESTURE_HEAD_NOD and coquette_player is not None:
                coquette_fired = coquette_player.update(
                    confirmed_action_g,
                    has_hand=obs.has_hand,
                    in_range=obs.in_range,
                    joy_blocking=joy_blocking,
                    fsm_ok=fsm_ok,
                    other_busy=action_player is not None and action_player.is_busy,
                )
            elif (
                action_player is not None
                and confirmed_action_g in GESTURE_ACTION_SPECS
                and not (coquette_player is not None and coquette_player.is_busy)
            ):
                action_fired = action_player.update(
                    confirmed_action_g,
                    has_hand=obs.has_hand,
                    in_range=obs.in_range,
                    joy_blocking=joy_blocking,
                    fsm_ok=fsm_ok,
                )

            if obs.palm_px is not None and not tracker.lite_display:
                col = (0, 255, 0) if obs.in_range else (0, 165, 255)
                cv2.circle(frame, obs.palm_px, 8, col, -1)
            cx = frame.shape[1] // 2
            cv2.drawMarker(
                frame, (cx, h // 2), (255, 255, 255),
                cv2.MARKER_CROSS, 24, 2,
            )

            if obs.gesture >= 0:
                gcol = GESTURE_COLORS_BGR[min(obs.gesture, 5)]
                draw_text(frame, f"Gesture: {obs.gesture}", (10, 40), gcol, 1.0, 3)
            elif obs.has_hand and not obs.in_range:
                draw_text(frame, "OUT OF RANGE", (10, 40), (0, 165, 255), 0.8, 2)

            neck_tag = "FACE_TRACK" if face_track_on else ""
            draw_text(
                frame,
                f"palm X:{palm_x:+.2f} Y:{palm_y:+.2f} Z:{obs.distance_m:.2f}m {neck_tag}",
                (10, 80), (255, 255, 255), 0.55, 1,
            )

            if zero_estop:
                remain = zero_handler.hold_remaining(zero_hold)
                draw_text(
                    frame,
                    f"G0 E-STOP exit {remain:.1f}s",
                    (10, 120), (0, 0, 255), 0.75, 2,
                )
            elif face_track_on:
                ov = face_track.overlay if face_track else None
                if ov is not None and ov.has_face:
                    yaw_d, pitch_d = face_track.get_neck_target_deg()
                    face_txt = f"FACE yaw{yaw_d:+.0f} pitch{pitch_d:+.0f}"
                else:
                    face_txt = "FACE search..."
                draw_text(frame, face_txt, (10, 120), (0, 255, 128), 0.75, 2)
            elif coquette_player is not None and (
                coquette_fired or coquette_player.is_busy
            ):
                draw_text(
                    frame, "ACT: coquette",
                    (10, 120), (255, 128, 200), 0.75, 2,
                )
            elif action_player is not None and (
                action_fired or action_player.is_busy
            ):
                draw_text(
                    frame, "ACT: running",
                    (10, 120), (255, 128, 0), 0.75, 2,
                )
            elif joy_blocking:
                draw_text(
                    frame, f"JOY ({joy.idle_remaining():.1f}s)",
                    (10, 120), (0, 128, 255), 0.75, 2,
                )
            elif (
                obs.has_hand
                and obs.in_range
                and obs.gesture in GESTURE_HOLD_GESTURES
                and action_hold.progress < 1.0
            ):
                remain = action_hold.hold_remaining
                pct = action_hold.progress * 100.0
                draw_text(
                    frame,
                    f"G{obs.gesture} hold {pct:.0f}% ({remain:.1f}s)",
                    (10, 120), (0, 200, 255), 0.7, 2,
                )
            else:
                if obs.has_hand and obs.in_range and obs.gesture in GESTURE_UI_EN:
                    hint = f"G{obs.gesture} {GESTURE_UI_EN[obs.gesture]}"
                elif obs.has_hand:
                    hint = "G0 estop G1-4 act | face on"
                else:
                    hint = "no hand"
                draw_text(frame, hint, (10, 120), (128, 128, 128), 0.7, 2)

            tag = "MOTION" if not dry_run else "PREVIEW"
            draw_text(
                frame, tag, (10, 160),
                (0, 0, 255) if not dry_run else (0, 128, 255), 0.8, 2,
            )
            fsm_txt = (
                "FSM n/a" if fsm is None
                else f"FSM {fsm.state}({fsm.state_name(fsm.state)})"
            )
            block_parts = []
            if dry_run:
                block_parts.append("preview")
            if not fsm_ok:
                block_parts.append("FSM!=5")
            if joy_blocking:
                block_parts.append("JOY")
            block_txt = " ".join(block_parts) if block_parts else "OK"
            draw_text(
                frame, f"{fsm_txt} | {block_txt}",
                (10, h - 28), (0, 200, 255) if block_parts else (0, 220, 0),
                0.45, 1,
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

    except KeyboardInterrupt:
        rospy.logwarn("[hand_follow] 用户中断")

    finally:
        if face_track is not None:
            face_track.shutdown()
        tracker.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        rospy.loginfo("[hand_follow] 已退出")


if __name__ == "__main__":
    main()
