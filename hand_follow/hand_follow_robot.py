#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hand_follow_robot.py — 脸跟踪常开 + 手势 1 抬头20° + 手势 2~4 动作库 + 手势 5 跟手
#
# 手势 0: 急停+中止动作；按住 5s 退出程序
# 手势 1: 抬头 20° 后复原（期间暂停脸跟踪脖子输出）
# 手势 2~4: 抬手 / 挥动双手 / 踢球（/joy_msg 动作库）
# 底盘: 误差 > 0.3m → 指令 ±0.3; 手静止不发指令
# 脸部跟踪: 常开，与手势识别共用 ZED RGB

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
    GESTURE_ACTION_LABELS,
    GESTURE_FACE_TRACK_LABEL,
    GESTURE_HEAD_NOD,
    GESTURE_HEAD_NOD_LABEL,
    GESTURE_STOP,
    GESTURE_ZERO_EXIT_SEC,
    GESTURE_ZERO_LABEL,
    GestureZeroHandler,
    log_gesture_head_nod,
    log_gesture_zero_estop,
    log_gesture_zero_exit,
)
from hand_action_library import GestureActionPlayer
from face_tracker import IntegratedFaceTracker
from hand_follow_control import (
    FsmStateMonitor,
    CmdVelPublisher,
    HandControlInput,
    HandFollowController,
    HandMotionGate,
    JoyMonitor,
    NeckController,
    VelCommand,
    GESTURE_FOLLOW,
    JOY_IDLE_SEC,
    POS_THRESH_M,
    CMD_MAG,
    HAND_MOVE_THRESH_M,
)

FULLSCREEN = True
WINDOW_NAME = "Hand Follow (Gesture 5)"


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.6, thick=2):
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA,
    )


def _fmt_cmd(v):
    if abs(v) < 1e-6:
        return "0"
    return f"{v:+.1f}"


def main():
    parser = argparse.ArgumentParser(
        description="手势跟手 + 脸跟踪(默认开) + 动作库",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="预览模式：不发 /cmd_vel、/joy_msg，不启脸跟踪控头",
    )
    parser.add_argument(
        "--enable-motion", action="store_true",
        help="(已默认开启) 与不加 --preview 相同，保留兼容",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--no-fsm", action="store_true")
    parser.add_argument("--no-joy", action="store_true")
    parser.add_argument("--no-neck", action="store_true", help="不控制脖子抬头")
    parser.add_argument(
        "--no-actions", action="store_true",
        help="禁用手势 2~4 动作库触发",
    )
    parser.add_argument(
        "--no-face-track", action="store_true",
        help="禁用内嵌脸部跟踪 (默认开启，共用 ZED RGB)",
    )
    parser.add_argument(
        "--no-face-track-auto", action="store_true",
        help="(已废弃) 脸跟踪现为常开，此参数无效果",
    )
    parser.add_argument(
        "--hd1080", action="store_true",
        help="使用 HD1080 (默认 HD720，更流畅)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="性能模式: 640px + 脸检测每2帧 (等同 --proc-max-w 640)",
    )
    parser.add_argument("--target-dist", type=float, default=1.0)
    parser.add_argument("--dist-min", type=float, default=DIST_MIN_M)
    parser.add_argument("--dist-max", type=float, default=DIST_MAX_M)
    parser.add_argument("--pos-thresh", type=float, default=POS_THRESH_M)
    parser.add_argument("--cmd-mag", type=float, default=CMD_MAG,
                        help="跟手指令幅度 (默认 0.3)")
    parser.add_argument(
        "--move-thresh", type=float, default=HAND_MOVE_THRESH_M,
        help="判定手在动的位移阈值/米",
    )
    parser.add_argument("--proc-max-w", type=int, default=640)
    parser.add_argument(
        "--zero-exit-sec", type=float, default=GESTURE_ZERO_EXIT_SEC,
        help="手势0持续按住多少秒后退出 (默认 5)",
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
    follow_ctrl = HandFollowController(
        target_distance_m=args.target_dist,
        pos_thresh_m=args.pos_thresh,
        cmd_mag=args.cmd_mag,
    )
    motion_gate = HandMotionGate(move_thresh_m=args.move_thresh)
    vel_cmd = VelCommand()
    fsm = None if args.no_fsm else FsmStateMonitor()
    joy = None if args.no_joy else JoyMonitor()
    pub_thread = CmdVelPublisher(vel_cmd, fsm, dry_run=dry_run)
    pub_thread.start()

    neck_thread = None
    if not args.no_neck:
        neck_thread = NeckController(fsm, dry_run=dry_run)
        neck_thread.start()

    action_player = None
    if not args.no_actions:
        action_player = GestureActionPlayer(dry_run=dry_run)

    face_track = None
    if not args.no_face_track:
        mesh_iv = 3 if args.fast else 2
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
        else "2抬手 3挥双手 4踢球"
    )
    zero_handler = GestureZeroHandler(exit_hold_sec=args.zero_exit_sec)
    if dry_run:
        rospy.logwarn(
            "[hand_follow] 预览模式 (--preview)：不发运动指令、不启脸跟踪子进程",
        )
    rospy.loginfo(
        "[hand_follow] 指令 ±%.2f 阈值 %.2fm | 手移动阈值 %.2fm | "
        "手势%d跟手 | 脸跟踪=%s | 动作库=%s | 手势0按住%.0fs退出 | 脖子=%s | %s | %s",
        args.cmd_mag, args.pos_thresh, args.move_thresh,
        GESTURE_FOLLOW,
        face_hint,
        action_hint,
        args.zero_exit_sec,
        "手势1抬头20°" if not args.no_neck else "关",
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

    prev_gesture = -1
    neck_nod_was_busy = False
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

            hand_seen = obs.has_hand and palm_pos is not None

            hand_moving = motion_gate.update(palm_pos)

            zero_estop, zero_exit, zero_hold = zero_handler.update(
                obs.gesture,
                has_hand=obs.has_hand,
                in_range=obs.in_range,
            )
            if zero_estop:
                if zero_hold < 0.15:
                    log_gesture_zero_estop(dry_run=dry_run)
                    rospy.logwarn("[hand_follow] 手势0急停: 停止动作与 cmd_vel")
                if action_player is not None:
                    action_player.abort()
                follow_ctrl.reset()
                motion_gate.reset()
                vel_cmd.stop()
            if zero_exit:
                log_gesture_zero_exit(zero_hold)
                rospy.loginfo(
                    "[hand_follow] 手势0保持 %.1fs，退出视觉识别",
                    zero_hold,
                )
                break

            fsm_ok = fsm is None or fsm.is_exec_default()
            joy_blocking = joy is not None and not joy.allow_program_cmd()
            face_track_on = face_track is not None and face_track.is_active

            if (
                neck_thread is not None
                and obs.gesture == GESTURE_HEAD_NOD
                and prev_gesture != GESTURE_HEAD_NOD
                and obs.has_hand
                and obs.in_range
                and fsm_ok
                and not joy_blocking
                and not zero_estop
            ):
                hold_yaw = 0.0
                start_pitch = 0.0
                if face_track is not None:
                    hold_yaw, start_pitch = face_track.get_neck_target()
                if neck_thread.trigger_gesture_nod(
                    hold_yaw_rad=hold_yaw,
                    start_pitch_rad=start_pitch,
                ):
                    if face_track is not None:
                        face_track.set_neck_output_enabled(False)
                    log_gesture_head_nod(dry_run=dry_run)

            neck_nod_busy = (
                neck_thread is not None and neck_thread.is_busy
            )
            if neck_nod_was_busy and not neck_nod_busy and face_track is not None:
                face_track.set_neck_output_enabled(True)
            neck_nod_was_busy = neck_nod_busy
            prev_gesture = obs.gesture

            action_fired = False
            if action_player is not None and obs.gesture != GESTURE_STOP:
                action_fired = action_player.update(
                    obs.gesture,
                    has_hand=obs.has_hand,
                    in_range=obs.in_range,
                    joy_blocking=joy is not None and not joy.allow_program_cmd(),
                    fsm_ok=fsm_ok,
                )

            ctrl_inp = HandControlInput(
                gesture=obs.gesture,
                distance_m=obs.distance_m,
                palm_x_m=palm_x,
                palm_y_m=palm_y,
                active=obs.valid_for_control,
            )

            action_busy = (
                action_player is not None and action_player.is_busy
            )
            if zero_estop or joy_blocking or action_fired or action_busy:
                follow_ctrl.reset()
                out_cmd_x, out_cmd_y, out_cmd_z = 0.0, 0.0, 0.0
                if zero_estop:
                    mode = "estop"
                elif joy_blocking:
                    mode = "joy"
                else:
                    mode = "action"
            elif not ctrl_inp.active or not hand_moving:
                follow_ctrl.reset()
                out_cmd_x, out_cmd_y, out_cmd_z = 0.0, 0.0, 0.0
                if not ctrl_inp.active:
                    mode = "idle"
                else:
                    mode = "hold"
            else:
                out = follow_ctrl.compute(ctrl_inp)
                out_cmd_x, out_cmd_y, out_cmd_z = out.cmd_x, out.cmd_y, out.cmd_z
                mode = out.mode

            vel_cmd.set(out_cmd_x, out_cmd_y, out_cmd_z)

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
                draw_text(frame, obs.direction, (10, 40), (0, 165, 255), 0.8, 2)

            if neck_nod_busy:
                phase = neck_thread.phase if neck_thread else "idle"
                if phase == "feedback":
                    neck_tag = "GESTURE1 抬头20°"
                elif phase == "return":
                    neck_tag = "GESTURE1 回中"
                else:
                    neck_tag = "GESTURE1"
            elif face_track_on:
                neck_tag = "FACE_TRACK"
            else:
                neck_tag = ""
            draw_text(
                frame,
                f"palm X:{palm_x:+.2f} Y:{palm_y:+.2f} Z:{obs.distance_m:.2f}m  "
                f"{'MOVE' if hand_moving else 'HOLD'} {neck_tag}",
                (10, 80), (255, 255, 255), 0.55, 1,
            )
            draw_text(
                frame,
                f"cmd ±{args.cmd_mag:.1f} thresh {args.pos_thresh:.1f}m",
                (10, h - 50), (180, 180, 180), 0.5, 1,
            )

            if zero_estop:
                remain = zero_handler.hold_remaining(zero_hold)
                draw_text(
                    frame,
                    f"手势0 {GESTURE_ZERO_LABEL} 退出倒计时 {remain:.1f}s",
                    (10, 120), (0, 0, 255), 0.75, 2,
                )
            elif neck_nod_busy:
                draw_text(
                    frame,
                    f"手势1 {GESTURE_HEAD_NOD_LABEL}",
                    (10, 120), (0, 255, 200), 0.75, 2,
                )
            elif face_track_on:
                draw_text(
                    frame,
                    f"{GESTURE_FACE_TRACK_LABEL}: 运行中",
                    (10, 120), (0, 255, 128), 0.75, 2,
                )
            elif action_player is not None and (
                action_fired or action_player.is_busy
            ):
                act_txt = action_player.last_label or "动作"
                draw_text(
                    frame, f"动作: {act_txt}",
                    (10, 120), (255, 128, 0), 0.75, 2,
                )
            elif joy_blocking:
                draw_text(
                    frame, f"JOY ({joy.idle_remaining():.1f}s)",
                    (10, 120), (0, 128, 255), 0.75, 2,
                )
            elif ctrl_inp.active and hand_moving:
                draw_text(
                    frame,
                    f"CMD X:{_fmt_cmd(out_cmd_x)} "
                    f"Y:{_fmt_cmd(out_cmd_y)} "
                    f"Z:{_fmt_cmd(out_cmd_z)} [{mode}]",
                    (10, 120), (0, 255, 255), 0.75, 2,
                )
            elif ctrl_inp.active and not hand_moving:
                draw_text(frame, "手势5 手静止-无指令", (10, 120), (128, 200, 255), 0.7, 2)
            else:
                if obs.has_hand and obs.in_range and obs.gesture == GESTURE_HEAD_NOD:
                    hint = f"手势1→{GESTURE_HEAD_NOD_LABEL}"
                elif obs.has_hand and obs.in_range and obs.gesture in GESTURE_ACTION_LABELS:
                    hint = f"手势{obs.gesture}→{GESTURE_ACTION_LABELS[obs.gesture]}"
                elif obs.has_hand:
                    hint = "0急停 1抬头20° 2~4动作 5跟手"
                else:
                    hint = "无手"
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
                block_parts.append("预览--preview")
            if not fsm_ok:
                block_parts.append("FSM!=5")
            if joy_blocking:
                block_parts.append("JOY占用")
            block_txt = " ".join(block_parts) if block_parts else "可控制"
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
        vel_cmd.stop()
        pub_thread.publish_stop_blocking(0.5)
        pub_thread.stop()
        if face_track is not None:
            face_track.shutdown()
        if neck_thread is not None:
            neck_thread.publish_center_blocking(0.5)
            neck_thread.stop()
        tracker.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        rospy.loginfo("[hand_follow] 已退出")


if __name__ == "__main__":
    main()
