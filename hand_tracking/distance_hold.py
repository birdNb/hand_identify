#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手部跟踪：左右居中（angular.z）+ 可选手势5前后距离（linear.x）。

特性：
- 掌心在画面左右偏移 → 底盘左右转，使手保持在正前方
- 横向死区：相对画面中心偏移 |dx_norm| ≤ 20% 不发转指令
- 左右转：强驱动 ±ANGULAR_Z_MAG（angular.z，大于前后速度）
- 手势5 + 进入跟随后：前后距离保持（linear.x，Z 深度，独立死区）
"""

import argparse
import os
import sys
import time

import rospy
from geometry_msgs.msg import Twist

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
from paths import setup_paths  # noqa: E402

setup_paths(tracking=True)

from gesture_actions import GESTURE_FOLLOW, GESTURE_FOLLOW_LOST_SEC, GestureFiveLostWatch
from handoff import (
    exec_gesture_recognition,
    log_return_to_gesture,
    release_before_gesture_return,
)
from ros_setup import require_sim2real_msg
from hand_perception import DIST_MAX_M, DIST_MIN_M, ZedHandTracker
from ros_control import (
    FsmStateMonitor,
    HAND_TRACKING_JOY_IDLE_SEC,
    JoyMonitor,
)

require_sim2real_msg()

CMD_VEL_TOPIC = "/cmd_vel"
TARGET_DISTANCE_M = 0.50
DIST_DEADBAND_M = 0.10
LATERAL_DEADBAND_NORM = 0.20
LINEAR_X_MAG = 0.5
ANGULAR_Z_MAG = 1.5
LOOP_HZ = 20.0
WINDOW_NAME = "5 Finger Distance Hold Debug"
FULLSCREEN_DEFAULT = True
LOST_TIMEOUT_SEC = 0.6
LOG_HZ = 5.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def strong_cmd(
    err_m: float,
    deadband_m: float = DIST_DEADBAND_M,
    magnitude: float = LINEAR_X_MAG,
) -> float:
    if abs(err_m) <= deadband_m:
        return 0.0
    return magnitude if err_m > 0 else -magnitude


def strong_angular_cmd(
    dx_norm: float,
    deadband: float = LATERAL_DEADBAND_NORM,
    magnitude: float = ANGULAR_Z_MAG,
) -> float:
    """手在画面右侧(dx>0)则右转(angular.z<0)，与 locomotion 符号一致。"""
    if abs(dx_norm) <= deadband:
        return 0.0
    return -magnitude if dx_norm > 0 else magnitude


class PalmBootState:
    """手掌识别状态机：detect -> follow（有效手掌即刻进入）。"""

    DETECT = "detect"
    FOLLOW = "follow"

    def __init__(self):
        self.mode = self.DETECT
        self._detect_since = 0.0
        self._boot_since = 0.0
        self._lost_since = 0.0
        self._locked = False

    def reset(self):
        self.mode = self.DETECT
        self._detect_since = 0.0
        self._lost_since = 0.0
        self._locked = False

    def update(self, has_palm: bool) -> None:
        now = time.time()
        if has_palm:
            self._lost_since = 0.0
        else:
            if self._lost_since <= 0.0:
                self._lost_since = now
            if now - self._lost_since > LOST_TIMEOUT_SEC:
                self.reset()
                return

        if self.mode == self.DETECT:
            if has_palm:
                self.mode = self.FOLLOW
                self._locked = True
                print("\n[htrack] 识别手掌，进入手部跟踪", flush=True)
            else:
                self._detect_since = 0.0
            return


def should_publish_hand_cmd(
    joy_blocking: bool, cmd_x: float, cmd_z: float
) -> bool:
    """仅在有跟手速度且手柄未占用时发布，避免覆盖 teleop 的 /cmd_vel。"""
    if joy_blocking:
        return False
    return cmd_x != 0.0 or cmd_z != 0.0


def publish_stop_once(pub: rospy.Publisher) -> None:
    pub.publish(Twist())


def main():
    parser = argparse.ArgumentParser(
        description="手部跟踪：左右居中(angular.z)+手势5距离保持(linear.x)"
    )
    parser.add_argument("--gui", action="store_true", help="开启可视化窗口（默认关闭）")
    parser.add_argument("--hd1080", action="store_true")
    parser.add_argument("--proc-max-w", type=int, default=640)
    parser.add_argument("--dist-min", type=float, default=DIST_MIN_M)
    parser.add_argument("--dist-max", type=float, default=DIST_MAX_M)
    parser.add_argument("--no-fsm", action="store_true", help="跳过 FSM=EXEC_DEFAULT 检查")
    parser.add_argument("--dry-run", action="store_true", help="不发 /cmd_vel，只打印")
    parser.add_argument(
        "--no-joy",
        action="store_true",
        help="不监听 /joy 手柄仲裁",
    )
    args = parser.parse_args()

    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")

    rospy.init_node("hand_tracking", anonymous=False)
    fsm = None if args.no_fsm else FsmStateMonitor()
    if fsm is not None:
        rospy.loginfo("[htrack] 等待 FSM EXEC_DEFAULT(5)...")
        if fsm.wait_for_exec_default(timeout=30.0):
            rospy.loginfo("[htrack] FSM OK")
        else:
            rospy.logwarn("[htrack] FSM 超时，将继续但控制会受限")

    tracker = ZedHandTracker(
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        use_hd1080=args.hd1080,
        proc_max_w=args.proc_max_w,
    )

    pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
    joy = (
        None
        if args.no_joy
        else JoyMonitor(idle_sec=HAND_TRACKING_JOY_IDLE_SEC)
    )
    boot_state = PalmBootState()
    rate = rospy.Rate(LOOP_HZ)
    last_log_t = 0.0
    last_pub_active = False

    rospy.loginfo(
        "[htrack] 左右转: |dx|>%.0f%% -> angular.z=±%.1f; 手势5距离: linear.x=±%.1f Z=%.2fm",
        LATERAL_DEADBAND_NORM * 100,
        ANGULAR_Z_MAG,
        LINEAR_X_MAG,
        TARGET_DISTANCE_M,
    )
    joy_hint = (
        "无"
        if joy is None
        else f"/joy 占用后暂停 {HAND_TRACKING_JOY_IDLE_SEC:.0f}s"
    )
    rospy.loginfo("[htrack] 手柄仲裁: %s", joy_hint)
    print(
        "[htrack] 手掌入画即左右居中；手势5另做前后距离；"
        f"跟手中 {GESTURE_FOLLOW_LOST_SEC:.0f}s 无五指 → 手势识别；"
        f"手柄={joy_hint}；ESC退出",
        flush=True,
    )
    g5_watch = GestureFiveLostWatch(lost_sec=GESTURE_FOLLOW_LOST_SEC)
    return_to_gesture = False
    if args.gui:
        import cv2

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if FULLSCREEN_DEFAULT:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

    try:
        while not rospy.is_shutdown():
            frame, obs = tracker.process_frame(draw_landmarks=args.gui)
            fsm_ok = (fsm is None) or fsm.is_exec_default()

            has_palm = (
                obs.has_hand
                and obs.in_range
                and obs.palm_pos is not None
                and fsm_ok
            )
            dist_x = obs.palm_pos[0] if obs.palm_pos is not None else 0.0
            dist_z = obs.palm_pos[2] if obs.palm_pos is not None else 0.0
            dx_norm = obs.dx_norm
            boot_state.update(has_palm)
            engaged = boot_state.mode == PalmBootState.FOLLOW
            dist_follow = engaged and obs.gesture == GESTURE_FOLLOW
            is_gesture_five = (
                obs.has_hand
                and obs.in_range
                and obs.gesture == GESTURE_FOLLOW
            )
            if g5_watch.should_return_to_gesture(
                engaged=engaged,
                is_gesture_five=is_gesture_five,
            ):
                log_return_to_gesture(lost_sec=GESTURE_FOLLOW_LOST_SEC)
                return_to_gesture = True
                break

            joy_blocking = (
                joy is not None and joy.blocks_hand_tracking()
            )
            if joy is not None and joy.poll_takeover_edge():
                rospy.loginfo(
                    "[htrack] 手柄接管：%ds 内停止发布 /cmd_vel（不发送零速，避免盖手柄）",
                    HAND_TRACKING_JOY_IDLE_SEC,
                )
                last_pub_active = False

            cmd_x = 0.0
            cmd_z = 0.0
            mode = "idle"
            if joy_blocking:
                mode = "joy"
            elif engaged:
                cmd_z = strong_angular_cmd(dx_norm)
                mode = "yaw"
                if dist_follow:
                    err = dist_z - TARGET_DISTANCE_M
                    cmd_x = strong_cmd(err)
                    mode = "yaw+distance"
            elif has_palm:
                mode = "detect"

            if not args.dry_run:
                if joy_blocking:
                    last_pub_active = False
                elif should_publish_hand_cmd(joy_blocking, cmd_x, cmd_z):
                    msg = Twist()
                    msg.linear.x = cmd_x
                    msg.angular.z = cmd_z
                    pub.publish(msg)
                    last_pub_active = True
                elif last_pub_active:
                    publish_stop_once(pub)
                    last_pub_active = False

            now_t = time.time()
            if now_t - last_log_t >= 1.0 / LOG_HZ:
                last_log_t = now_t
                joy_left = (
                    joy.idle_remaining() if joy_blocking and joy is not None else 0.0
                )
                g5_left = (
                    max(0.0, GESTURE_FOLLOW_LOST_SEC - g5_watch.lost_elapsed())
                    if engaged and not is_gesture_five
                    else 0.0
                )
                tip = (
                    f"[htrack] g={obs.gesture} dx={dx_norm:+.2f} z={dist_z:.2f}m "
                    f"cmd_x={cmd_x:+.2f} cmd_z={cmd_z:+.2f} mode={mode}"
                )
                if engaged and g5_left > 0:
                    tip += f" g5_back={g5_left:.1f}s"
                if joy_blocking:
                    tip += f" joy_wait={joy_left:.1f}s"
                print(f"\r{tip:100s}", end="", flush=True)

            if args.gui:
                import cv2

                cv2.putText(
                    frame,
                    f"G:{obs.gesture} dx:{dx_norm:+.2f} Z:{dist_z:.2f}m",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"MODE:{mode} deadband_dx:{LATERAL_DEADBAND_NORM:.0%}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 128) if engaged else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"linear.x={cmd_x:+.2f} angular.z={cmd_z:+.2f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if return_to_gesture and not args.dry_run:
            release_before_gesture_return(
                tracker=tracker,
                pub=pub,
                last_pub_active=last_pub_active,
                dry_run=args.dry_run,
                no_gui=not args.gui,
            )
            exec_gesture_recognition()
        if not args.dry_run and last_pub_active:
            publish_stop_once(pub)
        tracker.close()
        if args.gui:
            import cv2

            cv2.destroyAllWindows()
        print("\n[htrack] 已退出", flush=True)


if __name__ == "__main__":
    main()
