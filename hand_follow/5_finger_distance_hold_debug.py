#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立调试：五指手势(5)前后距离保持（仅 linear.x）。

特性：
- 仅测试距离保持，不接入脸跟踪与动作库
- 距离轴使用 palm_pos[2]（Z 深度）
- 识别到手势5后立即进入距离跟随
- 前后速度使用强驱动符号控制：误差超阈值直接发送 +1/-1
"""

import argparse
import os
import sys
import time

import rospy
from geometry_msgs.msg import Twist

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ros_setup import require_sim2real_msg
from hand_perception import DIST_MAX_M, DIST_MIN_M, ZedHandTracker
from hand_follow_control import FsmStateMonitor

require_sim2real_msg()

CMD_VEL_TOPIC = "/cmd_vel"
GESTURE_FOLLOW = 5
TARGET_DISTANCE_M = 0.50
DIST_DEADBAND_M = 0.10
CMD_MAX = 1.0
LOOP_HZ = 20.0
WINDOW_NAME = "5 Finger Distance Hold Debug"
FULLSCREEN_DEFAULT = True
LOST_TIMEOUT_SEC = 0.6
LOG_HZ = 5.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def strong_cmd(err_m: float, deadband_m: float = DIST_DEADBAND_M) -> float:
    if abs(err_m) <= deadband_m:
        return 0.0
    return CMD_MAX if err_m > 0 else -CMD_MAX


class PalmBootState:
    """手掌识别状态机：detect -> follow（手势5即刻进入）。"""

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
                print("\n[5hold] 识别手势5，进入距离跟随模式", flush=True)
            else:
                self._detect_since = 0.0
            return


def publish_stop(pub: rospy.Publisher, sec: float = 0.4):
    msg = Twist()
    end_t = time.time() + sec
    while time.time() < end_t and not rospy.is_shutdown():
        pub.publish(msg)
        time.sleep(1.0 / LOOP_HZ)


def main():
    parser = argparse.ArgumentParser(description="五指(5)距离保持独立调试：仅前后控制")
    parser.add_argument("--gui", action="store_true", help="开启可视化窗口（默认关闭）")
    parser.add_argument("--hd1080", action="store_true")
    parser.add_argument("--proc-max-w", type=int, default=640)
    parser.add_argument("--dist-min", type=float, default=DIST_MIN_M)
    parser.add_argument("--dist-max", type=float, default=DIST_MAX_M)
    parser.add_argument("--no-fsm", action="store_true", help="跳过 FSM=EXEC_DEFAULT 检查")
    parser.add_argument("--dry-run", action="store_true", help="不发 /cmd_vel，只打印")
    args = parser.parse_args()

    if args.dist_min >= args.dist_max:
        raise SystemExit("--dist-min 必须小于 --dist-max")

    rospy.init_node("five_finger_distance_hold_debug", anonymous=False)
    fsm = None if args.no_fsm else FsmStateMonitor()
    if fsm is not None:
        rospy.loginfo("[5hold] 等待 FSM EXEC_DEFAULT(5)...")
        if fsm.wait_for_exec_default(timeout=30.0):
            rospy.loginfo("[5hold] FSM OK")
        else:
            rospy.logwarn("[5hold] FSM 超时，将继续但控制会受限")

    tracker = ZedHandTracker(
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        use_hd1080=args.hd1080,
        proc_max_w=args.proc_max_w,
    )

    pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
    boot_state = PalmBootState()
    rate = rospy.Rate(LOOP_HZ)
    last_log_t = 0.0

    rospy.loginfo(
        "[5hold] 五指距离保持：目标Z=%.2fm, 手势5即刻进入, 强驱动(+/-1), deadband=%.2fm",
        TARGET_DISTANCE_M,
        DIST_DEADBAND_M,
    )
    print("[5hold] 识别到手势5即进入距离保持；ESC退出", flush=True)
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

            active = (
                obs.has_hand
                and obs.in_range
                and obs.gesture == GESTURE_FOLLOW
                and fsm_ok
            )

            dist_x = obs.palm_pos[0] if obs.palm_pos is not None else 0.0
            dist_z = obs.palm_pos[2] if obs.palm_pos is not None else 0.0
            has_palm = active and (obs.palm_pos is not None)
            boot_state.update(has_palm)
            engaged = boot_state.mode == PalmBootState.FOLLOW

            cmd_x = 0.0
            mode = "idle"
            if engaged:
                err = dist_z - TARGET_DISTANCE_M
                cmd_x = strong_cmd(err)
                mode = "follow"
            elif has_palm:
                mode = "detect"

            if not args.dry_run:
                msg = Twist()
                msg.linear.x = cmd_x
                pub.publish(msg)

            now_t = time.time()
            if now_t - last_log_t >= 1.0 / LOG_HZ:
                last_log_t = now_t
                tip = (
                    f"[5hold] g={obs.gesture} x={dist_x:.2f} z={dist_z:.2f}m "
                    f"cmd_x={cmd_x:+.2f} mode={mode}"
                )
                print(f"\r{tip:100s}", end="", flush=True)

            if args.gui:
                import cv2

                cv2.putText(
                    frame,
                    f"G:{obs.gesture} X:{dist_x:.2f}m Z:{dist_z:.2f}m",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"TARGET_Z:{TARGET_DISTANCE_M:.2f}m MODE:{mode}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 128) if engaged else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"linear.x={cmd_x:+.2f}",
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
        if not args.dry_run:
            publish_stop(pub, 0.5)
        tracker.close()
        if args.gui:
            import cv2

            cv2.destroyAllWindows()
        print("\n[5hold] 已退出", flush=True)


if __name__ == "__main__":
    main()
