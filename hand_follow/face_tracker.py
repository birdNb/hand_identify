#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内嵌脸部跟踪：与 hand_perception 共用 ZED 降采样 RGB，不再启动 locate_face 子进程。

控制律与 locate_face/locate_face.py 一致，发布 /pi_plus_absolute。
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from gesture_actions import GESTURE_FACE_TRACK, log_face_track_toggle

# ----- 与 locate_face.py 对齐 -----
ABSOLUTE_TOPIC = "/pi_plus_absolute"
HEAD_YAW_JOINT = "head_yaw_joint"
HEAD_PITCH_JOINT = "head_pitch_joint"

DETECT_CONFIDENCE = 0.4
TRACK_CONFIDENCE = 0.5
ROI_PAD_RATIO = 0.30

DEAD_BAND_X = 0.04
DEAD_BAND_Y = 0.05
K_YAW_DEG = 20.0
K_PITCH_DEG = 15.0
MAX_STEP_YAW_DEG = 6.0
MAX_STEP_PITCH_DEG = 5.0
TARGET_EMA_ALPHA = 0.6
YAW_LIMIT_DEG = 80.0
PITCH_UP_DEG = -40.0
PITCH_DOWN_DEG = 60.0
PUBLISH_RATE_HZ = 50
NO_FACE_RETURN_HOME_SEC = 1.0
RETURN_HOME_RATE_DEG_PER_SEC = 45.0

TOGGLE_COOLDOWN_SEC = 1.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _face_bbox_from_landmarks(landmarks, w, h, pad=10):
    xs = [p.x for p in landmarks.landmark]
    ys = [p.y for p in landmarks.landmark]
    x1 = max(0, int(min(xs) * w) - pad)
    y1 = max(0, int(min(ys) * h) - pad)
    x2 = min(w - 1, int(max(xs) * w) + pad)
    y2 = min(h - 1, int(max(ys) * h) + pad)
    return x1, y1, x2, y2


def _detect_face_roi_bbox(face_detector, rgb, w, h, pad_ratio=ROI_PAD_RATIO):
    det = face_detector.process(rgb)
    if not det.detections:
        return None
    best = max(det.detections, key=lambda d: d.score[0])
    rel = best.location_data.relative_bounding_box
    bx = rel.xmin * w
    by = rel.ymin * h
    bw = rel.width * w
    bh = rel.height * h
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    x1 = max(0, int(bx - pad_x))
    y1 = max(0, int(by - pad_y))
    x2 = min(w, int(bx + bw + pad_x))
    y2 = min(h, int(by + bh + pad_y))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


def _update_target_from_error(
    yaw_cur_rad: float,
    pitch_cur_rad: float,
    dx_n: float,
    dy_n: float,
    state: Dict[str, float],
) -> Tuple[float, float]:
    if abs(dx_n) < DEAD_BAND_X:
        dx_n = 0.0
    if abs(dy_n) < DEAD_BAND_Y:
        dy_n = 0.0

    delta_yaw_deg = _clamp(-K_YAW_DEG * dx_n, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG)
    delta_pitch_deg = _clamp(
        K_PITCH_DEG * dy_n, -MAX_STEP_PITCH_DEG, MAX_STEP_PITCH_DEG,
    )

    base_yaw_rad = state.get("yaw_rad", yaw_cur_rad)
    base_pitch_rad = state.get("pitch_rad", pitch_cur_rad)
    raw_yaw_rad = base_yaw_rad + math.radians(delta_yaw_deg)
    raw_pitch_rad = base_pitch_rad + math.radians(delta_pitch_deg)

    a = TARGET_EMA_ALPHA
    yaw_new = base_yaw_rad * (1 - a) + raw_yaw_rad * a
    pitch_new = base_pitch_rad * (1 - a) + raw_pitch_rad * a

    yaw_new = _clamp(
        yaw_new, -math.radians(YAW_LIMIT_DEG), math.radians(YAW_LIMIT_DEG),
    )
    pitch_new = _clamp(
        pitch_new, math.radians(PITCH_UP_DEG), math.radians(PITCH_DOWN_DEG),
    )
    state["yaw_rad"] = yaw_new
    state["pitch_rad"] = pitch_new
    return yaw_new, pitch_new


class _NeckTarget:
    def __init__(self):
        self._lock = threading.Lock()
        self._yaw = 0.0
        self._pitch = 0.0

    def set(self, yaw_rad: float, pitch_rad: float):
        with self._lock:
            self._yaw = yaw_rad
            self._pitch = pitch_rad

    def get(self):
        with self._lock:
            return self._yaw, self._pitch


class _FaceNeckPublisher(threading.Thread):
    """固定频率发布脖子目标到 /pi_plus_absolute。"""

    def __init__(self, target: _NeckTarget, fsm, dry_run: bool):
        super().__init__(daemon=True)
        self._target = target
        self._fsm = fsm
        self._dry_run = dry_run
        self._stop_evt = threading.Event()
        self._pub = rospy.Publisher(ABSOLUTE_TOPIC, JointState, queue_size=10)
        self._rate = rospy.Rate(PUBLISH_RATE_HZ)

    def stop(self):
        self._stop_evt.set()

    def publish_center_blocking(self, duration: float = 0.5):
        msg = JointState()
        msg.name = [HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
        msg.position = [0.0, 0.0]
        msg.velocity = []
        msg.effort = []
        end_t = time.time() + duration
        while time.time() < end_t:
            if not self._dry_run:
                msg.header.stamp = rospy.Time.now()
                self._pub.publish(msg)
            time.sleep(1.0 / max(PUBLISH_RATE_HZ, 1))

    def run(self):
        msg = JointState()
        msg.name = [HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
        msg.velocity = []
        msg.effort = []
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            if self._fsm is not None and not self._fsm.is_exec_default():
                self._rate.sleep()
                continue
            yaw, pitch = self._target.get()
            msg.position = [yaw, pitch]
            msg.header.stamp = rospy.Time.now()
            if not self._dry_run:
                self._pub.publish(msg)
            self._rate.sleep()


@dataclass
class FaceOverlay:
    has_face: bool = False
    used_roi: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None


class IntegratedFaceTracker:
    """
    与 ZedHandTracker 共用 proc RGB；手势 1 开关；默认可由 hand_follow 自动开启。
    """

    def __init__(
        self,
        fsm,
        *,
        dry_run: bool = False,
        cooldown_sec: float = TOGGLE_COOLDOWN_SEC,
    ):
        self._fsm = fsm
        self._dry_run = dry_run
        self._cooldown_sec = max(0.2, float(cooldown_sec))
        self._enabled = False
        self._last_gesture = -1
        self._last_toggle_t = 0.0

        self._target = _NeckTarget()
        self._ctrl_state: Dict[str, float] = {"yaw_rad": 0.0, "pitch_rad": 0.0}
        self._last_face_t = time.time()
        self._last_loop_t = time.time()
        self._homing_logged = False
        self._frame_i = 0
        self._overlay = FaceOverlay()
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=DETECT_CONFIDENCE,
            min_tracking_confidence=TRACK_CONFIDENCE,
        )
        self._face_mesh_roi = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=DETECT_CONFIDENCE,
            min_tracking_confidence=TRACK_CONFIDENCE,
        )
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=DETECT_CONFIDENCE,
        )
        self._publisher = _FaceNeckPublisher(self._target, fsm, dry_run)
        self._publisher.start()
        rospy.loginfo(
            "[face_track] 内嵌脸跟踪(共用ZED RGB), dry_run=%s", dry_run,
        )

    @property
    def is_active(self) -> bool:
        return self._enabled

    @property
    def overlay(self) -> FaceOverlay:
        return self._overlay

    def start(self) -> bool:
        if self._enabled:
            return False
        self._enabled = True
        self._last_face_t = time.time()
        self._homing_logged = False
        if not self._dry_run:
            log_face_track_toggle(enabled=True, dry_run=False)
        else:
            log_face_track_toggle(enabled=True, dry_run=True)
        rospy.loginfo("[face_track] 脸部跟踪已开启 (共用相机RGB)")
        return True

    def stop(self, *, reason: str = "关闭") -> bool:
        if not self._enabled:
            return False
        self._enabled = False
        self._target.set(0.0, 0.0)
        self._ctrl_state = {"yaw_rad": 0.0, "pitch_rad": 0.0}
        if not self._dry_run:
            self._publisher.publish_center_blocking(0.35)
        log_face_track_toggle(
            enabled=False, dry_run=self._dry_run, reason=reason,
        )
        rospy.loginfo("[face_track] 脸部跟踪已关闭 (%s)", reason)
        return True

    def toggle(self) -> bool:
        if self._enabled:
            self.stop(reason="手势1")
            return True
        return self.start()

    def update(
        self,
        gesture: int,
        *,
        has_hand: bool,
        in_range: bool,
        joy_blocking: bool = False,
        fsm_ok: bool = True,
    ) -> bool:
        prev = self._last_gesture
        self._last_gesture = gesture
        if joy_blocking or not fsm_ok:
            return False
        if not has_hand or not in_range:
            return False
        if gesture != GESTURE_FACE_TRACK:
            return False
        if gesture == prev:
            return False
        if time.time() - self._last_toggle_t < self._cooldown_sec:
            return False
        self._last_toggle_t = time.time()
        self.toggle()
        return True

    def process_shared_rgb(
        self,
        rgb_mp,
        frame_bgr,
        proc_w: int,
        proc_h: int,
    ) -> FaceOverlay:
        """在手势 MediaPipe 之前/之后调用，rgb_mp 与手部检测共用。"""
        overlay = FaceOverlay()
        if not self._enabled:
            self._overlay = overlay
            return overlay

        if self._fsm is not None and not self._fsm.is_exec_default():
            self._overlay = overlay
            return overlay

        loop_now = time.time()
        dt_frame = max(1e-3, min(0.2, loop_now - self._last_loop_t))
        self._last_loop_t = loop_now

        h, w = frame_bgr.shape[:2]
        rgb_in = rgb_mp
        if not rgb_mp.flags["C_CONTIGUOUS"]:
            rgb_in = np.ascontiguousarray(rgb_mp)

        res = self._face_mesh.process(rgb_in)
        used_roi = False

        self._frame_i += 1
        # ROI 兜底较重，每 3 帧跑一次，减轻 CPU 卡顿
        if not res.multi_face_landmarks and (self._frame_i % 3 == 0):
            bbox = _detect_face_roi_bbox(
                self._face_detector, rgb_in, proc_w, proc_h,
            )
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = np.ascontiguousarray(rgb_in[y1:y2, x1:x2])
                if roi.size > 0 and roi.shape[0] >= 20 and roi.shape[1] >= 20:
                    res2 = self._face_mesh_roi.process(roi)
                else:
                    res2 = None
                if res2 is not None and res2.multi_face_landmarks:
                    rw_roi = x2 - x1
                    rh_roi = y2 - y1
                    for lms in res2.multi_face_landmarks:
                        for lm in lms.landmark:
                            lm.x = (lm.x * rw_roi + x1) / proc_w
                            lm.y = (lm.y * rh_roi + y1) / proc_h
                    res = res2
                    used_roi = True

        cx_img, cy_img = w / 2.0, h / 2.0
        scale_x = w / max(proc_w, 1)
        scale_y = h / max(proc_h, 1)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            px1, py1, px2, py2 = _face_bbox_from_landmarks(
                lm, proc_w, proc_h, pad=12,
            )
            fx1 = int(px1 * scale_x)
            fy1 = int(py1 * scale_y)
            fx2 = int(px2 * scale_x)
            fy2 = int(py2 * scale_y)
            face_cx = (fx1 + fx2) / 2.0
            face_cy = (fy1 + fy2) / 2.0
            dx_n = (face_cx - cx_img) / (w / 2.0)
            dy_n = (face_cy - cy_img) / (h / 2.0)

            cur_yaw, cur_pitch = self._target.get()
            new_yaw, new_pitch = _update_target_from_error(
                cur_yaw, cur_pitch, dx_n, dy_n, self._ctrl_state,
            )
            if not self._dry_run:
                self._target.set(new_yaw, new_pitch)
            self._last_face_t = loop_now
            self._homing_logged = False

            overlay.has_face = True
            overlay.used_roi = used_roi
            overlay.bbox = (fx1, fy1, fx2, fy2)
            overlay.center = (int(face_cx), int(face_cy))
        else:
            lost_dur = loop_now - self._last_face_t
            if (
                NO_FACE_RETURN_HOME_SEC > 0
                and lost_dur > NO_FACE_RETURN_HOME_SEC
            ):
                cur_yaw, cur_pitch = self._target.get()
                step_rad = math.radians(
                    RETURN_HOME_RATE_DEG_PER_SEC * dt_frame,
                )
                new_yaw = (
                    cur_yaw - math.copysign(min(step_rad, abs(cur_yaw)), cur_yaw)
                    if abs(cur_yaw) > 1e-4 else 0.0
                )
                new_pitch = (
                    cur_pitch - math.copysign(
                        min(step_rad, abs(cur_pitch)), cur_pitch,
                    )
                    if abs(cur_pitch) > 1e-4 else 0.0
                )
                if not self._dry_run:
                    self._target.set(new_yaw, new_pitch)
                self._ctrl_state["yaw_rad"] = new_yaw
                self._ctrl_state["pitch_rad"] = new_pitch
                if not self._homing_logged:
                    rospy.loginfo_throttle(
                        5.0,
                        "[face_track] 无人脸 %.1fs, 平滑回中",
                        lost_dur,
                    )
                    self._homing_logged = True

        self._overlay = overlay
        return overlay

    def draw_overlay(self, frame_bgr) -> None:
        ov = self._overlay
        if not self._enabled or not ov.has_face or ov.bbox is None:
            return
        fx1, fy1, fx2, fy2 = ov.bbox
        col = (0, 255, 255) if ov.used_roi else (0, 255, 0)
        cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), col, 2)
        if ov.center is not None:
            cx_i = frame_bgr.shape[1] // 2
            cy_i = frame_bgr.shape[0] // 2
            cv2.circle(frame_bgr, ov.center, 6, col, -1)
            cv2.line(
                frame_bgr, (cx_i, cy_i), ov.center, col, 2,
            )

    def shutdown(self):
        self.stop(reason="程序退出")
        self._publisher.stop()
        self._face_mesh.close()
        self._face_mesh_roi.close()
        self._face_detector.close()


# 兼容旧导入名
FaceTrackingLauncher = IntegratedFaceTracker
