#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Optional, Tuple


class DropTagDetector:
    def __init__(self,
                 lower_red1=np.array([0, 50, 50], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 50, 50], dtype=np.uint8),
                 upper_red2=np.array([179, 255, 255], dtype=np.uint8),
                 min_area=5000,
                 pause_threshold=0.4,
                 far_min_pixels=1000,
                 roi_margin=0,
                 close_iters=2):
        # 색 범위
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

        # 임계
        self.min_area = float(min_area)
        self.pause_threshold = float(pause_threshold)
        self.far_min_pixels = int(far_min_pixels)

        # 최적화
        self.roi_margin = float(roi_margin)
        self.close_iters = int(close_iters)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 상태
        self.perma_paused = False
        self.detection_paused = False
        self.last_mode = ""

    def detect_drop_tag(self, frame):
        if frame is None or len(frame.shape) != 3:
            return None, None

        if self.perma_paused:
            self.detection_paused = True
            return None, None

        h, w = frame.shape[:2]

        # ---------- ROI 영역만 ----------
        x0, y0, x1, y1 = self._roi_box(w, h, self.roi_margin)
        roi_bgr = frame[y0:y1, x0:x1]

        # HSV 변환은 ROI에서만
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask_roi = self._mask_red(hsv_roi)

        if self.close_iters > 0:
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        # ROI 내 red ratio 계산 (전체 대비 비율로 맞추기 위해 ROI 사이즈 사용)
        red_ratio = cv2.countNonZero(mask_roi) / ((y1 - y0) * (x1 - x0))

        # pause 
        if red_ratio > self.pause_threshold:
            self.perma_paused = True
            self.detection_paused = True
            return None, None
        self.detection_paused = False

        # 가까이: contour 중심
        center_roi = self._centroid_from_contours(mask_roi)
        if center_roi is not None:
            self.last_mode = "close"
            cx, cy = self._map_from_roi(center_roi, x0, y0)
            return np.array([cx, cy]), np.array([red_ratio])

        # 멀리: 픽셀 평균
        far_center = self._mean_from_mask(mask_roi, self.far_min_pixels)
        if far_center is not None:
            self.last_mode = "far"
            cx, cy = self._map_from_roi(far_center, x0, y0)
            return np.array([cx, cy]), np.array([red_ratio])

        self.last_mode = ""
        return None, None

    # 내부 함수들
    def _mask_red(self, hsv):
        m1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        m2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        return cv2.bitwise_or(m1, m2)

    def _roi_box(self, w, h, margin):
        x0 = int(w * margin); x1 = int(w * (1.0 - margin))
        y0 = int(h * margin); y1 = int(h * (1.0 - margin))
        if x1 <= x0 or y1 <= y0:
            return 0, 0, w, h
        return x0, y0, x1, y1

    def _centroid_from_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area:
            return None
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

    def _mean_from_mask(self, mask, min_pix):
        ys, xs = np.where(mask > 0)
        if len(xs) < min_pix:
            return None
        return (int(xs.mean()), int(ys.mean()))

    def _map_from_roi(self, pt, x0, y0):
        cx, cy = pt
        return x0 + cx, y0 + cy


# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="/workspace/data/droptag_drone.avi")  # 0 for testing with webcam
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height") # can change resolution (640x480 or 320x240)
    parser.add_argument("--skip-sec", type=int, default=3, help="skip first n seconds")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if not args.video.isdigit() else int(args.video))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {args.video}")
        exit(1)

    # 처음부터 해상도 낮춰 읽기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * args.skip_sec))

    det = DropTagDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection, red_ratio = det.detect_drop_tag(frame)
        disp = frame.copy()

        if detection is not None:
            cv2.circle(disp, tuple(detection), 10, (0, 0, 255), 2)
            cv2.putText(disp, f"({detection[0]}, {detection[1]}) mode={det.last_mode}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            msg = "PAUSED" if det.perma_paused else "No detection"
            cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,215,255), 2)

        cv2.imshow("DropTag Test Optimized", disp)
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
