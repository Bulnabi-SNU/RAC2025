
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Optional, Tuple


class DropTagDetector:
    def __init__(self,
                 lower_red1=np.array([0, 100, 120], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 100, 120], dtype=np.uint8),
                 upper_red2=np.array([179, 255, 255], dtype=np.uint8),
                 # 비율 입력(≤1.0) 시 해상도/ROI 무관 동작
                 # 1280x720 기준 예: 5000/921600 ≈ 0.0054253472
                 min_area=0.0054253472,
                 pause_threshold=0.25,
                 # far 모드 제거했지만, 하위호환을 위해 파라미터는 보존
                 far_min_pixels=0.0010850694,
                 roi_margin=0.0,
                 close_iters=2):
        # 색 범위
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

        # 임계(≤1.0이면 비율, >1.0이면 절대 픽셀)
        self.min_area = float(min_area)
        self.pause_threshold = float(pause_threshold)
        self.far_min_pixels = float(far_min_pixels)  # (미사용; 하위호환 유지)

        # 최적화
        self.roi_margin = float(roi_margin)
        self.close_iters = int(close_iters)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 상태
        self.perma_paused = False
        self.detection_paused = False
        self.last_mode = ""  # "close"만 사용 (far 제거)

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    def detect_drop_tag(self, frame):
        if frame is None or len(frame.shape) != 3:
            return None, None

        if self.perma_paused:
            self.detection_paused = True
            return None, None

        h, w = frame.shape[:2]

        # ---------- ROI 계산 ----------
        x0, y0, x1, y1 = self._roi_box(w, h, self.roi_margin)
        roi_bgr = frame[y0:y1, x0:x1]
        rh, rw = roi_bgr.shape[:2]
        if rh <= 0 or rw <= 0:
            return None, None

        # ---------- ROI 기준 절대 임계치 환산(비율 입력 지원) ----------
        min_area_abs, _ = self._resolve_thresholds(rw, rh)  # far_min은 미사용

        # ---------- HSV/마스크 (ROI만) ----------
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask_roi = self._mask_red(hsv_roi)

        # 아크릴 반사로 생기는 구멍 메우기
        if self.close_iters > 0:
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        # ---------- ROI 내 red ratio ----------
        roi_area = float(rw * rh)
        red_ratio = cv2.countNonZero(mask_roi) / roi_area if roi_area > 0 else 0.0

        # 퍼머넌트 일시정지 (태그가 화면 대부분일 때 하강/정지 상황 가정)
        if red_ratio > self.pause_threshold:
            self.perma_paused = True
            self.detection_paused = True
            return None, None
        self.detection_paused = False

        # ---------- 가까이: 컨투어 중심(유일 모드, far 제거) ----------
        center_roi = self._centroid_from_contours(mask_roi, min_area_abs)
        if center_roi is not None:
            self.last_mode = "close"
            cx, cy = self._map_from_roi(center_roi, x0, y0)
            return np.array([cx, cy]), np.array([red_ratio], dtype=np.float32)

        # 미검출
        self.last_mode = ""
        return None, None

    # ---------------- 내부 함수들 ----------------
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

    def _centroid_from_contours(self, mask, min_area_abs: float):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < float(min_area_abs):
            return None
        M = cv2.moments(cnt)
        m00 = M.get('m00', 0.0)
        if m00 <= 1e-6:
            return None
        return (int(M['m10'] / m00), int(M['m01'] / m00))

    def _map_from_roi(self, pt, x0, y0):
        cx, cy = pt
        return x0 + cx, y0 + cy

    def _resolve_thresholds(self, w: int, h: int):
        """
        전달된 w*h를 전체 픽셀수로 보고,
        - min_area: ≤1.0 → 비율*픽셀수, >1.0 → 절대 픽셀
        - far_min_pixels: 동일 규칙(현재 미사용)
        """
        total = float(w * h)
        if self.min_area <= 1.0:
            min_area_abs = self.min_area * total
        else:
            min_area_abs = self.min_area

        if self.far_min_pixels <= 1.0:
            far_min_abs = self.far_min_pixels * total
        else:
            far_min_abs = self.far_min_pixels

        return float(min_area_abs), float(far_min_abs)


# ==============================
# Test code 
# ==============================
if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="/workspace/src/vision_processing_nodes/vision_processing_nodes/detection/path.mp4")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--skip-sec", type=int, default=10, help="start playing from t = n seconds")
    parser.add_argument("--roi-margin", type=float, default=0.0, help="0.0~0.45 권장, ROI 크기 비율")
    parser.add_argument("--close-iters", type=int, default=2, help="morph close iterations (fill holes)")
    args = parser.parse_args()

    # 웹캠/파일 구분
    is_camera = args.video.isdigit()

    cap = cv2.VideoCapture(int(args.video) if is_camera else args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {args.video}")
        exit(1)

    # 입력 해상도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 시크/지연
    def jump_to_seconds(capture, seconds):
        if seconds <= 0:
            return
        if is_camera:
            time.sleep(seconds); return
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000.0)
        except Exception:
            pass
        fps_local = capture.get(cv2.CAP_PROP_FPS)
        if not fps_local or fps_local <= 1e-3:
            fps_local = 30.0
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(fps_local * seconds))
        except Exception:
            pass

    jump_to_seconds(cap, args.skip_sec)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    det = DropTagDetector(roi_margin=args.roi_margin, close_iters=args.close_iters)

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection, red_ratio = det.detect_drop_tag(frame)
        disp = frame.copy()

        if detection is not None:
            cv2.circle(disp, tuple(detection.astype(int)), 10, (0, 0, 255), 2)
            cv2.putText(disp, f"({int(detection[0])}, {int(detection[1])}) mode={det.last_mode}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if red_ratio is not None:
                cv2.putText(disp, f"red ratio={float(red_ratio[0]):.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            msg = "PAUSED" if det.perma_paused else "No detection"
            cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,215,255), 2)

        cv2.imshow("DropTag Detection (Contour-only, no FAR)", disp)
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
