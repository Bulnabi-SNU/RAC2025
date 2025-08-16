#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
수정사항
1.pause 이후의 resume 코드 없앰
2. cpu optimize: 해상도 downscale && ROI->가장자리는 인식 안함 시야각이 좁아지는 것은 감수해야(조정 가능)
3.영상 n초 뒤부터 테스트 가능(시작하자마자 pause 되는 것 때문에)

부족한 부분
1.웹캠으로 테스트 하는 부분 다시 추가해야
2.짐벌 조리개 때문에 고고도에서는 하기태그 색이 변해서 인식이 잘 안됨->고도를 조금 낮춰서 비행해야 할 듯
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class DropTagDetector:
    def __init__(self,
                 # HSV 빨강 범위 
                 lower_red1: np.ndarray = np.array([0,   100, 50], dtype=np.uint8),
                 upper_red1: np.ndarray = np.array([10,  255, 255], dtype=np.uint8),
                 lower_red2: np.ndarray = np.array([170, 100, 50], dtype=np.uint8),
                 upper_red2: np.ndarray = np.array([179, 255, 255], dtype=np.uint8),  
                 # 면적/임계
                 min_area: float = 500.0,
                 pause_threshold: float = 0.4,     # red_ratio 임계
                 # 최적화 옵션
                 process_scale: float = 0.6,       # 0.5~0.7 is recommended (i guess)
                 roi_margin: float = 0.15,         # ROI margin 15% wont be detected -> only for 70% of center 
                 close_iters: int = 2              # morphology close iterations
                 ):
        # 색 범위
        self.lower_red1 = lower_red1.astype(np.uint8)
        self.upper_red1 = upper_red1.astype(np.uint8)
        self.lower_red2 = lower_red2.astype(np.uint8)
        self.upper_red2 = np.array([min(int(upper_red2[0]), 179), upper_red2[1], upper_red2[2]], dtype=np.uint8)

        # 임계
        self.min_area = float(min_area)
        self.pause_threshold = float(pause_threshold)

        # 최적화
        self.process_scale = float(process_scale)
        self.roi_margin = float(roi_margin)
        self.close_iters = int(close_iters)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 상태
        self.detection_paused: bool = False   
        self.perma_paused: bool = False       

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    # ==============================
    # Public API
    # ==============================
    def detect_drop_tag(self, image):
        """
        Args:
            image: BGR frame
        Returns:
            (np.array([cx, cy]), np.array([red_ratio]))  또는 (None, None)
        """
        if image is None or len(image.shape) != 3:
            return None, None

        if self.perma_paused:
            self.detection_paused = True
            return None, None

        src_h, src_w = image.shape[:2]

        # ---------- downscale----------
        if self.process_scale != 1.0:
            small = cv2.resize(image, (int(src_w * self.process_scale), int(src_h * self.process_scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = image

        sh, sw = small.shape[:2]

        # ---------- red ratio ----------
        hsv_full = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask_full = self._mask_red(hsv_full)
        red_ratio = cv2.countNonZero(mask_full) / mask_full.size

        # 임계 초과 시: pause
        if red_ratio > self.pause_threshold:
            self.perma_paused = True
            self.detection_paused = True
            return None, None

        self.detection_paused = False

        # ---------- 중앙 ROI ----------
        mx = self.roi_margin
        x0 = int(sw * mx); x1 = int(sw * (1.0 - mx))
        y0 = int(sh * mx); y1 = int(sh * (1.0 - mx))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, sw, sh

        roi = small[y0:y1, x0:x1]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = self._mask_red(hsv_roi)

        # 노이즈 제거
        if self.close_iters > 0:
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        # ---------- 컨투어 중심 ----------
        center_roi = self._centroid_from_contours(mask_roi, (y1 - y0, x1 - x0))
        if center_roi is None:
            return None, None

        cx_r, cy_r = center_roi
        # ROI -> small 전체
        cx_s = x0 + cx_r
        cy_s = y0 + cy_r
        # small -> 원본
        cx = int(cx_s / self.process_scale)
        cy = int(cy_s / self.process_scale)

        return np.array([cx, cy], dtype=np.int32), np.array([red_ratio], dtype=np.float32)

    def update_param(self, lower_red1=None, upper_red1=None,
                     lower_red2=None, upper_red2=None,
                     min_area=None, pause_threshold=None,
                     process_scale=None, roi_margin=None,
                     close_iters=None):
        if lower_red1 is not None: self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        if upper_red1 is not None: self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        if lower_red2 is not None: self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        if upper_red2 is not None:
            u = np.array(upper_red2, dtype=np.uint8)
            self.upper_red2 = np.array([min(int(u[0]), 179), u[1], u[2]], dtype=np.uint8)
        if min_area is not None: self.min_area = float(min_area)
        if pause_threshold is not None: self.pause_threshold = float(pause_threshold)
        if process_scale is not None: self.process_scale = float(process_scale)
        if roi_margin is not None: self.roi_margin = float(roi_margin)
        if close_iters is not None: self.close_iters = int(close_iters)

    def print_param(self):
        print("DropTagDetector Parameters:")
        print(f"- HSV Red1: {self.lower_red1} ~ {self.upper_red1}")
        print(f"- HSV Red2: {self.lower_red2} ~ {self.upper_red2} (Hue<=179)")
        print(f"- Min Area(px): {self.min_area}")
        print(f"- Pause Threshold: {self.pause_threshold}")
        print(f"- Process Scale: {self.process_scale}")
        print(f"- ROI Margin: {self.roi_margin}")
        print(f"- CLOSE iterations: {self.close_iters}")
        print(f"- PermaPaused: {self.perma_paused}, Paused(now): {self.detection_paused}")

    # ==============================
    # Internals
    # ==============================
    def _mask_red(self, hsv: np.ndarray) -> np.ndarray:
        m1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        m2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        return cv2.bitwise_or(m1, m2)

    def _centroid_from_contours(self, mask: np.ndarray, hw: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.min_area:
            return None
        M = cv2.moments(cnt)
        if M.get('m00', 0) == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)


# ==============================
# Standalone test (recorded video / webcam)
# ==============================
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="DropTagDetector Tester")
    parser.add_argument("--video", type=str, default="video_path",
                        help="Path to video file or camera index")
    parser.add_argument("--display-scale", type=float, default=0.7, help="Display downscale factor")
    args = parser.parse_args()

    src = args.video
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/video: {src}")
        raise SystemExit(1)
    
    # --- Skip first n seconds ---드론을 하기태그 위에서 띄우는 바람에 시작하자마자 pause 되는 거 방지
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps_read * 0))  # Jump to frame at n seconds

    det = DropTagDetector(
        process_scale=0.6,
        roi_margin=0.15,
        close_iters=2,
        pause_threshold=0.4
    )

    print("Keys: 'q' quit | 'i' toggle info | 'b' toggle ROI box")
    show_info = True
    show_roi_box = True
    fps = 0.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection, red_ratio = det.detect_drop_tag(frame)

        # FPS 계산
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        disp = frame.copy()
        h, w = disp.shape[:2]

        # 중앙 십자
        cx0, cy0 = w // 2, h // 2
        cv2.line(disp, (cx0 - 20, cy0), (cx0 + 20, cy0), (0, 255, 0), 2)
        cv2.line(disp, (cx0, cy0 - 20), (cx0, cy0 + 20), (0, 255, 0), 2)

        # ROI 박스 표시
        if show_roi_box and 0 <= det.roi_margin < 0.5:
            sw = int(w * det.process_scale)
            sh = int(h * det.process_scale)
            mx = det.roi_margin
            x0 = int(sw * mx); x1 = int(sw * (1.0 - mx))
            y0 = int(sh * mx); y1 = int(sh * (1.0 - mx))
            # small 좌표 → 원본 좌표
            x0_f = int(x0 / det.process_scale); x1_f = int(x1 / det.process_scale)
            y0_f = int(y0 / det.process_scale); y1_f = int(y1 / det.process_scale)
            cv2.rectangle(disp, (x0_f, y0_f), (x1_f, y1_f), (255, 255, 0), 2)

        # 결과 표시
        if detection is not None:
            cx, cy = int(detection[0]), int(detection[1])
            cv2.circle(disp, (cx, cy), 10, (0, 0, 255), 2)
            cv2.putText(disp, f"DropTag: ({cx}, {cy})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            msg = "PAUSED " if det.perma_paused else "No detection"
            cv2.putText(disp, msg, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        # 정보 오버레이
        if show_info:
            rtxt = f"{red_ratio[0]:.3f}" if isinstance(red_ratio, np.ndarray) else "N/A"
            state = "PAUSED" if det.detection_paused else "RUN"
            latch = "ON" if det.perma_paused else "OFF"
            cv2.putText(disp, f"State: {state} | Latch: {latch}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(disp, f"Red ratio: {rtxt} | FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 디스플레이 스케일
        if args.display_scale != 1.0:
            disp = cv2.resize(disp, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow("DropTag Detection Test", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            show_info = not show_info
        elif key == ord('b'):
            show_roi_box = not show_roi_box
        
    cap.release()
    cv2.destroyAllWindows()
