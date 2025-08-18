#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Casualty Detection Function
Takes an image and outputs pixel position of detected object's center

- HSV 마스크 기반(빨강) 탐지
- CPU 절약: 다운스케일(process_scale) 후 처리, 커널 캐싱
- 안정성: (옵션) OPEN → CLOSE 형태 전처리
- 반환 형식: (np.array([cx, cy]), None) 또는 (None, None)
"""

import numpy as np
import cv2


class CasualtyDetector:
    def __init__(self, 
                 lower_red1: np.ndarray = np.array([0, 70, 150]),
                 upper_red1: np.ndarray = np.array([10, 255, 255]),
                 lower_red2: np.ndarray = np.array([170, 70, 150]),
                 upper_red2: np.ndarray = np.array([180, 255, 255]),
                 min_area: float = 500.0,
                 process_scale: float = 0.6,
                 use_open: bool = True,
                 open_iters: int = 1,
                 close_iters: int = 2):
        """
        Args:
            lower_red1/upper_red1/lower_red2/upper_red2: HSV thresholds (np.uint8로 내부 캐스팅)
            min_area: Minimum contour area 
            process_scale: Downscale factor (0.1~1.0, default 0.6)
            use_open: 작은 점 노이즈 제거용 OPEN 사용 여부
            open_iters: OPEN 반복 횟수
            close_iters: CLOSE 반복 횟수
        """
        # HSV 임계 안전 캐스팅
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)

        self.min_area = float(min_area)
        self.process_scale = float(np.clip(process_scale, 0.1, 1.0))

        self.use_open = bool(use_open)
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))

        # 커널 캐싱(매 프레임 생성 방지)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    def detect_casualty(self, image):
        src_h, src_w = image.shape[:2]

        # ---------- 다운스케일 ----------
        if self.process_scale != 1.0:
            small = cv2.resize(
                image,
                (int(src_w * self.process_scale), int(src_h * self.process_scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            small = image

        # 스케일에 맞춘 면적 임계값(면적은 scale^2로 줄어듦)
        scaled_min_area = self.min_area * (self.process_scale ** 2)

        # 1) 가까이용(가장 큰 컨투어 중심)
        pt = self.detect_casualty_close(small, min_area_px=scaled_min_area)
        # 2) 실패 시 원거리용(마스크 평균)
        if pt is None:
            pt = self.detect_casualty_far(small)
        if pt is None:
            return None, None

        # 좌표를 원본 해상도로 환산
        if self.process_scale != 1.0:
            cx = int(round(pt[0] / self.process_scale))
            cy = int(round(pt[1] / self.process_scale))
        else:
            cx, cy = pt

        return np.array([cx, cy], dtype=np.int32), None
    
    def detect_casualty_close(self, frame, min_area_px: float = None):
        """가까이용: 면적이 충분한 가장 큰 컨투어 중심 반환 (frame 크기 기준)"""
        area_th = float(min_area_px) if min_area_px is not None else self.min_area

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # noise 제거
        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < area_th:
            return None

        M = cv2.moments(cnt)
        if M.get('m00', 0) == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    def detect_casualty_far(self, frame):
        """멀리용: 마스크된 모든 픽셀의 평균 좌표 반환 (frame 크기 기준)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return (int(xs.mean()), int(ys.mean()))   
    
    def update_param(self, lower_red1=None, upper_red1=None,
                     lower_red2=None, upper_red2=None, min_area=None,
                     process_scale=None, use_open=None,
                     open_iters=None, close_iters=None):
        if lower_red1 is not None:
            self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        if upper_red1 is not None:
            self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        if lower_red2 is not None:
            self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        if upper_red2 is not None:
            self.upper_red2 = np.array(upper_red2, dtype=np.uint8)

        if min_area is not None:
            self.min_area = float(min_area)
        if process_scale is not None:
            self.process_scale = float(np.clip(process_scale, 0.1, 1.0))
        if use_open is not None:
            self.use_open = bool(use_open)
        if open_iters is not None:
            self.open_iters = int(max(0, int(open_iters)))
        if close_iters is not None:
            self.close_iters = int(max(0, int(close_iters)))
    
    def print_param(self):
        print("Casualty Detector Parameters:")
        print(f"- Lower Red1: {self.lower_red1}")
        print(f"- Upper Red1: {self.upper_red1}")
        print(f"- Lower Red2: {self.lower_red2}")
        print(f"- Upper Red2: {self.upper_red2}")
        print(f"- Minimum Area (orig px^2): {self.min_area}")
        print(f"- Process scale: {self.process_scale}")
        print(f"- Use OPEN: {self.use_open} (iters={self.open_iters})")
        print(f"- CLOSE iters: {self.close_iters}")


# Test code for standalone usage
if __name__ == "__main__":
    import cv2
    import sys

    detector = CasualtyDetector(process_scale=0.7, use_open=True, open_iters=1, close_iters=2)  # 70% 처리
    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/test_video.mp4"  # 또는 "0"(웹캠)
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        exit()
    
    print("Press 'q' to quit, 'c' to print detection info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        detection, _ = detector.detect_casualty(frame)
        
        if detection is not None:
            cx, cy = int(detection[0]), int(detection[1])
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Red: ({cx}, {cy})", (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"Casualty detected: pixel=({cx}, {cy})")
        
        cv2.imshow("Casualty Detection Test", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.print_param()
    
    cap.release()
    cv2.destroyAllWindows()
