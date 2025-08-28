#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class CasualtyDetector:
    def __init__(self, 
                 lower_red1=np.array([0, 50, 100], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 50, 100], dtype=np.uint8),
                 upper_red2=np.array([180, 255, 255], dtype=np.uint8),
                 min_area: float = 500.0,
                 use_open: bool = True,
                 open_iters: int = 1,
                 close_iters: int = 2):
        """
        Args:
            lower_red1/upper_red1/lower_red2/upper_red2: HSV thresholds
            min_area: Minimum contour area 
            use_open: 작은 점 노이즈 제거용 OPEN 사용 여부
            open_iters: OPEN 반복 횟수
            close_iters: CLOSE 반복 횟수
        """
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

        self.min_area = float(min_area)

        self.use_open = bool(use_open)
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))

        # 커널 캐싱
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    def detect_casualty(self, frame):
        # 1) 가까이용 (컨투어)
        pt = self.detect_casualty_close(frame, min_area_px=self.min_area)
        # 2) 실패 시 멀리용 (픽셀 평균)
        if pt is None:
            pt = self.detect_casualty_far(frame)
        if pt is None:
            return None, None

        cx, cy = pt
        return np.array([cx, cy], dtype=np.int32), None
    
    def detect_casualty_close(self, frame, min_area_px: float = None):
        """가까이용: 면적이 충분한 가장 큰 컨투어 중심 반환"""
        area_th = float(min_area_px) if min_area_px is not None else self.min_area

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거
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
        """멀리용: 마스크된 모든 픽셀의 평균 좌표 반환"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return (int(xs.mean()), int(ys.mean()))   
    
    def print_param(self):
        print("Casualty Detector Parameters:")
        print(f"- Lower Red1: {self.lower_red1}")
        print(f"- Upper Red1: {self.upper_red1}")
        print(f"- Lower Red2: {self.lower_red2}")
        print(f"- Upper Red2: {self.upper_red2}")
        print(f"- Minimum Area: {self.min_area}")
        print(f"- Use OPEN: {self.use_open} (iters={self.open_iters})")
        print(f"- CLOSE iters: {self.close_iters}")


# Test code
if __name__ == "__main__":
    import sys

    detector = CasualtyDetector(use_open=True, open_iters=1, close_iters=2)
    src = sys.argv[1] if len(sys.argv) > 1 else "path.mp4"
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        exit()
    
    # 여기서 해상도 낮춰서 캡처하기 (예: 320x240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
        
        cv2.imshow("Casualty Detection Test", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.print_param()
    
    cap.release()
    cv2.destroyAllWindows()
