#Detecting only red part for test flight
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class CasualtyDetector:
    def __init__(self, 
                 lower_red1=np.array([0, 100, 190], dtype=np.uint8), # 비행시험용 코드만 volume값 많이 올렸음(비행시험하고 재조정 필요할 듯)
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 100, 190], dtype=np.uint8),
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


'''
# detecting green part first
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class BasketTwoStageDetector:

    def __init__(self,
                 # --- Green (판) HSV ---
                 lower_green=np.array([35, 70, 70], dtype=np.uint8),
                 upper_green=np.array([85, 255, 255], dtype=np.uint8),

                 # --- Red (바구니 그립) HSV (양 끝 영역) ---
                 lower_red1=np.array([0, 100, 150], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 100, 150], dtype=np.uint8),
                 upper_red2=np.array([180, 255, 255], dtype=np.uint8),

                 min_area_green: float = 1500.0,   # 초록 큰 판으로 가정 → 컨투어 최소 면적
                 min_area_red: float = 500.0,      # 빨간 그립 최소 면적
                 green_ratio_threshold: float = 0.40,  # 전환 임계비율 (40%)
                 use_open: bool = True,
                 open_iters: int = 1,
                 close_iters: int = 2,
                 mask_scale: float = 0.6           # 마스크 계산 해상도 (CPU 절약)
                 ):
        # HSV 범위
        self.lower_green = lower_green
        self.upper_green = upper_green

        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

        # 파라미터
        self.min_area_green = float(min_area_green)
        self.min_area_red = float(min_area_red)
        self.green_ratio_threshold = float(green_ratio_threshold)
        self.use_open = bool(use_open)
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))
        self.mask_scale = float(np.clip(mask_scale, 0.2, 1.0))

        # 모폴로지 커널
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 상태: False → Green 모드, True → Red 모드
        self.red_mode = False

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    # ---------- Public API ----------
    def detect(self, frame):
        """
        반환:
          mode, center, info
            - mode: 'GREEN' 또는 'RED'
            - center: np.array([cx, cy], dtype=int) 또는 None
            - info: 디버깅용 dict
        """
        if not self.red_mode:
            # 1) 초록 비율 계산 (다운스케일)
            green_ratio, green_mask_small_shape = self._green_ratio(frame)

            # 전환 판단
            if green_ratio >= self.green_ratio_threshold:
                self.red_mode = True  

            # 전환 전이라면 초록 중심 반환
            if not self.red_mode:
                center = self._largest_contour_center(
                    frame, self.lower_green, self.upper_green, self.min_area_green
                )
                return 'GREEN', center, {
                    'green_ratio': green_ratio,
                    'mask_small_shape': green_mask_small_shape
                }

        # 2) RED 모드: 빨간 그립 탐지
        center = self._red_center(frame)
        return 'RED', center, {}

    def print_param(self):
        print("[Detector Parameters]")
        print(f"- Green HSV: {self.lower_green} ~ {self.upper_green}")
        print(f"- Red HSV1 : {self.lower_red1} ~ {self.upper_red1}")
        print(f"- Red HSV2 : {self.lower_red2} ~ {self.upper_red2}")
        print(f"- min_area_green: {self.min_area_green}")
        print(f"- min_area_red  : {self.min_area_red}")
        print(f"- green_ratio_threshold: {self.green_ratio_threshold}")
        print(f"- use_open: {self.use_open}, open_iters: {self.open_iters}, close_iters: {self.close_iters}")
        print(f"- mask_scale: {self.mask_scale}")

    def reset_to_green(self):
        """테스트용: 수동으로 GREEN 모드로 되돌림 (실전에서는 호출하지 않는 것을 권장)"""
        self.red_mode = False

    # ---------- Internal helpers ----------
    def _green_ratio(self, frame):
        """
        초록 픽셀 비율(0~1)을 계산 (다운스케일로 CPU 절약)
        """
        h, w = frame.shape[:2]
        sw = int(w * self.mask_scale)
        sh = int(h * self.mask_scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        green_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.size
        ratio = green_pixels / float(total_pixels) if total_pixels > 0 else 0.0
        return ratio, (sh, sw)

    def _largest_contour_center(self, frame, lower, upper, min_area):
        """
        주어진 HSV 범위에서 가장 큰 컨투어의 중심(Moment) 반환
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < float(min_area):
            return None

        M = cv2.moments(cnt)
        m00 = M.get('m00', 0.0)
        if m00 == 0.0:
            return None
        cx = int(M['m10'] / m00)
        cy = int(M['m01'] / m00)
        return np.array([cx, cy], dtype=np.int32)

    def _red_center(self, frame):
        """
        빨간(두 구간) 마스크 → 가장 큰 컨투어 중심(가까이) 시도,
        실패 시 모든 빨간 픽셀 평균 좌표(멀리) 반환
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        # 가까이: 컨투어
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) >= self.min_area_red:
                M = cv2.moments(cnt)
                if M.get('m00', 0.0) != 0.0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return np.array([cx, cy], dtype=np.int32)

        # 멀리: 모든 픽셀 평균
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.array([int(xs.mean()), int(ys.mean())], dtype=np.int32)


# ---------------------- Test code ----------------------
if __name__ == "__main__":
    import sys

    # 파라미터는 필요에 맞게 조정
    detector = BasketTwoStageDetector(
        min_area_green=2000.0,
        min_area_red=500.0,
        green_ratio_threshold=0.40,
        use_open=True, open_iters=1, close_iters=2,
        mask_scale=0.5
    )

    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/1000011262.mp4"
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        sys.exit(1)

    # 미리보기 해상도(저해상도면 처리속도↑)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Press 'q' to quit, 'p' to print parameters, 'r' to reset to GREEN (test only)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mode, center, info = detector.detect(frame)

        # 화면 오버레이
        if mode == 'GREEN':
            cv2.putText(frame, f"[MODE] GREEN  (ratio: {info.get('green_ratio', 0.0):.2f})",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            if center is not None:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame, (cx, cy), 10, (0, 220, 0), 2)
                cv2.putText(frame, f"Green Center: ({cx}, {cy})", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        else:
            cv2.putText(frame, "[MODE] RED", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if center is not None:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                cv2.putText(frame, f"Red Center: ({cx}, {cy})", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Basket Two-Stage Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.print_param()
        elif key == ord('r'):
            # 테스트용 리셋 (실전에서는 사용 금지 권장)
            detector.reset_to_green()
            print("[INFO] Manually reset to GREEN mode")

    cap.release()
    cv2.destroyAllWindows()
    '''
