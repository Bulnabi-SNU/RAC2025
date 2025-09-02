#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class CasualtyDetector:

    def __init__(self,
                 # --- Green (판) HSV ---
                 #lower_green=np.array([45, 50, 40], dtype=np.uint8),
                 #upper_green=np.array([85, 255, 255], dtype=np.uint8),

                 # real range for "yawn green" (지금은 red-only 테스트라 미사용 가능)
                 lower_green=np.array([50, 70, 140], dtype=np.uint8),
                 upper_green=np.array([65, 255, 255], dtype=np.uint8),

                 # --- Red (바구니 그립) HSV (양 끝 영역) ---
                 lower_red1=np.array([0, 100, 120], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 100, 120], dtype=np.uint8),
                 upper_red2=np.array([180, 255, 255], dtype=np.uint8),

                 # 1280x720 기준 (총 921,600px):
                 #   green 5000px  ≈ 0.0054253472
                 #   red    500px  ≈ 0.0005425347
                 min_area_green_ratio: float = 0.0054253472,
                 min_area_red_ratio: float = 0.0005425347,

                 green_ratio_threshold: float = 0.40,  # (red_only=False일 때 사용)
                 use_open: bool = True,
                 open_iters: int = 1,
                 close_iters: int = 2,
                 mask_scale: float = 0.6,          # 마스크 계산 해상도 (CPU 절약)

                 # ------ (참고) 기존 GREEN 게이트 파라미터들: 더 이상 사용하지 않음 ------
                 min_solidity: float = 0.60,       # (미사용)
                 min_bbox_fill: float = 0.40,      # (미사용)
                 ar_min: float = 0.30,             # (미사용)
                 ar_max: float = 3.50,             # (미사용)
                 min_mean_s: int = 60,             # (미사용)

                 # ------ 추가: 빨간 전용 테스트 스위치 ------
                 red_only: bool = True
                 ):
        # HSV 범위
        self.lower_green = lower_green
        self.upper_green = upper_green
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

        # 비율 파라미터
        self.min_area_green_ratio = float(min_area_green_ratio)
        self.min_area_red_ratio = float(min_area_red_ratio)

        # 기타 파라미터
        self.green_ratio_threshold = float(green_ratio_threshold)
        self.use_open = bool(use_open)
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))
        self.mask_scale = float(np.clip(mask_scale, 0.2, 1.0))

        # 모폴로지 커널
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 상태: False → Green 모드, True → Red 모드
        self.red_mode = False

        # (호환용) 미사용 게이트 파라미터 보관만
        self.min_solidity = float(min_solidity)
        self.min_bbox_fill = float(min_bbox_fill)
        self.ar_min = float(ar_min)
        self.ar_max = float(ar_max)
        self.min_mean_s = int(min_mean_s)

        # RED 전용 테스트 스위치
        self.red_only = bool(red_only)
        if self.red_only:
            # 시작부터 RED 모드로
            self.red_mode = True

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
        # 빨간 전용 모드면, 곧장 RED 탐지 (레드에는 게이트 적용 안 함)
        if self.red_only:
            center = self._red_center_ratio(frame)
            return 'RED', center, {}

        # (일반) 초록→빨강 2단계
        if not self.red_mode:
            # 1) 초록 비율 계산 (다운스케일)
            green_ratio, green_mask_small_shape = self._green_ratio(frame)

            # 전환 판단
            if green_ratio >= self.green_ratio_threshold:
                self.red_mode = True

            # 전환 전이라면 초록 중심 반환
            #  ※ 요청대로 GREEN에는 면적 비율만 사용(다른 모든 게이트 제거)
            if not self.red_mode:
                center = self._largest_contour_center_ratio(
                    frame, self.lower_green, self.upper_green, self.min_area_green_ratio
                )
                return 'GREEN', center, {
                    'green_ratio': green_ratio,
                    'mask_small_shape': green_mask_small_shape
                }

        # 2) RED 모드: 빨간 그립 탐지 (※ 레드에는 FP 게이트 미적용)
        center = self._red_center_ratio(frame)
        return 'RED', center, {}

    def print_param(self):
        print("[Detector Parameters]")
        print(f"- red_only: {self.red_only}")
        print(f"- Green HSV: {self.lower_green} ~ {self.upper_green}")
        print(f"- Red HSV1 : {self.lower_red1} ~ {self.upper_red1}")
        print(f"- Red HSV2 : {self.lower_red2} ~ {self.upper_red2}")
        print(f"- min_area_green_ratio: {self.min_area_green_ratio:.8f} (GREEN: area-only)")
        print(f"- min_area_red_ratio  : {self.min_area_red_ratio:.8f} (RED: area-only)")
        print(f"- green_ratio_threshold: {self.green_ratio_threshold}")
        print(f"- use_open: {self.use_open}, open_iters: {self.open_iters}, close_iters: {self.close_iters}")
        print(f"- mask_scale: {self.mask_scale}")
        print(f"- [GREEN gates] disabled (solidity/bbox_fill/AR/meanS not used)")
        print(f"- [RED gates] disabled (area ratio only)")

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

    def _largest_contour_center_ratio(self, frame, lower, upper, min_area_ratio: float):
        """
        (GREEN) 주어진 HSV 범위에서 가장 큰 컨투어 중심 반환
        - 임계값: 전체 픽셀 대비 면적 비율 기준만 사용 (게이트 전부 제거)
        """
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = float(min_area_ratio) * float(total_pixels)

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
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            return None

        M = cv2.moments(cnt)
        m00 = M.get('m00', 0.0)
        if m00 <= 1e-6:   # 0 근처 값 방지
            return None
        cx = int(M['m10'] / m00)
        cy = int(M['m01'] / m00)
        return np.array([cx, cy], dtype=np.int32)

    def _red_center_ratio(self, frame):
        """
        (RED) 빨간(두 구간) 마스크 → 가장 큰 컨투어 중심만 사용
        - FP 게이트는 적용하지 않음
        - 최소 면적 비율(min_area_red_ratio)만 적용
        """
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = float(self.min_area_red_ratio) * float(total_pixels)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            return None

        # --- 게이트 없음: 바로 중심 계산 ---
        M = cv2.moments(cnt)
        m00 = M.get('m00', 0.0)
        if m00 <= 1e-6:
            return None
        cx = int(M['m10'] / m00)
        cy = int(M['m01'] / m00)
        return np.array([cx, cy], dtype=np.int32)


# ---------------------- Test code ----------------------
if __name__ == "__main__":
    import sys

    # 예: RED-ONLY 테스트 시 red_only=True, 2단계 전체 동작 시 red_only=False
    detector = CasualtyDetector(
        min_area_green_ratio=0.0054253472,   # (red_only=True면 미사용)
        min_area_red_ratio=0.0005425347,     # ≈  500 / 921600
        green_ratio_threshold=0.25,          # (red_only=True면 미사용)
        use_open=True, open_iters=1, close_iters=2,
        mask_scale=0.5,
        # (참고) GREEN 게이트 파라미터는 더 이상 사용하지 않음
        min_solidity=0.65,
        min_bbox_fill=0.45,
        ar_min=0.35, ar_max=3.20,
        min_mean_s=65,
        # 스위치
        red_only=True
    )

    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/src/vision_processing_nodes/vision_processing_nodes/detection/path.mp4"
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        sys.exit(1)

    # 미리보기 해상도- 실제 컨투어/임계값은 원해상도 프레임 기준
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Press 'q' to quit, 'p' to print parameters")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mode, center, info = detector.detect(frame)

        title = "[MODE] RED-ONLY" if detector.red_only else ("[MODE] " + mode)
        color = (0, 0, 255) if "RED" in title else (0, 220, 0)
        cv2.putText(frame, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not detector.red_only and mode == 'GREEN':
            cv2.putText(frame, f"Green ratio: {info.get('green_ratio', 0.0):.2f}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.circle(frame, (cx, cy), 10, color, 2)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Center: None", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("Basket RED-Only / Two-Stage Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.print_param()

    cap.release()
    cv2.destroyAllWindows()
