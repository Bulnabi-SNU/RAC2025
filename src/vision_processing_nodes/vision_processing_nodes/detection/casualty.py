#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class CasualtyDetector:
    def __init__(self,
                 # --- Fake Green (판) HSV ---
                 #lower_green=np.array([45, 50, 40], dtype=np.uint8),
                 #upper_green=np.array([85, 255, 255], dtype=np.uint8),

                 # --- Real Green (판) HSV (optional; only used when red_only=False) ---
                 lower_green=np.array([50, 70, 140], dtype=np.uint8),
                 upper_green=np.array([65, 255, 255], dtype=np.uint8),

                 # --- Red (casualty) HSV (two ranges) ---
                 lower_red1=np.array([0, 100, 120], dtype=np.uint8),
                 upper_red1=np.array([10, 255, 255], dtype=np.uint8),
                 lower_red2=np.array([170, 100, 120], dtype=np.uint8),
                 upper_red2=np.array([179, 255, 255], dtype=np.uint8),

                 # Area thresholds
                 # If `min_area` <= 1.0 it's treated as a ratio of the frame pixels.
                 # If `min_area` > 1.0 it's treated as absolute pixels.
                 # If `min_area` is None, fall back to *_ratio values below.
                 min_area: float = None,
                 min_area_green_ratio: float = 0.0054253472,  # ≈ 5000 / 921600 @ 1280x720
                 min_area_red_ratio: float = 0.0005425347,    # ≈  500 / 921600 @ 1280x720

                 green_ratio_threshold: float = 0.25,  # switch from GREEN to RED if exceeded
                 use_open: bool = True,
                 open_iters: int = 1,
                 close_iters: int = 2,
                 mask_scale: float = 0.6,

                 # Test switch: if True, skip GREEN and detect RED only
                 red_only: bool = True):
        
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)
        self.lower_red1  = np.array(lower_red1,  dtype=np.uint8)
        self.upper_red1  = np.array(upper_red1,  dtype=np.uint8)
        self.lower_red2  = np.array(lower_red2,  dtype=np.uint8)
        self.upper_red2  = np.array(upper_red2,  dtype=np.uint8)

        # Area thresholds
        self.min_area_green_ratio = float(min_area_green_ratio)
        self.min_area_red_ratio = float(min_area_red_ratio)
        self.min_area_input = float(min_area) if min_area is not None else None

        # Other params
        self.green_ratio_threshold = float(green_ratio_threshold)
        self.use_open = bool(use_open)
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))
        self.mask_scale = float(np.clip(mask_scale, 0.2, 1.0))

        # Morphology kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Mode: False → GREEN, True → RED
        self.red_mode = False
        self.red_only = bool(red_only)
        if self.red_only:
            self.red_mode = True

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    # ======= Adapters for VisionProcessorNode (ROS2 dynamic params & API) =======
    def update_param(self, **kwargs):
        """Accepts updates from ROS2 parameter server."""
        if 'lower_red1' in kwargs and kwargs['lower_red1'] is not None:
            self.lower_red1 = np.array(kwargs['lower_red1'], dtype=np.uint8)
        if 'upper_red1' in kwargs and kwargs['upper_red1'] is not None:
            arr = np.array(kwargs['upper_red1'], dtype=np.uint8)
            # Hue clamp to 0..179 (OpenCV HSV)
            arr[0] = np.uint8(min(int(arr[0]), 179))
            self.upper_red1 = arr
        if 'lower_red2' in kwargs and kwargs['lower_red2'] is not None:
            self.lower_red2 = np.array(kwargs['lower_red2'], dtype=np.uint8)
        if 'upper_red2' in kwargs and kwargs['upper_red2'] is not None:
            arr = np.array(kwargs['upper_red2'], dtype=np.uint8)
            arr[0] = np.uint8(min(int(arr[0]), 179))
            self.upper_red2 = arr

        if 'min_area' in kwargs and kwargs['min_area'] is not None:
            self.min_area_input = float(kwargs['min_area'])

        if 'green_ratio_threshold' in kwargs and kwargs['green_ratio_threshold'] is not None:
            self.green_ratio_threshold = float(kwargs['green_ratio_threshold'])
        if 'use_open' in kwargs and kwargs['use_open'] is not None:
            self.use_open = bool(kwargs['use_open'])
        if 'open_iters' in kwargs and kwargs['open_iters'] is not None:
            self.open_iters = int(max(0, kwargs['open_iters']))
        if 'close_iters' in kwargs and kwargs['close_iters'] is not None:
            self.close_iters = int(max(0, kwargs['close_iters']))

    def detect_casualty(self, frame):
        """
        VisionProcessorNode expects: (center, extra)
        Here, extra is unused (None).
        """
        # ===== two-stage 스위칭 사용 =====
        center = self.detect(frame)
        return center, None
    # ===========================================================================

    def detect(self, frame):
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Lenient color ranges for various green shades and white/light surfaces
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 40, 255])
        
        # Create initial color masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological kernel for cleaning operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Clean up masks
        green_mask_clean = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        white_mask_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask_clean = cv2.morphologyEx(white_mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Find all green contours
        green_contours, _ = cv2.findContours(green_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not green_contours:
            print("No green areas found")
            return None
        
        # Sort contours by area (largest first) and select top 10
        green_contours_sorted = sorted(green_contours, key=cv2.contourArea, reverse=True)
        top_green_contours = green_contours_sorted[:min(10, len(green_contours_sorted))]
        
        print(f"Processing {len(top_green_contours)} green areas")
        
        # Process each green contour individually
        valid_white_areas = []
        min_white_area = 200  # Minimum white area threshold (adjust as needed)
        
        for i, contour in enumerate(top_green_contours):
            green_area = cv2.contourArea(contour)
            
            # Filter out very small green contours
            if green_area < 100:
                continue
                
            # Create mask for this specific green contour (with holes filled)
            single_green_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(single_green_mask, [contour], 255)
            
            # Find white areas within this green contour
            white_in_this_green = cv2.bitwise_and(white_mask_clean, single_green_mask)
            
            # Calculate white area size
            white_area = np.sum(white_in_this_green) / 255  # Convert to pixel count
            
            print(f"Green contour {i+1}: Green area = {green_area:.0f}, White area = {white_area:.0f}")
            
            # Check if white area is large enough
            if white_area >= min_white_area:
                print(f"✓ Green contour {i+1} has sufficient white area")
                
                # Calculate centroid for this valid white area
                moments = cv2.moments(white_in_this_green)
                if moments["m00"] != 0:
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])
                    
                    # Store valid white area info
                    valid_white_areas.append({
                        'centroid': (centroid_x, centroid_y),
                        'white_area': white_area,
                        'green_area': green_area,
                        'mask': white_in_this_green
                    })
            else:
                print(f"✗ Green contour {i+1} white area too small ({white_area:.0f} < {min_white_area})")
        
        # Return result based on valid areas found
        if not valid_white_areas:
            print("No green areas contain sufficient white regions")
            return None
        
        # Option 1: Return centroid of the largest valid white area
        largest_white = max(valid_white_areas, key=lambda x: x['white_area'])
        print(f"Returning centroid of largest valid white area: {largest_white['centroid']}")
        return largest_white['centroid']

    def print_param(self):
        print("[CasualtyDetector Parameters]")
        print(f"- red_only: {self.red_only}")
        print(f"- Green HSV: {self.lower_green} ~ {self.upper_green}")
        print(f"- Red HSV1 : {self.lower_red1} ~ {self.upper_red1}")
        print(f"- Red HSV2 : {self.lower_red2} ~ {self.upper_red2}")
        print(f"- min_area_green_ratio: {self.min_area_green_ratio:.8f} (GREEN area ratio)")
        if self.min_area_input is None:
            print(f"- min_area_red_ratio  : {self.min_area_red_ratio:.8f} (RED area ratio)")
        else:
            print(f"- min_area_red_input  : {self.min_area_input:.8f} (<=1: ratio, >1: pixels)")
        print(f"- green_ratio_threshold: {self.green_ratio_threshold}")
        print(f"- use_open: {self.use_open}, open_iters: {self.open_iters}, close_iters: {self.close_iters}")
        print(f"- mask_scale: {self.mask_scale}")

    # ---------- Internal helpers ----------
    def _green_ratio(self, frame):
        """Compute green pixel ratio (downscaled for speed)."""
        h, w = frame.shape[:2]
        sw = max(1, int(w * self.mask_scale))
        sh = max(1, int(h * self.mask_scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        if self.use_open and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        green_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size) if mask.size > 0 else 1
        ratio = green_pixels / float(total_pixels)
        return ratio, (sh, sw)

    def _largest_contour_center_ratio(self, frame, lower, upper, min_area_ratio: float):
        """Return centroid of the largest contour if area exceeds ratio threshold."""
        h, w = frame.shape[:2]
        total_pixels = float(h * w)
        min_area = float(min_area_ratio) * total_pixels

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
        if m00 <= 1e-6:
            return None
        cx = int(M['m10'] / m00)
        cy = int(M['m01'] / m00)
        return np.array([cx, cy], dtype=np.int32)

    def _red_center_ratio(self, frame):
        """
        Red detection: largest contour centroid if area exceeds threshold.
        Threshold uses:
          - min_area_input (<=1: ratio, >1: absolute pixels), if provided;
          - otherwise min_area_red_ratio (ratio).
        """
        h, w = frame.shape[:2]
        total_pixels = float(h * w)

        if self.min_area_input is None:
            min_area = float(self.min_area_red_ratio) * total_pixels
        else:
            if self.min_area_input <= 1.0:
                min_area = float(self.min_area_input) * total_pixels
            else:
                min_area = float(self.min_area_input)

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
        if cv2.contourArea(cnt) < float(min_area):
            return None

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

    # Red-only quick test by default. Set red_only=False to test GREEN→RED switching.
    detector = CasualtyDetector(
        min_area=None,                      # if None → uses min_area_red_ratio
        min_area_green_ratio=0.0054253472,  # ~5000 / 921600
        min_area_red_ratio=0.0005425347,    # ~ 500 / 921600
        green_ratio_threshold=0.25,
        use_open=True, open_iters=1, close_iters=2,
        mask_scale=0.5,
        red_only=True
    )

    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/src/vision_processing_nodes/vision_processing_nodes/detection/video.mp4"
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        sys.exit(1)

    # Preview resolution (actual detection uses full frames read by cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Press 'q' to quit, 'p' to print parameters")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For testing UI: use the two-stage `detect` method to show mode/ratio on-screen.
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Casualty RED-Only / Two-Stage Detection (Test)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.print_param()

    cap.release()
    cv2.destroyAllWindows()
