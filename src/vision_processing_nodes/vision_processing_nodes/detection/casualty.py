# casualty.py

"""
Casualty Detection Function
Takes in input of phase, raw image
Outputs relative position of detected object
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import os, math
import numpy as np
import cv2


# landingtag.py

"""
LandingTAG Detection Function
Takes in input of phase, raw image
Outputs relative position of detected object
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats

# import utilities functions
from .utils import pixel_to_fov

class CasualtyDetector:
    def __init__(self, lower_orange=None, upper_orange=None, min_area=500):
        """
        Initialize Casualty Detector for orange basket detection
        
        Args:
            lower_orange (numpy.ndarray): Lower HSV threshold for orange color
            upper_orange (numpy.ndarray): Upper HSV threshold for orange color  
            min_area (int): Minimum contour area for valid detection
        """
        # Default orange color range in HSV
        self.lower_orange = lower_orange if lower_orange is not None else np.array([5, 150, 150])
        self.upper_orange = upper_orange if upper_orange is not None else np.array([20, 255, 255])
        self.min_area = min_area

    def detect_casualty(self, image):
        h, w, _ = image.shape
        # 1) 가까이용 시도
        pt = self.detect_casualty_close(image)
        mode = 'close'
        # 2) 실패 시 원거리 시도
        if pt is None:
            pt = self.detect_casualty_far(image)
            mode = 'far' if pt else 'none'
        if pt is None:
            return None
        cx, cy = pt
        # z, yaw, height는 필요에 따라 조정
        return np.array([cx, cy])
    
    def detect_casualty_close(self,frame):
        """가까이용: 면적이 충분한 가장 큰 컨투어 중심 반환"""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area:
            return None
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)

    def detect_casualty_far(self,frame):
        """멀리용: 마스크된 모든 픽셀의 평균 좌표 반환"""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return (int(xs.mean()), int(ys.mean()))   
    
    def detect_casualty_with_angles(self, image):
        """
        Detect casualty and return both pixel coordinates and angular positions
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict or None: Detection result with pixel and angular coordinates
                         Returns None if no detection
        """
        pixel_detection = self.detect_casualty(image)
        
        if pixel_detection is None:
            return None
            
        cx, cy = pixel_detection
        h, w = image.shape[:2]
        
        # Calculate angular position in field of view
        angle_x, angle_y = pixel_to_fov(cx, cy, w, h)
        
        return {
            'pixel_x': cx,
            'pixel_y': cy,
            'angle_x': angle_x,
            'angle_y': angle_y,
            'image_width': w,
            'image_height': h
        }

# Test code for standalone usage
if __name__ == "__main__":
    import cv2
    
    # Initialize detector with default parameters
    detector = CasualtyDetector()
    
    # Test with webcam or video file
    cap = cv2.VideoCapture(0)  # Use webcam
    # cap = cv2.VideoCapture("path/to/video.mp4")  # Use video file
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        exit()
    
    print("Press 'q' to quit, 'c' to print detection info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        detection = detector.detect_casualty_with_angles(frame)
        
        if detection:
            # Draw detection result on frame
            cx, cy = detection['pixel_x'], detection['pixel_y']
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Orange: ({cx}, {cy})", (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            print(f"Casualty detected: pixel=({cx}, {cy}), "
                  f"angles=({detection['angle_x']:.1f}°, {detection['angle_y']:.1f}°)")
        
        cv2.imshow("Casualty Detection Test", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and detection:
            print(f"Full detection info: {detection}")
    
    cap.release()
    cv2.destroyAllWindows()