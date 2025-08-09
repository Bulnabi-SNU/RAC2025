# casualty.py

"""
Casualty Detection Function
Takes an image and outputs pixel position of detected object's center
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import numpy as np
import cv2

class CasualtyDetector:
    def __init__(self, 
                 lower_red1:np.ndarray=np.array([0, 150, 150]),
                 upper_red1:np.ndarray=np.array([10, 255, 255]),
                 lower_red2:np.ndarray=np.array([170, 150, 150]),
                 upper_red2:np.ndarray=np.array([180, 255, 255]),
                 min_area:float=500.0):
        """
        Initialize Casualty Detector for red basket detection
        
        Args:
            lower_red1 (numpy.ndarray): Lower HSV threshold for red color (0-10)
            upper_red1 (numpy.ndarray): Upper HSV threshold for red color (0-10)
            lower_red2 (numpy.ndarray): Lower HSV threshold for red color (170-180)
            upper_red2 (numpy.ndarray): Upper HSV threshold for red color (170-180)
            min_area (int): Minimum contour area for valid detection
        """
        # Default red color range in HSVs
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2
        self.min_area = min_area

    def detect_casualty(self, image):
        h, w, _ = image.shape
        # 1) 가까이용 시도
        pt = self.detect_casualty_close(image)
        # 2) 실패 시 원거리 시도
        if pt is None:
            pt = self.detect_casualty_far(image)
        if pt is None:
            return None, None
        cx, cy = pt
        # z, yaw, height는 필요에 따라 조정
        return np.array([cx, cy]), None
    
    def detect_casualty_close(self, frame):
        """가까이용: 면적이 충분한 가장 큰 컨투어 중심 반환"""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
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

    def detect_casualty_far(self, frame):
        """멀리용: 마스크된 모든 픽셀의 평균 좌표 반환"""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return (int(xs.mean()), int(ys.mean()))   
    
    def update_param(self, lower_red1=None, upper_red1=None,
                     lower_red2=None, upper_red2=None, min_area=None):
        if lower_red1 is not None:
            self.lower_red1 = lower_red1
        if upper_red1 is not None:
            self.upper_red1 = upper_red1
        if lower_red2 is not None:
            self.lower_red2 = lower_red2
        if upper_red2 is not None:
            self.upper_red2 = upper_red2
        if min_area is not None:
            self.min_area = min_area
    
    def print_param(self):
        print("Casualty Detector Parameters:")
        print(f"- Lower Red1: {self.lower_red1}")
        print(f"- Upper Red1: {self.upper_red1}")
        print(f"- Lower Red2: {self.lower_red2}")
        print(f"- Upper Red2: {self.upper_red2}")
        print(f"- Minimum Area: {self.min_area}")


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
        detection = detector.detect_casualty(frame)
        
        if detection and detection[0] is not None:
            # Draw detection result on frame
            cx, cy = detection[0]
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Red: ({cx}, {cy})", (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            print(f"Casualty detected: pixel=({cx}, {cy})")
        
        cv2.imshow("Casualty Detection Test", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and detection:
            print(f"Full detection info: {detection}")
    
    cap.release()
    cv2.destroyAllWindows()
