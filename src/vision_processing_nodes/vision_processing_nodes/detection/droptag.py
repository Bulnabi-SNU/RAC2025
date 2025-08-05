# vision_processing_nodes/detection/droptag.py

"""
DropTag Detection Function
Detects red-colored drop zones using HSV color filtering with pause/resume logic
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import numpy as np
import cv2

class DropTagDetector:
    def __init__(self, 
                 lower_red1:np.ndarray=np.array([0, 100, 50]), 
                 upper_red1:np.ndarray=np.array([10, 255, 255]), 
                 lower_red2:np.ndarray=np.array([170, 100, 50]), 
                 upper_red2:np.ndarray=np.array([180, 255, 255]), 
                 min_area=500, pause_threshold=0.4):
        """
        Initialize DropTag Detector for red drop zone detection
        
        Args:
            lower_red1 (numpy.ndarray): Lower HSV threshold for red range 1
            upper_red1 (numpy.ndarray): Upper HSV threshold for red range 1
            lower_red2 (numpy.ndarray): Lower HSV threshold for red range 2  
            upper_red2 (numpy.ndarray): Upper HSV threshold for red range 2
            min_area (int): Minimum contour area for valid detection
            pause_threshold (float): Red ratio threshold for pausing detection (0.0-1.0)
        """
        
        # Default red color ranges in HSV (red wraps around in HSV)
        self.lower_red1 = lower_red1 
        self.upper_red1 = upper_red1 
        self.lower_red2 = lower_red2 
        self.upper_red2 = upper_red2
        
        self.min_area = min_area
        self.pause_threshold = pause_threshold
        
        # State management for pause/resume functionality
        self.detection_paused = False
    
    def detect_drop_tag(self, image):
        """
        Detect red drop tag in the image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict or None: Detection result with center, confidence, and distance
                         Returns None if no valid detection
        """
        # Simple validation - replaced validate_image()
        if image is None or len(image.shape) != 3:
            return None, None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Get ratio of red pixels. If too much, stop tracking entirely (hopefully not needed?)
        red_ratio = cv2.countNonZero(combined_mask) / combined_mask.size
        if(red_ratio > self.pause_threshold): return None
       
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return  None, None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check if contour meets minimum area requirement
        if area < self.min_area:
            return  None, None
            
        # Calculate centroid using moments
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:  # Avoid division by zero
            return  None, None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
                
        return np.array([cx,cy]), np.array([red_ratio])

    def update_param(self, lower_red1=None, upper_red1=None, 
                     lower_red2=None, upper_red2=None, 
                     min_area=None, pause_threshold=None):
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
        if pause_threshold is not None:
            self.pause_threshold = pause_threshold
    
    def print_param(self):
        print(f"DropTagDetector Parameters:\n"
              f"- Lower Red1: {self.lower_red1}\n"
              f"- Upper Red1: {self.upper_red1}\n"
              f"- Lower Red2: {self.lower_red2}\n"
              f"- Upper Red2: {self.upper_red2}\n"
              f"- Minimum Area: {self.min_area}\n"
              f"- Pause Threshold: {self.pause_threshold}")


# Test code for standalone usage
if __name__ == "__main__":
    import cv2
    
    # Initialize detector with default parameters
    detector = DropTagDetector()
    
    # Test with webcam or video file
    cap = cv2.VideoCapture(0)  # Use webcam
    # cap = cv2.VideoCapture("path/to/video.mp4")  # Use video file
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        exit()
    
    print("Press 'q' to quit, 'd' to print detection info, 'p' to toggle pause info")
    show_pause_info = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        detection, red_ratio = detector.detect_droptag_with_position(frame)
        
        if detection:
            # Draw detection result on frame
            cx, cy = detection[0],detection[1]
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
            cv2.putText(frame, f"DropTag: ({cx}, {cy})", (cx + 15, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # Show screen center when paused
            cv2.putText(frame, "PAUSED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show pause info if enabled
        if show_pause_info:
            cv2.putText(frame, f"Red ratio: {red_ratio[0]:.3f}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("DropTag Detection Test", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and 'detection' in locals() and detection:
            print(f"Full detection info: {detection}")
        elif key == ord('p'):
            show_pause_info = not show_pause_info
            print(f"Pause info display: {'ON' if show_pause_info else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()