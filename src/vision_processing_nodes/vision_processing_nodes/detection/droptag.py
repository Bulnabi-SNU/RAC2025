# vision_processing_nodes/detection/droptag.py

"""
DropTag Detection Function
Detects red-colored drop zones using HSV color filtering with pause/resume logic
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import numpy as np
import cv2

# Import utility functions
from .utils import pixel_to_fov

class DropTagDetector:
    def __init__(self, lower_red1=None, upper_red1=None, lower_red2=None, upper_red2=None, 
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
        self.lower_red1 = lower_red1 if lower_red1 is not None else np.array([0, 100, 50])
        self.upper_red1 = upper_red1 if upper_red1 is not None else np.array([10, 255, 255])
        self.lower_red2 = lower_red2 if lower_red2 is not None else np.array([170, 100, 50])
        self.upper_red2 = upper_red2 if upper_red2 is not None else np.array([180, 255, 255])
        
        self.min_area = min_area
        self.pause_threshold = pause_threshold
        
        # State management for pause/resume functionality
        self.detection_paused = False

    def should_pause_detection(self, image):
        """
        Check if detection should be paused based on red ratio in image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            tuple: (should_pause, red_ratio, screen_center)
        """
        # Simple validation - replaced validate_image()
        if image is None or len(image.shape) != 3:
            return False, 0.0, (0, 0)
        
        # Convert to HSV and create red mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate red pixel ratio
        red_ratio = cv2.countNonZero(combined_mask) / combined_mask.size
        
        # Get screen center
        h, w = image.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Determine if detection should be paused
        should_pause = red_ratio > self.pause_threshold
        
        return should_pause, red_ratio, screen_center

    def detect_droptag(self, image):
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
            return None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check if contour meets minimum area requirement
        if area < self.min_area:
            return None
            
        # Calculate centroid using moments
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:  # Avoid division by zero
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate confidence based on area ratio
        h, w = image.shape[:2]
        total_pixels = h * w
        confidence = area / total_pixels
        
        # Estimate distance (simple inverse relationship)
        # Note: This is a basic estimation - real applications should use proper calibration
        distance = 1.0 / (confidence + 0.001)  # Prevent division by zero
        
        return {
            'center': (cx, cy),
            'confidence': confidence,
            'distance': distance,
            'area': area,
            'contour': largest_contour
        }

    def detect_droptag_with_position(self, image):
        """
        Detect drop tag and calculate relative position from screen center
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict or None: Complete detection result with positions and angles
        """
        detection = self.detect_droptag(image)
        
        if detection is None:
            return None
            
        cx, cy = detection['center']
        confidence = detection['confidence']
        distance = detection['distance']
        
        # Get image dimensions
        h, w = image.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Calculate relative position from screen center
        dx = cx - screen_center[0]
        dy = screen_center[1] - cy  # Invert y-axis
        
        # Convert pixel coordinates to real-world coordinates
        # Simple conversion using focal length approximation
        focal_length = 1000  # pixels (should be from camera calibration)
        real_x = (dx * distance) / focal_length
        real_y = (dy * distance) / focal_length
        
        # Calculate angular position in field of view
        angle_x, angle_y = pixel_to_fov(cx, cy, w, h)
        
        return {
            'pixel_center': (cx, cy),
            'screen_center': screen_center,
            'pixel_offset': (dx, dy),
            'real_position': (real_x, real_y, distance),
            'angles': (angle_x, angle_y),
            'confidence': confidence,
            'area': detection['area'],
            'image_size': (w, h)
        }

    def update_pause_state(self, should_pause):
        """
        Update the internal pause state
        
        Args:
            should_pause (bool): Whether detection should be paused
            
        Returns:
            tuple: (state_changed, new_state)
        """
        previous_state = self.detection_paused
        self.detection_paused = should_pause
        state_changed = previous_state != should_pause
        
        return state_changed, should_pause

    def is_detection_paused(self):
        """Check if detection is currently paused"""
        return self.detection_paused

    def set_color_ranges(self, lower_red1, upper_red1, lower_red2, upper_red2):
        """
        Update the HSV color ranges for red detection
        
        Args:
            lower_red1, upper_red1: First red range
            lower_red2, upper_red2: Second red range (for red wrap-around)
        """
        self.lower_red1 = lower_red1
        self.upper_red1 = upper_red1
        self.lower_red2 = lower_red2
        self.upper_red2 = upper_red2

    def set_detection_parameters(self, min_area=None, pause_threshold=None):
        """
        Update detection parameters
        
        Args:
            min_area (int): Minimum contour area
            pause_threshold (float): Red ratio threshold for pausing
        """
        if min_area is not None:
            self.min_area = min_area
        if pause_threshold is not None:
            self.pause_threshold = pause_threshold


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
        
        # Check pause condition
        should_pause, red_ratio, screen_center = detector.should_pause_detection(frame)
        state_changed, paused = detector.update_pause_state(should_pause)
        
        # Show pause state changes
        if state_changed:
            if paused:
                print(f"Detection PAUSED - Red ratio: {red_ratio:.3f}")
            else:
                print(f"Detection RESUMED - Red ratio: {red_ratio:.3f}")
        
        # Only detect if not paused
        if not paused:
            detection = detector.detect_droptag_with_position(frame)
            
            if detection:
                # Draw detection result on frame
                cx, cy = detection['pixel_center']
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                cv2.putText(frame, f"DropTag: ({cx}, {cy})", (cx + 15, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                real_x, real_y, dist = detection['real_position']
                print(f"DropTag detected: pixel=({cx}, {cy}), "
                      f"real=({real_x:.3f}, {real_y:.3f}, {dist:.3f}), "
                      f"confidence={detection['confidence']:.3f}")
        else:
            # Show screen center when paused
            cv2.circle(frame, screen_center, 15, (255, 255, 0), 3)
            cv2.putText(frame, "PAUSED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show pause info if enabled
        if show_pause_info:
            cv2.putText(frame, f"Red ratio: {red_ratio:.3f}", (10, frame.shape[0] - 10), 
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