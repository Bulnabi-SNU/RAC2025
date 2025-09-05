# landingtag.py

"""
LandingTAG Detection Function
Takes in input of phase, raw image
Outputs relative position of detected object
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import numpy as np
import cv2
import os
import torch
from ultralytics import YOLO
from packaging import version 

class LandingTagDetector:
    def __init__(self, tag_size, K, D):
        
        
        self.tag_size = tag_size
        self.K = K
        self.D = D
        # YOLO attributes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.tag_template = cv2.imread(os.path.join(script_dir, "apriltag_scaled.png"), cv2.IMREAD_GRAYSCALE)

        model_file = "best_final_n.pt"
        model_path = os.path.join(script_dir, model_file)
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda:0')  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def _create_params(self):
        if hasattr(cv2.aruco, 'DetectorParameters'):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        
        params.minMarkerPerimeterRate = 0.02

        # Performance optimizations
        params.adaptiveThreshWinSizeMin = 3   
        params.adaptiveThreshWinSizeMax = 20
        params.adaptiveThreshWinSizeStep = 4 
        
        params.maxErroneousBitsInBorderRate = 0.5

        params.polygonalApproxAccuracyRate = 0.08

        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        params.perspectiveRemovePixelPerCell = 12
        params.perspectiveRemoveIgnoredMarginPerCell = 0.2

        # Error correction
        params.errorCorrectionRate = 0.8  # Allow some error correction
        
        # Marker border
        params.markerBorderBits = 1  # Standard border width

        return params
    

    # TODO: Use SIFT at high alts, use aruco at low alts
    def detect_landing_tag(self, image):
        a,b = self.detect_landing_tag_aruco(image)
        if a is not None or True:
            return a,b
        else:
            return self.detect_landing_tag_2(image)

    def detect_landing_tag_aruco(self, image):
        if image is None:
            return None, None
        
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        parameters = self._create_params() # or cv2.aruco.DetectorParameters_create()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        corners, ids, _  = cv2.aruco.detectMarkers(sharpened, dictionary, parameters=parameters)
        if ids is not None and len(ids) > 0:
            img_pts = corners[0].reshape(-1, 2).astype(np.float32)
            tag_center = np.mean(img_pts, axis=0)
            return (tag_center[0], tag_center[1]), "Apriltag"
        return None, None

    def detect_landing_tag_2(self, image):
        """
        Detect landing tag using SIFT feature matching with FLANN
        Robust to scale, rotation, and distance variations
        """
        if image is None:
            return None, None
        
        # Initialize feature detector on first run
        if not hasattr(self, '_feature_detector') or self._feature_detector is None:
            self._feature_detector = cv2.SIFT_create(
                nfeatures=500,
                contrastThreshold=0.05,  # Higher = fewer but stronger features
                edgeThreshold=10,
                sigma=1.6
            )
        
            
            # FLANN matcher setup for SIFT (using KD-Tree)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(
                algorithm=FLANN_INDEX_KDTREE,
                trees=5  # Number of trees (5 is good balance)
            )
            search_params = dict(
                checks=50  # How many leafs to check (50 is good balance of speed/accuracy)
            )
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self._feature_type = "SIFT"
        
        # Load template features on first run
        if not hasattr(self, '_template_kp') or self._template_kp is None:
            template = self.tag_template
            
            # Store template dimensions
            self._template_h, self._template_w = template.shape[:2]
            
            # Compute template features
            self._template_kp, self._template_desc = self._feature_detector.detectAndCompute(template, None)
            
            if self._template_desc is None:
                print("Warning: No features found in template")
                return None, None
            
            # Convert to float32 for FLANN
            self._template_desc = self._template_desc.astype(np.float32)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        kp, desc = self._feature_detector.detectAndCompute(gray, None)
        
        if desc is None or len(desc) < 4:
            return None, None
        
        # Convert to float32 for FLANN
        desc = desc.astype(np.float32)
        
        # Match features using KNN
        try:
            matches = self._matcher.knnMatch(self._template_desc, desc, k=2)
        except:
            return None, None
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Ratio test - adjust threshold for sensitivity (0.7-0.8 typical)
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Need minimum number of matches for reliable detection
        min_matches = 10
        
        if len(good_matches) >= min_matches:
            # Extract matched keypoints
            src_pts = np.float32([self._template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
            
            if M is not None and mask is not None:
                # Count inliers
                inliers = sum(mask)
                # Require minimum inliers for valid detection
                if inliers >= len(good_matches) * 0.5:
                    # Transform template corners to find object in image
                    center_x = np.mean(dst_pts[mask.ravel() == 1, :, 0])
                    center_y = np.mean(dst_pts[mask.ravel() == 1, :, 1])
                    
                    # Validate detection geometry
                    return (float(center_x), float(center_y)), f"{self._feature_type}"
        
        return None, None


    def update_param(self, tag_size=None, K=None, D=None):
        if tag_size is not None:
            self.tag_size = tag_size
        if K is not None:
            self.K = K
        if D is not None:
            self.D = D

    
    def print_param(self):
        print(f"LandingTagDetector Parameters:\n"
              f"- Tag Size: {self.tag_size}\n"
              f"- Camera Intrinsics (K):\n{self.K}\n"
              f"- Distortion Coefficients (D):\n{self.D}")
            

if __name__ == "__main__":
    # === Define dummy calibration (replace with your real values) ===
    K = np.array([
        [1070.089695, 0.0, 1045.772015],
        [0.0, 1063.560096, 566.257075],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    D = np.array([-0.090292, 0.052332, 0.000171, 0.006618, 0.0], dtype=np.float64)
    
    detector = LandingTagDetector(tag_size=0.1, K=K, D=D)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "landingtag.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Clean interface: pass image, get result
        detection = detector.detect_landing_tag(frame)
        
        if detection is not None:  # Success
            cx, cy = detection[0],detection[1]
            cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), 2)

        cv2.imshow("LandingTag Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
