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

class LandingTagDetector:
    def __init__(self, tag_size, K, D):
        
        
        self.tag_size = tag_size
        self.K = K
        self.D = D
        

    def _create_params(self):
        p = (cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create())
        # 강건한 검출을 위한 파라미터
        p.adaptiveThreshWinSizeMin     = 3
        p.adaptiveThreshWinSizeMax     = 71
        p.adaptiveThreshWinSizeStep    = 3
        p.adaptiveThreshConstant       = 2
        p.minMarkerPerimeterRate       = 0.005
        p.maxMarkerPerimeterRate       = 5.0
        p.minCornerDistanceRate        = 0.005
        p.minOtsuStdDev                = 1.0
        p.maxErroneousBitsInBorderRate = 0.7
        p.detectInvertedMarker         = True
        p.cornerRefinementMethod       = (cv2.aruco.CORNER_REFINE_SUBPIX
                                        if hasattr(cv2.aruco,"CORNER_REFINE_SUBPIX") else 1)
        p.cornerRefinementWinSize      = 3
        return p

    def detect_landing_tag(self, image):
        if image is None:
            return None

        # Apriltag detection - 느리고 ellipse랑 비슷한 거리까지 측정 가능
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        detector   = cv2.aruco.ArucoDetector(dictionary, self._create_params())

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            img_pts = corners[0].reshape(-1, 2).astype(np.float32)
            tag_center = np.mean(img_pts, axis=0)
            return np.array([tag_center[0], tag_center[1]])

        # Ellipse fitting - 굉장히 빠름, 그러나 apriltag와 마찬가지로 거리 제한. 최대 거리는 apriltag와 비슷
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 70, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        ellipses = []
        for c in contours:
            if len(c) < 50:  continue
            area = cv2.contourArea(c)
            if area < 2000:   continue
            peri = cv2.arcLength(c, True)
            circ = 4*np.pi*area/(peri**2 + 1e-6)
            if circ < 0.82:    continue
            ellipses.append(cv2.fitEllipse(c))

        if len(ellipses) > 0:
            (cx, cy), (maj, minr), ang = ellipses[0]
            return np.array([cx, cy])

        # V-marker detection
        """
        To-do
        - V-marker을 인식할 수 있는 알고리즘 만들기. 이미지가 멀어지만 타원 인식도 선이 너무 얇아져서 인식이 잘 안 된다. V-marker는 멀어도 선명하게 보인다.
        - Canny 무언가를 쓰던가 yolo를 써보는 것도 괜찮을 것 같다.
        """



        # If everything fails
        return None



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

    video_path = "/home/jungwon/Downloads/landingtag.mp4"
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