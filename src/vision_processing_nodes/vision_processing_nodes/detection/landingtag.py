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

class LandingTagDetector:
    def __init__(self, tag_size, K, D):
        
        
        self.tag_size = tag_size
        self.K = K
        self.D = D

        # YOLO attributes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = "best_n.torchscript"
        model_path = os.path.join(script_dir, model_file)
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

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
        """
        Apriltag detection - 느리고 ellipse랑 비슷한 거리까지 측정 가능
        """
        
        # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        # detector   = cv2.aruco.ArucoDetector(dictionary, self._create_params())

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # corners, ids, _ = detector.detectMarkers(gray)

        # if ids is not None and len(ids) > 0:
        #     img_pts = corners[0].reshape(-1, 2).astype(np.float32)
        #     tag_center = np.mean(img_pts, axis=0)
        #     return np.array([tag_center[0], tag_center[1]])

        """
        Ellipse fitting - 굉장히 빠름, 그러나 apriltag와 마찬가지로 거리 제한. 최대 거리는 apriltag와 비슷
        """
        # blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # edges = cv2.Canny(blur, 70, 180)
        # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # ellipses = []
        # for c in contours:
        #     if len(c) < 50:  continue
        #     area = cv2.contourArea(c)
        #     if area < 2000:   continue
        #     peri = cv2.arcLength(c, True)
        #     circ = 4*np.pi*area/(peri**2 + 1e-6)
        #     if circ < 0.82:    continue
        #     ellipses.append(cv2.fitEllipse(c))

        # if len(ellipses) > 0:
        #     (cx, cy), (maj, minr), ang = ellipses[0]
        #     return np.array([cx, cy])

        """
        Yolo detection
        """
        yolo_result = self.model(image) # 색깔 있는 cv 형식 이미지로 예측
                
        
        for result in yolo_result:
            items = [result.names[cls.item()] for cls in result.boxes.cls.int()] # ['basket', 이런식으로 나옴]
            bound_coordinates = result.boxes.xyxyn # [[0,0,0,0], [1,1,1,1]] 이런식으로 tensor로 나옴
            confidence_level = result.boxes.conf # [0.8, 0.9, ...] 이런식으로 나옴


        # 태그인지 확인, confidence 제일 높은거 골라서 center coordinates 찾기.
        if len(items) > 0:
            
            # Choose the bounding box with highest confidence if multiple landingtags are detected
            for i in range(len(confidence_level)):
                max_confidence = 0
                index = 0
                if(confidence_level[i]>=max_confidence and items[i]=="landingtag"):
                    max_confidence = confidence_level[i]
                    index = i
        
                # Extract center coordinates
                x1 = bound_coordinates[index][0] * frame.shape[1]
                y1 = bound_coordinates[index][1] * frame.shape[0]
                x2 = bound_coordinates[index][2] * frame.shape[1]
                y2 = bound_coordinates[index][3] * frame.shape[0]
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                return np.array([cx, cy])


        """
        Feature matching (simple white/black mask)
        """
        # # Reduce resolution for speed
        # scale = 0.2
        # img_small = cv2.resize(image, (0,0), fx=scale, fy=scale)
        # gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        # # --- Keep only black and white regions ---
        # _, mask_black = cv2.threshold(gray_small, 50, 255, cv2.THRESH_BINARY_INV)
        # _, mask_white = cv2.threshold(gray_small, 200, 255, cv2.THRESH_BINARY)
        # mask_bw = cv2.bitwise_or(mask_black, mask_white)
        # gray_small = cv2.bitwise_and(gray_small, gray_small, mask=mask_bw)

        # # ORB keypoint detection
        # orb = cv2.ORB_create(500)
        # kp1, des1 = orb.detectAndCompute(gray_small, None)

        # # Load reference tag image (grayscale)
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # ref_path = os.path.join(script_dir, "apriltag.png")
        # ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        # if ref_img is None:
        #     print("[ERROR] Reference image not found:", ref_path)
        #     return None

        # ref_small = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
        # kp2, des2 = orb.detectAndCompute(ref_small, None)

        # # BFMatcher + Homography + RANSAC
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(des2, des1)
        # if len(matches) < 4:
        #     return None
        # matches = sorted(matches, key=lambda x:x.distance)
        # src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        # dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        # mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # if mtrx is None:
        #     return None

        # h, w = ref_small.shape[:2]
        # pts = np.float32([[[0,0]], [[0,h-1]], [[w-1,h-1]], [[w-1,0]]])
        # dst = cv2.perspectiveTransform(pts, mtrx)

        # # Rescale coordinates back to original image
        # dst = dst / scale
        # center = dst.mean(axis=0).ravel()

        # # Draw detected box
        # image = cv2.polylines(image, [np.int32(dst)], True, (0,255,0), 2, cv2.LINE_AA)

        # return np.array([center[0], center[1]])




                




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