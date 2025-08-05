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

class LandingTagDetector:
    def __init__(self, tag_size, K, D):

        self.tag_size = tag_size
        self.K = K
        self.D = D

    def detect_landing_tag(self, image): #to-do
        if image is None:
            return None, None
        
        # Opencv aruco settings for apriltag detection
        dictionary   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        det_params   = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(dictionary, det_params)

        # Gray scale conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 태그 검출
        corners, ids, rejected = detector.detectMarkers(gray_image)
        if ids is None or len(ids) == 0:
            return None, None
        
        # 일단 그냥 첫번째 태그 사용, 나중에 위에처럼 수정
        img_pts = corners[0].reshape(-1, 2).astype(np.float32)  # (4,2)
        tag_center = np.mean(img_pts, axis=0)  # (2,)

        # =========== This is the end of basic image detection, the latter is not 100% necessary =======
        
        # solvePnP용 3D 좌표계 정의 (april_tag 좌표계 만드는 코드)
        s = self.tag_size / 2.0
        obj_pts = np.array([[-s,  s, 0],
                            [ s,  s, 0],
                            [ s, -s, 0],
                            [-s, -s, 0]], dtype=np.float32)
        
        """
        Solve PnP - 여기서 camera intrinsics and distortion 고려
        success - 성공 여부에 대한 Boolean
        rvec - rotation vector (단위)
        tvec - transformation vector
        """
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                        self.K, self.D,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return np.array([tag_center[0],tag_center[1]]), None # pose estimate fail, don't return additional info

        # Compute Rotation matrix and its inverse for camera position calculation
        R, _ = cv2.Rodrigues(rvec)
        T = tvec                        # T vector renamed for 통일

        # Calculate camera position - https://darkpgmr.tistory.com/122 math behind these calculations, https://m.blog.naver.com/jewdsa813/222210464716 for code
        tag_pose_camera_frame = (-np.linalg.inv(R)@T).flatten()
        
        # Calculate yaw
        unit_z_tag_frame = np.array([0.,0.,1.])
        unit_z_camera_frame = np.linalg.inv(R)@unit_z_tag_frame         # No need to consider transformation in this case (only consider rotation)
        yaw = np.arctan2(unit_z_camera_frame[1],unit_z_camera_frame[0]) - np.pi/2

        # Return values
        x = tag_pose_camera_frame[0]
        y = tag_pose_camera_frame[1]
        z = tag_pose_camera_frame[2]
        
        return np.array([tag_center[0],tag_center[1]]), np.array([x,y,z,yaw])

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

    video_path = "path/to/your/video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Clean interface: pass image, get result
        detection, additional = detector.detect_landing_tag(frame)
        
        if detection != None:  # Success
            cx, cy = detection[0],detection[1]
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
            
            if additional != None:
                x, y, z, yaw = additional
                print(f"Landing tag detected: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        cv2.imshow("LandingTag Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()