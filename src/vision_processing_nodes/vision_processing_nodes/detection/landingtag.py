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
import threading
import queue
import sys
# import torch
# from ultralytics import YOLO

class LandingTagDetector:
    def __init__(self, tag_size, K, D):
        
        
        self.tag_size = tag_size
        self.K = K
        self.D = D
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        self.detector_params   = self._create_params()

        # YOLO attributes
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # model_file = "best_n.torchscript"
        # model_path = os.path.join(script_dir, model_file)
        # self.model = YOLO(model_path)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OpenCV 4.7+ 에서는 ArucoDetector 클래스 사용
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            # 구버전 호환
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.detector_params
            )

        if ids is not None and len(ids) > 0:
            # 첫 번째 태그만 사용 (필요시 가장 큰 태그 선택 로직으로 변경 가능)
            img_pts = corners[0].reshape(-1, 2).astype(np.float32)
            tag_center = np.mean(img_pts, axis=0)
            return float(tag_center[0]), float(tag_center[1])

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
        # yolo_result = self.model(image) # 색깔 있는 cv 형식 이미지로 예측
                
        
        # for result in yolo_result:
        #     items = [result.names[cls.item()] for cls in result.boxes.cls.int()] # ['basket', 이런식으로 나옴]
        #     bound_coordinates = result.boxes.xyxyn # [[0,0,0,0], [1,1,1,1]] 이런식으로 tensor로 나옴
        #     confidence_level = result.boxes.conf # [0.8, 0.9, ...] 이런식으로 나옴


        # # 태그인지 확인, confidence 제일 높은거 골라서 center coordinates 찾기.
        # if len(items) > 0:
            
        #     # Choose the bounding box with highest confidence if multiple landingtags are detected
        #     for i in range(len(confidence_level)):
        #         max_confidence = 0
        #         index = 0
        #         if(confidence_level[i]>=max_confidence and items[i]=="landingtag"):
        #             max_confidence = confidence_level[i]
        #             index = i
        
        #         # Extract center coordinates
        #         x1 = bound_coordinates[index][0] * frame.shape[1]
        #         y1 = bound_coordinates[index][1] * frame.shape[0]
        #         x2 = bound_coordinates[index][2] * frame.shape[1]
        #         y2 = bound_coordinates[index][3] * frame.shape[0]
        #         cx = int((x1+x2)/2)
        #         cy = int((y1+y2)/2)
        #         return np.array([cx, cy])


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



    # def update_param(self, tag_size=None, K=None, D=None):
    #     if tag_size is not None:
    #         self.tag_size = tag_size
    #     if K is not None:
    #         self.K = K
    #     if D is not None:
    #         self.D = D

    
    # def print_param(self):
    #     print(f"LandingTagDetector Parameters:\n"
    #           f"- Tag Size: {self.tag_size}\n"
    #           f"- Camera Intrinsics (K):\n{self.K}\n"
    #           f"- Distortion Coefficients (D):\n{self.D}")
            

def worker_loop(in_q, out_q, detector):
    """큐에서 (idx, frame)을 받아 태그 검출 후 (idx, frame, detection) 반환."""
    while True:
        item = in_q.get()
        if item is None:  # 종료 신호
            in_q.task_done()
            break
        idx, frame = item
        detection = detector.detect_landing_tag(frame)
        out_q.put((idx, frame, detection))
        in_q.task_done()

if __name__ == "__main__":
    # === Define dummy calibration (replace with your real values) ===
    K = np.array([
        [1070.089695, 0.0, 1045.772015],
        [0.0, 1063.560096, 566.257075],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    D = np.array([-0.090292, 0.052332, 0.000171, 0.006618, 0.0], dtype=np.float64)
    
    detector = LandingTagDetector(tag_size=0.1, K=K, D=D)

    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "landingtag.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        exit()
        
    # ===== 멀티스레드 파이프라인 구성 =====
    import os as _os
    NUM_WORKERS = max(1, (_os.cpu_count() or 4) - 1)  # CPU 하나는 OS/표시에 남겨둠
    in_q = queue.Queue(maxsize=NUM_WORKERS * 2)
    out_q = queue.Queue()

    workers = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=worker_loop, args=(in_q, out_q, detector), daemon=True)
        t.start()
        workers.append(t)

    next_index_to_show = 0
    buffer = {}  # 순서 복원을 위한 임시 저장소: idx -> (frame, detection)
    frame_index = 0

    # ===== 메인 루프: 프레임 읽기 & 결과 표시 =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 워커 큐로
        in_q.put((frame_index, frame))
        frame_index += 1

        # 워커가 완료한 결과들을 가능한 만큼 수거
        try:
            while True:
                idx, f, detection = out_q.get_nowait()
                buffer[idx] = (f, detection)
        except queue.Empty:
            pass

        # 원래 순서대로 가능한 만큼 표시
        while next_index_to_show in buffer:
            f, detection = buffer.pop(next_index_to_show)
            if detection is not None:
                cx, cy = detection
                cv2.circle(f, (int(cx), int(cy)), 10, (0, 0, 255), 2)
            cv2.imshow("LandingTag Detection [CPU multithreaded]", f)
            next_index_to_show += 1

            # UI 반응성 확보
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                # 워커 종료 신호
                for _ in range(NUM_WORKERS):
                    in_q.put(None)
                in_q.join()
                for t in workers:
                    t.join(timeout=0.2)
                sys.exit(0)

    # ===== 모든 입력을 보냈으므로 워커 종료 처리 =====
    for _ in range(NUM_WORKERS):
        in_q.put(None)
    in_q.join()

    # 남은 결과 수거
    while not out_q.empty():
        idx, f, detection = out_q.get()
        buffer[idx] = (f, detection)

    # 남은 것들 순서대로 표시
    while next_index_to_show in buffer:
        f, detection = buffer.pop(next_index_to_show)
        if detection is not None:
            cx, cy = detection
            cv2.circle(f, (int(cx), int(cy)), 10, (0, 0, 255), 2)
        cv2.imshow("LandingTag Detection [CPU multithreaded]", f)
        next_index_to_show += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()