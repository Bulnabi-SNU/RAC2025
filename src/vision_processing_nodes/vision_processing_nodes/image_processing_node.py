# image_processing_node.py

"""
Image Processor Node
Takes in input of phase, raw image
Outputs relative position of detected object
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

# import rclpy: ros library
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image                   ### 나중에 센서 이미지 받아오는것 찾아보기 ###

# import custom msg
'''msgs for subscription to phase'''
from custom_msgs.msg import VehicleState  
'''msgs for publishing positions'''
from custom_msgs.msg import TargetLocation  

# import other libraries
import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats

# define parameters for target detection (color range in hsv)
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([20, 255, 255])
min_area = 500

use_gazebo = True  # Set to True if running in Gazebo, False for live camera or video feed

class ImageProcessor(Node):

    #============================================
    # Initialize Node
    #============================================

    def __init__(self):
        super().__init__('image_processor_node')
        """
        0. Configure QoS profile for publishing and subscribing
        : communication settings with px4
        """
        qos_profile=  QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        image_qos_profile = None
        
        if use_gazebo:
            image_qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,  
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        else:
            image_qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,   # Change back to TRANSIENT_LOCAL for actual test
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

        # Lines for System Initialization
        #self.phase = None

        """ for VehiclePhase callback"""
        self.vehicle_state = VehicleState()

        """ drop tag """
        self.drop_tag_detected = False
        self.drop_tag_center = None
        self.drop_tag_confidence = 0.0
        # DropTag 일시정지 상태 관리
        self.drop_tag_paused = False

        self.bridge = CvBridge()

        if use_gazebo:
            self.topicNameFrames = "/world/RAC_2025/model/standard_vtol_0/link/camera_link/sensor/camera/image"     # Change to for real camera'topic_camera_image'
        else:
            self.topicNameFrames = "topic_camera_image"  # Change to for real camera'topic_camera_image'
        
        self.queueSize=20
        self.last_image = None

        # Camera intrinsics
        self.K = np.array([
                            [1070.089695, 0.0, 1045.772015],
                            [0.0, 1063.560096, 566.257075],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float64)
        # Camera distortion coefficients
        self.D = np.array([-0.090292, 0.052332, 0.000171, 0.006618, 0.0], dtype=np.float64)
        self.tag_size=0.5

        """
        1. Subscribers & Publishers & Timers
        """
        # [Subscribers]
        # 1. VehicleState          : current state of vehicle (ex. CASUALTY_TRACK, DESCEND_PICKUP, etc.)
        #    - detect_target_type   : 0: No detection, 1: Casualty detection, 2: Casualty dropoff detection, 3: Landing tag detection
        # 2. CameraRawImage         : Raw Image received from the Camera
        self.vehicle_state_subscriber = self.create_subscription(VehicleState, '/vehicle_state', self.state_callback, qos_profile)
        self.camera_image_subscriber = self.create_subscription(Image, self.topicNameFrames, self.image_callback, image_qos_profile)

        # [Publishers]
        # 1. TargetLocation          : publish the target location based on the type given by VehicleState
        self.target_pub = self.create_publisher(TargetLocation, '/target_position', qos_profile)

        # [Timers]
        timer_period = 0.05
        self.main_timer = self.create_timer(timer_period, self.main_timer_callback)



    #============================================
    # "Timer" Callback Functions
    #============================================     

    def main_timer_callback(self):
        if self.last_image is None:
            self.get_logger().warn("No image received yet, skipping processing")
            return
        if self.vehicle_state.detect_target_type == 0:
            return  # Skip processing if no detection is required
        elif self.vehicle_state.detect_target_type == 1:
            # Casualty 감지 및 중심으로 이동
            self.publish_casualty_location()
        elif self.vehicle_state.detect_target_type == 2:
            # DropTag 감지 및 중심으로 이동
            self.publish_casualty_dropoff_location()
        elif self.vehicle_state.detect_target_type == 3:
            # Landing Tag 감지 및 중심으로 이동
            self.publish_landing_tag_location()



    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    def state_callback(self, msg):
        self.vehicle_state = msg
        self.get_logger().debug(f"Vehicle state updated: detect_target_type = {msg.detect_target_type}") #for logging vehiclestate

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    #============================================
    # Landing Tag (AprilTag) Related Functions
    #============================================ 

    def publish_landing_tag_location(self):
        
        detection = self.detect_landing_tag(self.last_image)
        if detection[0] == -1:
            self.get_logger().warn("No apriltags detected")
            return
        if detection[0] == -2:
            self.get_logger().warn("Pose estimation failed")
            return
        x, y, z, yaw, angle_x, angle_y = detection
        pos_msg = TargetLocation()
        pos_msg.x = x                                   # x coordinate of the object's center position relative to the screen
        pos_msg.y = y                                   # y coordinate of the object's center position relative to the screen
        pos_msg.z = z                                   # distance of the object's center position relative to the camera
        pos_msg.yaw = yaw                               # angle deviation away from proper alignment 
        pos_msg.angle_x = angle_x
        pos_msg.angle_y = angle_y
        
        self.target_pub.publish(pos_msg)
        self.get_logger().info(f"Publishing landing tag location: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}, angle_x={angle_x:.2f}, angle_y={angle_y:.2f}")
        


    def detect_landing_tag(self, image): #to-do
        if image is None:
            return np.array([-1,0,0,0,0,0]) # detection fail, return -1
        
        # Opencv aruco settings for apriltag detection
        dictionary   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        det_params   = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(dictionary, det_params)

        # Gray scale conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 태그 검출
        corners, ids, rejected = detector.detectMarkers(gray_image)
        if ids is None or len(ids) == 0:
            return np.array([-1,0,0,0,0,0]) # detection fail, return -1
        
        # 일단 그냥 첫번째 태그 사용, 나중에 위에처럼 수정
        img_pts = corners[0].reshape(-1, 2).astype(np.float32)  # (4,2)
        tag_center = np.mean(img_pts, axis=0)  # (2,)

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
            return np.array([-2,0,0,0,0,0]) # pose estimate fail, return -2

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
        
        angle_x, angle_y = pixel_to_fov(
            tag_center[0], tag_center[1],
            image.shape[1], image.shape[0]
        )
        
        return np.array([x,y,z,yaw,angle_x,angle_y])

    #============================================
    # Target (Basket) Related Functions
    #============================================ 

    # ====== 바구니(오렌지색) HSV 캘리브레이션 범위 & 최소 면적 ======
    def publish_casualty_location(self):
        detection = self.detect_casualty(self.last_image)
        if detection is None:
            return
        
        pos_msg = TargetLocation()
        pos_msg.x = float('nan')
        pos_msg.y = float('nan')
        pos_msg.z = float('nan')
        pos_msg.yaw = float('nan')
        
        pos_msg.angle_x, pos_msg.angle_y = pixel_to_fov(
            detection[0], detection[1],
            self.last_image.shape[1], self.last_image.shape[0]
        )
        
        self.target_pub.publish(pos_msg)
        # self.get_logger().info(f"Publishing target location: angle_x={pos_msg.angle_x:.2f}, angle_y={pos_msg.angle_y:.2f}")
        
        
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
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
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
    

    #============================================
    # DropTag Related Functions
    #============================================ 

    def publish_casualty_dropoff_location(self):
        """DropTag를 감지하고 중심으로 이동하는 메인 함수"""
        if self.last_image is None:
            return
            
        # HSV 색상 범위 (빨간색)
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # 1) 전체 이미지 빨간 비율 계산 (0.4 기준으로 일시정지/재개)
        hsv_tmp = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2HSV)
        m1_tmp = cv2.inRange(hsv_tmp, lower_red1, upper_red1)
        m2_tmp = cv2.inRange(hsv_tmp, lower_red2, upper_red2)
        mask_tmp = cv2.bitwise_or(m1_tmp, m2_tmp)
        red_ratio = cv2.countNonZero(mask_tmp) / mask_tmp.size
        h, w = self.last_image.shape[:2]
        screen_center = (w // 2, h // 2)
        
        if red_ratio > 0.4:
            # 비율 높으면 일시정지 후 화면 중앙 좌표 출력
            if not self.drop_tag_paused:
                self.drop_tag_paused = True
                self.get_logger().info(f"DropTag detection paused. Center: {screen_center}")
            return
        else:
            # 비율 낮아지면 재개 알림
            if self.drop_tag_paused:
                self.drop_tag_paused = False
                self.get_logger().info("DropTag detection resumed.")
     

        # DropTag 감지
        detection_result = self.detect_casualty_dropoff(self.last_image)
        
        if detection_result is not None:
            tag_center, confidence, distance = detection_result
            self.drop_tag_detected = True
            self.drop_tag_center = tag_center
            self.drop_tag_confidence = confidence
            
            # 화면 중심으로부터의 상대 위치 계산
            h, w = self.last_image.shape[:2]
            screen_center = (w // 2, h // 2)
            dx = tag_center[0] - screen_center[0]
            dy = screen_center[1] - tag_center[1]  # y축은 반전
            
            # 화면 좌표를 실제 거리로 변환 (카메라 캘리브레이션 필요)
            # 간단한 변환: 픽셀 단위를 미터 단위로 변환
            focal_length = 1000  # 픽셀 단위 (실제 카메라 캘리브레이션 값 사용)
            real_x = (dx * distance) / focal_length
            real_y = (dy * distance) / focal_length
            
            # DropTagLocation 메시지로 발행 (드론 위치 기준 상대 좌표)
            pos_msg = TargetLocation()
            pos_msg.x = real_x      # 드론 기준 전방 거리 (미터)
            pos_msg.y = real_y      # 드론 기준 좌우 거리 (미터)
            pos_msg.z = distance    # 드론 기준 상하 거리 (미터)
            pos_msg.yaw = float('nan')       # 필요시 계산
            pos_msg.angle_x, pos_msg.angle_y = pixel_to_fov(
                tag_center[0], tag_center[1],
                self.last_image.shape[1], self.last_image.shape[0]
            )
            
            self.target_pub.publish(pos_msg)
            self.get_logger().info(f"DropTag detected! Real position: ({real_x:.3f}, {real_y:.3f}, {distance:.3f}), Confidence: {confidence:.2f}, Angle: ({pos_msg.angle_x:.2f}, {pos_msg.angle_y:.2f})")
            
        else:
            self.drop_tag_detected = False
            self.get_logger().info("DropTag not detected")

    def detect_casualty_dropoff(self, image):
        """
        DropTag 감지 함수 (DropTagDetection_RedHsv.py 기반)
        Returns: (tag_center, confidence, distance) or None
        """
        # HSV 색상 범위 (빨간색)
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        
        min_area = 500
        
        # HSV 변환 및 마스크 생성
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < min_area:
            return None
            
        # 중심점 계산
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # 신뢰도 계산 (면적 기반)
        h, w = image.shape[:2]
        total_pixels = h * w
        confidence = area / total_pixels
        
        # 거리 추정 (면적 기반, 실제로는 카메라 캘리브레이션 필요)
        # 간단한 추정: 면적이 클수록 가까움
        distance = 1.0 / (confidence + 0.001)  # 0으로 나누기 방지
        
        return ((cx, cy), confidence, distance)



#============================================
# Helper Functions
#============================================

def pixel_to_fov(x, y, image_width, image_height, h_fov_deg=81, d_fov_deg=93):
    # Calculate vertical FOV
    h_fov_rad = math.radians(h_fov_deg)
    d_fov_rad = math.radians(d_fov_deg)
    
    # For 16:9 aspect ratio, calculate vertical FOV
    aspect_ratio = 16/9
    v_fov_rad = 2 * math.atan(math.tan(h_fov_rad/2) / aspect_ratio)
    v_fov_deg = math.degrees(v_fov_rad)
    
    # Normalize pixel coordinates to [-1, 1]
    norm_x = (2 * x / image_width) - 1
    norm_y = (2 * y / image_height) - 1
    
    # Convert to angular coordinates (invert y since increase in y = going down)
    angle_x = norm_x * (h_fov_deg / 2)
    angle_y =- norm_y * (v_fov_deg / 2)
    
    return angle_x, angle_y

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
