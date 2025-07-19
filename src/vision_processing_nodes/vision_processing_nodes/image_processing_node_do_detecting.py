# image_processor_node.py

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
from custom_msgs.msg import VehiclePhase  
'''msgs for publishing positions'''
from custom_msgs.msg import LandingTagLocation  
from custom_msgs.msg import TargetLocation  
from custom_msgs.msg import DropTagLocation  # HagiTag 위치 정보용

# import other libraries
import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats


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
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Lines for System Initialization
        #self.phase = None

        """ for VehiclePhase callback"""
        self.vehicle_state = None
        self.vehicle_phase = None
        self.vehicle_subphase = None
        self.do_landing = False
        self.do_detecting = True
        self.do_drop = False

        """ drop tag """
        self.hagi_tag_detected = False
        self.hagi_tag_center = None
        self.hagi_tag_confidence = 0.0

        self.bridge = CvBridge()
        self.topicNameFrames ='topic_camera_image'
        self.queueSize=20
        self.last_image = None

        """
        1. Subscribers & Publishers & Timers
        """
        # [Subscribers]
        # 1. VehiclePhase           : current phase of vehicle (ex. do_landing, do_detecting)
        # 2. CameraRawImage         : Raw Image received from the Camera
        self.vehicle_phase_subscriber = self.create_subscription(VehiclePhase, '/vehicle_phase', self.phase_callback, qos_profile)
        self.camera_image_subscriber = self.create_subscription(Image, self.topicNameFrames, self.image_callback, qos_profile)

        # [Publishers]
        # 1. LandingTagLocation      : publish the landing tag location
        # 2. TargetLocation          : publish the target location
        # 3. DropTagLocation         : publish the HagiTag location (드론 위치 기준 상대 좌표)
        self.landing_pub = self.create_publisher(LandingTagLocation, '/landing_tag_position', qos_profile)
        self.target_pub = self.create_publisher(TargetLocation, '/target_position', qos_profile)
        self.drop_tag_pub = self.create_publisher(DropTagLocation, '/drop_tag_position', qos_profile)

        # [Timers]
        timer_period = 0.05
        self.main_timer = self.create_timer(timer_period, self.main_timer_callback)



    #============================================
    # "Timer" Callback Functions
    #============================================     

    def main_timer_callback(self):
        if self.do_landing:
            self.publish_landing_tag_location()

        if self.do_detecting:
            self.publish_target_location()
            
        if self.do_drop:
            # HagiTag 감지 및 중심으로 이동
            self.detect_and_track_hagi_tag()


    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    def phase_callback(self, msg):
        self.phase = msg
        self.vehicle_state = msg.vehicle_state
        self.vehicle_phase = msg.vehicle_phase
        self.vehicle_subphase = msg.vehicle_subphase
        self.do_landing = msg.do_landing
        self.do_detecting = msg.do_detecting
        self.do_drop = msg.do_drop

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


    #============================================
    # "Publisher" Callback Functions
    #============================================ 

    def publish_landing_tag_location(self):
        if self.phase is None:
            return  # Skip processing if phase is not defined
        detection = self.detect_landing_tag(self.last_image)
        if detection[0] == -1:
            print("No apriltags detected")
            return
        if detection[0] == -2:
            print("Pose estimation failed")
            return
        x, y, z, yaw, height = detection
        pos_msg = LandingTagLocation()
        pos_msg.x = x                                   # x coordinate of the object's center position relative to the screen
        pos_msg.y = y                                   # y coordinate of the object's center position relative to the screen
        pos_msg.z = z                                   # distance of the object's center position relative to the camera
        pos_msg.yaw = yaw                               # angle deviation away from proper alignment 
        pos_msg.height = height                         # height of the vehicle if probably not necessary here
        self.landing_pub.publish(pos_msg)
        self.get_logger().info(f"Publishing landing tag location: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f},height={height:.2f}")

    def publish_target_location(self):
        if self.phase is None:
            return  # Skip processing if phase is not defined
        detection = self.detect_target(self.last_image)
        if not detection:
            return
        x, y, z, yaw, height = detection
        pos_msg = TargetLocation()
        pos_msg.x = x                                   # x coordinate of the object's center position relative to the screen
        pos_msg.y = y                                   # y coordinate of the object's center position relative to the screen
        pos_msg.z = z                                   # distance of the object's center position relative to the camera
        pos_msg.yaw = yaw                               # angle deviation away from proper alignment 
        pos_msg.height = height                         # height of the vehicle if probably not necessary here
        self.target_pub.publish(pos_msg)
        self.get_logger().info(f"Publishing target location: x={x:.2f}, y={y:.2f}")

    def detect_and_track_hagi_tag(self):
        """HagiTag를 감지하고 중심으로 이동하는 메인 함수"""
        if self.last_image is None:
            return
            
        # HagiTag 감지
        detection_result = self.detect_hagi_tag(self.last_image)
        
        if detection_result is not None:
            tag_center, confidence, distance = detection_result
            self.hagi_tag_detected = True
            self.hagi_tag_center = tag_center
            self.hagi_tag_confidence = confidence
            
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
            pos_msg = DropTagLocation()
            pos_msg.x = real_x      # 드론 기준 전방 거리 (미터)
            pos_msg.y = real_y      # 드론 기준 좌우 거리 (미터)
            pos_msg.z = distance    # 드론 기준 상하 거리 (미터)
            pos_msg.yaw = 0.0       # 필요시 계산
            pos_msg.height = 0.0    # 필요시 계산
            
            self.drop_tag_pub.publish(pos_msg)
            self.get_logger().info(f"HagiTag detected! Real position: ({real_x:.3f}, {real_y:.3f}, {distance:.3f}), Confidence: {confidence:.2f}")
            
        else:
            self.hagi_tag_detected = False
            self.get_logger().info("HagiTag not detected")




    #============================================
    # Helper Functions
    #============================================

    def detect_landing_tag(self, image): #to-do
        # Opencv aruco settings for apriltag detection
        dictionary   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        det_params   = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(dictionary, det_params)

        # Gray scale conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 태그 검출
        corners, ids, rejected = detector.detectMarkers(gray_image)
        if ids is None or len(ids) == 0:
            return np.array([-1,0,0,0,0]) # detection fail, return -1
        
        # 일단 그냥 첫번째 태그 사용, 나중에 위에처럼 수정
        img_pts = corners[0].reshape(-1, 2).astype(np.float32)  # (4,2)

        # solvePnP용 3D 좌표계 정의 (april_tag 좌표게 만드는 코드)
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
            return np.array([-2,0,0,0,0]) # pose estimate fail, return -2

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

        return np.array([x,y,z,yaw,z])


    def detect_target(self, image): #to-do
        # --- Used to detect the target basket for rescue ---
        # Replace with your algorithm (e.g., ArUco, color blob, YOLO, etc.)
        height, width, _ = image.shape

        # Simulated center
        cx, cy = width // 2, height // 2

        # Fake data (replace this with real detection outputs)
        return (0.1, -0.05, 0.0, 0.0, 1.2)  # x, y, z, yaw, height

    def detect_hagi_tag(self, image):
        """
        HagiTag 감지 함수 (HagiTagDetection_RedHsv.py 기반)
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


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
