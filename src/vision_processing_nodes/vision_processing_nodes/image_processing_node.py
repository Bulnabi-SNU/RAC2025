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

# import detection functions and utilities
from vision_processing_nodes.detection.landingtag import LandingTagDetector
from vision_processing_nodes.detection.casualty import CasualtyDetector
from vision_processing_nodes.detection.droptag import DropTagDetector
from vision_processing_nodes.detection.utils import pixel_to_fov

### MODE SETTING (GAZEBO)
use_gazebo = False  # Set to True if running in Gazebo, False for live camera or video feed

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

        '''
        2. Instantiating Different Detectors
        '''
        # [Landing Tag Detector]
        self.landing_tag_detector = LandingTagDetector(
            tag_size=self.tag_size,       
            K=self.K,                     
            D=self.D,                    
        )

        # [Casualty Detector]
        self.casualty_detector = CasualtyDetector(
            lower_orange=np.array([5, 150, 150]),
            upper_orange=np.array([20, 255, 255]),
            min_area=500
        )

        # [DropTag Detector]
        self.droptag_detector = DropTagDetector(
            lower_red1=np.array([0, 100, 50]),
            upper_red1=np.array([10, 255, 255]),
            lower_red2=np.array([170, 100, 50]),
            upper_red2=np.array([180, 255, 255]),
            min_area=500,
            pause_threshold=0.4
        )

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
    # Landing Tag (AprilTag) Publisher
    #============================================ 
    def publish_landing_tag_location(self):
        if self.last_image is None:
            self.get_logger().warn("No image received in LandingTagDetector.")
            return
            
        detection = self.landing_tag_detector.detect_landing_tag(self.last_image)
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
  

    #============================================
    # Casualty (Basket) Publisher
    #============================================ 

    # ====== 바구니(오렌지색) HSV 캘리브레이션 범위 & 최소 면적 ======
    def publish_casualty_location(self):
        detection = self.casualty_detector.detect_casualty_with_angles(self.last_image)
        if detection is None:
            return
        
        pos_msg = TargetLocation()
        pos_msg.x = float('nan')
        pos_msg.y = float('nan')
        pos_msg.z = float('nan')
        pos_msg.yaw = float('nan')
         # Set angular coordinates from detection
        pos_msg.angle_x = detection['angle_x']
        pos_msg.angle_y = detection['angle_y']
        
        self.target_pub.publish(pos_msg)
        self.get_logger().info(f"Publishing target location: angle_x={pos_msg.angle_x:.2f}, angle_y={pos_msg.angle_y:.2f}") #optional

    #============================================
    # DropTag Related Functions
    #============================================ 

    def publish_casualty_dropoff_location(self):
        """
        Detect and publish DropTag location using DropTagDetector
        Includes pause/resume logic based on red pixel ratio
        """
        if self.last_image is None:
            return
        
        # Check if detection should be paused
        should_pause, red_ratio, screen_center = self.droptag_detector.should_pause_detection(self.last_image)
        state_changed, is_paused = self.droptag_detector.update_pause_state(should_pause)
        
        # Log pause state changes
        if state_changed:
            if is_paused:
                self.get_logger().info(f"DropTag detection paused. Center: {screen_center}")
            else:
                self.get_logger().info("DropTag detection resumed.")
        
        # Skip detection if paused
        if is_paused:
            return
        
        # Perform detection using the detector
        detection = self.droptag_detector.detect_droptag_with_position(self.last_image)
        
        if detection is not None:
            # Update state tracking
            self.drop_tag_detected = True
            self.drop_tag_center = detection['pixel_center']
            self.drop_tag_confidence = detection['confidence']
            
            # Extract position data
            real_x, real_y, distance = detection['real_position']
            angle_x, angle_y = detection['angles']
            
            # Create and publish ROS message
            pos_msg = TargetLocation()
            pos_msg.x = real_x          # Real-world x position (meters)
            pos_msg.y = real_y          # Real-world y position (meters)
            pos_msg.z = distance        # Distance to target (meters)
            pos_msg.yaw = float('nan')  # Orientation not calculated
            pos_msg.angle_x = angle_x   # Angular position in FOV
            pos_msg.angle_y = angle_y   # Angular position in FOV
            
            self.target_pub.publish(pos_msg)
            
            # Log detection info
            confidence = detection['confidence']
            self.get_logger().info(
                f"DropTag detected! Real position: ({real_x:.3f}, {real_y:.3f}, {distance:.3f}), "
                f"Confidence: {confidence:.2f}, Angle: ({angle_x:.2f}, {angle_y:.2f})"
            )
        else:
            # Update state for no detection
            self.drop_tag_detected = False
            # Optional: Log no detection (might be too verbose)
            # self.get_logger().info("DropTag not detected")

#============================================
# Main Function
#============================================

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
