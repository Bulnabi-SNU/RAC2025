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

        '''Use temporarily while no phase publisher exists'''
        self.phase = VehiclePhase()
        self.phase.do_detecting = True  # Force target detection mode
        self.phase.do_landing = False   # Make sure landing is off


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
        self.vehicle_phase_subscriber = self.create_subscription(VehiclePhase, '/phase', self.phase_callback, qos_profile)
        self.camera_image_subscriber = self.create_subscription(Image, self.topicNameFrames, self.image_callback, qos_profile)

        # [Publishers]
        # 1. LandingTagLocation      : publish the landing tag location
        # 2. TargetLocation          : publish the target location
        self.landing_pub = self.create_publisher(LandingTagLocation, '/landing_tag_position', qos_profile)
        self.target_pub = self.create_publisher(TargetLocation, '/target_position', qos_profile)



    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    def phase_callback(self, msg):
        self.phase = msg

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.phase is None:
            return  # Skip processing if phase is not defined
        detection = None
        if self.phase.do_landing:
            self.publish_landing_tag_location()
        elif self.phase.do_detecting:
            self.publish_target_location()

    

    #============================================
    # "Publisher" Callback Functions
    #============================================ 

    def publish_landing_tag_location(self):
        if self.phase is None:
            return  # Skip processing if phase is not defined
        detection = self.detect_landing_tag(self.last_image)
        if not detection:
            return
        x, y, z, yaw, height = detection
        pos_msg = LandingTagLocation()
        pos_msg.x = x                                   # x coordinate of the object's center position relative to the screen
        pos_msg.y = y                                   # y coordinate of the object's center position relative to the screen
        pos_msg.z = z                                   # distance of the object's center position relative to the camera
        pos_msg.yaw = yaw                               # angle deviation away from proper alignment 
        pos_msg.height = height                         # height of the vehicle if probably not necessary here
        self.landing_pub.publish(pos_msg)
        self.get_logger().info(f"Publishing landing tag location: x={x:.2f}, y={y:.2f}")


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




    #============================================
    # Helper Functions
    #============================================

    def detect_landing_tag(self, image): #to-do
        # --- Used for Auto-Landing using AprilTag Markers ---
        # Replace with your algorithm (e.g., ArUco, color blob, YOLO, etc.)
        height, width, _ = image.shape

        # Simulated center
        cx, cy = width // 2, height // 2

        # Fake data (replace this with real detection outputs)
        return (0.1, -0.05, 0.0, 0.0, 1.2)  # x, y, z, yaw, height


    def detect_target(self, image): #to-do
        # --- Used to detect the target basket for rescue ---
        # Replace with your algorithm (e.g., ArUco, color blob, YOLO, etc.)
        height, width, _ = image.shape

        # Simulated center
        cx, cy = width // 2, height // 2

        # Fake data (replace this with real detection outputs)
        return (0.1, -0.05, 0.0, 0.0, 1.2)  # x, y, z, yaw, height


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
