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

class ImageProcessor(Node):

    #============================================
    # Initialize Node
    #============================================

    def __init__(self, use_gazebo=False):
        super().__init__('image_processor_node')

        """
        -1. set camera info
        : real camera or Gazebo camera
        """
        if use_gazebo:
            self.topicNameFrames = "/world/RAC_2025/model/standard_vtol_0/link/camera_link/sensor/camera/image"     # Change to for real camera'topic_camera_image'
        else:
            self.topicNameFrames = "topic_camera_image"  # Change to for real camera'topic_camera_image'
        
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
            image_qos_profile = qos_profile

        """ for VehicleState callback"""
        self.vehicle_state = VehicleState()
        
        """ for tracking detection states"""
        self.detection_status = -1
        self.detection_cx = 0
        self.detection_cy = 0

        """ CVBridge configuration """
        self.bridge = CvBridge()
        self.queueSize=20
        self.last_image = None

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
        self.timer_period = 0.05
        self.streaming_period = 0.1
        self.main_timer = self.create_timer(self.timer_period, self.main_timer_callback)
        self.streaming_timer = self.create_timer(self.streaming_period, self.streaming_timer_callback)

        '''
        2. Instantiating Different Detectors
        '''

        # [Casualty Detector]
        self.casualty_detector = CasualtyDetector(
            lower_orange=np.array([5, 150, 150]),
            upper_orange=np.array([20, 255, 255]),
            min_area=500
        )

        # [DropTag Detector]
        self.drop_tag_detector = DropTagDetector(
            lower_red1=np.array([0, 100, 50]),
            upper_red1=np.array([10, 255, 255]),
            lower_red2=np.array([170, 100, 50]),
            upper_red2=np.array([180, 255, 255]),
            min_area=500,
            pause_threshold=0.4
        )

        # [Landing Tag Detector]
        self.landing_tag_detector = LandingTagDetector(
            tag_size=0.5,       
            K=np.array([
                            [1070.089695, 0.0, 1045.772015],
                            [0.0, 1063.560096, 566.257075],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float64),                     
            D=np.array([-0.090292, 0.052332, 0.000171, 0.006618, 0.0], dtype=np.float64),                    
        )
        
    #============================================
    # "Timer" Callback Functions
    #============================================     

    def main_timer_callback(self):
        # Assume no detection unless proven otherwise
        self.detection_status = -1
        
        if self.last_image is None:
            self.get_logger().warn("No image received yet, skipping processing")
            return
        
        if self.vehicle_state.detect_target_type == 0:
            return  # Skip processing if no detection is required
        elif self.vehicle_state.detect_target_type == 1:
            # Casualty 감지 및 중심으로 이동
            self.handle_detect_casualty()
        elif self.vehicle_state.detect_target_type == 2:
            # DropTag 감지 및 중심으로 이동
            self.handle_detect_drop_tag()
        elif self.vehicle_state.detect_target_type == 3:
            # Landing Tag 감지 및 중심으로 이동
            self.handle_detect_landing_tag()
        
        h, w, _ = self.last_image.shape
        
        targetLocation = TargetLocation()
        targetLocation.status = self.detection_status
        targetLocation.angle_x, targetLocation.angle_y = \
            pixel_to_fov(self.detection_cx,self.detection_cy,w,h,81,93)
        
        self.target_pub.publish(targetLocation)
    
    def streaming_timer_callback(self):
        if self.last_image is None:
            self.get_logger().warn("No image received yet, skipping processing")
            return
        
        output_image = self.last_image.copy()
        
        self.render_image(output_image)

        # show image to monitor
        resized_frame = cv2.resize(output_image, (960, 540))
        cv2.imshow("Image Processor", resized_frame)
        cv2.waitKey(1)  
        #print(self.last_image) 

    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    def state_callback(self, msg):
        self.vehicle_state = msg
        self.get_logger().debug(f"Vehicle state updated: detect_target_type = {msg.detect_target_type}") #for logging vehiclestate

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # TODO maybe add additional logic by-case
    #============================================
    # Detection Related Functions
    #============================================ 
    
    def handle_detect_casualty(self):
        
        detection, _ =  self.casualty_detector.detect_casualty(self.last_image)
        
        if detection is None:
            self.detection_status = -1
            return
        
        self.detection_status = 0
        self.detection_cx = int(detection[0])
        self.detection_cy = int(detection[1])

    def handle_detect_drop_tag(self):

        detection, _ =  self.drop_tag_detector.detect_drop_tag(self.last_image)
        
        if detection is None:
            self.detection_status = -1
            return
        
        self.detection_status = 0
        self.detection_cx = int(detection[0])
        self.detection_cy = int(detection[1])

    def handle_detect_landing_tag(self):
        
        detection, _ = self.landing_tag_detector.detect_landing_tag(self.last_image)
        
        if detection is None:
            self.detection_status = -1
            return
        
        if _ is None:
            # Set detection_status to -2 if pose isn't calculated
            # For now just assume it's all fine.
            pass
            
        self.detection_status = 0
        self.detection_cx = int(detection[0])
        self.detection_cy = int(detection[1])
        
    """ Function to render shapes/text onto camera feed before streaming """
    def render_image(self, image):
        h, w, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        center_x, center_y = w // 2, h // 2
        cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

        if self.detection_status == 0:
            cv2.circle(image, 
                    (self.detection_cx, self.detection_cy), 
                    5, (0, 0, 255), -1)
            angle_x, angle_y = pixel_to_fov(
                self.detection_cx, self.detection_cy, w, h, 81, 93
            )

            detect_type_label = {
                1: "Casualty",
                2: "DropTag",
                3: "LandingTag"
            }.get(self.vehicle_state.detect_target_type, "Unknown")

            info_lines = [
                f"Detection Mode : {detect_type_label}",
                f"Pixel (x, y)   : ({self.detection_cx}, {self.detection_cy})",
                f"Angle (x, y)   : ({angle_x:.2f}, {angle_y:.2f})"
            ]

            for i, line in enumerate(info_lines):
                y = 20 + i * 20
                (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                cv2.rectangle(image, (10, y - 15), (10 + text_w + 4, y + 5), bg_color, -1)
                cv2.putText(image, line, (12, y), font, font_scale, text_color, font_thickness)
        else:
            cv2.putText(image, "No detection", (10, 30), font, font_scale, (0, 0, 255), font_thickness)



#============================================
# Main Function
#============================================

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def main_gazebo(args=None):
    rclpy.init(args=args)
    node = ImageProcessor(use_gazebo=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()