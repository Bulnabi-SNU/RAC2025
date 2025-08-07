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
# Messages for changing params dynamically
from rcl_interfaces.msg import SetParametersResult

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
            self.topicNameFrames = "/world/RAC_2025/model/standard_vtol_gimbal_0/link/camera_link/sensor/camera/image"     # Change to for real camera'topic_camera_image'
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
        
        image_qos_profile = qos_profile
        
        if use_gazebo:
            image_qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,  
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

        """ Parameters """
        self.declare_parameters(
            namespace='',
            parameters=[
                ('do_streaming', False),
                
                ('casualty_lower_orange', [5.0,150.0,150.0]),
                ('casualty_upper_orange', [20.0, 255.0, 255.0]),
                ('casualty_min_area', 500.0),

                ('drop_tag_lower_red1', [0.0,100.0,50.0]),
                ('drop_tag_upper_red1', [10.0,255.0,255.0]),
                ('drop_tag_lower_red2', [170.0,100.0,50.0]),
                ('drop_tag_upper_red2', [180.0,255.0,255.0]),
                ('drop_tag_min_area', 500.0),
                ('drop_tag_pause_threshold', 0.4),

                ('landing_tag_tag_size', 0.5),
                ('landing_tag_K', [1070.089695, 0.0, 1045.772015, 
                                    0.0, 1063.560096, 566.257075,
                                    0.0, 0.0, 1.0]),
                ('landing_tag_D', [-0.090292, 0.052332, 0.000171, 0.006618, 0.0])
        ])
        
        self.horizontal_fov = 123
        self.vertical_fov = 123

        self.do_streaming = self.get_parameter('do_streaming').value

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
        # 1. VehicleState           : current state of vehicle (ex. CASUALTY_TRACK, DESCEND_PICKUP, etc.)
        #    - detect_target_type   : 0: No detection, 1: Casualty detection, 2: Casualty dropoff detection, 3: Landing tag detection
        # 2. CameraRawImage         :  Image received from the Camera
        self.vehicle_state_subscriber = self.create_subscription(VehicleState, '/vehicle_state', self.state_callback, qos_profile)
        self.camera_image_subscriber = self.create_subscription(Image, self.topicNameFrames, self.image_callback, image_qos_profile)

        # [Publishers]
        # 1. TargetLocation         : publish the target location based on the type given by VehicleState
        self.target_pub = self.create_publisher(TargetLocation, '/target_position', qos_profile)

        # [Timers]
        self.timer_period = 0.05 # Maybe set this into a parameter?
        self.streaming_period = 0.1
        self.main_timer = self.create_timer(self.timer_period, self.main_timer_callback)

        if(self.do_streaming): self.streaming_timer = self.create_timer(self.streaming_period, self.streaming_timer_callback)

        # [Parameter callback] 
        # : for dynamically updating parameters while flying
        self.add_on_set_parameters_callback(self.param_update_callback)
        
        '''
        2. Instantiating Different Detectors
        '''        

        # [Casualty Detector]
        self.casualty_detector = CasualtyDetector(
            lower_orange=np.array(self.get_parameter('casualty_lower_orange').value),
            upper_orange=np.array(self.get_parameter('casualty_upper_orange').value),
            min_area=(self.get_parameter('casualty_min_area').value)
        )

        # [DropTag Detector]
        self.drop_tag_detector = DropTagDetector(
            lower_red1=np.array(self.get_parameter('drop_tag_lower_red1').value),
            upper_red1=np.array(self.get_parameter('drop_tag_upper_red1').value),
            lower_red2=np.array(self.get_parameter('drop_tag_lower_red2').value),
            upper_red2=np.array(self.get_parameter('drop_tag_upper_red2').value),
            min_area=self.get_parameter('drop_tag_min_area').value,
            pause_threshold=self.get_parameter('drop_tag_pause_threshold').value
        )

        # [Landing Tag Detector]
        self.landing_tag_detector = LandingTagDetector(
            tag_size=self.get_parameter('landing_tag_tag_size').value,       
            K=np.reshape(np.array(self.get_parameter('landing_tag_K').value,dtype=np.float64),(3,3)),                     
            D=np.array(self.get_parameter('landing_tag_D').value, dtype=np.float64),                    
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
        # TODO: SET ROS PARAMETERS for fov 
        targetLocation.angle_x, targetLocation.angle_y = \
            pixel_to_fov(self.detection_cx, self.detection_cy, w, h,114,159)
        
        self.target_pub.publish(targetLocation)
    
    def streaming_timer_callback(self):
        if self.last_image is None:
            self.get_logger().warn("No image received yet, skipping processing")
            return
        
        output_image = self.last_image.copy()
        
        self.render_image(output_image)

        # Show image to connected monitor
        resized_frame = cv2.resize(output_image, (960, 540))
        cv2.imshow("Image Processor", resized_frame)
        cv2.waitKey(1)  
        #print(self.last_image) 

    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    def state_callback(self, msg):
        self.vehicle_state = msg
        self.get_logger().debug(f"Vehicle state updated: detect_target_type = {msg.detect_target_type}")

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # TODO maybe add additional logic by-case
    #============================================
    # Detection Related Functions
    # All detectors return a 2d array (cx,cy) and additional info when it's needed.
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
        
    #============================================
    # Streaming Related Functions
    #============================================     
    
    def render_image(self, image):
        """ Function to render shapes/text onto camera feed before streaming """
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
    # Parameter callback
    #============================================     

    def param_update_callback(self, params):
        """ Parameter callback : for dynamically updating parameters while flying """
        successful = True
        reason = ''

        for p in params:
            # ────────── Casualty ──────────
            if p.name == 'casualty_lower_orange' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.casualty_detector.lower_orange = np.array(p.value, dtype=np.float32)
            elif p.name == 'casualty_upper_orange' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.casualty_detector.upper_orange = np.array(p.value, dtype=np.float32)

            # ────────── Drop-tag ──────────
            elif p.name == 'drop_tag_lower_red1' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.drop_tag_detector.lower_red1 = np.array(p.value, dtype=np.float32)
            elif p.name == 'drop_tag_upper_red1' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.drop_tag_detector.upper_red1 = np.array(p.value, dtype=np.float32)
            elif p.name == 'drop_tag_lower_red2' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.drop_tag_detector.lower_red2 = np.array(p.value, dtype=np.float32)
            elif p.name == 'drop_tag_upper_red2' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.drop_tag_detector.upper_red2 = np.array(p.value, dtype=np.float32)

            # ────────── Landing-tag ──────────
            # elif p.name == 'landing_tag_tag_size' and p.type_ == p.Type.DOUBLE:
            #     self.landing_tag_detector.tag_size = p.value
            elif p.name == 'landing_tag_K' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.landing_tag_detector.K = np.reshape(np.array(p.value, dtype=np.float64), (3, 3))
            elif p.name == 'landing_tag_D' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.landing_tag_detector.D = np.array(p.value, dtype=np.float64)

            # 예상하지 못한/지원하지 않는 파라미터
            else:
                self.get_logger().warn(f"Ignoring unknown parameter {p.name}")
                continue
        
        print("[Parameter Update] Current parameters:")
        self.casualty_detector.print_param()
        print("\n")
        self.drop_tag_detector.print_param()
        print("\n")
        self.landing_tag_detector.print_param()
        print("\n\n")

        return SetParametersResult(successful=successful, reason=reason)

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
