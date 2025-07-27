# camera_publisher_node.py
### IMPORTANT: go to main and check for the correct mode (camera for live, video for testing)

"""
Camera Publisher Node
Publishes the Raw Image from the Camera to the Image processing Node
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

# import rclpy: ros library
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image                   

# import other libraries
import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats


class ImagePublisher(Node):

    #============================================
    # Initialize Node
    #============================================

    def __init__(self, mode='camera', video_path='video.mp4'):
        super().__init__('image_publisher_node')
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
        self.mode = mode
        self.videopath = video_path
        self.cameraDeviceNumber=0

        if self.mode == 'camera':
            self.camera = cv2.VideoCapture(self.cameraDeviceNumber)
        elif self.mode == 'video':
            if not os.path.exists(video_path):
                self.get_logger().error(f"Video file not found: {video_path}")
                return
            self.camera = cv2.VideoCapture(video_path)
        else:
            self.get_logger().error("invalid mode. Use 'camera' or 'video'.")
            return
        
        self.bridgeObject = CvBridge()
        self.topicNameFrames='topic_camera_image'
        self.queueSize=20

        """
        1. Subscribers & Publishers & Timers
        """
        # [Publishers]
        # 1. Image      : publish the Raw Image to the image processing node
        self.image_publisher = self.create_publisher(Image, self.topicNameFrames, qos_profile)
        self.periodCommunication = 0.02
        self.timer = self.create_timer(self.periodCommunication,self.timer_callbackFunction)
        self.i=0
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera.")
            return
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)


    #============================================
    # "Subscriber" Callback Functions
    #============================================     
        
    
    #============================================
    # "Publisher" Callback Functions
    #============================================ 


    #============================================
    # "Timer" Callback Functions
    #============================================
    def timer_callbackFunction(self):
        success, frame = self.camera.read()
        if self.mode == 'video' and not success:
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.camera.read()
        if success:
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)  
            ROS2ImageMessage=self.bridgeObject.cv2_to_imgmsg(frame,encoding='bgr8')
            self.image_publisher.publish(ROS2ImageMessage)
            self.get_logger().info('Publishing image number %d' % self.i)
            self.i += 1
        


    #============================================
    # Helper Functions
    #============================================

  
#============================================
# Main
#============================================

def main(args=None):
    rclpy.init(args=args)

    #============================================
    # Choose 'camera' or 'video' mode for testing
    #============================================
    mode = 'video'  # or 'camera'
    video_path = '/workspace/src/vision_processing_nodes/vision_processing_nodes/videos/droptag_video.mov'
    node = ImagePublisher(mode=mode, video_path=video_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.camera.release()
        rclpy.shutdown()
