# camera_publisher_node.py

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
from sensor_msgs.msg import Image                   ### 나중에 센서 이미지 받아오는것 찾아보기 ###

# import other libraries
import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats


class ImagePublisher(Node):

    #============================================
    # Initialize Node
    #============================================

    def __init__(self):
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
        self.cameraDeviceNumber=1
        self.camera = cv2.VideoCapture(self.cameraDeviceNumber)
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
        if success == True:
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
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    node.camera.release()
    rclpy.shutdown()
