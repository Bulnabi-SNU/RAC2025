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
from custom_msgs.msg import DropTagLocation  # DropTag 위치 정보용

# import other libraries
import os, math
import numpy as np
import cv2
from cv_bridge import CvBridge #helps convert ros2 images to OpenCV formats

class VehiclePhasePublisher(Node):
    def __init__(self):
        super().__init__('vehicle_phase_test_publisher')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher = self.create_publisher(VehiclePhase, '/vehicle_phase', qos_profile)

        # Timer: publish once every 1 second
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # You can modify these flags to test different scenarios
        self.test_phase = VehiclePhase()
        self.test_phase.vehicle_state = ""
        self.test_phase.vehicle_phase = ""
        self.test_phase.vehicle_subphase = ""
        self.test_phase.do_landing = True
        self.test_phase.do_detecting = False
        self.test_phase.do_drop = False

        self.get_logger().info("VehiclePhase test publisher initialized.")

    def timer_callback(self):
        self.publisher.publish(self.test_phase)
        self.get_logger().info("Published VehiclePhase message with:")
        self.get_logger().info(f"  do_detecting: {self.test_phase.do_detecting}, "
                               f"do_landing: {self.test_phase.do_landing}, "
                               f"do_drop: {self.test_phase.do_drop}")


def main(args=None):
    rclpy.init(args=args)
    node = VehiclePhasePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
