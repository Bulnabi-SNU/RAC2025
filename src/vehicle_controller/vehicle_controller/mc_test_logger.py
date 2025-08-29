#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation

# Import the logger class from your package
from vehicle_controller.core.logger import Logger

# Import all the necessary PX4 message types
from px4_msgs.msg import SensorGps, VehicleLocalPosition, VehicleAcceleration, VehicleAttitude

class LoggerTestNode(Node):
    """
    A simple ROS 2 node to test the Logger class functionality.
    It subscribes to sensor topics and logs the data to a CSV file.
    """
    def __init__(self):
        super().__init__('logger_test_node')

        # 1. Initialize the Logger
        self.logger = Logger(log_path="/workspace/flight_logs")
        self.logger.start_logging()
        self.get_logger().info(f"Logger test started. Writing to: {self.logger.log_path}")

        # 2. Initialize data holders with the correct message types
        self.vehicle_gps = SensorGps()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_acceleration = VehicleAcceleration()
        self.vehicle_attitude = VehicleAttitude()

        # 3. Create Subscribers
        # This QoS profile is robust for receiving sensor data
        self.qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.gps_sub = self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position', self.gps_callback, self.qos_profile)
        
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_pos_callback, self.qos_profile)
        
        self.accel_sub = self.create_subscription(
            VehicleAcceleration, '/fmu/out/vehicle_acceleration', self.accel_callback, self.qos_profile)
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, self.qos_profile)

        # 4. Create a timer to call the logging function periodically
        self.log_timer = self.create_timer(0.1, self.log_timer_callback) # Log at 10 Hz
        self.get_logger().info("Logger test node initialized and subscribers created.")

    # Subscriber Callbacks to store the latest message
    def gps_callback(self, msg):
        self.vehicle_gps = msg

    def local_pos_callback(self, msg):
        self.vehicle_local_position = msg

    def accel_callback(self, msg):
        self.vehicle_acceleration = msg

    def attitude_callback(self, msg):
        self.vehicle_attitude = msg

    # The main logging function, called by the timer
    def log_timer_callback(self):
        # Sanity check: wait for all data to arrive before logging to prevent errors
        if self.vehicle_attitude.timestamp == 0: #or self.vehicle_local_position.timestamp == 0:  #or self.vehicle_gps.timestamp == 0 
            self.get_logger().warn("Waiting for all sensor data to be received...")
            return

        # Convert Quaternion to Euler angles
        q_px4 = self.vehicle_attitude.q
        r = Rotation.from_quat([q_px4[1], q_px4[2], q_px4[3], q_px4[0]])
        roll_rad, pitch_rad, yaw_rad = r.as_euler('zyx')

        # Prepare dummy variables that are part of the logger's function signature
        auto_flag = 0  # Stationary, so we can consider it "manual"
        camera_detection_flag = 0
        waypoint_num = 0

        # Call the logger with all the required data points
        self.logger.log_data(
            auto_flag,
            self.vehicle_gps.latitude_deg,
            self.vehicle_gps.longitude_deg,
            self.vehicle_gps.altitude_ellipsoid_m,
            roll_rad,
            pitch_rad,
            yaw_rad,
            self.vehicle_local_position.timestamp,
            self.vehicle_acceleration.xyz[0],
            self.vehicle_acceleration.xyz[1],
            self.vehicle_acceleration.xyz[2],
            self.vehicle_local_position.ax,
            self.vehicle_local_position.ay,
            self.vehicle_local_position.az,
            camera_detection_flag,
            waypoint_num
        )
        self.get_logger().info("Logged one data point.")


def main(args=None):
    rclpy.init(args=args)
    node = LoggerTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nLogger test stopped by user.")
    finally:
        # Cleanly destroy the node and shut down rclpy
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
