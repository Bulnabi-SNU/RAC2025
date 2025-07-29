__author__ = "PresidentPlant"
__contact__ = ""

# import rclpy: ros library
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from enum import Enum

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from vehicle_controller.core.logger import Logger

# import px4_msgs
from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition

# import math library
import numpy as np
import math

# gps
import pymap3d as p3d

# Custom Messages
from custom_msgs.msg import VehicleState, TargetLocation

class MissionState(Enum):
    INIT = "INIT"
    TRACK = "TRACK"
    DESCEND = "DESCEND"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    ASCEND = "ASCEND"

class MissionController(PX4BaseController):

    def __init__(self):
        super().__init__("mc_main")

        # External data placeholders
        self.target = None
        
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False

        # State machine
        self.state = MissionState.INIT  # Initial state
        
        # Offboard controller
        self.drone_target_controller = DroneTargetController(target_distance=0.35, target_altitude=self.track_min_altitude, 
                                                             acceptance_radius=0.1)

        # CV Detection Subscriber
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )

        self.get_logger().info("Mission Controller initialized")

    def main_loop(self):
        """Main control loop - implements the state machine"""

        if self.state == MissionState.INIT:
            self._handle_init()
        elif self.state == MissionState.TRACK:
            self._handle_track_target(MissionState.DESCEND)
        elif self.state == MissionState.DESCEND:
            self._handle_descend(MissionState.GRIPPER_CLOSE)
        elif self.state == MissionState.GRIPPER_CLOSE:
            self._handle_gripper_close()
        elif self.state == MissionState.ASCEND:
            self._handle_ascend()

        # 1: Casualty
        # 2: Drop Tag
        # 3: Landing Tag
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=1
                ),
            )

    def on_target_update(self, msg):
        """Callback for target coordinates from image_processing_node"""
        if msg is not None:
            self.target = msg
        else:
            self.target = None
            # self.get_logger().info(f"Received target coordinates: {self.target.x}, {self.target.y}")

    # =======================================
    # State Machine Functions
    # =======================================

    def _handle_init(self):
        """Initialize system and check status"""
        if not self.get_position_flag:
            self.get_logger().info("Waiting for position data...")
            return

        self.set_home_position()
        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = MissionState.TRACK

    def _handle_track_target(self, nextState: MissionState):
        """Track target using vision and transition to next state when arrived"""
        if self.target is None:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return

        next_setpoint, arrived = self.drone_target_controller.update(
            self.pos, self.yaw, self.target.angle_x, self.target.angle_y
        )
        self.publish_setpoint(pos_sp=next_setpoint)

        if arrived:
            self.state = nextState

    def _handle_descend(self, nextState: MissionState):
        """Descend to casualty pickup position"""
        # Set position setpoint to pickup altitude
        pickup_pos = np.array([self.pos[0], self.pos[1], self.gripper_altitude])
        self.publish_setpoint(pos_sp=pickup_pos)

        # Check if at pickup altitude
        if -self.pos[2] < self.gripper_altitude:
            self.state=nextState

    def _handle_gripper_close(self):
        """Close gripper to pick up casualty"""
        # TODO: Implement gripper control
        
        self.get_logger().info("Closing gripper to pick up casualty")

        # TODO: Add logic to check if this state is complete
        if True:
            self.dropoff_complete = True
            self.state = MissionState.ASCEND

    def _handle_ascend(self):
        """Return to mission altitude with casualty"""
        ascend_pos = np.array([self.pos[0], self.pos[1], self.mission_altitude])
        self.publish_setpoint(pos_sp=ascend_pos)

        if abs(self.pos[2] - self.mission_altitude) < 0.2:
            return

    
    # =======================================
    # Additional Functions
    # =======================================
    

    def on_vehicle_status_update(self, msg):
        """Override to handle vehicle status updates"""
        # Could add additional status monitoring here
        pass

    def on_local_position_update(self, msg):
        """Override to handle local position updates"""
        # Could add position monitoring here
        pass

    def on_global_position_update(self, msg):
        """Override to handle global position updates"""
        # Could add GPS monitoring here
        pass


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        controller = MissionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Mission interrupted by user")
    except Exception as e:
        print(f"Mission failed with error: {e}")
    finally:
        if "controller" in locals():
            controller.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
