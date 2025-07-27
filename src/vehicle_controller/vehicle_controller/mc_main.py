__author__ = "PresidentPlant"
__contact__ = ""
# import rclpy: ros library
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController

# import px4_msgs
from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition

# import math library
import numpy as np
import math

# gps
import pymap3d as p3d

# Custom Messages
from custom_msgs.msg import VehicleState, TargetLocation


class MissionController(PX4BaseController):

    def __init__(self):
        super().__init__("mc_main")

        # Mission parameters
        self.pickup_waypoint = 14  # waypoint number for pickup
        self.dropoff_waypoint = 15  # waypoint number for dropoff
        self.landing_waypoint = 16  # waypoint number for landing
        self.pickup_altitude = 0.3  # altitude for pickup/dropoff operations
        self.track_min_altitude = -4.0 # minimum altitude for tracking - just go down without tracking from here
        self.mission_altitude = -15.0  # normal mission altitude

        # External data placeholders
        self.target = None
        
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False

        # State machine
        self.state = "INIT"  # Initial state
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        
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

        if self.state == "INIT":
            self._handle_init()
        elif self.state == "OFFBOARD_ARM":
            self._handle_offboard_arm()
        elif self.state == "TO_MISSION":
            self._handle_to_mission()
        elif self.state == "MISSION_EXECUTE":
            self._handle_mission_execute()
        elif self.state == "MISSION_TO_OFFBOARD":
            self._handle_mission_to_offboard()
        elif self.state == "CASUALTY_TRACK":
            self._handle_casualty_track()
        elif self.state == "DESCEND_PICKUP":
            self._handle_descend_pickup()
        elif self.state == "GRIPPER_CLOSE":
            self._handle_gripper_close()
        elif self.state == "ASCEND_PICKUP":
            self._handle_ascend_pickup()
        elif self.state == "OFFBOARD_TO_MISSION":
            self._handle_offboard_to_mission()
        elif self.state == "MISSION_CONTINUE":
            self._handle_mission_continue()
        elif self.state == "MISSION_TO_OFFBOARD_DROPOFF":
            self._handle_mission_to_offboard_dropoff()
        elif self.state == "DROPOFF_CASUALTY_TRACK":
            self._handle_dropoff_casualty_track()
        elif self.state == "DESCEND_DROPOFF":
            self._handle_descend_dropoff()
        elif self.state == "GRIPPER_OPEN":
            self._handle_gripper_open()
        elif self.state == "ASCEND_DROPOFF":
            self._handle_ascend_dropoff()
        elif self.state == "LANDING_TAG_TRACK":
            self._handle_landing_tag_track()
        elif self.state == "FINAL_DESCEND":
            self._handle_final_descend()
        elif self.state == "LAND":
            self._handle_land()
        elif self.state == "MISSION_COMPLETE":
            self._handle_mission_complete()
        elif self.state == "ERROR":
            self._handle_error()

        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state,
                detect_target_type=(
                    1
                    if self.state
                    in [
                        "CASUALTY_TRACK",
                        "DESCEND_PICKUP",
                        "GRIPPER_CLOSE",
                        "ASCEND_PICKUP",
                    ]
                    else (
                        2
                        if self.state
                        in [
                            "DROPOFF_CASUALTY_TRACK",
                            "DESCEND_DROPOFF",
                            "GRIPPER_OPEN",
                            "ASCEND_DROPOFF",
                        ]
                        else (
                            3
                            if self.state in ["LANDING_TAG_TRACK", "FINAL_DESCEND"]
                            else 0
                        )
                    )
                ),
            )
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
            self.state = "OFFBOARD_ARM"

    def _handle_offboard_arm(self):
        """If at offboard mode, arm"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode")
            self.state = "TO_MISSION"

    def _handle_to_mission(self):
        """Transition to mission mode"""
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.get_logger().info("Successfully switched to mission mode")
            self.state = "MISSION_EXECUTE"

    def _handle_mission_execute(self):
        """Execute mission"""
        if not self.is_mission_mode():
            #    self.get_logger().error("Not in mission mode at MISSION_EXECUTE")
            return

        current_wp = self.mission_wp_num

        # Check if reached pickup waypoint
        if current_wp == self.pickup_waypoint and not self.pickup_complete:
            self.mission_paused_waypoint = current_wp
            self.state = "MISSION_TO_OFFBOARD"
            return

        # Check if reached dropoff waypoint
        if current_wp == self.dropoff_waypoint and not self.dropoff_complete:
            self.mission_paused_waypoint = current_wp
            self.state = "MISSION_TO_OFFBOARD_DROPOFF"
            return

        # Check if reached landing waypoint
        if current_wp == self.landing_waypoint:
            self.state = "LANDING_TAG_TRACK"
            return

    def _handle_mission_to_offboard(self):
        """Switch from mission to offboard for pickup"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = "CASUALTY_TRACK"

    def _handle_casualty_track(self):
        """Track casualty using CV"""
        
        if self.target is None:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return
        
        next_setpoint, arrived = self.drone_target_controller.update(self.pos, self.yaw, self.target.angle_x, self.target.angle_y)
        self.publish_setpoint(pos_sp=next_setpoint)

        # Check if over casualty
        if arrived:
            self.state = "DESCEND_PICKUP"

    def _handle_descend_pickup(self):
        """Lower altitude to pickup position"""
        pickup_pos = np.array([self.pos[0], self.pos[1], 0])
        self.publish_setpoint(pos_sp=pickup_pos)

        if -self.pos[2] < self.pickup_altitude:
            self.state = "GRIPPER_CLOSE"

    def _handle_gripper_close(self):
        """Close gripper to pick up casualty"""
        # TODO: Implement gripper control
        # Send gripper close command
        self.get_logger().info("Closing gripper to pick up casualty")
        # Simulate gripper operation delay
        self.state = "ASCEND_PICKUP"

    def _handle_ascend_pickup(self):
        """Return to mission altitude with casualty"""
        ascend_pos = np.array([self.pos[0], self.pos[1], self.mission_altitude])
        self.publish_setpoint(pos_sp=ascend_pos)

        if abs(self.pos[2] - self.mission_altitude) < 0.2:
            self.pickup_complete = True
            self.state = "OFFBOARD_TO_MISSION"

    def _handle_offboard_to_mission(self):
        """Resume mission mode after pickup"""
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.state = "MISSION_CONTINUE"

    def _handle_mission_continue(self):
        """Continue mission from paused waypoint"""
        # TODO: Debate whether to remove this?
        self.get_logger().info(
            f"Resuming mission from waypoint {self.mission_paused_waypoint}"
        )
        self.target = None
        self.state = "MISSION_EXECUTE"

    def _handle_mission_to_offboard_dropoff(self):
        """Switch from mission to offboard for dropoff"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = "DROPOFF_CASUALTY_TRACK"

    def _handle_dropoff_casualty_track(self):
        """Track dropoff point (red cross) using CV/ArUco"""
        if self.target is None:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return
        
        next_setpoint, arrived = self.drone_target_controller.update(self.pos, self.yaw, self.target.angle_x, self.target.angle_y)
        self.publish_setpoint(pos_sp=next_setpoint)
        
        if arrived:
            self.state = "DESCEND_DROPOFF"

    def _handle_descend_dropoff(self):
        """Lower altitude to dropoff position"""
        dropoff_pos = np.array([self.pos[0], self.pos[1], 0])
        self.publish_setpoint(pos_sp=dropoff_pos)

        if -self.pos[2] < self.pickup_altitude:
            self.state = "GRIPPER_OPEN"

    def _handle_gripper_open(self):
        """Open gripper to release casualty at dropoff point"""
        # TODO: Implement gripper control
        # Send gripper open command
        self.get_logger().info("Opening gripper to release casualty at dropoff point")
        self.state = "ASCEND_DROPOFF"

    def _handle_ascend_dropoff(self):
        """Return to mission altitude after dropoff"""
        ascend_pos = np.array([self.pos[0], self.pos[1], self.mission_altitude])
        self.publish_setpoint(pos_sp=ascend_pos)

        if abs(self.pos[2] - self.mission_altitude) < 0.2:
            self.dropoff_complete = True
            self.get_logger().info("Dropoff complete, returning to mission mode")
            self.state = "OFFBOARD_TO_MISSION"

    def _handle_landing_tag_track(self):
        """Track landing target (ArUco tag) using CV/ArUco"""
        
        if self.target is None:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return
        
        next_setpoint, arrived = self.drone_target_controller.update(self.pos, self.yaw, self.target.angle_x, self.target.angle_y)
        self.publish_setpoint(pos_sp=next_setpoint)
        
        if arrived:
            self.state = "DESCEND_DROPOFF"

    def _handle_final_descend(self):
        """Descend for landing"""
        landing_pos = np.array([self.pos[0], self.pos[1], self.pickup_altitude])
        self.publish_setpoint(pos_sp=landing_pos)

        if abs(self.pos[2] - self.pickup_altitude) < 0.2:
            self.state = "LAND"

    def _handle_land(self):
        """Final landing sequence"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.state = "MISSION_COMPLETE"

    def _handle_mission_complete(self):
        """Mission finished"""
        self.get_logger().info("Mission Complete!")
        pass

    def _handle_error(self):
        """Error handling state"""
        self.get_logger().error("Mission in error state")
        # TODO: Implement error recovery or emergency procedures
        pass

    # =======================================
    # Additional Functions
    # =======================================

    def set_casualty_coordinates(self, x, y):
        """Set casualty coordinates from external CV/ArUco system"""
        self.casualty_coordinates = [x, y]

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
