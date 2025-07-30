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
    OFFBOARD_ARM = "OFFBOARD_ARM"
    MISSION_EXECUTE = "MISSION_EXECUTE"
    MISSION_TO_OFFBOARD_CASUALTY = "MISSION_TO_OFFBOARD_CASUALTY"
    CASUALTY_TRACK = "CASUALTY_TRACK"
    CASUALTY_DESCEND = "CASUALTY_DESCEND"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    CASUALTY_ASCEND = "CASUALTY_ASCEND"
    OFFBOARD_TO_MISSION = "OFFBOARD_TO_MISSION"
    MISSION_CONTINUE = "MISSION_CONTINUE"
    MISSION_TO_OFFBOARD_DROP_TAG = "MISSION_TO_OFFBOARD_DROP_TAG"
    DROP_TAG_TRACK = "DROP_TAG_TRACK"
    DROP_TAG_DESCEND = "DROP_TAG_DESCEND"
    GRIPPER_OPEN = "GRIPPER_OPEN"
    DROP_TAG_ASCEND = "DROP_TAG_ASCEND"
    MISSION_TO_OFFBOARD_LANDING_TAG = "MISSION_TO_OFFBOARD_LANDING_TAG"
    LANDING_TAG_TRACK = "LANDING_TAG_TRACK"
    FINAL_DESCEND = "FINAL_DESCEND"
    LAND = "LAND"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"

class MissionController(PX4BaseController):

    def __init__(self):
        super().__init__("mc_main")

        # Mission parameters
        self.pickup_waypoint = 14  # waypoint number for pickup
        self.dropoff_waypoint = 15  # waypoint number for dropoff
        self.landing_waypoint = 16  # waypoint number for landing
        self.gripper_altitude = 0.3  # altitude for pickup/dropoff operations
        self.track_min_altitude = -4.0 # minimum altitude for tracking - just go down without tracking from here
        self.mission_altitude = -15.0  # normal mission altitude

        # External data placeholders
        self.target = None
        
        # Offboard params - use position mode for now
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False

        # State machine
        self.state = MissionState.INIT  # Initial state
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        
        # Logger
        self.doLogging = False
        self.logger = Logger(log_path="./flight_logs/")
        
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
        elif self.state == MissionState.OFFBOARD_ARM:
            self._handle_offboard_arm()
        elif self.state == MissionState.OFFBOARD_TO_MISSION:
            self._handle_offboard_to_mission()
        elif self.state == MissionState.MISSION_EXECUTE:
            self._handle_mission_execute()
        elif self.state == MissionState.MISSION_TO_OFFBOARD_CASUALTY:
            self._handle_mission_to_offboard(MissionState.CASUALTY_TRACK)
        elif self.state == MissionState.CASUALTY_TRACK:
            self._handle_track_target(MissionState.CASUALTY_DESCEND)
        elif self.state == MissionState.CASUALTY_DESCEND:
            self._handle_descend(MissionState.GRIPPER_CLOSE)
        elif self.state == MissionState.GRIPPER_CLOSE:
            self._handle_gripper_close()
        elif self.state == MissionState.CASUALTY_ASCEND:
            self._handle_ascend()
        elif self.state == MissionState.MISSION_CONTINUE:
            self._handle_mission_continue()
        elif self.state == MissionState.MISSION_TO_OFFBOARD_DROP_TAG:
            self._handle_mission_to_offboard(MissionState.DROP_TAG_TRACK)
        elif self.state == MissionState.DROP_TAG_TRACK:
            self._handle_track_target(MissionState.DROP_TAG_DESCEND)
        elif self.state == MissionState.DROP_TAG_DESCEND:
            self._handle_descend(MissionState.GRIPPER_OPEN)
        elif self.state == MissionState.GRIPPER_OPEN:
            self._handle_gripper_open()
        elif self.state == MissionState.DROP_TAG_ASCEND:
            self._handle_ascend()
        elif self.state == MissionState.MISSION_TO_OFFBOARD_LANDING_TAG:
            self._handle_mission_to_offboard(MissionState.LANDING_TAG_TRACK)
        elif self.state == MissionState.LANDING_TAG_TRACK:
            self._handle_track_target(MissionState.FINAL_DESCEND)
        elif self.state == MissionState.FINAL_DESCEND:
            self._handle_final_descend()
        elif self.state == MissionState.LAND:
            self._handle_land()
        elif self.state == MissionState.MISSION_COMPLETE:
            self._handle_mission_complete()
        elif self.state == MissionState.ERROR:
            self._handle_error()

        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=(
                    1
                    if self.state
                    in [
                        MissionState.CASUALTY_TRACK,
                        MissionState.CASUALTY_DESCEND,
                        MissionState.GRIPPER_CLOSE,
                        MissionState.CASUALTY_ASCEND,
                    ]
                    else (
                        2
                        if self.state
                        in [
                            MissionState.DROP_TAG_TRACK,
                            MissionState.DROP_TAG_DESCEND,
                            MissionState.GRIPPER_OPEN,
                            MissionState.DROP_TAG_ASCEND,
                        ]
                        else (
                            3
                            if self.state in [
                                MissionState.LANDING_TAG_TRACK, 
                                MissionState.FINAL_DESCEND
                                ]
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
            self.state = MissionState.OFFBOARD_ARM

    def _handle_offboard_arm(self):
        """If at offboard mode, arm"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, starting logger")
            
            if self.doLogging:
                self.logger.start_logging()
                self.log_timer = self.create_timer(0.1, self._log_timer_callback)
            
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _handle_offboard_to_mission(self):
        """Resume mission mode after pickup"""
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.state = MissionState.MISSION_CONTINUE

    def _handle_mission_continue(self):
        """Continue mission from paused waypoint"""
        # TODO: Debate whether to remove this?
        self.get_logger().info(
            f"Resuming mission from waypoint {self.mission_paused_waypoint}"
        )
        self.target = None
        self.state = MissionState.MISSION_EXECUTE
        
    def _handle_mission_execute(self):
        """Execute mission"""
        if not self.is_mission_mode():
            #    self.get_logger().error("Not in mission mode at MISSION_EXECUTE")
            return

        current_wp = self.mission_wp_num

        # Check if reached pickup waypoint
        if current_wp == self.pickup_waypoint and not self.pickup_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_CASUALTY
            return

        # Check if reached dropoff waypoint
        if current_wp == self.dropoff_waypoint and not self.dropoff_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_DROP_TAG
            return

        # Check if reached landing waypoint
        if current_wp == self.landing_waypoint:
            self.state = MissionState.MISSION_TO_OFFBOARD_LANDING_TAG
            return

    def _handle_mission_to_offboard(self, nextState: MissionState):
        """Switch from mission to offboard for pickup"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = nextState

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
            self.pickup_complete = True
            self.state = MissionState.CASUALTY_ASCEND

    def _handle_gripper_open(self):
        """Open gripper to release casualty at dropoff point"""
        # TODO: Implement gripper control
        # Send gripper open command
        self.get_logger().info("Opening gripper to release casualty at dropoff point")
        
        # TODO: Add logic to check if this state is complete
        if True:
            self.dropoff_complete = True
            self.state = MissionState.DROP_TAG_ASCEND

    def _handle_ascend(self):
        """Return to mission altitude with casualty"""
        ascend_pos = np.array([self.pos[0], self.pos[1], self.mission_altitude])
        self.publish_setpoint(pos_sp=ascend_pos)

        if abs(self.pos[2] - self.mission_altitude) < 0.2:
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _handle_final_descend(self):
        """Descend for landing"""
        landing_pos = np.array([self.pos[0], self.pos[1], self.gripper_altitude])
        self.publish_setpoint(pos_sp=landing_pos)

        if abs(self.pos[2] - self.gripper_altitude) < 0.2:
            self.state = MissionState.LAND

    def _handle_land(self):
        """Final landing sequence"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.state = MissionState.MISSION_COMPLETE

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
    
    def _log_timer_callback(self):
        """Timer callback to log vehicle data"""
        if self.logger is None:
            self.get_logger().warn("logger called while not initalized")
            return

        # Log current vehicle state
        auto_flag = 0 if self.state is MissionState.INIT else 1
        
        # TODO: Set event_flag in accordance with mission regulations
        event_flag = self.mission_wp_num 
        
        gps_time = self.vehicle_gps.time_utc_usec / 1e6  # Convert microseconds to seconds
        lat = self.vehicle_gps.latitude_deg
        long = self.vehicle_gps.longitude_deg
        alt = self.vehicle_gps.altitude_ellipsoid_m

        self.logger.log_data(auto_flag, event_flag, gps_time, lat, long, alt)

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
