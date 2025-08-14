__author__ = "PresidentPlant"
__contact__ = ""

import time
import rclpy
import numpy as np
from enum import Enum
from typing import Optional

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from vehicle_controller.core.logger import Logger
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
    """Mission Controller for the actual competition."""

    # Constants
    TARGET_TYPES = {
        MissionState.CASUALTY_TRACK: 1,
        MissionState.CASUALTY_DESCEND: 1,
        MissionState.GRIPPER_CLOSE: 1,
        MissionState.CASUALTY_ASCEND: 1,
        MissionState.DROP_TAG_TRACK: 2,
        MissionState.DROP_TAG_DESCEND: 2,
        MissionState.GRIPPER_OPEN: 2,
        MissionState.DROP_TAG_ASCEND: 2,
        MissionState.LANDING_TAG_TRACK: 3,
        MissionState.FINAL_DESCEND: 3,
    }

    def __init__(self):
        super().__init__("mc_main")
        
        self._load_parameters()
        self._initialize_components()
        self._setup_subscribers()
        
        self.state = MissionState.INIT
        self.target: Optional[TargetLocation] = None
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        self.target_position = None  # Store position when entering descend/ascend
        
        self.get_logger().info("Mission Controller initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        params = [
            ('casualty_waypoint', 14),
            ('drop_tag_waypoint', 15),
            ('landing_tag_waypoint', 16),
            ('mission_altitude', 15.0),
            ('track_min_altitude', 4.0),
            ('gripper_altitude', 0.3),
            ('tracking_target_offset', 0.35),
            ('tracking_acceptance_radius_xy', 0.2),
            ('tracking_acceptance_radius_z', 0.2),
            ('do_logging', True),
        ]
        
        self.declare_parameters(namespace='', parameters=params)
        
        # Cache parameter values
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)

    def _initialize_components(self):
        """Initialize controllers and logger"""
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        self.logger = Logger(log_path="./flight_logs/")
        self.log_timer = None
        
        self.drone_target_controller = DroneTargetController(
            target_offset=self.tracking_target_offset,
            target_altitude=self.track_min_altitude,
            acceptance_radius=self.tracking_acceptance_radius_xy
        )

    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )

    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            MissionState.INIT: self._handle_init,
            MissionState.OFFBOARD_ARM: self._handle_offboard_arm,
            MissionState.OFFBOARD_TO_MISSION: self._handle_mission_continue,
            MissionState.MISSION_EXECUTE: self._handle_mission_execute,
            MissionState.MISSION_TO_OFFBOARD_CASUALTY: lambda: self._handle_mission_to_offboard(MissionState.CASUALTY_TRACK),
            MissionState.CASUALTY_TRACK: lambda: self._handle_track_target(MissionState.CASUALTY_DESCEND),
            MissionState.CASUALTY_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_CLOSE, self.gripper_altitude),
            MissionState.GRIPPER_CLOSE: self._handle_gripper_close,
            MissionState.CASUALTY_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            MissionState.MISSION_CONTINUE: self._handle_mission_continue,
            MissionState.MISSION_TO_OFFBOARD_DROP_TAG: lambda: self._handle_mission_to_offboard(MissionState.DROP_TAG_TRACK),
            MissionState.DROP_TAG_TRACK: lambda: self._handle_track_target(MissionState.DROP_TAG_DESCEND),
            MissionState.DROP_TAG_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_OPEN, self.gripper_altitude),
            MissionState.GRIPPER_OPEN: self._handle_gripper_open,
            MissionState.DROP_TAG_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            MissionState.MISSION_TO_OFFBOARD_LANDING_TAG: lambda: self._handle_mission_to_offboard(MissionState.LANDING_TAG_TRACK),
            MissionState.LANDING_TAG_TRACK: lambda: self._handle_track_target(MissionState.LAND),
            MissionState.LAND: self._handle_land,
            MissionState.MISSION_COMPLETE: self._handle_mission_complete,
            MissionState.ERROR: self._handle_error,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
        
        self._publish_vehicle_state()

    def _publish_vehicle_state(self):
        """Publish current vehicle state"""
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=self.TARGET_TYPES.get(self.state, 0)
            )
        )

    def on_target_update(self, msg: TargetLocation):
        """Callback for target coordinates from image_processing_node"""
        self.target = msg if msg is not None else None

    # =======================================
    # State Machine Handlers
    # =======================================

    def _handle_init(self):
        """Initialize system and check status"""
        if not self.get_position_flag:
            self.get_logger().info("Waiting for global position data...")
            return

        self.set_home_position()
        
        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = MissionState.OFFBOARD_ARM

    def _handle_offboard_arm(self):
        """Wait for offboard mode, then arm / start the mission"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, starting logger")
            self._start_logging()
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _start_logging(self):
        """Start flight logging if enabled"""
        if self.do_logging:
            self.logger.start_logging()
            self.log_timer = self.create_timer(0.1, self._log_timer_callback)

    def _handle_mission_continue(self):
        """Resume mission mode or continue from paused waypoint"""
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.get_logger().info(f"Resuming mission from waypoint {self.mission_paused_waypoint}")
            self.target = None
            self.state = MissionState.MISSION_EXECUTE

    def _handle_mission_execute(self):
        """Execute mission"""
        if not self.is_mission_mode():
            self.get_logger().warn("Not in mission mode, cannot execute mission")
            return

        current_wp = self.mission_wp_num

        # Check if reached pickup waypoint
        if current_wp == self.casualty_waypoint and not self.pickup_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_CASUALTY
            return

        # Check if reached dropoff waypoint
        if current_wp == self.drop_tag_waypoint and not self.dropoff_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_DROP_TAG
            return

        # Check if reached landing waypoint
        if current_wp == self.landing_tag_waypoint:
            self.state = MissionState.MISSION_TO_OFFBOARD_LANDING_TAG
            return

    def _handle_mission_to_offboard(self, next_state: MissionState):
        """Switch from mission to offboard for specific operations"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = next_state

    def _handle_track_target(self, next_state: MissionState):
        """Track target using vision and transition to next state when arrived"""
        if self.target is None or self.target.status != 0:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return

        target_pos, arrived = self.drone_target_controller.update(
            self.pos, self.yaw, self.target.angle_x, self.target.angle_y
        )

        self.publish_setpoint(pos_sp=target_pos)

        if arrived:
            self.drone_target_controller.reset()
            self.state = next_state

    def _handle_descend_ascend(self, next_state: MissionState, target_altitude: float):
        """Descend to target position"""
        if self.target_position is None:
            self.target_position = np.array([self.pos[0], self.pos[1], -target_altitude])
        
        self.publish_setpoint(pos_sp=self.target_position)

        # Assume drone can hold position well. If not, add checking for acceptance radius xy
        if abs(self.pos[2] - self.target_position[2]) < self.tracking_acceptance_radius_z:
            self.publish_setpoint(pos_sp=self.pos)  # Hold current position
            self.target_position = None  # Reset for next use
            self.state = next_state

    def _handle_gripper_close(self):
        """Close gripper to pick up casualty"""
        self.get_logger().info("Closing gripper to pick up casualty")
        # TODO: Implement gripper control
        
        self.pickup_complete = True
        self.state = MissionState.CASUALTY_ASCEND

    def _handle_gripper_open(self):
        """Open gripper to release casualty at dropoff point"""
        self.get_logger().info("Opening gripper to release casualty at dropoff point")
        # TODO: Implement gripper control
        
        self.dropoff_complete = True
        self.state = MissionState.DROP_TAG_ASCEND

    def _handle_land(self):
        """Final landing sequence"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.get_logger().info("Mission Complete!")
        self.log_timer.cancel() if self.log_timer else None
        self.state = MissionState.MISSION_COMPLETE

    def _handle_mission_complete(self):
        """Mission finished"""
        pass
        
    def _handle_error(self):
        """Error handling state"""
        self.get_logger().error("Mission in error state")
        # TODO: Implement error recovery or emergency procedures

    def _log_timer_callback(self):
        """Timer callback to log vehicle data"""
        if self.logger is None:
            self.get_logger().warn("Logger called while not initialized")
            return

        auto_flag = 0 if self.state is MissionState.INIT else 1
        event_flag = self.mission_wp_num
        gps_time = self.vehicle_gps.time_utc_usec / 1e6
        
        self.logger.log_data(
            auto_flag, event_flag, gps_time,
            self.vehicle_gps.latitude_deg,
            self.vehicle_gps.longitude_deg,
            self.vehicle_gps.altitude_ellipsoid_m
        )

    # Override methods (placeholders for additional functionality)
    def on_vehicle_status_update(self, msg): pass
    def on_local_position_update(self, msg): pass
    def on_attitude_update(self, msg): pass
    def on_global_position_update(self, msg): pass


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    controller = None

    try:
        controller = MissionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Mission interrupted by user")
    except Exception as e:
        print(f"Mission failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()