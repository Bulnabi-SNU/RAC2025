__author__ = "PresidentPlant"
__contact__ = ""

import os
import time
import rclpy
from rcl_interfaces.msg import SetParametersResult

import numpy as np
from enum import Enum
from typing import Optional

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from vehicle_controller.core.logger import Logger
from custom_msgs.msg import VehicleState, TargetLocation
from px4_msgs.msg import VehicleAcceleration
from scipy.spatial.transform import Rotation
from px4_msgs.msg import VehicleAttitude


class MissionState(Enum):
    INIT = "INIT"
    OFFBOARD_ARM = "OFFBOARD_ARM" #+ ascend
    MISSION_EXECUTE = "MISSION_EXECUTE" #d
    MISSION_TO_OFFBOARD_CASUALTY = "MISSION_TO_OFFBOARD_CASUALTY"#d
    CASUALTY_TRACK = "CASUALTY_TRACK"
    CASUALTY_DESCEND = "CASUALTY_DESCEND"
    GRIPPER_CLOSE = "GRIPPER_CLOSE" #d hover
    CASUALTY_ASCEND = "CASUALTY_ASCEND" #altitude change
    OFFBOARD_TO_MISSION = "OFFBOARD_TO_MISSION" #d
    MISSION_CONTINUE = "MISSION_CONTINUE" #d
    MISSION_TO_OFFBOARD_DROP_TAG = "MISSION_TO_OFFBOARD_DROP_TAG" #d 
    DROP_TAG_TRACK = "DROP_TAG_TRACK"
    DROP_TAG_DESCEND = "DROP_TAG_DESCEND" #+hover
    GRIPPER_OPEN = "GRIPPER_OPEN" #d 
    DROP_TAG_ASCEND = "DROP_TAG_ASCEND"
    MISSION_TO_OFFBOARD_LANDING_TAG = "MISSION_TO_OFFBOARD_LANDING_TAG" #d
 

    DESCEND = "DESCEND"
    HOVER = "HOVER"
    ASCEND = "ASCEND"
    MISSION_TO_OFFBOARD = "MISSION_TO_OFFBOARD"



    LANDING_TAG_TRACK = "LANDING_TAG_TRACK"
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
        MissionState.LANDING_TAG_TRACK: 3
    }

<<<<<<< HEAD
=======

>>>>>>> ac5dd9070cc89c92a40349365c503eb8669220a5
    def __init__(self):
        super().__init__("mc_main")
        
        self._load_parameters()
        self._initialize_components()
        self._setup_subscribers()
        self.vehicle_acc = VehicleAcceleration()
        self.vehicle_attitude = VehicleAttitude()

        self.state = MissionState.INIT
        self.target: Optional[TargetLocation] = None
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        self.target_position = None  # Store position when entering descend/ascend


        self.start_time=0.0
        self.end_time=0.0
        self.hover_position = None

        self.descend_waypoint_flag = 0 # 0: 시작 안함, 1: 시작함, 2:끝남
        
        
        
        self.get_logger().info("Mission Controller initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        # Mission parameters
        params = [
            ('descend_waypoint', 2),
            ('landing_tag_waypoint', 16),
            ('mission_altitude', 2.0),
            ('track_min_altitude', 4.0),
            ('gripper_altitude', 0.3),
            ('tracking_target_offset', 0.35),
            ('tracking_acceptance_radius_xy', 0.2),
            ('tracking_acceptance_radius_z', 0.1),
            ('do_logging', True),
        ]
        
        # DroneTargetController parameters
        drone_controller_params = [
            ('drone_target_controller.xy_v_max_default', 3.0),
            ('drone_target_controller.xy_a_max_default', 1.0),
            ('drone_target_controller.xy_v_max_close', 0.5),
            ('drone_target_controller.xy_a_max_close', 0.1),
            ('drone_target_controller.z_v_max_default', 10.0),
            ('drone_target_controller.z_a_max_default', 4.0),
            ('drone_target_controller.close_distance_threshold', 0.5),
            ('drone_target_controller.far_distance_threshold', 2.0),
            ('drone_target_controller.descend_radius', 0.5),
            ('drone_target_controller.z_velocity_exp_coefficient', 3.0),
            ('drone_target_controller.z_velocity_exp_offset', 0.5),
        ]
        
        # Declare all parameters
        self.declare_parameters(namespace='', parameters=params + drone_controller_params)
        
        # Cache mission parameter values
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)
        
        # Store drone controller parameters in a dict
        self.drone_controller_params = {}
        for param_name, _ in drone_controller_params:
            # Extract the actual parameter name (remove 'drone_target_controller.' prefix)
            actual_name = param_name.replace('drone_target_controller.', '')
            self.drone_controller_params[actual_name] = self.get_parameter(param_name).value

    def _initialize_components(self):
        """Initialize controllers and logger"""
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        self.logger = Logger(log_path="/workspace/flight_logs")
        self.log_timer = None
        
        self.add_on_set_parameters_callback(self.param_update_callback)
        
        # Initialize DroneTargetController with parameters from ROS
        self.drone_target_controller = DroneTargetController(
            target_offset=self.tracking_target_offset,
            target_altitude=self.track_min_altitude,
            acceptance_radius=self.tracking_acceptance_radius_xy,
            dt=self.timer_period,
            **self.drone_controller_params
        )

    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )
        self.accel_subscriber = self.create_subscription(
            VehicleAcceleration,
            "/fmu/out/vehicle_acceleration",
            self.on_vehicle_accel_update,
            self.qos_profile
        )
        self.attitude_subscriber = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.on_attitude_update,
            self.qos_profile
        )


    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            MissionState.INIT: self._handle_init,
            MissionState.OFFBOARD_ARM: self._handle_offboard_arm,
            MissionState.OFFBOARD_TO_MISSION: self._handle_mission_continue,
            MissionState.MISSION_EXECUTE: self._handle_mission_execute,
            MissionState.MISSION_TO_OFFBOARD: lambda: self._handle_mission_to_offboard(MissionState.DESCEND),
            MissionState.DESCEND: lambda: self._handle_descend_ascend(MissionState.HOVER, 0.3),
            MissionState.HOVER: lambda: self._handle_hover(MissionState.CASUALTY_ASCEND, 5.0),
            MissionState.CASUALTY_ASCEND: lambda: self._handle_descend_ascend(MissionState.MISSION_CONTINUE, 2.0),
            MissionState.MISSION_CONTINUE: self._handle_mission_continue,



            # MissionState.MISSION_TO_OFFBOARD_CASUALTY: lambda: self._handle_mission_to_offboard(MissionState.CASUALTY_TRACK),
            # MissionState.CASUALTY_TRACK: lambda: self._handle_track_target(MissionState.CASUALTY_DESCEND),
            # MissionState.CASUALTY_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_CLOSE, self.gripper_altitude),
            # MissionState.GRIPPER_CLOSE: self._handle_gripper_close,
            # MissionState.CASUALTY_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            # MissionState.MISSION_CONTINUE: self._handle_mission_continue,
            # MissionState.MISSION_TO_OFFBOARD_DROP_TAG: lambda: self._handle_mission_to_offboard(MissionState.DROP_TAG_TRACK),
            # MissionState.DROP_TAG_TRACK: lambda: self._handle_track_target(MissionState.DROP_TAG_DESCEND),
            # MissionState.DROP_TAG_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_OPEN, self.gripper_altitude),
            # MissionState.GRIPPER_OPEN: self._handle_gripper_open,
            # MissionState.DROP_TAG_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            # MissionState.MISSION_TO_OFFBOARD_LANDING_TAG: lambda: self._handle_mission_to_offboard(MissionState.LANDING_TAG_TRACK),
            # MissionState.LANDING_TAG_TRACK: lambda: self._handle_track_target(MissionState.LAND),
            # MissionState.LAND: self._handle_land,
            # MissionState.MISSION_COMPLETE: self._handle_mission_complete,
            # MissionState.ERROR: self._handle_error,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
        
        self._publish_vehicle_state()

    def param_update_callback(self, params):
        """Parameter callback for dynamically updating parameters while flying"""
        successful = True
        reason = ''
        
        for p in params:
            # Mission parameters
            if p.name == 'mission_altitude':
                self.mission_altitude = p.value
            elif p.name == 'track_min_altitude':
                self.track_min_altitude = p.value
            elif p.name == 'gripper_altitude':
                self.gripper_altitude = p.value
            elif p.name == 'tracking_target_offset':
                self.tracking_target_offset = p.value
                # Update drone target controller
                self.drone_target_controller.target_offset = p.value
            elif p.name == 'tracking_acceptance_radius_xy':
                self.tracking_acceptance_radius_xy = p.value
                self.drone_target_controller.acceptance_radius = p.value
            elif p.name == 'tracking_acceptance_radius_z':
                self.tracking_acceptance_radius_z = p.value
            elif p.name == 'casualty_waypoint' :
                self.casualty_waypoint = p.value
            elif p.name == 'drop_tag_waypoint':
                self.drop_tag_waypoint = p.value
            elif p.name == 'landing_tag_waypoint':
                self.landing_tag_waypoint = p.value
            elif p.name == "descend_waypoint":
                self.descend_waypoint = p.value
                
            # DroneTargetController parameters
            elif p.name.startswith('drone_target_controller.'):
                param_key = p.name.replace('drone_target_controller.', '')
                self.drone_controller_params[param_key] = p.value
                # Update the controller directly
                if hasattr(self.drone_target_controller, param_key):
                    setattr(self.drone_target_controller, param_key, p.value)
            else:
                self.get_logger().warn(f"Ignoring unknown parameter: {p.name}")
                continue
        
        self.get_logger().info("[Parameter Update] Mission parameters updated successfully")
        
        self.drone_target_controller.reset()  # Reset controller to apply new parameters
        self.print_current_parameters()
        return SetParametersResult(successful=successful, reason=reason)

    def print_current_parameters(self):
        """Print current parameter values for debugging"""
        self.get_logger().info("Current Mission Parameters:")
        self.get_logger().info(f"  mission_altitude: {self.mission_altitude}")
        self.get_logger().info(f"  track_min_altitude: {self.track_min_altitude}")
        self.get_logger().info(f"  gripper_altitude: {self.gripper_altitude}")
        self.get_logger().info(f"  tracking_target_offset: {self.tracking_target_offset}")
        self.get_logger().info(f"  tracking_acceptance_radius_xy: {self.tracking_acceptance_radius_xy}")
        self.get_logger().info(f"  tracking_acceptance_radius_z: {self.tracking_acceptance_radius_z}")
        self.get_logger().info(f"  casualty_waypoint: {self.casualty_waypoint}")
        self.get_logger().info(f"  drop_tag_waypoint: {self.drop_tag_waypoint}")
        self.get_logger().info(f"  landing_tag_waypoint: {self.landing_tag_waypoint}")
        self.get_logger().info("DroneTargetController Parameters:")
        for key, value in self.drone_controller_params.items():
            self.get_logger().info(f"  {key}: {value}")

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

    def on_vehicle_accel_update(self, msg: VehicleAcceleration):
        self.vehicle_acc = msg

    def on_attitude_update(self, msg: VehicleAttitude):
        self.vehicle_attitude = msg

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
            self.get_logger().info("Arming vehicle in offboard mode")
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, starting logger")
            self._start_logging()
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _start_logging(self):
        """Start flight logging if enabled"""
        if self.do_logging:
            self.logger.start_logging()
            self.get_logger().info(f"[Logger] Writing to: {os.path.abspath(self.logger.log_path)}")
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

        # Check if reached descend waypoint
        if current_wp == self.descend_waypoint:

            if self.descend_waypoint_flag == 0:
                self.descend_waypoint_flag = 1
                self.mission_paused_waypoint = current_wp
                self.state = MissionState.MISSION_TO_OFFBOARD
                self.get_logger().info(f"Transitioning into offboard. Current paused waypoint: {self.mission_paused_waypoint}")
                return
            
            if self.descend_waypoint_flag == 1:
                current_wp += self.mission_paused_waypoint
                self.state = MissionState.MISSION_CONTINUE
        
        


    def _handle_mission_to_offboard(self, next_state: MissionState):
        """Switch from mission to offboard for specific operations"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.get_logger().info(f"Transitioned into offboard. Paused waypoint: {self.mission_paused_waypoint}")
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
            self.target_position = None  # Reset for next use
            self.state = next_state

    def _handle_hover(self, next_state: MissionState, duration=10.0):
        now_sec = self.get_clock().now().nanoseconds / 1e9  # float in seconds

        if self.start_time == 0.0 and self.end_time == 0.0:
            self.start_time = now_sec
            self.end_time = self.start_time + duration
        
        if self.hover_position is None:
            self.hover_position = self.pos

        self.get_logger().info(f"Hovering for {duration} seconds")
        self.publish_setpoint(pos_sp=self.hover_position)

        if now_sec >= self.end_time:
            self.get_logger().info("Hover complete")
            self.start_time = 0.0
            self.end_time = 0.0
            self.hover_position = None
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
        gps_time = int(getattr(self.vehicle_gps, "time_utc_usec", 0))
        if gps_time <= 0:
            gps_time = self.get_clock().now().nanoseconds // 1000 


        if self.vehicle_attitude.timestamp == 0:
            return
        # --- Convert Quaternion to Euler ---
        # PX4 quaternion order is w, x, y, z. Scipy expects x, y, z, w.
        q_px4 = self.vehicle_attitude.q
        r = Rotation.from_quat([q_px4[1], q_px4[2], q_px4[3], q_px4[0]])
        
        # Get Euler angles in radians
        # Scipy default order is ZYX: roll, pitch, yaw
        roll_rad, pitch_rad, yaw_rad = r.as_euler('zyx')
        
        
        self.logger.log_data(
            auto_flag,                                          #1                                         
            self.vehicle_gps.latitude_deg,                      #2
            self.vehicle_gps.longitude_deg,                     #3                
            self.vehicle_gps.altitude_ellipsoid_m,              #4
            gps_time,                                           #5
            self.vehicle_acc.xyz[0],                            #6
            self.vehicle_acc.xyz[1],                            #7
            self.vehicle_acc.xyz[2],                            #8
            self.vehicle_local_position.ax,                     #9
            self.vehicle_local_position.ay,                     #10
            self.vehicle_local_position.az,                     #11
            roll_rad,                                           #12 
            pitch_rad,                                          #13
            yaw_rad,                                            #14
            auto_flag,                                          #15
            event_flag                                          #16
        )

    # Override methods (placeholders for additional functionality)
    def on_vehicle_status_update(self, msg): pass
    def on_local_position_update(self, msg): pass
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