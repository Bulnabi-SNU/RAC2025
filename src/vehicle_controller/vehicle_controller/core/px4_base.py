"""
Base PX4 Controller Class
Handles all the boilerplate ROS2 and PX4 communications
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


# PX4 Messages

"""Messages for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import SensorGps  # for log file, not used anywhere else
from px4_msgs.msg import MissionResult

"""Messages for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import GimbalManagerSetAttitude

import numpy as np
from abc import ABC, abstractmethod

# GPS coordinate (LLA) to Local (NED) conversion
import pymap3d as p3d

# Custom Message
from custom_msgs.msg import VehicleState


class PX4BaseController(Node, ABC):
    """
    Base class for PX4 controllers that handles all the boilerplate
    ROS2 and PX4 communication setup.
    """

    def __init__(self, node_name: str, timer_period: float = 0.01):
        super().__init__(node_name)

        self.timer_period = timer_period

        # Configure QoS profile
        self._setup_qos()

        self._init_state_variables()

        self._create_subscribers()
        self._create_publishers()

        self._setup_timers()
        
        self.get_logger().info(f"{node_name} initialized")

    # =======================================
    # Setup functions (__init__)
    # =======================================

    def _setup_qos(self):
        """Configure QoS profile for publishing and subscribing"""
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

    def _init_state_variables(self):
        """Initialize all state variables"""
        # Vehicle status
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_global_position = VehicleGlobalPosition()

        self.offboard_control_mode_params = {
            "position": True,
            "velocity": False,
            "acceleration": False,
            "attitude": False,
            "body_rate": False,
            "thrust_and_torque": False,
            "direct_actuator": False,
        }

        # Position, velocity, and yaw
        self.pos = np.array([0.0, 0.0, 0.0])  # local NED coordinates (0 = home (usually takeoff point))
        self.pos_gps = np.array([0.0, 0.0, 0.0])  # global GPS coordinates (LLA)
        self.vel = np.array([0.0, 0.0, 0.0]) # NED coordinates
        self.yaw = 0.0 # Radians
        self.attitude_q = np.zeros(4)
        
        self.vehicle_gps = None

        # Home position flags
        self.get_position_flag = False 
        self.home_set_flag = False
        
        # Home position state: (basically the "zero point" of the drone)
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.home_position_gps = np.array([0.0, 0.0, 0.0])
        self.home_yaw = 0.0

        # Offboard setpoint publishing
        self.blocking_setpoint_publish = False

        # Mission variables
        self.mission_result = None
        self.mission_wp_num = None

        # Heartbeat timing
        self.last_heartbeat_time = None

    def _create_subscribers(self):
        """Create all ROS2 subscribers"""
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._vehicle_status_callback,
            self.qos_profile,
        )

        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._vehicle_local_position_callback,
            self.qos_profile,
        )

        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._vehicle_attitude_callback,
            self.qos_profile,
        )

        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self._vehicle_global_position_callback,
            self.qos_profile,
        )
        
        self.vehicle_gps_subscriber = self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self._vehicle_gps_callback,
            self.qos_profile,
        )

        self.vehicle_mission_subscriber = self.create_subscription(
            MissionResult,
            "/fmu/out/mission_result",
            self._mission_result_callback,
            self.qos_profile,
        )

    def _create_publishers(self):
        """Create all ROS2 publishers"""
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", self.qos_profile
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", self.qos_profile
        )

        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.qos_profile
        )

        self.vehicle_state_publisher = self.create_publisher(
            VehicleState, "/vehicle_state", self.qos_profile
        )
        
        self.gimbal_manager_set_attitude_publisher = self.create_publisher(
            GimbalManagerSetAttitude, "/fmu/in/gimbal_manager_set_attitude", 
            self.qos_profile
        )

    def _setup_timers(self):
        """Setup ROS2 timers"""
        self.offboard_heartbeat = self.create_timer(
            self.timer_period, self._offboard_heartbeat_callback
        )

        self.main_timer = self.create_timer(self.timer_period, self._main_timer_callback)

    # =======================================
    # Timer Callback functions
    # =======================================

    def _offboard_heartbeat_callback(self):
        """Heartbeat callback to maintain offboard mode"""
        now = self.get_clock().now()
        if self.last_heartbeat_time is not None:
            # TODO: are heartbeats really 1ns apart?
            delta = (now - self.last_heartbeat_time).nanoseconds / 1e9
            if delta > 1.0:
                self.get_logger().warn(f"Heartbeat delay detected: {delta:.3f}s")
        self.last_heartbeat_time = now

        # Publish offboard control mode (can be overridden via setting offboard_control_mode_params)
        self.publish_offboard_control_mode(**self.offboard_control_mode_params)

    def _main_timer_callback(self):
        """Main timer callback - calls the abstract main_loop method"""
        try:
            self.main_loop()
        except Exception as e:
            self.get_logger().error(f"Error in main loop: {e}")

    @abstractmethod
    def main_loop(self):
        """
        Abstract method to be implemented by subclasses.
        This is where the main control logic should go.
        """
        pass

    # =======================================
    # Subscriber Callback functions
    # =======================================

    def _vehicle_status_callback(self, msg):
        """Callback for vehicle status updates"""
        self.vehicle_status = msg
        self.on_vehicle_status_update(msg)

    def _vehicle_local_position_callback(self, msg):
        """Callback for local position updates"""
        self.vehicle_local_position = msg
        self.pos = np.array([msg.x, msg.y, msg.z])
        self.vel = np.array([msg.vx, msg.vy, msg.vz])
        self.yaw = msg.heading
         
        # If home is set then make self.pos relative to home,
        # instead of the EKF2-module start location, except for altitude.
        if self.get_position_flag and self.home_set_flag:
            self.pos[:2] = self.pos[:2] - self.home_position[:2]
        
        self.on_local_position_update(msg)

    def _vehicle_attitude_callback(self, msg):
        """Callback for attitude updates"""
        self.attitude_q = msg.q
        self.on_attitude_update(msg)

    def _vehicle_global_position_callback(self, msg):
        """Callback for global position updates"""
        self.get_position_flag = True
        self.vehicle_global_position = msg
        self.pos_gps = np.array([msg.lat, msg.lon, msg.alt])
        self.on_global_position_update(msg)
        

    def _vehicle_gps_callback(self, msg):
        """Callback for GPS updates"""
        if msg.fix_type >= 3:
            # Check if GPS has at least FIX_TYPE_3D
            self.vehicle_gps = msg
        else:
            self.get_logger().warn("GPS fix type is less than 3, no valid GPS data")          

    def _mission_result_callback(self, msg):
        # This is called only when the waypoint changes.
        self.mission_result = msg
        self.mission_wp_num = msg.seq_current

    # =======================================
    # Additional Overridable Callback Functions
    # =======================================

    # Probably won't be used at all, but keep them for now just in case?
    
    def on_vehicle_status_update(self, msg):
        """Override to handle vehicle status updates"""
        # Could add additional status monitoring here
        pass

    def on_local_position_update(self, msg):
        """Override to handle local position updates"""
        # Could add position monitoring here
        pass

    def on_attitude_update(self, msg):
        """Override to handle attitude updates"""
        # Could add GPS monitoring here
        pass

    def on_global_position_update(self, msg):
        """Override to handle global position updates"""
        # Could add GPS monitoring here
        pass

    # =======================================
    # Publisher Callback functions
    # =======================================

    def publish_vehicle_command(self, command: int, **kwargs):
        """Publish a vehicle command (One of the Vehicle_CMD Mavlink Messages)"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = kwargs.get("param1", float("nan"))
        msg.param2 = kwargs.get("param2", float("nan"))
        msg.param3 = kwargs.get("param3", float("nan"))
        msg.param4 = kwargs.get("param4", float("nan"))
        msg.param5 = kwargs.get("param5", float("nan"))
        msg.param6 = kwargs.get("param6", float("nan"))
        msg.param7 = kwargs.get("param7", float("nan"))
        
        # Since this is not a GCS but an onboard computer, it's sending the command to itself.
        # (if MAV_SYS_ID is set != 1 this will need to be changed.)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def publish_offboard_control_mode(self, **kwargs):
        """Publish offboard control mode"""
        msg = OffboardControlMode()
        msg.position = kwargs.get("position", False)
        msg.velocity = kwargs.get("velocity", False)
        msg.acceleration = kwargs.get("acceleration", False)
        msg.attitude = kwargs.get("attitude", False)
        msg.body_rate = kwargs.get("body_rate", False)
        msg.thrust_and_torque = kwargs.get("thrust_and_torque", False)
        msg.direct_actuator = kwargs.get("direct_actuator", False)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_setpoint(self, **kwargs):
        """Publish trajectory setpoint (relative to home position)"""
        if not self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().warn(
                "Blocking offboard commands while not in offboard mode."
            )
            return
        
        msg = TrajectorySetpoint()
        msg.position = list(
            kwargs.get("pos_sp", np.nan * np.zeros(3)) + self.home_position
        )
        msg.velocity = list(kwargs.get("vel_sp", np.nan * np.zeros(3)))
        msg.yaw = kwargs.get("yaw_sp", float("nan"))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_gimbal_attitude(self, **kwargs):
        msg = GimbalManagerSetAttitude()
        
        msg.origin_sysid = kwargs.get("origin_sysid", 0)
        msg.origin_compid = kwargs.get("origin_compid", 0)
        msg.target_system = kwargs.get("target_system", 0)
        msg.target_component = kwargs.get("target_component", 0) # NOTE: FOR SIYI USE 154 INSTEAD # NOTE: NEVER MIND. FIXED WITH NEW FIRMWARE
        
        # Only set roll and pitch lock
        msg.flags = kwargs.get("flags",12 )
        
        msg.gimbal_device_id = kwargs.get("gimbal_device_id", 0)
        
        msg.q = list(kwargs.get("q", np.nan * np.zeros(4)))

        msg.angular_velocity_x = kwargs.get("angular_velocity_x", float("nan"))
        msg.angular_velocity_y = kwargs.get("angular_velocity_y", float("nan"))
        msg.angular_velocity_z = kwargs.get("angular_velocity_z", float("nan"))

        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.gimbal_manager_set_attitude_publisher.publish(msg)
        
    # TODO: Add helper function for MAV_CMD_DO_SET_ACTUATOR
    # or at least document usage (https://mavlink.io/en/messages/common.html#MAV_CMD_DO_SET_ACTUATOR)

    # =======================================
    # Set Home position functions
    # =======================================

    def set_home_position(self):
        if not self.get_position_flag:
            self.get_logger().warn("Global position hasn't been received yet. Skipping")
            return
        
        self.home_position = self.pos
        self.home_position_gps = self.pos_gps
        self.home_yaw = self.yaw
        self.home_set_flag = True

    def set_gps_to_local(self, num_wp, gps_WP, home_gps_pos):
        """
        Convert GPS waypoints to local coordinates relative to home position.
        input: num_wp, gps_WP, home_gps_pos
        output: local_wp array
        """
        # home_gps_pos = currunt self.pos
        if self.home_set_flag:
            # convert GPS waypoints to local coordinates relative to home position
            local_wp = []
            for i in range(0, num_wp):
                # gps_WP = [lat, lon, rel_alt]
                wp_position = p3d.geodetic2ned(
                    self.gps_WP[i][0],
                    self.gps_WP[i][1],
                    self.gps_WP[i][2] + home_gps_pos[2],
                    home_gps_pos[0],
                    home_gps_pos[1],
                    home_gps_pos[2],
                )
                wp_position = np.array(wp_position)
                local_wp.append(wp_position)
            return local_wp
        else:
            self.get_logger().error(
                "Home position not set. Cannot convert GPS to local coordinates."
            )
            return None

    # =======================================
    # Utility methods
    # check & change vehicle mode
    # =======================================

    # Check Current Status

    def is_armed(self):
        """Check if vehicle is armed"""
        return self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def is_disarmed(self):
        """Check if vehicle is disarmed"""
        return self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED

    def is_offboard_mode(self):
        """Check if vehicle is in offboard mode"""
        return self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def is_mission_mode(self):
        """Check if vehicle is in mission mode"""
        return (
            self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION
        )

    def is_auto_takeoff(self):
        """Check if vehicle is in auto takeoff mode"""
        return (
            self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF
        )

    def is_auto_loiter(self):
        """Check if vehicle is in auto loiter mode"""
        return (
            self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER
        )

    # Change Current Status

    def arm(self):
        """Arm the vehicle"""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )

    def disarm(self):
        """Disarm the vehicle"""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0
        )

    # Note: The numbers corresponding to the custom modes are all defined in src/modules/commander/px4_custom_mode.h

    def set_offboard_mode(self):
        """Set vehicle to offboard mode"""
        if not self.is_offboard_mode():
            # param1: mode flags (CUSTOM_MODE_ENABLED)
            # param2: OFFBOARD mode
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
            )

    def set_mission_mode(self):
        """Set vehicle to mission mode"""
        if not self.is_mission_mode():
            # param1: mode flags (CUSTOM_MODE_ENABLED)
            # param2: AUTO mode 
            # param3: AUTO_MISSION submode
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=4.0,
                param3=4.0,
            )

    def takeoff(self, altitude: float = None):
        """Command takeoff"""
        if altitude is not None:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=altitude
            )
        else:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF)

    def land(self, altitude: float = 0.0):
        """Command landing"""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND, param7=altitude
        )

    ## TODO: Add checks for parameters. This is completely awful non-checkable code.
    def set_actuator(self,actuator_nubmer:int = 2, actuator_value: float=0.0, actuator_index: int=0):
        params = {f"param{actuator_nubmer}":actuator_value,
                  "param7":actuator_index}        
        self.publish_vehicle_command(**params)
