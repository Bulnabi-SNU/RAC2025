__author__ = "Chaewon Yun"
__contact__ = "gbll0305@snu.ac.kr"

# import rclpy: ros library
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# import px4_msgs
"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import SensorGps # for log
"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

# import custom msg

# import other libraries
import os, math
import numpy as np
# log
import logging
from datetime import datetime, timedelta
# gimbal
import serial


class VehicleController(Node):

    #============================================
    # Initialize Node
    #============================================

    def __init__(self):
        super().__init__('vehicle_controller')

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

        """
        1. Subscribers & Publishers & Timers
        """
        # [Subscribers]
        # 1. VehicleStatus          : current status of vehicle (ex. offboard)
        # 2. VehicleLocalPosition   : NED position.
        # 3. VehicleGlobalPosition  : vehicle global position (lat, lon, alt)
        self.vehicle_status_subscriber          = self.create_subscription(VehicleStatus,         '/fmu/out/vehicle_status',          self.vehicle_status_callback,          qos_profile)
        self.vehicle_local_position_subscriber  = self.create_subscription(VehicleLocalPosition,  '/fmu/out/vehicle_local_position',  self.vehicle_local_position_callback,  qos_profile) 
        self.vehicle_global_position_subscriber = self.create_subscription(VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile)

        # [Publishers]
        # 1. VehicleCommand          : vehicle command (e.g., takeoff, land, arm, disarm)
        # 2. OffboardControlMode     : send offboard signal. important to maintain offboard mode in PX4
        # 3. TrajectorySetpoint      : trajectory setpoint. send setpoint (position, velocity, acceleration)
        self.vehicle_command_publisher          = self.create_publisher(VehicleCommand,        '/fmu/in/vehicle_command',          qos_profile)
        self.offboard_control_mode_publisher    = self.create_publisher(OffboardControlMode,   '/fmu/in/offboard_control_mode',    qos_profile)
        self.trajectory_setpoint_publisher      = self.create_publisher(TrajectorySetpoint,    '/fmu/in/trajectory_setpoint',      qos_profile)
        
        # [Timers]
        # 1. offboard_heartbeat : send offboard heartbeat to maintain offboard mode
        # 2. main_timer         : main loop timer for controlling vehicle
        self.time_period = 0.05     # 20Hz
        self.offboard_heartbeat = self.create_timer(self.time_period, self.offboard_heartbeat_callback)
        self.main_timer         = self.create_timer(self.time_period, self.main_callback)


        """
        2. phase & Subphase
        """
        # [Phase discription]
        # 1. ready2flight   : vehicle is ready to take off
        # 2. takeoff        : vehicle is taking off
        # ...
        # n-1. audo_landing : aline and approach to tag
        # n. Landing        : vehicle is landing
        self.phase = 'ready2flight'

        # [Subphase description]
        # 1. takeoff : vehicle is taking off
        # ...
        self.subphase = 'none'


        """
        3. Load GPS Variables
        """
        self.home_position = np.array([0.0, 0.0, 0.0])  # set home position
        self.WP = [np.array([0.0, 0.0, 0.0])]           # waypoints, local coordinates
        self.gps_WP = [np.array([0.0, 0.0, 0.0])]       # waypoints, global coordinates

        # fetch GPS waypoints from parameters in yaml file
        self.declare_parameter(f'num_wp', None)
        num_wp = self.get_parameter(f'num_wp').value
        for i in range(1, num_wp+1):
            self.declare_parameter(f'gps_WP{i}', None)
            gps_wp_value = self.get_parameter(f'gps_WP{i}').value
            self.gps_WP.append(np.array(gps_wp_value))


        """
        4. Variables: Vehicle phase
        : vehicle informations
        """
        # vehicle status
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()

        # vehicle position, velocity, and yaw
        self.pos        = np.array([0.0, 0.0, 0.0])     # local
        self.pos_gps    = np.array([0.0, 0.0, 0.0])     # global
        self.vel        = np.array([0.0, 0.0, 0.0])
        self.yaw        = 0.0


        """
        5: Variables & Constants
        : bezier, alinement, approach, landing, etc...
        """
        # 1. Hardware constants(given)
        self.camera_to_center = 0.2     # distance from camera to center of vehicle (m)
        self.limit_acc = 9.0            # maximum acceleration (m/s^2)
        
        # 2. Goal position and yaw
        # Variables
        self.goal_position = None
        self.goal_yaw = None
        # Constants
        self.mc_acceptance_radius = 0.3
        self.nearby_acceptance_radius = 30
        self.offboard_acceptance_radius = 10.0      # mission -> offboard acceptance radius
        self.transition_acceptance_angle = 0.8      # 0.8 rad = 45.98 deg
        self.landing_acceptance_angle = 0.8         # 0.8 rad = 45.98 deg
        self.heading_acceptance_angle = 0.1         # 0.1 rad = 5.73 deg

        # 3. Bezier curve - todo
        # Variables
        self.num_bezier = 0
        self.bezier_counter = 0 
        self.bezier_points = None
        # Constants
        self.very_fast_vmax = 7.0
        self.fast_vmax = 5.0
        self.slow_vmax = 2.5
        self.very_slow_vmax = 1.0
        self.max_acceleration = 9.81 * np.tan(10 * np.pi / 180)  # 10 degree tilt angle
        self.mc_start_speed = 0.0001
        self.mc_end_speed = 0.0001
        self.bezier_threshold_speed = 0.7
        self.bezier_minimum_time = 3.0

        # 4. Auto Landing
        # todo


        """
        6. Detecting target in Image
        """
        # todo

        
        """
        6. Logging
        """
        # todo


        """
        7. Gimbal
        """
        # todo

    

    #============================================
    # Helper Functions
    #============================================

    def set_home_position(self):
        """Convert global GPS coordinates to local coordinates relative to home position"""     
        R = 6371000.0  # Earth radius in meters
        try:
            lat1 = float(os.environ.get('PX4_HOME_LAT', 0.0))
            lon1 = float(os.environ.get('PX4_HOME_LON', 0.0))
            alt1 = float(os.environ.get('PX4_HOME_ALT', 0.0))
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"Error converting environment variables: {e}")
            lat1, lon1, alt1 = 0.0, 0.0, 0.0
            
        lat2, lon2, alt2 = self.pos_gps
        
        if lat1 == 0.0 and lon1 == 0.0:
            self.get_logger().warn("No home position in environment variables, using current position")
            self.home_position = np.array([0.0, 0.0, 0.0])
            return self.home_position

        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        
        x_ned = R * (lat2 - lat1)  # North
        y_ned = R * (lon2 - lon1) * np.cos(lat1)  # East
        z_ned = -(alt2 - alt1)  # Down

        return np.array([x_ned, y_ned, z_ned])

    #============================================
    # "Subscriber" Callback Functions
    #============================================     
      
    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = msg
    
    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg
        self.pos = np.array([msg.x, msg.y, msg.z])
        self.vel = np.array([msg.vx, msg.vy, msg.vz])
        self.yaw = msg.heading

    def vehicle_global_position_callback(self, msg):
        self.vehicle_global_position = msg
        self.pos_gps = np.array([msg.lat, msg.lon, msg.alt])
    
    #============================================
    # "Publisher" Callback Functions
    #============================================     
     
    def publish_vehicle_command(self, command, **kwargs):
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = kwargs.get("param1", float('nan'))
        msg.param2 = kwargs.get("param2", float('nan'))
        msg.param3 = kwargs.get("param3", float('nan'))
        msg.param4 = kwargs.get("param4", float('nan'))
        msg.param5 = kwargs.get("param5", float('nan'))
        msg.param6 = kwargs.get("param6", float('nan'))
        msg.param7 = kwargs.get("param7", float('nan'))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)
    
    def publish_offboard_control_mode(self, **kwargs):
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
        msg = TrajectorySetpoint()
        msg.position = list(kwargs.get("pos_sp", np.nan * np.zeros(3)) + self.home_position )
        msg.velocity = list(kwargs.get("vel_sp", np.nan * np.zeros(3)))
        msg.yaw = kwargs.get("yaw_sp", float('nan'))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_local2global_setpoint(self, **kwargs):
        msg = TrajectorySetpoint()
        local_setpoint = kwargs.get("pos_sp", np.nan * np.zeros(3))
        global_setpoint = local_setpoint - self.home_position
        msg.position = list(global_setpoint)
        msg.velocity = list(kwargs.get("vel_sp", np.nan * np.zeros(3)))
        msg.yaw = kwargs.get("yaw_sp", float('nan'))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
    
    #============================================
    # "Timer" Callback Functions
    #============================================
      
    def offboard_heartbeat_callback(self):
        now = self.get_clock().now()
        if hasattr(self, 'last_heartbeat_time'):
            delta = (now - self.last_heartbeat_time).nanoseconds / 1e9
            if delta > 1.0:
                self.get_logger().warn(f"Heartbeat delay detected: {delta:.3f}s")
        self.last_heartbeat_time = now
        self.publish_offboard_control_mode(position=True, velocity=False)

    def main_callback(self):
        if self.phase == 'ready2flight':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED: # Disarm 이면
                    print("Arming...")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0) # Arming 하기
                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF) # Take off 하기, param7 = height
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                if not self.get_position_flag: # set home position 하기 전에 position topic을 받았는 지 확인
                    print("Waiting for position data")
                    return
                self.home_position = self.set_home_position() # home position 설정
                print("Taking off...")
                self.phase = 'takeoff'

        if self.phase == 'takeoff':
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: # Take off 끝나면 Auto Loiter로 바뀜
                if (self.hold_timer > self.hold_timer_threshold):
                    print("landing")
                    self.phase = 'Landing'
                else:
                    if self.hold_timer%100 == 0:
                        print("holding...")
                    self.hold_timer += 1

        if self.phase == 'Landing':
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, param7=0.0) # 착륙
            print("Mission complete")
    





def main(args = None):
    rclpy.init(args=args)

    vehicle_controller = VehicleController()
    rclpy.spin(vehicle_controller)

    vehicle_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)