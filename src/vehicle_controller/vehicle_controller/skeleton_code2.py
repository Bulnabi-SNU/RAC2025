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
# from custom_msgs.msg import VehiclePhase

# import other libraries
import os, math
import numpy as np
# gps
import pymap3d as p3d
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
        self.phase = 0

        # [Subphase description]
        # 1. takeoff : vehicle is taking off
        # ...
        self.subphase = 'none'


        """
        3. Load GPS Variables
        """
        # take off -> update
        self.home_position = np.array([0.0, 0.0, 0.0])  # set home position -> local position
        self.start_yaw = 0.0                            # set start yaw
        self.WP = [np.array([0.0, 0.0, 0.0])]           # waypoints, local coordinates. 1~num_wp
        self.gps_WP = [np.array([0.0, 0.0, 0.0])]       # waypoints, global coordinates. 1~num_wp
        self.WP_tag = np.array([0.0, 0.0, 0.0])         # WP for recognizing tag, (not used yet)

        # fetch GPS waypoints from parameters in yaml file
        self.declare_parameter(f'num_wp', None)
        self.num_wp = self.get_parameter(f'num_wp').value
        for i in range(1, self.num_wp+1):
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

    def set_home_local_position(self, home_gps_pos):
        # 'start point' of mission in local coordinates
        self.home_position = self.pos
        self.start_yaw = self.yaw

        # convert GPS waypoints to local coordinates relative to home position
        for i in range(len(self.num_wp)):
            # gps_WP = [lat, lon, rel_alt]
            wp_position = p3d.geodetic2ned(self.gps_WP[i][0], self.gps_WP[i][1], self.gps_WP[i][2] + home_gps_pos[2],
                                            home_gps_pos[0], home_gps_pos[1], home_gps_pos[2])
            wp_position = np.array(wp_position)
            self.WP.append(wp_position)

    def generate_bezier_curve(self, xi, xf, vmax):
        # reset counter
        self.bezier_counter = 0

        # total time calculation
        total_time = np.linalg.norm(xf - xi) / vmax * 2      # Assume that average velocity = vmax / 2.     real velocity is lower then vmax
        if total_time <= self.bezier_minimum_time:
            total_time = self.bezier_minimum_time

        direction = np.array((xf - xi) / np.linalg.norm(xf - xi))
        vf = self.mc_end_speed * direction
        if np.linalg.norm(self.vel) < self.bezier_threshold_speed:
            vi = self.mc_start_speed * direction
        else:
            vi = self.vel
        self.bezier_counter = int(1 / self.time_period) - 1

        point1 = xi
        point2 = xi + vi * total_time / 3
        point3 = xf - vf * total_time / 3
        point4 = xf

        # Bezier curve
        self.num_bezier = int(total_time / self.time_period)
        bezier = np.linspace(0, 1, self.num_bezier).reshape(-1, 1)
        bezier = point4 * bezier**3 +                             \
                3 * point3 * bezier**2 * (1 - bezier) +           \
                3 * point2 * bezier**1 * (1 - bezier)**2 +        \
                1 * point1 * (1 - bezier)**3
        
        return bezier
    
    def run_bezier_curve(self, bezier_points, goal_yaw=None):
        if goal_yaw is None:
            goal_yaw = self.yaw
        
        if self.bezier_counter < self.num_bezier:
            self.publish_trajectory_setpoint(
                position_sp = bezier_points[self.bezier_counter],
                yaw_sp = self.yaw + np.sign(np.sin(goal_yaw - self.yaw)) * self.yaw_speed
            )
            self.bezier_counter += 1
        else:
            self.publish_trajectory_setpoint(
                position_sp = bezier_points[-1],        # last point (goal position)
                yaw_sp = self.yaw + np.sign(np.sin(goal_yaw - self.yaw)) * self.yaw_speed
            )

    def get_braking_position(self, pos, vel):
        braking_distance = (np.linalg.norm(vel))**2 / (2 * self.max_acceleration)
        return pos + braking_distance * vel / np.linalg.norm(vel)



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
        if self.phase != -1:
            # set position relative to the home position after takeoff
            self.pos = self.pos - self.home_position

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
        if self.phase == 0:
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Arming...")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF) # Take off
                    self.set_home_local_position(self.pos_gps) # set home position & convert wp gps -> local
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                print("Taking off...")
                self.phase = 1
                self.subphase = 'takeoff'

        if self.phase == 1:
            if self.subphase == 'takeoff':
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: 
                    # Take off -> Auto Loiter
                    self.publish_vehicle_command(
                            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                            param1=1.0, # main mode
                            param2=6.0  # offboard
                        )
                elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.subphase = 'generate_path'
            
            elif self.subphase == 'generate_path':
                self.goal_position = self.WP[1]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                self.subphase = 'run_path'
                
            elif self.subphase == 'run_path':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.phase = 2
                    self.subphase = 'generate_path'
        
        if self.phase == 2:
            if self.subphase == 'generate_path':
                self.goal_position = self.WP[2]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                self.subphase = 'run_path'
                
            elif self.subphase == 'run_path':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.phase = 3
                    self.subphase = 'generate_path'
        
        if self.phase == 3:
            if self.subphase == 'generate_path':
                self.goal_position = self.WP[3]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                self.subphase = 'run_path'
                
            elif self.subphase == 'run_path':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.phase = 4
                    self.subphase = 'generate_path'
        
        if self.phase == 4:
            if self.subphase == 'generate_path':
                self.goal_position = self.WP[4]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                self.subphase = 'run_path'
                
            elif self.subphase == 'run_path':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.phase = 5
                    self.subphase = 'generate_path'

        if self.phase == 5:
            if self.subphase == 'generate_path':
                self.goal_position = self.WP[1]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                self.subphase = 'run_path'
                
            elif self.subphase == 'run_path':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.subphase = 'landing'

            elif self.subphase == 'landing':
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, param7=0.0) # 착륙
                print("Mission complete")



#============================================
# Additional Functions
# (e.g., Gimbal, jetson, etc.)
#============================================
def is_jetson():
    # if jetson, return True
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        return False



#============================================
# Main
#============================================

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