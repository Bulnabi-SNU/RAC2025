__author__ = "Kyungjun Oh"
__contact__ = "frankok@snu.ac.kr"

# import rclpy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# import px4_msgs
"""msgs for subscription"""
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import SensorGps
"""msgs for publishing"""
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

# import message for auto landing, YOLOv5
from std_msgs.msg import Float32MultiArray
from my_bboxes_msg.msg import VehiclePhase
from my_bboxes_msg.msg import YoloObstacle

# import other libraries
import os
import time
import serial
import logging
import numpy as np
import pymap3d as p3d
from datetime import datetime, timedelta
import cv2

class VehicleController(Node):

    def __init__(self):
        super().__init__('vehicle_controller')

        """
        0. Configure QoS profile for publishing and subscribing
        """
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        """
        1. Constants
        """
        # given constants
        self.camera_to_center = 0.5                             # distance from camera to center of the vehicle
        self.landing_height = 5.0                               # prepare auto landing at 5m
        self.corridor_radius = 2.0

        # time period
        self.time_period = 0.05                                  # 20 Hz

        # acceptance constants
        self.mc_acceptance_radius = 0.3
        self.nearby_acceptance_radius = 30
        self.offboard_acceptance_radius = 10.0                   # mission -> offboard acceptance radius
        self.transition_acceptance_angle = 0.8                   # 0.8 rad = 45.98 deg
        self.landing_acceptance_angle = 0.8                      # 0.8 rad = 45.98 deg
        self.heading_acceptance_angle = 0.1                      # 0.1 rad = 5.73 deg

        # bezier curve constants
        self.very_fast_vmax = 7.0
        self.fast_vmax = 5.0
        self.slow_vmax = 2.5
        self.very_slow_vmax = 1.0
        self.max_acceleration = 9.81 * np.tan(10 * np.pi / 180)  # 10 degree tilt angle
        self.mc_start_speed = 0.0001
        self.mc_end_speed = 0.0001
        self.bezier_threshold_speed = 0.7
        self.bezier_minimum_time = 3.0

        # alignment constants
        self.yaw_speed = 0.1                                    # 0.1 rad = 5.73 deg

        # yolo constants
        self.image_size = np.array([1280, 720])
        self.critical_threshold = 10                            # 10 pixels
        self.yolo_hz = 10                                       # theoretically 30Hz, but 10Hz in practice
        self.quick_time = 1.0                                   # 1 seconds
        self.focus_time = 5.0                                   # 5 seconds

        # auto landing constants
        self.gimbal_time = 5.0
        self.auto_landing_height = 10.0

        # vehicle status
        self.vehicle_status_array = ['Manual', 'Altitude', 'Position', 'Mission', 'Hold', 'RTL', 'PositionSlow', 'free5', 'free4', 'free3', 'Acro', 'free2', 'descend', 'Termination', 'Offboard', 'Stablized', 'free1', 'Takeoff', 'Land']

        # emergency detection
        self.emergency_detecting_again = False
        self.align_emergency_threshold = int(60 / self.time_period)
        self.landing_align_emergency_threshold = int(60 / self.time_period)
        self.emergency_corridor_radius = 2.0
        self.emergency_obstacle_direction = 1                 # right: 1, left: -1

        """
        2. Logging setup
        """
        log_dir = os.path.join(os.getcwd(), 'src/vehicle_controller/test_nodes/log')
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = os.path.join(log_dir,  f'log_{current_time}.txt')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # initialize log info & error
        self.log_dict = {
            'auto': [],
            'subphase': [],
            'utc_time': [],
            'pos[0]' : [],
            'pos[1]': [],
            'pos[2]': [],
            'pos_gps[0]': [],
            'pos_gps[1]': [],
            'pos_gps[2]': []
        }
        self.error = [np.inf]

        """
        3. Load waypoints (GPS)
        """
        self.WP = [np.array([0.0, 0.0, 0.0])]
        self.gps_WP = [np.array([0.0, 0.0, 0.0])]
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.start_yaw = 0.0

        for i in range(1, 8):
            self.declare_parameter(f'gps_WP{i}', None)
            gps_wp_value = self.get_parameter(f'gps_WP{i}').value
            self.gps_WP.append(np.array(gps_wp_value))

        """
        4. Phase and subphase
        """
        # phase description
        # 0 : before flight
        # i >= 1 : moving toward WP_i
        # 9 : landing
        self.phase = 0

        # subphase description
        # takeoff -> heading to WP[1] -> collecting log info -> heading to WP[2] -> collecting log info -> ... -> heading to WP[7]
        # pause -> align to vertiport -> go slow -> landing align -> auto landing (obstacle not detected)
        # pause -> align to vertiport -> go slow -> pause -> align -> detecting obstacle -> avoiding obstacle -> landing align -> auto landing (obstacle detected and avoided)
        # pause -> align to vertiport -> go slow -> pause -> align -> detecting obstacle -> avoiding obstacle -> pause -> ... (obstacle detected and avoided but detected again)
        # pause -> align to vertiport -> go slow -> pause -> align -> detecting obstacle -> go slow -> ... (thought obstacle was detected but it was not)
        self.subphase = 'before flight'

        """
        5. State variables
        """
        # vehicle status
        self.auto = 0                           # 0: manual, 1: auto
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()

        # vehicle position, velocity, and yaw
        self.pos = np.array([0.0, 0.0, 0.0])        # local
        self.pos_gps = np.array([0.0, 0.0, 0.0])    # global
        self.vel = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        
        # goal position and yaw
        self.goal_position = None
        self.goal_yaw = None

        # Bezier curve
        self.num_bezier = 0
        self.bezier_counter = 0
        self.bezier_points = None

        # YOLOv5
        self.obstacle_label = ''
        self.obstacle_x = 0
        self.obstacle_y = 0
        self.left_or_right = 0
        self.ladder_count = 0
        self.time_count = 0
        self.yolo_time_count = 0
        self.yolo_wp_checker = 1
        self.yolo_WP = [np.array([0.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0])]

        # gimbal
        self.gimbal_pitch = 0.0
        self.gimbal_counter = 0
        if is_jetson():
            self.ser = serial.Serial('/dev/ttyGimbal', 115200)

        # emergency detection
        self.emergency_time_checker = 0

        # UTC time
        self.utc_time = 0.0
        self.utc_year = 0
        self.utc_month = 0
        self.utc_day = 0
        self.utc_hour = 0
        self.utc_min = 0
        self.utc_sec = 0
        self.utc_ms = 0

        """
        6. Create Subscribers
        """
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile
        )
        self.yolo_obstacle_subscriber = self.create_subscription(
            YoloObstacle, '/yolo_obstacle', self.yolo_obstacle_callback, qos_profile
        )
        self.gps_subscriber = self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position', self.vehicle_gps_callback, qos_profile
        )

        """
        7. Create Publishers
        """
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile
        )
        self.vehicle_phase_publisher = self.create_publisher(
            VehiclePhase, '/vehicle_phase', qos_profile
        )
        self.start_yaw_publisher = self.create_publisher(
            Float32MultiArray, '/auto_land_home_info', 10
        )

        """
        8. timer setup
        """
        self.offboard_heartbeat = self.create_timer(self.time_period, self.offboard_heartbeat_callback)
        self.vehicle_phase_publisher_timer = self.create_timer(self.time_period, self.vehicle_phase_publisher_callback)
        self.gimbal_control_callback_timer = self.create_timer(self.time_period, self.gimbal_control_callback)
        self.log_timer = self.create_timer(self.time_period, self.log_timer_callback)
        self.main_timer = self.create_timer(self.time_period, self.main_timer_callback)
        # self.show_to_monitor_timer = self.create_timer(self.time_period, self.show_to_monitor_callback)
        
        # self.print("Successfully executed: vehicle_controller")
        # self.print("Start the mission\n")
        self.print("Auto Latitude	Longtitude	 Altitude	  UTC Year	 UTC Month	  UTC Day	  UTC Hour	  UTC Min	  UTC Sec	   UTC ms  WPT")

    
    """
    Services
    """   
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        self.logger.info(*args, **kwargs)
    
    def convert_global_to_local_waypoint(self, home_position_gps):
        self.home_position = self.pos   # set home position
        self.start_yaw = self.yaw     # set initial yaw
        for i in range(1, 8):
            # gps_WP = [lat, lon, rel_alt]
            wp_position = p3d.geodetic2ned(self.gps_WP[i][0], self.gps_WP[i][1], self.gps_WP[i][2] + home_position_gps[2],
                                            home_position_gps[0], home_position_gps[1], home_position_gps[2])
            wp_position = np.array(wp_position)
            self.WP.append(wp_position)
        self.WP.append(np.array([0.0, 0.0, -self.landing_height]))  # landing position
        self.WP.append(np.array([-self.camera_to_center * np.cos(self.start_yaw), -self.camera_to_center * np.sin(self.start_yaw), -self.auto_landing_height]))  # set the camera's position to the home position
        # self.print(f'WP: {self.WP}\n')

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
    
    def get_bearing_to_next_waypoint(self, now, next):
        now2d = now[0:2]
        next2d = next[0:2]
        direction = (next2d - now2d) / np.linalg.norm(next2d - now2d) # NED frame
        yaw = np.arctan2(direction[1], direction[0])
        return yaw
    
    def find_indices_below_threshold(self, arr, threshold):
        return [i for i, value in enumerate(arr) if value < threshold]
    
    def intersection(self, arr1, arr2):
        return [x for x in arr1 if x in arr2]
    
    def gimbal_reboot(self):
        if is_jetson():
            data_fix = bytes([0x55, 0x66, 0x01, 0x02, 0x00, 0x00, 0x00, 0x80, 0x00, 0x01])
            data_crc = crc_xmodem(data_fix)
            packet = bytearray(data_fix + data_crc)
            self.ser.write(packet)

    """
    Callback functions for the timers
    """    
    def offboard_heartbeat_callback(self):
        """offboard heartbeat signal"""
        self.publish_offboard_control_mode(position=True)

    # YOLOv5
    def vehicle_phase_publisher_callback(self):
        msg = VehiclePhase()
        msg.phase = str(self.phase)
        msg.subphase = str(self.subphase)
        self.vehicle_phase_publisher.publish(msg)

    # Gimbal
    def gimbal_control_callback(self):
        if is_jetson():
            data_fix = bytes([0x55, 0x66, 0x01, 0x04, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00])
            data_var = to_twos_complement(10 * int(self.gimbal_pitch))
            data_crc = crc_xmodem(data_fix + data_var)
            packet = bytearray(data_fix + data_var + data_crc)
            self.ser.write(packet)

    # Logging
    # auto_states = {3, 4, 5, 14, 17, 18, 19, 20}
    def log_timer_callback(self): # auto latitude longtitude altitude year month day hour min sec ms WPT
        self.auto = int(self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_FOLLOW_TARGET \
                        or self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND)
        if self.phase in [0, 1, 7, 8, 9] :
            self.print(f"{self.auto}\t{self.pos_gps[0]:.6f}\t{self.pos_gps[1]:.6f}\t{-self.pos[2]:.6f}\t{self.utc_year}\t{self.utc_month}\t{self.utc_day}\t{self.utc_hour}\t{self.utc_min}\t{self.utc_sec}\t{self.utc_ms}\t{self.phase}")

    def show_to_monitor_callback(self):
        image = np.zeros((180,500,3),np.uint8)
        cv2.putText(image, f'Pos: [{round(self.pos[0],2)},{round(self.pos[1],2)},{round(self.pos[2],2)}]', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Vel: {round(np.linalg.norm(self.vel), 2)}. [{round(self.vel[0],2)},{round(self.vel[1],2)},{round(self.vel[2],2)}]', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'State: {self.vehicle_status_array[self.vehicle_status.nav_state]}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if (self.phase == 8) and (self.subphase == 'avoiding obstacle'):
            if self.left_or_right > 0:
                cv2.putText(image, f'Obstacle direction: right', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(image, f'Obstacle direction: left', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Vehicle Information', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def main_timer_callback(self):       
        if self.phase == 0:
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                # self.print('\n\n<< final_taean_left >>\n\n')
                # self.print("Offboard mode requested\n")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                self.convert_global_to_local_waypoint(self.pos_gps)
                self.phase = 1
                self.subphase = 'takeoff'
                # self.print('\n[phase : 0 -> 1]\n')

        elif self.phase == 1:
            if self.subphase == 'takeoff':
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self.publish_vehicle_command(
                        VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                        param1=1.0, # main mode
                        param2=6.0  # offboard
                    )
                elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.goal_yaw = self.get_bearing_to_next_waypoint(self.WP[1], self.WP[2])
                    self.goal_position = self.WP[1]
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.fast_vmax)
                    self.subphase = 'heading to WP[1]'
                    # self.print('\n[subphase : takeoff -> heading to WP[1]]\n')

            elif self.subphase == 'heading to WP[1]':
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                    if np.abs((self.yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi) < self.transition_acceptance_angle and np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                        self.publish_vehicle_command(
                            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                            param1=1.0, # main mode
                            param2=4.0, # auto
                            param3=4.0  # mission
                        )
                elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                    self.phase = 2
                    self.subphase = 'heading to WP[2]'
                    # self.print('\n[phase : 1 -> 2]')
                    # self.print('[subphase : heading to WP[1] -> heading to WP[2]]\n')

        elif self.phase in range(2, 7):
            if np.linalg.norm(self.pos - self.WP[self.phase]) >= self.nearby_acceptance_radius: # far from WP
                if len(self.log_dict['utc_time']) == 0:
                    self.print(f"{self.auto}\t{self.pos_gps[0]:.6f}\t{self.pos_gps[1]:.6f}\t{-self.pos[2]:.6f}\t{self.utc_year}\t{self.utc_month}\t{self.utc_day}\t{self.utc_hour}\t{self.utc_min}\t{self.utc_sec}\t{self.utc_ms}\t{self.phase}")
                else:
                    # horziontal error: np.linalg.norm(x error, y error)
                    # vertical error: z error
                    # self.print("--------------------------------------------")
                    # find min_idx
                    horizontal_min_indices = self.find_indices_below_threshold(self.horizontal_error, 2 )
                    vertical_min_indices = self.find_indices_below_threshold(self.vertical_error, 4)
                    min_indices = self.intersection(horizontal_min_indices, vertical_min_indices)

                    if min_indices != [] : # (2,4) -> 20 points
                        print(min_indices)
                        if len(min_indices) == 1 :
                            errors = [self.error[min_indices[0]]]
                        else :
                            errors = self.error[min_indices[0]:min_indices[-1]]
                        min_idx = self.error.index(min(errors))
                        
                        
                    else :
                        horizontal_min_indices = self.find_indices_below_threshold(self.horizontal_error, 4)
                        vertical_min_indices = self.find_indices_below_threshold(self.vertical_error, 8)
                        min_indices = self.intersection(horizontal_min_indices, vertical_min_indices)
                        if min_indices != [] :
                            print(min_indices)
                            if len(min_indices) == 1 :
                                errors = [self.error[min_indices[0]]]
                            else :
                                errors = self.error[min_indices[0]:min_indices[-1]]
                            min_idx = self.error.index(min(errors))
                        else :
                            horizontal_min_indices = self.find_indices_below_threshold(self.horizontal_error, 6)
                            vertical_min_indices = self.find_indices_below_threshold(self.vertical_error, 12)
                            min_indices = self.intersection(horizontal_min_indices, vertical_min_indices)
                            if min_indices != [] :
                                print(min_indices)
                                if len(min_indices) == 1 :
                                    errors = [self.error[min_indices[0]]]
                                else :
                                    errors = self.error[min_indices[0]:min_indices[-1]]
                                min_idx = self.error.index(min(errors))
                            else :
                                min_idx = self.error.index(min(self.error))


                    for i in range(len(self.log_dict['utc_time'])):
                        vertical_error = self.vertical_error[min_idx]
                        horizontal_error = self.horizontal_error[min_idx]
                        if i < min_idx:
                            phase = self.phase
                        else:
                            phase = self.phase + 1
                        self.print(f"{self.auto}\t{self.pos_gps[0]:.6f}\t{self.pos_gps[1]:.6f}\t{-self.pos[2]:.6f}\t{self.utc_year}\t{self.utc_month}\t{self.utc_day}\t{self.utc_hour}\t{self.utc_min}\t{self.utc_sec}\t{self.utc_ms}\t{self.phase}")


                    # self.print("--------------------------------------------")
                    # self.print(f"vertical_error: {vertical_error}, horizontal_error: {horizontal_error}")
                    # self.print("--------------------------------------------")

                    self.phase += 1
                    self.subphase = f'heading to WP[{phase}]'

                # initialize log info & error
                self.log_dict = {
                    'auto': [],
                    'subphase': [],
                    'utc_time': [],
                    'pos[0]': [],
                    'pos[1]': [],
                    'pos[2]': [],
                    'pos_gps[0]': [],
                    'pos_gps[1]': [],
                    'pos_gps[2]': []
                }
                self.vertical_error = [np.inf]
                self.horizontal_error = [np.inf]
                self.error = [np.inf]

            elif np.linalg.norm(self.pos - self.WP[self.phase]) < self.nearby_acceptance_radius: # near WP
                # log stops printing, log info still being collected in array 
                self.subphase = f'collecting log info'

                self.log_dict['auto'].append(self.auto)
                self.log_dict['subphase'].append(self.subphase)
                self.log_dict['utc_time'].append(self.utc_time)
                self.log_dict['pos[0]'].append(self.pos[0])
                self.log_dict['pos[1]'].append(self.pos[1])
                self.log_dict['pos[2]'].append(self.pos[2])
                self.log_dict['pos_gps[0]'].append(self.pos_gps[0])
                self.log_dict['pos_gps[1]'].append(self.pos_gps[1])
                self.log_dict['pos_gps[2]'].append(self.pos_gps[2]) # collecting log info

                self.vertical_error.append(np.abs(self.pos[2] - self.WP[self.phase][2]))
                self.horizontal_error.append(np.linalg.norm(np.array([self.pos[0], self.pos[1]]) - self.WP[self.phase][:2])) # collecting error info
                self.error.append(np.sqrt((self.pos[2] - self.WP[self.phase][2])**2 + (np.linalg.norm(np.array([self.pos[0], self.pos[1]]) - self.WP[self.phase][:2]))**2))
                

        elif self.phase == 7:
            if self.subphase == 'heading to WP[7]':
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                    if np.linalg.norm(self.pos - self.WP[7]) < self.offboard_acceptance_radius:
                        self.publish_vehicle_command(
                            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                            param1=1.0, # main mode
                            param2=6.0  # offboard mode
                        )
                
                elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.goal_position = self.get_braking_position(self.pos, self.vel)
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, np.linalg.norm(self.vel))
                    self.subphase = 'pause'
                    # self.print("\nOffboard control mode activated")
                    # self.print('[subphase : heading to WP[7] -> pause]\n')

            elif self.subphase == 'pause':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.gimbal_reboot()    # gimbal reboot for safety
                    self.gimbal_pitch = -15.0
                    self.goal_yaw = self.get_bearing_to_next_waypoint(self.WP[7], self.WP[8])
                    self.goal_position = self.WP[7]
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                    self.emergency_time_checker = 0
                    self.subphase = 'align to vertiport'
                    # self.print('\n[subphase : pause -> align to vertiport]\n')
            
            elif self.subphase == 'align to vertiport':
                self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                if np.abs((self.yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi) < self.heading_acceptance_angle and np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    vertical_error = np.abs(self.pos[2] - self.WP[7][2])
                    horizontal_error = np.linalg.norm(self.pos[:2] - self.WP[7][:2])
                    # self.print("--------------------------------------------")
                    # self.print(f"vertical_error: {vertical_error}, horizontal_error: {horizontal_error}")
                    # self.print("--------------------------------------------")

                    self.goal_position = self.WP[8]
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                    self.phase = 8
                    self.subphase = 'go slow'
                    # self.print('\n[phase: 7 -> 8]')
                    # self.print('[subphase : align to vertiport -> go slow]\n')
                
                else:
                    # emergency detection
                    self.emergency_time_checker += 1
                    if self.emergency_time_checker >= self.align_emergency_threshold:
                        # self.print('Yaw alignment failed')
                        self.emergency_time_checker = 0
                        self.phase = 8
                        self.subphase = 'path generated'
                        # self.print('\n[phase: 7 -> 8]')
                        # self.print('[subphase : align to vertiport -> path generated]\n')

        elif self.phase == 8:
            if self.subphase == 'go slow':
                self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                if self.obstacle_label == 'ladder':
                    self.obstacle_label = ''
                    self.ladder_count += 1
                    # self.print(f'Detected obstacle : {self.ladder_count} times')
                    if self.ladder_count >= self.yolo_hz * self.quick_time:
                        self.ladder_count = 0
                        self.goal_position = self.get_braking_position(self.pos, self.vel)
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, np.linalg.norm(self.vel))
                        self.subphase = 'pause'
                        # self.print('\n[subphase : go slow -> pause]\n')

                # if reach the WP8 without confronting obstacle
                if np.linalg.norm(self.pos - self.WP[8]) < self.mc_acceptance_radius:
                    self.goal_yaw = self.start_yaw
                    self.goal_position = self.WP[9]
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                    self.phase = 9
                    self.subphase = 'landing align'
                    # self.print('\n[phase : 8 -> 9]')
                    # self.print('[subphase : go slow -> landing align]\n')
            
            elif self.subphase == 'pause':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.gimbal_reboot()    # gimbal reboot for safety
                    self.goal_yaw = self.get_bearing_to_next_waypoint(self.WP[7], self.WP[8])
                    direction = (self.WP[8] - self.WP[7]) / np.linalg.norm(self.WP[8] - self.WP[7])
                    self.goal_position = self.WP[7] + direction * np.dot(self.pos - self.WP[7], direction)
                    self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.very_slow_vmax)
                    self.emergency_time_checker = 0
                    self.subphase = 'align'
                    # self.print('\n[subphase : pause -> align]\n')
            
            elif self.subphase == 'align':
                self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                if np.abs((self.yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi) < self.heading_acceptance_angle and np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.left_or_right = 0
                    self.ladder_count = 0
                    self.subphase = 'detecting obstacle'
                    # self.print('\n[subphase : align -> detecting obstacle]\n')
                
                else:
                    # emergency detection
                    self.emergency_time_checker += 1
                    if self.emergency_time_checker >= self.align_emergency_threshold:
                        # self.print('Yaw alignment failed')
                        self.emergency_time_checker = 0
                        self.subphase = 'path generated'
                        # self.print('[subphase : align to vertiport -> path generated]\n')
            
            elif self.subphase == 'detecting obstacle':
                self.publish_trajectory_setpoint(position_sp=self.goal_position, yaw_sp=self.goal_yaw)
                if self.time_count >= self.yolo_hz * self.focus_time:
                    obstacle_detect_ratio = self.ladder_count / self.yolo_time_count
                    if obstacle_detect_ratio >= 0.30:
                        # self.print(f'Ladder-truck Orientation: {"right" if self.left_or_right > 0 else "left"}')
                        self.goal_yaw = self.get_bearing_to_next_waypoint(self.WP[7], self.WP[8])
                        avoid_direction = np.array([np.cos(self.goal_yaw - np.sign(self.left_or_right) * np.pi/2), np.sin(self.goal_yaw - np.sign(self.left_or_right) * np.pi/2), 0.0])
                        self.yolo_WP[0] = self.pos
                        self.yolo_WP[1] = self.pos + self.corridor_radius * avoid_direction
                        self.yolo_WP[2] = self.WP[8] + self.corridor_radius * avoid_direction
                        self.yolo_WP[3] = self.WP[8]
                        self.ladder_count = 0
                        self.time_count = 0
                        self.yolo_time_count = 0
                        self.goal_position = self.yolo_WP[self.yolo_wp_checker]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.very_slow_vmax)  # very slow (not to go out of the corridor)
                        self.subphase = 'avoiding obstacle'
                        # self.print('\n[subphase : detecting obstacle -> avoiding obstacle]\n')
                    else:
                        self.ladder_count = 0
                        self.time_count = 0
                        self.yolo_time_count = 0
                        self.left_or_right = 0
                        self.goal_position = self.WP[8]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                        self.subphase = 'go slow'
                        # self.print('\n[subphase : detecting obstacle -> go slow]\n')

                else:
                    self.time_count += 1
                    if self.obstacle_label != '':
                        if self.obstacle_label == 'ladder':
                            self.ladder_count += 1
                            self.left_or_right += 1 if self.obstacle_x > (self.image_size[0] / 2) else -1
                            # self.print(f'Detected obstacle : {self.ladder_count} times, ({self.obstacle_x}, {self.obstacle_y})')
                        self.obstacle_label = ''
                        self.yolo_time_count += 1

            elif self.subphase == 'avoiding obstacle':
                # go along with ractangle path. but if you find obstacle being in front of you  =>  'pause'.
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    # self.print(f"\nyolo wp{self.yolo_wp_checker} reached\n")
                    if self.yolo_wp_checker == 2:

                        self.yolo_wp_checker = 1
                        self.goal_yaw = self.start_yaw
                        self.goal_position = self.WP[9]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.very_slow_vmax)
                        self.phase = 9
                        self.subphase = 'landing align'
                        self.emergency_time_checker = 0
                        # self.print('\n[phase : 8 -> 9]')
                        # self.print('[subphase : avoiding obstacle -> landing align]\n')
                    else:
                        self.yolo_wp_checker += 1
                        self.goal_position = self.yolo_WP[self.yolo_wp_checker]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.fast_vmax)
                else:
                    self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                    if self.obstacle_label == 'ladder' and self.yolo_wp_checker == 2 and np.sign(self.left_or_right) * (self.obstacle_x-(self.image_size[0]/2)) < self.critical_threshold:
                        self.obstacle_label = ''
                        self.ladder_count += 1
                        # self.print(f'Detected obstacle in critical section: {self.ladder_count} times')
                        if self.ladder_count >= self.yolo_hz * self.quick_time:
                            self.ladder_count = 0
                            self.yolo_wp_checker = 1
                            self.goal_position = self.get_braking_position(self.pos, self.vel)
                            self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, np.linalg.norm(self.vel))
                            if self.emergency_detecting_again:
                                self.subphase = 'path generated'
                                # self.print('\n[subphase : avoiding obstacle -> path generated]\n')
                            else:
                                self.emergency_detecting_again = True
                                self.subphase = 'pause'
                                # self.print('\n[subphase : avoiding obstacle -> pause]\n')

            elif self.subphase == 'path generated':
                self.goal_yaw = self.get_bearing_to_next_waypoint(self.WP[7], self.WP[8])
                avoid_direction = np.array([np.cos(self.goal_yaw - np.sign(self.emergency_obstacle_direction) * np.pi/2), np.sin(self.goal_yaw - np.sign(self.emergency_obstacle_direction) * np.pi/2), 0.0])
                
                direction = (self.WP[8] - self.WP[7]) / np.linalg.norm(self.WP[8] - self.WP[7])
                self.yolo_WP[0] = self.WP[7] + direction * np.dot(self.pos - self.WP[7], direction)
                self.yolo_WP[1] = self.WP[7] + direction * np.dot(self.pos - self.WP[7], direction) + self.emergency_corridor_radius * avoid_direction
                self.yolo_WP[2] = self.WP[8] + self.emergency_corridor_radius * avoid_direction
                self.yolo_WP[3] = self.WP[8]
                
                self.ladder_count = 0
                self.yolo_time_count = 0
                self.goal_position = self.yolo_WP[self.yolo_wp_checker]
                self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.very_slow_vmax)  # very slow (not to go out of the corridor)
                self.subphase = 'path following'
                # self.print('\n[subphase : emergency detected -> path following]\n')
                
            elif self.subphase == 'path following':
                self.run_bezier_curve(self.bezier_points)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    # self.print(f"\nyolo wp{self.yolo_wp_checker} reached\n")
                    if self.yolo_wp_checker == 2:
                        self.yolo_wp_checker = 1
                        self.goal_yaw = self.start_yaw
                        self.goal_position = self.WP[9]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.slow_vmax)
                        self.phase = 9
                        self.subphase = 'landing align'
                        # self.print('\n[phase : 8 -> 9]')
                        # self.print('[subphase : avoiding obstacle -> landing align]\n')
                    else:
                        self.yolo_wp_checker += 1
                        self.goal_position = self.yolo_WP[self.yolo_wp_checker]
                        self.bezier_points = self.generate_bezier_curve(self.pos, self.goal_position, self.very_fast_vmax)

        elif self.phase == 9:
            if self.subphase == 'landing align':
                self.run_bezier_curve(self.bezier_points, self.goal_yaw)
                if np.abs((self.yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi) < self.landing_acceptance_angle and np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.gimbal_reboot()    # gimbal reboot for safety
                    self.gimbal_pitch = -90.0
                    self.subphase = 'prepare landing'
                    # self.print('\n[subphase : landing align -> prepare landing]\n')
                
                # emergency detection
                self.emergency_time_checker += 1
                # self.print(f'Emergency time checker: {self.emergency_time_checker}')
                if self.emergency_time_checker >= self.landing_align_emergency_threshold:
                    self.gimbal_reboot()    # gimbal reboot for safety
                    self.gimbal_pitch = -90.0
                    self.subphase = 'prepare landing'
                    # self.print('\nLanding align failed\n')
                    # self.print('\n[subphase : landing align -> prepare landing]\n')

            elif self.subphase == 'prepare landing':
                self.gimbal_counter += 1
                self.publish_trajectory_setpoint(position_sp=self.goal_position, yaw_sp=self.goal_yaw)
                if self.gimbal_counter >= self.gimbal_time / self.time_period:
                    home_info = Float32MultiArray()
                    home_info.data = list(self.home_position) + [self.start_yaw]      # [N, E, D, yaw]
                    self.start_yaw_publisher.publish(home_info)
                    self.subphase = 'auto landing'
                    # self.print('\n[subphase : prepare landing -> auto landing]\n')

            elif self.subphase == 'auto landing':
                if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    horizontal_error = np.linalg.norm(self.pos[:2])
                    # self.print("--------------------------------------------")
                    # self.print(f"horizontal_error: {horizontal_error}")
                    # self.print("--------------------------------------------")
                    self.subphase = 'mission complete'
                    # self.print('\n[subphase : auto landing -> mission complete]\n')

            elif self.subphase == 'mission complete':
                self.print("\nMission complete")
                self.print("Congratulations!\n")
                self.destroy_node()
                rclpy.shutdown()
            

    """
    Callback functions for subscribers.
    """        
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

    def vehicle_gps_callback(self, msg):
        self.utc_time = msg.time_utc_usec
        self.utc_datetime = datetime(1970, 1, 1) + timedelta(microseconds=self.utc_time)
        self.utc_year = self.utc_datetime.year
        self.utc_month = self.utc_datetime.month
        self.utc_day = self.utc_datetime.day
        self.utc_hour = self.utc_datetime.hour
        self.utc_minute = self.utc_datetime.minute
        self.utc_sec = self.utc_datetime.second
        self.utc_ms = self.utc_datetime.microsecond // 1000  # Convert microseconds to milliseconds
    
    def yolo_obstacle_callback(self, msg):
        self.obstacle_label = msg.label
        self.obstacle_x = int(msg.x)
        self.obstacle_y = int(msg.y)

    """
    Functions for publishing topics.
    """
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
    
    def publish_trajectory_setpoint(self, **kwargs):
        msg = TrajectorySetpoint()
        # position setpoint is relative to the home position
        msg.position = list( kwargs.get("position_sp", np.nan * np.zeros(3)) + self.home_position )
        msg.velocity = list( kwargs.get("velocity_sp", np.nan * np.zeros(3)) )
        msg.yaw = kwargs.get("yaw_sp", float('nan'))
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)


"""
Functions for Gimbal Control
"""

def crc_xmodem(data: bytes) -> bytes:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc.to_bytes(2, 'little')

def to_twos_complement(number: int) -> bytes:
    if number < 0:
        number &= 0xFFFF
    return number.to_bytes(2, 'little')

def format_bytearray(byte_array: bytearray) -> str:
    return ' '.join(f'{byte:02x}' for byte in byte_array)

def is_jetson():
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        return False
    
    
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
