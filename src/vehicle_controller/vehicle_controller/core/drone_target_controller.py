import numpy as np
import math

from vehicle_controller.core.slew import SlewRate

class DroneTargetController:
    def __init__(self, 
                 target_offset: float = 0.35, 
                 target_altitude: float = 5.0,
                 acceptance_radius: float = 0.05, 
                 descend_radius: float = 0.5,
                 dt: float = 0.01,
                 # Slew rate parameters for XY movement
                 xy_v_max_default: float = 3.0,
                 xy_a_max_default: float = 1.0,
                 xy_v_max_close: float = 0.5,
                 xy_a_max_close: float = 0.1,
                 # Slew rate parameters for Z movement
                 z_v_max_default: float = 10.0,
                 z_a_max_default: float = 4.0,
                 # Distance thresholds
                 close_distance_threshold: float = 0.5,
                 far_distance_threshold: float = 2.0,
                 # Vertical tracking parameters
                 z_velocity_exp_coefficient: float = 3.0,
                 z_velocity_exp_offset: float = 0.5):
        
        self.target_offset = target_offset
        self.dt = dt
        self.target_altitude = target_altitude
        self.acceptance_radius = acceptance_radius
        self.descend_radius = descend_radius
        
        # XY slew rate parameters
        self.xy_v_max_default = xy_v_max_default
        self.xy_a_max_default = xy_a_max_default
        self.xy_v_max_close = xy_v_max_close
        self.xy_a_max_close = xy_a_max_close
        
        # Z slew rate parameters
        self.z_v_max_default = z_v_max_default
        self.z_a_max_default = z_a_max_default

        # Vertical tracking parameters
        self.z_velocity_exp_coefficient = z_velocity_exp_coefficient
        self.z_velocity_exp_offset = z_velocity_exp_offset
        
        # Distance thresholds
        self.close_distance_threshold = close_distance_threshold
        self.far_distance_threshold = far_distance_threshold
    
        # Initialize slew rate controllers
        self.slew_xy = SlewRate(dt, self.xy_v_max_default, self.xy_a_max_default)
        self.slew_z = SlewRate(dt, self.z_v_max_default, self.z_a_max_default)

    def calculate_target_position(self,
                                drone_position: np.ndarray,  # [N, E, D] in global frame
                                drone_yaw: float,  # radians
                                target_angle_x: float,  # degrees (right position)
                                target_angle_y: float) -> np.ndarray:  # degrees (forward position)
        
        drone_altitude = -drone_position[2]  # Convert to positive altitude
        horizontal_distance_x = drone_altitude * math.tan(np.deg2rad(target_angle_x))  # right distance
        horizontal_distance_y = drone_altitude * math.tan(np.deg2rad(target_angle_y))  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, drone_altitude])
        
        # Adjust for target distance behind camera
        target_local += np.array([0, 1, 0]) * self.target_offset
        
        # Rotate to NED frame using yaw
        cos_yaw = np.cos(drone_yaw)
        sin_yaw = np.sin(drone_yaw)
        
        # (x,y) to (N,E) conversion - (1,0) goes to (-sin,cos), (0,1) goes to (cos, sin)
        target_world_offset = np.array([
            -sin_yaw * target_local[0] + cos_yaw * target_local[1],  # world N
            cos_yaw * target_local[0] + sin_yaw * target_local[1],   # world E
            drone_altitude - self.target_altitude  # world D is just towards ground
        ])
        
        return target_world_offset

    def update(self,
               drone_position: np.ndarray,  # [x, y, z]
               drone_yaw: float,  # radians
               target_angle_x: float,  # degrees (right is +ve)
               target_angle_y: float) -> np.ndarray:  # degrees (forward is +ve)
        
        target_world_offset = self.calculate_target_position(
            drone_position, drone_yaw, target_angle_x, target_angle_y
        )
        
        distance_xy = np.linalg.norm(target_world_offset[:2])
        
        # Horizontal tracking - adjust slew parameters based on distance
        self.slew_xy.v_max = self.xy_v_max_default
        self.slew_xy.a_max = self.xy_a_max_default
        
        if distance_xy < self.close_distance_threshold:
            self.slew_xy.v_max = self.xy_v_max_close
            self.slew_xy.a_max = self.xy_a_max_close
            
        slewed_xy = self.slew_xy.slew_rate(drone_position[:2], 
                                          drone_position[:2] + target_world_offset[:2])
        
        # Vertical tracking - adjust velocity based on horizontal distance
        self.slew_z.v_max = self.z_v_max_default
        if distance_xy > self.close_distance_threshold:
            self.slew_z.v_max = self.z_velocity_exp_coefficient * np.exp(
                self.z_velocity_exp_offset - distance_xy
            )
        if distance_xy > self.far_distance_threshold:
            self.slew_z.v_max = 0.05

        # Vertical tracking
        slewed_z = self.slew_z.slew_rate(drone_position[2], -self.target_altitude)
        print(drone_position[2], slewed_z)
        target_pos = [*slewed_xy, slewed_z]
        
        # Check if drone is close to target position and hovering
        # NOTE: the 2nd self.acceptance_radius is in place of a maximum velocity while hovering. Maybe add a separate parameter?
        is_close = (np.linalg.norm(target_world_offset) < self.acceptance_radius and 
                   np.linalg.norm(self.slew_xy.sp_last - self.slew_xy.sp_last2) < 
                   self.acceptance_radius * self.dt)
        
        
        return target_pos, is_close

    def reset(self):
        """Reset the controller state"""
        self.slew_xy.reset()
        self.slew_z.reset()
