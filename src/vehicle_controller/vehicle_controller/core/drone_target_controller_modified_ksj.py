import numpy as np
import math
from typing import Tuple, Optional

class DroneTargetController:
    """
    Simplified controller for positioning drone to keep target 0.35m behind camera at 3m altitude
    Assumes target is on the ground and camera faces straight down.
    """
    
    def __init__(self, target_distance: float = 0.35, target_altitude: float = 5.0, acceptance_radius: float = 0.1):
        """
        Initialize the controller
        
        Args:
            target_distance: Desired distance behind camera (meters)
            target_altitude: Desired drone altitude (meters)
            acceptance_radius: Radius within which the drone considers itself close enough to the target
        """
        
        # Initialize parameters
        self.target_distance = target_distance
        self.target_altitude = target_altitude
        self.acceptance_radius = acceptance_radius
        
        # Previous target positions
        self.prev_desired_xy: Optional[np.ndarray] = None  # Previous target position for horizontal slew
        self.prev_desired_z: Optional[float] = None  # Previous target altitude for vertical
        
        # Slew-rate parameters
        self.max_horizontal_speed = 1  # Maximum horizontal speed in m/s
        self.max_vertical_speed = 1 # Maximum vertical speed in m/s
        self.dt = 0.2  # Time step in seconds for control updates
    
    def calculate_target_position(self, 
                                drone_position: np.ndarray,  # [N, E, D] in global frame
                                drone_yaw: float,            # radians
                                target_angle_x: float,       # degrees (right position)
                                target_angle_y: float) -> np.ndarray:  # degrees (forward position)
        """
        Calculate the target position for the drone
        
        Args:
            drone_position: Current drone position [N, E, D] in global frame
            drone_yaw: Current drone yaw angle (radians)    
            target_angle_x: Target angle in x direction (degrees, right positive)
            target_angle_y: Target angle in y direction (degrees, forward positive)
            
        Returns:
            target_local: Target position in drone's local frame [x, y, z]
            target_world_offset: Target position offset in global frame [N, E, D]
            
        Note: Assumes target is on the ground (z=0) and camera faces straight down
        """
        
        # Check if detection angles are valid, if not valid angles, set drone altitude to 10m
        if np.isnan(target_angle_x):
            print("Target missing, going up")
            drone_position[2] = -10 # NED
            return drone_position

        drone_altitude = -drone_position[2]  # Convert to positive altitude
        horizontal_distance_x = drone_altitude * math.tan(np.deg2rad(target_angle_x))  # right distance
        horizontal_distance_y = drone_altitude * math.tan(np.deg2rad(target_angle_y))  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, -drone_altitude])
        
        # Adjust for target distance behind camera
        target_local += np.array([0, 1, 0]) * self.target_distance
        
        # Rotate to NED frame using yaw
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        
        # (x,y) to (N,E) conversion - (1,0) goes to (-sin,cos), (0,1) goes to (cos, sin)
        target_world_offset = np.array([
             -sin_yaw * target_local[0] + cos_yaw * target_local[1],  # world N
             cos_yaw * target_local[0] + sin_yaw * target_local[1],  # world E
              - 3.0 + drone_altitude                                         # world D is  just towards ground
        ])
        
        return target_world_offset
    
    def limit_vel(self, target_world_offset_orig):

        target_world_offset = target_world_offset_orig.copy()

        xy_delta = np.linalg.norm(target_world_offset[:2])
        z_delta = np.abs(target_world_offset[2])

        max_xy_delta = self.dt * self.max_horizontal_speed
        max_z_delta = self.dt * self.max_vertical_speed

        if (z_delta < 2):
            max_xy_delta *= 0.5
            max_z_delta *= 0.5
        

        def sigmoid_deadband_3d(arr, threshold=0.3, sharpness=10):
           return arr * (1 / (1 + np.exp(-sharpness * (np.abs(arr) - threshold))))

        target_world_offset[:2] = sigmoid_deadband_3d(target_world_offset[:2])

        if(xy_delta > max_xy_delta):
           target_world_offset[:2] = target_world_offset[:2] / xy_delta * max_xy_delta

        if(z_delta > max_z_delta):
           target_world_offset[2] = np.sign(target_world_offset[2]) * max_z_delta



        return target_world_offset

    def update(self, 
               drone_position: np.ndarray,  # [x, y, z]
               drone_yaw: float,            # radians
               target_angle_x: float,       # degrees (right is +ve)
               target_angle_y: float) -> np.ndarray:  # degrees (forward is +ve)
        """
        Main update function - call this at 100Hz
        
        Args:
            drone_position: Current drone position [x, y, z]
            drone_yaw: Current drone yaw angle (radians)
            target_angle_x: Target angle in x direction (right positive)  
            target_angle_y: Target angle in y direction (forward positive)
            
        Returns:
            Target position [x, y, z] for drone controller
        """
        
        target_world_offset = self.calculate_target_position(
            drone_position, drone_yaw, target_angle_x, target_angle_y
        )

        print(target_world_offset)

        limited_offset = self.limit_vel(target_world_offset)

        print(limited_offset)
         
        is_close = np.linalg.norm(target_world_offset) < self.acceptance_radius

        return drone_position+limited_offset, is_close

    def reset(self):
        """Reset the controller state"""
        self.prev_desired_xy = None
        self.prev_desired_z = None
