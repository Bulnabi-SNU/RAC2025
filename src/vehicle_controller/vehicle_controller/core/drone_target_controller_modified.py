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
        self.max_horizontal_speed = 0.001  # Maximum horizontal speed in m/s
        self.max_vertical_speed = 0.01  # Maximum vertical speed in m/s
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
            drone_position[2] = -10 # NED
            return drone_position

        drone_altitude = -drone_position[2]  # Convert to positive altitude
        horizontal_distance_x = drone_altitude * math.tan(np.deg2rad(target_angle_x))  # right distance
        horizontal_distance_y = drone_altitude * math.tan(np.deg2rad(target_angle_y))  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, -drone_altitude])
        print(f"Target local position: {target_local}")
        
        # Adjust for target distance behind camera
        target_local += np.array([0, 1, 0]) * self.target_distance
        
        # Rotate to NED frame using yaw
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        print(f"[Drone Pos] yaw: {drone_yaw}, cos yaw: {cos_yaw}, sin yaw: {sin_yaw}")
        
        # (x,y) to (N,E) conversion - (1,0) goes to (-sin,cos), (0,1) goes to (cos, sin)
        target_world_offset = np.array([
             -sin_yaw * target_local[0] + cos_yaw * target_local[1],  # world N
             cos_yaw * target_local[0] + sin_yaw * target_local[1],  # world E
            0                                         # world D is ignored for now
        ])
        print(f"Target world offset: {target_world_offset}")
        
        return target_local, target_world_offset
    
    
    def horizontal_slew(self,
                drone_position: np.ndarray,  # [x, y, z]
                target_world_offset: np.ndarray) -> np.ndarray: # 3 dimension setpoints
        """
        Calculate continuous horizontal setpoints to limit acceleration and avoid overshoot
        
        Args:
            drone_position: Current drone position in global frame [N, E, D]
            target_world_offset: Target position offset in global frame [N, E, D]

        Returns:
            np.ndarray: Continuous horizontal setpoints [N, E, D]
        """
        
        desired_xy = target_world_offset[:2] + drone_position[:2]  # Only x and y components in global frame
        print(f"desired xy in global : {desired_xy}")
        
        # If previous target is NaN, initialize it
        if self.prev_desired_xy is None:
            self.prev_desired_xy = desired_xy.copy()
            
        # Calculate delta
        delta = desired_xy - self.prev_desired_xy
        
        # Limit the step size based on horizontal distance
        max_speed = 1.0 # Maximum speed in m/s
        dt = 0.2  # Time step in seconds
        max_delta = max_speed * dt
        delta_norm = np.linalg.norm(delta)
        
        # Limit the movement delta to max_delta
        if delta_norm > max_delta:
            delta = (delta / delta_norm) * max_delta
        
        # Calculate the next setpoint
        next_setpoint = self.prev_desired_xy + delta
        
        # Update the previous target position
        self.prev_desired_xy = next_setpoint.copy()
        
        return np.array([next_setpoint[0], next_setpoint[1], 0])  # Return as [x, y, z]
    
    
    def vertical_slew(self,
                drone_position: np.ndarray,  # [x, y, z]
                target_world_offset: np.ndarray) -> np.ndarray: # 3 dimension setpoints
                
        """
        Calculate continuous vertical setpoints to limit acceleration and avoid overshoot
        
        Args:
            drone_position: Current drone position [N, E, D]
            target_world_offset: Target position offset in global frame [N, E, D]
            
        Returns:
            np.ndarray: Continuous vertical setpoint [z]
        """
        
        # Calculate the vertical delta
        desired_z = target_world_offset[2] + drone_position[2]  # Only z component in global frame
        print(f"desired_z : {desired_z}")
        
        # If previous target is NaN, initialize it
        if self.prev_desired_z is None:
            self.prev_desired_z = desired_z
            
        # Calculate delta
        delta = desired_z - self.prev_desired_z
        
        # Limit the step size based on vertical distance
        max_speed = 0.5  # Maximum vertical speed in m/s
        dt = 0.2  # Time step in seconds
        max_delta = max_speed * dt
        
        # Limit the movement delta to max_delta
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
            
        # Calculate the next setpoint
        next_setpoint = self.prev_desired_z + delta
        
        # Update the previous target position
        self.prev_desired_z = next_setpoint
        
        return np.array([drone_position[0], drone_position[1], next_setpoint])  # Return as [x, y, z] with x and y as 0
        

    def approach_decision(self, 
               drone_position: np.ndarray,  # [x, y, z]
               target_world_offset: np.ndarray) -> np.ndarray:
        """
        Decide the approach setpoint based on current drone position and target offset

        Args:
            drone_position : Current drone position in global frame [x, y, z]
            target_world_offset: Target position offset in global frame [N, E, D]

        Returns:
            np.ndarray: Approach setpoint [x, y, z]
        """
        
        horizontal_distance = np.linalg.norm(target_world_offset[:2])  # Only x and y components
        vertical_distance = abs(target_world_offset[2])  # Only z component
        
        # If the horizontal distance is greater than acceptance radius, use horizontal slew
        if horizontal_distance > self.acceptance_radius:
            desired_position = self.horizontal_slew(drone_position, target_world_offset)
        elif vertical_distance > self.acceptance_radius:
            desired_position = self.vertical_slew(drone_position, target_world_offset)
        else:
            desired_position = drone_position.copy()  # No movement needed
        
        is_close = horizontal_distance < self.acceptance_radius and vertical_distance < self.acceptance_radius
        
        if is_close:
            print("Drone is close enough to the target.")
            
        return desired_position, is_close
    

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
        
        target_local, target_world_offset = self.calculate_target_position(
            drone_position, drone_yaw, target_angle_x, target_angle_y
        )
        return self.approach_decision(drone_position, target_world_offset)
    
    def reset(self):
        """Reset the controller state"""
        self.prev_desired_xy = None
        self.prev_desired_z = None