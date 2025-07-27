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
        """
        self.target_distance = target_distance
        self.target_altitude = target_altitude
        self.acceptance_radius = acceptance_radius
        
        # Smoothing parameters for gradual transitions
        self.position_smoothing = 0.1  # Lower = smoother, higher = more responsive
        self.altitude_smoothing = 0.05  # Separate smoothing for altitude (typically slower)
        
        # Previous target for smoothing
        self.prev_target_position: Optional[np.ndarray] = None
    
    def calculate_target_position(self, 
                                drone_position: np.ndarray,  # [x, y, z]
                                drone_yaw: float,            # radians
                                target_angle_x: float,       # radians (right is +ve)
                                target_angle_y: float) -> np.ndarray:  # radians (forward is +ve)
        """
        Calculate the target position for the drone
        
        Args:
            drone_position: Current drone position [x, y, z] in world frame
            drone_yaw: Current drone yaw angle (radians)
            target_angle_x: Target angle in x direction (right positive)
            target_angle_y: Target angle in y direction (forward positive)
            
        Returns:
            Target position [N,E,D] for the drone
            
        Note: Assumes target is on the ground (z=0) and camera faces straight down
        """
        
        if np.isnan(target_angle_x):
            drone_position[2] = -15
            return drone_position

        # Convert target angles to direction vector in camera/body frame
        # Camera points straight down, target is on ground
        # We can calculate exact distances using trigonometry
        drone_altitude = -drone_position[2]
        
        # Calculate horizontal distances to target
        horizontal_distance_x = drone_altitude * math.tan(target_angle_x)  # right distance
        horizontal_distance_y = drone_altitude * math.tan(target_angle_y)  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, -drone_altitude])
        # Adjust for target distance behind camera
        target_local += np.array([0,1,0]) * self.target_distance
        
        # Rotate to NED frame using yaw
        
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        
        # (x,y) to (N,E) conversion - (1,0) goes to (-sin,cos), (0,1) goes to (cos, sin)
        target_world_offset = np.array([
             -sin_yaw * target_local[0] + cos_yaw * target_local[1],  # world N
             cos_yaw * target_local[0] + sin_yaw * target_local[1],  # world E
            0                                         # world D is ignored for now
        ])
        
        distance = np.linalg.norm(target_world_offset) 
        norm_target_world_offset  = target_world_offset / distance
        
        if distance > 0.1:
            # Normalize if too far
            target_world_offset = norm_target_world_offset * 0.1
        if drone_altitude < 7.0:
            # If altitude is low, limit the offset to avoid too aggressive movements
            target_world_offset = norm_target_world_offset * 0.05
        
        # Absolute target position in world frame
        target_world_pos = drone_position + target_world_offset     
        
        # Position the drone so the target appears at the desired distance behind camera
        desired_position = target_world_pos
        # desired_position = target_world_pos
        
        # Override altitude to target altitude
        desired_position[2] = 0.9*drone_position[2] + 0.1*self.target_altitude
        
        # # Apply smoothing if we have a previous target
        # if self.prev_target_position is not None:
        #     # Smooth x and y
        #     desired_position[0] = self.prev_target_position[0] + \
        #                         self.position_smoothing * (desired_position[0] - self.prev_target_position[0])
        #     desired_position[1] = self.prev_target_position[1] + \
        #                         self.position_smoothing * (desired_position[1] - self.prev_target_position[1])
        #     # Smooth altitude separately
        #     desired_position[2] = self.prev_target_position[2] + \
        #                         self.altitude_smoothing * (desired_position[2] - self.prev_target_position[2])
        
        # Store for next iteration
        self.prev_target_position = desired_position.copy()
        
        is_close = distance < self.acceptance_radius and abs(drone_position[2] - desired_position[2]) < self.acceptance_radius
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
        
                
        drone_yaw = drone_yaw # CAlready in rad
        target_angle_x = math.radians(target_angle_x)  # Convert to radians
        target_angle_y = math.radians(target_angle_y)  # Convert to radians
        
        return self.calculate_target_position(drone_position, drone_yaw, target_angle_x, target_angle_y)
    
    def reset(self):
        """Reset the controller state"""
        self.prev_target_position = None
    
    def set_parameters(self, target_distance: Optional[float] = None, 
                      target_altitude: Optional[float] = None,
                      acceptance_radius: Optional[float] = None,
                      position_smoothing: Optional[float] = None,
                      altitude_smoothing: Optional[float] = None):
        """
        Update controller parameters
        
        Args:
            target_distance: Distance behind camera (meters)
            target_altitude: Target altitude (meters)  
            position_smoothing: Position smoothing factor (0-1)
            altitude_smoothing: Altitude smoothing factor (0-1)
        """
        if target_distance is not None:
            self.target_distance = target_distance
        if target_altitude is not None:
            self.target_altitude = target_altitude
        if position_smoothing is not None:
            self.position_smoothing = max(0.01, min(1.0, position_smoothing))
        if altitude_smoothing is not None:
            self.altitude_smoothing = max(0.01, min(1.0, altitude_smoothing))
        if acceptance_radius is not None:
            self.acceptance_radius = acceptance_radius