import numpy as np
import math
from typing import Tuple, Optional

class DroneTargetController:
    """
    Simplified controller for positioning drone to keep target 0.35m behind camera at 3m altitude
    Assumes target is on the ground and camera faces straight down.
    """
    
    def __init__(self, target_distance: float = 0.35, target_altitude: float = 3.0, acceptance_radius: float = 0.5):
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

        # Convert target angles to direction vector in camera/body frame
        # Camera points straight down, target is on ground
        # We can calculate exact distances using trigonometry
        drone_altitude = -drone_position[2]
        
        # Calculate horizontal distances to target
        horizontal_distance_x = drone_altitude * math.tan(target_angle_x)  # right distance
        horizontal_distance_y = drone_altitude * math.tan(target_angle_y)  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, -drone_altitude])
        print(f"Target local position: {target_local}")
        
        
        # Rotate to world frame using yaw
        
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        
        target_world_offset = np.array([
            cos_yaw * target_local[1] - sin_yaw * target_local[0],  # world N
            sin_yaw * target_local[1] + cos_yaw * target_local[0],  # world E
            target_local[2]                                         # world D
        ])
        
        # Absolute target position in world frame
        target_world_pos = drone_position + target_world_offset
        
        # Calculate drone forward direction in world frame (+y in drone frame)
        drone_forward_world = np.array([
            cos_yaw,  # world N (drone's +y rotated to world frame)
            sin_yaw,   # world E
            0          # world D (no pitch/roll)
        ])
        
        # Position the drone so the target appears at the desired distance behind camera
        # desired_position = target_world_pos - drone_forward_world * self.target_distance
        desired_position = target_world_pos
        
        # Override altitude to target altitude
        desired_position[2] = self.target_altitude
        
        # Apply smoothing if we have a previous target
        if self.prev_target_position is not None:
            # Smooth x and y
            desired_position[0] = self.prev_target_position[0] + \
                                self.position_smoothing * (desired_position[0] - self.prev_target_position[0])
            desired_position[1] = self.prev_target_position[1] + \
                                self.position_smoothing * (desired_position[1] - self.prev_target_position[1])
            # Smooth altitude separately
            desired_position[2] = self.prev_target_position[2] + \
                                self.altitude_smoothing * (desired_position[2] - self.prev_target_position[2])
        
        # Store for next iteration
        self.prev_target_position = desired_position.copy()
        
        return desired_position
    
    # TODO: Reduce velocity at lower altitudes, rename arguments since max_vel isn't what it does
    def get_smoothed_target_position(self, 
                                drone_position: np.ndarray,  # [x, y, z]
                                drone_yaw: float,            # radians
                                target_angle_x: float,       # radians (right is +ve)
                                target_angle_y: float,       # radians (forward is +ve)
                                max_velocity: float = 25,   # m/s max movement speed
                                dt: float = 0.01) -> np.ndarray:  # time step (100Hz = 0.01s)
        """
        Get smoothed target position with velocity limitings
        
        Args:
            drone_position: Current drone position [x, y, z]
            drone_yaw: Current drone yaw angle (radians)
            target_angle_x: Target angle in x direction (right positive)
            target_angle_y: Target angle in y direction (forward positive)
            max_velocity: Maximum movement velocity (m/s)
            dt: Time step between calls (seconds)
            
        Returns:
            Velocity-limited target position [x, y, z]
        """
        # Calculate ideal target position
        ideal_target = self.calculate_target_position(drone_position, drone_yaw, target_angle_x, target_angle_y)
        
        if self.prev_target_position is None:
            return ideal_target
        
        # Calculate maximum allowed movement this timestep
        max_movement = max_velocity * dt
        
        # Calculate desired movement vector
        movement = ideal_target - drone_position
        movement_magnitude = np.linalg.norm(movement)
        
        # Limit movement if too aggressive
        if movement_magnitude > max_movement:
            movement = movement * (max_movement / movement_magnitude)
        
        # Return limited target position
        limited_target = drone_position + movement
        self.prev_target_position = limited_target.copy()
        
        return limited_target
    
    def is_close_to_target(self, 
                     drone_position: np.ndarray,  # [x, y, z]
                     drone_yaw: float,            # radians
                     target_angle_x: float,       # radians (right is +ve)
                     target_angle_y: float,       # radians (forward is +ve)
                      radius: float) -> bool:      # meters
        """
        Check if drone is within specified radius of the target
        
        Args:
            drone_position: Current drone position [x, y, z]
            drone_yaw: Current drone yaw angle (radians)
            target_angle_x: Target angle in x direction (right positive)
            target_angle_y: Target angle in y direction (forward positive)
            radius: Distance threshold (meters)
            
        Returns:
            True if drone is within radius of target
        """
        # Calculate target position on ground
        drone_altitude = drone_position[2]
        horizontal_distance_x = drone_altitude * math.tan(target_angle_x)
        horizontal_distance_y = drone_altitude * math.tan(target_angle_y)
        
        # Target position in drone's local frame
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, -drone_altitude])
        
        # Rotate to world frame using yaw
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        
        target_world_offset = np.array([
            cos_yaw * target_local[1] - sin_yaw * target_local[0],
            sin_yaw * target_local[1] + cos_yaw * target_local[0],
            target_local[2]
        ])
        
        target_world_pos = drone_position + target_world_offset
    
        # Calculate 2D distance (ignore altitude difference)
        distance_2d = math.sqrt((drone_position[0] - target_world_pos[0])**2 + 
                            (drone_position[1] - target_world_pos[1])**2)
    
        return (distance_2d <= radius) and (abs(drone_position[2] - target_world_pos[2]) <= radius)
    
    def update(self, 
               drone_position: np.ndarray,  # [x, y, z]
               drone_yaw: float,            # degrees
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
        
                
        drone_yaw = math.radians(drone_yaw)  # Convert to radians
        target_angle_x = math.radians(target_angle_x)  # Convert to radians
        target_angle_y = math.radians(target_angle_y)  # Convert to radians
        
        
        
        return self.get_smoothed_target_position(drone_position, drone_yaw, target_angle_x, target_angle_y), self.is_close_to_target(
            drone_position, drone_yaw, target_angle_x, target_angle_y, self.acceptance_radius)
    
    def reset(self):
        """Reset the controller state"""
        self.prev_target_position = None
    
    def set_parameters(self, target_distance: Optional[float] = None, 
                      target_altitude: Optional[float] = None,
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
