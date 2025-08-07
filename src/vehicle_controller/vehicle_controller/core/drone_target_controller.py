import numpy as np
import math
from typing import Tuple, Optional

from vehicle_controller.core.slew import SlewRate

class DroneTargetController:
    
    def __init__(self, target_distance: float = 0.35, target_altitude: float = 5.0, acceptance_radius: float = 0.1):
        self.target_distance = target_distance

        self.slew_xy = SlewRate(0.01, 3.0, 1.0)
        self.slew_z = SlewRate(0.01, 10.0, 4.0)

        self.dt = 0.01

    def calculate_target_position(self, 
                                drone_position: np.ndarray,  # [N, E, D] in global frame
                                drone_yaw: float,            # radians
                                target_angle_x: float,       # degrees (right position)
                                target_angle_y: float) -> np.ndarray:  # degrees (forward position)
        
        drone_altitude = -drone_position[2]  # Convert to positive altitude
        horizontal_distance_x = drone_altitude * math.tan(np.deg2rad(target_angle_x))  # right distance
        horizontal_distance_y = drone_altitude * math.tan(np.deg2rad(target_angle_y))  # forward distance
        
        # Target position in drone's local frame (relative to drone)
        target_local = np.array([horizontal_distance_x, horizontal_distance_y, drone_altitude])
        
        # Adjust for target distance behind camera
        target_local += np.array([0, 1, 0]) * self.target_distance
        
        # Rotate to NED frame using yaw
        cos_yaw = np.cos(drone_yaw)
        sin_yaw = np.sin(drone_yaw)
        
        # (x,y) to (N,E) conversion - (1,0) goes to (-sin,cos), (0,1) goes to (cos, sin)
        target_world_offset = np.array([
             -sin_yaw * target_local[0] + cos_yaw * target_local[1],  # world N
             cos_yaw * target_local[0] + sin_yaw * target_local[1],  # world E
             drone_altitude                                         # world D is  just towards ground
        ])
        
        return target_world_offset

    def update(self, 
               drone_position: np.ndarray,  # [x, y, z]
               drone_yaw: float,            # radians
               target_angle_x: float,       # degrees (right is +ve)
               target_angle_y: float) -> np.ndarray:  # degrees (forward is +ve)

        target_world_offset = self.calculate_target_position(
            drone_position, drone_yaw, target_angle_x, target_angle_y
        )

        distance_xy = np.linalg.norm(target_world_offset[:2])
        
        # Horizontal tracking
        self.slew_xy.v_max = 3.0
        self.slew_xy.a_max = 1.0

        if distance_xy < 0.5:
            self.slew_xy.v_max = 0.5
            self.slew_xy.a_max = 0.1

        slewed_xy = self.slew_xy.slew_rate(drone_position[:2],drone_position[:2]+target_world_offset[:2])

        # Horizontal tracking

        if distance_xy > 0.5:
            self.slew_z.v_max = 3*np.exp(0.5-distance_xy) 
        if distance_xy > 2:
            self.slew_z.v_max= 0.0

        # Vertical tracking
        slewed_z = self.slew_z.slew_rate(drone_position[2],-3.0)

        target_pos = [*slewed_xy,slewed_z]

        is_close = distance_xy < 0.05 and np.linalg.norm(self.slew_xy.sp_last - self.slew_xy.sp_last2) < 0.04 * self.slew_xy.dt
        
        return target_pos, is_close

    def reset(self):
        """Reset the controller state"""
        self.slew_xy.reset()
        self.slew_z.reset()
