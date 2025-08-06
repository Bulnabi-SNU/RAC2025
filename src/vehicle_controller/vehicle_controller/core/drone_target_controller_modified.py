import numpy as np
import math
from typing import Tuple, Optional

class DroneTargetController:
    
    def __init__(self, target_distance: float = 0.35, target_altitude: float = 5.0, acceptance_radius: float = 0.1):
        self.target_distance = target_distance
        self.logger =0

        # Hate this but it's the most easy way
        self.tracking_state = 0
        
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

    def xy_deadband(self, x:float):
        a1=0.5
        d1=0.05
        k1=1


        return a1*np.tanh(k1*max(0.0,x - d1))

    def limit_xy(self, target_offset: np.ndarray):
        distance = np.linalg.norm(target_offset)
        limited_dist = self.xy_deadband(distance)

        if limited_dist == 0: return [0.0,0.0],distance
        else: return target_offset / distance * limited_dist, distance

    def limit_z(self, drone_pos_z, xy_distance):
        if(xy_distance > 0.2 or xy_distance / drone_pos_z > np.tan(np.deg2rad(20))): 
            if(drone_pos_z < -10): return -10-drone_pos_z # Set altitude when "lost target"
            return -0.1
        return 0.8 


    def update(self, 
               drone_position: np.ndarray,  # [x, y, z]
               drone_yaw: float,            # radians
               target_angle_x: float,       # degrees (right is +ve)
               target_angle_y: float) -> np.ndarray:  # degrees (forward is +ve)
        


        self.logger +=1 
        target_world_offset = self.calculate_target_position(
            drone_position, drone_yaw, target_angle_x, target_angle_y
        )

        if(self.logger%30 == 0):
            print(target_world_offset, drone_yaw)


        # Track initial state just with px4 (High speed)
        # Horizontal tracking

        offset_limit_xy, distance_xy = self.limit_xy(np.array(target_world_offset[:2]))

        # Vertical tracking
        offset_z = self.limit_z(drone_position[2],distance_xy)

        limited_offset = [*offset_limit_xy,offset_z]


        #is_close = np.linalg.norm(target_world_offset) < self.acceptance_radius
        is_close = False

        return drone_position+limited_offset, is_close

    def reset(self):
        """Reset the controller state"""
        self.prev_desired_xy = None
        self.prev_desired_z = None
