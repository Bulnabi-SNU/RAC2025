"""
Simple Bezier Curve Generator
A clean implementation for generating smooth trajectory curves between waypoints.
"""

import numpy as np


class BezierCurve:
    def __init__(self, time_step=0.05):
        """
        Initialize Bezier curve generator
        
        Args:
            time_step: Time step for trajectory points (default: 0.05 = 20Hz)
        """
        # fetch from arg
        self.time_step = time_step

        # for run bezier curve
        self.current_index = 0
        self.trajectory_points = None
        self.num_trajectory_points = 0

        # for thrashold or constants
        self.max_acceleration = 9.81 * np.tan(10 * np.pi / 180)  # 10 degree tilt angle
        self.mc_start_speed = 0.0001    # when start_vel=None
        self.mc_end_speed = 0.0001      # when end_vel=None
        self.bezier_threshold_speed = 0.7
        self.bezier_minimum_time = 3.0
        
    def generate_curve(self, start_pos, end_pos, start_vel=None, end_vel=None, max_velocity=None, total_time=None):
        """
        Generate a cubic Bezier curve between two points
        
        Args:
            start_pos   : Starting position [x, y, z]
            end_pos     : Ending position [x, y, z]
            start_vel   : Starting velocity [vx, vy, vz]
            end_vel     : Ending velocity [vx, vy, vz]
            max_velocity: Maximum velocity for trajectory
            total_time  : Total time for trajectory (auto-calculated if None)

        Returns:
            numpy array of trajectory points (not necessary)
        """
        #-------------------------------------
        # calculate direction / velocity
        #-------------------------------------
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # calculate direction
        if np.linalg.norm(end_pos - start_pos) > 0: direction = (end_pos - start_pos)/ np.linalg.norm(end_pos - start_pos)
        else: direction = np.zeros(3)

        if (start_vel is None) and (direction != np.zeros(3)):
            start_vel = self.mc_start_speed * direction / np.linalg.norm(direction)
        else:
            start_vel = np.array(start_vel)
        
        if (end_vel is None) and (direction != np.zeros(3)):
            end_vel = self.mc_end_speed * direction / np.linalg.norm(direction)
        else:
            end_vel = np.array(end_vel)
        



        # Calculate total time if not provided
        if total_time is None:
            distance = np.linalg.norm(end_pos - start_pos)
            total_time = max(distance / max_velocity, 1.0)  # Minimum 1 second
        
        # Number of points in trajectory
        num_points = int(total_time / self.time_step)
        
        # Control points for smooth curve
        direction = end_pos - start_pos
        control_offset = np.linalg.norm(direction) * 0.3  # 30% of distance
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Bezier control points
        p0 = start_pos                           # Start point
        p1 = start_pos + direction * control_offset  # First control point
        p2 = end_pos - direction * control_offset    # Second control point
        p3 = end_pos                             # End point
        
        # Generate curve points
        t_values = np.linspace(0, 1, num_points)
        trajectory_points = []
        
        for t in t_values:
            # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
            trajectory_points.append(point)
        
        self.trajectory_points = np.array(trajectory_points)
        self.num_trajectory_points = len(trajectory_points)
        self.current_index = 0
        
        #return self.trajectory_points
    
    def get_next_point(self):
        """
        Get the next point in the trajectory
        
        Returns:
            Next trajectory point or None if trajectory is complete
        """
        if self.trajectory_points is None:
            return None
            
        if self.current_index < len(self.trajectory_points):
            point = self.trajectory_points[self.current_index]
            self.current_index += 1
            return point
        
        return None
    
    def get_current_point(self):
        """
        Get the current point without advancing the index
        
        Returns:
            Current trajectory point or None
        """
        if self.trajectory_points is None or self.current_index >= len(self.trajectory_points):
            return None
            
        return self.trajectory_points[self.current_index]
    
    def is_complete(self):
        """
        Check if trajectory is complete
        
        Returns:
            True if trajectory is finished, False otherwise
        """
        return (self.trajectory_points is None or 
                self.current_index >= self.num_trajectory_points)
    
    def reset(self):
        """Reset the trajectory to start from beginning"""
        self.current_index = 0