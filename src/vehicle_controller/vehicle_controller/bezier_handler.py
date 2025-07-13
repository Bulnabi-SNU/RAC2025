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
        self.time_step = time_step
        self.current_index = 0
        self.trajectory_points = None
        
    def generate_curve(self, start_pos, end_pos, max_velocity=2.0, total_time=None):
        """
        Generate a cubic Bezier curve between two points
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            max_velocity: Maximum velocity for trajectory
            total_time: Total time for trajectory (auto-calculated if None)
            
        Returns:
            numpy array of trajectory points
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
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
        self.current_index = 0
        
        return self.trajectory_points
    
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
                self.current_index >= len(self.trajectory_points))
    
    def reset(self):
        """Reset the trajectory to start from beginning"""
        self.current_index = 0