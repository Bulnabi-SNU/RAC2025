"""
Simple Bezier Curve Generator
A clean implementation for generating smooth trajectory curves between waypoints.
"""

import numpy as np

# for visualization bezier curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        self.start_pos = None
        self.end_pos = None

        # for run bezier curve
        self.current_index = 0
        self.trajectory_points = None
        self.num_trajectory_points = 0

        # for thrashold or constants
        # self.max_acceleration = 9.81 * np.tan(10 * np.pi / 180)  # 10 degree tilt angle
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
            numpy array of trajectory points
        """
        #-------------------------------------
        # calculate direction / velocity
        #-------------------------------------
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # calculate direction and distance
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        if np.linalg.norm(direction) > 0:
            direction_norm = direction/ distance
        else: 
            direction_norm = np.zeros(3)

        # set start/end velocity
        if ((start_vel is None) or (np.linalg.norm(start_vel) < self.bezier_threshold_speed)) and (direction_norm != np.zeros(3)):
            start_vel = self.mc_start_speed * direction_norm
        else:
            start_vel = np.array(start_vel)
        
        if (end_vel is None) and (direction_norm != np.zeros(3)):
            end_vel = self.mc_end_speed * direction_norm
        else:
            end_vel = np.array(end_vel)
        
        # Calculate total time if not provided
        if total_time is None:
            total_time = max(distance / max_velocity * 2, 1.0)  # Minimum 1 second
        

        #-------------------------------------
        # compute bezier control points
        #-------------------------------------
        
        # Number of points in trajectory
        self.num_trajectory_points = int(total_time / self.time_step)
        
        # for maintain current velocity and match to differencial equation
        control_offset = np.linalg.norm(direction) / 3.0

        # Bezier control points
        p0 = start_pos                              # Start point
        p1 = start_pos + start_vel * control_offset # First control point
        p2 = end_pos - end_vel * control_offset     # Second control point
        p3 = end_pos                                # End point

        # Generate curve points
        bezier = np.linspace(0, 1, self.num_trajectory_points).reshape(-1, 1)
        
        bezier = p3 * bezier**3 +                             \
                3 * p2 * bezier**2 * (1 - bezier) +           \
                3 * p1 * bezier**1 * (1 - bezier)**2 +        \
                1 * p0 * (1 - bezier)**3
        
        self.trajectory_points = np.array(bezier)

        # reset bezier index counter
        self.current_index = 0

    def get_bezier_points(self):
        """
        Get the generated Bezier trajectory points
        
        Returns:
            numpy array of trajectory points or None if not generated
        """
        if self.trajectory_points is None:
            return None
        
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
    
    def get_current_point(self, update=1):
        """
        Get the current point without advancing the index
        
        Returns:
            Current trajectory point or None
        """
        # 1. if trajectory is not generated => None
        if self.trajectory_points is None:
            return None
        
        # 2. give end point if current index is out of range
        if self.current_index >= self.num_trajectory_points:
            return self.end_pos
        
        # 3. give current point
        # update idx when update is 1 or give nothing
        # if update is 0 -> do not update index
        point = self.trajectory_points[self.current_index]
        if update == 1:
            self.current_index += 1
        
        return point

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
        self.start_pos = None
        self.end_pos = None


def main():
    bezier = BezierCurve()
    start = [0, 0, 0]
    end = [10, 10, 0]
    start_vel = [1, 0, 0]
    end_vel = [1, 0, 0]
    trajectory = bezier.generate_curve(start, end, start_vel=start_vel, end_vel=end_vel, max_velocity=5.0, total_time=2.0)
    
    # Extract trajectory points
    bezier_points = bezier.get_bezier_points()

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory points
    ax.plot(bezier_points[:, 0], bezier_points[:, 1], bezier_points[:, 2], label='Bezier Trajectory', color='b')

    # Plot start and end positions
    ax.scatter(start[0], start[1], start[2], color='r', s=100, label='Start Position')
    ax.scatter(end[0], end[1], end[2], color='g', s=100, label='End Position')

    # Plot vectors from start_pos and end_pos
    ax.quiver(start[0], start[1], start[2], start_vel[0], start_vel[1], start_vel[2], color='r', length=0.5, label='Start Velocity')
    ax.quiver(end[0], end[1], end[2], end_vel[0], end_vel[1], end_vel[2], color='r', length=0.5, label='End Velocity')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bezier Curve in 3D')
    
    # Show legend
    ax.legend()

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()