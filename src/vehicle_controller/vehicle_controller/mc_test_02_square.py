"""
Square Navigation Mission Controller
Implements a basic takeoff, waypoint nav, and land mission using the PX4BaseController
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
from px4_msgs.msg import VehicleStatus, VehicleCommand
from vehicle_controller.px4_base import PX4BaseController

import numpy as np
import math

class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_02_square')
        
        # Mission parameters
        self.square_points = [
            np.array([0.0, 5.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0]),
            np.array([-5.0, 0.0, 0.0])
        ]
        self.start_position = None
        self.current_point_index = 0
        self.arrival_radius = 1.0

        self.speed = 2.0  # m/s


        self.offboard_control_mode_params['position'] = False
        self.offboard_control_mode_params['velocity'] = True

        # State machine
        self.state = 'READY_TO_FLIGHT'
        
        self.get_logger().info("Mission Controller Test02 initialized")
    
    def main_loop(self):
        """Main control loop - implements the state machine"""
        
        if self.state == 'READY_TO_FLIGHT':
            self._handle_ready_to_flight()
        
        elif self.state == 'TAKEOFF':
            self._handle_takeoff()
        
        elif self.state == 'SQUARE_NAV':
            self._handle_square_nav()
        
        elif self.state == 'LANDING':
            self._handle_landing()
        
        elif self.state == 'MISSION_COMPLETE':
            self._handle_mission_complete()
    
    def _handle_ready_to_flight(self):
        """Handle the initial ready to flight state"""
        if self.is_offboard_mode():
            if self.is_disarmed():
                self.get_logger().info("Vehicle in offboard mode but disarmed. Arming...")
                self.arm()
            else:
                self.get_logger().info("Vehicle armed. Commanding takeoff...")
                self.takeoff()
        
        elif self.is_auto_takeoff():
            self.get_logger().info("Vehicle in auto takeoff mode")
            if not self.get_position_flag:
                self.get_logger().info("Waiting for position data...")
                return
            
            # Set home position
            self.set_home_position()
            self.get_logger().info(f"Home position set: {self.home_position}")
            self.state = 'TAKEOFF'
    
    def _handle_takeoff(self):
        """Handle takeoff state"""
        if self.is_auto_loiter():
            self.get_logger().info("Takeoff complete. Starting square navigation phase...")
            self.set_offboard_mode()
            self.state = 'SQUARE_NAV'
    
    def _handle_square_nav(self):
        """Handle square navigation state"""
        if self.current_point_index < len(self.square_points):
                if self.start_position is None:
                    self.start_position = np.copy(self.pos)

                
                target = self.start_position + self.square_points[self.current_point_index]
                
                direction = target - self.pos
                distance = np.linalg.norm(direction)

                direction /= distance if distance > 0 else 1.0  # Avoid division by zero

                if distance < self.arrival_radius:
                    print(f"Reached point {self.current_point_index + 1}")
                    self.current_point_index += 1
                    self.start_position = None
                else:
                    yaw_sp = math.atan2(direction[1], direction[0])
                    vel = self.speed * direction
                    self.publish_setpoint(vel_sp=vel, yaw_sp=yaw_sp)
        else:
            print("Square flight complete. Landing...")
            self.state = 'LANDING'
    
    def _handle_landing(self):
        """Handle landing state"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.get_logger().info("Mission complete!")
        self.state = 'MISSION_COMPLETE'
    
    def _handle_mission_complete(self):
        """Handle mission complete state"""
        pass
        
    
    def on_vehicle_status_update(self, msg):
        """Override to handle vehicle status updates"""
        # Could add additional status monitoring here
        pass
    
    def on_local_position_update(self, msg):
        """Override to handle local position updates"""
        # Could add position monitoring here
        pass
    
    def on_global_position_update(self, msg):
        """Override to handle global position updates"""
        # Could add GPS monitoring here
        pass


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        controller = MissionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Mission interrupted by user")
    except Exception as e:
        print(f"Mission failed with error: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
