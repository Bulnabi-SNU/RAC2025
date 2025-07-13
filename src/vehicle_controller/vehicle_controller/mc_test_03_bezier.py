"""
Bezier Navigation Mission Controller
Implements a basic takeoff, waypoint nav (with bezier curve generation), and land mission using the PX4BaseController
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
from px4_msgs.msg import VehicleStatus, VehicleCommand
from vehicle_controller.px4_base import PX4BaseController
from vehicle_controller.bezier_handler import BezierCurve

import numpy as np
import math

class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_03_bezier')
    
        self.num_wp = 4
        self.WP = [[5,0,10],[5,5,10],[0,5,10],[0,0,10],]           # waypoints, local coordinates. 0~num_wp-15
        self.gps_WP = [      # TODO: Convert GPS to local(NED) coords
            [47.397191,   8.546472, 0.0],
            [47.397191,   8.546672, 0.0],
            [47.397391,   8.546672, 0.0],
            [47.397391,   8.546472, 0.0] ]
        

        self.start_position = None
        self.current_point_index = 0

        self.bezier_handler = BezierCurve(time_step=0.05)

        self.vmax = 2.0  # m/s
        self.arrival_radius = 0.5

        self.offboard_control_mode_params['position'] = True
        self.offboard_control_mode_params['velocity'] = False

        # State machine
        self.state = 'READY_TO_FLIGHT'
        
        self.get_logger().info("Mission Controller Test03 initialized")
    
    def main_loop(self):
        """Main control loop - implements the state machine"""
        
        if self.state == 'READY_TO_FLIGHT':
            self._handle_ready_to_flight()
        
        elif self.state == 'TAKEOFF':
            self._handle_takeoff()
        
        elif self.state == 'BEZIER_NAV':
            self._handle_bezier_nav()
        
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
            self.get_logger().info("Takeoff complete. Starting bezir navigation phase...")
            self.set_offboard_mode()
            self.state = 'BEZIER_NAV'
    
    def _handle_bezier_nav(self):
        """Handle bezir navigation state"""
        if self.current_point_index < self.num_wp:
                if self.start_position is None:
                    self.start_position = np.copy(self.pos)
                    self.bezier_handler.generate_curve(
                        start_pos=self.start_position,
                        end_pos=self.WP[self.current_point_index],
                        max_velocity=self.vmax,
                        total_time=None
                    )
                    self.publish_setpoint(position_sp = self.bezier_handler.get_current_point()) 

                distance = np.linalg.norm(self.bezier_handler.get_current_point() - self.pos)

                if distance < self.arrival_radius:
                    if self.bezier_handler.get_next_point() is None:
                        self.current_point_index += 1
                        self.start_position = np.copy(self.pos)
                        if self.current_point_index >= self.num_wp:
                            self.get_logger().info("All waypoints reached. Landing...")
                            self.state = 'LANDING'
                            return
                        self.get_logger().info("Waypoints reached. Moving to next waypoint")
                        self.bezier_handler.generate_curve(
                            start_pos=self.start_position,
                            end_pos=self.WP[self.current_point_index],
                            max_velocity=self.vmax,
                            total_time=None
                        )
                    self.publish_setpoint(position_sp = self.bezier_handler.get_current_point()) 
        else:
            print("Bezier flight complete. Landing...")
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
