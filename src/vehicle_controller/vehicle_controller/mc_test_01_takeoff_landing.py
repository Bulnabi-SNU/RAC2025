"""
Takeoff and Landing Mission Controller
Implements a basic takeoff, hold, and land mission using the PX4BaseController
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
from px4_msgs.msg import VehicleStatus, VehicleCommand
from vehicle_controller.code_basic.px4_base import PX4BaseController


class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_01_takeoff_landing')
        
        # Mission parameters
        self.hold_time_seconds = 10.0
        self.hold_timer = 0
        self.hold_timer_threshold = int(self.hold_time_seconds / self.time_period)
        
        # State machine
        self.state = 'READY_TO_FLIGHT'
        
        self.get_logger().info("Mission Controller Test01 initialized")
    
    def main_loop(self):
        """Main control loop - implements the state machine"""
        
        if self.state == 'READY_TO_FLIGHT':
            self._handle_ready_to_flight()
        
        elif self.state == 'TAKEOFF':
            self._handle_takeoff()
        
        elif self.state == 'HOLDING':
            self._handle_holding()
        
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
            self.get_logger().info("Takeoff complete. Starting hold phase...")
            self.state = 'HOLDING'
            self.hold_timer = 0
    
    def _handle_holding(self):
        """Handle holding state"""
        if self.hold_timer >= self.hold_timer_threshold:
            self.get_logger().info("Hold phase complete. Initiating landing...")
            self.state = 'LANDING'
        else:
            # Log hold status every second
            if self.hold_timer % int(1.0 / self.time_period) == 0:
                remaining_time = (self.hold_timer_threshold - self.hold_timer) * self.time_period
                self.get_logger().info(f"Holding... {remaining_time:.1f}s remaining")
            
            self.hold_timer += 1
    
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
