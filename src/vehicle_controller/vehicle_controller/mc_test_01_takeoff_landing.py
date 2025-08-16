"""
Test 01: Simple Takeoff and Landing
Implements a basic takeoff, hold, and land mission using the PX4BaseController
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
import numpy as np
from enum import Enum
from vehicle_controller.core.px4_base import PX4BaseController
from custom_msgs.msg import VehicleState
from rcl_interfaces.msg import SetParametersResult


class TestState(Enum):
    INIT = "INIT"
    OFFBOARD_ARM = "OFFBOARD_ARM"
    TAKEOFF = "TAKEOFF"
    HOLDING = "HOLDING"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"


class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_01')
        
        self._load_parameters()
        self._initialize_components()
        
        self.state = TestState.INIT
        self.hold_timer = 0
        
        self.get_logger().info("Test 01: Takeoff and Landing initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        params = [
            ('hold_time_seconds', 10.0),
            ('takeoff_altitude', 5.0),
        ]
        
        self.declare_parameters(namespace='', parameters=params)
        
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)
        
        self.hold_timer_threshold = int(self.hold_time_seconds / self.timer_period)

    def _initialize_components(self):
        """Initialize controllers and logger"""
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        # Add parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.param_update_callback)

    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            TestState.INIT: self._handle_init,
            TestState.OFFBOARD_ARM: self._handle_offboard_arm,
            TestState.TAKEOFF: self._handle_takeoff,
            TestState.HOLDING: self._handle_holding,
            TestState.LANDING: self._handle_landing,
            TestState.COMPLETE: self._handle_complete,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
        
        self._publish_vehicle_state()

    def _publish_vehicle_state(self):
        """Publish current vehicle state"""
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=0  # No detection needed for this test
            )
        )

    # =======================================
    # State Machine Handlers
    # =======================================

    def _handle_init(self):
        """Initialize system and check status"""
        if not self.get_position_flag:
            self.get_logger().info("Waiting for global position data...")
            return

        self.set_home_position()
        
        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = TestState.OFFBOARD_ARM

    def _handle_offboard_arm(self):
        """Wait for offboard mode, then arm"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, taking off")
            self.state = TestState.TAKEOFF

    def _handle_takeoff(self):
        """Handle takeoff to specified altitude"""
        takeoff_pos = np.array([0.0, 0.0, -self.takeoff_altitude])
        self.publish_setpoint(pos_sp=takeoff_pos)
        
        # Check if reached takeoff altitude
        if abs(self.pos[2] - (-self.takeoff_altitude)) < 0.5:
            self.get_logger().info(f"Reached takeoff altitude of {self.takeoff_altitude}m. Starting hold phase...")
            self.hold_timer = 0
            self.state = TestState.HOLDING

    def _handle_holding(self):
        """Handle holding state"""
        # Hold position at takeoff altitude
        hold_pos = np.array([0.0, 0.0, -self.takeoff_altitude])
        self.publish_setpoint(pos_sp=hold_pos)
        
        if self.hold_timer >= self.hold_timer_threshold:
            self.get_logger().info("Hold phase complete. Landing...")
            self.state = TestState.LANDING
        else:
            # Log hold status every second
            if self.hold_timer % int(1.0 / self.timer_period) == 0:
                remaining_time = (self.hold_timer_threshold - self.hold_timer) * self.timer_period
                self.get_logger().info(f"Holding... {remaining_time:.1f}s remaining")
            
            self.hold_timer += 1

    def _handle_landing(self):
        """Handle landing sequence"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.get_logger().info("Test 01 Complete!")
        self.state = TestState.COMPLETE

    def _handle_complete(self):
        """Test complete"""
        pass

    def param_update_callback(self, params):
        """Parameter callback for dynamically updating parameters while flying"""
        successful = True
        reason = ''
        
        for p in params:
            if p.name == 'hold_time_seconds' and p.type_ == p.Type.DOUBLE:
                self.hold_time_seconds = p.value
                self.hold_timer_threshold = int(self.hold_time_seconds / self.timer_period)
            elif p.name == 'takeoff_altitude' and p.type_ == p.Type.DOUBLE:
                self.takeoff_altitude = p.value
            else:
                self.get_logger().warn(f"Ignoring unknown parameter: {p.name}")
                continue
        
        self.get_logger().info("[Parameter Update] Test 01 parameters updated successfully")
        return SetParametersResult(successful=successful, reason=reason)

    # Override methods (placeholders)
    def on_vehicle_status_update(self, msg): pass
    def on_local_position_update(self, msg): pass
    def on_attitude_update(self, msg): pass
    def on_global_position_update(self, msg): pass


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    controller = None

    try:
        controller = MissionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Test 01 interrupted by user")
    except Exception as e:
        print(f"Test 01 failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()