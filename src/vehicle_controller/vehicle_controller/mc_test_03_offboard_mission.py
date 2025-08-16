"""
Test 03: Offboard -> Arm -> Mission Mode
Tests the transition from offboard mode to mission mode execution
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
    MISSION_EXECUTE = "MISSION_EXECUTE"
    COMPLETE = "COMPLETE"


class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_03')
        
        self._load_parameters()
        self._initialize_components()
        
        self.state = TestState.INIT
        
        self.get_logger().info("Test 03: Offboard -> Mission Mode initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        params = [
            ('timer_period', 0.01),
        ]
        
        self.declare_parameters(namespace='', parameters=params)
        
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)

    def _initialize_components(self):
        """Initialize controllers and logger"""
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        self.time_counter=0
        
        # Add parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.param_update_callback)

    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            TestState.INIT: self._handle_init,
            TestState.OFFBOARD_ARM: self._handle_offboard_arm,
            TestState.MISSION_EXECUTE: self._handle_mission_execute,
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
        """Wait for offboard mode, then arm and switch to mission"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, switching to mission mode")
            self.set_mission_mode()
            self.state = TestState.MISSION_EXECUTE

    def _handle_mission_execute(self):
        """Monitor mission execution"""
        if not self.is_mission_mode():
            self.get_logger().warn("Not in mission mode, mission may have completed or failed")
            self.state = TestState.COMPLETE
            return

        # Check for mission timeout
        self.time_counter += 1

        # Log mission progress every 5 seconds
        if self.time_counter % int(5.0 / self.timer_period) == 0:
            if self.mission_wp_num is not None:
                self.get_logger().info(f"Mission executing... Current WP: {self.mission_wp_num}")

        # Check if mission completed (vehicle landed and disarmed)
        if self.is_disarmed():
            self.get_logger().info("Mission completed - vehicle disarmed")
            self.state = TestState.COMPLETE

    def _handle_complete(self):
        """Test complete"""
        self.get_logger().info("Test 03 Complete!")

    def param_update_callback(self, params):
        """Parameter callback for dynamically updating parameters while flying"""
        successful = True
        reason = ''
        
        for p in params:
            if p.name == 'mission_timeout':
                self.mission_timeout = p.value
                self.mission_timeout_threshold = int(self.mission_timeout / self.timer_period)
            else:
                self.get_logger().warn(f"Ignoring parameter: {p.name}")
                continue
        
        self.get_logger().info("[Parameter Update] Test 03 parameters updated successfully")
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
        print("Test 03 interrupted by user")
    except Exception as e:
        print(f"Test 03 failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()