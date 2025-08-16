"""
Test 04: Offboard -> Track Target -> Land/Hover
Tests target tracking functionality with vision system
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
import numpy as np
from enum import Enum
from typing import Optional

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from custom_msgs.msg import VehicleState, TargetLocation
from rcl_interfaces.msg import SetParametersResult


class TestState(Enum):
    INIT = "INIT"
    OFFBOARD_ARM = "OFFBOARD_ARM"
    TAKEOFF = "TAKEOFF"
    TRACK = "TRACK"
    DESCEND = "DESCEND"
    LAND_OR_HOVER = "LAND_OR_HOVER"
    COMPLETE = "COMPLETE"


class MissionController(PX4BaseController):

    def __init__(self):
        super().__init__("mc_test_04")

        self._load_parameters()
        self._initialize_components()
        self._setup_subscribers()
        
        self.state = TestState.INIT
        self.target: Optional[TargetLocation] = None
        
        self.get_logger().info("Test 04: Track and Hover initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        # NOTE: Copy this function structure from mc_main but adapt parameter names
        params = [
            ('timer_period', 0.01),
            ('mission_altitude', 10.0),
            ('track_min_altitude', 4.0),
            ('gripper_altitude', 0.3),
            ('tracking_target_offset', 0.35),
            ('tracking_acceptance_radius_xy', 0.05),
            ('tracking_acceptance_radius_z', 0.2),
            ('detect_target_type', 3),  # Default to landing tag
            ('land_after_track', False),  # If false, just hover
        ]
        
        # DroneTargetController parameters - copy from mc_main
        drone_controller_params = [
            ('drone_target_controller.xy_v_max_default', 3.0),
            ('drone_target_controller.xy_a_max_default', 1.0),
            ('drone_target_controller.xy_v_max_close', 0.5),
            ('drone_target_controller.xy_a_max_close', 0.1),
            ('drone_target_controller.z_v_max_default', 10.0),
            ('drone_target_controller.z_a_max_default', 4.0),
            ('drone_target_controller.close_distance_threshold', 0.5),
            ('drone_target_controller.far_distance_threshold', 2.0),
            ('drone_target_controller.descend_radius', 0.5),
            ('drone_target_controller.z_velocity_exp_coefficient', 3.0),
            ('drone_target_controller.z_velocity_exp_offset', 0.5),
        ]
        
        self.declare_parameters(namespace='', parameters=params + drone_controller_params)
        
        # Cache parameter values
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)
        
        # Store drone controller parameters in a dict
        self.drone_controller_params = {}
        for param_name, _ in drone_controller_params:
            actual_name = param_name.replace('drone_target_controller.', '')
            self.drone_controller_params[actual_name] = self.get_parameter(param_name).value

    def _initialize_components(self):
        """Initialize controllers and logger"""
        # NOTE: Copy this function structure from mc_main
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        # Initialize DroneTargetController with parameters from ROS
        self.drone_target_controller = DroneTargetController(
            target_offset=self.tracking_target_offset,
            target_altitude=self.track_min_altitude,
            acceptance_radius=self.tracking_acceptance_radius_xy,
            dt=self.timer_period,
            **self.drone_controller_params
        )
        
        # Add parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.param_update_callback)

    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )

    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            TestState.INIT: self._handle_init,
            TestState.TRACK: self._handle_track_target,
            TestState.DESCEND: self._handle_descend,
            TestState.LAND_OR_HOVER: self._handle_land_or_hover,
            TestState.COMPLETE: self._handle_complete,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
        
        self._publish_vehicle_state()
        
        # NOTE: target_component should be set to 0 for gazebo
        self.publish_gimbal_attitude(target_component=154, flags = 12,
                                     q = [0.0, 0.0, 1.0, 0.0])

    def _publish_vehicle_state(self):
        """Publish current vehicle state"""
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=self.detect_target_type if self.state == TestState.TRACK else 0
            )
        )

    def on_target_update(self, msg):
        """Callback for target coordinates from image_processing_node"""
        self.target = msg if msg is not None else None

    # =======================================
    # State Machine Functions
    # =======================================

    def _handle_init(self):
        """Initialize system and check status"""
        if not self.get_position_flag:
            self.get_logger().info("Waiting for position data...")
            return

        self.set_home_position()
        
        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = TestState.TRACK

  
    def _handle_track_target(self):
        """Track target using vision and transition to next state when arrived"""
        if self.target is None or self.target.status != 0:
            self.get_logger().warn("No target coordinates available")
            return

        target_pos, arrived = self.drone_target_controller.update(
            self.pos, self.yaw, self.target.angle_x, self.target.angle_y
        )

        self.publish_setpoint(pos_sp=target_pos)

        if arrived:
            self.drone_target_controller.reset()
            self.get_logger().info("Target tracking complete")
            self.state = TestState.DESCEND

    def _handle_descend(self):
        """Descend to target position"""
        target_pos = np.array([self.pos[0], self.pos[1], -self.gripper_altitude])
        self.publish_setpoint(pos_sp=target_pos)

        if abs(self.pos[2] - (-self.gripper_altitude)) < self.tracking_acceptance_radius_z:
            self.get_logger().info("Reached descent altitude")
            self.state = TestState.LAND_OR_HOVER

    def _handle_land_or_hover(self):
        """Either land or hover based on parameter"""
        if self.land_after_track:
            self.land()
            self.get_logger().info("Landing command sent")
            self.state = TestState.COMPLETE
        else:
            # Just hover at current position
            self.publish_setpoint(pos_sp=self.pos)
            self.get_logger().info("Hovering at target position - Test 04 Complete!")
            self.state = TestState.COMPLETE
            # Could add a timer here to hover for a specific duration

    def _handle_complete(self):
        """Test complete"""
        pass

    def param_update_callback(self, params):
        """Parameter callback for dynamically updating parameters while flying"""
        # NOTE: Copy this function from mc_main and adapt for test 04 parameters
        successful = True
        reason = ''
        
        for p in params:
            # Mission parameters
            if p.name == 'mission_altitude':
                self.mission_altitude = p.value
            elif p.name == 'track_min_altitude':
                self.track_min_altitude = p.value
            elif p.name == 'gripper_altitude':
                self.gripper_altitude = p.value
            elif p.name == 'tracking_target_offset':
                self.tracking_target_offset = p.value
                self.drone_target_controller.target_offset = p.value
            elif p.name == 'tracking_acceptance_radius_xy':
                self.tracking_acceptance_radius_xy = p.value
                self.drone_target_controller.acceptance_radius = p.value
            elif p.name == 'tracking_acceptance_radius_z':
                self.tracking_acceptance_radius_z = p.value
            elif p.name == 'detect_target_type':
                self.detect_target_type = p.value
            elif p.name == 'land_after_track':
                self.land_after_track = p.value
            # DroneTargetController parameters
            elif p.name.startswith('drone_target_controller.'):
                param_key = p.name.replace('drone_target_controller.', '')
            
                self.drone_controller_params[param_key] = p.value
                if hasattr(self.drone_target_controller, param_key):
                    setattr(self.drone_target_controller, param_key, p.value)
            else:
                self.get_logger().warn(f"Ignoring parameter: {p.name}")
                continue
        
        self.get_logger().info("[Parameter Update] Test 04 parameters updated successfully")
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
        print("Test 04 interrupted by user")
    except Exception as e:
        print(f"Test 04 failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()