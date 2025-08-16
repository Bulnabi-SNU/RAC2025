"""
Test 05: Gimbal Quaternion Test
Tests gimbal control by setting different gimbal orientations
Start this test midair - no takeoff/landing logic
"""

__author__ = "PresidentPlant"
__contact__ = ""

import rclpy
import numpy as np
import math
from enum import Enum
from vehicle_controller.core.px4_base import PX4BaseController
from custom_msgs.msg import VehicleState
from rcl_interfaces.msg import SetParametersResult




class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_gimbal')
        
        self.get_logger().info("Test 05: Gimbal Control initialized")

    def main_loop(self):
        
        # NOTE: Siyi may not like quaternions and flags (yaw follow etc.)
        # https://mavlink.io/en/messages/common.html#GIMBAL_MANAGER_FLAGS
        # If so, change to GimbalManagerSetManualControl instead.
        
        # NOTE v2: Updated firmware made everything work well. So target component 0, flag 12, and q 0.7 0 -0.7 0 works super nice.
        self.publish_gimbal_attitude(target_component=154, 
                                     flags = 4+8+32, 
                                     # 4: roll lock, 8: pitch lock, 32: GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME
                                     q = [0.0, 0.0, 1.0, 0.0])
        
        self._publish_vehicle_state()

    def _publish_vehicle_state(self):
        """Publish current vehicle state"""
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=0  
            )
        )

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
        print("Test 05 interrupted by user")
    except Exception as e:
        print(f"Test 05 failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()