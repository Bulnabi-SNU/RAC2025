__author__ = "PresidentPlant"
__contact__ = ""

import time
import rclpy
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from enum import Enum

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from vehicle_controller.core.logger import Logger

# import px4_msgs
# from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition

# import math library
import numpy as np

# Custom Messages
from custom_msgs.msg import VehicleState, TargetLocation

class MissionState(Enum):
    INIT = "INIT"
    OFFBOARD_ARM = "OFFBOARD_ARM"
    MISSION_EXECUTE = "MISSION_EXECUTE"
    MISSION_TO_OFFBOARD_CASUALTY = "MISSION_TO_OFFBOARD_CASUALTY"
    CASUALTY_TRACK = "CASUALTY_TRACK"
    CASUALTY_DESCEND = "CASUALTY_DESCEND"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    CASUALTY_ASCEND = "CASUALTY_ASCEND"
    OFFBOARD_TO_MISSION = "OFFBOARD_TO_MISSION"
    MISSION_CONTINUE = "MISSION_CONTINUE"
    MISSION_TO_OFFBOARD_DROP_TAG = "MISSION_TO_OFFBOARD_DROP_TAG"
    DROP_TAG_TRACK = "DROP_TAG_TRACK"
    DROP_TAG_DESCEND = "DROP_TAG_DESCEND"
    GRIPPER_OPEN = "GRIPPER_OPEN"
    DROP_TAG_ASCEND = "DROP_TAG_ASCEND"
    MISSION_TO_OFFBOARD_LANDING_TAG = "MISSION_TO_OFFBOARD_LANDING_TAG"
    LANDING_TAG_TRACK = "LANDING_TAG_TRACK"
    FINAL_DESCEND = "FINAL_DESCEND"
    LAND = "LAND"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"

class MissionController(PX4BaseController):
    """Mission Controller for the actual competition."""
    def __init__(self):
        super().__init__("mc_main")

        self.get_ros_parameters()        

        # Placeholder for target
        self.target = None
        
        # Choose where to override the control loop. 
        # For now, just set position and don't override anything 
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False

        # State machine variables
        self.state = MissionState.INIT  # Initial state
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        
        # Logger
        self.logger = Logger(log_path="./flight_logs/")
        
        # Offboard controller
        # TODO: change target_distance argument name
        self.drone_target_controller = DroneTargetController(
                target_offset=self.tracking_target_offset, 
                target_altitude=self.track_min_altitude, 
                acceptance_radius=self.tracking_acceptance_radius_xy)

        # CV Detection Subscriber
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )
        
        # TODO: Add callback for ROS parameter change      

        self.get_logger().info("Mission Controller initialized")

    def get_ros_parameters(self):
        """Get ROS2 parameters and save them to local variables"""
        self.declare_parameters(
            namespace='',
            parameters=[
                ('casualty_waypoint', 14),
                ('drop_tag_waypoint', 15),
                ('landing_tag_waypoint', 16),
                ('mission_altitude', 15.0),
                ('track_min_altitude', 4.0),
                ('gripper_altitude', 0.3),
                ('tracking_target_offset', 0.35),
                ('tracking_acceptance_radius_xy', 0.2),
                ('tracking_acceptance_radius_z', 0.2),
                ('do_logging', True),
            ])

        self.casualty_waypoint = self.get_parameter('casualty_waypoint').value
        self.drop_tag_waypoint = self.get_parameter('drop_tag_waypoint').value
        self.landing_tag_waypoint = self.get_parameter('landing_tag_waypoint').value

        self.mission_altitude = self.get_parameter('mission_altitude').value
        self.gripper_altitude = self.get_parameter('gripper_altitude').value
        self.track_min_altitude = self.get_parameter('track_min_altitude').value

        self.tracking_target_offset = self.get_parameter('tracking_target_offset').value

        self.tracking_acceptance_radius_xy = \
            self.get_parameter('tracking_acceptance_radius_xy').value
        self.tracking_acceptance_radius_z = \
            self.get_parameter('tracking_acceptance_radius_z').value
            
        self.do_log_flight = self.get_parameter('do_logging').value
    
    def main_loop(self):
        """Main control loop - implements the state machine"""

        if self.state == MissionState.INIT:
            self._handle_init()
        elif self.state == MissionState.OFFBOARD_ARM:
            self._handle_offboard_arm()
        elif self.state == MissionState.OFFBOARD_TO_MISSION:
            self._handle_offboard_to_mission()
        elif self.state == MissionState.MISSION_EXECUTE:
            self._handle_mission_execute()
        elif self.state == MissionState.MISSION_TO_OFFBOARD_CASUALTY:
            self._handle_mission_to_offboard(MissionState.CASUALTY_TRACK)
        elif self.state == MissionState.CASUALTY_TRACK:
            self._handle_track_target(MissionState.CASUALTY_DESCEND)
        elif self.state == MissionState.CASUALTY_DESCEND:
            self._handle_descend(MissionState.GRIPPER_CLOSE, self.gripper_altitude)
        elif self.state == MissionState.GRIPPER_CLOSE:
            self._handle_gripper_close()
        elif self.state == MissionState.CASUALTY_ASCEND:
            self._handle_ascend(ascendAlt=self.mission_altitude)
        elif self.state == MissionState.MISSION_CONTINUE:
            self._handle_mission_continue()
        elif self.state == MissionState.MISSION_TO_OFFBOARD_DROP_TAG:
            self._handle_mission_to_offboard(MissionState.DROP_TAG_TRACK)
        elif self.state == MissionState.DROP_TAG_TRACK:
            self._handle_track_target(MissionState.DROP_TAG_DESCEND)
        elif self.state == MissionState.DROP_TAG_DESCEND:
            self._handle_descend(MissionState.GRIPPER_OPEN, self.gripper_altitude)
        elif self.state == MissionState.GRIPPER_OPEN:
            self._handle_gripper_open()
        elif self.state == MissionState.DROP_TAG_ASCEND:
            self._handle_ascend(ascendAlt=self.mission_altitude)
        elif self.state == MissionState.MISSION_TO_OFFBOARD_LANDING_TAG:
            self._handle_mission_to_offboard(MissionState.LANDING_TAG_TRACK)
        elif self.state == MissionState.LANDING_TAG_TRACK:
            self._handle_track_target(MissionState.FINAL_DESCEND)
        elif self.state == MissionState.FINAL_DESCEND:
            self._handle_final_descend()
        elif self.state == MissionState.LAND:
            self._handle_land()
        elif self.state == MissionState.MISSION_COMPLETE:
            self._handle_mission_complete()
        elif self.state == MissionState.ERROR:
            self._handle_error()

        target_type_dict = {
            # Target type 1 - Casualty
            MissionState.CASUALTY_TRACK: 1,
            MissionState.CASUALTY_DESCEND: 1,
            MissionState.GRIPPER_CLOSE: 1,
            MissionState.CASUALTY_ASCEND: 1,
            
            # Target type 2 - Drop tag
            MissionState.DROP_TAG_TRACK: 2,
            MissionState.DROP_TAG_DESCEND: 2,
            MissionState.GRIPPER_OPEN: 2,
            MissionState.DROP_TAG_ASCEND: 2,
            
            # Target type 3 - Landing Tag
            MissionState.LANDING_TAG_TRACK: 3,
            MissionState.FINAL_DESCEND: 3,
        }

        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type= target_type_dict.get(self.state, 0)
            )
        )

    def on_target_update(self, msg):
        """Callback for target coordinates from image_processing_node"""
        
        # NOTE: Why the hell is this here? lmao
        if msg is not None:
            self.target = msg
        else:
            self.target = None

    # =======================================
    # State Machine Functions
    # =======================================

    def _handle_init(self):
        """Initialize system and check status"""
        if not self.get_position_flag:
            self.get_logger().info("Waiting for global position data...")
            return

        self.set_home_position()

        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = MissionState.OFFBOARD_ARM

    def _handle_offboard_arm(self):
        """Wait for offboard mode, then arm / start the mission"""
        if not self.is_offboard_mode():
            return

        if self.is_disarmed():
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, starting logger")
            
            if self.do_log_flight:
                self.logger.start_logging()
                self.log_timer = self.create_timer(0.1, self._log_timer_callback)
            
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _handle_offboard_to_mission(self):
        """Resume mission mode after pickup"""
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.state = MissionState.MISSION_CONTINUE

    def _handle_mission_continue(self):
        """Continue mission from paused waypoint"""
        # TODO: Debate whether to remove this? Probably should. Doesn't really mean much.
        self.get_logger().info(
            f"Resuming mission from waypoint {self.mission_paused_waypoint}"
        )
        self.target = None
        self.state = MissionState.MISSION_EXECUTE
        
    def _handle_mission_execute(self):
        """Execute mission"""
        if not self.is_mission_mode():
            #    self.get_logger().error("Not in mission mode at MISSION_EXECUTE")
            return

        current_wp = self.mission_wp_num

        # Check if reached pickup waypoint
        if current_wp == self.casualty_waypoint and not self.pickup_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_CASUALTY
            return

        # Check if reached dropoff waypoint
        if current_wp == self.drop_tag_waypoint and not self.dropoff_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_DROP_TAG
            return

        # Check if reached landing waypoint
        if current_wp == self.landing_tag_waypoint:
            self.state = MissionState.MISSION_TO_OFFBOARD_LANDING_TAG
            return

    def _handle_mission_to_offboard(self, nextState: MissionState):
        """Switch from mission to offboard for pickup"""
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = nextState

    def _handle_track_target(self, nextState: MissionState):
        """Track target using vision and transition to next state when arrived"""
        # TODO: (Maybe) add logic to make it so that intermittent losses (e.g. loss for 1-2 frames) is ignored.
        # Only continuous loss of target should be seen as fatal. 

        if self.target is None or self.target.status != 0:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return

        target_pos, arrived = self.drone_target_controller.update(
            self.pos, self.yaw, self.target.angle_x, self.target.angle_y
        )

        self.publish_setpoint(pos_sp = target_pos)

        if arrived:
            self.drone_target_controller.reset()
            self.state = nextState

    def _handle_descend(self, nextState: MissionState, descendAlt: float = 0.5):
        """Descend to casualty pickup position"""
        # Set position setpoint to pickup altitude
        # NOTE: you can use slewing if it's needed. 
        pickup_pos = np.array([self.pos[0], self.pos[1], 0])
        self.publish_setpoint(pos_sp=pickup_pos)

        # Check if at pickup altitude
        if -self.pos[2] < descendAlt:
            # Stop moving
            self.publish_setpoint(pos_sp=self.pos)
            self.state=nextState

    def _handle_gripper_close(self):
        """Close gripper to pick up casualty"""
        # TODO: Implement gripper control
        
        self.get_logger().info("Closing gripper to pick up casualty")

        # TODO: Add logic to check if this state is complete
        if True:
            self.pickup_complete = True
            self.state = MissionState.CASUALTY_ASCEND

    def _handle_gripper_open(self):
        """Open gripper to release casualty at dropoff point"""
        # TODO: Implement gripper control
        
        self.get_logger().info("Opening gripper to release casualty at dropoff point")
        
        # TODO: Add logic to check if this state is complete
        if True:
            self.dropoff_complete = True
            self.state = MissionState.DROP_TAG_ASCEND

    def _handle_ascend(self, 
                       nextState: MissionState=MissionState.OFFBOARD_TO_MISSION,
                       ascendAlt: float = 15.0):
        """Return to mission altitude with casualty"""
        ascend_pos = np.array([self.pos[0], self.pos[1], -ascendAlt])
        self.publish_setpoint(pos_sp=ascend_pos)

        if abs((-self.pos[2]) - ascendAlt) < self.tracking_acceptance_radius_z:
            self.state = nextState

    # TODO: maybe remove this state depending on how stable landing mode is?
    def _handle_final_descend(self):
        """Descend for landing"""
        landing_pos = np.array([self.pos[0], self.pos[1], 0])
        self.publish_setpoint(pos_sp=landing_pos)

        if abs(self.pos[2] - self.gripper_altitude) < self.tracking_acceptance_radius_z:
            self.state = MissionState.LAND

    def _handle_land(self):
        """Final landing sequence"""
        self.land()
        self.get_logger().info("Landing command sent")
        self.state = MissionState.MISSION_COMPLETE

    def _handle_mission_complete(self):
        """Mission finished"""
        self.get_logger().info("Mission Complete!")
        pass

    def _handle_error(self):
        """Error handling state"""
        self.get_logger().error("Mission in error state")
        # TODO: Implement error recovery or emergency procedures
        pass

    # =======================================
    # Additional Functions
    # =======================================
    
    def _log_timer_callback(self):
        """Timer callback to log vehicle data"""
        if self.logger is None:
            self.get_logger().warn("logger called while not initalized")
            return

        # Log current vehicle state
        auto_flag = 0 if self.state is MissionState.INIT else 1
        
        # TODO: Set event_flag in accordance with mission regulations
        event_flag = self.mission_wp_num 
        
        gps_time = self.vehicle_gps.time_utc_usec / 1e6  # Convert microseconds to seconds
        lat = self.vehicle_gps.latitude_deg
        long = self.vehicle_gps.longitude_deg
        alt = self.vehicle_gps.altitude_ellipsoid_m

        self.logger.log_data(auto_flag, event_flag, gps_time, lat, long, alt)

    def on_vehicle_status_update(self, msg):
        """Override to handle vehicle status updates"""
        # Could add additional status monitoring here
        pass

    def on_local_position_update(self, msg):
        """Override to handle local position updates"""
        # Could add position monitoring here
        pass

    def on_attitude_update(self, msg):
        """Stabilize gimbal manually (for gazebo only, this is a placeholder)"""

        def quaternion_multiply(q1, q2):
            """
            Multiply two quaternions q1 * q2
            Quaternions are in format [w, x, y, z]
            """
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
                w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
                w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
                w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
            ])
        
        def quaternion_conjugate(q):
            """
            Calculate the conjugate of a quaternion
            """
            w, x, y, z = q
            return np.array([w, -x, -y, -z])
        
        def quaternion_normalize(q):
            """
            Normalize a quaternion
            """
            return q /np.linalg.norm(q) 
        
        
        def calculate_gimbal_quaternion(q_drone):
            """
            Calculate the gimbal quaternion q_gimbal such that:
            q_gimbal * q_drone = q_final
            where q_final has pitch = -90° and same yaw as q_drone
            
            Args:
                q_drone: numpy array [w, x, y, z] representing drone quaternion
            
            Returns:
                q_gimbal: numpy array [w, x, y, z] representing gimbal quaternion
            """
            # Normalize input quaternion
            #q_drone = [1,0,0,0]
            q_drone = quaternion_normalize(q_drone)

            #q_final = quaternion_multiply([np.cos(self.yaw/2),0.0,.0,-np.sin(self.yaw/2)] ,[1,0,0.0,0])
            q_final = [1.0,0.,0.,0.]
            
            # Extract yaw from drone quaternion
            # roll_drone, pitch_drone, yaw_drone = quaternion_to_euler(q_drone)
            
            # Create desired final quaternion: roll=0, pitch=-90°, yaw=same as drone
            #q_final = euler_to_quaternion(0, -np.pi/2, yaw_drone)
            q_final = q_final
            
            # Calculate gimbal quaternion: q_gimbal = q_final * q_drone_conjugate
            q_drone_conjugate = quaternion_conjugate(q_drone)
            q_gimbal = quaternion_multiply(q_final, q_drone_conjugate)
            

            # FUCK THIS SHIT GAZEBO SUCKS
            return quaternion_normalize(np.array([0.7,0.0,-0.7,0.0]))

        if False:
            return

        gimbal_q = calculate_gimbal_quaternion(self.attitude_q)
        #self.get_logger().info("Publishing gimbal quaternions") 
        #print(gimbal_q)
        # If this works i'll be happy
        # self.publish_gimbal_attitude(flags=12)

        # Probably won't though.
        #self.publish_gimbal_attitude(flags=12,q=[0.7,0.,-0.7,0.0])
        #self.publish_gimbal_attitude(flags=12,q=[0.6,0.,-0.0,0.0])
        self.publish_gimbal_attitude(flags=0,q=gimbal_q)

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
        if "controller" in locals():
            controller.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
