__author__  = "SeongWon"
__contact__ = ""

# ----------------------------------------------------
# ROS / PX4 imports
# ----------------------------------------------------
import rclpy
from rclpy.qos          import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from vehicle_controller.code_basic.px4_base import PX4BaseController
from vehicle_controller.code_basic.bezier_handler import BezierCurve

from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition
from custom_msgs.msg import LandingTagLocation # AprilTag detector output (tag pose relative to camera)

import numpy as np
import math


class AutoLandingController(PX4BaseController):
    """
    Autonomous landing mission
    1. 대회 규정 고도(예: 5 m)까지 이륙
    2. AprilTag 탐지 → 태그 상공 0.5 m 목표점 설정
    3. Bezier 곡선으로 목표점까지 수평 이동
    4. 하강 속도 0.3 m/s(Down +)로 접지
    5. LAND 커맨드 전송
    """

    # ------------------------------------------------
    # Tunables
    # ------------------------------------------------
    TAG_TOPIC           = "/landing_tag_position"   # AprilTag `PoseStamped` (tag pose in camera frame)
    TARGET_OFFSET_Z     = 0.5                      # [m] 태그 상공 (positive value means above the tag in NED-Z, so reduce NED Z)
    DESCENT_SPEED_D     = 0.3                      # [m/s]  (NED Down +)
    ARRIVAL_RADIUS_XY   = 0.25                     # [m] Bezier 끝점 XY 수렴 기준
    LAND_ALT_THRESH     = 0.05                     # [m] z(D) 값이 이보다 작으면 착지로 판정 (for controlled descent before PX4 land)
    VMAX_XY             = 3.0                      # [m/s] Bezier 생성 시 사용
    TAG_LOST_TIMEOUT    = 2.0                      # [s] 태그를 찾지 못했을 때 SEARCH 단계로 돌아갈 시간

    # ------------------------------------------------
    def __init__(self):
        super().__init__('mc_auto_land')

        # ───── Mission-level state ──────────────────
        self.state          = 'READY_TO_FLIGHT'    # READY → TAKEOFF → SEARCH → ALIGN → DESCEND → LANDING → COMPLETE
        self.bezier         = BezierCurve(time_step=0.05)
        self.bezier_i       = 0
        self.path           = None

        # ───── AprilTag data ────────────────────────
        self.has_tag        = False
        self.tag_rel_ned    = np.zeros(3) # Tag's position relative to drone in NED frame
        self.last_tag_time  = self.get_clock().now() # For tag lost timeout

        # Match QOS profile with image_processing_node when subscribing for landingtag, etc....
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,   # Change back to TRANSIENT_LOCAL for actual test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 아니면 에러 뜸
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # AprilTag subscriber
        self.create_subscription(
            LandingTagLocation,
            self.TAG_TOPIC,
            self.image_callback,
            qos_profile
        )

        self.offboard_control_mode_params['position'] = True
        self.offboard_control_mode_params['velocity'] = False

        self.get_logger().info("Auto-landing controller initialised")




    # =================================================
    # Main state-machine loop (called by PX4Base)
    # =================================================
    def main_loop(self):
        # Always check for tag loss, especially during ALIGN and DESCEND
        if self.state in ['ALIGN', 'DESCEND'] and \
           (self.get_clock().now() - self.last_tag_time).nanoseconds / 1e9 > self.TAG_LOST_TIMEOUT:
            self.get_logger().warn(f"Tag lost for {self.TAG_LOST_TIMEOUT}s! Re-entering SEARCH phase.")
            self.has_tag = False # Explicitly set to false
            self.state = 'SEARCH' # Go back to search
            self.path = None # Clear current bezier path

        if self.state == 'READY_TO_FLIGHT':
            self._ready_to_flight()

        elif self.state == 'TAKEOFF':
            self._takeoff_phase()

        elif self.state == 'SEARCH':
            self._search_phase()

        elif self.state == 'ALIGN':
            self._align_phase()

        elif self.state == 'DESCEND':
            self._descent_phase()

        elif self.state == 'LANDING':
            self._landing_phase()

        elif self.state == 'COMPLETE':
            pass

    # ==============================
    # Phase handlers
    # ==============================
    def _ready_to_flight(self):
        """
        Arm → Takeoff (or switch to AUTO.TAKEOFF) as soon as offboard is allowed
        """
        if self.is_offboard_mode():
            if self.is_disarmed():
                self.arm()
                self.get_logger().info("Arming…")
            else:
                # PX4Base helper à 이륙 고도 5 m
                # Note: `takeoff` usually publishes a position setpoint,
                # PX4 will then transition to AUTO.TAKEOFF/LOITER
                self.takeoff()
                self.get_logger().info("Takeoff command sent")
                self.state = 'TAKEOFF'

    def _takeoff_phase(self):
        """
        Wait until PX4 reports LOITER (take-off complete) then start searching tag
        """
        if self.is_auto_loiter(): # is_auto_loiter() checks if vehicle is in LOITER state.
            self.get_logger().info("Take-off done → switching to tag search")
            self.state = 'SEARCH'

    def _search_phase(self):
        """
        Hover until tag detected
        """
        if not self.has_tag:
            # Keep offboard alive by publishing current position setpoint
            # This implicitly means the drone will hover at its current position after takeoff.
            self.publish_setpoint(pos_sp=self.pos)
            self.get_logger().info("Searching for AprilTag...")
            return

        # Once first tag pose is received → generate Bezier path
        # self.pos: drone's current position in NED (local)
        # self.tag_rel_ned: tag's position relative to drone in NED
        # So, drone's position + tag's relative position = tag's absolute position in local NED
        target_ned = self.pos + self.tag_rel_ned # coordinate transformation 잘 해주기
        
        # Adjust Z to be TARGET_OFFSET_Z meters *above* the tag
        # If target_ned[2] is positive down, to move up (above tag), we decrease the Z value.
        target_ned[2] -= self.TARGET_OFFSET_Z

        self.bezier.generate_curve(
            start_pos    = self.pos,
            end_pos      = target_ned,
            start_vel    = self.vel,
            end_vel      = np.zeros(3), # End velocity zero to hover
            max_velocity = self.VMAX_XY,
            total_time   = None # Bezier handler calculates time based on max_velocity
        )
        self.path       = self.bezier.trajectory_points
        self.bezier_i   = 0
        self.state      = 'ALIGN'
        self.get_logger().info(f"Tag locked at local NED: ({target_ned[0]:.2f}, {target_ned[1]:.2f}, {target_ned[2]:.2f}) – generated Bezier path, entering ALIGN")

    def _align_phase(self):
        """
        Publish successive Bezier set-points until within ARRIVAL_RADIUS_XY of the Bezier end point.
        """
        if self.path is None: # Should not happen if _search_phase works correctly
            self.get_logger().error("Bezier path is null in ALIGN phase. Returning to SEARCH.")
            self.state = 'SEARCH'
            self.has_tag = False # Force re-detection
            return

        # Check if we have processed all points in the path
        if self.bezier_i >= len(self.path):
            # Bezier path is exhausted, but we might not have met the arrival radius condition yet.
            self.get_logger().warning("Bezier path exhausted. Forcing DESCEND or waiting for convergence.")
            # If path is done, check distance to final target.
            dist_to_final_target_xy = np.linalg.norm(self.path[-1][:2] - self.pos[:2])
            if dist_to_final_target_xy < self.ARRIVAL_RADIUS_XY:
                self.get_logger().info("Reached hover point above tag (via path exhaustion) → start descent")
                self.state = 'DESCEND'
            else:
                # If still outside, hover at the end of the path.
                self.publish_setpoint(pos_sp=self.path[-1])
                self.get_logger().info(f"Hovering at end of Bezier path, waiting for convergence. Distance: {dist_to_final_target_xy:.2f}m")
            return

        next_sp = self.path[self.bezier_i]
        self.publish_setpoint(pos_sp = next_sp)
        self.bezier_i += 1

        # Check distance to the final target of the Bezier path (self.path[-1]), not the current setpoint
        dist_to_final_target_xy = np.linalg.norm(self.path[-1][:2] - self.pos[:2])

        if dist_to_final_target_xy < self.ARRIVAL_RADIUS_XY:
            self.get_logger().info("Reached hover point above tag → start descent")
            self.state = 'DESCEND'

    def _descent_phase(self):
        """
        Send velocity set-point straight down until z (D) ≤ threshold,
        then issue LAND command (or directly issue LAND command).
        """
        # Option 1: Controlled descent until threshold, then PX4 LAND command
        # This gives you finer control over the initial descent speed.
        self.publish_setpoint(vel_sp = np.array([0.0, 0.0, self.DESCENT_SPEED_D]),
                              switch_to_velocity=True)
        self.get_logger().info(f"Descending. Current altitude (NED-Z): {self.pos[2]:.2f}m")

        if self.pos[2] >= (self.path[-1][2] + self.TARGET_OFFSET_Z - self.LAND_ALT_THRESH): # Check if drone is close to the tag's ground level
            # More robust check: use the target Z from ALIGN phase, then add a small threshold
            # Example: self.path[-1][2] is the target Z over the tag, so check when current Z is close to that.
            self.get_logger().info(f"Altitude {self.pos[2]:.2f}m. Sending LAND command.")
            self.land() # Issue PX4's built-in LAND command
            self.state = 'LANDING'

        # Option 2: Directly switch to PX4's LAND mode from ALIGN phase (more robust for final touchdown)
        # If you prefer this, _descent_phase can be simplified to just calling self.land() once.
        # This means the _align_phase would transition directly to LANDING after arrival.
        # def _descent_phase(self):
        #     self.get_logger().info("Initiating PX4 LAND command.")
        #     self.land()
        #     self.state = 'LANDING'


    def _landing_phase(self):
        """
        No action – wait until motors disarm automatically
        """
        if self.is_disarmed():
            self.get_logger().info("Landed & disarmed. Mission complete!")
            self.state = 'COMPLETE'

    # ==============================
    # Callbacks
    # ==============================
    def image_callback(self, msg: LandingTagLocation):
        """
        Converts camera-frame tag pose → NED relative vector.
        Assumes msg.pose.position is the tag's position relative to the camera.
        Assumes camera is mounted such that:
        Camera +X ~ Drone +X (North/Forward)
        Camera +Y ~ Drone +Y (East/Right)
        Camera +Z ~ Drone +Z (Down)

        Based on these assumptions, the transformation would be direct.
        If your ImageProcessor provides camera position relative to tag, the logic needs to be flipped.
        """
        p_cam = np.array([msg.x,
                          msg.y,
                          msg.z])
        
        # Assuming camera +X is drone +X (North), camera +Y is drone +Y (East), camera +Z is drone +Z (Down)
        # This implies:
        # p_cam[0] -> relative North
        # p_cam[1] -> relative East
        # p_cam[2] -> relative Down
        # So, the tag's position relative to the drone in NED is simply p_cam.
        self.tag_rel = np.array([ -p_cam[1],  # Tag's X in camera frame -> Drone's X (North)
                                      -p_cam[0],  # Tag's Y in camera frame -> Drone's Y (East)
                                      p_cam[2] ]) # Tag's Z in camera frame -> Drone's Z (Down)
        R_z = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                        [np.sin(self.yaw), np.cos(self.yaw), 0],
                        [0, 0, 1]])
        self.tag_rel_ned = np.dot(R_z,self.tag_rel)

        # Update tag detection status and timestamp
        self.has_tag = True
        self.last_tag_time = self.get_clock().now()
        # self.get_logger().info(f"AprilTag detected at relative NED: {self.tag_rel_ned}")


    # (Vehicle status / position hooks can be overridden if desired)
    def on_vehicle_status_update(self, msg: VehicleStatus):
        pass # Call base class method
        # Add any specific logic for auto-landing if needed
    def on_local_position_update(self, msg):
        pass # Call base class method
        # Add any specific logic for auto-landing if needed
    def on_global_position_update(self, msg: VehicleGlobalPosition):
        pass # Call base class method
        # Add any specific logic for auto-landing if needed


# =====================================================
# Entrypoint
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    try:
        ctl = AutoLandingController()
        rclpy.spin(ctl)
    except KeyboardInterrupt:
        pass
    finally:
        if 'ctl' in locals():
            ctl.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()