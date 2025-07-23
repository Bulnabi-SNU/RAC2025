__author__  = "SeongWon"
__contact__ = ""

# ----------------------------------------------------
# ROS / PX4 imports
# ----------------------------------------------------
import rclpy
from rclpy.qos               import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from vehicle_controller.code_basic.px4_base     import PX4BaseController
from vehicle_controller.code_basic.bezier_handler import BezierCurve

from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition

import numpy as np
import math
import pymap3d as p3d
from geometry_msgs.msg import PoseStamped        # AprilTag detector output


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
    TAG_TOPIC          = "/tag_detections/pose"   # AprilTag `PoseStamped`
    TARGET_OFFSET_Z    = 0.5                      # [m] 태그 상공
    DESCENT_SPEED_D    = 0.3                      # [m/s]  (NED Down +)
    ARRIVAL_RADIUS_XY  = 0.25                     # [m] Bezier 끝점 XY 수렴 기준
    LAND_ALT_THRESH    = 0.05                     # [m] z(D) 값이 이보다 작으면 착지로 판정
    VMAX_XY            = 3.0                      # [m/s] Bezier 생성 시 사용

    # ------------------------------------------------
    def __init__(self):
        super().__init__('mc_auto_land')

        # ───── Mission-level state ──────────────────
        self.state     = 'READY_TO_FLIGHT'        # READY → TAKEOFF → SEARCH → ALIGN → DESCEND → LANDING → COMPLETE
        self.bezier    = BezierCurve(time_step=0.05)
        self.bezier_i  = 0
        self.path      = None

        # ───── AprilTag data ────────────────────────
        self.has_tag   = False
        self.tag_rel_ned = np.zeros(3)

        # AprilTag subscriber
        self.create_subscription(
            PoseStamped,
            self.TAG_TOPIC,
            self._on_tag_pose,
            QoSProfile(depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history   =HistoryPolicy.KEEP_LAST)
        )

        self.offboard_control_mode_params['position'] = True
        self.offboard_control_mode_params['velocity'] = False

        self.get_logger().info("Auto-landing controller initialised")

    # =================================================
    # Main state-machine loop (called by PX4Base)
    # =================================================
    def main_loop(self):
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
                self.takeoff( altitude_m=5.0 )   # PX4Base helper à 이륙 고도 5 m
                self.get_logger().info("Takeoff command sent")
                self.state = 'TAKEOFF'

    def _takeoff_phase(self):
        """
        Wait until PX4 reports LOITER (take-off complete) then start searching tag
        """
        if self.is_auto_loiter():
            self.get_logger().info("Take-off done → switching to tag search")
            self.state = 'SEARCH'

    def _search_phase(self):
        """
        Hover until tag detected
        """
        if not self.has_tag:
            # keep offboard alive but no new set-point
            self.publish_setpoint(pos_sp=self.pos)
            return

        # Once first tag pose is received → generate Bezier path
        target_ned = self.pos + self.tag_rel_ned
        target_ned[2] += self.TARGET_OFFSET_Z   # 상공 0.5 m

        self.bezier.generate_curve(
            start_pos = self.pos,
            end_pos   = target_ned,
            start_vel = self.vel,
            end_vel   = np.zeros(3),
            max_velocity = self.VMAX_XY,
            total_time   = None
        )
        self.path     = self.bezier.path
        self.bezier_i = 0
        self.state    = 'ALIGN'
        self.get_logger().info("Tag locked – generated Bezier path, entering ALIGN")

    def _align_phase(self):
        """
        Publish successive Bezier set-points until within ARRIVAL_RADIUS_XY
        """
        if self.bezier_i >= len(self.path):
            self.get_logger().warning("Bezier finished but radius check failed — forcing DESCEND")
            self.state = 'DESCEND'
            return

        next_sp = self.path[self.bezier_i]
        self.publish_setpoint(pos_sp = next_sp)
        self.bezier_i += 1

        dist_xy = np.linalg.norm(next_sp[:2] - self.pos[:2])

        if dist_xy < self.ARRIVAL_RADIUS_XY:
            self.get_logger().info("Reached hover point above tag → start descent")
            self.state = 'DESCEND'

    def _descent_phase(self):
        """
        Send velocity set-point straight down until z (D) ≤ threshold,
        then issue LAND command
        """
        self.publish_setpoint(vel_sp = np.array([0.0, 0.0, self.DESCENT_SPEED_D]),
                              switch_to_velocity=True)

        if abs(self.pos[2]) < self.LAND_ALT_THRESH:
            self.get_logger().info("Ground detected (< 5 cm). Sending LAND command.")
            self.land()
            self.state = 'LANDING'

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
    def _on_tag_pose(self, msg: PoseStamped):
        """
        Convert camera-frame tag pose → NED relative vector.
        Frame transform depends on your camera TF; here Z is flipped.
        """
        p_cam = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          msg.pose.position.z])
        self.tag_rel_ned = np.array([ p_cam[0],          # +X body-forward → +N
                                      p_cam[1],          # +Y body-right   → +E
                                     -p_cam[2] ])        # camera +Z down  → NED –Z(up)
        self.has_tag = True

    # (Vehicle status / position hooks can be overridden if desired)
    def on_vehicle_status_update(self, msg: VehicleStatus):
        pass
    def on_local_position_update(self, msg):
        pass
    def on_global_position_update(self, msg: VehicleGlobalPosition):
        pass


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
