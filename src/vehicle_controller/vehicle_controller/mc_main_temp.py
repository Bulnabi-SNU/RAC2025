# 중원, 실행할거면 이거 다 주석처리하고 너 거 붙이면 될 듯

import os
import time
import rclpy
from rcl_interfaces.msg import SetParametersResult

import numpy as np
from enum import Enum
from typing import Optional

from vehicle_controller.core.px4_base import PX4BaseController
from vehicle_controller.core.drone_target_controller import DroneTargetController
from vehicle_controller.core.logger import Logger
from custom_msgs.msg import VehicleState, TargetLocation
from px4_msgs.msg import VehicleAcceleration


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
    LAND = "LAND"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"

    # >>> ADDED: 상세 고도정렬·탐지 전환 상태들
    NAV_TO_CASUALTY_35 = "NAV_TO_CASUALTY_35"
    ALIGN_20 = "ALIGN_20"
    ALIGN_15 = "ALIGN_15"
    ALIGN_10 = "ALIGN_10"
    GREEN_DESCENT = "GREEN_DESCENT"
    SWITCH_TO_RED = "SWITCH_TO_RED"
    RED_STABILITY_CHECK = "RED_STABILITY_CHECK"
    RETRY_ASCEND_10 = "RETRY_ASCEND_10"

    NAV_TO_DROPTAG_35 = "NAV_TO_DROPTAG_35"
    DROP_ALIGN_20 = "DROP_ALIGN_20"
    DROP_ALIGN_15 = "DROP_ALIGN_15"
    DROP_ALIGN_10 = "DROP_ALIGN_10"
    DROPTAG_DESCENT = "DROPTAG_DESCENT"


class MissionController(PX4BaseController):
    """Mission Controller for the actual competition."""

    # Constants
    TARGET_TYPES = {
        MissionState.CASUALTY_TRACK: 1,
        MissionState.CASUALTY_DESCEND: 1,
        MissionState.GRIPPER_CLOSE: 1,
        MissionState.CASUALTY_ASCEND: 1,
        MissionState.DROP_TAG_TRACK: 2,
        MissionState.DROP_TAG_DESCEND: 2,
        MissionState.GRIPPER_OPEN: 2,
        MissionState.DROP_TAG_ASCEND: 2,
        MissionState.LANDING_TAG_TRACK: 3,

        # >>> ADDED: 새 상태에 대한 타입 매핑
        MissionState.GREEN_DESCENT: 1,
        MissionState.RED_STABILITY_CHECK: 1,
        MissionState.DROPTAG_DESCENT: 2
    }

    def __init__(self):
        super().__init__("mc_main")
        
        self._load_parameters()
        self._initialize_components()
        self._setup_subscribers()
        self.vehicle_acc = VehicleAcceleration()

        self.state = MissionState.INIT
        self.target: Optional[TargetLocation] = None
        self.mission_paused_waypoint = 0
        self.pickup_complete = False
        self.dropoff_complete = False
        self.target_position = None  # Store position when entering descend/ascend

        # >>> ADDED: 내부 상태/디텍션 모드
        self._state_entered_t = time.time()
        self._detector_mode = "green"  # green | red | drop_tag
        self._set_detector_mode("green")

        self.get_logger().info("Mission Controller initialized")

    def _load_parameters(self):
        """Load ROS parameters"""
        # Mission parameters (기존)
        params = [
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
        ]

        # >>> ADDED: 절대좌표/정렬/안정성/디텍션 파라미터
        params += [
            ('basket_xy', [0.0, 0.0]),      # ENU [x, y]
            ('drop_tag_xy', [0.0, 0.0]),    # ENU [x, y]
            ('arrive_altitude', 35.0),      # 35 m 상공
            ('arrive_radius', 5.0),         # 반경 5 m 판정
            ('green_stop_ratio', 0.40),     # 초록 면적비 트리거
            ('red_stable_secs', 10.0),      # 빨강 안정성 시간
            ('retry_altitude', 10.0),       # 실패 시 재상승 고도
            ('descent_speed', 0.7),         # 하강 속도 느낌값
            ('image_node_name', 'image_processing_node'),
            ('detection_mode_param_key', 'detection_mode'),
        ]
        
        # DroneTargetController parameters (기존)
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
        
        # Declare all parameters
        self.declare_parameters(namespace='', parameters=params + drone_controller_params)
        
        # Cache mission parameter values
        for param_name, _ in params:
            setattr(self, param_name, self.get_parameter(param_name).value)
        
        # Store drone controller parameters in a dict
        self.drone_controller_params = {}
        for param_name, _ in drone_controller_params:
            actual_name = param_name.replace('drone_target_controller.', '')
            self.drone_controller_params[actual_name] = self.get_parameter(param_name).value

    def _initialize_components(self):
        """Initialize controllers and logger"""
        self.offboard_control_mode_params["position"] = True
        self.offboard_control_mode_params["velocity"] = False
        
        self.logger = Logger(log_path="/workspace/flight_logs")
        self.log_timer = None
        
        self.add_on_set_parameters_callback(self.param_update_callback)
        
        # Initialize DroneTargetController with parameters from ROS
        self.drone_target_controller = DroneTargetController(
            target_offset=self.tracking_target_offset,
            target_altitude=self.track_min_altitude,
            acceptance_radius=self.tracking_acceptance_radius_xy,
            dt=self.timer_period,
            **self.drone_controller_params
        )

    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        self.target_subscriber = self.create_subscription(
            TargetLocation, "/target_position", self.on_target_update, self.qos_profile
        )
        self.accel_subscriber = self.create_subscription(
            VehicleAcceleration,
            "/fmu/out/vehicle_acceleration",
            self.on_vehicle_accel_update,
            self.qos_profile
        )

    # >>> ADDED: 유틸(상태진입, ENU setpoint, 도착판정, 고도도달, 디텍션모드)
    def _now(self) -> float:
        return time.time()

    def _enter(self, new_state: MissionState):
        self.state = new_state
        self._state_entered_t = self._now()
        self.get_logger().info(f"→ STATE: {new_state.value}")

    def _enu_goto(self, x: float, y: float, z_up: float):
        self.publish_setpoint(pos_sp=np.array([x, y, -z_up]))  # PX4 내부 NED 가정

    def _hold_alt(self, z_up: float):
        self._enu_goto(self.pos[0], self.pos[1], z_up)

    def _arrived_xy(self, target_xy, radius_m) -> bool:
        dx = self.pos[0] - target_xy[0]
        dy = self.pos[1] - target_xy[1]
        return (dx*dx + dy*dy) ** 0.5 <= radius_m

    def _alt_reached(self, alt_up: float, tol: float = 0.5) -> bool:
        return abs(self.pos[2] - (-alt_up)) < tol

    def _set_detector_mode(self, mode: str):
        try:
            from rclpy.parameter import Parameter
            self._detector_mode = mode
            self.get_logger().info(f"[detector] set mode → {mode}")
            self.set_parameters_atomically(
                node_name=self.image_node_name,
                parameters=[Parameter(
                    name=self.detection_mode_param_key,
                    type_=Parameter.Type.STRING,
                    value=mode
                )]
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to set detector mode: {e}")

    # =======================================
    # 메인 루프 (상태 디스패치)
    # =======================================
    def main_loop(self):
        """Main control loop - implements the state machine"""
        state_handlers = {
            # (기존) — 그대로 유지
            MissionState.INIT: self._handle_init,
            MissionState.OFFBOARD_ARM: self._handle_offboard_arm,
            MissionState.OFFBOARD_TO_MISSION: self._handle_mission_continue,
            MissionState.MISSION_EXECUTE: self._handle_mission_execute,
            MissionState.MISSION_TO_OFFBOARD_CASUALTY: lambda: self._handle_mission_to_offboard(MissionState.NAV_TO_CASUALTY_35),  # >>> CHANGED: 다음상태 변경
            MissionState.CASUALTY_TRACK: lambda: self._handle_track_target(MissionState.CASUALTY_DESCEND),
            MissionState.CASUALTY_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_CLOSE, self.gripper_altitude),
            MissionState.GRIPPER_CLOSE: self._handle_gripper_close,
            MissionState.CASUALTY_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            MissionState.MISSION_CONTINUE: self._handle_mission_continue,
            MissionState.MISSION_TO_OFFBOARD_DROP_TAG: lambda: self._handle_mission_to_offboard(MissionState.NAV_TO_DROPTAG_35),  # >>> CHANGED
            MissionState.DROP_TAG_TRACK: lambda: self._handle_track_target(MissionState.DROP_TAG_DESCEND),
            MissionState.DROP_TAG_DESCEND: lambda: self._handle_descend_ascend(MissionState.GRIPPER_OPEN, self.gripper_altitude),
            MissionState.GRIPPER_OPEN: self._handle_gripper_open,
            MissionState.DROP_TAG_ASCEND: lambda: self._handle_descend_ascend(MissionState.OFFBOARD_TO_MISSION, self.mission_altitude),
            MissionState.MISSION_TO_OFFBOARD_LANDING_TAG: lambda: self._handle_mission_to_offboard(MissionState.LANDING_TAG_TRACK),
            MissionState.LANDING_TAG_TRACK: lambda: self._handle_track_target(MissionState.LAND),
            MissionState.LAND: self._handle_land,
            MissionState.MISSION_COMPLETE: self._handle_mission_complete,
            MissionState.ERROR: self._handle_error,

            # >>> ADDED: 신규 상세 상태
            MissionState.NAV_TO_CASUALTY_35: self._handle_nav_to_casualty_35,
            MissionState.ALIGN_20: lambda: self._handle_align_alt(self.basket_xy, 20.0, MissionState.ALIGN_15),
            MissionState.ALIGN_15: lambda: self._handle_align_alt(self.basket_xy, 15.0, MissionState.ALIGN_10),
            MissionState.ALIGN_10: lambda: self._handle_align_alt(self.basket_xy, 10.0, MissionState.GREEN_DESCENT),
            MissionState.GREEN_DESCENT: self._handle_green_descent,
            MissionState.SWITCH_TO_RED: self._handle_switch_to_red,
            MissionState.RED_STABILITY_CHECK: self._handle_red_stability_check,
            MissionState.RETRY_ASCEND_10: lambda: self._handle_align_alt(self.basket_xy, self.retry_altitude, MissionState.GREEN_DESCENT),

            MissionState.NAV_TO_DROPTAG_35: self._handle_nav_to_droptag_35,
            MissionState.DROP_ALIGN_20: lambda: self._handle_align_alt(self.drop_tag_xy, 20.0, MissionState.DROP_ALIGN_15),
            MissionState.DROP_ALIGN_15: lambda: self._handle_align_alt(self.drop_tag_xy, 15.0, MissionState.DROP_ALIGN_10),
            MissionState.DROP_ALIGN_10: lambda: self._handle_align_alt(self.drop_tag_xy, 10.0, MissionState.DROPTAG_DESCENT),
            MissionState.DROPTAG_DESCENT: self._handle_droptag_descent,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
        
        self._publish_vehicle_state()

    # (파라미터 콜백/로깅/기존 핸들러들은 그대로 — 중략 없이 유지)
    # -------------------------------------------------------------------------
    def param_update_callback(self, params):
        successful = True
        reason = ''
        for p in params:
            # (기존 파라미터 갱신) — 그대로
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
            elif p.name == 'casualty_waypoint' :
                self.casualty_waypoint = p.value
            elif p.name == 'drop_tag_waypoint':
                self.drop_tag_waypoint = p.value
            elif p.name == 'landing_tag_waypoint':
                self.landing_tag_waypoint = p.value

            # >>> ADDED: 신규 파라미터 갱신
            elif p.name == 'basket_xy':
                self.basket_xy = p.value
            elif p.name == 'drop_tag_xy':
                self.drop_tag_xy = p.value
            elif p.name == 'arrive_altitude':
                self.arrive_altitude = p.value
            elif p.name == 'arrive_radius':
                self.arrive_radius = p.value
            elif p.name == 'green_stop_ratio':
                self.green_stop_ratio = p.value
            elif p.name == 'red_stable_secs':
                self.red_stable_secs = p.value
            elif p.name == 'retry_altitude':
                self.retry_altitude = p.value
            elif p.name == 'descent_speed':
                self.descent_speed = p.value
            elif p.name == 'image_node_name':
                self.image_node_name = p.value
            elif p.name == 'detection_mode_param_key':
                self.detection_mode_param_key = p.value

            # (드론 타깃 컨트롤러 파라미터) — 그대로
            elif p.name.startswith('drone_target_controller.'):
                param_key = p.name.replace('drone_target_controller.', '')
                self.drone_controller_params[param_key] = p.value
                if hasattr(self.drone_target_controller, param_key):
                    setattr(self.drone_target_controller, param_key, p.value)
            else:
                self.get_logger().warn(f"Ignoring unknown parameter: {p.name}")
                continue
        
        self.get_logger().info("[Parameter Update] Mission parameters updated successfully")
        self.drone_target_controller.reset()
        self.print_current_parameters()
        return SetParametersResult(successful=successful, reason=reason)

    def print_current_parameters(self):
        self.get_logger().info("Current Mission Parameters:")
        self.get_logger().info(f"  mission_altitude: {self.mission_altitude}")
        self.get_logger().info(f"  track_min_altitude: {self.track_min_altitude}")
        self.get_logger().info(f"  gripper_altitude: {self.gripper_altitude}")
        self.get_logger().info(f"  tracking_target_offset: {self.tracking_target_offset}")
        self.get_logger().info(f"  tracking_acceptance_radius_xy: {self.tracking_acceptance_radius_xy}")
        self.get_logger().info(f"  tracking_acceptance_radius_z: {self.tracking_acceptance_radius_z}")
        self.get_logger().info(f"  casualty_waypoint: {self.casualty_waypoint}")
        self.get_logger().info(f"  drop_tag_waypoint: {self.drop_tag_waypoint}")
        self.get_logger().info(f"  landing_tag_waypoint: {self.landing_tag_waypoint}")
        self.get_logger().info(f"  basket_xy: {self.basket_xy}")
        self.get_logger().info(f"  drop_tag_xy: {self.drop_tag_xy}")
        self.get_logger().info(f"  arrive_altitude: {self.arrive_altitude}")
        self.get_logger().info(f"  arrive_radius: {self.arrive_radius}")
        self.get_logger().info(f"  green_stop_ratio: {self.green_stop_ratio}")
        self.get_logger().info(f"  red_stable_secs: {self.red_stable_secs}")
        self.get_logger().info(f"  retry_altitude: {self.retry_altitude}")
        self.get_logger().info(f"  descent_speed: {self.descent_speed}")
        self.get_logger().info(f"  image_node_name: {self.image_node_name}")
        self.get_logger().info(f"  detection_mode_param_key: {self.detection_mode_param_key}")
        self.get_logger().info("DroneTargetController Parameters:")
        for key, value in self.drone_controller_params.items():
            self.get_logger().info(f"  {key}: {value}")

    def _publish_vehicle_state(self):
        self.vehicle_state_publisher.publish(
            VehicleState(
                vehicle_state=self.state.value,
                detect_target_type=self.TARGET_TYPES.get(self.state, 0)
            )
        )

    def on_target_update(self, msg: TargetLocation):
        # GREEN 모드: confidence == green_ratio
        self.target = msg if msg is not None else None

    def on_vehicle_accel_update(self, msg: VehicleAcceleration):
        self.vehicle_acc = msg

    # ---------------- (기존 핸들러들: 그대로) ----------------
    def _handle_init(self):
        if not self.get_position_flag:
            self.get_logger().info("Waiting for global position data...")
            return
        self.set_home_position()
        if self.home_set_flag:
            self.get_logger().info("Home position set, ready for offboard mode")
            self.state = MissionState.OFFBOARD_ARM

    def _handle_offboard_arm(self):
        if not self.is_offboard_mode():
            return
        if self.is_disarmed():
            self.get_logger().info("Arming vehicle in offboard mode")
            self.arm()
        else:
            self.get_logger().info("Armed in offboard mode, starting logger")
            self._start_logging()
            self.state = MissionState.OFFBOARD_TO_MISSION

    def _start_logging(self):
        if self.do_logging:
            self.logger.start_logging()
            self.get_logger().info(f"[Logger] Writing to: {os.path.abspath(self.logger.log_path)}")
            self.log_timer = self.create_timer(0.1, self._log_timer_callback)

    def _handle_mission_continue(self):
        if self.is_offboard_mode():
            self.set_mission_mode()
        elif self.is_mission_mode():
            self.get_logger().info(f"Resuming mission from waypoint {self.mission_paused_waypoint}")
            self.target = None
            self.state = MissionState.MISSION_EXECUTE

    def _handle_mission_execute(self):
        if not self.is_mission_mode():
            self.get_logger().warn("Not in mission mode, cannot execute mission")
            return

        current_wp = self.mission_wp_num

        # >>> CHANGED: 픽업/드롭 구간에서 상세 절차 상태로 진입
        if current_wp == self.casualty_waypoint and not self.pickup_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_CASUALTY
            return

        if current_wp == self.drop_tag_waypoint and not self.dropoff_complete:
            self.mission_paused_waypoint = current_wp
            self.state = MissionState.MISSION_TO_OFFBOARD_DROP_TAG
            return

        if current_wp == self.landing_tag_waypoint:
            self.state = MissionState.MISSION_TO_OFFBOARD_LANDING_TAG
            return

    def _handle_mission_to_offboard(self, next_state: MissionState):
        if self.is_mission_mode():
            self.set_offboard_mode()
        elif self.is_offboard_mode():
            self.state = next_state

    def _handle_track_target(self, next_state: MissionState):
        # >>> REUSED: 기존 각도기반 추종(필요 시)
        if self.target is None or self.target.status != 0:
            self.get_logger().warn("No target coordinates available, waiting for CV detection")
            return
        target_pos, arrived = self.drone_target_controller.update(
            self.pos, self.yaw, self.target.angle_x, self.target.angle_y
        )
        self.publish_setpoint(pos_sp=target_pos)
        if arrived:
            self.drone_target_controller.reset()
            self.state = next_state

    def _handle_descend_ascend(self, next_state: MissionState, target_altitude: float):
        # >>> REUSED: 기존 수직 이동 핸들러
        if self.target_position is None:
            self.target_position = np.array([self.pos[0], self.pos[1], -target_altitude])
        self.publish_setpoint(pos_sp=self.target_position)
        if abs(self.pos[2] - self.target_position[2]) < self.tracking_acceptance_radius_z:
            self.publish_setpoint(pos_sp=self.pos)
            self.target_position = None
            self.state = next_state

    def _handle_gripper_close(self):
        self.get_logger().info("Closing gripper to pick up casualty")
        # TODO: 실제 그리퍼 제어 넣기
        self.pickup_complete = True
        self.state = MissionState.CASUALTY_ASCEND

    def _handle_gripper_open(self):
        self.get_logger().info("Opening gripper to release casualty at dropoff point")
        # TODO: 실제 그리퍼 제어 넣기
        self.dropoff_complete = True
        self.state = MissionState.DROP_TAG_ASCEND

    def _handle_land(self):
        self.land()
        self.get_logger().info("Landing command sent")
        self.get_logger().info("Mission Complete!")
        self.log_timer.cancel() if self.log_timer else None
        self.state = MissionState.MISSION_COMPLETE

    def _handle_mission_complete(self):
        pass
        
    def _handle_error(self):
        self.get_logger().error("Mission in error state")
        # TODO: 오류 복구/비상절차

    # ---------------- (신규 상세 상태 핸들러) ----------------
    def _handle_nav_to_casualty_35(self):
        bx, by = self.basket_xy
        self._enu_goto(bx, by, self.arrive_altitude)
        if self._arrived_xy(self.basket_xy, self.arrive_radius) and self._alt_reached(self.arrive_altitude, 0.8):
            self.get_logger().info("Arrived over basket (35m, r<=5m).")
            self._enter(MissionState.ALIGN_20)

    def _handle_align_alt(self, target_xy, alt_up: float, next_state: MissionState):
        self._enu_goto(target_xy[0], target_xy[1], alt_up)
        if self._alt_reached(alt_up, 0.5):
            self._enter(next_state)

    def _handle_green_descent(self):
        if self._detector_mode != "green":
            self._set_detector_mode("green")

        if self.target is None or self.target.status != 0:
            self._hold_alt(max(10.0, self.arrive_altitude))
            return

        gx, gy = self.target.x, self.target.y
        ratio = float(getattr(self.target, 'confidence', 0.0))

        cur_alt = -self.pos[2]
        dt = max(self.timer_period, 0.05)
        target_alt = max(0.5, cur_alt - self.descent_speed * dt)
        self._enu_goto(gx, gy, target_alt)

        if ratio >= float(self.green_stop_ratio):
            self.get_logger().info(f"Green ratio {ratio:.2f} >= {self.green_stop_ratio:.2f}. Switch to RED.")
            self._enter(MissionState.SWITCH_TO_RED)

    def _handle_switch_to_red(self):
        if self._detector_mode != "red":
            self._set_detector_mode("red")
            self._enter(MissionState.RED_STABILITY_CHECK)
        else:
            if self._now() - self._state_entered_t > 1.0:
                self._enter(MissionState.RED_STABILITY_CHECK)

    def _handle_red_stability_check(self):
        t0 = self._now()
        last = None
        ok = True
        while self._now() - t0 < float(self.red_stable_secs) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            det = self.target
            if det is None or det.status != 0 or float(getattr(det, 'confidence', 0.0)) <= 0.0:
                ok = False
                break
            cur = (det.x, det.y)
            if last is not None:
                dx = cur[0] - last[0]; dy = cur[1] - last[1]
                if (dx*dx + dy*dy) ** 0.5 > 0.5:
                    ok = False
                    break
            last = cur

        if ok:
            self.get_logger().info("RED detection stable.")
            self._enter(MissionState.CASUALTY_DESCEND)  # 최종 하강(그리퍼)
        else:
            self.get_logger().warn("RED unstable. Retry: ascend to 10m and restart GREEN.")
            self._enter(MissionState.RETRY_ASCEND_10)

    def _handle_nav_to_droptag_35(self):
        dx, dy = self.drop_tag_xy
        self._enu_goto(dx, dy, self.arrive_altitude)
        if self._arrived_xy(self.drop_tag_xy, self.arrive_radius) and self._alt_reached(self.arrive_altitude, 0.8):
            self.get_logger().info("Arrived over drop tag (35m, r<=5m).")
            self._enter(MissionState.DROP_ALIGN_20)

    def _handle_droptag_descent(self):
        if self._detector_mode != "drop_tag":
            self._set_detector_mode("drop_tag")

        if self.target is None or self.target.status != 0:
            self._enter(MissionState.DROP_ALIGN_10)  # 재정렬 후 재시도
            return

        tx, ty = self.target.x, self.target.y
        cur_alt = -self.pos[2]
        dt = max(self.timer_period, 0.05)
        target_alt = max(0.4, cur_alt - self.descent_speed * dt)
        self._enu_goto(tx, ty, target_alt)

        if target_alt <= 0.45:
            self._enter(MissionState.GRIPPER_OPEN)

    # ---------------- 로깅 ----------------
    def _log_timer_callback(self):
        if self.logger is None:
            self.get_logger().warn("Logger called while not initialized")
            return

        auto_flag = 0 if self.state is MissionState.INIT else 1
        event_flag = self.mission_wp_num
        gps_time = self.vehicle_gps.time_utc_usec / 1e6
        
        self.logger.log_data(
            auto_flag, event_flag, gps_time,
            self.vehicle_gps.latitude_deg,
            self.vehicle_gps.longitude_deg,
            self.vehicle_gps.altitude_ellipsoid_m,
            self.vehicle_acc.xyz[0],
            self.vehicle_acc.xyz[1],
            self.vehicle_acc.xyz[2]
        )

    # Override methods (placeholders for additional functionality)
    def on_vehicle_status_update(self, msg): pass
    def on_local_position_update(self, msg): pass
    def on_attitude_update(self, msg): pass
    def on_global_position_update(self, msg): pass


def main(args=None):
    rclpy.init(args=args)
    controller = None

    try:
        controller = MissionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Mission interrupted by user")
    except Exception as e:
        print(f"Mission failed with error: {e}")
    finally:
        if controller:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
