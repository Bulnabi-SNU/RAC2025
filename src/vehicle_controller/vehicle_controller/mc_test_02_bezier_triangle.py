__author__ = "Chaewon"
__contact__ = ""

# import rclpy: ros library
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from vehicle_controller.vehicle_controller.code_basic.px4_base import PX4BaseController
from vehicle_controller.vehicle_controller.code_basic.bezier_handler import BezierCurve

# import px4_msgs
from px4_msgs.msg import VehicleStatus, VehicleCommand, VehicleGlobalPosition

# import math library
import numpy as np
import math
# gps
import pymap3d as p3d

class MissionController(PX4BaseController):
    
    def __init__(self):
        super().__init__('mc_test_02_bezier_triangle')
    
        self.num_wp = 4
        self.WP = []           # waypoints, local coordinates.
        self.gps_WP = [      # TODO: Convert GPS to local(NED) coords
            [37.455914, 126.954218, 0.0],       # 1 (home)
            [37.456108, 126.954093, 0.0],       # 2
            [37.456110, 126.954329, 0.0],       # 3
            [37.455914, 126.954218, 0.0] ]      # 1 (home)

        self.bezier_flag = False
        self.start_position = None
        self.end_position = None
        self.current_point_index = 0

        self.bezier_handler = BezierCurve(time_step=0.05)

        self.vmax = 3.0  # m/s
        self.mc_arrival_radius = 0.5

        self.offboard_control_mode_params['position'] = True
        self.offboard_control_mode_params['velocity'] = False

        # State machine
        self.state = 'READY_TO_FLIGHT'
        self.phase = 0
        self.subphase = ''
        
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
    
    #=======================================
    # Flight State Functions
    #=======================================
    
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
            if self.home_set_flag:
                self.WP = self.set_gps_to_local(self.num_wp, self.gps_WP, self.home_position_gps)
                self.state = 'TAKEOFF'
    
    def _handle_takeoff(self):
        """Handle takeoff state"""
        if self.is_auto_loiter():
            self.get_logger().info("Takeoff complete. Starting bezir navigation phase...")
            self.set_offboard_mode()
            self.current_point_index = 0
            self.state = 'BEZIER_NAV'
            self.phase = 1
    
    def _handle_bezier_nav(self):
        """Handle bezier navigation state"""
        if self.current_point_index < self.num_wp:
            if self.bezier_flag is False:
                # generate bezier curve just once per WP
                self.start_position = np.copy(self.pos)
                self.end_position = self.WP[self.current_point_index]
                self.bezier_handler.generate_curve(
                        start_pos=self.start_position,
                        end_pos=self.end_position,
                        start_vel=self.vel,
                        end_vel=np.array([0.0, 0.0, 0.0]),
                        max_velocity=self.vmax,
                        total_time=None
                    )
                
                self.publish_setpoint(position_sp = self.bezier_handler.get_current_point()) 
                distance = np.linalg.norm(self.end_position - self.pos)

                if distance < self.mc_arrival_radius:
                    self.get_logger().info("4")
                    self.current_point_index += 1
                    self.bezier_flag = False
                    if self.current_point_index >= self.num_wp:
                        self.get_logger().info("All waypoints reached. Landing...")
                        self.state = 'LANDING'
                        return
                    self.get_logger().info("Waypoints reached. Moving to next waypoint")
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
    
    #=======================================
    # Additional Fuctions
    #=======================================

    def set_home_local_position(self, home_gps_pos):
        # 'start point' of mission in local coordinates
        self.home_position = self.pos
        self.start_yaw = self.yaw

        # convert GPS waypoints to local coordinates relative to home position
        for i in range(len(self.num_wp)):
            # gps_WP = [lat, lon, rel_alt]
            wp_position = p3d.geodetic2ned(self.gps_WP[i][0], self.gps_WP[i][1], self.gps_WP[i][2] + home_gps_pos[2],
                                            home_gps_pos[0], home_gps_pos[1], home_gps_pos[2])
            wp_position = np.array(wp_position)
            self.WP.append(wp_position)

    
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
