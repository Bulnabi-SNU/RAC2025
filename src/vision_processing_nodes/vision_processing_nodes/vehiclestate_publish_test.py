# import rclpy: ros library
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# import custom msg
from custom_msgs.msg import VehicleState  # updated message

class VehicleStatePublisher(Node):
    def __init__(self):
        super().__init__('vehicle_state_test_publisher')

        # Set QoS to match the image_processing_node
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher = self.create_publisher(VehicleState, '/vehicle_state', qos_profile)

        # Timer: publish once every 1 second
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Message to publish
        self.test_state = VehicleState()
        self.test_state.vehicle_state = "Apriltag"
        self.test_state.detect_target_type = 3  # 0: none, 1: casualty, 2: dropoff, 3: apriltag

        self.get_logger().info("VehicleState test publisher initialized.")

    def timer_callback(self):
        self.publisher.publish(self.test_state)
        self.get_logger().info("Published VehicleState message with:")
        self.get_logger().info(f"  vehicle_state: {self.test_state.vehicle_state}")
        self.get_logger().info(f"  detect_target_type: {self.test_state.detect_target_type}")


def main(args=None):
    rclpy.init(args=args)
    node = VehicleStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
