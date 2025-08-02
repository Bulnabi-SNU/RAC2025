from launch import LaunchDescription

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare 


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_processing_nodes',
            executable='image_processing_node_gazebo',
            parameters = [PathJoinSubstitution([
        FindPackageShare('vision_processing_nodes'), 'config', 'params.yaml'])],
            output='screen'
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name="gz_image_bridge",
            arguments=['/world/RAC_2025/model/standard_vtol_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'],
            output='screen'
        ),
    ])
