from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch argument for gazebo/real mode
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to use Gazebo simulation (true) or real hardware (false)'
    )
        # Declare launch argument for streaming
    do_streaming_arg = DeclareLaunchArgument(
        'do_streaming',
        default_value='false',
        description='Whether to enable streaming functionality'
    )
    
    # Get the launch configuration
    use_gazebo = LaunchConfiguration('use_gazebo')
    do_streaming = LaunchConfiguration('do_streaming')
    
    # Image processing node with conditional executable name
    image_processing_node = Node(
        package='vision_processing_nodes',
        executable=PythonExpression([
            "'image_processing_node_gazebo' if '", use_gazebo, "' == 'true' else 'image_processing_node'"
        ]),
        parameters=[PathJoinSubstitution([
            FindPackageShare('vision_processing_nodes'), 'config', 'params.yaml'
        ]),
                    {'do_streaming': do_streaming}
                    ],
        output='screen'
    )
    
    # ROS-Gazebo bridge node (only runs when use_gazebo is true)
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name="gz_image_bridge",
        #arguments=['/world/RAC_2025/model/standard_vtol_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        arguments=['/world/RAC_2025/model/standard_vtol_gimbal_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        output='screen',
        condition=IfCondition(use_gazebo)
    )
    
    return LaunchDescription([
        use_gazebo_arg,
        image_processing_node,
        gz_bridge_node,
    ])
