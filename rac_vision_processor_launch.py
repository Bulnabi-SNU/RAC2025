from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='False',
        description='Whether to use Gazebo simulation (true) or real hardware (false)'
    )
    
    use_jetson_arg = DeclareLaunchArgument(
        'use_jetson',
        default_value='False',
        description='Whether running on Jetson platform (true) or other hardware (false)'
    )
    
    show_debug_stream_arg = DeclareLaunchArgument(
        'show_debug_stream',
        default_value='alse',
        description='Whether to show debug stream via imshow (true/false)'
    )
    
    # Get launch configurations
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_jetson = LaunchConfiguration('use_jetson')
    show_debug_stream = LaunchConfiguration('show_debug_stream')
    
    # Main vision processing node - handles everything in one process for efficiency
    vision_node = Node(
        package='vision_processing_nodes',
        executable='vision_processor_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('vision_processing_nodes'), 'config', 'params.yaml'
            ]),
            {'use_gazebo': use_gazebo},
            {'use_jetson': use_jetson},
            {'show_debug_stream': show_debug_stream}
        ],
        output='screen',
        emulate_tty=True
    )
    
    # ROS-Gazebo bridge node (only runs when use_gazebo is true)
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name="gz_image_bridge",
        arguments=['/world/RAC_2025/model/standard_vtol_gimbal_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        output='screen',
        condition=IfCondition(use_gazebo)
    )
    
    return LaunchDescription([
        use_gazebo_arg,
        use_jetson_arg,
        show_debug_stream_arg,
        vision_node,
        gz_bridge_node
    ])