from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare 


def get_mission_controller_node(context):
    """Generate mission controller node based on launch arguments"""
    mc_type = LaunchConfiguration("mc_type").perform(context)
    use_config = LaunchConfiguration("use_config").perform(context).lower() == 'true'
    config_file = LaunchConfiguration("config_file").perform(context)
    
    # Base parameters path
    if use_config:
        if config_file and config_file != '':
            # Use custom config file if specified
            params_file = config_file
        else:
            # Use default params.yaml
            params_file = PathJoinSubstitution([
                FindPackageShare('vehicle_controller'), 'config', 'params.yaml'
            ])
    else:
        params_file = None
    
    # Create the node configuration
    node_config = {
        "package": "vehicle_controller",
        "executable": f"mc_{mc_type}",
        "name": f"mission_controller_{mc_type}",
        "output": "screen",
        "emulate_tty": True,  # Better output formatting
    }
    
    # Add parameters if config is enabled
    if params_file:
        node_config["parameters"] = [params_file]
    
    return [
        LogInfo(msg=f"Starting mission controller: mc_{mc_type}"),
        Node(**node_config)
    ]


def generate_launch_description():
    return LaunchDescription([
        
        # Mission Controller Type
        DeclareLaunchArgument(
            "mc_type",
            default_value="main",
            choices=['main', 'test_01', 'test_02', 'test_03', 'test_04', 'test_05'],
            description="Type of mission controller to run"
        ),
        
        # Configuration
        DeclareLaunchArgument(
            "use_config",
            default_value="true",
            choices=['true', 'false'],
            description="Whether to use configuration file for parameters"
        ),
        
        DeclareLaunchArgument(
            "config_file",
            default_value="",
            description="Custom configuration file path (leave empty for default params.yaml)"
        ),
        
        # Launch Information
        LogInfo(msg="=== RAC 2025 Vehicle Controller Launch ==="),
        LogInfo(msg="Mission Controller Type: "),
        LogInfo(msg=LaunchConfiguration("mc_type")),
        
        # Generate mission controller node
        OpaqueFunction(function=get_mission_controller_node),
        
    ])


# Usage examples (as comments):
"""
# Basic usage - start main mission controller:
ros2 launch vehicle_controller rac_mission_controller_launch.py

# Start specific test controller:
ros2 launch vehicle_controller rac_mission_controller_launch.py mc_type:=test_01

# Start without configuration file:
ros2 launch vehicle_controller rac_mission_controller_launch.py use_config:=false

# Use custom configuration:
ros2 launch vehicle_controller rac_mission_controller_launch.py config_file:=/path/to/custom_params.yaml
"""