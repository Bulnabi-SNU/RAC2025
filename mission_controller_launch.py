from launch import LaunchDescription

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue 
from launch_ros.substitutions import FindPackageShare 

import os
from ament_index_python.packages import get_package_share_directory

def get_node(context):
    mc_type = LaunchConfiguration("mc_type").perform(context)

    params = PathJoinSubstitution([
        FindPackageShare('vehicle_controller'), 'config', 'params.yaml'])

    return [Node(package="vehicle_controller",
        executable=f"mc_{mc_type}",
        parameters=[params],
        output="screen")]

def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
              "mc_type",
              default_value="main",
              choices=['main','test_01','test_02','test_03','test_04'],
              description="Type of controller to run"
            ),

        OpaqueFunction(function=get_node)
    ])
