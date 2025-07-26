# RAC2025
Robot Aircraft Competition 2025

![PX4 SITL](assets/3.png)

## Environment
Ubuntu 22.04
ROS2 Humble
PX4_Autopilot: v1.15.4
px4_msgs: main

gazebo는 나는 fortress 버전 쓰긴 했는데 classic도 될거임 아마.

## Installation
'''
git clone https://github.com/Bulnabi-SNU/RAC2025.git
cd RAC2025
colcon build --symlink-install
'''


## How to Use

Terminal 1 : PX4 SITL
```
cd PX4-Autopilot
make px4_sitl gz_standard_vtol
```


Terminal 2 : Micro-XRCE-DDS
```
MicroXRCEAgent udp4 -p 8888
```


Terminal 3 : Q Ground Control
```
./QGroundControl.AppImage
```


Terminal 4 : run ROS2 node
```
cd RAC2025
roshumble # alias
ros2 run vehicle_controller mc_test_01

# run SITL with yaml file
ros2 run <package> <node> --ros-args --params-file <.yaml file path>

ros2 run test_nodes mc_test_01 --ros-args --params-file ~/RAC2025/src/vehicle_controller/waypoints/skeleton_code.yaml
```


## Changed Params:

Disabled QuadChute
Disabled Weathervaning
Increased NAV_ACC_RAD to 20 m (set radii on a per-waypoint basis (param2=0.5) for accurate ones)
Decreased NAV_FW_ALT_RAD to 3 m
Decreased VT_B_TRANS_DUR to 8 s
Set MPC_YAW_MODE to along trajectory
