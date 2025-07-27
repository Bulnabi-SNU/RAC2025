# RAC2025
Robot Aircraft Competition 2025

![PX4 SITL](assets/3.png)

## Environment
Ubuntu 22.04
ROS2 Humble
PX4_Autopilot: main
px4_msgs: main

# TODO
Refactor image recognition (WIP)
Create/separate target tracking code into separate file. (DONE)
- Create tracking code that uses beizier (if needed)
Add more test cases (esp. for tracking)
Update README

## Changed Params:

Disabled QuadChute
Disabled Weathervaning
Increased NAV_ACC_RAD to 20 m (set radii on a per-waypoint basis (param2=0.5) for accurate ones)
Decreased NAV_FW_ALT_RAD to 3 m
Decreased VT_B_TRANS_DUR to 8 s
Set MPC_YAW_MODE to along trajectory
