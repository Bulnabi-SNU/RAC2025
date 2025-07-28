# RAC2025
Robot Aircraft Competition 2025

![PX4 SITL](assets/3.png)

## Environment
Ubuntu 22.04 \
ROS2 Humble \
PX4_Autopilot: main \
px4_msgs: main

# TODO
## Controller Related
- Improve tracking logic
    - Horizontally Align -> Descend instead of always descending
    - Maybe try bezier? 
    - Improve stability (Need to tune w/ actual hardware)
- Add more test cases (esp. for tracking)

## CV related
- Refactor image recognition (WIP)
    - Standardize recognition messages
- Handle streaming on a separate node
- Create launch file for image recognition + bridge / streaming nodes

## Misc.
- Update README
- Convert mission waypoint to event flags i.a.w. mission rules for logs

## Changed Params:

- Disabled QuadChute
- Disabled Weathervaning
- Increased NAV_ACC_RAD to 20 m (set radii on a per-waypoint basis (param2=0.5) for accurate ones)
- Decreased NAV_FW_ALT_RAD to 3 m
- Decreased VT_B_TRANS_DUR to 8 s
- Set MPC_YAW_MODE to along trajectory
