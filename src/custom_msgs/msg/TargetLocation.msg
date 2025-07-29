# for give target location
# relative coordinates to vehicle position
# camera nodes -> vehicle_controller

# 0 if OK, -1 if completely failed, other values for debugging only
int8 status

# angle from the camera planes
float64 angle_x
float64 angle_y

# If needed, send additional info (prob not)
float64[] additional