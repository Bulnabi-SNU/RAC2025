# msg for publish phase
# vehicle_controller -> nodes using camera
string vehicle_state
string phase

# 0: Don't detect 
# 1: Detect casualty
# 2: Detect casualty dropoff
# 3: Detect landing
int8 detect_target_type