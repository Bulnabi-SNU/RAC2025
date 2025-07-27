# vision_processing_nodes/detection/utils.py

"""
Shared utility functions for vision processing modules
Contains common helper functions to avoid code duplication
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import math

def pixel_to_fov(x, y, image_width, image_height, h_fov_deg=81, d_fov_deg=93):
    # Calculate vertical FOV
    h_fov_rad = math.radians(h_fov_deg)
    d_fov_rad = math.radians(d_fov_deg)
    
    # For 16:9 aspect ratio, calculate vertical FOV
    aspect_ratio = 16/9
    v_fov_rad = 2 * math.atan(math.tan(h_fov_rad/2) / aspect_ratio)
    v_fov_deg = math.degrees(v_fov_rad)
    
    # Normalize pixel coordinates to [-1, 1]
    norm_x = (2 * x / image_width) - 1
    norm_y = (2 * y / image_height) - 1
    
    # Convert to angular coordinates (invert y since increase in y = going down)
    angle_x = norm_x * (h_fov_deg / 2)
    angle_y =- norm_y * (v_fov_deg / 2)
    
    return angle_x, angle_y