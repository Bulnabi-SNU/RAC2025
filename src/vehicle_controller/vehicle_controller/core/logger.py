import time
import csv
import numpy as np
import os

# Log format: ascii csv
# Columns: manual(0) or auto(1) mode, event flag (= headed waypoint #), gps time (s), LLA coordinates
# Creates log with filename that contains the current date and time

class Logger:
    """Logger class to handle logging of vehicle data for judging"""
    
    def __init__(self, log_path:str = "flight_logs"):
        self.log_path = log_path
        self.log_file = None
        self.log_writer = None

    def __del__(self):
        if self.log_file:
            self.log_file.close()

    def log_data(self, auto_flag, wp_num_internal, lat, long, alt, gps_time, est_ax, est_ay, est_az, roll, pitch, yaw):
        """Log data to the CSV file."""

        # Format gps_time in scientific notation (no rounding specified)
        gps_time = np.format_float_scientific(gps_time, trim='k')

        waypoint_range_tuples = [
            ((1, 3), 1),    # Internal WPs from 1 to 4 (inclusive) -> Group 1
            ((4, 4), 2),    # Internal WPs from 5 to 6 (inclusive) -> Group 2
            ((5, 5), 3),   # Internal WPs from 7 to 11 (inclusive) -> Group 3
            ((6,6),4)
        ]

        # Find the final waypoint number by checking each range
        final_wp_num = 0  # Default value
        for (start, end), group_num in waypoint_range_tuples:
            if start <= wp_num_internal <= end:
                final_wp_num = group_num
                break

        # Format latitude, longitude, and altitude to specified precision
        lat = np.format_float_positional(lat, precision=6, trim='k', unique=False)
        long = np.format_float_positional(long, precision=6, trim='k', unique=False)
        alt = np.format_float_positional(alt,precision=1,trim='k', unique=False)

        est_ax = np.format_float_positional(est_ax, precision=3, trim='k')
        est_ay = np.format_float_positional(est_ay, precision=3, trim='k')
        est_az = np.format_float_positional(est_az, precision=3, trim='k')

        roll = np.format_float_positional(roll, precision=4, trim='k') 
        pitch = np.format_float_positional(pitch, precision=4, trim='k') 
        yaw = np.format_float_positional(yaw, precision=4, trim='k') 
        
        if self.log_writer:
            self.log_writer.writerow([auto_flag, final_wp_num, lat, long, alt, gps_time, est_ax, est_ay, est_az, roll, pitch, yaw])
            self.log_file.flush()

    def start_logging(self):   
        """Start logging data to a new CSV file."""
        if self.log_file:
            self.log_file.close()
        self.log_path = "/workspace/flight_logs"
        os.makedirs(self.log_path, exist_ok=True)
        # Create a new log file with the current date and time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = open(os.path.join(self.log_path, f"vehicle_log_{timestamp}.csv"), "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        
        # Write header
        self.log_writer.writerow(["Auto Flag", "WP", "L", "L", "A", "Time", "Ax", "Ay", "Az", "R", "P", "Y"])

if __name__ == "__main__":
    logger = Logger()
    logger.start_logging()
    # Example data logging
    logger.log_data(1, 0, 37.604256123, 37.604256123, 76.54321, 100.0123)
    logger.log_data(0, 1, 123456.8, 12.34568, 100.154123657, 100.154123457)
    del logger