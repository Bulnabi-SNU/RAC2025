import time
import csv
import numpy as np
import os
from datetime import datetime, timedelta
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

    def log_data(self, auto_flag, lat, long, alt, gps_time, ax, ay, az, est_ax, est_ay, est_az, roll, pitch, yaw, cam, wp):
        """Log data to the CSV file."""
        # Format gps_time in scientific notation (no rounding specified)
        # gps_time = np.format_float_scientific(gps_time, trim='k')
        utc = datetime(1970, 1, 1) + timedelta(microseconds=gps_time)

        # Format latitude, longitude, and altitude to specified precision
        lat = np.format_float_positional(lat, precision=6, trim='k', unique=False)
        long = np.format_float_positional(long, precision=6, trim='k', unique=False)
        alt = np.format_float_positional(alt,precision=1,trim='k', unique=False)
        ax = np.format_float_positional(ax, precision=3, trim='k')
        ay = np.format_float_positional(ay, precision=3, trim='k')
        az = np.format_float_positional(az, precision=3, trim='k')

        est_ax = np.format_float_positional(est_ax, precision=3, trim='k')
        est_ay = np.format_float_positional(est_ay, precision=3, trim='k')
        est_az = np.format_float_positional(est_az, precision=3, trim='k')
        roll = np.format_float_positional(roll, precision=4, trim='k') 
        pitch = np.format_float_positional(pitch, precision=4, trim='k') 
        yaw = np.format_float_positional(yaw, precision=4, trim='k') 
        
        if self.log_writer:
            self.log_writer.writerow([auto_flag, lat, long, alt, utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second, utc.microsecond // 1000, ax, ay, az, est_ax, est_ay, est_az, roll, pitch, yaw, cam, wp])
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
        self.log_writer.writerow(["Mode", "Latitude", "Longitude", "Altitude", "Year", "Month", "Day", "Hour", "Minute", "Second", "Millisecond", "Ax", "Ay", "Az", "EstAx", "EstAy", "EstAz", "Roll", "Pitch", "Yaw", "ImageDetection", "Waypoint"])

if __name__ == "__main__":
    logger = Logger()
    logger.start_logging()
    # Example data logging
    logger.log_data(1, 0, 37.604256123, 37.604256123, 76.54321, 100.0123)
    logger.log_data(0, 1, 123456.8, 12.34568, 100.154123657, 100.154123457)
    del logger