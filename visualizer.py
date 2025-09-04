import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_log_data(filename='log_data.csv'):
    """
    Reads log data from a CSV file and plots various parameters.

    Args:
        filename (str): The name of the CSV file to read.
    """
    # --- Data Reading and Parsing ---
    # Initialize lists to store data from the CSV
    auto_flag = []
    final_wp_num = []
    lat = []
    long = []
    alt = []
    gps_time = []
    est_ax = []
    est_ay = []
    est_az = []
    roll = []
    pitch = []
    yaw = []

    try:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            # Skip header row if it exists
            # next(csvreader) 
            for row in csvreader:
                try:
                    # Append data, converting to float for plotting
                    auto_flag.append(int(row[0]))
                    final_wp_num.append(int(row[1]))
                    lat.append(float(row[2]))
                    long.append(float(row[3]))
                    alt.append(float(row[4]))
                    gps_time.append(float(row[5]))
                    est_ax.append(float(row[6]))
                    est_ay.append(float(row[7]))
                    est_az.append(float(row[8]))
                    roll.append(float(row[9]))
                    pitch.append(float(row[10]))
                    yaw.append(float(row[11]))
                except (ValueError, IndexError) as e:
                    print(f"Skipping row due to data conversion error: {row} -> {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the log file is in the same directory as the script or provide the full path.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
        
    if not gps_time:
        print("No data was loaded. Check the CSV file format and content.")
        return

    # Convert lists to numpy arrays for easier calculations if needed
    gps_time = np.array(gps_time)
    # Normalize gps_time to start from 0 for cleaner plots
    gps_time_normalized = gps_time - gps_time[0]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Position Data (Lat, Long, Alt)
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle('Aircraft Position vs. Time', fontsize=16)

    ax1.plot(gps_time_normalized, lat, label='Latitude', color='b')
    ax1.set_ylabel('Latitude (degrees)')
    ax1.legend()

    ax2.plot(gps_time_normalized, long, label='Longitude', color='r')
    ax2.set_ylabel('Longitude (degrees)')
    ax2.legend()

    ax3.plot(gps_time_normalized, alt, label='Altitude', color='g')
    ax3.set_ylabel('Altitude (units)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()

    # Figure 2: Attitude Data (Roll, Pitch, Yaw)
    fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle('Aircraft Attitude vs. Time', fontsize=16)

    ax4.plot(gps_time_normalized, np.rad2deg(np.array(roll)), label='Roll', color='purple')
    ax4.set_ylabel('Roll (degrees)')
    ax4.legend()

    ax5.plot(gps_time_normalized, np.rad2deg(np.array(pitch)), label='Pitch', color='orange')
    ax5.set_ylabel('Pitch (degrees)')
    ax5.legend()

    ax6.plot(gps_time_normalized, np.rad2deg(np.array(yaw)), label='Yaw', color='teal')
    ax6.set_ylabel('Yaw (degrees)')
    ax6.set_xlabel('Time (s)')
    ax6.legend()

    # Figure 3: Estimated Acceleration
    fig3, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig3.suptitle('Estimated Acceleration vs. Time', fontsize=16)

    ax7.plot(gps_time_normalized, est_ax, label='Est. Accel X', color='c')
    ax7.set_ylabel('Est. Accel X (units/s^2)')
    ax7.legend()

    ax8.plot(gps_time_normalized, est_ay, label='Est. Accel Y', color='m')
    ax8.set_ylabel('Est. Accel Y (units/s^2)')
    ax8.legend()

    ax9.plot(gps_time_normalized, est_az, label='Est. Accel Z', color='y')
    ax9.set_ylabel('Est. Accel Z (units/s^2)')
    ax9.set_xlabel('Time (s)')
    ax9.legend()
    
    # Figure 4: 2D Flight Path
    fig4, ax10 = plt.subplots(1, 1, figsize=(10, 8))
    fig4.suptitle('2D Flight Path (Longitude vs. Latitude)', fontsize=16)
    ax10.plot(long, lat, marker='.', linestyle='-', markersize=2)
    ax10.set_xlabel('Longitude (degrees)')
    ax10.set_ylabel('Latitude (degrees)')
    ax10.set_title('Top-down View of Flight Path')
    ax10.set_aspect('equal', adjustable='box')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    plot_log_data('flight_logs/vehicle_log_20250904_125448.csv')
