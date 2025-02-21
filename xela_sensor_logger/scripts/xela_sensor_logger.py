#!/usr/bin/env python3

import rospy
from xela_server_ros.msg import SensStream  # Correctly importing SensStream
import os
import csv
import time

# File path for storing data
output_file_path = os.path.join(os.getcwd(), "xela_sensor_data.csv")

# Ensure the CSV file has headers (only write them once)
if not os.path.exists(output_file_path):
    with open(output_file_path, "w", newline='') as file:
        headers = ["Timestamp"] + [f"Sensor_{i}_Force_{axis}" for i in range(24) for axis in ['x', 'y', 'z']]
        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)

def sensor_callback(data):
    # Open file in append mode inside the callback to ensure proper flushing
    with open(output_file_path, "a", newline='') as file:
        csv_writer = csv.writer(file)

        # Get the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Initialize a list to store all force values
        all_forces = []

        # Iterate through all sensors by index
        for i, sensor in enumerate(data.sensors):
            # Ensure we don't exceed 24 sensors
            if i >= 24:
                rospy.logwarn(f"Extra sensor data detected (index {i}). Skipping additional sensors.")
                break

            # Extract forces for each sensor
            for force in sensor.forces:
                all_forces.extend([force.x, force.y, force.z])

        # Ensure we have exactly 72 force values (24 sensors × 3 axes)
        if len(all_forces) != 72:
            rospy.logerr(f"Incorrect number of force values. Expected 72, got {len(all_forces)}")
            return

        # Write the timestamp and all forces to the CSV file
        csv_writer.writerow([timestamp] + all_forces)

def main():
    rospy.init_node('xela_sensor_logger', anonymous=True)

    # Subscribe to the sensor topic
    rospy.Subscriber('xServTopic', SensStream, sensor_callback)

    rospy.loginfo(f"Logging sensor data to {output_file_path}")

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
