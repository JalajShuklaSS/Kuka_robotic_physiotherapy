#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import WrenchStamped
import os
import csv
import time

# File path for storing data
output_file_path = os.path.join(os.getcwd(), "force_torque_sensor_data.csv")

# Ensure the CSV file has headers (only write them once)
if not os.path.exists(output_file_path):
    with open(output_file_path, "w", newline='') as file:
        headers = [
            "Timestamp",
            "Force_X", "Force_Y", "Force_Z",
            "Torque_X", "Torque_Y", "Torque_Z"
        ]
        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)

def sensor_callback(data):
    # Open file in append mode inside the callback to ensure proper flushing
    with open(output_file_path, "a", newline='') as file:
        csv_writer = csv.writer(file)

        # Get the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Extract force and torque values from the message
        force_x = data.wrench.force.x
        force_y = data.wrench.force.y
        force_z = data.wrench.force.z

        torque_x = data.wrench.torque.x
        torque_y = data.wrench.torque.y
        torque_z = data.wrench.torque.z

        # Write the timestamp, forces, and torques to the CSV file
        csv_writer.writerow([timestamp, force_x, force_y, force_z, torque_x, torque_y, torque_z])

def main():
    rospy.init_node('force_torque_sensor_logger', anonymous=True)

    # Subscribe to the sensor topic
    rospy.Subscriber('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, sensor_callback)

    rospy.loginfo(f"Logging force-torque sensor data to {output_file_path}")

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
