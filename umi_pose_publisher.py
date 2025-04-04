#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
import tf

np.set_printoptions(suppress=True)

# Initialize ROS node
rospy.init_node('umi_gripper_pose_publisher')
pub_cam = rospy.Publisher('/umi_gripper/pose_camera', PoseStamped, queue_size=10)
pub_world = rospy.Publisher('/umi_gripper/pose_world', PoseStamped, queue_size=10)
tf_broadcaster = tf.TransformBroadcaster()

# Load calibration: tag → gripper transform
def find_transform(tag_id):
    transforms = np.load("calibrated_transforms.npz")
    return transforms[f'from_{tag_id}']

# Transform from camera to base (world) frame
camera_to_base = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])

# Start RealSense
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

options = apriltag.DetectorOptions(families='tag36h11', nthreads=4)
detector = apriltag.Detector(options)

profile = pipe.start(config)
color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()

K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]])
camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
tag_size = 0.050

gripper_tags = {579, 580, 581, 582, 583, 584}
pixel_coordinates_list = []

def create_pose_msg(matrix, frame_id):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = matrix[0, 3]
    pose_msg.pose.position.y = matrix[1, 3]
    pose_msg.pose.position.z = matrix[2, 3]
    quat = tft.quaternion_from_matrix(matrix)
    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]
    return pose_msg

while not rospy.is_shutdown():
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    detected_tags = detector.detect(gray_image)
    ee_poses = []

    for tag in detected_tags:
        tag_id = tag.tag_id
        if tag_id in gripper_tags:
            tag_pose = detector.detection_pose(tag, camera_params, tag_size, z_sign=1)[0]
            transform = find_transform(tag_id)
            ee_pose = tag_pose @ transform
            ee_poses.append(ee_pose)

        # Draw tag
        (ptA, ptB, ptC, ptD) = tag.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        cv2.line(color_image, ptA, ptB, (0, 255, 0), 1)
        cv2.line(color_image, ptB, ptC, (0, 255, 0), 1)
        cv2.line(color_image, ptC, ptD, (0, 255, 0), 1)
        cv2.line(color_image, ptD, ptA, (0, 255, 0), 1)
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

    if len(ee_poses) > 0:
        ee_poses = np.array(ee_poses)
        avg_ee_cam = np.mean(ee_poses, axis=0)

        if avg_ee_cam.shape == (4, 4):
            # Visualization
            point_3d = avg_ee_cam[:3, 3]
            point_2d = K @ point_3d
            pixel_coords = point_2d[:2] / point_2d[2]
            pixel_coordinates_list.append(pixel_coords)
            if len(pixel_coordinates_list) > 100:
                pixel_coordinates_list.pop(0)

            for pt in pixel_coordinates_list:
                cv2.circle(color_image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

            # Pose in world frame
            avg_ee_base = camera_to_base @ avg_ee_cam

            # Publish camera frame pose
            pose_cam_msg = create_pose_msg(avg_ee_cam, "camera_link")
            pub_cam.publish(pose_cam_msg)

            # Publish world frame pose
            pose_world_msg = create_pose_msg(avg_ee_base, "base_link")
            pub_world.publish(pose_world_msg)

            # TF broadcast for RViz
            quat = tft.quaternion_from_matrix(avg_ee_base)
            tf_broadcaster.sendTransform(
                (avg_ee_base[0, 3], avg_ee_base[1, 3], avg_ee_base[2, 3]),
                quat,
                rospy.Time.now(),
                "umi_gripper",
                "base_link"
            )

    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)
