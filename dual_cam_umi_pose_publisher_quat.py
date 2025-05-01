#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
import tf
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
from OPF_Quat import OPF_3d  # UPDATED to use quaternion-enabled OPF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_opf_particles(opf_obj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    particles = opf_obj.particles
    ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='blue', s=1, label='Particles')

    filtered_pos = opf_obj.curr_pos
    ax.scatter(filtered_pos[0], filtered_pos[1], filtered_pos[2], c='red', s=50, label='Filtered Pose')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('OPF Particle Filter Visualization')
    ax.legend()
    plt.show(block=False)
    plt.pause(0.01)
    plt.close(fig)

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

umi_tag_size = 0.050
base_tag_size = 0.040

base_tags = { 17: np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    18: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    19: np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
    20: np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    21: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    22: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) }

umi_tags = { 579: np.eye(3), 580: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    581: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), 582: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    583: np.eye(3), 584: np.eye(3) }

base_to_world = np.array([[1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[0, 0, 0, 1]])

umi_to_gripper = np.array([[1, 0, 0, 0],[0, 1, 0, umi_tag_size/2+0.025],[0, 0, 1, 0.26],[0, 0, 0, 1]])

context = rs.context()
print("Connected devices:")
for device in context.devices:
    print(f"Device Name: {device.get_info(rs.camera_info.name)}")
    print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")

rospy.init_node('dual_cam_umi_gripper_pose_publisher')

base_pose_pub = rospy.Publisher('/base_pose_fused', PoseStamped, queue_size=10)
umi_cube_pose_pub = rospy.Publisher('/umi_pose_fused', PoseStamped, queue_size=10)
umi_ee_pose_pub = rospy.Publisher('/umi_ee_pose_fused', PoseStamped, queue_size=10)
umi_pose_world_pub = rospy.Publisher('/umi_pose_base', PoseStamped, queue_size=10)
cam1_pose_pub = rospy.Publisher('/camera1_pose', PoseStamped, queue_size=10)
cam2_pose_pub = rospy.Publisher('/camera2_pose', PoseStamped, queue_size=10)

tf_broadcaster = tf.TransformBroadcaster()

obj_OPF = OPF_3d(num_particles=5000, name="umi_gripper")

pipe1 = rs.pipeline()
config1 = rs.config()
config1.enable_device('215322079295')
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

pipe2 = rs.pipeline()
config2 = rs.config()
config2.enable_device('146322071961')
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

profile1 = pipe1.start(config1)
profile2 = pipe2.start(config2)

color_profile1 = profile1.get_stream(rs.stream.color)
intr1 = color_profile1.as_video_stream_profile().get_intrinsics()
K1 = np.array([[intr1.fx, 0, intr1.ppx],[0, intr1.fy, intr1.ppy],[0, 0, 1]])

color_profile2 = profile2.get_stream(rs.stream.color)
intr2 = color_profile2.as_video_stream_profile().get_intrinsics()
K2 = np.array([[intr2.fx, 0, intr2.ppx],[0, intr2.fy, intr2.ppy],[0, 0, 1]])

options = apriltag.DetectorOptions(families='tag36h11', nthreads=4)
detector = apriltag.Detector(options)

def process_camera(pipe, camera_index):
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(f'Camera {camera_index}', cv2.WINDOW_AUTOSIZE)
    if not color_frame:
        print(f"Camera {camera_index}: No frame captured.")
        return None, None

    detections = []
    for tag in detector.detect(gray_image):
        pose = detector.detection_pose(tag, (intr1.fx, intr1.fy, intr1.ppx, intr1.ppy) if camera_index == 1 else (intr2.fx, intr2.fy, intr2.ppx, intr2.ppy), umi_tag_size, z_sign=1)[0]
        detections.append({'id': tag.tag_id, 'pose': pose, 'corners': tag.corners})
        (ptA, ptB, ptC, ptD) = tag.corners
        ptA, ptB, ptC, ptD = map(lambda pt: (int(pt[0]), int(pt[1])), [ptA, ptB, ptC, ptD])
        cv2.line(color_image, ptA, ptB, (0, 255, 0), 1)
        cv2.line(color_image, ptB, ptC, (0, 255, 0), 1)
        cv2.line(color_image, ptC, ptD, (0, 255, 0), 1)
        cv2.line(color_image, ptD, ptA, (0, 255, 0), 1)
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(color_image, f'ID: {tag.tag_id}', (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return detections, color_image

pixel_coordinates_list_1 = []
pixel_coordinates_list_2 = []

base_cube_pose_1 = np.eye(4)
base_cube_pose_2 = np.eye(4)

while not rospy.is_shutdown():
    detections1, img1 = process_camera(pipe1, 1)    
    detections2, img2 = process_camera(pipe2, 2)    

    umi_poses_1 = []
    base_poses_1 = []
    umi_poses_2 = []
    base_poses_2 = []

    rospy.loginfo_throttle(1.0, f"CAM1 Detected tag IDs: {[det['id'] for det in detections1]}")
    rospy.loginfo_throttle(1.0, f"CAM2 Detected tag IDs: {[det['id'] for det in detections2]}")

    for det in detections1:
        if det['id'] in umi_tags:
            umi_poses_1.append(det['pose'])
        elif det['id'] in base_tags:
            base_poses_1.append(det['pose'])

    for det in detections2:
        if det['id'] in umi_tags:
            umi_poses_2.append(det['pose'])
        elif det['id'] in base_tags:
            base_poses_2.append(det['pose'])

    camera1_to_base = np.eye(4)
    camera2_to_base = np.eye(4)

    if len(base_poses_1) > 0:
        base_cube_pose_1 = base_poses_1[0]
        camera1_to_base = np.linalg.inv(base_cube_pose_1)

    if len(base_poses_2) > 0:
        base_cube_pose_2 = base_poses_2[0]
        camera2_to_base = np.linalg.inv(base_cube_pose_2)

    valid_measurements = []

    if len(umi_poses_1) > 0 and len(base_poses_1) > 0:
        umi_cube_wrt_base_1 = np.linalg.inv(base_poses_1[0]) @ umi_poses_1[0]
        trans = umi_cube_wrt_base_1[:3, 3]
        quat = Rotation.from_matrix(umi_cube_wrt_base_1[:3, :3]).as_quat()
        measurement = np.array([*trans, *quat])
        valid_measurements.append(measurement)

    if len(umi_poses_2) > 0 and len(base_poses_2) > 0:
        umi_cube_wrt_base_2 = np.linalg.inv(base_poses_2[0]) @ umi_poses_2[0]
        trans = umi_cube_wrt_base_2[:3, 3]
        quat = Rotation.from_matrix(umi_cube_wrt_base_2[:3, :3]).as_quat()
        measurement = np.array([*trans, *quat])
        valid_measurements.append(measurement)

    if len(valid_measurements) == 0:
        rospy.logwarn_throttle(3.0, "No valid umi + base pose pairs found. Skipping frame.")
        continue

    obj_OPF.predict()
    for measurement in valid_measurements:
        obj_OPF.update(measurement)
    obj_OPF.systematic_resample()
    obj_OPF.resample_from_index()

    filtered_translation = obj_OPF.curr_pos
    filtered_quat = obj_OPF.curr_quat
    filtered_rot_matrix = Rotation.from_quat(filtered_quat).as_matrix()

    umi_cube_pose = np.eye(4)
    umi_cube_pose[:3, 3] = filtered_translation
    umi_cube_pose[:3, :3] = filtered_rot_matrix

    umi_ee_pose = umi_cube_pose @ umi_to_gripper

    umi_pose_msg = create_pose_msg(umi_cube_pose, "umi_cube")
    umi_cube_pose_pub.publish(umi_pose_msg)

    umi_pose_msg = create_pose_msg(umi_ee_pose, "umi_ee")
    umi_ee_pose_pub.publish(umi_pose_msg)

    tf_broadcaster.sendTransform(
        (umi_cube_pose[0, 3], umi_cube_pose[1, 3], umi_cube_pose[2, 3]),
        tft.quaternion_from_matrix(umi_cube_pose),
        rospy.Time.now(), "umi_cube", "april_base")

    tf_broadcaster.sendTransform(
        (umi_ee_pose[0, 3], umi_ee_pose[1, 3], umi_ee_pose[2, 3]),
        tft.quaternion_from_matrix(umi_ee_pose),
        rospy.Time.now(), "umi_ee", "april_base")

    tf_broadcaster.sendTransform(
        (camera1_to_base[0, 3], camera1_to_base[1, 3], camera1_to_base[2, 3]),
        tft.quaternion_from_matrix(camera1_to_base),
        rospy.Time.now(), "cam1", "april_base")

    tf_broadcaster.sendTransform(
        (camera2_to_base[0, 3], camera2_to_base[1, 3], camera2_to_base[2, 3]),
        tft.quaternion_from_matrix(camera2_to_base),
        rospy.Time.now(), "cam2", "april_base")

    cv2.imshow('Camera 1', img1)
    cv2.imshow('Camera 2', img2)
    cv2.waitKey(1)

    visualize_opf_particles(obj_OPF)
