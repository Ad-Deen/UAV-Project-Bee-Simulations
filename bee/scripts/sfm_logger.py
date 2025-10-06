#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
from math import atan2, asin, pi

class CameraOdomNode(Node):
    def __init__(self):
        super().__init__('camera_odom_node')
        self.bridge = CvBridge()
        self.data_buffer = []  # Buffer to store odometry and image data
        self.start_time = time.time()

        # Subscription to the /camera topic
        self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            5
        )

        # Subscription to the /odom topic
        self.create_subscription(
            Odometry,
            '/model/vehicle_blue/odometry',
            self.odom_callback,
            5
        )

        # Publisher for /model/vehicle_blue/cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/model/vehicle_blue/cmd_vel',
            10
        )

        # Publish velocity commands periodically
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)  # Publish at 10 Hz

        self.current_odom = None  # Placeholder for the latest odom data
        self.current_image = None  # Placeholder for the latest image data

        self.get_logger().info("CameraOdomNode is up and running!")

    def quaternion_to_euler(self, x, y, z, w):
        """
        Converts quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
        in radians.
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = min(t2, 1.0)  # Clamp to avoid gimbal lock
        t2 = max(t2, -1.0)  # Clamp to avoid gimbal lock
        pitch_y = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z
    

    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.store_data()
        except Exception as e:
            self.get_logger().info(f"Error processing image: {e}")

    def odom_callback(self, msg):
        # Extract position and orientation data
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        print(ori)
        # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
        roll, pitch, yaw = self.quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)

        # Save as a 6-element NumPy array: [Tx, Ty, Tz, Rx, Ry, Rz]
        self.current_odom = np.array([pos.x, pos.y+4.0, pos.z+1.325, roll, pitch, yaw-1.57])
        self.get_logger().info(f"Angles {self.current_odom}")

    def publish_cmd_vel(self):
        # Create a Twist message
        twist_msg = Twist()
        twist_msg.linear.x = 1.0  # Forward throttle (1 m/s)
        twist_msg.angular.z = -0.25  # Yaw rate (0.25 rad/s)

        # Publish the Twist message
        self.cmd_vel_publisher.publish(twist_msg)

    def store_data(self):
        # Ensure both odom and image data are available
        if self.current_odom is not None and self.current_image is not None:
            # Save odometry and image data as a NumPy array pair
            data_pair = [self.current_odom, self.current_image]
            self.data_buffer.append(data_pair)
            
            self.get_logger().info(f"Time elapsed {time.time()-self.start_time:.3f}")

        # Check if 20 seconds have elapsed
        if time.time() - self.start_time >= 25.0:
            self.save_data()

    def save_data(self):
        # Convert the buffer to a NumPy array
        np_data = np.array(self.data_buffer, dtype=object)  # Use dtype=object for mixed data
        filename = "/media/deen/Extended-Linux/sfm_resources/camera_odom_data.npy"
        np.save(filename, np_data)
        self.get_logger().info(f"Saved data to {filename}")
        self.get_logger().info("Shutting down after saving.")
        rclpy.shutdown()  # Shut down the node after saving

def main(args=None):
    rclpy.init(args=args)
    node = CameraOdomNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
