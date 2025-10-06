#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge

class CameraOdomNode(Node):
    def __init__(self):
        super().__init__('camera_odom_node')
        self.bridge = CvBridge()
        
        # Subscription to the /camera topic
        self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            10
        )
        
        # Subscription to the /odom topic
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info("CameraOdomNode is up and running!")

    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Display the video frame
            cv2.imshow("Camera Frame", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def odom_callback(self, msg):
        # Extract position and orientation data
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        # Format and print as [Tx, Ty, Tz, Rx, Ry, Rz]
        odom_data = f"[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}, {ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}]"
        self.get_logger().info(f"Odom: {odom_data}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraOdomNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
