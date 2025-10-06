#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R

class DroneAscend(Node):
    def __init__(self):
        super().__init__('drone_ascend_controller')

        # Target altitude
        self.target_z = 3.0  # Desired altitude in meters

        # Hovering throttle
        self.hover_throttle = 0.3593  # Base throttle for hovering

        # PID control constants for altitude adjustment
        self.kp = 0.05
        self.ki = 0.01
        self.kd = 0.05

        # PID control variables
        self.prev_error = 0.0
        self.integral = 0.0

        # Current altitude
        self.current_z = 0.0

        # Roll and pitch
        self.roll = 0.0
        self.pitch = 0.0

        # Subscriber for odometry
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publishers for propellers
        self.propeller1_pub = self.create_publisher(Float64, '/propeller1', 10)
        self.propeller2_pub = self.create_publisher(Float64, '/propeller2', 10)
        self.propeller3_pub = self.create_publisher(Float64, '/propeller3', 10)
        self.propeller4_pub = self.create_publisher(Float64, '/propeller4', 10)

        # Timer to send commands periodically
        self.timer = self.create_timer(0.1, self.send_throttle_commands)

    def odom_callback(self, msg):
        # Update current altitude from odometry
        self.current_z = msg.pose.pose.position.z

        # Extract quaternion from odometry message
        quaternion = msg.pose.pose.orientation
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)

        self.roll = euler[0]
        self.pitch = euler[1]

        # Log roll and pitch for debugging
        self.get_logger().info(
            f"Roll: {self.roll:.3f}, Pitch: {self.pitch:.3f}"
        )

    def send_throttle_commands(self):
        # Calculate the error for altitude
        error = self.target_z - self.current_z
        
        # PID calculations
        self.integral += error * 0.1  # Integral term
        derivative = (error - self.prev_error) / 0.1  # Derivative term
        throttle_adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Calculate the final throttle value
        throttle = self.hover_throttle + throttle_adjustment

        # Ensure throttle is within bounds [0.0, 1.0]
        throttle = max(0.0, min(0.5, throttle))

        # Log the throttle value and altitude for debugging
        self.get_logger().info(
            f"Altitude:{self.current_z:.2f},Throttle:{throttle:.3f},"
            f"Roll: {self.roll}, Pitch: {self.pitch}"
        )

        # Publish throttle values equally to all four propellers
        self.publish_throttle(self.propeller1_pub, throttle+(self.pitch*0.1/2))
        self.publish_throttle(self.propeller2_pub, throttle-(self.pitch*0.1/2))
        self.publish_throttle(self.propeller3_pub, throttle-(self.roll*0.1/2))
        self.publish_throttle(self.propeller4_pub, throttle+(self.roll*0.1/2))

    def publish_throttle(self, publisher, value):
        # Publish throttle value
        msg = Float64()
        msg.data = value
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DroneAscend()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
