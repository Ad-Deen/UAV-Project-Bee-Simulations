#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

class JointForcePublisher(Node):
    def __init__(self):
        super().__init__('joint_force_publisher')
        
        # Initialize target position (Z-axis)
        self.target_z = 2.0  # Default target Z value (this can be changed by the user)

        # PID Constants
        self.kp = 0.05  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.1  # Derivative gain
        
        # Variables to store previous error and integral term for PID
        self.prev_error = 0.0
        self.integral = 0.0

        # Set up a timer to call the command sending function periodically (10 Hz)
        self.timer = self.create_timer(0.1, self.send_joint_commands)

        # Subscriber to Odometry topic to get the drone's current position
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Current Z position of the drone
        self.current_z = 0.0

        # Publishers for propeller control
        self.propeller1_pub = self.create_publisher(Float64, '/propeller1', 10)
        self.propeller2_pub = self.create_publisher(Float64, '/propeller2', 10)
        self.propeller3_pub = self.create_publisher(Float64, '/propeller3', 10)
        self.propeller4_pub = self.create_publisher(Float64, '/propeller4', 10)

    def odom_callback(self, msg):
        # Extract the current Z position from the odometry message
        self.current_z = msg.pose.pose.position.z
        self.get_logger().info(f"Current Z position: {self.current_z}")

    def send_joint_commands(self):
        # Calculate the error (difference between target and current Z position)
        z_diff = self.target_z - self.current_z
        
        # PID calculations
        error = z_diff
        self.integral += error * 0.1  # Integral term (based on time step of 0.1s)
        derivative = (error - self.prev_error) / 0.1  # Derivative term
        throttle_adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Store the current error for the next cycle
        self.prev_error = error

        # Calculate the new throttle value for each rotor (keeping it between 0 and 1)
        new_throttle = max(0.0, min(1.0, throttle_adjustment))
        
        # Log the PID values for debugging
        self.get_logger().info(f"Target Z: {self.target_z}, Current Z: {self.current_z}, "
                               f"Error: {error}, Throttle Adjustment: {throttle_adjustment}, "
                               f"New Throttle: {new_throttle}")

        # Publish throttle value to each propeller
        msg = Float64()
        msg.data = new_throttle
        
        self.propeller1_pub.publish(msg)
        self.propeller2_pub.publish(msg)
        self.propeller3_pub.publish(msg)
        self.propeller4_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointForcePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
