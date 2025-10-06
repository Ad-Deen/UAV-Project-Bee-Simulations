#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        # Clamp output to range [-1, 1]
        return max(-1, min(1, output))

# Initialize PID Controllers for Pitch, Roll, and Yaw
pitch_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
roll_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
yaw_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)

class DroneAscend(Node):
    def __init__(self):
        super().__init__('drone_ascend_controller')

        # Target altitude
        self.target_tx = 0.0  # Desired altitude in meters
        self.target_ty = 0.0  # Desired altitude in meters
        self.target_tz = 3.0  # Desired altitude in meters
        self.target_roll = 0.0  # Desired orientation in rad
        self.target_pitch = 0.0  # Desired orientation in rad
        self.target_yaw = 0.0  # Desired orientation in rad

        # Hovering throttle
        self.hover_throttle = 0.3593  # Base throttle for hovering

        # PID control constants for altitude adjustment
        self.kp = 0.1
        self.ki = 0.01
        self.kd = 0.05

        # PID control variables
        self.prev_error = 0.0
        self.integral = 0.0

        # Current altitude
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

        # Roll and pitch
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.pitch_trigger = 0      #pitch =tx+(+1) tx-(-1)
        self.roll_trigger = 0       #roll =ty+(+1) ty-(-1)
        self.yaw_trigger = 0       #Yaw =Rz+(+1) Rz-(-1)

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
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z

        # Extract quaternion from odometry message
        quaternion = msg.pose.pose.orientation
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)

        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]

        # Log roll and pitch for debugging
        self.get_logger().info(
            f"Roll: {self.roll:.3f}, Pitch: {self.pitch:.3f}"
        )

    def send_throttle_commands(self):

        target_pos = (2.0, 0, 5.0 , 0.0)  # Example target point (x, y, z , yaw)

        # Current position and orientation
        current_pos = (self.current_x, self.current_y, self.current_z)
        current_orientation = (self.roll, self.pitch, self.yaw)

        # Calculate navigation triggers
        triggers = navigate_to_target(current_pos, current_orientation, target_pos,0.1)

        self.pitch_trigger = triggers["pitch_trigger"]
        self.roll_trigger = triggers["roll_trigger"]
        self.yaw_trigger = triggers["yaw_trigger"]
        self.target_tz = triggers["target_tz"]

        # Calculate the error for altitude
        error = self.target_tz - self.current_z

        pitch_offset = 0.001*self.pitch_trigger       #if pitch offset +ve(pitch_trigger +1) we get Tx -> +ve translation and vice versa
        roll_offset = 0.001*self.roll_trigger        #if roll offset +ve we get Ty -> +ve translation and vice versa
        yaw_offset = 0.001*self.yaw_trigger*0              #if yaw offset +v we cw Rz -> +ve rotation and vice versa
        
        # PID calculations
        self.integral += error * 0.1  # Integral term
        derivative = (error - self.prev_error) / 0.1  # Derivative term
        throttle_adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Calculate the final throttle value
        throttle = self.hover_throttle + throttle_adjustment

        # Ensure throttle is within bounds [0.0, 1.0]
        throttle = max(0.0, min(self.hover_throttle*2, throttle))

        # Log the throttle value and altitude for debugging
        self.get_logger().info(
            f"Altitude:{self.current_z:.2f},Throttle:{throttle:.3f},"
            f"Roll: {self.roll}, Pitch: {self.pitch}"
        )

        # Publish throttle values equally to all four propellers
        self.publish_throttle(self.propeller1_pub, throttle+(self.pitch*0.1/2)-pitch_offset + yaw_offset)
        self.publish_throttle(self.propeller2_pub, throttle-(self.pitch*0.1/2)+pitch_offset + yaw_offset)
        self.publish_throttle(self.propeller3_pub, throttle-(self.roll*0.1/2)-roll_offset - yaw_offset)
        self.publish_throttle(self.propeller4_pub, throttle+(self.roll*0.1/2)+roll_offset - yaw_offset)

    def publish_throttle(self, publisher, value):
        # Publish throttle value
        msg = Float64()
        msg.data = value
        publisher.publish(msg)

def navigate_to_target(current_pos, current_orientation, target_pos,dt):
    """
    Calculate triggers to navigate the drone to the target position.
    
    Parameters:
    - current_pos: Tuple (x, y, z) representing the current position of the drone.
    - current_orientation: Tuple (roll, pitch, yaw) representing the current orientation.
    - target_pos: Tuple (x_target, y_target, z_target) representing the target position.
    
    Returns:
    - Dictionary with triggers for pitch, roll, yaw, and target_tz.
    """
    current_x, current_y, current_z = current_pos
    current_roll, current_pitch, current_yaw = current_orientation
    x_target, y_target, z_target , yaw_target = target_pos

    # Calculate the positional differences
    delta_x = x_target - current_x  # Tx adjustment
    delta_y = y_target - current_y  # Ty adjustment
    delta_z = z_target - current_z  # Tz 
    delta_yaw = yaw_target - current_yaw

     # Normalize yaw error to range [-pi, pi] for correct rotation direction
    delta_yaw = (delta_yaw + 3.14159) % (2 * 3.14159) - 3.14159

    # Use PID controllers for smooth triggers
    pitch_trigger = pitch_pid.compute(delta_x, dt)  # Smooth adjustment for forward/backward motion
    roll_trigger = roll_pid.compute(delta_y, dt)   # Smooth adjustment for lateral motion
    yaw_trigger = yaw_pid.compute(delta_yaw, dt)   # Smooth adjustment for yaw alignment


    # Set target altitude
    target_tz = z_target

    return {
        "pitch_trigger": pitch_trigger,
        "roll_trigger": roll_trigger,
        "yaw_trigger": yaw_trigger,
        "target_tz": target_tz,
    }

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
