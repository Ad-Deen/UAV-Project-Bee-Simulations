#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import socket
import struct
from std_msgs.msg import UInt8MultiArray, Float32MultiArray

class FullDuplexUDP(Node):
    def __init__(self):
        super().__init__('full_duplex_udp')

        # ESP32 config
        self.esp_ip = '192.168.0.110'  # explore
        self.esp_port = 4210

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 4210))  # Bind to same port to receive
        self.sock.setblocking(False)

        # ROS interfaces
        self.command_sub = self.create_subscription(
            UInt8MultiArray,
            'rotor_commands',
            self.send_to_esp,
            10
        )
        self.feedback_pub = self.create_publisher(
            Float32MultiArray,
            'quadcopter_feedback',
            10
        )

        # Timer to check incoming data
        self.timer = self.create_timer(0.01, self.read_from_esp)

    def send_to_esp(self, msg):
        if len(msg.data) == 4:
            packet = bytes(msg.data)
            self.sock.sendto(packet, (self.esp_ip, self.esp_port))
        else:
            self.get_logger().warn("Expected 4 rotor command bytes.")

    def read_from_esp(self):
        try:
            data, _ = self.sock.recvfrom(64)
            if len(data) == 12:
                r1, r2, r3, r4 = data[0:4]
                gyroX, gyroY = struct.unpack('ff', data[4:12])
                msg = Float32MultiArray()
                msg.data = [r1, r2, r3, r4, gyroX, gyroY]
                self.feedback_pub.publish(msg)
        except BlockingIOError:
            pass  # no data

def main(args=None):
    rclpy.init(args=args)
    node = FullDuplexUDP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
