import argparse
import cv2
import matplotlib
import numpy as np
import torch
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'point_cloud', 10)
        self.clock = self.get_clock()

    def publish_pointcloud(self, points):
        header = Header()
        header.stamp = self.clock.now().to_msg()
        header.frame_id = 'map'

        # Set all RGB to white
        rgb = np.ones((points.shape[0], 3), dtype=np.uint8) * 255
        rgb_packed = (rgb[:, 0].astype(np.uint32) << 16) | \
                    (rgb[:, 1].astype(np.uint32) << 8) | \
                    (rgb[:, 2].astype(np.uint32))
        rgb_packed = rgb_packed.view(np.float32)

        points_xyzrgb = np.hstack([points.astype(np.float32), rgb_packed.reshape(-1, 1)])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = pc2.create_cloud(header, fields, points_xyzrgb)
        self.publisher_.publish(msg)



from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Camera Livestream')
    
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path (e.g., /dev/video0 or 0 for default camera)')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # Device setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize and load the model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Open camera stream
    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera at {args.camera_device}")
        exit()
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Visualization settings
    margin_width = 50
    cmap = matplotlib.colormaps['Spectral_r']
    
    # Camera intrinsics (Fantech C30, DFOV=106°, HFOV=93.48°, VFOV=77.04°)
    f_x = 301.24
    f_y = 301.24
    c_x = 320.0
    c_y = 240.0
    
    # Initialize Open3D visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True
    
    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")
    rclpy.init()
    pointcloud_publisher_node = PointCloudPublisher()

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Generate depth map
        depth = depth_anything.infer_image(raw_frame, args.input_size)  # Shape: (518, 518)
        depth = cv2.resize(depth, (640, 480))  # Resize to (480, 640)
        
        # Compute 3D points (downsampled for performance)
        small_size = (100, 100)  # Downsample to 100x100 for real-time
        depth_small = cv2.resize(depth, small_size)
        u_small, v_small = np.meshgrid(np.arange(small_size[0]), np.arange(small_size[1]))
        scale_x = small_size[0] / 640
        scale_y = small_size[1] / 480
        X = (u_small - c_x * scale_x) * depth_small / (f_x * scale_x)
        Y = -(v_small - c_y * scale_y) * depth_small / (f_y * scale_y)  # Flip Y-axis
        Z = depth_small
        points_3d = np.stack([Z, Y, X], axis=-1).reshape(-1, 3)  # Shape: (10000, 3)
        
        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pointcloud_publisher_node.publish_pointcloud(points_3d)
        rclpy.spin_once(pointcloud_publisher_node, timeout_sec=0)  # Keeps ROS ticking

    #     if first_frame:
    #         vis.add_geometry(pcd)
    #         first_frame = False
    #     else:
    #         vis.update_geometry(pcd)
        
    #     # Visualize depth map (original 2D visualization)
    #     depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    #     depth_normalized = depth_normalized.astype(np.uint8)
    #     if args.grayscale:
    #         depth_vis = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
    #     else:
    #         depth_vis = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
    #     # Prepare output frame
    #     if args.pred_only:
    #         display_frame = depth_vis
    #     else:
    #         split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
    #         display_frame = cv2.hconcat([raw_frame, split_region, depth_vis])
        
    #     # Display 2D frame
    #     cv2.imshow('Depth Anything V2 - Live', display_frame)
        
    #     # Update 3D visualization
    #     vis.poll_events()
    #     vis.update_renderer()
        
    #     # Exit on 'q' key press
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # # Cleanup
    # cap.release()
    # vis.destroy_window()
    # cv2.destroyAllWindows()

    pointcloud_publisher_node.destroy_node()
    rclpy.shutdown()
