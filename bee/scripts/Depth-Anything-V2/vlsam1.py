import argparse
import cv2
import numpy as np
import torch
import open3d as o3d

from depth_anything_v2.dpt import DepthAnythingV2

def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Camera Livestream - Point Cloud Only Visualization')
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path (e.g., /dev/video0 or 0 for default camera)')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
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
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    # checkpoints/depth_anything_v2_vitb.pth
    # Open camera stream
    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera at {args.camera_device}")
        exit()
    
    # Camera intrinsics (example Fantech C30) - adjust if you have exact calibration
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_x = 301.24  # You had divided by 10 before; use original or calibrated values
    f_y = 301.24
    c_x = frame_width
    c_y = frame_height
    
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.904, 1.0, 0.0, 0.0, -0.193])
    
    # Precompute undistort maps for efficiency
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (frame_width, frame_height), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (frame_width, frame_height), cv2.CV_32FC1)
    
    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True
    
    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Undistort the frame first
        undistorted_frame = cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        # Generate depth map from undistorted frame
        depth = depth_anything.infer_image(undistorted_frame, args.input_size)  # Shape: (518, 518)
        depth = cv2.resize(depth, (frame_width, frame_height))  # Resize to camera resolution
        
        # Compute 3D points (downsampled for performance)
        small_size = (100, 100)  # Downsample to 100x100 for real-time
        depth_small = cv2.resize(depth, small_size)
        u_small, v_small = np.meshgrid(np.arange(small_size[0]), np.arange(small_size[1]))
        
        scale_x = small_size[0] / frame_width
        scale_y = small_size[1] / frame_height
        
        X = (u_small - c_x * scale_x) * depth_small / (f_x * scale_x)
        Y = -(v_small - c_y * scale_y) * depth_small / (f_y * scale_y)  # Flip Y-axis
        Z = depth_small
        
        points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # Shape: (10000, 3)
        
        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    vis.destroy_window()

if __name__ == '__main__':
    main()
