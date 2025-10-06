import argparse
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with FOV-Based 3D Projection')
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera at {args.camera_device}")
        exit()

    width = 640
    height = 480
    HFOV = 93.48  # degrees
    VFOV = 77.04  # degrees

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    cmap = matplotlib.colormaps['Spectral_r']

    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Generate depth map
        depth = depth_anything.infer_image(raw_frame, args.input_size)  # Shape: (518, 518)
        depth = cv2.resize(depth, (width, height))  # Resize to original camera resolution

        # Convert FOV to radians
        HFOV_rad = np.deg2rad(HFOV)
        VFOV_rad = np.deg2rad(VFOV)

        dtheta = HFOV_rad / width
        dphi = VFOV_rad / height

        u = np.arange(width)
        v = np.arange(height)
        theta = (u - width / 2) * dtheta  # Horizontal angles
        phi = -(v - height / 2) * dphi    # Vertical angles

        theta_grid, phi_grid = np.meshgrid(theta, phi)

        X = depth * np.cos(phi_grid) * np.sin(theta_grid)
        Y = depth * np.sin(phi_grid)
        Z = depth * np.cos(phi_grid) * np.cos(theta_grid)

        points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        # 2D Depth visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        if args.grayscale:
            depth_vis = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            display_frame = depth_vis
        else:
            margin = np.ones((height, 10, 3), dtype=np.uint8) * 255
            display_frame = cv2.hconcat([raw_frame, margin, depth_vis])

        cv2.imshow('Depth Anything V2 - Live', display_frame)
        vis.poll_events()
        vis.update_renderer()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()