import argparse
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with FOV-based 3D Projection')
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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

    frame_width, frame_height = 640, 480

    # FOV projection parameters
    diag_fov_deg = 106
    aspect = frame_width / frame_height
    diag_fov_rad = np.deg2rad(diag_fov_deg)
    hfov_rad = 2 * np.arctan(np.tan(diag_fov_rad / 2) * (aspect / np.sqrt(1 + aspect**2)))
    vfov_rad = 2 * np.arctan(np.tan(diag_fov_rad / 2) * (1 / np.sqrt(1 + aspect**2)))

    x_angles = np.linspace(-hfov_rad/2, hfov_rad/2, frame_width)
    y_angles = np.linspace(-vfov_rad/2, vfov_rad/2, frame_height)
    x_grid, y_grid = np.meshgrid(x_angles, y_angles)

    ray_dirs = np.stack([
        np.tan(x_grid),
        -np.tan(y_grid),
        np.ones_like(x_grid)
    ], axis=-1)
    ray_dirs /= np.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        depth = depth_anything.infer_image(frame, args.input_size)
        depth = cv2.resize(depth, (frame_width, frame_height))

        points_3d = depth[..., None] * ray_dirs
        points_3d = points_3d.reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(points_3d)
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
