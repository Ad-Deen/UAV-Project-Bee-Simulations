import argparse
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-device', type=str, default='/dev/video0')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb')
    parser.add_argument('--pred-only', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print("Error: Camera not opened")
        exit()

    W, H = 640, 480
    HFOV_deg = 93.48
    VFOV_deg = 77.04
    HFOV = np.deg2rad(HFOV_deg)
    VFOV = np.deg2rad(VFOV_deg)
    f_x = W / (2 * np.tan(HFOV / 2))
    f_y = H / (2 * np.tan(VFOV / 2))
    c_x, c_y = W / 2, H / 2

    cmap = matplotlib.colormaps['Spectral_r']
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first = True

    print("Streaming. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth = model.infer_image(frame, args.input_size)
        # print(depth.shape)
        # depth = cv2.resize(depth, (W, H))
        depth = np.round(1.0 / (depth + 1e-6), 5)

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        X = np.round((u - c_x) * depth / f_x, 5)
        Y = np.round(-(v - c_y) * depth / f_y, 5)
        Z = -depth
        pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(pts)
        if first:
            vis.add_geometry(pcd)
            first = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        # Display depth map
        d_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        if args.grayscale:
            d_vis = np.stack([d_norm] * 3, axis=-1)
        else:
            d_vis = (cmap(d_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]

        if args.pred_only:
            display = d_vis
        else:
            margin = np.ones((H, 10, 3), dtype=np.uint8) * 255
            display = cv2.hconcat([frame, margin, d_vis])

        cv2.imshow('Depth Visualization', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
