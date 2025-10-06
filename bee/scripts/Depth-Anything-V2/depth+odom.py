import argparse
import cv2
import matplotlib
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from depth_anything_v2.dpt import DepthAnythingV2

def backproject(u, v, depth, f_x, f_y, c_x, c_y):
    X = (u - c_x) * depth / f_x
    Y = (v - c_y) * depth / f_y
    Z = depth
    return np.array([X, Y, Z])

def compute_rigid_transform(A, B):
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am.T @ Bm
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    t = centroid_B - R_mat @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with Feature-based Odometry')
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path (e.g., /dev/video0 or 0 for default camera)')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera at {args.camera_device}")
        exit()
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    margin_width = 50
    cmap = matplotlib.colormaps['Spectral_r']
    
    # Camera intrinsics (Fantech C30)
    f_x = 301.24
    f_y = 301.24
    c_x = 320.0
    c_y = 240.0

    # Initialize ORB detector and BF matcher
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_frame = None
    prev_kp = None
    prev_des = None
    prev_depth = None

    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Depth inference and resize to original frame size (640x480)
        depth = depth_anything.infer_image(raw_frame, args.input_size)
        depth = cv2.resize(depth, (640, 480))

        # Extract ORB features from current RGB frame
        kp_curr, des_curr = orb.detectAndCompute(raw_frame, None)

        if prev_frame is not None and prev_des is not None and prev_kp is not None:
            # Match features to previous frame
            matches = bf.match(prev_des, des_curr)
            matches = sorted(matches, key=lambda x: x.distance)

            pts_3d_prev = []
            pts_3d_curr = []

            for m in matches:
                u1, v1 = map(int, prev_kp[m.queryIdx].pt)
                u2, v2 = map(int, kp_curr[m.trainIdx].pt)

                # Depth check, avoid zero or invalid depth
                if 0 <= v1 < prev_depth.shape[0] and 0 <= u1 < prev_depth.shape[1] and 0 <= v2 < depth.shape[0] and 0 <= u2 < depth.shape[1]:
                    d1 = prev_depth[v1, u1]
                    d2 = depth[v2, u2]
                    if d1 > 0 and d2 > 0:
                        p1 = backproject(u1, v1, d1, f_x, f_y, c_x, c_y)
                        p2 = backproject(u2, v2, d2, f_x, f_y, c_x, c_y)
                        pts_3d_prev.append(p1)
                        pts_3d_curr.append(p2)

            pts_3d_prev = np.array(pts_3d_prev)
            pts_3d_curr = np.array(pts_3d_curr)

            if len(pts_3d_prev) >= 4:
                T = compute_rigid_transform(pts_3d_prev, pts_3d_curr)
                
                # Extract translation
                tx, ty, tz = T[:3, 3]

                # Extract rotation matrix and convert to Euler angles
                rot = R.from_matrix(T[:3, :3])
                rx, ry, rz = rot.as_euler('xyz', degrees=True)  # degrees for readability
                
                print("Estimated Odometry Transform:\n", T)
                print(f"Translation: tx={tx:.3f}, ty={ty:.3f}, tz={tz:.3f}")
                print(f"Rotation (degrees): rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")
            else:
                print("Not enough 3D correspondences for odometry.")

        # Prepare depth visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        if args.grayscale:
            depth_vis = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Compose display frame
        if args.pred_only:
            display_frame = depth_vis
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            display_frame = cv2.hconcat([raw_frame, split_region, depth_vis])

        # Show frame with depth and RGB
        cv2.imshow('Depth Anything V2 - Live', display_frame)

        # Update previous frame data for next iteration
        prev_frame = raw_frame.copy()
        prev_kp = kp_curr
        prev_des = des_curr
        prev_depth = depth.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
