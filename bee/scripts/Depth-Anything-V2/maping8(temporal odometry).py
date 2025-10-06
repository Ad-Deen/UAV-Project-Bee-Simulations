import argparse
import cv2
import matplotlib
import numpy as np
import torch
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.spatial.transform import Rotation as R


# Global state variables
frame_count = 0
features = None
prev_gray = None
benchmark = None
current = None
FEATURE_SHUFFLE_FREQ = 10


def feature_track(rgb_image, FEATURE_SHUFFLE_FREQ=7):
    global prev_gray, features, benchmark_features, frame_count

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    output_frame = rgb_image.copy()

    # Reinitialize features every N frames or if features are gone
    if frame_count % FEATURE_SHUFFLE_FREQ == 0 or features is None or len(features) == 0:
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=20, qualityLevel=0.3, minDistance=10)
        
        if features is not None:
            benchmark_features = features.copy()  # Save for benchmark tracking
            prev_gray = gray.copy()
        else:
            benchmark_features = None  # No features found

    else:
        # Track features using optical flow
        if prev_gray is not None and features is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, features, None)

            if next_pts is not None and status is not None:
                good_new = next_pts[status.flatten() == 1]
                good_old = benchmark_features[status.flatten() == 1]

                # Draw tracked features
                for pt in good_new:
                    x, y = pt.ravel()
                    cv2.circle(output_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Update for next tracking
                features = good_new.reshape(-1, 1, 2)
                benchmark_features = good_old.reshape(-1, 1, 2)
                prev_gray = gray.copy()
            else:
                features = None
                benchmark_features = None

    frame_count += 1

    # Format output arrays
    if benchmark_features is not None and features is not None:
        # int_pts = np.round(features.reshape(-1, 2)).astype(int)
        return output_frame, benchmark_features.reshape(-1, 2), np.round(features.reshape(-1, 2)).astype(int)
    else:
        return output_frame, None, None

def pixel_to_3d_points(pixel_coords, depth_map, f_x, f_y, c_x, c_y):
    """
    Convert 2D pixel coordinates to 3D points using depth and camera intrinsics.

    Args:
        pixel_coords (ndarray): (N, 2) array of [x, y] pixel coordinates (float or int).
        depth_map (ndarray): (H, W) depth map with depth at each pixel.
        f_x, f_y: focal lengths.
        c_x, c_y: principal point offsets.

    Returns:
        points_3d (ndarray): (N, 3) array of corresponding 3D points [X, Y, Z].
    """

    # Ensure pixel coordinates are rounded and cast to integers for indexing
    pixel_coords = np.round(pixel_coords).astype(int)

    # Clip pixel coords to image bounds
    h, w = depth_map.shape
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, w - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, h - 1)

    # Extract u and v
    u = pixel_coords[:, 0]  # x
    v = pixel_coords[:, 1]  # y

    # Get depth values at those pixel locations
    Z = depth_map[v, u]  # Note: depth map indexing is (row=v, col=u)

    # Avoid divide-by-zero or invalid depth
    valid = Z > 0
    Z = Z[valid]
    u = u[valid]
    v = v[valid]

    # Back-project to 3D space
    X = (u - c_x) * Z / f_x
    Y = (v - c_y) * Z / f_y

    # Stack into (N, 3) array
    points_3d = np.stack([X, -Y, -Z], axis=-1)

    return points_3d


def estimate_odometry(ref_points, matched_points):
    """
    Estimate 6-DoF odometry from two matched 3D point sets.

    Args:
        ref_points (ndarray): (N, 3) reference 3D points.
        matched_points (ndarray): (N, 3) current 3D points corresponding to ref_points.

    Returns:
        odometry (list): [x, y, z, roll, pitch, yaw]
    """

    assert ref_points.shape == matched_points.shape and ref_points.shape[1] == 3, \
        "Point arrays must be shape (N, 3) and match in size."

    # Step 1: Compute centroids
    centroid_ref = np.mean(ref_points, axis=0)
    centroid_matched = np.mean(matched_points, axis=0)

    # Step 2: Center the points
    ref_centered = ref_points - centroid_ref
    matched_centered = matched_points - centroid_matched

    # Step 3: Compute covariance and SVD
    H = ref_centered.T @ matched_centered
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T

    # Step 4: Fix improper rotation
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    # Step 5: Translation vector
    t = centroid_matched - R_mat @ centroid_ref

    # Step 6: Extract Euler angles (roll, pitch, yaw) from rotation matrix
    r = R.from_matrix(R_mat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    # Step 7: Return as odometry list
    odometry = [round(val, 3) for val in [t[0], t[1], t[2], roll, pitch, yaw]]
    return odometry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Camera Livestream')
    parser.add_argument('--camera-device', type=str, default='/dev/video0')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
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

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    margin_width = 50
    cmap = matplotlib.colormaps['Spectral_r']

    f_x = 301.24
    f_y = 301.24
    c_x = 320.0
    c_y = 240.0

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    # 3-frame moving average buffer on linear depth
    depth_buffer = []
    buffer_size = 3

    # Resize to small size before storing in buffer
    small_size = (int(640/3), int(480/3))
    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Get linear depth prediction
        depth = depth_anything.infer_image(raw_frame, args.input_size)
        depth = cv2.resize(depth, (640, 480))
        
        
        depth_small_raw = cv2.resize(depth, small_size, interpolation=cv2.INTER_LINEAR)
        resized_raw = cv2.resize(raw_frame, small_size, interpolation=cv2.INTER_LINEAR)

        # Store raw depth before inverse
        depth_buffer.append(depth_small_raw.copy())
        if len(depth_buffer) > buffer_size:
            depth_buffer.pop(0)

        # Apply moving average to raw depth
        if len(depth_buffer) == buffer_size:
            depth_small_avg = np.mean(depth_buffer, axis=0)
        else:
            depth_small_avg = depth_small_raw

        # Inverse depth after averaging
        depth_small = np.round(1.0 / (depth_small_avg + 1e-6), 5)
        # print(depth_small.shape)

        #Function takes frame1 and frame4 to output 2 indexes array 
        #(matched index of frame 1 and frame 4) of the depth features matched
        resized_RGB , benchmark , current = feature_track(rgb_image=resized_raw, FEATURE_SHUFFLE_FREQ= FEATURE_SHUFFLE_FREQ)
        # print(current)
        # current_pnts = np.round(features).astype(int)
        # Prepare intrinsics
        u_small, v_small = np.meshgrid(np.arange(small_size[0]), np.arange(small_size[1]))
        scale_x = small_size[0] / 640
        scale_y = small_size[1] / 480

        f_x_small = f_x * scale_x
        f_y_small = f_y * scale_y
        c_x_small = c_x * scale_x
        c_y_small = c_y * scale_y
        matched_points = pixel_to_3d_points(current, depth_small, f_x_small, f_y_small, c_x_small, c_y_small)
        ref_points = pixel_to_3d_points(benchmark, depth_small, f_x_small, f_y_small, c_x_small, c_y_small)
        # odom = estimate_odometry(ref_points, matched_points)
        # print(odom)
        # Generate point cloud
        X = np.round((u_small - c_x_small) * depth_small / f_x_small, 5)
        Y = np.round(-(v_small - c_y_small) * depth_small / f_y_small, 5)
        Z = np.round(-depth_small, 5)

        points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # pcd.points = o3d.utility.Vector3dVector(ref_points)
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)


        vis.poll_events()
        vis.update_renderer()
        cv2.imshow('2D Array Visualization', resized_RGB)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
