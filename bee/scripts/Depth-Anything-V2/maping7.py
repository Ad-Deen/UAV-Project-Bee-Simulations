import argparse
import cv2
import matplotlib
import numpy as np
import torch
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
# Global state variables
frame_count = 0
features = None
prev_gray = None

def feature_track(rgb_image, FEATURE_SHUFFLE_FREQ=7):
    global prev_gray, features, frame_count

    # Convert current frame to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    output_frame = rgb_image.copy()

    # Every Nth frame or if no features exist: extract new features
    if frame_count % FEATURE_SHUFFLE_FREQ == 0 or features is None or len(features) == 0:
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=20, qualityLevel=0.1, minDistance=10)
        prev_gray = gray.copy()

    else:
        # Use optical flow to track features
        if prev_gray is not None and features is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, features, None)

            # Ensure tracking succeeded
            if next_pts is not None and status is not None:
                good_new = next_pts[status == 1]
                good_old = features[status == 1]

                # Draw tracked points
                for pt in good_new:
                    x, y = pt.ravel()
                    cv2.circle(output_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Update features and previous gray frame
                features = good_new.reshape(-1, 1, 2)
                prev_gray = gray.copy()
            else:
                features = None  # Trigger re-extraction next frame

    frame_count += 1
    return output_frame


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
        
        # Prepare intrinsics
        u_small, v_small = np.meshgrid(np.arange(small_size[0]), np.arange(small_size[1]))
        scale_x = small_size[0] / 640
        scale_y = small_size[1] / 480

        f_x_small = f_x * scale_x
        f_y_small = f_y * scale_y
        c_x_small = c_x * scale_x
        c_y_small = c_y * scale_y

        # Generate point cloud
        X = np.round((u_small - c_x_small) * depth_small / f_x_small, 5)
        Y = np.round(-(v_small - c_y_small) * depth_small / f_y_small, 5)
        Z = np.round(-depth_small, 5)

        points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(points_3d)
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)


        vis.poll_events()
        vis.update_renderer()
        cv2.imshow('2D Array Visualization', feature_track(rgb_image=resized_raw))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
