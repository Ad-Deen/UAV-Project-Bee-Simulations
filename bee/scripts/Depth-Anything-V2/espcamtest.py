import cv2
import numpy as np
import torch
import requests
import open3d as o3d
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration ---
ESP32_CAM_URL = "http://192.168.0.109"
STREAM_URL = ESP32_CAM_URL + ":81/stream"
INPUT_RESOLUTION_INDEX = 6  # 6 = 640x480 VGA

# --- Camera Intrinsics (approximate) ---
f_x = 301.24
f_y = 301.24
c_x = 320.0
c_y = 240.0

# --- Set ESP32-CAM resolution ---
def set_resolution(index=6):
    try:
        requests.get(f"{ESP32_CAM_URL}/control?var=framesize&val={index}", timeout=2)
        print(f"Set resolution index: {index}")
    except:
        print("Failed to set resolution")

# --- Main ---
def main():
    set_resolution(INPUT_RESOLUTION_INDEX)

    # Load model
    model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load('Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
    model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ Failed to open ESP32 stream.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window('3D Point Cloud', 800, 600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True
    cmap = matplotlib.colormaps['Spectral_r']

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Infer depth
        depth = model.infer_image(frame, input_size=518)
        depth = cv2.resize(depth, (640, 480))

        # Normalize depth for visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_vis = cmap(depth_vis.astype(np.uint8))[:, :, :3]
        depth_vis = (depth_vis * 255).astype(np.uint8)[:, :, ::-1]  # BGR

        # Generate 3D points (downsampled)
        small = cv2.resize(depth, (100, 100))
        u, v = np.meshgrid(np.arange(100), np.arange(100))
        scale_x = 100 / 640
        scale_y = 100 / 480

        X = (u - c_x * scale_x) * small / (f_x * scale_x)
        Y = -(v - c_y * scale_y) * small / (f_y * scale_y)
        Z = small
        pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(pts)
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        cv2.imshow("ESP32-CAM + Depth Anything", depth_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
