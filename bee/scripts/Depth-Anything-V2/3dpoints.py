import numpy as np
import cv2

# Load and resize depth map
depth = np.load('my_depth.npy')  # Shape: (518, 518) or (480, 640)
if depth.shape != (480, 640):
    depth = cv2.resize(depth, (640, 480))
    np.save('my_depth_resized.npy', depth)
    print("Resized and saved depth map to my_depth_resized.npy")

# Intrinsics (from DFOV=106°, HFOV=93.48°, VFOV=77.04°)
f_x = 301.24
f_y = 301.24
c_x = 320.0
c_y = 240.0

# Create pixel coordinate grids
u, v = np.meshgrid(np.arange(640), np.arange(480))  # Shapes: (480, 640)

# Compute 3D coordinates
X = (u - c_x) * depth / f_x  # Shape: (480, 640)
Y = (v - c_y) * depth / f_y  # Shape: (480, 640)
Z = depth                    # Shape: (480, 640)

# Stack into a 3D point cloud
points_3d = np.stack([X, Y, Z], axis=-1)  # Shape: (480, 640, 3)
points_3d_flat = points_3d.reshape(-1, 3)  # Shape: (307200, 3)

# Save 3D points
np.save('points_3d.npy', points_3d_flat)
print("Saved 3D points to points_3d.npy")

# Optional: Visualize point cloud
try:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d_flat)
    o3d.visualization.draw_geometries([pcd])
except ImportError:
    print("Open3D not installed. Install with: pip install open3d")