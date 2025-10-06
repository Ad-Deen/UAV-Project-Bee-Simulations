import numpy as np
import cv2
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R
#==============================================================================================================================
def plot_points_and_rays_with_axes(points, data):
    """
    Combines and plots 3D triangulated points, rays, and origin axes in a single visualization using Open3D.
    
    Parameters:
    points : list of numpy arrays
        List of 3D points to be plotted.
    data : list of numpy arrays
        Each array contains [tx, ty, tz, Rx, Ry, Rz], where:
        - (tx, ty, tz) represent the ray origin.
        - (Rx, Ry, Rz) represent the orientation in radians.
    """
    geometries = []

    # Add origin axes
    axis_length = 1.0  # Length of the axes
    axis_radius = 0.01  # Thickness of the axes
    #By default rx,ry,rx (xyz convention) is directed towerds Z direction
    # X-axis (red) 
    x_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=axis_radius, height=axis_length)
    x_axis.translate([0, 0, 0])
    x_axis.rotate(R.from_euler('xyz', [0, np.pi/2, 0]).as_matrix(), center=[0, 0, 0])
    x_axis.paint_uniform_color([1, 0, 0])
    geometries.append(x_axis)

    # Y-axis (green)
    y_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=axis_radius, height=axis_length)
    y_axis.translate([0, 0, 0])
    x_axis.rotate(R.from_euler('xyz', [0, 0, 0]).as_matrix(), center=[0, 0, 0])
    y_axis.paint_uniform_color([0, 1, 0])
    geometries.append(y_axis)

    # Z-axis (blue)
    z_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=axis_radius, height=axis_length)
    z_axis.translate([0, 0, 0 ])
    z_axis.rotate(R.from_euler('xyz', [np.pi/2, 0, 0]).as_matrix(), center=[0, 0, 0])
    z_axis.paint_uniform_color([0, 0, 1])
    geometries.append(z_axis)

    # Plot triangulated points
    if points:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in points])  # Red color for points
        geometries.append(point_cloud)

    # Plot rays
    for entry in data:
        origin = entry[:3]
        euler_angles = entry[3:]  # (Rx, Ry, Rz)

        # Create an arrow to represent the ray
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.8,
            cone_height=0.2
        )

        arrow.rotate(R.from_euler('xyz', [np.pi/2+euler_angles[0],np.pi/2+euler_angles[1],euler_angles[2]+np.pi/2]).as_matrix(), center=[0, 0, 0])

        # Translate the arrow to the origin
        arrow.translate(origin)

        # Set the arrow color
        arrow.paint_uniform_color([0, 0, 1])  # Blue color for arrows
        geometries.append(arrow)

    # Visualize combined geometries
    o3d.visualization.draw_geometries(geometries, window_name="3D Points, Rays, and Axes")


def euler_to_rotation_matrix(data):
    """
    Converts Euler angles (in radians) to a rotation matrix with the 'xyz' order.
    The rotation order is: X → Y → Z.

    Parameters:
    roll (float): Rotation around the X-axis (in radians).
    pitch (float): Rotation around the Y-axis (in radians).
    yaw (float): Rotation around the Z-axis (in radians).

    Returns:
    np.ndarray: The 3x3 rotation matrix.
    """
    roll, pitch, yaw = data
    # Rotation about X-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation about Y-axis (pitch)
    pitch = pitch + np.pi/2
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation about Z-axis (yaw)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Final rotation matrix (R = Rz * Ry * Rx)
    R = R_z @ R_y @ R_x
    return R

def triangulate_point(x1, x2, P1, P2):
    """
    Triangulates a 3D point from two 2D projections.
    x1, x2: 2D points in image 1 and image 2 (each is a 2D vector)
    P1, P2: Camera projection matrices for the two cameras (3x4 matrices)
    
    Returns: 3D point X (in Euclidean coordinates)
    """
    # Set up the linear system
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    
    # Solve using Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    
    # Convert to Euclidean coordinates
    X = X_homogeneous[:3] / X_homogeneous[3]
    return X
#==============================================================================================================================
# rolling_buffer = RollingListBuffer(max_size=20)
# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Parameters for Shi-Tomasi Corner Detection
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=5, blockSize=5)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get the first frame and convert it to grayscale
prev_frame = data[0][1]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Detect initial features to track
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Threshold for re-detecting features
REDETECT_THRESHOLD = 500  # Minimum number of tracked features to trigger re-detection

# Frame counter for feature removal
loc = '/media/deen/Extended-Linux/sfm_resources/'
frame_count = 0
data_to_save = []
all_3D = []
frame = []
features = []
# Iterate over frames to track features
for i in range(1, len(data)):
    # Get the current frame and convert to grayscale
    curr_frame = data[i][1]
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if prev_points is not None and len(prev_points) > 0:
        # Calculate optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

        # Select good points
        good_new = curr_points[status == 1]

        # Draw only the points
        for new in good_new:
            a, b = new.ravel()
            curr_frame = cv2.circle(curr_frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        # Update points for next iteration
        prev_points = good_new.reshape(-1, 1, 2)
        
    else:
        # If no points are left, re-detect features
        prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
        # print(len(prev_points))
    #==============================frame code========================================================================
    if (i-1)%20 == 0:
    # if i == 2:
        prev_pos= data[i][0]
        prev_feat = prev_points

    if (i-19)%20 == 0:
    # if i == 18:
        pos1 = prev_pos
        point1 = prev_feat
        
        pos2 = data[i][0]
        point2 = prev_points
        frame.append(pos2)
        frame.append(pos1)
        # print(point1[0][0])
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(point2[0][0])
        # Known camera parameters
        K = np.array([
            [554.38270568847656, 0, 320],
            [0, 554.38270568847656, 240],
            [0, 0, 1]
        ])  # Intrinsic matrix
        for i in range(0,len(point1)-1):
            # Extrinsic parameters for camera 1 (identity, origin)
            R1 = euler_to_rotation_matrix(pos1[3:])  # No rotation
            t1 = np.array([pos1[:3]]).T  # Camera 1 at origin

            # Extrinsic parameters for camera 2 (example: translate along X-axis)
            R2 = euler_to_rotation_matrix(pos2[3:])  # No rotation
            t2 = np.array([pos1[:3]]).T  # Camera 2 translated 1 unit along X-axis
            
            # Compute projection matrices
            P1 = K @ np.hstack((R1, t1))
            P2 = K @ np.hstack((R2, t2))
            # print(len(point1))
            # print(len(point2))
            
            # 2D points in image coordinates
            x1 = np.array(point1[i][0])  # Example point in Image 1
            x2 = np.array(point2[i][0])  # Example point in Image 2
            # Triangulate the 3D point
            X = triangulate_point(x1, x2, P1, P2)
            
            all_3D.append(X)
        # plot_points_and_rays_with_axes(all_3D,frame)
        print(len(all_3D))

    
    #================================================================================================================
    # Remove old features every 20 frames
    frame_count += 1
    if frame_count % 20 == 0:
        prev_points = None
    
    # Display the output (only points)
    cv2.imshow("Feature Tracking", curr_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    # Update the previous frame for the next iteration
    prev_gray = curr_gray.copy()
plot_points_and_rays_with_axes(all_3D,frame)
# Clean up windows
cv2.destroyAllWindows()

