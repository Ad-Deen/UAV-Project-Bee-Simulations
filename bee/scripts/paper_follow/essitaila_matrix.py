import numpy as np
import cv2
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as SciRot

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

###########################################################################################################################################
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
K = np.array([
        [554.38270568847656, 0, 320],
        [0, 554.38270568847656, 240],
        [0, 0, 1]
    ])  # Intrinsic matrix
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
    ################################################# Code here #####################################################
    
    if (i-1)%20 == 0:           #frame1 data
    # if i == 2:
        prev_pos= data[i][0]
        prev_feat = prev_points
        # print(prev_feat)

    if (i-19)%20 == 0:        # frame2 data
    # if i == 18:
        pos1 = prev_pos       # frame1 
        point1 = prev_feat    # frame1 feature x,y
        
        pos2 = data[i][0]
        point2 = prev_points  # frame2 feature x,


        # Ensure points are in the correct format
        # point1 = np.array(point1, dtype=np.float32).reshape(-1, 2)
        # point2 = np.array(point2, dtype=np.float32).reshape(-1, 2)

        # Check shapes
        # print("Point1 shape:", point1.shape)  # Should be (N, 2)
        # print("Point2 shape:", point2.shape)  # Should be (N, 2)

        if point1.shape == point2.shape:

            # Compute the fundamental matrix
            F, mask = cv2.findFundamentalMat(point1, point2, method=cv2.FM_RANSAC)

            # print("Fundamental Matrix:\n", F)

        # Compute the essential matrix
            E = K.T @ F @ K

        # print("Essential Matrix:\n", E)

            # Decompose the essential matrix
            _, R, t, _ = cv2.recoverPose(E, point1, point2, K)

            # print("Rotation Matrix:\n", R)
            # print("Translation Vector:\n", t)

            # Convert rotation matrix to Euler angles (in radians)
            rotation = SciRot.from_matrix(R)
            euler_angles = rotation.as_euler('xyz', degrees=False)  # Rotation order: x -> y -> z

            # Extract translation components
            tx, ty, tz = t.flatten()

            # Extract Euler angles (rx, ry, rz)
            rx, ry, rz = euler_angles

            # Combine into a single list
            result = [tx, ty, tz, rx, ry, rz]

            # Print the result
            print(f" result of eular form : {result}")
            print(f"position of frame2 : {pos2}")









    #################################################################################################################


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
# plot_points_and_rays_with_axes(all_3D,frame)
# Clean up windows
cv2.destroyAllWindows()


