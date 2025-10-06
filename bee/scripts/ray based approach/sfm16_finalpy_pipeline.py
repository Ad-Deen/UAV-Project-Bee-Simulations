import numpy as np
import cv2
import math
import open3d as o3d
from pycpd import RigidRegistration
from sklearn.cluster import DBSCAN

#==============================================================================================================================
def plot_multiple_rays_with_points(origins, direction_vectors_list, points, ray_length=1.0):
    """
    Plots rays and 3D points in Open3D given multiple origins, corresponding sets of direction vectors,
    and a list of 3D points.

    Args:
        origins (list of np.array): A list of 3D origins, one for each set of direction vectors.
        direction_vectors_list (list of list of np.array): A list where each element is a list of 
            3D direction vectors corresponding to an origin.
        points (list of np.array): A list of 3D points to plot.
        ray_length (float): The length of each ray for visualization.
    """
    all_lines = []  # To store all line connections
    all_points = []  # To store all points
    spheres = []  # To store origin spheres
    offset = 0  # Index offset for connecting lines

    for origin, direction_vectors in zip(origins, direction_vectors_list):
        # Add the origin to points
        all_points.append(origin)

        # Create a sphere at the current origin
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # Adjust radius as needed
        sphere.translate(origin)  # Move sphere to origin
        sphere.paint_uniform_color([0, 1, 0])  # Set color to green
        spheres.append(sphere)

        # Process direction vectors for the current origin
        for i, direction in enumerate(direction_vectors):
            endpoint = origin + direction * ray_length  # Compute the endpoint
            all_points.append(endpoint)  # Add endpoint to points
            all_lines.append([offset, offset + i + 1])  # Connect origin to endpoint

        # Update offset for the next set of rays
        offset += len(direction_vectors) + 1

    # Convert to Open3D format for rays
    all_points = np.array(all_points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)

    # Optionally set colors for the rays
    colors = [[1, 0, 0] for _ in range(len(all_lines))]  # Red color for all rays
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create spheres for the 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0, 0, 1])  # Blue color for points

    # Visualize the rays, origins, and points
    o3d.visualization.draw_geometries([line_set, point_cloud] + spheres)

def compute_orientations(points, image_width=640, image_height=480, hfov=60, vfov=40):
    """
    Converts a list of 2D image points into orientation angles [rx, ry, rz].

    Parameters:
    points: list of lists, where each sublist contains a point [[x1, y1], [x2, y2], ..., [xn, yn]]
    image_width: int, width of the image in pixels (default: 640)
    image_height: int, height of the image in pixels (default: 480)
    hfov: float, horizontal field of view in degrees (default: 60)
    vfov: float, vertical field of view in degrees (default: 40)

    Returns:
    orientations: list of lists, where each sublist contains [rx, ry, rz]
    """
    orientations = []

    # Convert FOV to radians
    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    # Compute pixel-to-angle conversion factors
    pixels_per_degree_h = image_width / hfov
    pixels_per_degree_v = image_height / vfov

    for point in points:
        x, y = point[0]

        # Map x (horizontal) to rz (-30 to +30 degrees)
        rz_deg = (x - image_width / 2) / pixels_per_degree_h
        rz_rad = np.radians(rz_deg)

        # Map y (vertical) to ry (+20 to -20 degrees, inverted axis)
        ry_deg = -(y - image_height / 2) / pixels_per_degree_v
        ry_rad = np.radians(ry_deg)

        orientations.append([0, ry_rad, -rz_rad])

    return orientations


def orientations_to_directions(orientations,offset):
    """
    Converts a list of orientations (rx, ry, rz) in radians to direction vectors.

    Args:
        orientations (list of lists): A list where each entry is [rx, ry, rz] in radians.

    Returns:
        list of np.array: A list of 3D direction vectors corresponding to the orientations.
    """
    directions = []

    for orientation in orientations:
        rx, ry, rz = orientation
        rx += offset[0]
        ry += offset[1]
        rz += offset[2]

        # Compute rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        R_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combine rotations: R = Rz * Ry * Rx
        R = R_z @ R_y @ R_x

        # Apply rotation to the default forward vector [0, 0, 1]
        forward_vector = np.array([1, 0, 0])
        direction = R @ forward_vector

        directions.append(direction)

    return directions



def triangulate_point(ray1, ray2):
    """
    Computes the 3D point where two rays intersect or come closest to each other.
    
    Parameters:
    ray1 : tuple
        A tuple (origin1, direction1) for the first ray.
        - origin1: numpy array of shape (3,) representing the starting point of the ray.
        - direction1: numpy array of shape (3,) representing the direction vector of the ray.
    ray2 : tuple
        A tuple (origin2, direction2) for the second ray.
        - origin2: numpy array of shape (3,) representing the starting point of the ray.
        - direction2: numpy array of shape (3,) representing the direction vector of the ray.
        
    Returns:
    point : numpy array
        The 3D point where the rays intersect or are closest.
    """
    origin1, end1 = ray1
    origin2, end2 = ray2

    # Direction vectors of the rays
    direction1 = end1 - origin1
    direction2 = end2 - origin2

    # Normalize direction vectors
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)

    # Compute the cross-product of the directions
    cross_dir = np.cross(direction1, direction2)
    cross_dir_norm = np.linalg.norm(cross_dir)

    if cross_dir_norm < 1e-6:  # Rays are nearly parallel
        return (origin1 + origin2) / 2  # Midpoint of origins as a fallback

    # Matrix to solve for the closest points
    A = np.array([direction1, -direction2]).T
    b = origin2 - origin1

    # Solve for the scalars (t1, t2) along the ray directions
    t = np.linalg.lstsq(A, b, rcond=None)[0]

    # Closest points on each ray
    closest_point_ray1 = origin1 + t[0] * direction1
    closest_point_ray2 = origin2 + t[1] * direction2

    # Return the midpoint of the closest points as the triangulated point
    triangulated_point = (closest_point_ray1 + closest_point_ray2) / 2
    return triangulated_point

def compute_mean_and_distance(pcd_points, cam1_pos, cam2_pos):
    """
    Computes the mean position of point cloud points, the mean position of two cameras,
    and the 3D distance between the two means.

    Args:
        pcd_points (list of np.array): List of 3D points (point cloud data).
        cam1_pos (np.array): 3D position of the first camera.
        cam2_pos (np.array): 3D position of the second camera.

    Returns:
        float: The 3D distance between the mean position of the point cloud and the mean camera position.
    """
    # Calculate the mean position of the point cloud
    mean_pcd_pos = np.mean(pcd_points, axis=0)

    # Calculate the mean position of the two cameras
    mean_cam_pos = (cam1_pos + cam2_pos) / 2

    # Compute the 3D distance between the two mean positions
    distance = np.linalg.norm(mean_pcd_pos - mean_cam_pos)

    return distance


def align_and_combine_points(old_points, new_points):
    """
    Aligns two sets of 3D points by translating them to a common midpoint 
    between their mean positions and combines them. If old_points is empty,
    returns new_points directly.

    Args:
        old_points (np.ndarray): A (N, 3) array of 3D coordinates for the first set of points.
                                 Can be an empty list or array.
        new_points (np.ndarray): A (M, 3) array of 3D coordinates for the second set of points.

    Returns:
        combined_points (np.ndarray): A (N + M, 3) array of the combined translated points.
    """
    if len(old_points) == 0:
        # If old points are empty, return the new points directly
        return np.array(new_points)

    # Step 1: Calculate the mean position of the point sets
    old_mean = np.mean(old_points, axis=0)
    new_mean = np.mean(new_points, axis=0)

    # Step 2: Calculate the midpoint between the two mean positions
    midpoint = (old_mean + new_mean) / 2

    # Step 3: Translate both point sets to the common midpoint
    old_translated = old_points - old_mean + midpoint
    new_translated = new_points - new_mean + midpoint

    # Step 4: Combine the translated point sets
    combined_points = np.vstack([old_translated, new_translated])

    return combined_points

def align_point_clouds_with_cpd(old_points, new_points):
    """
    Aligns two point clouds using the CPD algorithm (rigid).
    
    Args:
        old_points (list or np.ndarray): Existing points (2D array or list of 3D points).
        new_points (list of np.ndarray or np.ndarray): List of 3D points for new points (can be 2D array).
        
    Returns:
        combined_points (list of np.ndarray): Combined old and aligned new points.
        aligned_points (np.ndarray): Aligned new points.
        transformation (dict): Transformation parameters (R, t, s).
    """
    # Convert new points from list of arrays to a single NumPy array if necessary
    new_points = np.asarray(new_points)
    # Ensure old_points is a NumPy array
    old_points = np.asarray(old_points)
    
    # If old points are empty, just use new points
    if old_points is None or old_points.size == 0:
        return list(new_points), new_points, {"R": np.eye(3), "t": np.zeros(3), "s": 1.0}
    
    # Ensure old_points is a NumPy array
    # old_points = np.asarray(old_points)
    
    # Check if old_points is in a list of arrays format (list of points as np.ndarrays)
    if isinstance(old_points, list):
        old_points = np.vstack(old_points)  # Convert the list to a 2D NumPy array

    # Set up CPD rigid registration
    registration = RigidRegistration(X=old_points, Y=new_points)
    
    # Perform registration
    aligned_points, (R, t, s) = registration.register()
    
    # Combine the old and aligned new points
    combined_points = list(old_points) + list(aligned_points)
    
    # Return results
    transformation = {"R": R, "t": t, "s": s}
    return combined_points, aligned_points, transformation

def transform_points_towards_new(all_3D, all_3D_new, interpolation_factor=0.5):
    """
    Transforms the points in all_3D towards the points in all_3D_new by the specified interpolation factor.
    
    Args:
        all_3D (list of np.ndarray): List of 3D points for the old set.
        all_3D_new (list of np.ndarray): List of 3D points for the new set.
        interpolation_factor (float): Fraction (0-1) to move all_3D points towards all_3D_new points.
        
    Returns:
        transformed_points (np.ndarray): Transformed all_3D points, moved towards all_3D_new.
    """
    # Convert lists to numpy arrays for easy manipulation
    all_3D = np.asarray(all_3D)
    all_3D_new = np.asarray(all_3D_new)
    
    # Compute the centroid (mean position) of each set of points
    centroid_all_3D = np.mean(all_3D, axis=0)
    centroid_all_3D_new = np.mean(all_3D_new, axis=0)
    
    # Compute the translation vector to move all_3D points towards all_3D_new points
    translation_vector = centroid_all_3D_new - centroid_all_3D
    
    # Interpolate the all_3D points towards the new points by the interpolation factor
    transformed_points = all_3D + interpolation_factor * translation_vector
    
    return transformed_points

def filter_3d_points(points, eps=0.1, min_samples=5):
    """
    Filters a list of 3D points using DBSCAN to remove outliers and unite close points.

    Parameters:
        points (list of np.ndarray): List of 3D points in the format [array([x, y, z]), ...].
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
        filtered_points (list of np.ndarray): Filtered list of 3D points.
    """
    # Convert the list of 3D points to a numpy array
    points_array = np.array(points)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points_array)

    # Extract clusters (ignore noise labeled as -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove the noise label

    filtered_points = []

    for label in unique_labels:
        # Get points in the current cluster
        cluster_points = points_array[labels == label]

        # Compute the centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)

        # Add the centroid to the filtered points
        filtered_points.append(centroid)

    # Convert the filtered points back to the original format
    filtered_points = [np.array(point) for point in filtered_points]

    return filtered_points
    
def fifo_check_and_reduce(points_list, buffer_size):
    """
    Maintains a fixed-size buffer of 3D points using FIFO (First-In-First-Out) logic.
    
    Parameters:
        points_list (list): The current list of 3D points (numpy arrays).
        buffer_size (int): The maximum number of points to retain in the buffer.

    Returns:
        list: The updated list of 3D points within the buffer size limit.
    """
    # Check if the list exceeds the buffer size
    if len(points_list) > buffer_size:
        excess_count = len(points_list) - buffer_size
        points_list = points_list[excess_count:]  # Remove the oldest elements
    
    return points_list
#==============================================================================================================================
# rolling_buffer = RollingListBuffer(max_size=20)
# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Parameters for Shi-Tomasi Corner Detection
feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=5, blockSize=5)

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
loc = "/media/deen/Extended-Linux/sfm_resources/camera_odom_data.npy"
frame_count = 0
data_to_save = []
all_3D = []
all_3D_old = []
aligned_points = []
frame = []
all_pos = []
features = []
vector = []
all_3D_accum = []
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
        rays1 = []
        rays2 = []
        all_3D_new = []
        
        pos1 = prev_pos
        point1 = prev_feat
        
        pos2 = data[i][0]
        point2 = prev_points
        all_pos.append(pos2)
        all_pos.append(pos1)
        frame.append(pos2[:3])
        frame.append(pos1[:3])
        # print([pos1[:3],pos2[:3]])
        # print(pos2[:3])

        # Compute orientations------------------------------------------------------------
        orientations1 = compute_orientations(point1)
        orientations2 = compute_orientations(point2)
        # print(orientations1)
        # Convert orientations to direction vectors---------------------------------------
        directions1 = orientations_to_directions(orientations1,[0,0,pos1[-1]])
        directions2 = orientations_to_directions(orientations2,[0,0,pos2[-1]])
        vector.append(directions1[::30])
        vector.append(directions2[::30])
        # for i, direction in enumerate(directions):
        #     print(f"Orientation {i+1}: {orientations1[i]}")
        #     print(f"Direction Vector: {direction}\n")
        # print(f' direct1 len : {len(directions1)} direct2 len : {len(directions2)}')
        # Plot the rays
        # for i in range(len(directions1[:5])):
        #     intersection = compute_ray_intersection(pos1[:3], directions1[i][:5], pos2[:3], directions2[i][:5])
        #     all_3D.append(intersection)
        # print(all_3D)
        for i in range(len(directions1)):
            endpoint1 = pos1[:3] + directions1[i] * 0.1  # Compute the endpoint1---------------------
            rays1.append((pos1[:3],endpoint1))
            
        for i in range(len(directions2)):
            endpoint2 = pos2[:3] + directions2[i] * 0.1  # Compute the endpoint2--------------------
            rays2.append((pos2[:3],endpoint2))
        
        # print(f'ray1 ------------- {ray1}')
        # print(f'ray2 ------------- {ray2}')
        # plot_multiple_rays_with_points([pos1[:3],pos2[:3]], [directions1[:5],directions2[:5]],all_3D)

        # # Compute rays
        # rays_frame1 = compute_rays(pos1, orientations1, length=1.0)
        # rays_frame2 = compute_rays(pos2, orientations2, length=1.0)

        # print(rays_frame1)

        for ray1, ray2 in zip(rays1, rays2):
            point = triangulate_point(ray1, ray2)
            all_3D_new.append(point)

        # distance = compute_mean_and_distance(all_3D, pos1[:3], pos2[:3])
        # print("3D Distance:", distance)
        # print(all_3D_new)
        
        # Align the example point clouds
        # all_3D_old, aligned_points, transform = align_point_clouds_with_cpd(all_3D_old[int(len(all_3D_old)/4):], all_3D_new)
        all_3D_old, aligned_points, transform = align_point_clouds_with_cpd(all_3D_old, all_3D_new)
        # all_3D_old = filter_3d_points(all_3D_old, eps=0.5, min_samples=5)
        all_3D_old = fifo_check_and_reduce(all_3D_old, 400)
        print(f'len of all 3D old {len(all_3D_old)}')
        # print(f'all_3D_old len: {len(all_3D_old)}')
        # print(all_3D_old)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        all_3D_accum.extend(all_3D_old)
        all_3D = transform_points_towards_new(all_3D_accum, all_3D_new, interpolation_factor=1.0)
        print(f'len of all 3D {len(all_3D)}')
        # all_3D.append(all_3D)

        # print(all_3D)
        # all_3D = align_and_combine_points(all_3D, all_3D_new)
        # plot_multiple_rays_with_points([pos1[:3],pos2[:3]], [directions1[:30],directions2[:30]],all_3D_old)
        # plot_multiple_rays_with_points(frame, vector,all_3D_old)
        # plot_multiple_rays_with_points(frame, vector,all_3D)
        # plot_oriented_rays(pos1[:3], orientations1)
        
       

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

# plot_points_and_rays_with_axes(all_3D,all_pos)
plot_multiple_rays_with_points(frame, vector,all_3D)
# Clean up windows
cv2.destroyAllWindows()

