import numpy as np
import math
import matplotlib.pyplot as plt
import open3d as o3d

def draw_ray(rx, ry, rz, origin, length=1.0):
    """
    Returns the ray equation components (start and end points in 3D space).

    Parameters:
    rx, ry, rz : float
        Direction angles (in radians) of the ray.
    origin : list or numpy array
        Origin of the ray [x, y, z].
    length : float, optional
        Length of the ray. Default is 1.0.

    Returns:
    ray : tuple
        (origin, end_point), where end_point is the calculated endpoint of the ray.
    """
    # Calculate the direction components
    x = math.cos(rz)
    y = math.sin(rz)
    x_rotated = x * math.cos(rx) - y * math.sin(rx)  # Tilt in YZ-plane
    y_rotated = y * math.cos(rx) + x * math.sin(rx)  # Tilt in YZ-plane
    z = math.sin(ry)  # Tilt in ZX-plane

    # Scale the vector by the desired length
    direction = np.array([x_rotated, y_rotated, z]) * length
    end_point = origin + direction

    return (origin, end_point)

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

def compute_rays(frame_pos, orientations, length=5.0):
    """
    Computes ray equations for all orientations in a frame.

    Parameters:
    frame_pos : list
        Position of the frame [tx, ty, tz, rx, ry, rz].
    orientations : list
        List of [rx, ry, rz] orientations in radians.
    length : float
        Length of the ray.

    Returns:
    rays : list
        List of ray equations [(origin, end_point)] for the frame.
    """
    tx, ty, tz, rxi, ryi, rzi = frame_pos
    origin = np.array([tx, ty, tz])
    rays = []

    for orientation in orientations:
        rx, ry, rz = orientation
        rx += rxi
        ry += ryi
        rz += rzi
        rays.append(draw_ray(rx, ry, rz, origin, length))

    return rays


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


def plot_triangulated_points_open3d(points):
    """
    Plots the 3D triangulated points using Open3D.
    
    Parameters:
    points : list of numpy arrays
        List of 3D points to be plotted.
    """
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Convert the list of points to an Open3D-compatible format
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

    # Optionally, set colors for the points
    colors = [[1, 0, 0] for _ in points]  # Red color for all points
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], window_name="3D Triangulated Points")

# Load data
loaded_data = np.load("bee/scripts/feature_data.npy", allow_pickle=True)
# print(loaded_data)

# Access the first two frames
frame1_data = loaded_data[0]
frame2_data = loaded_data[1]

# Extract relevant data
pos1 = frame1_data['position']
points1 = frame1_data['feature_points']
# pos1[-1] = pos1[-1] - 1.5703  # Adjust rotation

pos2 = frame2_data['position']
points2 = frame2_data['feature_points']
# pos2[-1] = pos2[-1] - 1.5703  # Adjust rotation

# Compute orientations
orientations1 = compute_orientations(points1)
orientations2 = compute_orientations(points2)

# Compute rays
rays_frame1 = compute_rays(pos1, orientations1, length=5.0)
rays_frame2 = compute_rays(pos2, orientations2, length=5.0)

# Output the rays
# print("Rays for Frame 1:")
# for ray in rays_frame1:
#     print(ray)

# print("\nRays for Frame 2:")
# for ray in rays_frame2:
#     print(ray)


triangulated_points = []
for ray1, ray2 in zip(rays_frame1, rays_frame2):
    point = triangulate_point(ray1, ray2)
    triangulated_points.append(point)

# # Output the triangulated points
# print("Triangulated Points:")
# for i, point in enumerate(triangulated_points):
#     print(f"Point {i}: {point}")

# Plot Rays and Triangulated Points
plot_triangulated_points_open3d(triangulated_points)