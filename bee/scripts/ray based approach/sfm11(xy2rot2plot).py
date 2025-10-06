import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import numpy as np

def draw_ray(rx, ry, rz, length=1.0):
    """
    Draws a ray in 3D space from the origin (0, 0, 0) based on the direction angles (rx, ry, rz) in radians.
    
    Parameters:
    rx, ry, rz : float
        Direction angles (in radians) of the ray.
    length : float, optional
        Length of the ray. Default is 1.0.
    """
    # Calculate the direction components
    # Rotation around Z-axis (rz)
    x = math.cos(rz)
    y = math.sin(rz)
    
    # Tilt in YZ-plane (rx) and ZX-plane (ry)
    x_rotated = x * math.cos(rx) - y * math.sin(rx)  # Rotation in YZ-plane (tilt)
    y_rotated = y * math.cos(rx) + x * math.sin(rx)  # Rotation in YZ-plane (tilt)
    z = math.sin(ry)  # Tilt in ZX-plane (tilt up/down from Y-axis)

    # Scale the vector by the desired length
    ray_end = [length * x_rotated, length * y_rotated, length * z]
    
    return ray_end

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
                  rx is fixed to 1.5078, ry and rz are in radians
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

def plot_multiple_rays(frame_pos,orientations, length=1.0):
    """
    Plots multiple rays with random colors in 3D space.
    
    Parameters:
    orientations : list of lists
        Each list contains [rx, ry, rz] in radians for a ray.
    length : float, optional
        Length of each ray. Default is 1.0.
    """
    txi , tyi , tzi , rxi , ryi , rzi = frame_pos
    print(frame_pos)
    # rzi=rzi+1.5703
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for orientation in orientations:
        rx, ry, rz = orientation
        rx = rx+ rxi
        ry = ry+ ryi
        rz = rz+ rzi 
        # print(orientation)
        # Get the ray end point for the current orientation
        ray_end = draw_ray(rx, ry, rz, length)
        # print(ray_end)
        
        # Generate a random color for the ray
        color = (random.random(), random.random(), random.random())  # Random RGB color
        
        # Draw the ray on the plot
        ax.quiver(txi, tyi, tzi, ray_end[0], ray_end[1], ray_end[2], color=color, linewidth=0.5)

    # Set plot limits for better visualization
    ax.set_xlim([0, length])
    ax.set_ylim([0, length])
    ax.set_zlim([0, length])

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

# Example usage: List of orientations [rx, ry, rz] in radians
# orientations = [
#     [0, 0.3490658503988659, 0.5235987755982988], 
#      [0, 0.3490658503988659, -0.5235987755982988], 
#      [0, -0.3490658503988659, 0.5235987755982988], 
#      [0, -0.3490658503988659, -0.5235987755982988]
# ]
loaded_data = np.load("bee/scripts/feature_data.npy", allow_pickle=True)

# Access the first two frames (assuming you saved data for frames 32 and 50)
frame1_data = loaded_data[0]
frame2_data = loaded_data[1]

# Extract image, position, and feature points for each frame
img1 = frame1_data['image']
pos1 = frame1_data['position']
points1 = frame1_data['feature_points']

# print(f"frame 1 features={points1}")

img2 = frame2_data['image']
pos2 = frame2_data['position']
points2 = frame2_data['feature_points']
# Example usage
# points = [[[0, 0]], [[640, 0]], [[0, 480]],[[640,480]]]
orientations1 = compute_orientations(points1)
orientations2 = compute_orientations(points2)
# frame_pos = [ 1.87464472, -0.28813097 , 0.        ,  0. ,         0.     ,    -0.3471698 ]
# Plot the rays
plot_multiple_rays(pos2,orientations2, length=5.0)
