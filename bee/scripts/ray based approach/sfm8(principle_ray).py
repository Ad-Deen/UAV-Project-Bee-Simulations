import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Frame 1 data
img1_pos = np.array([1.87464472, -0.28813097, 0.0])
img1_rotation = -0.3471698

# Frame 2 data
img2_pos = np.array([3.28395308, -1.14392845, 0.0])
img2_rotation = -0.74914014

# Function to calculate direction vector from rotation angle
def calculate_direction(rotation_angle):
    direction = np.array([np.cos(rotation_angle), np.sin(rotation_angle), 0])
    return direction

# Function to plot a 3D ray
def plot_ray(ax, origin, direction, color='blue', length=10):
    """
    Plots a 3D ray from the given origin in the specified direction.

    Parameters:
    ax: matplotlib 3D axes object
    origin: numpy array, origin of the ray [x, y, z]
    direction: numpy array, direction vector [dx, dy, dz]
    color: string, color of the ray (default: 'blue')
    length: float, length of the ray (default: 10)
    """
    x, y, z = origin
    dx, dy, dz = direction
    x1, y1, z1 = x + dx * length, y + dy * length, z + dz * length
    ax.plot([x, x1], [y, y1], [z, z1], color=color)

# Function to plot a 3D point
def plot_point(ax, point, color='red'):
    """
    Plots a 3D point.

    Parameters:
    ax: matplotlib 3D axes object
    point: numpy array, coordinates of the point [x, y, z]
    color: string, color of the point (default: 'red')
    """
    x, y, z = point
    ax.scatter(x, y, z, color=color, s=50)  # Use ax.scatter for 3D points

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the rays and points
plot_ray(ax, img1_pos, calculate_direction(img1_rotation), 'red')
plot_ray(ax, img2_pos, calculate_direction(img2_rotation), 'blue')
plot_point(ax, img1_pos, 'red')
plot_point(ax, img2_pos, 'blue')

# Set plot limits and labels
ax.set_xlim([-5, 10])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
