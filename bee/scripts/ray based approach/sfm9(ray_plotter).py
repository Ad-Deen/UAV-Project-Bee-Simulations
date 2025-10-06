import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

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
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for orientation in orientations:
        rx, ry, rz = orientation
        rx = rx+ rxi
        ry = ry+ ryi
        rz = rz+ rzi 
        # Get the ray end point for the current orientation
        ray_end = draw_ray(rx, ry, rz, length)
        # print(ray_end)
        
        # Generate a random color for the ray
        color = (random.random(), random.random(), random.random())  # Random RGB color
        
        # Draw the ray on the plot
        ax.quiver(txi, tyi, tzi, ray_end[0], ray_end[1], ray_end[2], color=color, linewidth=2)

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
orientations = [
    [math.radians(0), math.radians(20), math.radians(30)],
    [math.radians(0), math.radians(20), math.radians(-30)],
    [math.radians(0), math.radians(-20), math.radians(30)],
    [math.radians(0), math.radians(-20), math.radians(-30)]
]
frame_pos = [ 1.87464472, -0.28813097 , 0.        ,  0. ,         0.     ,    -0.3471698+1.5703 ]
# Plot the rays
plot_multiple_rays(frame_pos,orientations, length=5.0)
