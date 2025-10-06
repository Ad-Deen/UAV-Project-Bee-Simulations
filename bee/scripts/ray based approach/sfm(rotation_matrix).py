import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_rotation_matrix(roll, pitch, yaw):
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

def create_arrow(position, rotation_matrix, length=1.0, radius=0.05):
    """
    Create an arrow in Open3D to visualize orientation.
    
    Parameters:
        position (np.array): 3D coordinates where the arrow starts.
        rotation_matrix (np.array): 3x3 rotation matrix to orient the arrow.
        length (float): Length of the arrow.
        radius (float): Radius of the arrow shaft.
    
    Returns:
        arrow (o3d.geometry.TriangleMesh): Arrow mesh in Open3D.
    """
    # Create an arrow mesh
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=2 * radius,
        cylinder_height=0.8 * length,
        cone_height=0.2 * length
    )
    
    # Apply the rotation matrix
    arrow.rotate(rotation_matrix, center=(0,0,0))
    
    # Translate the arrow to the desired position
    arrow.translate(position)
    
    # Set the color for visibility
    arrow.paint_uniform_color([1.0, 0.0, 0.0])  # Red arrow
    
    return arrow

# Example usage
if __name__ == "__main__":
    # Define the arrow's position
    position = np.array([0, 0, 0.5])

    # Define orientation using Euler angles (in radians)
    rx, ry, rz = np.radians([0, 0, 130])  # Example angles
    rotation_matrix = euler_to_rotation_matrix(rx, ry, rz)

    # Create the arrow
    arrow = create_arrow(position, rotation_matrix, length=1.0, radius=0.01)

    # Create a coordinate frame for reference (showing X, Y, Z axes)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    

    # Visualize using Open3D
    o3d.visualization.draw_geometries([arrow, coordinate_frame])
