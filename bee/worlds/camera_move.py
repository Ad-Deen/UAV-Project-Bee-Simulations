import math
import subprocess
import time

def set_camera_pose(position, orientation):
    """Send the camera pose to Ignition using a command."""
    command = (
        f"ign service -s /world/default/set_pose "
        f"--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean "
        f"--timeout 300 --req 'name: \"camera\", "
        f"position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, "
        f"orientation: {{x: {orientation[0]}, y: {orientation[1]}, z: {orientation[2]}, w: {orientation[3]}}}'"
    )
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            print(f"Error: {result.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr.decode()}")

def update_camera_position_and_orientation(radius, angular_step, total_steps, initial_position, initial_orientation):
    """Move the camera in a circular path and update its orientation to focus on the origin."""
    
    # Start with initial position and orientation
    current_position = initial_position
    current_orientation = initial_orientation

    # Convert angular step from degrees to radians
    angular_step_radians = math.radians(angular_step)

    # Move the camera in a circular path
    for step in range(total_steps):
        # Calculate new position on the circle
        angle = step * angular_step_radians
        
        # Compute the new camera position along the circle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = current_position[2]  # Keep the z-coordinate constant (unless you want to change altitude)

        # For orientation: Keep the camera pointing at the origin
        # Orientation quaternion to keep camera facing the origin
        # Assuming the camera rotates along the horizontal plane (no tilt)
        yaw = angle  # Set yaw to match the angle of the camera's position
        
        # Convert yaw to quaternion (assuming no pitch or roll)
        qw = math.cos(yaw / 2)
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2)

        # Update the camera pose
        set_camera_pose([x, y, z], [qx, qy, qz, qw])
        
        # Wait for the next interval (0.5 seconds)
        time.sleep(0.5)

if __name__ == "__main__":
    # Define the initial position and orientation of the camera
    initial_position = [4.0, 0.0, 1.0]
    initial_orientation = [0.0, 0.0, 0.001, -0.999]  # Initial quaternion

    # Parameters
    radius = 4.0  # Set the radius of the circular path
    angular_step = 0.1  # Angular step in degrees
    total_steps = 3600  # Total number of steps (for example, a full circle over 0.1 degrees)

    # Start moving the camera
    update_camera_position_and_orientation(radius, angular_step, total_steps, initial_position, initial_orientation)
