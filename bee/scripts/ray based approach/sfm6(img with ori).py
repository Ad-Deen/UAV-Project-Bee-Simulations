import numpy as np
import cv2

# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

def display_frame(frame, pose):
    # Extract translation and rotation (unpack only 3 elements)
    translation = pose[:3]
    rotation = pose[3:]
    
    # Create a text string with the information
    text = f"Position: ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})\n"
    text += f"Rotation: ({rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f})"

    # Put the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(30)

# Iterate over the frames and display them
for i in range(len(data)):
    frame = data[i][1]
    pose = data[i][0]
    display_frame(frame, pose)
    print(pose)

# Close all windows
cv2.destroyAllWindows()