import numpy as np
import cv2

# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Inspect the loaded array
print("Type of data:", type(data))  # Should be <class 'numpy.ndarray'>
print("Shape of data:", data.shape)  # Check dimensions
print("Data type of elements:", data.dtype)  # Should be uint8 for RGB format
print(len(data))
# Extract the first frame
first_frame = data[120][1]
print("Shape of the first frame:", first_frame.shape)

# Display the RGB frame using OpenCV
cv2.imshow("RGB Frame", first_frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
