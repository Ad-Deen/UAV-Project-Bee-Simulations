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

# Parameters for Shi-Tomasi Corner Detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get the first frame and convert it to grayscale
prev_frame = data[0][1]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Detect initial features to track
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)

# Iterate over frames to track features
for i in range(1, len(data)):
    # Get the current frame and convert to grayscale
    curr_frame = data[i][1]
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    # Select good points
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]

    # Draw the tracks
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        curr_frame = cv2.circle(curr_frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Overlay the mask on the current frame
    output = cv2.add(curr_frame, mask)

    # Display the output
    cv2.imshow("Feature Tracking", output)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    # Update previous frame and points for the next iteration
    prev_gray = curr_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

# Clean up windows
cv2.destroyAllWindows()
