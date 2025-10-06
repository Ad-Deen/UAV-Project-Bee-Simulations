import numpy as np
import cv2

# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

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

# Threshold for re-detecting features
REDETECT_THRESHOLD = 50  # Minimum number of tracked features to trigger re-detection

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
        good_old = prev_points[status == 1]

        # Draw the tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            curr_frame = cv2.circle(curr_frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Overlay the mask on the current frame
        output = cv2.add(curr_frame, mask)

        # Update points for next iteration
        prev_points = good_new.reshape(-1, 1, 2)
    else:
        # If no points are left, re-detect features
        prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)

    # Re-detect features if too few remain
    if prev_points is None or len(prev_points) < REDETECT_THRESHOLD:
        new_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
        if new_points is not None:
            prev_points = new_points if prev_points is None else np.vstack((prev_points, new_points))

    # Display the output
    cv2.imshow("Feature Tracking", output if 'output' in locals() else curr_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    # Update the previous frame for the next iteration
    prev_gray = curr_gray.copy()

# Clean up windows
cv2.destroyAllWindows()
