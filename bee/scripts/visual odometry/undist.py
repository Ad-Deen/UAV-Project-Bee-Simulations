import cv2
import numpy as np

# Camera intrinsics (adjust if known)
frame_width = 640
frame_height = 480
focal_length = frame_width  # Approximate
cx = frame_width / 2
cy = frame_height / 2
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Distortion coefficients from your tuning
dist_coeffs = np.array([-0.904, 1.0, 0, 0, -0.193])  # k1, k2, p1, p2, k3

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Get one frame to calculate the undistortion map
ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam.")
    cap.release()
    exit()

h, w = frame.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)

print("Running undistortion. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply undistortion
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Show result
    cv2.imshow("Undistorted Live Feed", undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
