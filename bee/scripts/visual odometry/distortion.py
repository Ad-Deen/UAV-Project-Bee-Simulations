import cv2
import numpy as np

# Callback for trackbars (does nothing, needed by OpenCV)
def nothing(x):
    pass

# Capture a single frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image from webcam.")
    exit()

# Resize for display
frame = cv2.resize(frame, (640, 480))

# Camera intrinsics (adjust to your specific camera if needed)
h, w = frame.shape[:2]
focal_length = w  # Approximate focal length in pixels
cx = w / 2
cy = h / 2
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Create OpenCV window and sliders
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)
cv2.createTrackbar("k1 x1000", "Undistorted", 1000, 2000, nothing)
cv2.createTrackbar("k2 x1000", "Undistorted", 1000, 2000, nothing)
cv2.createTrackbar("k3 x1000", "Undistorted", 1000, 2000, nothing)

print("Use sliders to tune distortion coefficients. Press 'q' to quit.")

while True:
    # Get values from sliders and map them to [-1.0, 1.0]
    k1 = (cv2.getTrackbarPos("k1 x1000", "Undistorted") - 1000) / 1000.0
    k2 = (cv2.getTrackbarPos("k2 x1000", "Undistorted") - 1000) / 1000.0
    k3 = (cv2.getTrackbarPos("k3 x1000", "Undistorted") - 1000) / 1000.0
    dist_coeffs = np.array([k1, k2, 0, 0, k3])  # Only using radial distortion

    # Compute undistortion map
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Combine and show both original and undistorted
    combined = np.hstack((frame, undistorted))
    cv2.imshow("Undistorted", combined)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
