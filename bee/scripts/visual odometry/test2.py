import cv2
import numpy as np

def get_strong_edge_features(gray, max_features=50):
    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Use goodFeaturesToTrack on edge regions to get corners at sharp transitions
    corners = cv2.goodFeaturesToTrack(
        edges,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=False
    )

    if corners is not None:
        return np.int0(corners)
    return []

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    frame_count = 0
    feature_points = []

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Refresh features every 10 frames
        if frame_count % 10 == 0:
            feature_points = get_strong_edge_features(gray, max_features=50)

        # Draw features
        display_frame = frame.copy()
        if feature_points is not None:
            for pt in feature_points:
                x, y = pt.ravel()
                cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)

        cv2.imshow("Edge-based Features", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
