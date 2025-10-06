import cv2
import numpy as np

def get_strong_edge_features(gray, max_features=50):
    edges = cv2.Canny(gray, 100, 200)
    corners = cv2.goodFeaturesToTrack(
        edges,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=False
    )
    return corners if corners is not None else []

def estimate_motion(good_old, good_new, image_shape):
    center = np.array([image_shape[1] / 2, image_shape[0] / 2])

    # Compute flow vectors
    flow = good_new - good_old
    avg_flow = np.mean(flow, axis=0)  # [tx, ty]
    tx, ty = avg_flow

    # Approximate tz (radial motion)
    old_dists = np.linalg.norm(good_old - center, axis=1)
    new_dists = np.linalg.norm(good_new - center, axis=1)
    tz = np.mean(new_dists - old_dists)

    # Approximate yaw (left vs right difference in horizontal flow)
    midpoint_x = center[0]
    left_flows = flow[good_old[:, 0] < midpoint_x][:, 0]
    right_flows = flow[good_old[:, 0] >= midpoint_x][:, 0]
    yaw = np.mean(right_flows) - np.mean(left_flows)

    return tx, ty, tz, yaw

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    frame_count = 0
    frames_per_track = 10
    tracking_frames_left = 0
    prev_gray = None
    feature_points = None

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Refresh features every 10 frames
        if tracking_frames_left == 0:
            feature_points = get_strong_edge_features(gray, max_features=50)
            prev_gray = gray.copy()
            tracking_frames_left = frames_per_track

        if feature_points is not None and len(feature_points) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, feature_points, None, **lk_params)
            good_new = next_pts[status == 1]
            good_old = feature_points[status == 1]

            # Draw optical flow
            for new, old in zip(good_new, good_old):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                cv2.circle(frame, (int(x_new), int(y_new)), 3, (0, 255, 0), -1)
                cv2.line(frame, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (255, 0, 0), 1)

            # Estimate motion
            tx, ty, tz, yaw = estimate_motion(good_old, good_new, gray.shape)

            # Display estimated motion
            cv2.putText(frame, f"tx: {tx:.2f} px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"ty: {ty:.2f} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"tz (radial): {tz:.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"yaw (px diff): {yaw:.2f} px", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Prepare for next frame
            feature_points = good_new.reshape(-1, 1, 2)
            prev_gray = gray.copy()
            tracking_frames_left -= 1

        cv2.imshow("Optical Flow & Motion Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
