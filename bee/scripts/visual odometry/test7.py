import cv2
import numpy as np
import open3d as o3d
import time

# ---------- Feature Detection ----------
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

# ---------- Optical Flow and Pose Estimation ----------
def estimate_motion(prev_pts, curr_pts):
    flow = curr_pts - prev_pts
    avg_flow = np.mean(flow, axis=0).ravel()
    tx = avg_flow[0]
    ty = avg_flow[1]
    tz = -np.mean(np.linalg.norm(flow, axis=2)) / 5
    yaw = np.arctan2(tx, 1.0) / 10
    return tx, ty, tz, yaw

# ---------- Create Open3D Visualizer ----------
def create_gimbal():
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Camera Pose', width=640, height=480)
    vis.add_geometry(axis)
    return vis, axis

# ---------- Main Tracking and Visualization ----------
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
    refresh_interval = 10
    prev_gray = None
    feature_points = None
    yaw_angle = 0.0
    pose = np.eye(4)

    vis, gimbal = create_gimbal()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % refresh_interval == 0 or feature_points is None:
            feature_points = get_strong_edge_features(gray, max_features=50)
            prev_gray = gray.copy()
            prev_points = feature_points

        elif feature_points is not None and len(feature_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, feature_points, None, **lk_params)
            good_new = next_points[status == 1].reshape(-1, 1, 2)
            good_old = feature_points[status == 1].reshape(-1, 1, 2)

            tx, ty, tz, dyaw = estimate_motion(good_old, good_new)
            yaw_angle += dyaw

            dx_world = tx * np.cos(yaw_angle) - tz * np.sin(yaw_angle)
            dz_world = tx * np.sin(yaw_angle) + tz * np.cos(yaw_angle)

            delta_pose = np.eye(4)
            delta_pose[:3, 3] = [dx_world / 50, -ty / 50, dz_world / 50]
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, yaw_angle, 0])
            delta_pose[:3, :3] = rotation

            pose = pose @ delta_pose
            gimbal.transform(pose)

            vis.update_geometry(gimbal)
            vis.poll_events()
            vis.update_renderer()

            feature_points = good_new.reshape(-1, 1, 2)
            prev_gray = gray.copy()

        cv2.imshow("Camera Optical Flow Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == "__main__":
    main()
