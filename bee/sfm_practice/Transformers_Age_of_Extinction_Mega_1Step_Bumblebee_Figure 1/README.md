# Untitled

# Structure-from-Motion (SfM) Core Script Documentation

This script implements a **real-time Structure-from-Motion (SfM)** pipeline using ROS2, OpenCV, Open3D, and CPD for rigid point cloud alignment. It subscribes to camera images and odometry data, computes 3D points from tracked image features, aligns successive frames, and visualizes the reconstructed 3D scene.

---

## Overview

The SfM pipeline performs the following tasks:

1. **Feature Detection and Tracking**
    - Detects Shi-Tomasi corners in incoming camera frames.
    - Tracks features across frames using Lucas-Kanade Optical Flow.
2. **Orientation and Ray Computation**
    - Converts 2D image points to orientation angles.
    - Transforms orientations into 3D direction vectors in the world frame.
3. **Triangulation and 3D Reconstruction**
    - Triangulates corresponding rays from consecutive frames to compute 3D points.
    - Aligns new points to the accumulated 3D point cloud using **Coherent Point Drift (CPD)** rigid registration.
4. **Filtering and Buffering**
    - Applies DBSCAN clustering to remove outliers and merge close points.
    - Maintains a FIFO buffer to manage accumulated points.
5. **Visualization**
    - Visualizes rays, origins, and 3D points in Open3D.
    - Shows tracked features in real-time on the camera feed.

---

## Workflow

### Mermaid Flowchart

```mermaid
flowchart TD
    A[Start: Subscribe to Camera & Odometry] --> B[Convert ROS Image to OpenCV Frame]
    B --> C[Detect or Track Feature Points]
    C --> D[Compute 2D Orientations from Image Points]
    D --> E[Convert Orientations to 3D Direction Vectors]
    E --> F[Triangulate Rays to Compute 3D Points]
    F --> G[Align New 3D Points to Accumulated Cloud (CPD)]
    G --> H[Filter Points using DBSCAN]
    H --> I[Update FIFO Buffer of 3D Points]
    I --> J[Visualize Rays, Origins, and Point Clouds (Open3D)]
    J --> K[Publish Velocity Commands (Optional)]
    K --> L[Loop for Next Frame]
    L --> C

```

This flowchart shows the **main SfM processing loop**, starting from camera input, feature tracking, ray triangulation, point cloud alignment, filtering, buffering, visualization, and optional velocity command publication.

---

## Key Functional Components

- **Pose to Euler Conversion**: Converts quaternion orientation to Euler angles and position.
- **Orientation Computation**: Maps 2D image points to 3D orientation angles based on camera FOV.
- **Direction Vector Calculation**: Converts Euler angles to 3D direction vectors for rays.
- **Triangulation**: Computes the 3D intersection point between two rays.
- **Point Cloud Alignment**: Aligns successive 3D point clouds using rigid CPD registration.
- **Point Cloud Filtering**: Removes outliers and merges nearby points using DBSCAN.
- **FIFO Buffer Management**: Maintains a fixed-size history of points for accumulation.
- **Visualization**: Displays rays, origins, and point clouds using Open3D for debugging and analysis.

---

## ROS2 Node: `DataSubscriberNode`

### Subscriptions

- `/camera`: Receives camera frames for SfM processing.
- `/model/vehicle_blue/odometry`: Receives odometry data for pose tracking.

### Publisher

- `/model/vehicle_blue/cmd_vel`: Publishes velocity commands for testing.

### Node Behavior

- Tracks feature points in real-time using optical flow.
- Every 20 frames, triangulates 3D points and aligns them with the accumulated cloud.
- Updates visualization buffers with new rays and points.
- Re-detects features when the number of tracked points falls below a threshold.

---

## Visualization

- Shows 3D rays, origins, and points using Open3D.
- Displays tracked 2D features in the camera frame for debugging.
- Visualizes accumulated 3D point cloud after alignment with CPD.

---

## Notes

- The system assumes sufficient vehicle motion (>0.3 meters) between frames for accurate triangulation.
- Velocity publishing is configurable and can be adjusted for the environment or robot.
- Interpolation factors control how old points are moved towards new points, balancing stability and responsiveness.