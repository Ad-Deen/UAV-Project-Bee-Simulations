# 🐝 BEE: Multi-Agent Autonomous Drone Project

The **BEE Project** is a modular research platform for developing and testing autonomous UAV capabilities in simulation and embedded hardware. It integrates multiple subsystems under one repository, enabling a complete workflow from **low-level flight control** to **high-level perception and mapping**.

BEE is designed as a flexible sandbox for experimenting with new ideas in **autonomous navigation, perception, and communication pipelines**.

---

✈️ **Core Components**

### [Flight Controller Development (Simulation)](bee/scripts/README.md)
Custom controllers for UAV stabilization and navigation, tested in simulation with Gazebo/ROS 2. Includes PID-based stabilization, teleoperation scripts, and prototype autonomous control pipelines.

### [Monocular Vision Depth Perception (Depth Anything V2)](bee/scripts/Depth-Anything-V2/README.md)
A deep-learning-based pipeline for monocular depth estimation using Depth Anything V2. Generates dense depth maps for 3D scene reconstruction, odometry, and SLAM integration.

### [Structure-from-Motion (SfM) & Visual Odometry](bee/sfm_practice/Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure_1/README.md)
Multiple pipelines for reconstructing 3D structure from images, including ray-based triangulation, feature-based SfM, and visual odometry.

    
---

## 📂 Repository Structure (Highlights)

- **`scripts/`** – UAV control logic, perception pipelines, SfM experiments.
- **`launch/`** – ROS 2 launch files for BEE simulation, RViz, and SfM scanning.
- **`CameraWebServer_OV5640AF/`** – ESP32-CAM firmware for video streaming.
- **`Depth-Anything-V2/`** – Monocular depth estimation pipelines.
- **`ray based approach/` & `visual odometry/`** – Research pipelines for SfM & VO.
- **`urdf/`, `meshes/`, `worlds/`** – UAV models and Gazebo simulation environments.

---

## 🚀 Usage Workflow

1. **Simulate the Drone** – Launch Gazebo worlds with the BEE URDF and control nodes.
2. **Enable Flight Control** – Run stabilization/teleop scripts for manual or assisted flight.
3. **Add Perception** – Launch the Depth Anything V2 pipeline for depth-aware perception.
4. **Transmit Data** – Use the ESP32-CAM telemetry stack for live video and state streaming.
5. **Experiment with SfM** – Run visual odometry and structure-from-motion pipelines for 3D reconstruction.

---

## 📚 Documentation Roadmap

- Flight Controller Development
- Monocular Depth Perception
- Structure-from-Motion Pipelines
