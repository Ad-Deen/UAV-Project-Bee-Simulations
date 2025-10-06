import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from pycpd import RigidRegistration
import numpy as np
from geometry_msgs.msg import Twist
from sklearn.cluster import DBSCAN
################################################# Utility Function ################################################################
def pose_to_euler(cur_pose):
    """
    Converts pose with quaternion orientation to a list of position and Euler angles in radians.
    
    Args:
        cur_pose (list): A list containing [x, y, z, qx, qy, qz, qw], where
                         - x, y, z are the position coordinates
                         - qx, qy, qz, qw are the quaternion components
    
    Returns:
        list: [tx, ty, tz, rx, ry, rz] where
              - tx, ty, tz are the position coordinates
              - rx, ry, rz are the Euler angles in radians
    """
    # Extract position and quaternion
    tx, ty, tz = cur_pose[0:3]  # Position
    qx, qy, qz, qw = cur_pose[3:7]  # Quaternion

    # Convert quaternion to Euler angles (in radians)
    rotation = R.from_quat([qx, qy, qz, qw])
    rx, ry, rz = rotation.as_euler('xyz', degrees=False)  # 'xyz' represents roll, pitch, yaw

    # Combine position and Euler angles
    return [tx, ty, tz, rx, ry, rz]


def plot_multiple_rays_with_points(origins, direction_vectors_list, points, ray_length=1.0):
    """
    Plots rays and 3D points in Open3D given multiple origins, corresponding sets of direction vectors,
    and a list of 3D points.

    Args:
        origins (list of np.array): A list of 3D origins, one for each set of direction vectors.
        direction_vectors_list (list of list of np.array): A list where each element is a list of 
            3D direction vectors corresponding to an origin.
        points (list of np.array): A list of 3D points to plot.
        ray_length (float): The length of each ray for visualization.
    """
    all_lines = []  # To store all line connections
    all_points = []  # To store all points
    spheres = []  # To store origin spheres
    offset = 0  # Index offset for connecting lines

    for origin, direction_vectors in zip(origins, direction_vectors_list):
        # Add the origin to points
        all_points.append(origin)

        # Create a sphere at the current origin
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # Adjust radius as needed
        sphere.translate(origin)  # Move sphere to origin
        sphere.paint_uniform_color([0, 1, 0])  # Set color to green
        spheres.append(sphere)

        # Process direction vectors for the current origin
        for i, direction in enumerate(direction_vectors):
            endpoint = origin + direction * ray_length  # Compute the endpoint
            all_points.append(endpoint)  # Add endpoint to points
            all_lines.append([offset, offset + i + 1])  # Connect origin to endpoint

        # Update offset for the next set of rays
        offset += len(direction_vectors) + 1

    # Convert to Open3D format for rays
    all_points = np.array(all_points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)

    # Optionally set colors for the rays
    colors = [[1, 0, 0] for _ in range(len(all_lines))]  # Red color for all rays
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create spheres for the 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0, 0, 1])  # Blue color for points

    # Visualize the rays, origins, and points
    o3d.visualization.draw_geometries([line_set, point_cloud] + spheres)


def compute_orientations(points, image_width=640, image_height=480, hfov=60, vfov=40):
    """
    Converts a list of 2D image points into orientation angles [rx, ry, rz].

    Parameters:
    points: list of lists, where each sublist contains a point [[x1, y1], [x2, y2], ..., [xn, yn]]
    image_width: int, width of the image in pixels (default: 640)
    image_height: int, height of the image in pixels (default: 480)
    hfov: float, horizontal field of view in degrees (default: 60)
    vfov: float, vertical field of view in degrees (default: 40)

    Returns:
    orientations: list of lists, where each sublist contains [rx, ry, rz]
    """
    orientations = []

    # Convert FOV to radians
    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    # Compute pixel-to-angle conversion factors
    pixels_per_degree_h = image_width / hfov
    pixels_per_degree_v = image_height / vfov

    for point in points:
        x, y = point[0]

        # Map x (horizontal) to rz (-30 to +30 degrees)
        rz_deg = (x - image_width / 2) / pixels_per_degree_h
        rz_rad = np.radians(rz_deg)

        # Map y (vertical) to ry (+20 to -20 degrees, inverted axis)
        ry_deg = -(y - image_height / 2) / pixels_per_degree_v
        ry_rad = np.radians(ry_deg)

        orientations.append([0, ry_rad, -rz_rad])

    return orientations

def orientations_to_directions(orientations,offset):
    """
    Converts a list of orientations (rx, ry, rz) in radians to direction vectors.

    Args:
        orientations (list of lists): A list where each entry is [rx, ry, rz] in radians.

    Returns:
        list of np.array: A list of 3D direction vectors corresponding to the orientations.
    """
    directions = []

    for orientation in orientations:
        rx, ry, rz = orientation
        rx += offset[0]
        ry += offset[1]
        rz += offset[2]

        # Compute rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        R_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combine rotations: R = Rz * Ry * Rx
        R = R_z @ R_y @ R_x

        # Apply rotation to the default forward vector [0, 0, 1]
        forward_vector = np.array([1, 0, 0])
        direction = R @ forward_vector

        directions.append(direction)

    return directions

def triangulate_point(ray1, ray2):
    """
    Computes the 3D point where two rays intersect or come closest to each other.
    
    Parameters:
    ray1 : tuple
        A tuple (origin1, direction1) for the first ray.
        - origin1: numpy array of shape (3,) representing the starting point of the ray.
        - direction1: numpy array of shape (3,) representing the direction vector of the ray.
    ray2 : tuple
        A tuple (origin2, direction2) for the second ray.
        - origin2: numpy array of shape (3,) representing the starting point of the ray.
        - direction2: numpy array of shape (3,) representing the direction vector of the ray.
        
    Returns:
    point : numpy array
        The 3D point where the rays intersect or are closest.
    """
    origin1, end1 = ray1
    origin2, end2 = ray2

    # Direction vectors of the rays
    direction1 = end1 - origin1
    direction2 = end2 - origin2

    # Normalize direction vectors
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)

    # Compute the cross-product of the directions
    cross_dir = np.cross(direction1, direction2)
    cross_dir_norm = np.linalg.norm(cross_dir)

    if cross_dir_norm < 1e-6:  # Rays are nearly parallel
        return (origin1 + origin2) / 2  # Midpoint of origins as a fallback

    # Matrix to solve for the closest points
    A = np.array([direction1, -direction2]).T
    b = origin2 - origin1

    # Solve for the scalars (t1, t2) along the ray directions
    t = np.linalg.lstsq(A, b, rcond=None)[0]

    # Closest points on each ray
    closest_point_ray1 = origin1 + t[0] * direction1
    closest_point_ray2 = origin2 + t[1] * direction2

    # Return the midpoint of the closest points as the triangulated point
    triangulated_point = (closest_point_ray1 + closest_point_ray2) / 2
    return triangulated_point


def align_point_clouds_with_cpd(old_points, new_points):
    """
    Aligns two point clouds using the CPD algorithm (rigid).
    
    Args:
        old_points (list or np.ndarray): Existing points (2D array or list of 3D points).
        new_points (list of np.ndarray or np.ndarray): List of 3D points for new points (can be 2D array).
        
    Returns:
        combined_points (list of np.ndarray): Combined old and aligned new points.
        aligned_points (np.ndarray): Aligned new points.
        transformation (dict): Transformation parameters (R, t, s).
    """
    # Convert new points from list of arrays to a single NumPy array if necessary
    new_points = np.asarray(new_points)
    # Ensure old_points is a NumPy array
    old_points = np.asarray(old_points)
    
    # If old points are empty, just use new points
    if old_points is None or old_points.size == 0:
        return list(new_points), new_points, {"R": np.eye(3), "t": np.zeros(3), "s": 1.0}
    
    # Ensure old_points is a NumPy array
    # old_points = np.asarray(old_points)
    
    # Check if old_points is in a list of arrays format (list of points as np.ndarrays)
    if isinstance(old_points, list):
        old_points = np.vstack(old_points)  # Convert the list to a 2D NumPy array

    # Set up CPD rigid registration
    registration = RigidRegistration(X=old_points, Y=new_points)
    
    # Perform registration
    aligned_points, (R, t, s) = registration.register()
    
    # Combine the old and aligned new points
    combined_points = list(old_points) + list(aligned_points)
    
    # Return results
    transformation = {"R": R, "t": t, "s": s}
    return combined_points, aligned_points, transformation

def transform_points_towards_new(all_3D, all_3D_new, interpolation_factor=0.5):
    """
    Transforms the points in all_3D towards the points in all_3D_new by the specified interpolation factor.
    
    Args:
        all_3D (list of np.ndarray): List of 3D points for the old set.
        all_3D_new (list of np.ndarray): List of 3D points for the new set.
        interpolation_factor (float): Fraction (0-1) to move all_3D points towards all_3D_new points.
        
    Returns:
        transformed_points (np.ndarray): Transformed all_3D points, moved towards all_3D_new.
    """
    # Convert lists to numpy arrays for easy manipulation
    all_3D = np.asarray(all_3D)
    all_3D_new = np.asarray(all_3D_new)
    
    # Compute the centroid (mean position) of each set of points
    centroid_all_3D = np.mean(all_3D, axis=0)
    centroid_all_3D_new = np.mean(all_3D_new, axis=0)
    
    # Compute the translation vector to move all_3D points towards all_3D_new points
    translation_vector = centroid_all_3D_new - centroid_all_3D
    
    # Interpolate the all_3D points towards the new points by the interpolation factor
    transformed_points = all_3D + interpolation_factor * translation_vector
    
    return transformed_points


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.

    Args:
        point1 (list or np.ndarray): The first 3D point [tx, ty, tz].
        point2 (list or np.ndarray): The second 3D point [tx, ty, tz].

    Returns:
        float: The distance between the two points.
    """
    # Convert inputs to numpy arrays if they are not already
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Check if both points are 3D
    if point1.shape != (3,) or point2.shape != (3,):
        raise ValueError("Both points must be 3D coordinates in the form [tx, ty, tz]")

    # Compute the Euclidean distance
    distance = np.linalg.norm(point1 - point2)
    return distance


def filter_3d_points(points, eps=0.1, min_samples=5):
    """
    Filters a list of 3D points using DBSCAN to remove outliers and unite close points.

    Parameters:
        points (list of np.ndarray): List of 3D points in the format [array([x, y, z]), ...].
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
        filtered_points (list of np.ndarray): Filtered list of 3D points.
    """
    # Convert the list of 3D points to a numpy array
    points_array = np.array(points)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points_array)

    # Extract clusters (ignore noise labeled as -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove the noise label

    filtered_points = []

    for label in unique_labels:
        # Get points in the current cluster
        cluster_points = points_array[labels == label]

        # Compute the centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)

        # Add the centroid to the filtered points
        filtered_points.append(centroid)

    # Convert the filtered points back to the original format
    filtered_points = [np.array(point) for point in filtered_points]

    return filtered_points
    
def fifo_check_and_reduce(points_list, buffer_size):
    """
    Maintains a fixed-size buffer of 3D points using FIFO (First-In-First-Out) logic.
    
    Parameters:
        points_list (list): The current list of 3D points (numpy arrays).
        buffer_size (int): The maximum number of points to retain in the buffer.

    Returns:
        list: The updated list of 3D points within the buffer size limit.
    """
    # Check if the list exceeds the buffer size
    if len(points_list) > buffer_size:
        excess_count = len(points_list) - buffer_size
        points_list = points_list[excess_count:]  # Remove the oldest elements
    
    return points_list

##################################################################################################################################
########################### Globals #######################################################################


# Parameters for Shi-Tomasi Corner Detection
feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=5, blockSize=5)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get the first frame and convert it to grayscale


# Threshold for re-detecting features
REDETECT_THRESHOLD = 500  # Minimum number of tracked features to trigger re-detection
#############################################################################################################
class DataSubscriberNode(Node):
    def __init__(self):
        super().__init__('data_subscriber_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscription to the /camera topic
        self.camera_subscription = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            10
        )

        # Subscription to the /model/vehicle_blue/odometry topic
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/model/vehicle_blue/odometry',
            self.odometry_callback,
            10
        )
        # Publisher for /model/vehicle_blue/cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/model/vehicle_blue/cmd_vel',
            10
        )
        # Publish velocity commands periodically
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)  # Publish at 10 Hz

        # Internal storage for the most recent data
        self.latest_camera_data = None
        self.latest_odometry_data = None

        self.get_logger().info("Data Subscriber Node has been started.")
        # Parameters for feature detection and tracking
        self.feature_params = dict(maxCorners=500, qualityLevel=0.4, minDistance=5, blockSize=5)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize variables for tracking
        self.frame_count = 0
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.cur_pose = []


        self.all_3D = []
        self.all_3D_old = []
        self.aligned_points = []
        self.frame = []
        self.all_pos = []
        self.features = []
        self.vector = []
        self.all_3D_accum = []
        self.prev_pos = None
        self.prev_feat = None
        self.acceptance_threshold = 0.3

    def camera_callback(self, msg):
        """
        Callback function to process incoming camera frames for SfM.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.frame_count == 0:
                # First frame setup
                self.prev_frame = cv_image
                self.prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            else:
                # Process subsequent frames
                curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

                if self.prev_points is not None and len(self.prev_points) > 0:
                    # Calculate optical flow
                    curr_points, status, err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, curr_gray, self.prev_points, None, **self.lk_params
                    )

                    # Select good points
                    good_new = curr_points[status == 1]

                    # Draw the tracked points
                    for new in good_new:
                        a, b = new.ravel()
                        cv_image = cv2.circle(cv_image, (int(a), int(b)), 3, (0, 0, 255), -1)

                    # Update points for the next iteration
                    self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    # Re-detect features if none are left
                    self.prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
                
                #==============================================Frame logic here ==================================================
                if (self.frame_count-1)%20 == 0:
                # if i == 2:
                    if self.cur_pose is not None and self.prev_points is not None:
            
                        self.prev_pos= self.cur_pose
                        self.prev_feat = self.prev_points
                        self.get_logger().info(f"frame1 {self.frame_count}")
                    
                    

                if (self.frame_count-19)%20 == 0:
                # if i == 18:
                    rays1 = []
                    rays2 = []
                    all_3D_new = []
                    self.get_logger().info(f"frame19 {self.frame_count}")
                    pos1 = self.prev_pos
                    point1 = self.prev_feat
                    
                    pos2 = self.cur_pose
                    point2 = self.prev_points
                    self.all_pos.append(pos2)
                    self.all_pos.append(pos1)
                    
                    

                    # # Compute orientations------------------------------------------------------------
                    orientations1 = compute_orientations(point1)
                    orientations2 = compute_orientations(point2)
                    
                    # # print(orientations1)
                    # # Convert orientations to direction vectors---------------------------------------
                    directions1 = orientations_to_directions(orientations1,[0,0,pos1[-1]])
                    directions2 = orientations_to_directions(orientations2,[0,0,pos2[-1]])
                    
                    
                    # #Check directions ok or not
                    if len(directions1) == 0 or len(directions2) == 0:
                        self.get_logger().error("Empty directions array. Skipping this frame.")
                        return
                    
                    for i in range(len(directions1)):
                        endpoint1 = pos1[:3] + directions1[i] * 0.1  # Compute the endpoint1---------------------
                        rays1.append((pos1[:3],endpoint1))
                        
                    for i in range(len(directions2)):
                        endpoint2 = pos2[:3] + directions2[i] * 0.1  # Compute the endpoint2--------------------
                        rays2.append((pos2[:3],endpoint2))

                    

                    for ray1, ray2 in zip(rays1, rays2):
                        point = triangulate_point(ray1, ray2)
                        all_3D_new.append(point)
                    
                    
                    # if len(all_3D_new) == 0:
                    #     self.get_logger().error("all_3D_new is empty. Skipping alignment.")
                    #     return

                    # self.get_logger().info(f"pos1 {pos1[:3]}")
                    # self.get_logger().info(f"pos2 {pos2[:3]}")
                    if calculate_distance(pos1[:3], pos2[:3]) > self.acceptance_threshold:
                        self.all_3D_old, aligned_points, transform = align_point_clouds_with_cpd(self.all_3D_old, all_3D_new)
                        # self.all_3D_old = filter_3d_points(self.all_3D_old, eps=0.5, min_samples=5)
                        self.all_3D_old = fifo_check_and_reduce(self.all_3D_old, 400)
                        self.get_logger().info(f"3D points found {self.all_3D_old}")
                        self.all_3D_accum.extend(self.all_3D_old)
                        self.all_3D = transform_points_towards_new(self.all_3D_accum, all_3D_new, interpolation_factor=1.0)
                        # self.all_3D_old, self.aligned_points, transform = align_point_clouds_with_cpd( self.aligned_points, all_3D_new) 
                    # self.get_logger().info(f"pos1 {rays1}")
                        self.get_logger().info(f"3D points found {len(self.all_3D_old)}")
                        self.frame.append(pos2[:3])
                        self.frame.append(pos1[:3])
                        self.vector.append(directions1[::30])
                        self.vector.append(directions2[::30])
                    else:
                        self.get_logger().info(f"Frames too close. They must be > {self.acceptance_threshold} m")
                    
                    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    # # self.all_3D_accum.extend(self.all_3D_old)
                    # self.all_3D = transform_points_towards_new(self.all_3D_accum, all_3D_new, interpolation_factor=1.0)
                
                #=================================================================================================================

                # Remove old features periodically
                if self.frame_count % 20 == 0:
                    self.prev_points = None

                # Update the previous frame for the next iteration
                self.prev_gray = curr_gray.copy()

            # Display the output (real-time feature tracking)
            cv2.imshow("Feature Tracking", cv_image)
            cv2.waitKey(1)

            # Increment frame counter
            self.frame_count += 1
            

        except Exception as e:
            self.get_logger().error(f"Failed to process camera data: {e}")

    def destroy_node(self):
        #plot_points_and_rays_with_axes(all_3D,all_pos)
        plot_multiple_rays_with_points(self.frame, self.vector,self.all_3D)
        super().destroy_node()
        cv2.destroyAllWindows()
        
    def publish_cmd_vel(self):
        # Create a Twist message
        twist_msg = Twist()
        twist_msg.linear.x = 1.0  # Forward throttle (1 m/s)
        twist_msg.angular.z = -0.25  # Yaw rate (0.25 rad/s)

        # Publish the Twist message
        self.cmd_vel_publisher.publish(twist_msg)
    
    def odometry_callback(self, msg):
        """
        Callback function for odometry data.
        """
        self.latest_odometry_data = msg
        self.cur_pose = [msg.pose.pose.position.x , msg.pose.pose.position.y+4.00 , msg.pose.pose.position.z+1.325 , msg.pose.pose.orientation.x , msg.pose.pose.orientation.y , msg.pose.pose.orientation.z , msg.pose.pose.orientation.w]
        self.cur_pose = pose_to_euler(self.cur_pose)
        self.cur_pose[-1] = self.cur_pose[-1] -1.57
        self.cur_pose = np.array(self.cur_pose)
        # self.get_logger().info(
        #     # f"Received odometry data: "
        #     f"Position: {self.cur_pose})"
        # )



def main(args=None):
    rclpy.init(args=args)
    node = DataSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        
        # Cleanup OpenCV windows and shutdown node
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
