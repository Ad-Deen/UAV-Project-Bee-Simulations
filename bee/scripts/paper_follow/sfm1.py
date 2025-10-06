import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
#================================================= Utility Functions ====================================================================
def correspondence_search(img1, img2, ratio_test=0.0, ransac_thresh=1.0):
    """
    Perform correspondence search using ORB feature extraction, BFMatcher for binary descriptors, and RANSAC-based geometric verification.
    
    Parameters:
        img1 (ndarray): First input image.
        img2 (ndarray): Second input image.
        ratio_test (float): Lowe's ratio test threshold for filtering matches.
        ransac_thresh (float): RANSAC threshold for geometric verification (in pixels).
    
    Returns:
        good_matches (list): Filtered and verified matches.
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): All matches (before RANSAC filtering).
    """
    # Step 1: Initialize the ORB detector
    orb = cv2.ORB_create()

    # Step 2: Detect keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Step 3: Use BFMatcher for binary descriptor matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # NORM_HAMMING for binary descriptors
    
    # Step 4: Perform matching using BFMatcher
    matches = bf.match(des1, des2)
    
    # Step 5: Sort the matches based on the distance (best matches first)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Step 6: Apply Lowe's ratio test to filter out poor matches
    good_matches = []
    for i in range(1, len(matches)):
        if matches[i].distance < ratio_test * matches[i-1].distance:
            good_matches.append(matches[i])

    # Step 7: Geometric verification using RANSAC to filter incorrect matches
    if len(good_matches) > 0:  # Need at least 4 matches for RANSAC
        # Extract location of matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find the homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        
        # Mask indicates the inliers
        matchesMask = mask.ravel().tolist()
        good_matches = [good_matches[i] for i in range(len(good_matches)) if matchesMask[i]]
    else:
        print("Not enough matches found for RANSAC.")

    # Return the filtered matches and keypoints
    return good_matches, kp1, kp2, matches

#===================================================================================================================================
# Path to your .npy file
file_path = "bee/scripts/camera_odom_data.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Load images
img1 = data[2][1]
img2 = data[50][1]

pos1 = data[2][0]
pos2 = data[50][0]

img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

good_matches, kp1, kp2, matches = correspondence_search(img1, img2)

# Draw the matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


