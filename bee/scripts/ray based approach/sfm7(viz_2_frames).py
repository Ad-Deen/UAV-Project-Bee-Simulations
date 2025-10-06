import numpy as np
import cv2
## HFOV 1.047 rad VFOV 0.785 rad
# Load the saved data
loaded_data = np.load("bee/scripts/feature_data.npy", allow_pickle=True)
print(len(loaded_data))
# Access the first two frames (assuming you saved data for frames 32 and 50)
frame1_data = loaded_data[0]
frame2_data = loaded_data[1]

# Extract image, position, and feature points for each frame
img1 = frame1_data['image']
pos1 = frame1_data['position']
points1 = frame1_data['feature_points']

# print(f"frame 1 features={points1}")

img2 = frame2_data['image']
pos2 = frame2_data['position']
points2 = frame2_data['feature_points']

for i in range(len(loaded_data)):
    print(loaded_data[i]['position'])
