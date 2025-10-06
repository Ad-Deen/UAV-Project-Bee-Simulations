import numpy as np

# Load the .npy file
depth = np.load('my_depth.npy')

# Inspect the loaded array
print("Shape:", depth.shape)
print("Dtype:", depth.dtype)
print("Range:", depth.min(), depth.max())
print("Sample values:", depth[0, :5])  # First row, first 5 values
print(depth)