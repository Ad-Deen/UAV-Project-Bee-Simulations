import numpy as np

# Create a 1x3 matrix (row vector)
row_vector = np.array([[1, 2, 3]])

# Convert it to a 3x1 matrix (column vector)
column_vector = row_vector.T  # Transpose the row vector

# Print the result
print("Row Vector (1x3):")
print(row_vector)
print("\nColumn Vector (3x1):")
print(column_vector)

