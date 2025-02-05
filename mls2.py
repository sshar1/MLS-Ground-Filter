from scipy.spatial import cKDTree
import numpy as np
import math
import pandas as pd

NEIGHBOR_RANGE = 2
SLOPE_THRESH = 0.3

def parse_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    
    # Ensure required columns are present
    if not all(col in data.columns for col in ['x', 'y', 'z']):
        raise ValueError("CSV file must contain 'x', 'y', 'z', columns.")
    
    # Extract points
    return data[['x', 'y', 'z']].values

def in_range_batch(point, neighbors):
    return np.all(np.abs(neighbors - point) < NEIGHBOR_RANGE, axis=1)

# Adjust get_all_neighbors to use the KD-tree
def get_all_neighbors(given_point, tree):
    # Query all neighbors within NEIGHBOR_RANGE
    indices = tree.query_ball_point(given_point, NEIGHBOR_RANGE)
    neighbors = points_array[indices]
    return neighbors

def get_slope(point1, point2):
    xy_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return xy_distance <= 1e-8 or abs(point2[2] - point1[2]) / xy_distance

def point_is_ground(point):
    neighbors = get_all_neighbors(point, tree)
    for neighbor in neighbors:
        if np.array_equal(point, neighbor):  # Skip itself
            continue
        if get_slope(point, neighbor) >= SLOPE_THRESH:
            return False
    return True

points = parse_data('pointclouddata/point_cloud_50.csv')

# Build the KD-tree once for all points
points_array = np.array(points)  # Convert to numpy array for efficient processing
# tree = cKDTree(points_array[:-1])

finite_mask = np.isfinite(points_array).all(axis=1)
points_array = points_array[finite_mask]

# Rebuild the KD-tree
tree = cKDTree(points_array)
