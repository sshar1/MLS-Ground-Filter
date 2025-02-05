from scipy.spatial import cKDTree
import numpy as np
import math
import pandas as pd

NEIGHBOR_RANGE = 0.1
SLOPE_THRESH = 0.3 # Higher means stricter (i.e. you will get fewer ground points)

def parse_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    if not all(col in data.columns for col in ['x', 'y', 'z']):
        raise ValueError("CSV file must contain 'x', 'y', 'z' columns.")
    return data[['x', 'y', 'z']].values

# Gets slope between two points using xy-distance
# and relative z values
def get_slope(point1, point2):
    xy_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return xy_distance <= 1e-8 or abs(point2[2] - point1[2]) / xy_distance

# Classifies a point as ground if it is below the slope
# threshold for all points neighboring it
def point_is_ground(point):
    neighbors = get_all_neighbors(point, tree)
    for neighbor in neighbors:
        if np.array_equal(point, neighbor):  # Skip itself
            continue
        if get_slope(point, neighbor) >= SLOPE_THRESH:
            return False
    return True

# Removes all points farther than the given distance
def filter_points_by_distance(points, reference_point, max_distance):
    distances = np.linalg.norm(points - reference_point, axis=1)
    return points[distances <= max_distance]

# Filters out inf and nan points, as well as points outside
# the distance threshold
def init_points(point_cloud_path, distance_threshold):
    global points_array, tree

    points = parse_data(point_cloud_path)
    points_array = points

    # Filter out invalid points
    finite_mask = np.isfinite(points_array).all(axis=1)
    if not np.all(finite_mask):
        print(f"Filtered out {len(points_array) - np.sum(finite_mask)} invalid points.")
    points_array = points_array[finite_mask]

    reference_point = np.array([0, 0, 0])
    points_array = filter_points_by_distance(points_array, reference_point, distance_threshold)

    # Build the KD-tree
    tree = cKDTree(points_array)
    print("KD-tree built with", len(points_array), "points.")

    return points_array

# Uses KD tree to get all points within a range of 'neighbor range'
def get_all_neighbors(given_point, tree):
    # Ensure given_point is finite
    if not np.isfinite(given_point).all():
        raise ValueError(f"Invalid given_point: {given_point}")
    
    # Query all neighbors within NEIGHBOR_RANGE
    indices = tree.query_ball_point(given_point, NEIGHBOR_RANGE)
    neighbors = points_array[indices]
    return neighbors