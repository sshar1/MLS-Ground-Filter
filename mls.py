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

# Get all points within neighbor range of given point
def get_all_neighbors(given_point, points):
    neighbors = set()
    for point in points:
        if point is given_point: continue
        if in_range(given_point, point):
            neighbors.add(tuple(point))
    return neighbors

# Return true if point2 is in neighbor range of point1
def in_range(point1, point2):
    return abs(point1[0] - point2[0]) < NEIGHBOR_RANGE and abs(point1[1] - point2[1]) < NEIGHBOR_RANGE and abs(point1[2] - point2[2]) < NEIGHBOR_RANGE

def get_slope(point1, point2):
    xy_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return xy_distance <= 1e-8 or abs(point2[2] - point1[2]) / xy_distance

def point_is_ground(point):
    neighbors = get_all_neighbors(point, points)
    for neighbor in neighbors:
        if get_slope(point, neighbor) >= SLOPE_THRESH:
            return False
    return True

points = parse_data('pointclouddata/point_cloud_120.csv')
# Format: [[x, y, z], [x, y, z], ...]

'''
Explanation:
1. Look at neighboring points for each point (e.g. points that fall within 5cm box)
2. Look at slope for each neighboring point
3. If there is one point with a slope above certain threshold (e.g. 0.3), that point is not ground
'''