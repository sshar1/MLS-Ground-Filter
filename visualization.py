import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import mls3

# Top-level variables to control display
DISPLAY_MIDLINE = True
DISPLAY_SECTION_CIRCLES = True
SHOW_GROUND = False
SHOW_ORIGINAL_POINTS = True
SHOW_ALL_PLOTS = False

DISTANCE_THRESHOLD = 30

POINT_CLOUD_PATH = 'pointclouddata/intensity_2.csv'

point_clouds = [
    'pointclouddata/point_cloud_73.csv',
    'pointclouddata/point_cloud_120.csv',
    'pointclouddata/point_cloud_221.csv',
    'pointclouddata/point_cloud_222.csv',
    'pointclouddata/intensity_1.csv',
    'pointclouddata/intensity_2.csv'
]

def filter_points_by_distance(points, reference_point, max_distance):
    distances = np.linalg.norm(points - reference_point, axis=1)
    return points[distances <= max_distance]

def parse_and_visualize_clusters(point_cloud_path):
    """
    Parse a CSV file with x, y, z information and visualize the clusters, midline points, and section circles.
    """
    start = time.time()

    # Initialize points and KD-tree
    true_points = mls3.init_points(point_cloud_path, DISTANCE_THRESHOLD)

    # Classify points as ground or non-ground
    print('classifying points...')
    ground_mask = np.array([mls3.point_is_ground(point) for point in true_points])
    ground_points = true_points[ground_mask]
    non_ground_points = true_points[~ground_mask]
    print('points classified!')
    end = time.time()
    print(f"Time taken: {(end-start)*10**3:.03f}ms")

    # Visualize points
    fig = plt.figure()
    fig.suptitle(f'Labeld points ({point_cloud_path})')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 40])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-15, 15])

    # Plot ground and non-ground points in batches
    print('start graph')
    if SHOW_GROUND:
        ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], s=1, color='black', label='Ground')
    ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], s=1, color='red', label='Non-Ground')
    ax.legend()

    if SHOW_ORIGINAL_POINTS:
        original_fig = plt.figure()
        original_fig.suptitle(f'Original points ({point_cloud_path})')
        og_ax = original_fig.add_subplot(111, projection='3d')
        og_ax.set_xlim([0, 40])
        og_ax.set_ylim([-20, 20])
        og_ax.set_zlim([-15, 15])

        og_ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1, color='black', label='All points')
        og_ax.legend()

if SHOW_ALL_PLOTS:
    for point_cloud in point_clouds:
        parse_and_visualize_clusters(point_cloud)
else:
    parse_and_visualize_clusters(POINT_CLOUD_PATH)

plt.show()