import numpy as np

def generate_random_points(n_points, width, height):
    """Generate random x, y coordinates."""
    # Creates an array of random points scaled to width and height
    return np.random.rand(n_points, 2) * [width, height]

def centroid_region(vertices):
    """
    Calculate the centroid of a polygon.
    Used for Lloyd's relaxation to make blocks uniform-ish.
    """
    length = vertices.shape[0]
    sum_x = np.sum(vertices[:, 0])
    sum_y = np.sum(vertices[:, 1])
    return sum_x / length, sum_y / length