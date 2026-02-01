# terrain.py
import numpy as np
import config

def get_river_y(x):
    """
    Returns the Y position of the river for a given X.
    Uses a sine wave to make it look organic.
    """
    # A sine wave river: Amplitude * sin(Frequency * x) + Offset
    return 150 * np.sin(x / 200) + (config.HEIGHT * 0.7)

def is_water(x, y):
    """
    Returns True if the point hits the river.
    """
    # Check distance to the river curve
    river_y = get_river_y(x)
    dist_to_river = abs(y - river_y)
    return dist_to_river < (config.RIVER_WIDTH / 2)

def is_land(x, y):
    """
    Land if it's within the circle AND not in the river.
    """
    # 1. Circular City Boundary
    nx = 2 * (x / config.WIDTH) - 1
    ny = 2 * (y / config.HEIGHT) - 1
    dist_from_center = np.sqrt(nx**2 + ny**2)
    
    if dist_from_center >= config.LAKE_THRESHOLD:
        return False

    # 2. River Check
    if is_water(x, y):
        return False
        
    return True