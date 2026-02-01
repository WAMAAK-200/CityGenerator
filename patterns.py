import numpy as np

def apply_default_pattern(city_map, centre):
    ci, cj = centre
    city_map[ci, :] = -1
    city_map[:, cj] = -1
    return city_map

def apply_grid_pattern(city_map, size, spacing=10):
    # Grid-like city pattern
    for i in range(0, size, spacing):
        city_map[i, :] = -1
        city_map[:, i] = -1
    return city_map

def apply_radial_pattern(city_map, size, centre):
    # Concentric rings and spokes.
    ci, cj = centre
    for i in range(size):
        for j in range(size):
            di, dj = i - ci, j - cj
            dist = np.sqrt(di**2 + dj**2)
            angle = np.arctan2(dj, di)
            
            # Rings every 12 units
            on_ring = (int(dist) % 12 == 0)
            # 8 spokes radiating from center
            on_spoke = any(abs(angle - (2 * np.pi / 8) * s) < 0.04 for s in range(8))
            
            if (on_ring or on_spoke) and dist < (size * 0.48):
                city_map[i, j] = -1
    return city_map