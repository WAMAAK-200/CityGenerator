# visualizer.py
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import config
from terrain import get_river_y

def draw_city(city):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Background is the "Ground/Road" color
    ax.set_facecolor(config.BG_COLOR)
    
    # 1. Draw River (visual representation)
    x_vals = np.linspace(0, config.WIDTH, 100)
    y_vals = get_river_y(x_vals)
    
    # Fill between river edges
    ax.fill_between(x_vals, 
                    y_vals - config.RIVER_WIDTH/2, 
                    y_vals + config.RIVER_WIDTH/2, 
                    color=config.WATER_COLOR, alpha=1.0)

    # 2. Draw Blocks
    for block in city.blocks:
        poly = block['poly']
        b_type = block['type']
        
        # Determine Color based on Type
        if b_type == 'castle':
            c = config.CASTLE_COLOR
            lw = 0  # No border for castle
        else:
            c = config.LAND_COLOR
            lw = 0.5

        # Handle Multipolygons (sometimes shrinking splits a block in two)
        if poly.geom_type == 'MultiPolygon':
            geoms = poly.geoms
        else:
            geoms = [poly]

        for geom in geoms:
            x, y = geom.exterior.xy
            patch = MplPolygon(list(zip(x, y)), 
                               facecolor=c, 
                               edgecolor='#5c4a3d', 
                               linewidth=lw)
            ax.add_patch(patch)

    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.axis('off')
    
    # Dark border frame
    fig.patch.set_facecolor('#202020') 
    
    print("Rendering detailed map...")
    plt.show()