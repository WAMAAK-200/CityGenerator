import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import os
import sys

def visualize_complete_map(filename="complete_map.json"):
    if not os.path.exists(filename):
        print("Error: complete_map.json not found. Run water_system.py first.")
        sys.exit(1)
        
    with open(filename, 'r') as f:
        data = json.load(f)
        
    heightmap = np.array(data["heightmap"])
    masks = data["masks"]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Final City Map (Seed: {data['seed']})")
    
    # 1. Base Terrain (Earth Tones)
    ax.imshow(heightmap, cmap='gist_earth', interpolation='bilinear')
    
    # 2. Create an Overlay Layer (RGBA)
    overlay = np.zeros((data["height"], data["width"], 4))
    
    # Helper to apply colors
    # Format: [R, G, B, Alpha]
    
    # OCEAN (Deep Blue)
    ocean_mask = np.array(masks["ocean"])
    overlay[ocean_mask] = [0.1, 0.2, 0.6, 1.0]
    
    # RIVERS (Bright Blue)
    river_mask = np.array(masks["river"])
    overlay[river_mask] = [0.2, 0.6, 0.9, 1.0]
    
    # LAKES (Teal)
    lake_mask = np.array(masks["lake"])
    overlay[lake_mask] = [0.2, 0.5, 0.8, 1.0]
    
    # SWAMPS (Murky Green/Purple - Semi transparent)
    swamp_mask = np.array(masks["swamp"])
    # Only apply swamp color where there ISN'T already water
    swamp_visible = swamp_mask & ~river_mask & ~lake_mask
    overlay[swamp_visible] = [0.3, 0.4, 0.2, 0.6] 
    
    # BEACHES (Sand Yellow)
    beach_mask = np.array(masks["beach"])
    overlay[beach_mask] = [0.9, 0.8, 0.6, 0.8]

    # Display Overlay
    ax.imshow(overlay)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a3399', label='Ocean'),
        Patch(facecolor='#3399e6', label='River (Bridgeable?)'),
        Patch(facecolor='#3380cc', label='Lake (No Bridge)'),
        Patch(facecolor='#4d6633', label='Swamp (Difficult Terrain)'),
        Patch(facecolor='#e6cc99', label='Beach'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_complete_map()