import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import sys
import os

class TerrainVisualizer:
    def __init__(self, filepath="terrain_data.json"):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found. Run perlin.py first!")
            sys.exit(1)
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load and convert lists back to numpy arrays
        self.heightmap = np.array(data["heightmap"])
        self.water_mask = np.array(data["water_mask"], dtype=bool)
        self.river_mask = np.array(data["river_mask"], dtype=bool)
        self.buildable_mask = np.array(data["buildable_mask"], dtype=bool)
        self.slope = np.array(data["slope"])
        
        self.width = data["metadata"]["width"]
        self.height = data["metadata"]["height"]
        self.seed = data["metadata"]["seed"]

    def show_dashboard(self):
        """Displays the 3-panel terrain analysis dashboard."""
        fig = plt.figure(figsize=(18, 6))
        plt.suptitle(f"Terrain Analysis (Seed: {self.seed})", fontsize=16, fontweight='bold')

        # --- PANEL 1: Composite Terrain Map ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Topographic Map")
        
        # 1. Base Earth Tones
        ax1.imshow(self.heightmap, cmap='gist_earth', interpolation='bilinear')
        
        # 2. Water Overlay (Blue)
        # Create a mask that is transparent everywhere except where there is water
        water_layer = np.zeros((self.height, self.width, 4))
        water_layer[self.water_mask] = [0.1, 0.3, 0.8, 0.8] # Deep Blue
        water_layer[self.river_mask] = [0.2, 0.6, 1.0, 1.0] # Bright Blue River
        ax1.imshow(water_layer)
        
        # --- PANEL 2: Habitability (For City Generator) ---
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("City Habitability Mask")
        
        # Green = Buildable, Red = No Build (Water/Steep)
        habitability_layer = np.zeros((self.height, self.width, 3))
        
        # Fill background Red (Unbuildable)
        habitability_layer[:, :] = [0.8, 0.2, 0.2] 
        
        # Fill Buildable Green
        habitability_layer[self.buildable_mask] = [0.2, 0.8, 0.2]
        
        # Darken water areas for clarity
        habitability_layer[self.water_mask] = [0.0, 0.1, 0.3]
        
        ax2.imshow(habitability_layer)
        
        # Stats
        buildable_percent = (np.sum(self.buildable_mask) / self.buildable_mask.size) * 100
        ax2.set_xlabel(f"Buildable Area: {buildable_percent:.1f}%")

        # --- PANEL 3: Slope / Cliffs ---
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Slope & Cliffs")
        
        # White = Flat, Black = Vertical Cliff
        # We invert the slope visualization so high slope is dark
        slope_viz = ax3.imshow(self.slope, cmap='Greys', vmin=0, vmax=0.2)
        plt.colorbar(slope_viz, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Check if a specific file was passed, otherwise default
    target_file = sys.argv[1] if len(sys.argv) > 1 else "terrain_data.json"
    
    print(f"Visualizing {target_file}...")
    viz = TerrainVisualizer(target_file)
    viz.show_dashboard()