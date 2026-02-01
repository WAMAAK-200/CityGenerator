import numpy as np
import matplotlib.pyplot as plt
import json
import sys

class TerrainVisualizer:
    def __init__(self, data_source):
        """
        data_source: either .npy file, .json file, or dict
        """
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                with open(data_source, 'r') as f:
                    data = json.load(f)
                    self.heightmap = np.array(data["heightmap"])
                    self.slope = np.array(data["slope"])
                    self.buildable = np.array(data["buildable_mask"])
                    self.city_sites = data["city_sites"]
                    self.barriers = np.array(data["barrier_mask"])
            elif data_source.endswith('.npy'):
                self.heightmap = np.load(data_source)
                self.slope = np.gradient(self.heightmap)
                self.buildable = np.zeros_like(self.heightmap, dtype=bool)
                self.city_sites = []
                self.barriers = np.zeros_like(self.heightmap, dtype=bool)
        else:
            # Assume dict from generator
            self.heightmap = data_source["heightmap"]
            self.slope = data_source["slope"]
            self.buildable = data_source["buildable"]
            self.city_sites = data_source["city_sites"]
            self.barriers = data_source["barriers"]
    
    def show_all(self, save_path=None):
        """Complete dashboard view"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Heightmap (3D-like shading)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.heightmap, cmap='terrain', interpolation='bilinear')
        ax1.set_title('Elevation (Terrain)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Slope/Steepness (Warith needs this for road calculation)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.slope, cmap='hot', interpolation='bilinear')
        ax2.set_title('Slope (Red = Cliffs/Barriers)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Buildable Areas (Green = Flat for cities)
        ax3 = fig.add_subplot(gs[0, 2])
        build_viz = np.zeros((*self.buildable.shape, 3))
        build_viz[self.buildable] = [0.2, 0.8, 0.2]  # Green
        build_viz[self.barriers] = [0.8, 0.2, 0.2]   # Red barriers
        ax3.imshow(build_viz)
        ax3.set_title('Buildable Zones (Green=Flat, Red=Cliffs)', fontsize=12, fontweight='bold')
        
        # 4. City Sites Overlay
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(self.heightmap, cmap='gray', alpha=0.5)
        ax4.imshow(self.buildable, cmap='Greens', alpha=0.3)
        
        # Mark city sites
        for site in self.city_sites:
            circle = plt.Circle((site["x"], site["y"]), site["radius"], 
                              fill=False, color='red', linewidth=2)
            ax4.add_patch(circle)
            ax4.plot(site["x"], site["y"], 'r+', markersize=10)
            ax4.annotate(f"Q:{site['quality']:.2f}", 
                        (site["x"], site["y"]), 
                        xytext=(5, 5), textcoords='offset points',
                        color='yellow', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax4.set_title(f'City Sites ({len(self.city_sites)} found)', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, self.heightmap.shape[1])
        ax4.set_ylim(self.heightmap.shape[0], 0)  # Flip Y to match array coords
        
        # 5. Cross-section (elevation profile through middle)
        ax5 = fig.add_subplot(gs[1, 1])
        mid_y = self.heightmap.shape[0] // 2
        profile = self.heightmap[mid_y, :]
        ax5.fill_between(range(len(profile)), profile, alpha=0.3)
        ax5.plot(profile, linewidth=2, color='brown')
        ax5.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Valley floor')
        ax5.set_title('Cross-Section (Elevation Profile)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Distance')
        ax5.set_ylabel('Height')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Height histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(self.heightmap.flatten(), bins=20, color='skyblue', edgecolor='black')
        ax6.axvline(x=0.3, color='r', linestyle='--', label='Valley threshold')
        ax6.set_title('Elevation Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Height')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        plt.suptitle('Dwarf Fortress Style Terrain Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def show_3d(self, elevation_scale=50):
        """3D wireframe view"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X = np.arange(self.heightmap.shape[1])
        Y = np.arange(self.heightmap.shape[0])
        X, Y = np.meshgrid(X, Y)
        
        surf = ax.plot_surface(X, Y, self.heightmap * elevation_scale, 
                              cmap='terrain', alpha=0.9, 
                              linewidth=0, antialiased=True)
        
        ax.set_title('3D Terrain View')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "terrain_data.json"
    
    print(f"Visualizing {file}...")
    viz = TerrainVisualizer(file)
    viz.show_all(save_path="terrain_analysis.png")
    # viz.show_3d()  # Uncomment for 3D view (slower)