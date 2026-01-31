import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional
from perlin import TerrainData, Biome, TerrainGenerator
from hydro import add_water, HydroData

class Visualizer:
    def __init__(self, terrain: TerrainData, modified_height: np.ndarray, 
                 hydro: Optional[HydroData] = None):
        self.terrain = terrain
        self.orig = terrain.heightmap
        self.height = modified_height
        self.hydro = hydro
        self.size = terrain.size
        
        # Calculate slope on modified terrain
        gy, gx = np.gradient(self.height)
        self.slope = np.clip(np.sqrt(gx**2 + gy**2) * 2, 0, 1)
        
        if hydro:
            self.water_pct = np.sum(hydro.water_mask) / (self.size**2)
            self.bridge_pct = np.sum(hydro.get_bridgeable()) / (self.size**2)
    
    def show(self, save: Optional[str] = None):
        has_water = self.hydro is not None
        
        rows = 3 if has_water else 2
        fig = plt.figure(figsize=(16, 4*rows))
        gs = GridSpec(rows, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Composite view
        ax = fig.add_subplot(gs[0, :2])
        if has_water:
            # Base terrain colors
            norm = (self.height - self.height.min()) / (self.height.max() - self.height.min() + 1e-10)
            rgb = plt.cm.terrain(norm)[:,:,:3]
            
            # Overlay water
            h = self.hydro
            rgb[h.ocean_mask] = [0.0, 0.2, 0.5]      # Deep ocean
            rgb[h.lake_mask] = [0.2, 0.5, 0.9]       # Lake
            rgb[h.river_mask] = [0.3, 0.7, 1.0]      # River
            rgb[h.delta_mask] = [0.5, 0.8, 1.0]      # Delta
            rgb[h.swamp_mask] = [0.4, 0.5, 0.2]      # Swamp
            rgb[h.island_mask] = [0.9, 0.8, 0.4]     # Sand island
            
            ax.imshow(rgb)
            title = f"{self.terrain.biome.upper()} + Water"
        else:
            ax.imshow(self.height, cmap='terrain')
            title = self.terrain.biome.upper()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 2. Bridgeable map
        if has_water:
            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(self.hydro.get_bridgeable(), cmap='RdYlGn')
            ax.set_title(f"Bridgeable\n{self.bridge_pct*100:.1f}% of map")
        
        # 3. Slope
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(self.slope, cmap='hot')
        ax.set_title("Slope (Modified)")
        
        # 4. Modified height
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(self.height, cmap='viridis')
        ax.set_title("Height (Carved)")
        
        # 5. Cross-section comparison
        ax = fig.add_subplot(gs[1, 2])
        mid = self.size // 2
        orig_line = self.orig[mid, :]
        new_line = self.height[mid, :]
        
        ax.plot(orig_line, 'g--', alpha=0.5, label='Original')
        ax.plot(new_line, 'brown', linewidth=2, label='With Water')
        ax.fill_between(range(self.size), new_line, alpha=0.3, color='brown')
        
        if has_water:
            wl = self.hydro.water_level
            ax.axhline(wl, color='blue', linestyle='--', alpha=0.7)
            # Show water presence
            water_line = np.where(self.hydro.water_mask[mid, :], wl, np.nan)
            ax.fill_between(range(self.size), water_line, wl, color='blue', alpha=0.4)
        
        ax.set_title("Cross-Section")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Water breakdown (if present)
        if has_water:
            ax = fig.add_subplot(gs[2, 0])
            viz = np.zeros((*self.size, 3))
            viz[self.hydro.ocean_mask] = [0, 0, 1]
            viz[self.hydro.lake_mask] = [0.3, 0.7, 1]
            viz[self.hydro.river_mask] = [0.5, 0.9, 1]
            viz[self.hydro.swamp_mask] = [0.4, 0.6, 0.2]
            ax.imshow(viz)
            ax.set_title("Water Types")
            
            ax = fig.add_subplot(gs[2, 1])
            ax.imshow(self.hydro.island_mask, cmap='YlOrBr')
            ax.set_title(f"Islands: {np.sum(self.hydro.island_mask)}px")
            
            ax = fig.add_subplot(gs[2, 2])
            ax.axis('off')
            stats = (f"Water: {self.water_pct*100:.1f}%\n"
                    f"Bridgeable: {self.bridge_pct*100:.1f}%\n"
                    f"Rivers: {len(self.hydro.river_paths)}\n"
                    f"Water Level: {self.hydro.water_level:.2f}")
            ax.text(0.1, 0.5, stats, transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved {save}")
        plt.show()


def run(biome: str, size: int = 512):
    """Full pipeline: generate -> add water -> visualize"""
    # Generate base terrain
    gen = TerrainGenerator(size=size, seed=42)
    terrain = gen.generate(Biome(biome))
    
    # Add water layer (modifies height)
    new_height, hydro = add_water(terrain, ocean=True, rivers=True, 
                                  lakes=True, swamps=True, islands=True)
    
    # Visualize combined
    viz = Visualizer(terrain, new_height, hydro)
    viz.show(f"{biome}_water.png")

if __name__ == "__main__":
    import sys
    b = sys.argv[1] if len(sys.argv) > 1 else "valley"
    run(b)