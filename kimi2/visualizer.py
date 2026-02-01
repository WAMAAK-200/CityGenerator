import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LightSource
from matplotlib.patches import Polygon

class CityVisualizer:
    def __init__(self, terrain, hydrology, city_gen):
        self.terrain = terrain
        self.hydro = hydrology
        self.city = city_gen
        
    def show_city_map(self, show_terrain=True):
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Background
        if show_terrain:
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(self.terrain.heightmap, plt.cm.terrain, blend_mode='overlay')
            ax.imshow(rgb, extent=[0, 1000, 0, 1000], alpha=0.3)
        else:
            ax.set_facecolor('#f0ebe0')  # Parchment
        
        # Streets - thin black lines
        if self.city.streets.streets:
            xs = [p[0] * 3.9 for p in self.city.streets.streets]
            ys = [p[1] * 3.9 for p in self.city.streets.streets]
            ax.scatter(xs, ys, c='#2a2a2a', s=0.8, alpha=0.7, zorder=2)
        
        # Main roads - slightly thicker
        if self.city.streets.main_roads:
            xs = [p[0] * 3.9 for p in self.city.streets.main_roads]
            ys = [p[1] * 3.9 for p in self.city.streets.main_roads]
            ax.scatter(xs, ys, c='#1a1a1a', s=2, alpha=0.9, zorder=3)
        
        # Walls - clean ring only if walls exist
        if self.city.streets.wall_points:
            wall_x = [p[0] * 3.9 for p in sorted(self.city.streets.wall_points, 
                    key=lambda p: np.arctan2(p[1]-128, p[0]-128))]
            wall_y = [p[1] * 3.9 for p in sorted(self.city.streets.wall_points,
                    key=lambda p: np.arctan2(p[1]-128, p[0]-128))]
            if len(wall_x) > 2:
                ax.plot(wall_x + [wall_x[0]], wall_y + [wall_y[0]], 
                       'k-', linewidth=3, alpha=0.9, zorder=4)
        
        # Buildings - POLYGONAL
        for bldg in self.city.placed_buildings:
            try:
                color_val = getattr(bldg.building_type, 'color', [140, 140, 140])
                if not isinstance(color_val, (list, tuple)) or len(color_val) != 3:
                    color_val = [140, 140, 140]
                color = [c/255 for c in color_val]
            except:
                color = [0.55, 0.55, 0.55]
            
            # Transform vertices to world space
            scale = 3.9
            world_verts = []
            for vx, vy in bldg.vertices:
                # Apply rotation
                rad = np.radians(bldg.rotation)
                rx = vx * np.cos(rad) - vy * np.sin(rad)
                ry = vx * np.sin(rad) + vy * np.cos(rad)
                # Translate to position
                world_verts.append((bldg.x * scale + rx * scale, 
                                   bldg.y * scale + ry * scale))
            
            if len(world_verts) >= 3:
                poly = Polygon(world_verts, facecolor=color, 
                              edgecolor='#1a1a1a', linewidth=0.8, 
                              alpha=0.95, zorder=5)
                ax.add_patch(poly)
        
        # Legend
        seen = set()
        legend_elements = []
        for bldg in self.city.placed_buildings:
            btype = bldg.building_type
            if btype.name not in seen:
                seen.add(btype.name)
                try:
                    c = [v/255 for v in getattr(btype, 'color', [140,140,140])]
                except:
                    c = [0.55, 0.55, 0.55]
                legend_elements.append(patches.Patch(facecolor=c, edgecolor='black', label=btype.name))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.02, 1), fontsize=9)
        
        ax.set_xlim(-50, 1050)
        ax.set_ylim(-50, 1050)
        ax.set_aspect('equal')
        ax.set_title('Medieval City Layout', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()