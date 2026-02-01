import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

class TerrainVisualizer:
    def __init__(self, terrain_gen, hydro_gen):
        self.terrain = terrain_gen
        self.hydro = hydro_gen
        self.config = terrain_gen.config
        self.fig = None
        self.axs = None
        
    def show(self, mode='2d'):
        if mode == '2d':
            self._show_2d()
        elif mode == '3d':
            self._show_3d()
        elif mode == 'analysis':
            self._show_analysis()
            
    def _show_2d(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 12))
        h = self.terrain.heightmap
        w = self.hydro.water_mask
        s = self.terrain.slope_map
        extent = [0, 1000, 0, 1000]
        
        ax1 = self.axs[0, 0]
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(h, plt.cm.terrain, blend_mode='overlay')
        ax1.imshow(rgb, extent=extent)
        ax1.set_title('Terrain (Hillshade)')
        ax1.set_xlabel('Meters')
        ax1.set_ylabel('Meters')
        
        ax2 = self.axs[0, 1]
        im2 = ax2.imshow(h, cmap='terrain', extent=extent)
        river_overlay = np.zeros((*h.shape, 4))
        river_overlay[w > 0] = [0.2, 0.4, 0.8, 0.8]
        ax2.imshow(river_overlay, extent=extent)
        ax2.set_title('Elevation + Rivers')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = self.axs[1, 0]
        im3 = ax3.imshow(s, cmap='hot', extent=extent)
        ax3.set_title('Slope (Buildability)')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        ax4 = self.axs[1, 1]
        X, Y = np.meshgrid(np.linspace(0, 1000, h.shape[0]), 
                           np.linspace(0, 1000, h.shape[1]))
        cs = ax4.contour(X, Y, h, levels=10, colors='black', alpha=0.5)
        ax4.clabel(cs, inline=True, fontsize=8)
        ax4.imshow(w, cmap='Blues', alpha=0.6, extent=extent)
        ax4.set_title('Contours + Hydrology')
        
        plt.tight_layout()
        plt.show()
    
    def _show_3d(self):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        size = self.config.size
        X = np.linspace(0, 1000, size)
        Y = np.linspace(0, 1000, size)
        X, Y = np.meshgrid(X, Y)
        Z = self.terrain.heightmap * self.config.max_height
        
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.9)
        
        water_level = self.config.water_level * self.config.max_height
        ax.contourf(X, Y, Z, levels=[0, water_level], 
                   colors=['#4444ff'], alpha=0.3, zdir='z', offset=0)
        
        if self.hydro.river_paths:
            for path in self.hydro.river_paths:
                xs = [p[0] * (1000/size) for p in path]
                ys = [p[1] * (1000/size) for p in path]
                zs = [self.terrain.get_height_at(p[0], p[1]) * 
                      self.config.max_height + 0.5 for p in path]
                ax.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Meters')
        ax.set_ylabel('Meters')
        ax.set_zlabel('Height (m)')
        ax.set_title(f'3D Terrain - {self.config.terrain_type.value}')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
    
    def _show_analysis(self):
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        h = self.terrain.heightmap
        s = self.terrain.slope_map
        
        axs[0, 0].hist(h.flatten(), bins=50, color='brown', alpha=0.7)
        axs[0, 0].axvline(self.config.water_level, color='blue', 
                         linestyle='--', label='Water Level')
        axs[0, 0].set_title('Elevation Distribution')
        axs[0, 0].legend()
        
        axs[0, 1].hist(s.flatten(), bins=50, color='orange', alpha=0.7)
        axs[0, 1].set_title('Slope Distribution')
        
        sample_idx = np.random.choice(h.size, 1000, replace=False)
        axs[0, 2].scatter(h.flatten()[sample_idx], s.flatten()[sample_idx], 
                         alpha=0.5, s=1)
        axs[0, 2].set_xlabel('Height')
        axs[0, 2].set_ylabel('Slope')
        axs[0, 2].set_title('Height vs Slope')
        
        step = 20
        x = np.arange(0, h.shape[0], step)
        y = np.arange(0, h.shape[1], step)
        X, Y = np.meshgrid(x, y)
        dx = np.gradient(h, axis=0)[::step, ::step]
        dy = np.gradient(h, axis=1)[::step, ::step]
        
        scale_factor = 1000 / self.config.size
        axs[1, 0].imshow(h, cmap='terrain', extent=[0, 1000, 0, 1000])
        axs[1, 0].quiver(X*scale_factor, Y*scale_factor, -dx, -dy, alpha=0.6)
        axs[1, 0].set_title('Gradient Field (Water Flow)')
        
        axs[1, 1].imshow(h, cmap='gray', extent=[0, 1000, 0, 1000])
        if self.hydro.river_paths:
            for path in self.hydro.river_paths:
                xs = [p[0] * scale_factor for p in path]
                ys = [p[1] * scale_factor for p in path]
                axs[1, 1].plot(xs, ys, 'cyan', linewidth=2)
        axs[1, 1].set_title('River Network')
        
        buildable = s < 0.15
        axs[1, 2].imshow(buildable, cmap='Greens', extent=[0, 1000, 0, 1000])
        axs[1, 2].set_title('Buildable Areas (Slope < 0.15)')
        
        plt.tight_layout()
        plt.show()