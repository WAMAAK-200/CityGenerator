import numpy as np
import noise
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

class TerrainType(Enum):
    PLAINS = "plains"
    MOUNTAINS = "mountains" 
    COASTAL = "coastal"
    PLATEAU = "plateau"
    VALLEY = "valley"
    ISLAND = "island"

@dataclass
class TerrainConfig:
    seed: int = 42
    size: int = 256
    terrain_type: TerrainType = TerrainType.MOUNTAINS
    water_level: float = 0.25
    max_height: float = 100.0  # meters
    # CRITICAL FIX: Reduced scale for city-level detail (1km / 256 = ~4m per pixel)
    scale: float = 25.0  # Was 100.0 - way too big for a city!
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    river_count: int = 2  # Fewer rivers for city scale
    min_river_length: int = 20

class NoiseGenerator:
    def __init__(self, seed: int):
        self.seed = seed
    
    def perlin(self, x: float, y: float, scale: float = 25.0, 
               octaves: int = 6, persistence: float = 0.5, 
               lacunarity: float = 2.0) -> float:
        return noise.pnoise2(
            x / scale, y / scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base=self.seed
        )
    
    def ridged(self, x: float, y: float, scale: float = 25.0,
               octaves: int = 6, persistence: float = 0.5,
               lacunarity: float = 2.0) -> float:
        n = noise.pnoise2(x / scale, y / scale, octaves=octaves, 
                         persistence=persistence, lacunarity=lacunarity, base=self.seed)
        return 1.0 - abs(n)
    
    def valley_noise(self, x: float, y: float, direction: float = 0.0) -> float:
        rad = np.radians(direction)
        nx = x * np.cos(rad) - y * np.sin(rad)
        ny = x * np.sin(rad) + y * np.cos(rad)
        return noise.pnoise2(nx / 40, ny / 15, octaves=4, base=self.seed)  # Adjusted for city scale

class TerrainGenerator:
    def __init__(self, config: TerrainConfig):
        self.config = config
        self.noise_gen = NoiseGenerator(config.seed)
        self.heightmap = None
        self.slope_map = None
        
    def generate(self) -> np.ndarray:
        size = self.config.size
        h = np.zeros((size, size))
        scale = self.config.scale
        
        if self.config.terrain_type == TerrainType.MOUNTAINS:
            for i in range(size):
                for j in range(size):
                    # Multiple octaves for detail
                    r = self.noise_gen.ridged(i, j, scale=scale * 0.5, octaves=self.config.octaves)
                    d = self.noise_gen.perlin(i, j, scale=scale * 0.2) * 0.3
                    h[i, j] = r ** 2 + d
                    
        elif self.config.terrain_type == TerrainType.PLAINS:
            for i in range(size):
                for j in range(size):
                    # Very subtle variation for plains
                    n = self.noise_gen.perlin(i, j, scale=scale, octaves=4)  # Fewer octaves = smoother
                    h[i, j] = n * 0.3 + 0.5  # Reduced amplitude
                    
        elif self.config.terrain_type == TerrainType.COASTAL:
            center = size // 2
            for i in range(size):
                for j in range(size):
                    # Shoreline closer to edge for city scale
                    dist = np.sqrt((i - center*0.7)**2 + (j - center)**2) / (size * 0.8)
                    n = self.noise_gen.perlin(i, j, scale=scale * 0.5) * 0.15
                    h[i, j] = (1.0 - dist) * 0.8 + n
                    
        elif self.config.terrain_type == TerrainType.ISLAND:
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i-center)**2 + (j-center)**2) / (size/2)
                    if dist > 1.0:
                        h[i, j] = -0.2
                    else:
                        cone = 1.0 - dist
                        h[i, j] = cone ** 3 + self.noise_gen.ridged(i, j, scale=scale*0.8) * 0.2
                        
        elif self.config.terrain_type == TerrainType.PLATEAU:
            for i in range(size):
                for j in range(size):
                    n = self.noise_gen.perlin(i, j, scale=scale)
                    if n > 0.6:
                        h[i, j] = 0.7 + self.noise_gen.perlin(i, j, scale=scale*0.3) * 0.1
                    else:
                        h[i, j] = n * 0.4
                        
        elif self.config.terrain_type == TerrainType.VALLEY:
            for i in range(size):
                for j in range(size):
                    v = self.noise_gen.valley_noise(i, j, 45)
                    cross = abs(noise.pnoise2(i/scale, j/(scale*0.3), base=self.config.seed))
                    h[i, j] = v * 0.4 + cross * 0.3
        
        h = (h - h.min()) / (h.max() - h.min())
        h = self._thermal_erosion(h)
        self.heightmap = h
        self._calculate_slope()
        return h
    
    def _thermal_erosion(self, h: np.ndarray) -> np.ndarray:
        size = h.shape[0]
        for _ in range(2):  # Reduced iterations for city scale
            new_h = h.copy()
            for i in range(1, size-1):
                for j in range(1, size-1):
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        diff = h[i, j] - h[i+di, j+dj]
                        if diff > 0.05:  # Lower threshold for finer detail
                            move = diff * 0.05
                            new_h[i, j] -= move
                            new_h[i+di, j+dj] += move
            h = new_h
        return h
    
    def _calculate_slope(self):
        dx = np.gradient(self.heightmap, axis=0)
        dy = np.gradient(self.heightmap, axis=1)
        self.slope_map = np.sqrt(dx**2 + dy**2)

class HydrologySystem:
    def __init__(self, terrain: TerrainGenerator, config: TerrainConfig):
        self.terrain = terrain
        self.config = config
        self.river_paths = []
        self.water_mask = np.zeros((config.size, config.size))
        
    def generate(self):
        sources = self._find_sources()
        for source in sources[:self.config.river_count]:
            path = self._carve_river(source)
            if len(path) > self.config.min_river_length:
                self.river_paths.append(path)
                self._apply_river(path)
        return self.water_mask
    
    def _find_sources(self):
        size = self.config.size
        h = self.terrain.heightmap
        sources = []
        # Denser sampling for city scale
        for i in range(3, size-3, 8):
            for j in range(3, size-3, 8):
                if h[i, j] > 0.5:  # Lower threshold for city hills
                    window = h[i-1:i+2, j-1:j+2]
                    if h[i, j] == window.max():
                        sources.append((i, j))
        sources.sort(key=lambda p: h[p[0], p[1]], reverse=True)
        return sources
    
    def _carve_river(self, start):
        path = [start]
        x, y = start
        visited = set()
        size = self.config.size
        
        for _ in range(500):  # Shorter max length for city
            visited.add((x, y))
            neighbors = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                    diff = self.terrain.heightmap[x, y] - self.terrain.heightmap[nx, ny]
                    if diff > 0:
                        neighbors.append((diff, nx, ny))
            
            if not neighbors:
                break
            neighbors.sort(reverse=True)
            _, x, y = neighbors[0] if np.random.random() > 0.3 else neighbors[np.random.randint(0, min(2, len(neighbors)))]
            path.append((x, y))
            if self.terrain.heightmap[x, y] < self.config.water_level:
                break
        return path
    
    def _apply_river(self, path):
        for idx, (x, y) in enumerate(path):
            width = 1 if idx < len(path) * 0.7 else 2  # Wider at mouth
            for dx in range(-width, width+1):
                for dy in range(-width, width+1):
                    if dx*dx + dy*dy <= width*width:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < self.config.size and 0 <= ny < self.config.size:
                            self.water_mask[nx, ny] = 1.0

class TerrainVisualizer:
    def __init__(self, terrain_gen, hydro_gen):
        self.terrain = terrain_gen
        self.hydro = hydro_gen
        # BUG FIX: Store config reference
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
        size = self.config.size
        meters = 1000  # 1km
        
        # 1. Hillshade
        ax1 = self.axs[0, 0]
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(h, plt.cm.terrain, blend_mode='overlay')
        ax1.imshow(rgb, extent=[0, meters, 0, meters])
        ax1.set_title('City Terrain (Hillshade)')
        ax1.set_xlabel('Meters')
        ax1.set_ylabel('Meters')
        
        # 2. Elevation + Rivers
        ax2 = self.axs[0, 1]
        im2 = ax2.imshow(h, cmap='terrain', extent=[0, meters, 0, meters])
        river_overlay = np.zeros((*h.shape, 4))
        river_overlay[w > 0] = [0.2, 0.4, 0.8, 0.8]
        ax2.imshow(river_overlay, extent=[0, meters, 0, meters])
        ax2.set_title(f'Elevation (Max: {self.config.max_height}m)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, label='Height (0-1)')
        
        # 3. Slope
        ax3 = self.axs[1, 0]
        im3 = ax3.imshow(s, cmap='hot', extent=[0, meters, 0, meters])
        ax3.set_title('Slope - Buildable Areas')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='Gradient')
        
        # 4. Contours
        ax4 = self.axs[1, 1]
        X, Y = np.meshgrid(np.linspace(0, meters, size), np.linspace(0, meters, size))
        cs = ax4.contour(X, Y, h, levels=8, colors='black', alpha=0.6)
        ax4.clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
        ax4.imshow(w, cmap='Blues', alpha=0.5, extent=[0, meters, 0, meters])
        ax4.set_title('Topographic Map + Rivers')
        ax4.set_xlabel('Meters')
        
        plt.tight_layout()
        plt.show()
    
    def _show_3d(self):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        size = self.config.size
        meters = 1000
        X = np.linspace(0, meters, size)
        Y = np.linspace(0, meters, size)
        X, Y = np.meshgrid(X, Y)
        Z = self.terrain.heightmap * self.config.max_height
        
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.9, 
                              linewidth=0, antialiased=True)
        
        # Water plane
        water_z = self.config.water_level * self.config.max_height
        ax.contourf(X, Y, Z, levels=[0, water_z], colors=['#4444ff'], 
                   alpha=0.4, zdir='z', offset=0)
        
        ax.set_xlabel('Meters')
        ax.set_ylabel('Meters')
        ax.set_zlabel('Height (m)')
        ax.set_title(f'3D City Terrain - {self.config.terrain_type.value}')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,0.3])  # Flatter for terrain
        
        plt.show()
    
    def _show_analysis(self):
        """BUG FIX: This was missing self.config reference"""
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        
        h = self.terrain.heightmap
        s = self.terrain.slope_map
        
        # 1. Elevation histogram
        axs[0, 0].hist(h.flatten(), bins=30, color='brown', alpha=0.7, edgecolor='black')
        axs[0, 0].axvline(self.config.water_level, color='blue', linestyle='--', 
                         label=f'Water Level ({self.config.water_level})')
        axs[0, 0].set_title('Elevation Distribution')
        axs[0, 0].set_xlabel('Normalized Height')
        axs[0, 0].legend()
        
        # 2. Slope distribution
        axs[0, 1].hist(s.flatten(), bins=30, color='orange', alpha=0.7, edgecolor='black')
        axs[0, 1].axvline(0.1, color='red', linestyle='--', label='Buildable Threshold')
        axs[0, 1].set_title('Slope Distribution')
        axs[0, 1].set_xlabel('Slope Gradient')
        axs[0, 1].legend()
        
        # 3. Height vs Slope
        sample_idx = np.random.choice(h.size, 2000, replace=False)
        axs[0, 2].scatter(h.flatten()[sample_idx], s.flatten()[sample_idx], 
                         alpha=0.4, s=1, c='green')
        axs[0, 2].set_xlabel('Height')
        axs[0, 2].set_ylabel('Slope')
        axs[0, 2].set_title('Height vs Slope')
        axs[0, 2].grid(True, alpha=0.3)
        
        # 4. Gradient field (subsampled)
        step = 16
        x = np.arange(0, h.shape[0], step)
        y = np.arange(0, h.shape[1], step)
        X, Y = np.meshgrid(x, y)
        dx = np.gradient(h, axis=0)[::step, ::step]
        dy = np.gradient(h, axis=1)[::step, ::step]
        
        axs[1, 0].imshow(h, cmap='terrain', extent=[0, 1000, 0, 1000])
        axs[1, 0].quiver(X*3.9, Y*3.9, -dx, -dy, alpha=0.6, scale=5)  # Scale to meters
        axs[1, 0].set_title('Water Flow Direction')
        axs[1, 0].set_xlabel('Meters')
        
        # 5. River network
        axs[1, 1].imshow(h, cmap='gray', extent=[0, 1000, 0, 1000])
        if self.hydro.river_paths:
            for path in self.hydro.river_paths:
                xs = [p[0]*3.9 for p in path]  # Convert to meters
                ys = [p[1]*3.9 for p in path]
                axs[1, 1].plot(xs, ys, 'cyan', linewidth=2)
        axs[1, 1].set_title('River Network')
        axs[1, 1].set_xlabel('Meters')
        
        # 6. Buildable mask
        buildable = s < 0.15  # Slightly higher threshold for cities
        axs[1, 2].imshow(buildable, cmap='RdYlGn', extent=[0, 1000, 0, 1000])
        build_percent = np.sum(buildable) / buildable.size * 100
        axs[1, 2].set_title(f'Buildable Area: {build_percent:.1f}%')
        axs[1, 2].set_xlabel('Meters')
        
        plt.tight_layout()
        plt.show()

def main():
    print("City Terrain Generator (1km²)")
    print("=" * 40)
    
    config = TerrainConfig(
        seed=123,
        size=256,  # 256x256 = ~4m resolution
        terrain_type=TerrainType.MOUNTAINS,
        water_level=0.3,
        max_height=60.0,  # 60m height difference max for city
        scale=25.0,  # CRITICAL: Small scale for city features
        river_count=2
    )
    
    print(f"Generating {config.terrain_type.value}...")
    terrain = TerrainGenerator(config)
    terrain.generate()
    
    print("Adding hydrology...")
    hydro = HydrologySystem(terrain, config)
    hydro.generate()
    
    viz = TerrainVisualizer(terrain, hydro)
    
    while True:
        print("\nOptions:")
        print("1. 2D Overview (with meter scale)")
        print("2. 3D View")
        print("3. Technical Analysis (FIXED)")
        print("4. Change terrain type")
        print("q. Quit")
        
        choice = input("> ").strip()
        
        if choice == '1':
            viz.show('2d')
        elif choice == '2':
            viz.show('3d')
        elif choice == '3':
            viz.show('analysis')
        elif choice == '4':
            print("Types: plains, mountains, coastal, plateau, valley, island")
            t = input("Type: ").strip()
            try:
                config.terrain_type = TerrainType(t)
                terrain = TerrainGenerator(config)
                terrain.generate()
                hydro = HydrologySystem(terrain, config)
                hydro.generate()
                viz = TerrainVisualizer(terrain, hydro)
            except:
                print("Invalid type")
        elif choice == 'q':
            break

if __name__ == "__main__":
    main()