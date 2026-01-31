import numpy as np
from typing import List, Tuple
from terrain_generator import TerrainLayer

class HydrologyLayer:
    """Separate water layer container"""
    def __init__(self, size: int):
        self.water_mask = np.zeros((size, size))  # 0 = land, 1 = water
        self.river_paths: List[List[Tuple[int, int]]] = []
        self.size = size

class HydrologyGenerator:
    """Generates water features independently of terrain"""
    def __init__(self, seed: int = 42, river_count: int = 2, 
                 min_river_length: int = 20, water_level: float = 0.25):
        self.seed = seed
        self.river_count = river_count
        self.min_river_length = min_river_length
        self.water_level = water_level
        self.rng = np.random.RandomState(seed)
        
    def generate(self, terrain: TerrainLayer) -> HydrologyLayer:
        """Generate water based on terrain, but keep separate"""
        layer = HydrologyLayer(terrain.size)
        sources = self._find_sources(terrain)
        
        for source in sources[:self.river_count]:
            path = self._carve_river(source, terrain)
            if len(path) > self.min_river_length:
                layer.river_paths.append(path)
                self._add_to_mask(layer, path)
        
        return layer
    
    def _find_sources(self, terrain: TerrainLayer) -> List[Tuple[int, int]]:
        """Find high points for rivers"""
        h = terrain.heightmap
        size = terrain.size
        sources = []
        
        for i in range(3, size-3, 8):
            for j in range(3, size-3, 8):
                if h[i,j] > 0.6:
                    window = h[i-1:i+2, j-1:j+2]
                    if h[i,j] == window.max():
                        sources.append((i,j))
        
        sources.sort(key=lambda p: h[p[0], p[1]], reverse=True)
        return sources
    
    def _carve_river(self, start: Tuple[int, int], 
                    terrain: TerrainLayer) -> List[Tuple[int, int]]:
        """Pathfinding for river"""
        path = [start]
        x, y = start
        visited = set()
        size = terrain.size
        
        for _ in range(500):
            visited.add((x, y))
            neighbors = []
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and (nx,ny) not in visited:
                    diff = terrain.heightmap[x,y] - terrain.heightmap[nx,ny]
                    if diff > 0:
                        neighbors.append((diff, nx, ny))
            
            if not neighbors:
                break
                
            neighbors.sort(reverse=True)
            if len(neighbors) > 1 and self.rng.random() > 0.3:
                _, x, y = neighbors[self.rng.randint(0, min(2, len(neighbors)))]
            else:
                _, x, y = neighbors[0]
                
            path.append((x, y))
            if terrain.heightmap[x,y] < self.water_level:
                break
                
        return path
    
    def _add_to_mask(self, layer: HydrologyLayer, 
                    path: List[Tuple[int, int]]):
        """Add river to mask"""
        for idx, (x,y) in enumerate(path):
            width = 1 if idx < len(path) * 0.7 else 2
            for dx in range(-width, width+1):
                for dy in range(-width, width+1):
                    if dx*dx + dy*dy <= width*width:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < layer.size and 0 <= ny < layer.size:
                            layer.water_mask[nx, ny] = 1.0