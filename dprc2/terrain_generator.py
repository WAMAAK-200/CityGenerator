import numpy as np
from config import TerrainConfig, TerrainType
from noise_utils import NoiseGenerator

class TerrainLayer:
    """Pure terrain data container"""
    def __init__(self, heightmap: np.ndarray, max_height: float):
        self.heightmap = heightmap
        self.max_height = max_height
        self.size = heightmap.shape[0]

class TerrainGenerator:
    def __init__(self, config: TerrainConfig):
        self.config = config
        self.noise_gen = NoiseGenerator(config.seed)
        
    def generate(self) -> TerrainLayer:
        """Generate terrain only - no water"""
        size = self.config.size
        h = np.zeros((size, size))
        
        if self.config.terrain_type == TerrainType.PLAINS:
            h = self._gen_plains(size)
        elif self.config.terrain_type == TerrainType.MOUNTAINS:
            h = self._gen_mountains(size)
        elif self.config.terrain_type == TerrainType.COASTAL:
            h = self._gen_coastal(size)
        elif self.config.terrain_type == TerrainType.PLATEAU:
            h = self._gen_plateau(size)
        elif self.config.terrain_type == TerrainType.VALLEY:
            h = self._gen_valley(size)
        elif self.config.terrain_type == TerrainType.ISLAND:
            h = self._gen_island(size)
        elif self.config.terrain_type == TerrainType.CANYON:
            h = self._gen_canyon(size)
        elif self.config.terrain_type == TerrainType.HILLS:
            h = self._gen_hills(size)
        
        h = (h - h.min()) / (h.max() - h.min())
        h = self._thermal_erosion(h)
        return TerrainLayer(h, self.config.max_height)
    
    def _gen_plains(self, size: int) -> np.ndarray:
        """Very flat, gentle rolling"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                n = self.noise_gen.perlin(i, j, scale=self.config.scale, octaves=4)
                h[i,j] = n * 0.2 + 0.5
        return h
    
    def _gen_mountains(self, size: int) -> np.ndarray:
        """Ridged peaks with detail"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                r = self.noise_gen.ridged(i, j, scale=self.config.scale*0.5)
                d = self.noise_gen.perlin(i, j, scale=self.config.scale*0.2) * 0.3
                h[i,j] = r**2 + d
        return h
    
    def _gen_coastal(self, size: int) -> np.ndarray:
        """Shoreline gradient with beach"""
        h = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center*0.7)**2 + (j - center)**2) / (size * 0.8)
                n = self.noise_gen.perlin(i, j, scale=self.config.scale) * 0.1
                h[i,j] = (1.0 - dist) * 0.8 + n
        return h
    
    def _gen_plateau(self, size: int) -> np.ndarray:
        """Flat top with steep cliffs"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                n = self.noise_gen.perlin(i, j, scale=self.config.scale)
                if n > 0.6:
                    h[i,j] = 0.8 + self.noise_gen.perlin(i, j, scale=10) * 0.05
                else:
                    h[i,j] = n * 0.5
        return h
    
    def _gen_valley(self, size: int) -> np.ndarray:
        """V-shaped trough"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                rad = np.radians(45)
                x = i - size/2
                y = j - size/2
                rx = x * np.cos(rad) + y * np.sin(rad)
                ry = -x * np.sin(rad) + y * np.cos(rad)
                along = self.noise_gen.perlin(rx, 0, scale=self.config.scale*2) * 0.15
                cross = abs(ry) / (size * 0.3)
                h[i,j] = 0.5 + along - cross * 0.6
        return h
    
    def _gen_island(self, size: int) -> np.ndarray:
        """Volcanic cone with beaches"""
        h = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i-center)**2 + (j-center)**2) / (size/2)
                if dist > 1.0:
                    h[i,j] = -0.2
                else:
                    cone = max(0, 1.0 - dist**1.5)
                    detail = self.noise_gen.ridged(i, j, scale=15) * 0.15
                    h[i,j] = cone + detail
        return h
    
    def _gen_canyon(self, size: int) -> np.ndarray:
        """Sharp cuts and ridges"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # Base height
                base = self.noise_gen.perlin(i, j, scale=self.config.scale)
                # Canyon cuts (absolute value creates cliffs)
                cuts = abs(self.noise_gen.perlin(i, j, scale=self.config.scale*0.3)) * 2 - 1
                h[i,j] = base * 0.3 + cuts * 0.5
        return h
    
    def _gen_hills(self, size: int) -> np.ndarray:
        """Rolling rounded hills"""
        h = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                n1 = self.noise_gen.perlin(i, j, scale=self.config.scale)
                n2 = self.noise_gen.perlin(i, j, scale=self.config.scale*0.5) * 0.5
                h[i,j] = (n1 + n2) * 0.4 + 0.5
        return h
    
    def _thermal_erosion(self, h: np.ndarray) -> np.ndarray:
        size = h.shape[0]
        for _ in range(2):
            new_h = h.copy()
            for i in range(1, size-1):
                for j in range(1, size-1):
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        diff = h[i,j] - h[i+di,j+dj]
                        if diff > 0.05:
                            move = diff * 0.05
                            new_h[i,j] -= move
                            new_h[i+di,j+dj] += move
            h = new_h
        return h