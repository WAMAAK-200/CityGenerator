import numpy as np
import noise

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
        n = noise.pnoise2(
            x / scale, y / scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base=self.seed
        )
        return 1.0 - abs(n)