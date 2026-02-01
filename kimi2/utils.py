import numpy as np
import noise

class NoiseGenerator:
    def __init__(self, seed: int):
        self.seed = seed
    
    def perlin(self, x: float, y: float, scale: float = 100.0, 
               octaves: int = 6, persistence: float = 0.5, 
               lacunarity: float = 2.0) -> float:
        """Standard smooth noise"""
        return noise.pnoise2(
            x / scale, y / scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base=self.seed
        )
    
    def ridged(self, x: float, y: float, scale: float = 100.0,
               octaves: int = 6, persistence: float = 0.5,
               lacunarity: float = 2.0) -> float:
        """Ridged noise for mountains (absolute value + inversion)"""
        n = noise.pnoise2(
            x / scale, y / scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            base=self.seed
        )
        # Ridged multifractal
        return 1.0 - abs(n)
    
    def billowy(self, x: float, y: float, scale: float = 100.0,
                octaves: int = 6) -> float:
        """Billowy noise (square of value)"""
        n = noise.pnoise2(x / scale, y / scale, 
                         octaves=octaves, base=self.seed)
        return n * n
    
    def turbulence(self, x: float, y: float, scale: float = 100.0,
                   octaves: int = 6) -> float:
        """Turbulent/swirly noise"""
        n1 = noise.pnoise2(x / scale, y / scale, 
                          octaves=octaves, base=self.seed)
        n2 = noise.pnoise2(x / scale + 5.2, y / scale + 1.3, 
                          octaves=octaves, base=self.seed)
        return np.sin(x / scale + n1) * np.cos(y / scale + n2)
    
    def valley_noise(self, x: float, y: float, direction: float = 0.0) -> float:
        """Directional noise for valleys"""
        # Rotate coordinates
        rad = np.radians(direction)
        nx = x * np.cos(rad) - y * np.sin(rad)
        ny = x * np.sin(rad) + y * np.cos(rad)
        
        n = noise.pnoise2(nx / 150, ny / 50, octaves=4, base=self.seed)
        return n