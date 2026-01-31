import numpy as np
from noise import pnoise2, snoise2
from enum import Enum
from typing import Dict
from dataclasses import dataclass

class Biome(Enum):
    MESA = "mesa"
    MOUNTAINS = "mountains" 
    VOLCANO = "volcano"
    PLATEAU = "plateau"
    VALLEY = "valley"
    PLAINS = "plains"
    ROLLING_HILLS = "rolling_hills"
    CLIFF = "cliff"

@dataclass
class TerrainData:
    heightmap: np.ndarray
    biome: str
    seed: int
    size: int

class TerrainGenerator:
    def __init__(self, size: int = 512, seed: int = 42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)
        
    def _noise(self, freq: float, octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        """Fractal Brownian Motion noise"""
        result = np.zeros((self.size, self.size))
        amplitude = 1.0
        frequency = freq
        
        for i in range(octaves):
            # Grid coordinates
            x = np.arange(self.size) * frequency
            y = np.arange(self.size) * frequency
            X, Y = np.meshgrid(x, y)
            
            # Simplex noise (smoother than Perlin)
            vec = np.vectorize(lambda x, y: snoise2(x, y, base=self.seed + i))
            result += vec(X, Y) * amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
            
        return result
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range"""
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    
    def _ridge(self, noise: np.ndarray, sharpness: float = 2.0) -> np.ndarray:
        """Convert noise to ridges (1 - abs(noise))^sharpness"""
        return np.power(1.0 - np.abs(noise), sharpness)
    
    def generate(self, biome: Biome) -> TerrainData:
        """Generate terrain for specific biome"""
        if biome == Biome.MESA:
            height = self._gen_mesa()
        elif biome == Biome.MOUNTAINS:
            height = self._gen_mountains()
        elif biome == Biome.VOLCANO:
            height = self._gen_volcano()
        elif biome == Biome.PLATEAU:
            height = self._gen_plateau()
        elif biome == Biome.VALLEY:
            height = self._gen_valley()
        elif biome == Biome.PLAINS:
            height = self._gen_plains()
        elif biome == Biome.ROLLING_HILLS:
            height = self._gen_rolling_hills()
        elif biome == Biome.CLIFF:
            height = self._gen_cliff()
        else:
            raise ValueError(f"Unknown biome: {biome}")
            
        return TerrainData(
            heightmap=height.astype(np.float32),
            biome=biome.value,
            seed=self.seed,
            size=self.size
        )
    
    def _gen_mesa(self) -> np.ndarray:
        """Stratified flat layers with steep cliffs between"""
        base = self._noise(freq=0.008, octaves=6, persistence=0.6)
        base = self._normalize(base)
        # Quantize to create terraces
        levels = 5
        stepped = np.floor(base * levels) / levels
        # Add slight variation within layers for texture
        detail = self._noise(freq=0.04, octaves=2) * 0.02
        return stepped * 0.6 + 0.2 + detail
    
    def _gen_mountains(self) -> np.ndarray:
        """Sharp alpine peaks with valleys"""
        # Ridged multifractal
        base = self._noise(freq=0.012, octaves=8, persistence=0.5)
        ridges = self._ridge(base, sharpness=3.0)
        # Erosion details
        detail = self._noise(freq=0.03, octaves=3) * 0.15
        height = ridges * 0.7 + detail
        # Bias toward high elevation (valleys are exception, not rule)
        return np.power(height, 0.6) * 0.8 + 0.2
    
    def _gen_volcano(self) -> np.ndarray:
        """Single cone mountain with possible crater"""
        cx, cy = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Cone shape
        max_dist = self.size * 0.45
        cone = 1.0 - (dist / max_dist)
        cone = np.clip(cone, 0, 1)
        # Steepness
        cone = np.power(cone, 0.7)
        
        # Noise for lava flows/asymmetry
        asym = self._noise(freq=0.015, octaves=4) * 0.1
        
        height = cone * 0.8 + asym + 0.1
        
        # Crater depression in center
        crater_radius = self.size * 0.08
        crater_mask = dist < crater_radius
        height[crater_mask] *= 0.6
        
        return height
    
    def _gen_plateau(self) -> np.ndarray:
        """High flat table with eroded/scarred edges"""
        # Base elevation (high)
        base = 0.7 + self._noise(freq=0.01, octaves=4) * 0.15
        
        # Distance from center for erosion
        cx, cy = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Edge falloff
        edge_start = self.size * 0.35
        edge_end = self.size * 0.45
        t = np.clip((dist - edge_start) / (edge_end - edge_start), 0, 1)
        erosion = np.power(t, 2)  # Smooth curve down
        
        return base * (1 - erosion * 0.9)
    
    def _gen_valley(self) -> np.ndarray:
        """U-shaped glacial valley with flat floor"""
        # Create meandering centerline using noise
        t = np.linspace(0, 4*np.pi, self.size)
        meander = self._noise(freq=0.008, octaves=2)[:, int(self.size/2)] * self.size * 0.2
        center_y = self.size//2 + meander
        
        # Cross-section profile
        y_idx = np.arange(self.size)[:, None]
        dist_from_center = np.abs(y_idx - center_y[None, :])
        
        # Normalize distance
        width = self.size // 3
        d = np.clip(dist_from_center / width, 0, 1)
        
        # U-shape function: d^0.3 (flat bottom, steep sides)
        profile = np.power(d, 0.35)
        height = 0.8 - profile * 0.6
        
        # Add lateral noise for irregular valley walls
        wall_noise = self._noise(freq=0.02, octaves=3) * 0.05
        height += wall_noise * d  # More noise on walls than floor
        
        return np.clip(height, 0, 1)
    
    def _gen_plains(self) -> np.ndarray:
        """Extremely flat, minimal variation"""
        base = self._noise(freq=0.005, octaves=2, persistence=0.4)
        # Suppress amplitude heavily
        return 0.4 + base * 0.05
    
    def _gen_rolling_hills(self) -> np.ndarray:
        """Gentle, rounded hills suitable for grazing/farms"""
        base = self._noise(freq=0.01, octaves=4, persistence=0.5)
        # Round the hills (power < 1 pushes values toward middle)
        rounded = np.sign(base) * np.abs(base) ** 0.7
        # Gentle amplitude
        return 0.45 + rounded * 0.2
    
    def _gen_cliff(self) -> np.ndarray:
        """Linear fault/escarpment"""
        angle = np.radians(self.seed % 180)  # Random angle from seed
        Y, X = np.ogrid[:self.size, :self.size]
        
        # Coordinate along cliff direction
        projected = X * np.cos(angle) + Y * np.sin(angle)
        normalized = (projected / self.size)
        
        # Displacement for natural look
        displacement = self._noise(freq=0.015, octaves=3) * 0.1
        
        # Step function with noise
        pos = normalized + displacement
        
        # Two levels: lowland and highland
        cliff_pos = 0.5
        height = np.where(pos < cliff_pos, 
                         0.2 + pos * 0.2,  # Low side
                         0.6 + (pos - cliff_pos) * 0.8)  # High side
        
        return height

def generate_biome(biome_name: str, size: int = 512, seed: int = 42) -> TerrainData:
    """Convenience function"""
    biome = Biome(biome_name.lower())
    gen = TerrainGenerator(size=size, seed=seed)
    return gen.generate(biome)

if __name__ == "__main__":
    # Test generate all
    gen = TerrainGenerator(size=256, seed=42)
    for biome in Biome:
        data = gen.generate(biome)
        print(f"{biome.value}: range [{data.heightmap.min():.2f}, {data.heightmap.max():.2f}]")