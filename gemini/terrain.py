import numpy as np
import noise
from scipy.ndimage import gaussian_filter, maximum_filter
import json
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class TerrainModule(ABC):
    """Abstract base for terrain features (cities will inherit later)"""
    @abstractmethod
    def apply(self, heightmap: np.ndarray, params: Dict) -> np.ndarray:
        pass

class TerrainGenerator:
    def __init__(self, size: int = 256, seed: int = 42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)
        
        # Core data layers - [y, x] indexing
        self.elevation = np.zeros((size, size), dtype=np.float32)
        self.water_level = 0.2
        self.terrain_type = "plains"  # Set by generator
        
        # Extension registry (for cities later)
        self.modules = []
        
    def generate(self, method: str = "valley", seed: Optional[int] = None) -> Dict:
        """
        Generate terrain using specified method.
        Methods: beach, valley, mesa, mountains, plateau, cliff
        """
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            
        self.terrain_type = method
        
        # Route to specific generator
        if method == "beach":
            self._gen_beach()
        elif method == "valley":
            self._gen_valley()
        elif method == "mesa":
            self._gen_mesa()
        elif method == "mountains":
            self._gen_mountains()
        elif method == "plateau":
            self._gen_plateau()
        elif method == "cliff":
            self._gen_cliff()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Post-processing (always applied)
        self._normalize()
        self._smooth(sigma=0.8)
        
        # EXTENSION POINT: City modules would hook here later
        # for module in self.modules:
        #     module.apply(self.elevation, {"seed": self.seed})
        
        return self._package_data()
    
    # --- TERRAIN ARCHETYPES ---
    
    def _gen_beach(self):
        """Coastal gradient: flat sand (0.0-0.2), steep cliffs, inland plateau"""
        for y in range(self.size):
            for x in range(self.size):
                n = noise.pnoise2(x/60.0, y/60.0, octaves=4, base=self.seed)
                gradient = (x / self.size) + (n * 0.15)
                
                # Terrace function
                if gradient < 0.3:  # Beach
                    h = gradient * 0.3
                elif gradient < 0.5:  # Cliff transition
                    h = 0.09 + (gradient - 0.3) * 2.0
                else:  # Inland
                    h = 0.5 + (gradient - 0.5) * 0.5
                    
                self.elevation[y, x] = h
        
        self.water_level = 0.05
    
    def _gen_valley(self):
        """U-shaped valley with winding river depression"""
        freq = 0.03 + (self.seed % 5) / 100.0
        river_path = np.zeros(self.size, dtype=int)
        
        for x in range(self.size):
            river_path[x] = int(self.size/2 + np.sin(x * freq) * self.size/3 + 
                               noise.pnoise1(x/50.0, base=self.seed) * 15)
            river_path[x] = np.clip(river_path[x], 5, self.size-5)
        
        for y in range(self.size):
            for x in range(self.size):
                dist = abs(y - river_path[x])
                # Valley cross-section curve
                base_h = np.power(dist / (self.size/2.2), 0.8)
                n = noise.pnoise2(x/40.0, y/40.0, octaves=3, base=self.seed) * 0.08
                self.elevation[y, x] = 1.0 - base_h + n
        
        # Carve river bed
        for x in range(self.size):
            y = river_path[x]
            if 0 <= y < self.size:
                for dy in range(-3, 4):
                    if 0 <= y+dy < self.size:
                        self.elevation[y+dy, x] *= 0.4
        
        self.water_level = 0.12
    
    def _gen_mesa(self):
        """Stratified layers with sharp cliffs"""
        for y in range(self.size):
            for x in range(self.size):
                n1 = noise.pnoise2(x/80.0, y/80.0, octaves=5, base=self.seed)
                ridge = 1.0 - abs(noise.pnoise2(x/30.0, y/30.0, octaves=3, base=self.seed+1))
                self.elevation[y, x] = n1 * 0.6 + ridge * 0.4
        
        self._normalize()
        # Quantize to strata
        layers = 5
        self.elevation = np.floor(self.elevation * layers) / layers
        self.water_level = 0.08
    
    def _gen_mountains(self):
        """Alpine: sharp ridges, high frequency, scattered flat pockets"""
        for y in range(self.size):
            for x in range(self.size):
                n = noise.pnoise2(x/35.0, y/35.0, octaves=8, persistence=0.55, base=self.seed)
                r = 1.0 - abs(noise.pnoise2(x/20.0, y/20.0, octaves=4, base=self.seed+2))
                self.elevation[y, x] = n * 0.3 + r * 0.7
        
        self._normalize()
        self.elevation = np.power(self.elevation, 0.35)  # Bias toward high
        
        # Create flat pockets (for future cities)
        low_mask = self.elevation < 0.25
        if np.any(low_mask):
            self.elevation[low_mask] *= 0.6  # Flatten valleys
        
        self.water_level = 0.15
    
    def _gen_plateau(self):
        """High flat table with eroded edges"""
        # Base high
        for y in range(self.size):
            for x in range(self.size):
                n = noise.pnoise2(x/70.0, y/70.0, octaves=4, base=self.seed)
                self.elevation[y, x] = 0.4 + n * 0.6  # Start at 0.4-1.0
        
        # Edge erosion
        cx, cy = self.size/2, self.size/2
        for y in range(self.size):
            for x in range(self.size):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                max_dist = self.size/2 * 0.85
                
                if dist > max_dist * 0.5:
                    drop = np.power((dist - max_dist*0.5) / (max_dist*0.5), 2) * 0.9
                    self.elevation[y, x] -= drop
        
        self.elevation = np.clip(self.elevation, 0, 1)
        self.water_level = 0.25
    
    def _gen_cliff(self):
        """Linear cliff with terraces"""
        angle = (self.seed % 360) * np.pi / 180
        
        for y in range(self.size):
            for x in range(self.size):
                proj = x * np.cos(angle) + y * np.sin(angle)
                n = noise.pnoise2(x/50.0, y/50.0, octaves=4, base=self.seed) * 0.25
                pos = (proj / self.size * 2 - 1) + n
                
                if pos < 0:  # Lowland
                    h = 0.15 + abs(pos) * 0.1
                else:  # Highland
                    h = 0.55 + pos * 0.4
                
                self.elevation[y, x] = h
        
        # Terrace it
        self.elevation = np.round(self.elevation * 6) / 6
        self.water_level = 0.12
    
    # --- UTILITY METHODS ---
    
    def _normalize(self):
        """Clamp 0-1 range"""
        e = self.elevation
        self.elevation = (e - e.min()) / (e.max() - e.min() + 1e-10)
    
    def _smooth(self, sigma: float = 1.0):
        """Gaussian blur to remove artifacts"""
        self.elevation = gaussian_filter(self.elevation, sigma=sigma)
    
    def get_slope(self) -> np.ndarray:
        """Calculate slope steepness (0=flat, 1=cliff)"""
        gy, gx = np.gradient(self.elevation)
        slope = np.sqrt(gx**2 + gy**2)
        return np.clip(slope * 4, 0, 1)
    
    def _package_data(self) -> Dict:
        """Standard output format for Warith"""
        return {
            "metadata": {
                "terrain_type": self.terrain_type,
                "seed": self.seed,
                "size": self.size,
                "water_level": float(self.water_level),
                "coordinate_system": "y_x_array_indices"
            },
            "elevation": self.elevation.tolist(),
            "slope": self.get_slope().tolist(),
            "water_mask": (self.elevation < self.water_level).tolist()
        }
    
    # --- EXPORT METHODS ---
    
    def save_json(self, filename: str = None):
        """Full data export"""
        if filename is None:
            filename = f"{self.terrain_type}_{self.seed}.json"
        
        with open(filename, 'w') as f:
            json.dump(self._package_data(), f)
        print(f"Saved: {filename}")
        return filename
    
    def save_raw(self, filename: str = "terrain.raw"):
        """Unity 16-bit raw"""
        h_uint16 = (self.elevation * 65535).astype(np.uint16)
        h_uint16.tofile(filename)
        print(f"Saved RAW: {filename}")
    
    def save_png(self, filename: str = "terrain.png"):
        """8-bit grayscale PNG"""
        from PIL import Image
        h_8bit = (self.elevation * 255).astype(np.uint8)
        Image.fromarray(h_8bit, mode='L').save(filename)
        print(f"Saved PNG: {filename}")

# --- EXTENSION INTERFACE (for later) ---
"""
class CityModule(TerrainModule):
    def __init__(self, density=0.5):
        self.density = density
    
    def apply(self, heightmap: np.ndarray, params: Dict) -> np.ndarray:
        # Find flat spots
        # Mark buildable areas
        # Return modified heightmap or separate layer
        pass
"""

if __name__ == "__main__":
    # Demo all types
    types = ["valley", "beach", "mesa", "mountains", "plateau", "cliff"]
    
    for t in types:
        print(f"\nGenerating {t}...")
        gen = TerrainGenerator(size=150, seed=42)
        gen.generate(t)
        gen.save_json(f"{t}_demo.json")
        gen.save_png(f"{t}_demo.png")