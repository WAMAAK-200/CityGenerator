import numpy as np
from noise import pnoise2, snoise2
from scipy.ndimage import gaussian_filter, distance_transform_edt, convolve
from typing import Dict, List, Tuple
import json

class NaturalTerrain:
    """
    Naturalistic terrain focused on flat, buildable ground with 
    occasional hills and gentle valleys. No crazy spikes.
    """
    
    def __init__(self, size: int = 512, seed: int = 42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)
        
        self.elevation = np.zeros((size, size), dtype=np.float32)
        self.water_mask = np.zeros((size, size), dtype=bool)
        self.river_mask = np.zeros((size, size), dtype=bool)
        
        # Analysis
        self.slope = np.zeros((size, size), dtype=np.float32)
        self.habitability = np.zeros((size, size), dtype=np.float32)
        
    def _noise(self, freq: float, octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
        """Simple fractal noise"""
        result = np.zeros((self.size, self.size))
        amp = 1.0
        f = freq
        
        for i in range(octaves):
            # Create grid
            x = np.linspace(0, f*self.size, self.size)
            y = np.linspace(0, f*self.size, self.size)
            X, Y = np.meshgrid(x, y)
            
            # Simplex noise (smoother than Perlin)
            vec = np.vectorize(lambda x, y: snoise2(x, y, base=self.seed + i*100))
            result += vec(X, Y) * amp
            
            amp *= persistence
            f *= 2.0
            
        return result
    
    def _erode(self, height: np.ndarray, iterations: int = 50) -> np.ndarray:
        """
        Thermal erosion: rounds off spikes, fills pits, creates natural slopes.
        Makes hills look like actual hills, not math functions.
        """
        h = height.copy()
        talus = 4.0 / self.size  # Max slope before soil slides
        
        for _ in range(iterations):
            # Get slopes to neighbors
            dy, dx = np.gradient(h)
            slope_mag = np.sqrt(dx**2 + dy**2)
            
            # Where too steep, move soil down
            too_steep = slope_mag > talus
            if not np.any(too_steep):
                break
                
            # Simple blur in steep areas (soil slide)
            h = gaussian_filter(h, sigma=0.8)
            
        return h
    
    def generate_plains(self, flatness: float = 0.8) -> Dict:
        """
        Mostly flat with gentle undulations. 
        flatness: 0.9 = very flat farmland, 0.5 = rolling
        """
        # Low frequency, low amplitude
        base = self._noise(0.005, octaves=3, persistence=0.4)
        
        # Bias toward middle (flat)
        base = base * (1 - flatness)
        
        # Height range: 0.3 to 0.5 (all above water, no mountains)
        self.elevation = 0.4 + base * 0.15
        
        # Small noise for texture
        detail = self._noise(0.02, octaves=2) * 0.02
        self.elevation += detail
        
        self._finalize()
        return self.export()
    
    def generate_hills(self, hill_height: float = 0.3) -> Dict:
        """
        Rolling hills - rounded tops, gentle slopes. Actually buildable on top.
        """
        # Medium frequency
        base = self._noise(0.008, octaves=4, persistence=0.5)
        
        # Power < 1 makes valleys wider (flatter bottom), hills rounded
        # Power > 1 makes sharp peaks
        shape = np.sign(base) * np.abs(base) ** 0.7  # Rounded hills
        
        self.elevation = 0.35 + shape * hill_height
        
        # Thermal erosion to make it look natural (no math-looking spikes)
        self.elevation = self._erode(self.elevation, iterations=30)
        
        # Carve occasional river if terrain varies enough
        if hill_height > 0.2:
            self._carve_natural_river()
            
        self._finalize()
        return self.export()
    
    def generate_valley(self, width_factor: float = 0.4) -> Dict:
        """
        Wide, flat valley floor with gentle sides. River in middle.
        No sine waves - uses noise-wander for natural river.
        """
        # Valley direction
        angle = np.random.random() * np.pi
        Y, X = np.ogrid[:self.size, :self.size]
        
        # Rotate coordinates
        Xr = X * np.cos(angle) + Y * np.sin(angle)
        Yr = -X * np.sin(angle) + Y * np.cos(angle)
        
        # Meander the valley center (noise-driven, not sine)
        wander = self._noise(0.01, octaves=2, persistence=0.5)[:, int(self.size/2)]
        center_y = (Yr - wander * self.size * 0.1) / (self.size * width_factor)
        
        # Soft U-shape: power 0.4 makes flat floor, steep sides
        dist = np.abs(center_y)
        profile = np.power(dist, 0.4)  # Flat bottom
        
        # Noise for irregularity
        surface = self._noise(0.015, octaves=3) * 0.05
        
        self.elevation = 0.3 + profile * 0.4 + surface
        
        # Flatten the floor even more for building
        floor_mask = dist < 0.3
        self.elevation[floor_mask] = 0.3 + surface[floor_mask] * 0.5
        
        self._carve_natural_river(path_randomness=0.6)
        self._finalize()
        return self.export()
    
    def generate_coastal(self, shore_complexity: float = 0.5) -> Dict:
        """
        Flat beach gradient with noisy coast. Inland is flat/low.
        """
        # Gentle slope from sea to land
        gradient = np.linspace(0, 1, self.size)
        X, Y = np.meshgrid(gradient, gradient)
        
        # Diagonal coast with noise
        coast_pos = (X + Y) / 2
        noise = self._noise(0.012, octaves=4) * 0.15 * shore_complexity
        coast_pos += noise
        
        # S-curve transition: flat underwater, flat beach, flat inland
        # Sigmoid-like shaping
        height = np.where(coast_pos < 0.35, 0.05,  # Sea
                 np.where(coast_pos < 0.45, 0.15 + (coast_pos-0.35)*0.5,  # Beach
                          0.2 + (coast_pos-0.45)*0.3))  # Low plain
        
        self.elevation = height
        
        # Mark water
        self.water_mask = self.elevation < 0.12
        self.coast_mask = (self.elevation > 0.1) & (self.elevation < 0.2)
        
        self._finalize()
        return self.export()
    
    def _carve_natural_river(self, path_randomness: float = 0.5):
        """
        River that wanders naturally from high to low ground.
        """
        # Start from high point
        high_y, high_x = np.unravel_index(np.argmax(self.elevation), self.elevation.shape)
        x, y = high_x, high_y
        
        path = [(x, y)]
        momentum = np.array([0.0, 0.0])
        
        for _ in range(self.size * 2):
            # Drop downhill with inertia
            gy, gx = np.gradient(self.elevation)
            grad = np.array([gx[y, x], gy[y, x]])
            
            if np.linalg.norm(grad) < 0.0001:
                break
            
            # Normalize and add wander
            grad = grad / (np.linalg.norm(grad) + 1e-10)
            wander = (np.random.randn(2) * path_randomness)
            
            momentum = momentum * 0.6 + grad * 0.4 + wander * 0.1
            momentum = momentum / (np.linalg.norm(momentum) + 1e-10)
            
            x = int(x + momentum[0] * 2)
            y = int(y + momentum[1] * 2)
            
            # Bounds check
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                break
                
            path.append((x, y))
            
            # Stop if hit existing water or very low
            if self.elevation[y, x] < 0.15:
                break
        
        # Carve gentle channel
        for i, (x, y) in enumerate(path):
            # Wider at end (estuary)
            width = 1 if i < len(path)*0.8 else 2
            
            for dy in range(-width, width+1):
                for dx in range(-width, width+1):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < self.size and 0 <= yy < self.size:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= width:
                            carve = (1 - dist/(width+1)) * 0.08
                            self.elevation[yy, xx] -= carve
                            if dist < 0.5:
                                self.river_mask[yy, xx] = True
        
        self.water_mask |= self.river_mask
    
    def _finalize(self):
        """Calculate derived layers"""
        # Slope
        gy, gx = np.gradient(self.elevation)
        self.slope = np.sqrt(gx**2 + gy**2)
        
        # Habitability for cities:
        # 1. Local flatness (can place building)
        # 2. Not water
        # 3. Not steep
        
        local_flat = gaussian_filter(self.elevation, sigma=2)
        local_var = (self.elevation - local_flat) ** 2
        flatness = np.exp(-local_var * 500)  # 1.0 = perfectly flat local
        
        buildable = flatness * (1 - np.clip(self.slope * 5, 0, 1)) * (~self.water_mask)
        
        self.habitability = buildable
        
        # Find buildable patches
        self._find_sites()
    
    def _find_sites(self, min_area: int = 200):
        """Find contiguous buildable areas"""
        from scipy.ndimage import label, center_of_mass
        
        good = self.habitability > 0.7
        labeled, n = label(good)
        
        self.sites = []
        for i in range(1, min(n+1, 6)):
            mask = (labeled == i)
            area = np.sum(mask)
            if area < min_area:
                continue
            
            cy, cx = center_of_mass(mask)
            self.sites.append({
                "x": int(cx),
                "y": int(cy),
                "radius": int(np.sqrt(area / np.pi)),
                "quality": float(np.mean(self.habitability[mask]))
            })
    
    def export(self) -> Dict:
        return {
            "heightmap": self.elevation.astype(np.float32),
            "slope": self.slope.astype(np.float32),
            "water": self.water_mask,
            "river": self.river_mask,
            "habitability": self.habitability.astype(np.float32),
            "buildable": (self.habitability > 0.6),
            "sites": self.sites,
            "meta": {"seed": self.seed, "size": self.size}
        }
    
    def save(self, filename: str = "terrain.json"):
        data = self.export()
        json_data = {
            "heightmap": data["heightmap"].tolist(),
            "slope": data["slope"].tolist(),
            "habitability": data["habitability"].tolist(),
            "buildable": data["buildable"].tolist(),
            "water": data["water"].tolist(),
            "river": data["river"].tolist(),
            "sites": data["sites"],
            "meta": data["meta"]
        }
        with open(filename, 'w') as f:
            json.dump(json_data, f)
        print(f"Saved {filename} with {len(self.sites)} buildable sites")

# Convenience functions for your hackathon
def flat_land(size=512, seed=42):
    """Maximum buildable area for testing city gen"""
    t = NaturalTerrain(size, seed)
    return t.generate_plains(flatness=0.95)

def gentle_hills(size=512, seed=42):
    """Rolling countryside"""
    t = NaturalTerrain(size, seed)
    return t.generate_hills(hill_height=0.25)

def river_valley(size=512, seed=42):
    """Flat valley perfect for linear city"""
    t = NaturalTerrain(size, seed)
    return t.generate_valley(width_factor=0.5)

def coast(size=512, seed=42):
    """Beach and lowland"""
    t = NaturalTerrain(size, seed)
    return t.generate_coastal(shore_complexity=0.6)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Show the natural types
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    gens = [
        ("Flat Plains", flat_land(256)),
        ("Gentle Hills", gentle_hills(256)),
        ("River Valley", river_valley(256)),
        ("Coast", coast(256))
    ]
    
    for ax, (name, data) in zip(axes.flat, gens):
        # Height as background
        ax.imshow(data["heightmap"], cmap="terrain", alpha=0.6)
        
        # Habitability overlay (green = good)
        hab = data["habitability"]
        green = np.zeros((*hab.shape, 4))
        green[..., 1] = 1  # Green channel
        green[..., 3] = hab * 0.5  # Alpha
        ax.imshow(green)
        
        # Sites
        for site in data["sites"]:
            circle = plt.Circle((site["x"], site["y"]), site["radius"], 
                              fill=False, color="red", linewidth=2)
            ax.add_patch(circle)
            ax.plot(site["x"], site["y"], "r+")
        
        ax.set_title(f"{name}\n{np.sum(data['buildable'])} buildable pixels")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("natural_terrain.png", dpi=150)
    plt.show()