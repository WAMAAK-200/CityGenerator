import numpy as np
import random
import noise  # pip install noise
from scipy.ndimage import gaussian_filter
import json
import sys

class TerrainEngine:
    def __init__(self, width=256, height=256, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 10000)
        
        # Data Layers
        self.heightmap = np.zeros((height, width), dtype=np.float32)
        self.water_mask = np.zeros((height, width), dtype=bool)
        self.river_mask = np.zeros((height, width), dtype=bool)
        self.buildable_mask = np.zeros((height, width), dtype=bool)
        
        print(f"Terrain Engine Initialized [Seed: {self.seed}]")

    def _get_noise(self, x, y, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0):
        return noise.pnoise2(x / scale, 
                             y / scale, 
                             octaves=octaves, 
                             persistence=persistence, 
                             lacunarity=lacunarity, 
                             repeatx=1024, 
                             repeaty=1024, 
                             base=self.seed)

    def generate_base_terrain(self, terrain_type="valley"):
        print(f"Generating base terrain: {terrain_type}...")
        if terrain_type == "valley":
            self._gen_natural_valley()
        elif terrain_type == "mesa":
            self._gen_mesa_canyon()
        elif terrain_type == "archipelago":
            self._gen_archipelago()
        else:
            self._gen_rolling_plains()
        self._normalize_heightmap()

    def _gen_natural_valley(self):
        for y in range(self.height):
            for x in range(self.width):
                base = self._get_noise(x, y, scale=150.0, octaves=4)
                warp_x = x + self._get_noise(x, y, scale=50, octaves=1) * 50
                dist_from_center = abs((warp_x / self.width) - 0.5) * 2.0
                valley_shape = np.power(dist_from_center, 1.5)
                self.heightmap[y, x] = base * 0.3 + valley_shape * 0.8

    def _gen_mesa_canyon(self):
        for y in range(self.height):
            for x in range(self.width):
                n = self._get_noise(x, y, scale=80.0, octaves=6)
                layers = 6
                self.heightmap[y, x] = np.floor((n + 1) / 2 * layers) / layers

    def _gen_archipelago(self):
        for y in range(self.height):
            for x in range(self.width):
                n = self._get_noise(x, y, scale=100.0, octaves=6)
                dx = x / self.width - 0.5
                dy = y / self.height - 0.5
                d = np.sqrt(dx*dx + dy*dy) * 2
                self.heightmap[y, x] = n - d + 0.5

    def _gen_rolling_plains(self):
        for y in range(self.height):
            for x in range(self.width):
                self.heightmap[y, x] = self._get_noise(x, y, scale=200.0, octaves=3)

    # --- FLUID DYNAMICS (FIXED) ---

    def add_ocean(self, water_level=0.3):
        print(f"Flooding oceans at level {water_level}...")
        self.water_mask = self.heightmap < water_level
        self.heightmap[self.water_mask] = water_level * 0.8

    def add_river(self, source_count=3):
        """
        Calculates river paths using simple gravity descent.
        Uses a FOR loop to guarantee no infinite hanging.
        """
        print(f"Carving {source_count} rivers...")
        
        for i in range(source_count):
            # 1. Find a random high point (Attempt 100 times then give up on this river)
            start_x, start_y = 0, 0
            found_start = False
            for _ in range(100):
                rx = random.randint(1, self.width - 2)
                ry = random.randint(1, self.height - 2)
                if self.heightmap[ry, rx] > 0.6:
                    start_x, start_y = rx, ry
                    found_start = True
                    break
            
            if not found_start:
                print(f"  - River {i+1}: Could not find high ground, skipping.")
                continue

            # 2. Flow Downhill
            cx, cy = start_x, start_y
            max_steps = self.width * 2  # Hard limit on length
            
            # Using a for loop prevents infinite while loops
            for step in range(max_steps):
                self.river_mask[cy, cx] = True
                
                # Dig the riverbed slightly (Erosion)
                self.heightmap[cy, cx] -= 0.01 

                # Look for lowest neighbor
                current_h = self.heightmap[cy, cx]
                lowest_h = current_h
                nx, ny = cx, cy
                
                # Check 8 neighbors
                found_downhill = False
                
                # Shuffle directions to avoid straight-line bias
                neighbors = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
                random.shuffle(neighbors)

                for dx, dy in neighbors:
                    tx, ty = cx + dx, cy + dy
                    
                    # Boundary check
                    if 0 <= tx < self.width and 0 <= ty < self.height:
                        h = self.heightmap[ty, tx]
                        
                        # Optimization: If we hit water, we are done immediately
                        if self.water_mask[ty, tx]:
                            self._create_delta(tx, ty)
                            nx, ny = tx, ty # Move to water
                            found_downhill = False # Signal to stop loop
                            break 

                        # Strict gravity: must be lower than current
                        if h < lowest_h:
                            lowest_h = h
                            nx, ny = tx, ty
                            found_downhill = True
                
                # Check why we stopped
                if self.water_mask[ny, nx]:
                    # We hit the ocean
                    break
                
                if not found_downhill:
                    # We are in a pit (local minimum) and not in water
                    # Create a small lake and stop
                    self._create_delta(cx, cy, radius=2)
                    break
                
                # Move the river head
                cx, cy = nx, ny

        print("Rivers generated.")

    def _create_delta(self, x, y, radius=4):
        y_min, y_max = max(0, y-radius), min(self.height, y+radius)
        x_min, x_max = max(0, x-radius), min(self.width, x+radius)
        self.river_mask[y_min:y_max, x_min:x_max] = True
        self.water_mask[y_min:y_max, x_min:x_max] = True # Make deltas actual water

    # --- POST PROCESSING ---

    def smooth_terrain(self, intensity=1.0):
        print("Smoothing terrain...")
        self.heightmap = gaussian_filter(self.heightmap, sigma=intensity)

    def calculate_buildable_areas(self, slope_tolerance=0.08):
        print("Calculating buildable zones...")
        gy, gx = np.gradient(self.heightmap)
        slope_map = np.sqrt(gx**2 + gy**2)
        
        is_flat = slope_map < slope_tolerance
        is_dry = ~self.water_mask
        is_not_river = ~self.river_mask
        
        self.buildable_mask = is_flat & is_dry & is_not_river
        return slope_map

    def _normalize_heightmap(self):
        min_val = np.min(self.heightmap)
        max_val = np.max(self.heightmap)
        if max_val - min_val > 0:
            self.heightmap = (self.heightmap - min_val) / (max_val - min_val)

    def export_data(self, filename="terrain_data.json"):
        gy, gx = np.gradient(self.heightmap)
        slope = np.sqrt(gx**2 + gy**2)

        data = {
            "metadata": {
                "width": self.width,
                "height": self.height,
                "seed": self.seed
            },
            "heightmap": self.heightmap.tolist(),
            "water_mask": self.water_mask.tolist(),
            "river_mask": self.river_mask.tolist(),
            "buildable_mask": self.buildable_mask.tolist(),
            "slope": slope.tolist(),
            "city_sites": [],
            "barrier_mask": (slope > 0.15).tolist()
        }
        
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Data exported to {filename}")

if __name__ == "__main__":
    # Settings
    SIZE = 256
    SEED = 42
    
    engine = TerrainEngine(SIZE, SIZE, SEED)
    engine.generate_base_terrain("valley") 
    engine.add_ocean(water_level=0.25)
    engine.add_river(source_count=5)
    engine.smooth_terrain(intensity=0.5)
    engine.calculate_buildable_areas()
    engine.export_data("terrain_data.json")
    print("Done! Run visualiser.py to view.")