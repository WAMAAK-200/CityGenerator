import numpy as np
import random
import json
import os
import sys
import noise # pip install noise
from scipy.ndimage import binary_dilation

class WaterSystem:
    def __init__(self, input_file="raw_terrain.json"):
        if not os.path.exists(input_file):
            print("Error: Generate terrain first using perlin_biomes.py")
            sys.exit(1)
            
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        self.heightmap = np.array(data["heightmap"])
        self.size = data["width"]
        self.seed = data["seed"]
        
        # New Layers
        self.water_mask = np.zeros_like(self.heightmap, dtype=bool)   # Deep water
        self.river_mask = np.zeros_like(self.heightmap, dtype=bool)   # Moving water
        self.lake_mask = np.zeros_like(self.heightmap, dtype=bool)    # Still water
        self.swamp_mask = np.zeros_like(self.heightmap, dtype=bool)   # Marsh
        self.beach_mask = np.zeros_like(self.heightmap, dtype=bool)   # Sand
        
        # Metadata for your friend (Road building logic)
        self.crossable_mask = np.zeros_like(self.heightmap, dtype=bool) 

    def _noise(self, x, y, scale=50.0):
        return noise.pnoise2(x/scale, y/scale, base=self.seed+99)

    def generate_water_features(self, sea_level=0.2):
        print(f"Adding Water Systems (Sea Level: {sea_level})...")
        
        # 1. Oceans (Base Level)
        self._add_oceans(sea_level)
        
        # 2. Rivers (Gravity Simulation)
        # We try to spawn rivers at high points
        self._add_rivers(count=8)
        
        # 3. Swamps (Proximity to water + low elevation)
        self._add_swamps()
        
        # 4. Beaches (Erosion around oceans)
        self._add_beaches()
        
        self._save()

    def _add_oceans(self, level):
        print("- Flooding Oceans...")
        # Simple flood fill
        self.water_mask = self.heightmap < level
        
        # Flatten ocean floor physically for the visualizer
        self.heightmap[self.water_mask] = level * 0.8

    def _add_rivers(self, count):
        print(f"- Simulating {count} Rivers...")
        
        for i in range(count):
            # Find a random start point high up
            start_x, start_y = self._find_high_ground()
            if start_x is None: continue

            cx, cy = start_x, start_y
            
            # Trace path
            path = []
            
            # Flow limit to prevent infinite loops
            for _ in range(self.size * 3):
                path.append((cx, cy))
                
                # Check neighbors for lowest point
                nx, ny = self._find_lowest_neighbor(cx, cy)
                
                # STOP CONDITIONS:
                
                # 1. Hit Ocean -> Form Delta
                if self.water_mask[ny, nx]:
                    self._create_delta(cx, cy)
                    break
                
                # 2. Hit Existing River -> Join it
                if self.river_mask[ny, nx]:
                    break
                    
                # 3. Local Minima (Pit) -> Form Lake
                if self.heightmap[ny, nx] >= self.heightmap[cy, cx]:
                    self._create_lake(cx, cy)
                    break
                    
                cx, cy = nx, ny
                
            # Draw the river
            self._carve_river(path)

    def _carve_river(self, path):
        """Draws the river with varying width and islands."""
        for i, (x, y) in enumerate(path):
            # Width increases as river flows downstream
            width = 1 + (i // 50)
            
            # Island Logic:
            # If river is wide (width > 2), check noise to leave a spot dry
            for dy in range(-width, width+1):
                for dx in range(-width, width+1):
                    tx, ty = x+dx, y+dy
                    if 0 <= tx < self.size and 0 <= ty < self.size:
                        
                        # Distance from center of river
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        if dist <= width:
                            # ISLAND CHECK:
                            # High noise value in middle of river = Island
                            island_noise = self._noise(tx, ty, scale=10.0)
                            if width > 2 and island_noise > 0.4 and dist < (width/2):
                                continue # Skip drawing water (Island!)
                                
                            self.river_mask[ty, tx] = True
                            
                            # Erode terrain slightly so river sits "in" the land
                            self.heightmap[ty, tx] -= 0.02
                            
                            # Mark as crossable if narrow
                            if width < 2:
                                self.crossable_mask[ty, tx] = True

    def _create_delta(self, x, y):
        """Spreads water out where river meets sea."""
        radius = 5
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                tx, ty = x+dx, y+dy
                if 0 <= tx < self.size and 0 <= ty < self.size:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Use noise to make "fingers" of the delta
                    if dist < radius and self._noise(tx, ty, scale=5.0) > 0.0:
                        self.river_mask[ty, tx] = True
                        self.heightmap[ty, tx] -= 0.01

    def _create_lake(self, x, y):
        """Fills a depression with water."""
        radius = random.randint(3, 8)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                tx, ty = x+dx, y+dy
                if 0 <= tx < self.size and 0 <= ty < self.size:
                    if dx**2 + dy**2 < radius**2:
                        self.lake_mask[ty, tx] = True
                        # Level the lake surface
                        self.heightmap[ty, tx] = self.heightmap[y, x]

    def _add_swamps(self):
        """
        Swamps appear where land is flat, low, and near water.
        """
        print("- Growing Swamps...")
        # 1. Identify low flat lands
        is_low = (self.heightmap > 0.2) & (self.heightmap < 0.35)
        
        # 2. Identify near-water areas (Dilation)
        all_water = self.water_mask | self.river_mask | self.lake_mask
        near_water = binary_dilation(all_water, iterations=4)
        
        # 3. Combine
        self.swamp_mask = is_low & near_water & ~all_water
        
        # Add noise to swamps (patchy)
        noise_mask = np.zeros_like(self.swamp_mask)
        for y in range(self.size):
            for x in range(self.size):
                if self._noise(x, y, scale=20.0) > 0.1:
                    noise_mask[y, x] = True
                    
        self.swamp_mask = self.swamp_mask & noise_mask

    def _add_beaches(self):
        print("- Spreading Sand...")
        # Dilate ocean mask to find coastline
        coast = binary_dilation(self.water_mask, iterations=2)
        # Beach is Coast minus Water
        self.beach_mask = coast & ~self.water_mask & ~self.river_mask

    def _find_high_ground(self):
        """Tries 100 times to find a high point."""
        for _ in range(100):
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if self.heightmap[y, x] > 0.6:
                return x, y
        return None, None

    def _find_lowest_neighbor(self, x, y):
        """Returns coordinates of the lowest neighbor."""
        min_h = self.heightmap[y, x]
        best_x, best_y = x, y
        
        # Check 8 directions
        dirs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        random.shuffle(dirs) # Randomize flow bias
        
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                h = self.heightmap[ny, nx]
                if h < min_h:
                    min_h = h
                    best_x, best_y = nx, ny
                    
        return best_x, best_y

    def _save(self):
        filename = "complete_map.json"
        
        # Combine all water layers into one generic "wet" mask for simple checking
        wet_mask = self.water_mask | self.river_mask | self.lake_mask
        
        data = {
            "width": self.size,
            "height": self.size,
            "seed": self.seed,
            "heightmap": self.heightmap.tolist(),
            "masks": {
                "ocean": self.water_mask.tolist(),
                "river": self.river_mask.tolist(),
                "lake": self.lake_mask.tolist(),
                "swamp": self.swamp_mask.tolist(),
                "beach": self.beach_mask.tolist(),
                "crossable": self.crossable_mask.tolist(), # For bridges
                "wet": wet_mask.tolist() # Helper
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Success! Map saved to {filename}")

if __name__ == "__main__":
    ws = WaterSystem()
    ws.generate_water_features(sea_level=0.15)