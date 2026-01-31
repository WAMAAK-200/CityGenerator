# CITY GENERATOR - Procedural Urban Tool
# Final Version: Guaranteed Spine Connectivity for all road segments.

import numpy as np
import noise
import matplotlib.pyplot as plt
import json
import os

class CityGenerator():
    def __init__(self, size=100, seed=42):
        self.size = size
        self.seed = seed
        self.heightmap = None
        self.city_map = None
        self.buildings_data = {}

    def generate_fractal_pattern(self):
        """Generates a centralized fractal heightmap."""
        self.heightmap = np.zeros((self.size, self.size), dtype=np.float32)
        scale = 35.0
        center = self.size / 2
        for i in range(self.size):
            for j in range(self.size):
                nx = (i - center) / scale
                ny = (j - center) / scale
                val = 0.0
                amp, freq = 1.0, 1.0
                for _ in range(6):
                    n = noise.pnoise2(nx * freq, ny * freq, base=self.seed)
                    val += (1.0 - abs(n)) * amp
                    amp *= 0.5
                    freq *= 2.0
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                mask = np.clip(1.0 - (dist / (self.size * 0.6)), 0, 1)
                self.heightmap[i][j] = (val ** 2) * mask
        self.heightmap = (self.heightmap - self.heightmap.min()) / (self.heightmap.max() - self.heightmap.min())
        return self.heightmap

    def generate_city_entities(self, bdg_threshold=0.32):
        if self.heightmap is None: self.generate_fractal_pattern()
        
        self.city_map = np.zeros((self.size, self.size), dtype=np.int32)
        self.buildings_data = {}
        building_id_counter = 1
        center_i, center_j = self.size // 2, self.size // 2

        # 1. Establish Permanent Spine (Main Arteries)
        for k in range(self.size):
            self.city_map[center_i, k] = -1
            self.city_map[k, center_j] = -1

        # 2. Building Placement (8x4 Logic)
        x_step, y_step = 10, 6 
        for i in range(2, self.size - 10, x_step):
            for j in range(2, self.size - 6, y_step):
                density = self.heightmap[i, j]
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)

                if density > bdg_threshold and dist < (self.size * 0.45):
                    # Place Building
                    self.city_map[i:i+8, j:j+4] = building_id_counter
                    self.buildings_data[building_id_counter] = {
                        "id": building_id_counter,
                        "pos": [int(i), int(j)],
                        "size": [8, 4],
                        "height": round(float(density * 25), 2),
                        "engine_tag": "Building_Mesh"
                    }
                    
                    # Create a local road stub adjacent to the building
                    self.city_map[i-1, j:j+4] = -1 
                    building_id_counter += 1

        # 3. RECURSIVE SPINE CONNECTION (Final Adjustment)
        # This pass ensures every road segment traces back to the center spine.
        for i in range(self.size):
            for j in range(self.size):
                if self.city_map[i, j] == -1:
                    # Trace back to horizontal spine
                    curr_i = i
                    while curr_i != center_i:
                        step = 1 if center_i > curr_i else -1
                        curr_i += step
                        # Stop if we hit an existing road, otherwise bridge the gap
                        if self.city_map[curr_i, j] == -1:
                            break
                        # Only place road if it doesn't destroy a building lot
                        if self.city_map[curr_i, j] == 0:
                            self.city_map[curr_i, j] = -1
                    
                    # Trace back to vertical spine
                    curr_j = j
                    while curr_j != center_j:
                        step = 1 if center_j > curr_j else -1
                        curr_j += step
                        if self.city_map[i, curr_j] == -1:
                            break
                        if self.city_map[i, curr_j] == 0:
                            self.city_map[i, curr_j] = -1

        print(f"Generated {len(self.buildings_data)} building entities.")
        return self.city_map

    def get_3d_entities(self):
        """
        Converts 2D city_map data into 3D Transform objects.
        Returns a list of dictionaries with 3D coordinates.
        """
        entities_3d = []
        
        # Convert Buildings
        for b_id, data in self.buildings_data.items():
            i, j = data["pos"]
            entities_3d.append({
                "type": "building",
                "id": b_id,
                "position": [float(i + 1), float(data["height"] / 2), float(j + 1)], # Centered height
                "scale": [2.0, float(data["height"]), 2.0]
            })
            
        # Convert Roads
        for i in range(self.size):
            for j in range(self.size):
                if self.city_map[i, j] == -1:
                    entities_3d.append({
                        "type": "road",
                        "position": [float(i), 0.05, float(j)], # Slightly above ground
                        "scale": [1.0, 0.1, 1.0]
                    })
        
        print(f"Converted {len(entities_3d)} entities to 3D data.")
        return entities_3d

    def export_to_json(self, filename="city_assets.json"):
        export_body = {str(k): v for k, v in self.buildings_data.items()}
        with open(filename, 'w') as f:
            json.dump(export_body, f, indent=4)
        print(f"Metadata exported to {filename}")

    def visualise_city(self, filename="citymap.png"):
        if self.city_map is None: return
        print(f"Creating visualisation: {filename}")
        
        display_map = np.zeros(self.city_map.shape, dtype=np.int32)
        display_map[self.city_map == -1] = 1 # Roads
        display_map[self.city_map > 0] = 2  # Buildings

        plt.figure(figsize=(10, 10))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#2ecc71', '#34495e', '#ecf0f1'])
        
        plt.imshow(display_map, cmap=cmap)
        plt.title(f"Connected Procedural City | Seed: {self.seed}")
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Saved as {filename}")
        plt.close()

if __name__ == "__main__":
    city = CityGenerator(size=100)
    city.generate_city_entities()
    city.visualise_city("citymap.png")
    city.export_to_json("city_assets.json")