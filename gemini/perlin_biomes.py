import numpy as np
import random
import noise
import json
import sys

class TerrainGenerator:
    def __init__(self, size=256, seed=None):
        self.size = size
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.heightmap = np.zeros((size, size), dtype=np.float32)
        print(f"Generator Ready [Seed: {self.seed}]")

    def _noise(self, x, y, scale, octaves=1, persistence=0.5, lacunarity=2.0, base=0):
        """Standard Noise"""
        return noise.pnoise2(x/scale, y/scale, octaves=octaves, 
                             persistence=persistence, lacunarity=lacunarity, 
                             base=self.seed + base)

    def _fbm(self, x, y, scale=100.0, octaves=6):
        """Fractal Brownian Motion (Cloudy/Rocky look)"""
        return self._noise(x, y, scale, octaves, persistence=0.5)

    def generate(self, biome_type):
        print(f"Generating Natural Biome: {biome_type.upper()}")
        
        # 1. Generate specific shape
        if biome_type == "mountains":
            self._gen_single_mountain()
        elif biome_type == "volcano":
            self._gen_volcano()
        elif biome_type == "mesa":
            self._gen_mesa()
        elif biome_type == "valley":
            self._gen_valley()
        elif biome_type == "cliff":
            self._gen_cliff()
        elif biome_type == "plateau":
            self._gen_plateau()
        elif biome_type == "hills":
            self._gen_hills()
        else:
            self._gen_plains()

        self._normalize()
        self._save()

    # --- SHAPE BASED GENERATION ---

    def _get_distorted_coords(self, x, y, strength=40.0):
        """
        DOMAN WARPING: This fixes the 'Straight Lines' / 'Stripes' issue.
        It pushes every pixel slightly left/right/up/down based on noise.
        """
        qx = self._noise(x, y, scale=150.0, base=100) * strength
        qy = self._noise(x, y, scale=150.0, base=200) * strength
        return x + qx, y + qy

    def _gen_single_mountain(self):
        """
        Shape: A distorted cone.
        Result: One major peak with rocky slopes.
        """
        cx, cy = self.size / 2, self.size / 2
        
        for y in range(self.size):
            for x in range(self.size):
                # 1. Warped Coordinates
                wx, wy = self._get_distorted_coords(x, y, strength=60)
                
                # 2. Base Shape: Cone (Distance from center)
                dist = np.sqrt((wx-cx)**2 + (wy-cy)**2)
                max_rad = self.size * 0.6
                
                # Height drops as we get further from center
                base_h = 1.0 - (dist / max_rad)
                base_h = np.clip(base_h, 0, 1) # Don't go below 0
                
                # Curve it to look like a mountain (exponential falloff)
                base_h = np.power(base_h, 1.5)
                
                # 3. Add Details (Rocks)
                detail = self._fbm(x, y, scale=40.0, octaves=6) * 0.2
                
                self.heightmap[y, x] = base_h + detail

    def _gen_mesa(self):
        """
        Shape: A flat-topped mountain (Cylinder).
        """
        cx, cy = self.size / 2, self.size / 2
        
        for y in range(self.size):
            for x in range(self.size):
                wx, wy = self._get_distorted_coords(x, y, strength=30)
                dist = np.sqrt((wx-cx)**2 + (wy-cy)**2)
                
                # Define the Mesa Radius
                radius = self.size * 0.35
                
                # Sigmoid function for steep walls, but not vertical
                # This creates the "Table" shape
                edge_steepness = 0.2
                base_h = 1.0 / (1.0 + np.exp((dist - radius) * edge_steepness))
                
                # Add Steps/Terraces
                # We multiply height by levels, floor it, then divide back
                levels = 4.0
                terraced_h = np.floor(base_h * levels) / levels
                
                # Add slight erosion noise
                noise_val = self._fbm(x, y, scale=30.0) * 0.05
                
                self.heightmap[y, x] = terraced_h + noise_val
    def _gen_plains(self):
        """
        Shape: Mostly flat with very gentle, realistic ground swells.
        1km view: Looks like a buildable grassy field.
        """
        print("Generating Plains...")
        for y in range(self.size):
            for x in range(self.size):
                # 1. The Swell: Very large scale, low height.
                # This prevents it from looking like a flat sheet of paper.
                swell = self._noise(x, y, scale=400.0, octaves=2) * 0.15
                
                # 2. The Texture: Tiny bumps for grass/soil.
                # High frequency, tiny amplitude.
                grass = self._noise(x, y, scale=20.0, octaves=2) * 0.02
                
                # 3. Micro-variation: Breaking up the surface slightly more
                dirt = self._noise(x, y, scale=5.0) * 0.005
                
                # Combine: The base height is 0.0, we just add the variations
                self.heightmap[y, x] = 0.2 + swell + grass + dirt
    
    def _gen_valley(self):
        """
        Shape: A U-Shape trough cutting through the map.
        """
        for y in range(self.size):
            for x in range(self.size):
                # We distort mostly X to make the river meander
                warp_val = self._noise(x, y * 0.5, scale=100.0) * 80.0
                river_center = (self.size / 2) + warp_val
                
                # Distance from the winding river center
                dist = abs(x - river_center)
                
                # Parabola (x^2) creates smooth slopes up from the river
                base_h = np.power(dist / (self.size * 0.4), 2.0)
                
                # Add ground noise
                ground = self._fbm(x, y, scale=60.0) * 0.1
                
                self.heightmap[y, x] = base_h + ground

    def _gen_cliff(self):
        """
        Shape: A diagonal split between high and low ground.
        """
        for y in range(self.size):
            for x in range(self.size):
                # Create a jagged diagonal line
                # y * 0.5 makes the line somewhat diagonal/curved
                warp = self._noise(x, y, scale=80.0) * 50.0
                split_point = (self.size / 2) + warp
                
                dist = x - split_point
                
                # Smooth transition (Sigmoid)
                base_h = 1.0 / (1.0 + np.exp(-dist * 0.1))
                
                # Add texture (Rough cliff face vs smoother top)
                detail = self._fbm(x, y, scale=30.0) * 0.15
                
                self.heightmap[y, x] = base_h + detail

    def _gen_volcano(self):
        """
        Shape: Cone minus a central inverted cone (Crater).
        """
        cx, cy = self.size / 2, self.size / 2
        for y in range(self.size):
            for x in range(self.size):
                wx, wy = self._get_distorted_coords(x, y, strength=20)
                dist = np.sqrt((wx-cx)**2 + (wy-cy)**2)
                
                # 1. The Mountain Base
                mount = 1.0 - (dist / (self.size * 0.6))
                mount = np.clip(mount, 0, 1)
                
                # 2. The Crater Cutout
                crater_rad = self.size * 0.15
                if dist < crater_rad:
                    # Invert height near center
                    # Simple linear dip
                    factor = dist / crater_rad 
                    mount *= factor
                
                detail = self._fbm(x, y, scale=20.0) * 0.1
                self.heightmap[y, x] = mount + detail

    def _gen_hills(self):
        """
        Shape: Multiple blobs (Voronoi-ish feel but smooth).
        """
        for y in range(self.size):
            for x in range(self.size):
                # Just medium scale noise, but warped so it's not grid-aligned
                wx, wy = self._get_distorted_coords(x, y, strength=50)
                
                # Scale 80 = medium hills
                h = self._noise(wx, wy, scale=80.0, octaves=3)
                
                self.heightmap[y, x] = h

    def _gen_plateau(self):
        """
        Shape: High ground that fills most of the map, drop off at one edge.
        """
        for y in range(self.size):
            for x in range(self.size):
                # Similar to cliff but fills more space
                warp = self._noise(x, y, scale=120.0) * 60.0
                edge = (self.size * 0.8) + warp # Edge is far to the right
                
                dist = x - edge
                
                # High ground (1.0) dropping to Low (0.0)
                # Reverse sigmoid
                base_h = 1.0 - (1.0 / (1.0 + np.exp(-dist * 0.05)))
                
                self.heightmap[y, x] = base_h + (self._fbm(x, y) * 0.05)

    def _gen_plains2(self):
        """
        Shape: Flat.
        """
        for y in range(self.size):
            for x in range(self.size):
                # Just light texture
                base = self._noise(x, y, scale=200.0) * 0.2
                detail = self._noise(x, y, scale=20.0) * 0.02
                self.heightmap[y, x] = base + detail

    def _normalize(self):
        """Scales map to 0.0 - 1.0"""
        min_val = self.heightmap.min()
        max_val = self.heightmap.max()
        if max_val - min_val > 0:
            self.heightmap = (self.heightmap - min_val) / (max_val - min_val)
        else:
            self.heightmap[:] = 0.0

    def _save(self):
        filename = "raw_terrain.json"
        data = {
            "width": self.size,
            "height": self.size,
            "seed": self.seed,
            "heightmap": self.heightmap.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    biome = sys.argv[1] if len(sys.argv) > 1 else "mountains"
    gen = TerrainGenerator(size=256, seed=None)
    gen.generate(biome)