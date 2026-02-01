import numpy as np
import noise
import matplotlib.pyplot as plt

class TerrainGenerator:
    def __init__(self, size=100, seed=42):
        self.size = size
        self.seed = seed
        self.heightmap = None

    def generate_fractal(self, sharpness=1.0):
        """
        Generates a heightmap using Perlin noise with octaves.
        sharpness: > 1.0 makes peaks pointier and valleys flatter.
                   < 1.0 makes the terrain rounder and softer.
        """
        self.heightmap = np.zeros((self.size, self.size))
        centre = self.size / 2
        
        # We use different frequencies (f) and amplitudes (0.5**k) 
        # to create fractal-like detail (octaves).
        scale = 30.0
        octaves = [1, 2, 4, 8]

        for i in range(self.size):
            for j in range(self.size):
                nx = (i - centre) / scale
                ny = (j - centre) / scale
                
                # Summing noise layers
                val = sum((1.0 - abs(noise.pnoise2(nx*f, ny*f, base=self.seed))) * (0.5**k) 
                          for k, f in enumerate(octaves))
                
                self.heightmap[i, j] = val

        # Normalize values to range 0.0 - 1.0
        min_val = self.heightmap.min()
        max_val = self.heightmap.max()
        self.heightmap = (self.heightmap - min_val) / (max_val - min_val)
        
        # Apply sharpness (exponentiation)
        # Power of >1 pushes mid-tones down (wider valleys, steeper peaks)
        self.heightmap = self.heightmap ** sharpness
        
        return self.heightmap

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Settings
    SIZE = 100
    SEED = 42
    SHARPNESS = 1.5  # Try 3.0 for very steep mountains, 0.5 for rolling hills

    # Generate
    gen = TerrainGenerator(size=SIZE, seed=SEED)
    terrain = gen.generate_fractal(sharpness=SHARPNESS)

    # Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(terrain, cmap='terrain')
    plt.colorbar(label="Height")
    plt.title(f"Terrain Generation\nSeed: {SEED} | Sharpness: {SHARPNESS}")
    plt.show()