# CITY GENERATOR
# Generates 2D cities
# Can be converted to 3D
# Can be integrated into game engines to serve as an internal city generator
# Can generate cities based on fractal patterns.

import numpy as np
import noise
import matplotlib.pyplot as pt
import json
import os

class CityGenerator():
    def __init__(self,size = 50, seed=42):
        self.size = size
        self.seed = seed
        self.heightmap = None
        self.districts = None
        self.roads = None


    # Generates a fractal pattern
    def generate_fractal_pattern(self):

        
        # Creates the heightmap for terrain and elevation
        heightmap = np.zeros((self.size, self.size))
        scale = 50
        octaves = 6
        for i in range(self.size):
            for j in range(self.size):
                value = 0.0
                amplitude = 1.0
                frequency = 1.0

                for oct in range(octaves):
                    # sample noise at frequency
                    nx = 
                heightmap[i][j] = noise.pnoise2 (
                    i/scale,
                    j/scale,
                    octaves = octaves,
                    persistence = 0.5,
                    lacunarity = 2.0,
                    repeatx = self.size,
                    repeaty = self.size,
                    base = self.seed
                )

                # Normalise to 0-1 range
                return (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())


            
    