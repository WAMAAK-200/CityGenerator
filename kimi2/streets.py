import numpy as np
from typing import List, Tuple, Set
import noise

class StreetNetwork:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.streets: Set[Tuple[int, int]] = set()
        self.main_roads: Set[Tuple[int, int]] = set()
        self.wall_points: Set[Tuple[int, int]] = set()
        
    def generate_organic(self, num_branches: int = 5) -> Set[Tuple[int, int]]:
        """Organic growth from center - like your reference image"""
        center = self.size // 2
        start = (center, center)
        
        # Main arterial roads first (3-4 main branches)
        for _ in range(3):
            angle = self.rng.uniform(0, np.pi * 2)
            length = self.size * 0.35
            self._carve_organic_road(start, angle, length, width=2, is_main=True)
        
        # Secondary winding streets
        for _ in range(num_branches * 3):
            # Pick random point on existing road
            if not self.streets:
                continue
            base = list(self.streets)[self.rng.randint(0, len(self.streets))]
            angle = self.rng.uniform(0, np.pi * 2)
            length = self.rng.randint(20, 50)
            self._carve_organic_road(base, angle, length, width=1, is_main=False)
            
        return self.streets
    
    def generate_grid(self, spacing: int = 8) -> Set[Tuple[int, int]]:
        """Roman/colonial grid"""
        center = self.size // 2
        
        for i in range(-center//spacing, center//spacing + 1):
            x = center + i * spacing
            for y in range(self.size):
                if 0 <= x < self.size:
                    self.streets.add((x, y))
                    if i % 2 == 0:
                        self.main_roads.add((x, y))
        
        for j in range(-center//spacing, center//spacing + 1):
            y = center + j * spacing
            for x in range(self.size):
                if 0 <= y < self.size:
                    self.streets.add((x, y))
                    if j % 2 == 0:
                        self.main_roads.add((x, y))
        
        return self.streets
    
    def generate_fractal(self, iterations: int = 4) -> Set[Tuple[int, int]]:
        """Recursive subdivision pattern"""
        center = self.size // 2
        
        def subdivide(x, y, w, h, depth):
            if depth == 0 or w < 10 or h < 10:
                return
            
            # Draw dividing lines
            if w > h:
                split_x = x + w // 2 + self.rng.randint(-w//4, w//4)
                for dy in range(h):
                    if 0 <= split_x < self.size and 0 <= y+dy < self.size:
                        self.streets.add((split_x, y+dy))
                subdivide(x, y, split_x-x, h, depth-1)
                subdivide(split_x, y, x+w-split_x, h, depth-1)
            else:
                split_y = y + h // 2 + self.rng.randint(-h//4, h//4)
                for dx in range(w):
                    if 0 <= x+dx < self.size and 0 <= split_y < self.size:
                        self.streets.add((x+dx, split_y))
                subdivide(x, y, w, split_y-y, depth-1)
                subdivide(x, split_y, w, y+h-split_y, depth-1)
        
        subdivide(center-40, center-40, 80, 80, iterations)
        
        # Loop roads around outside
        radius = 45
        for angle in np.linspace(0, 2*np.pi, 100):
            x = int(center + np.cos(angle) * radius)
            y = int(center + np.sin(angle) * radius)
            if 0 <= x < self.size and 0 <= y < self.size:
                self.streets.add((x, y))
                self.wall_points.add((x, y))
        
        return self.streets
    
    def generate_walls(self, radius: float = None):
        """Circular walls like in reference image"""
        center = self.size // 2
        if radius is None:
            radius = self.size * 0.35
        
        points = set()
        for angle in np.linspace(0, 2*np.pi, int(radius * 3)):
            x = int(center + np.cos(angle) * radius)
            y = int(center + np.sin(angle) * radius)
            if 0 <= x < self.size and 0 <= y < self.size:
                points.add((x, y))
        
        self.wall_points = points
        return points
    
    def _carve_organic_road(self, start: Tuple[int, int], angle: float, 
                           length: float, width: int = 1, is_main: bool = False):
        """Curved road using Perlin noise for natural flow"""
        x, y = float(start[0]), float(start[1])
        
        for step in range(int(length)):
            # Add curve variation
            curve = noise.pnoise2(step*0.1, self.seed) * 0.5
            current_angle = angle + curve
            
            x += np.cos(current_angle)
            y += np.sin(current_angle)
            
            ix, iy = int(x), int(y)
            if not (0 <= ix < self.size and 0 <= iy < self.size):
                break
            
            # Carve width
            for dx in range(-width//2, width//2+1):
                for dy in range(-width//2, width//2+1):
                    nx, ny = ix+dx, iy+dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        self.streets.add((nx, ny))
                        if is_main:
                            self.main_roads.add((nx, ny))