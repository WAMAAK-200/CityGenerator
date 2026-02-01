import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

from terrain_generator import TerrainLayer
from hydrology import HydrologyLayer
from streets import StreetNetwork
from building_registry import BuildingRegistry, BuildingType

class LayoutType(Enum):
    ORGANIC = "organic"
    GRID = "grid"
    FRACTAL = "fractal"

@dataclass
class PlacedBuilding:
    building_type: BuildingType
    x: int  # Center X
    y: int  # Center Y
    vertices: List[Tuple[float, float]]  # Polygon vertices (relative to center)
    rotation: float
    width: int  # Bounding box for collision
    depth: int

class CityGenerator:
    def __init__(self, size: int, seed: int, 
                 terrain: TerrainLayer, 
                 hydrology: Optional[HydrologyLayer] = None):
        self.size = size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.terrain = terrain
        self.hydrology = hydrology
        
        self.streets = StreetNetwork(size, seed)
        self.registry = BuildingRegistry()
        
        self.placed_buildings: List[PlacedBuilding] = []
        self.occupied: Set[Tuple[int, int]] = set()
        self.walled_area: Set[Tuple[int, int]] = set()
        
    def generate(self, layout: LayoutType, population: int, 
                 has_walls: bool = True) -> List[PlacedBuilding]:
        
        print(f"Generating {layout.value} street network...")
        if layout == LayoutType.ORGANIC:
            self.streets.generate_organic()
        elif layout == LayoutType.GRID:
            self.streets.generate_grid()
        elif layout == LayoutType.FRACTAL:
            self.streets.generate_fractal()
        
        if has_walls:
            self.streets.generate_walls()
            center = self.size // 2
            for i in range(self.size):
                for j in range(self.size):
                    dist = np.sqrt((i-center)**2 + (j-center)**2)
                    if dist < self.size * 0.35:
                        self.walled_area.add((i, j))
        
        print("Placing buildings...")
        buildings_to_place = self.registry.get_by_priority()
        
        for btype in buildings_to_place:
            count = self._calculate_count(btype, population)
            if count > 0:
                print(f"  Placing {count} {btype.name}(s)...")
            
            for _ in range(count):
                pos = self._find_placement(btype)
                if pos:
                    cx, cy, vertices, bbox_w, bbox_h = pos
                    building = PlacedBuilding(
                        building_type=btype,
                        x=cx, y=cy,
                        vertices=vertices,
                        rotation=self.rng.uniform(0, 360),
                        width=bbox_w, depth=bbox_h
                    )
                    self.placed_buildings.append(building)
                    self._mark_occupied(cx, cy, bbox_w, bbox_h)
        
        return self.placed_buildings
    
    def _calculate_count(self, btype: BuildingType, population: int) -> int:
        """Fixed spawn logic: rarity 1=rare, rarity 10=common"""
        if btype.min_population <= 0:
            return 0
        if population < btype.min_population:
            return 0
        
        theoretical = population / btype.min_population
        
        # Rarity interpretation: 10=common (full spawn), 1=rare (1/10 spawn)
        # Priority 1-2 (Castle/Cathedral) max 1 only
        if btype.priority <= 2:
            probability = min(1.0, theoretical) * (btype.rarity / 10.0)
            return 1 if self.rng.random() < probability else 0
        
        # Common buildings (high rarity) spawn many
        if btype.rarity >= 8:
            count = int(theoretical * self.rng.uniform(0.8, 1.3))
            return max(1, count)  # At least 1 for common
        
        # Medium rarity
        count = int(theoretical * (btype.rarity / 10.0) * self.rng.uniform(0.8, 1.2))
        return max(0, count)
    
    def _generate_polygon(self, size: int, btype: BuildingType) -> Tuple[List[Tuple[float, float]], int, int]:
        """Generate irregular polygonal building shape"""
        w = self.rng.randint(*btype.size_range)
        d = self.rng.randint(*btype.size_range)
        
        # Building types have different shapes
        if btype.name in ["Castle", "Cathedral"]:
            # Large compounds with wings
            vertices = [
                (0, 0), (w*0.7, 0), (w*0.7, d*0.3), 
                (w, d*0.3), (w, d), (0, d)
            ]
        elif btype.name == "Market":
            # Open central area (courtyard)
            vertices = [
                (0, 0), (w, 0), (w, d*0.2), (w*0.8, d*0.2),
                (w*0.8, d*0.8), (w, d*0.8), (w, d), (0, d),
                (0, d*0.8), (w*0.2, d*0.8), (w*0.2, d*0.2), (0, d*0.2)
            ]
        elif btype.district == "slum":
            # Irregular, jagged shapes
            vertices = []
            num_points = self.rng.randint(5, 8)
            for i in range(num_points):
                angle = (i / num_points) * 2 * np.pi
                radius_w = w/2 + self.rng.uniform(-w*0.2, w*0.2)
                radius_d = d/2 + self.rng.uniform(-d*0.2, d*0.2)
                x = w/2 + np.cos(angle) * radius_w
                y = d/2 + np.sin(angle) * radius_d
                vertices.append((x, y))
        else:
            # Standard house: slightly irregular rectangle
            jitter = 0.15
            vertices = [
                (self.rng.uniform(0, w*jitter), self.rng.uniform(0, d*jitter)),
                (w - self.rng.uniform(0, w*jitter), self.rng.uniform(0, d*jitter)),
                (w - self.rng.uniform(0, w*jitter), d - self.rng.uniform(0, d*jitter)),
                (self.rng.uniform(0, w*jitter), d - self.rng.uniform(0, d*jitter))
            ]
        
        return vertices, w, d
    
    def _find_placement(self, btype: BuildingType) -> Optional[Tuple[int, int, List[Tuple[float, float]], int, int]]:
        """Find valid spot and generate polygon"""
        vertices, w, d = self._generate_polygon(0, btype)  # Get shape template
        
        # Get candidates based on district
        if btype.priority <= 2:
            candidates = self._get_premium_locations(w, d)
        elif btype.district == "slum":
            candidates = self._get_slum_locations(w, d)
        elif btype.district == "wealthy":
            candidates = self._get_road_locations(w, d, prefer_main=True)
        else:
            candidates = self._get_standard_locations(w, d)
        
        best = None
        
        for x, y in candidates[:30]:
            cx, cy = x + w//2, y + d//2
            if not self._is_valid(cx, cy, w, d, btype):
                continue
            
            # Generate actual polygon for this spot
            actual_vertices, _, _ = self._generate_polygon(0, btype)
            # Center the vertices
            centered_vertices = [(vx - w/2, vy - d/2) for vx, vy in actual_vertices]
            
            return (cx, cy, centered_vertices, w, d)
        
        return None
    
    def _is_valid(self, cx: int, cy: int, w: int, d: int, btype: BuildingType) -> bool:
        """Check bounding box validity"""
        x, y = cx - w//2, cy - d//2
        
        if x < 0 or y < 0 or x+w >= self.size or y+d >= self.size:
            return False
        
        # Check occupation (simplified to bounding box)
        for dx in range(w):
            for dy in range(d):
                if (x+dx, y+dy) in self.occupied:
                    return False
        
        # Check water
        if self.hydrology:
            for dx in range(w):
                for dy in range(d):
                    if self.hydrology.water_mask[x+dx, y+dy] > 0.5:
                        if not btype.requires_water:
                            return False
        
        # Check walls
        if btype.requires_wall:
            if not all((x+dx, y+dy) in self.walled_area for dx in range(w) for dy in range(d)):
                return False
        
        # Check slope
        if btype.max_slope < 0.4:
            hmap = self.terrain.heightmap
            corners = [
                hmap[x, y], hmap[min(x+w, self.size-1), y],
                hmap[x, min(y+d, self.size-1)], hmap[min(x+w, self.size-1), min(y+d, self.size-1)]
            ]
            max_diff = max(corners) - min(corners)
            if max_diff > btype.max_slope:
                return False
        
        # Check road access
        has_road = False
        perimeter = (
            [(x-1, y+dy) for dy in range(d)] +
            [(x+w+1, y+dy) for dy in range(d)] +
            [(x+dx, y-1) for dx in range(w)] +
            [(x+dx, y+d+1) for dx in range(w)]
        )
        for px, py in perimeter:
            if (px, py) in self.streets.streets:
                has_road = True
                break
        
        return has_road
    
    def _get_premium_locations(self, w: int, h: int) -> List[Tuple[int, int]]:
        center = self.size // 2
        candidates = []
        for radius in range(5, 30, 2):
            for angle in np.linspace(0, 2*np.pi, 12):
                x = int(center + np.cos(angle) * radius - w/2)
                y = int(center + np.sin(angle) * radius - h/2)
                if 0 <= x < self.size-w and 0 <= y < self.size-h:
                    candidates.append((x, y))
        return candidates
    
    def _get_slum_locations(self, w: int, h: int) -> List[Tuple[int, int]]:
        candidates = []
        for _ in range(50):
            angle = self.rng.uniform(0, 2*np.pi)
            dist = self.rng.uniform(self.size*0.4, self.size*0.48)
            x = int(self.size/2 + np.cos(angle) * dist - w/2)
            y = int(self.size/2 + np.sin(angle) * dist - h/2)
            if 0 <= x < self.size-w and 0 <= y < self.size-h:
                candidates.append((x, y))
        return candidates
    
    def _get_road_locations(self, w: int, h: int, prefer_main: bool = False) -> List[Tuple[int, int]]:
        roads = list(self.streets.main_roads if prefer_main else self.streets.streets)
        if not roads:
            return []
        samples = self.rng.choice(len(roads), min(50, len(roads)), replace=False)
        return [(roads[i][0] - w//2, roads[i][1] - h//2) for i in samples]
    
    def _get_standard_locations(self, w: int, h: int) -> List[Tuple[int, int]]:
        return self._get_road_locations(w, h, prefer_main=False)
    
    def _mark_occupied(self, cx: int, cy: int, w: int, d: int):
        x, y = cx - w//2, cy - d//2
        for dx in range(w):
            for dy in range(d):
                self.occupied.add((x+dx, y+dy))