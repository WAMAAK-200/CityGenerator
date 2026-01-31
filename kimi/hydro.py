import numpy as np
from scipy.ndimage import distance_transform_edt, label, binary_dilation, maximum_filter
from typing import List, Tuple, Optional
from dataclasses import dataclass
from perlin import TerrainData
import noise

@dataclass
class HydroData:
    """Water overlay data - completely separate from terrain"""
    water_mask: np.ndarray          # All water
    river_mask: np.ndarray          # Flowing water (bridgeable)
    lake_mask: np.ndarray           # Standing water (not bridgeable)
    ocean_mask: np.ndarray          # Ocean water (not bridgeable)
    swamp_mask: np.ndarray          # Wetlands
    coast_mask: np.ndarray          # Beach/transition zones
    delta_mask: np.ndarray          # River deltas
    island_mask: np.ndarray         # Islands in rivers
    water_level: float
    river_paths: List[np.ndarray]   # For rendering paths
    
    def get_bridgeable(self) -> np.ndarray:
        """Where roads can cross: rivers yes, lakes/ocean no"""
        return self.river_mask & ~self.lake_mask & ~self.ocean_mask


class WaterLayer:
    """Adds hydrology to existing terrain"""
    
    def __init__(self, terrain: TerrainData, water_level: float = 0.15):
        self.base = terrain
        self.height = terrain.heightmap.copy()
        self.size = terrain.size
        self.wl = water_level
        self.seed = terrain.seed
        
        # Masks start empty
        self.water = np.zeros((self.size, self.size), dtype=bool)
        self.river = np.zeros((self.size, self.size), dtype=bool)
        self.lake = np.zeros((self.size, self.size), dtype=bool)
        self.ocean = np.zeros((self.size, self.size), dtype=bool)
        self.swamp = np.zeros((self.size, self.size), dtype=bool)
        self.coast = np.zeros((self.size, self.size), dtype=bool)
        self.delta = np.zeros((self.size, self.size), dtype=bool)
        self.island = np.zeros((self.size, self.size), dtype=bool)
        self.paths: List[np.ndarray] = []
    
    def add_ocean(self, width: float = 0.12):
        """Create ocean at edges with beaches"""
        y, x = np.ogrid[:self.size, :self.size]
        
        # Distance to nearest edge
        dx = np.minimum(x, self.size - 1 - x)
        dy = np.minimum(y, self.size - 1 - y)
        dist = np.minimum(dx, dy)
        
        # Falloff curve
        limit = self.size * width
        t = np.clip(1 - dist / limit, 0, 1)
        
        # Lower terrain at edges
        self.height -= t * 0.4
        
        # Mark water
        self.ocean = self.height < self.wl
        self.water |= self.ocean
        
        # Beach is transition zone
        edge_dist = distance_transform_edt(~self.ocean)
        self.coast = (edge_dist <= 6) & ~self.ocean
    
    def add_valley_river(self, meander: float = 0.5):
        """River that follows valley floor (meanders horizontally)"""
        xs = np.arange(self.size)
        y_center = np.full(self.size, self.size // 2, dtype=float)
        
        # Perlin meander
        for o in range(3):
            freq = 0.008 * (2 ** o)
            amp = 30 * (meander ** o)
            for i in range(self.size):
                y_center[i] += noise.pnoise1(xs[i] * freq + self.seed, base=self.seed) * amp
        
        y_center = np.clip(y_center, 10, self.size-10).astype(int)
        path = np.column_stack([xs, y_center])
        
        self._carve(path, width=3, depth=0.1)
        self.paths.append(path)
        
        # Check for delta at end
        if self.ocean[path[-1, 1], path[-1, 0]]:
            self._make_delta(path[-10:], width=6)
    
    def add_rivers(self, count: int = 2):
        """Spawn rivers from high points, flow to ocean"""
        # Find peaks
        peaks = (self.height > 0.65) & (maximum_filter(self.height, size=10) == self.height)
        py, px = np.where(peaks)
        
        if len(py) == 0:
            # Fallback
            starts = [(self.size//4, self.size//4), (3*self.size//4, self.size//3)]
        else:
            # Take highest 2
            idx = np.argsort(self.height[py, px])[-count:]
            starts = [(int(px[i]), int(py[i])) for i in idx]
        
        for sx, sy in starts:
            path = self._flow(sx, sy)
            if len(path) > 30:
                self._carve(np.array(path), width=2, depth=0.08)
                self.paths.append(np.array(path))
                
                # Delta if reached ocean
                ex, ey = int(path[-1][0]), int(path[-1][1])
                if self.ocean[ey, ex]:
                    self._make_delta(np.array(path[-6:]), width=4)
    
    def add_lakes(self):
        """Fill depressions with water"""
        # Low areas not connected to ocean
        low = (self.height < self.wl + 0.05) & ~self.ocean
        labeled, n = label(low)
        
        for i in range(1, n+1):
            region = (labeled == i)
            size = np.sum(region)
            if 100 < size < 800:
                self.lake |= region
                self.height[region] = self.wl - 0.02
        
        self.water |= self.lake
    
    def add_swamps(self):
        """Wet ground near water"""
        if not np.any(self.water):
            return
            
        dist = distance_transform_edt(~self.water)
        near = (dist < 10) & ~self.water
        
        # Flat and low
        gy, gx = np.gradient(self.height)
        flat = np.sqrt(gx**2 + gy**2) < 0.02
        low = (self.height > self.wl) & (self.height < self.wl + 0.08)
        
        self.swamp = near & flat & low
        
        # River-lake interfaces especially swampy
        if np.any(self.river) and np.any(self.lake):
            interface = binary_dilation(self.river) & binary_dilation(self.lake)
            self.swamp |= interface
    
    def add_islands(self):
        """Sandbars in rivers"""
        if not np.any(self.river):
            return
            
        for path in self.paths:
            if len(path) < 50:
                continue
            
            # Every ~80 pixels along river
            for i in range(30, len(path)-30, 80):
                cx, cy = int(path[i, 0]), int(path[i, 1])
                # Small mound
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dx*dx + dy*dy <= 9:
                            x, y = cx+dx, cy+dy
                            if 0 <= x < self.size and 0 <= y < self.size:
                                self.height[y, x] = self.wl + 0.05
                                self.island[y, x] = True
        
        self.water &= ~self.island
        self.river &= ~self.island
    
    def _flow(self, sx: int, sy: int) -> List[Tuple[int, int]]:
        """Walk downhill with meandering"""
        path = [(sx, sy)]
        x, y = sx, sy
        mom = np.array([0.0, 0.0])
        
        for _ in range(1500):
            if not (0 < x < self.size-1 and 0 < y < self.size-1):
                break
            
            if self.ocean[y, x]:
                break
            
            gy, gx = np.gradient(self.height)
            grad = np.array([gx[y, x], gy[y, x]])
            norm = np.linalg.norm(grad)
            
            if norm < 1e-8:
                break
            
            grad = grad / norm
            
            # Meander perpendicular to flow
            perp = np.array([-grad[1], grad[0]])
            wander = perp * (np.random.rand() - 0.5) * 0.8
            
            mom = mom * 0.3 + (grad + wander) * 0.7
            mom = mom / (np.linalg.norm(mom) + 1e-10)
            
            step = 1.5
            nx = int(x + mom[0] * step)
            ny = int(y + mom[1] * step)
            
            if (nx, ny) == (x, y):
                nx = int(x + mom[0])
                ny = int(y + mom[1])
            
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                break
            
            x, y = nx, ny
            path.append((x, y))
        
        return path
    
    def _carve(self, path: np.ndarray, width: int, depth: float):
        """Cut river channel"""
        for x, y in path:
            x, y = int(x), int(y)
            w = width
            
            for dy in range(-w, w+1):
                for dx in range(-w, w+1):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < self.size and 0 <= yy < self.size:
                        d = np.sqrt(dx*dx + dy*dy)
                        if d <= w:
                            self.height[yy, xx] = min(self.height[yy, xx], 
                                                     self.wl - 0.02)
                            if d < w * 0.6:
                                self.river[yy, xx] = True
        
        self.water |= self.river
    
    def _make_delta(self, mouth: np.ndarray, width: int):
        """Fan out at river mouth"""
        if len(mouth) == 0:
            return
        
        cx, cy = int(mouth[-1, 0]), int(mouth[-1, 1])
        r = width * 2
        
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    x, y = cx+dx, cy+dy
                    if 0 <= x < self.size and 0 <= y < self.size:
                        self.height[y, x] = self.wl + 0.01
                        self.delta[y, x] = True
                        if np.random.rand() < 0.3:
                            self.river[y, x] = True
                            self.water[y, x] = True
    
    def export(self) -> HydroData:
        """Package results"""
        return HydroData(
            water_mask=self.water.copy(),
            river_mask=self.river.copy(),
            lake_mask=self.lake.copy(),
            ocean_mask=self.ocean.copy(),
            swamp_mask=self.swamp.copy(),
            coast_mask=self.coast.copy(),
            delta_mask=self.delta.copy(),
            island_mask=self.island.copy(),
            water_level=self.wl,
            river_paths=[p.copy() for p in self.paths]
        )
    
    def get_height(self) -> np.ndarray:
        """Modified terrain height (carved)"""
        return self.height.copy()


def add_water(terrain: TerrainData, ocean: bool = True, rivers: bool = True,
              lakes: bool = True, swamps: bool = True, islands: bool = True) -> Tuple[np.ndarray, HydroData]:
    """
    Apply full hydrology to terrain.
    Returns: (modified_heightmap, hydro_data)
    """
    layer = WaterLayer(terrain)
    
    if ocean:
        layer.add_ocean()
    
    if rivers:
        if terrain.biome == 'valley':
            layer.add_valley_river()
        else:
            count = 2 if terrain.biome in ['mountains', 'mountains_hydro'] else 1
            layer.add_rivers(count=count)
    
    if lakes:
        layer.add_lakes()
    
    if swamps:
        layer.add_swamps()
    
    if islands:
        layer.add_islands()
    
    return layer.get_height(), layer.export()