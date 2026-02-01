# generator.py
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from geom_utils import generate_random_points, centroid_region
from terrain import is_land
import config

class CityGenerator:
    def __init__(self):
        self.points = generate_random_points(config.NUM_POINTS, config.WIDTH, config.HEIGHT)
        # We will store dictionaries now: {'poly': shape, 'type': 'house'}
        self.blocks = [] 

    def relax_points(self):
        vor = Voronoi(self.points)
        new_points = []
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_points.append(self.points[i])
                continue
            vertices = [vor.vertices[i] for i in region]
            cx, cy = centroid_region(np.array(vertices))
            new_points.append([cx, cy])
        self.points = np.array(new_points)

    def generate_map(self):
        # 1. Relax
        for _ in range(config.RELAXATION_STEPS):
            self.relax_points()
            
        # 2. Final Voronoi
        vor = Voronoi(self.points)
        
        raw_blocks = []
        
        # 3. Create Polygons
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region: continue
            
            vertices = [vor.vertices[v] for v in region]
            poly = Polygon(vertices)
            
            # Check center of polygon for Land validity
            cx, cy = poly.centroid.x, poly.centroid.y
            if is_land(cx, cy) and poly.area > config.MIN_BLOCK_SIZE:
                raw_blocks.append(poly)

        # 4. Assign Types (Zoning)
        # Sort blocks by distance to center
        raw_blocks.sort(key=lambda p: (p.centroid.x - config.WIDTH/2)**2 + (p.centroid.y - config.HEIGHT/2)**2)

        if not raw_blocks: return

        # A. The Castle (The central block)
        self.blocks.append({'poly': raw_blocks[0], 'type': 'castle'})
        
        # B. Plazas (The immediate neighbors of the castle)
        # We simply skip adding them to the list to leave an empty hole (Plaza)
        # OR we add them as specific 'plaza' type if we want to draw pavement.
        # Let's skip the next 2 blocks to make an open courtyard.
        
        # C. The Rest (Houses)
        for poly in raw_blocks[3:]:
            # Main Road Logic:
            # If a block is close to the horizontal center line, make it smaller (wider street)
            is_main_road = abs(poly.centroid.y - config.HEIGHT/2) < 20
            
            gap = config.MAIN_ROAD_GAP if is_main_road else config.STREET_GAP
            
            shrunk_poly = poly.buffer(-gap)
            if not shrunk_poly.is_empty:
                self.blocks.append({'poly': shrunk_poly, 'type': 'district'})