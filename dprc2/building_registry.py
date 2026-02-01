import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

@dataclass
class BuildingType:
    name: str = "Unknown"
    min_population: int = 100
    rarity: int = 5
    size_range: Tuple[int, int] = (1, 2)
    height_range: Tuple[float, float] = (2.0, 4.0)
    priority: int = 5
    color: List[int] = field(default_factory=lambda: [128, 128, 128])
    requires_wall: bool = False
    requires_water: bool = False
    max_slope: float = 0.3
    requires_flat: bool = False
    district: str = "common"
    road_access: str = "normal"

class BuildingRegistry:
    def __init__(self, json_path: Optional[str] = None):
        self.buildings: Dict[str, BuildingType] = {}
        
        # FIX: Default to buildings.json in same folder as this file
        if json_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(module_dir, "buildings.json")
        
        if os.path.exists(json_path):
            self.load_from_json(json_path)
        else:
            print(f"Warning: {json_path} not found, using defaults")
            self._load_defaults()
    
    def _load_defaults(self):
        """Fallback if JSON missing"""
        defaults = [
            BuildingType("Castle", 5000, 1, (10, 14), (15, 25), 1, [100, 105, 110], True),
            BuildingType("Cathedral", 3000, 1, (8, 12), (20, 35), 2, [210, 205, 180]),
            BuildingType("House", 100, 10, (2, 3), (4, 6), 8, [170, 150, 120]),
        ]
        for b in defaults:
            self.register(b)
    
    def load_from_json(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            for b in data['buildings']:
                raw_color = b.get('color', [128, 128, 128])
                if not isinstance(raw_color, list) or len(raw_color) != 3:
                    raw_color = [128, 128, 128]
                
                building = BuildingType(
                    name=b['name'],
                    min_population=b.get('min_population', 100),
                    rarity=b.get('rarity', 5),
                    size_range=tuple(b.get('size_range', [1, 2])),
                    height_range=tuple(b.get('height_range', [2.0, 4.0])),
                    priority=b.get('priority', 5),
                    color=raw_color,
                    requires_wall=b.get('requires_wall', False),
                    requires_water=b.get('requires_water', False),
                    max_slope=b.get('max_slope', 0.3),
                    requires_flat=b.get('requires_flat', False),
                    district=b.get('district', 'common'),
                    road_access=b.get('road_access', 'normal')
                )
                self.register(building)
    
    def register(self, building: BuildingType):
        self.buildings[building.name] = building
    
    def get_by_priority(self) -> List[BuildingType]:
        return sorted(self.buildings.values(), key=lambda b: b.priority)