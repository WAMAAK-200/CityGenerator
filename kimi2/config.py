from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

class TerrainType(Enum):
    PLAINS = "plains"
    MOUNTAINS = "mountains"
    COASTAL = "coastal"
    PLATEAU = "plateau"
    VALLEY = "valley"
    ISLAND = "island"
    CANYON = "canyon"
    HILLS = "hills"

@dataclass
class TerrainConfig:
    seed: int = 42
    size: int = 256
    terrain_type: TerrainType = TerrainType.MOUNTAINS
    water_level: float = 0.25
    max_height: float = 60.0
    scale: float = 25.0
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    
    # Biome-specific params (auto-set by preset)
    biome_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Auto-configure biome-specific settings"""
        presets = {
            TerrainType.PLAINS: {
                'scale': 40.0, 'octaves': 4, 'max_height': 15.0,
                'persistence': 0.3, 'water_level': 0.15
            },
            TerrainType.MOUNTAINS: {
                'scale': 20.0, 'octaves': 6, 'max_height': 100.0,
                'persistence': 0.5, 'water_level': 0.25
            },
            TerrainType.COASTAL: {
                'scale': 30.0, 'octaves': 4, 'max_height': 30.0,
                'persistence': 0.4, 'water_level': 0.35
            },
            TerrainType.PLATEAU: {
                'scale': 35.0, 'octaves': 5, 'max_height': 80.0,
                'persistence': 0.5, 'water_level': 0.2
            },
            TerrainType.VALLEY: {
                'scale': 25.0, 'octaves': 5, 'max_height': 50.0,
                'persistence': 0.4, 'water_level': 0.3
            },
            TerrainType.ISLAND: {
                'scale': 20.0, 'octaves': 5, 'max_height': 40.0,
                'persistence': 0.5, 'water_level': 0.25
            },
            TerrainType.CANYON: {
                'scale': 15.0, 'octaves': 6, 'max_height': 70.0,
                'persistence': 0.6, 'water_level': 0.2
            },
            TerrainType.HILLS: {
                'scale': 30.0, 'octaves': 5, 'max_height': 35.0,
                'persistence': 0.4, 'water_level': 0.2
            }
        }
        
        if self.terrain_type in presets and self.biome_params is None:
            preset = presets[self.terrain_type]
            for key, value in preset.items():
                setattr(self, key, value)