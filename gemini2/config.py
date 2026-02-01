# config.py

# Canvas
WIDTH = 1000
HEIGHT = 1000

# Generation
NUM_POINTS = 1500        
RELAXATION_STEPS = 3     
LAKE_THRESHOLD = 0.85    

# Features
RIVER_WIDTH = 40.0       # Width of the river strip
MIN_BLOCK_SIZE = 100.0   # Filter out tiny broken polygons

# Visuals
BG_COLOR = "#dcbfa6"     # Parchment-like background (Road Color)
LAND_COLOR = "#e0cda6"   # Slightly lighter block color
WATER_COLOR = "#7ca1bf"  # Blue river
CASTLE_COLOR = "#2b2b2b" # Dark grey/black for the castle (like the image)

STREET_GAP = 2.0         # Standard alley width
MAIN_ROAD_GAP = 8.0      # Main road width