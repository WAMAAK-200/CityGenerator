from generator import CityGenerator
from visualizer import draw_city

def main():
    print("Initializing City Generator...")
    city = CityGenerator()
    
    print("Relaxing Voronoi grid (Lloyd's Algorithm)...")
    city.generate_map()
    
    # Updated: accessed 'blocks' instead of 'land_polygons'
    print(f"Generated {len(city.blocks)} city blocks.")
    
    draw_city(city)

if __name__ == "__main__":
    main()