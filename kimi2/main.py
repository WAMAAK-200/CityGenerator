from config import TerrainConfig, TerrainType
from terrain_generator import TerrainGenerator
from hydrology import HydrologyGenerator
from city_generator import CityGenerator, LayoutType

def main():
    print("=== Medieval City Generator MVP ===\n")
    
    # 1. Generate terrain
    print("1. TERRAIN GENERATION")
    t_config = TerrainConfig(seed=42, size=256, terrain_type=TerrainType.PLAINS)
    terrain = TerrainGenerator(t_config).generate()
    
    hydro_config = HydrologyGenerator(seed=42, river_count=2, water_level=0.25)
    hydro = hydro_config.generate(terrain)
    
    # 2. Get population from user
    print("\n2. CITY CONFIGURATION")
    layout = input("Layout type (organic/grid/fractal): ").strip()
    layout_enum = LayoutType(layout)
    
    # NEW: Population prompt
    while True:
        pop_input = input("Enter population (500-20000, affects building count): ").strip()
        try:
            population = int(pop_input)
            if 500 <= population <= 20000:
                break
            else:
                print("Please enter a value between 500 and 20000")
        except ValueError:
            print("Please enter a valid number")
    
    has_walls = input("Include city walls? (y/n): ").strip().lower() == 'y'
    
    # 3. Generate city
    print("\n3. GENERATING CITY...")
    city = CityGenerator(size=256, seed=123, terrain=terrain, hydrology=hydro)
    
    buildings = city.generate(
        layout=layout_enum,
        population=population,
        has_walls=has_walls
    )
    
    print(f"\nGenerated {len(buildings)} buildings for population of {population}:")
    by_type = {}
    for b in buildings:
        by_type[b.building_type.name] = by_type.get(b.building_type.name, 0) + 1
    for name, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    
    # 4. Visualize
    print("\n4. VISUALIZATION")
    from visualizer import CityVisualizer
    viz = CityVisualizer(terrain, hydro, city)
    viz.show_city_map()

if __name__ == "__main__":
    main()