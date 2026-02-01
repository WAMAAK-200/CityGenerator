import numpy as np
import noise
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter import ttk, messagebox
import random

# --- PATTERNS LOGIC ---
def apply_default_pattern(city_map, centre):
    ci, cj = centre
    city_map[ci, :] = -1
    city_map[:, cj] = -1
    return city_map

def apply_grid_pattern(city_map, size, spacing=10):
    spacing = max(2, spacing) 
    for i in range(0, size, spacing):
        city_map[i, :] = -1
        city_map[:, i] = -1
    return city_map

def apply_radial_pattern(city_map, size, centre, ring_dist=12):
    ci, cj = centre
    ring_dist = max(5, ring_dist)
    
    y, x = np.ogrid[:size, :size]
    dist_map = np.sqrt((x - ci)**2 + (y - cj)**2)
    angle_map = np.arctan2(y - cj, x - ci)
    
    on_ring = (dist_map.astype(int) % ring_dist == 0)
    spoke_angles = [2 * np.pi / 8 * s for s in range(8)]
    on_spoke = np.zeros((size, size), dtype=bool)
    for sa in spoke_angles:
        diff = np.abs(angle_map - sa)
        diff = np.minimum(diff, 2*np.pi - diff) 
        on_spoke |= (diff < 0.04)

    mask = (on_ring | on_spoke) & (dist_map < (size * 0.48))
    city_map[mask] = -1
    return city_map

def apply_fractal_pattern(city_map, size, depth=3):
    def draw_recursive(x0, y0, s, level):
        if level <= 0 or s < 4: return
        mid = s // 2
        city_map[x0+mid, y0:y0+s] = -1
        city_map[x0:x0+s, y0+mid] = -1
        
        new_level = level - 1
        draw_recursive(x0, y0, mid, new_level)
        draw_recursive(x0+mid, y0, mid, new_level)
        draw_recursive(x0, y0+mid, mid, new_level)
        draw_recursive(x0+mid, y0+mid, mid, new_level)

    depth = max(1, min(depth, 5)) 
    draw_recursive(0, 0, size, depth)
    return city_map

def apply_voronoi_pattern(city_map, size, density=20):
    num_cells = max(4, min(density, 100)) 
    points = np.random.randint(0, size, (num_cells, 2))
    y, x = np.ogrid[:size, :size]
    
    dist_sq = (x[..., np.newaxis] - points[:, 0])**2 + (y[..., np.newaxis] - points[:, 1])**2
    nearest_idx = np.argmin(dist_sq, axis=2)
    
    diff_x = np.diff(nearest_idx, axis=1, append=nearest_idx[:, -1:]) != 0
    diff_y = np.diff(nearest_idx, axis=0, append=nearest_idx[-1:, :]) != 0
    
    city_map[diff_x | diff_y] = -1
    return city_map

# --- CITY GENERATOR CLASS ---
class CityGenerator:
    def __init__(self, size=100, seed=42):
        self.size = size
        self.seed = seed
        self.heightmap = None
        self.city_map = None
        self.buildings_data = {}
        self.existing_positions = []
        
        # --- NEW: BUILDING TEMPLATES ---
        # w=width, d=depth, h_mod=height multiplier
        self.building_templates = [
            {"name": "Tiny Hut", "w": 3, "d": 3, "h_mod": 0.5},
            {"name": "Cottage",  "w": 4, "d": 4, "h_mod": 0.8},
            {"name": "House",    "w": 6, "d": 4, "h_mod": 1.0},
            {"name": "Longhall", "w": 8, "d": 3, "h_mod": 0.9},
            {"name": "Estate",   "w": 7, "d": 6, "h_mod": 1.2},
            {"name": "Tower",    "w": 4, "d": 4, "h_mod": 2.5} 
        ]

    def generate_fractal(self, sharpness=1.0):
        self.heightmap = np.zeros((self.size, self.size))
        centre = self.size / 2
        scale = 100.0 
        
        for i in range(self.size):
            for j in range(self.size):
                nx, ny = (i - centre) / scale, (j - centre) / scale
                val = sum((1.0 - abs(noise.pnoise2(nx*f, ny*f, base=self.seed))) * (0.5**k) 
                          for k, f in enumerate([1, 2, 4, 8]))
                self.heightmap[i,j] = val

        self.heightmap = (self.heightmap - self.heightmap.min()) / (self.heightmap.max() - self.heightmap.min())
        self.heightmap = self.heightmap ** sharpness

    def get_local_slope(self, r, c):
        if r <= 0 or r >= self.size - 1 or c <= 0 or c >= self.size - 1:
            return 0.0
        current_h = self.heightmap[r, c]
        neighbors = [
            self.heightmap[r+1, c], self.heightmap[r-1, c],
            self.heightmap[r, c+1], self.heightmap[r, c-1]
        ]
        return max(abs(current_h - n) for n in neighbors)

    def prune_unused_roads(self):
        if not self.buildings_data: return
        rows = [d["pos"][0] for d in self.buildings_data.values()]
        cols = [d["pos"][1] for d in self.buildings_data.values()]
        
        min_r, max_r = max(0, min(rows) - 10), min(self.size, max(rows) + 18)
        min_c, max_c = max(0, min(cols) - 10), min(self.size, max(cols) + 14)

        keep_mask = np.zeros_like(self.city_map, dtype=bool)
        keep_mask[min_r:max_r, min_c:max_c] = True
        self.city_map[(self.city_map == -1) & (~keep_mask)] = 0

    def run_generation(self, pattern_type, max_buildings, pattern_param, min_dist=5.0, sharpness=1.0, avoid_steep=True):
        np.random.seed(self.seed)
        random.seed(self.seed) # Ensure python random is also seeded
        
        self.generate_fractal(sharpness)
        self.city_map = np.zeros((self.size, self.size), dtype=int)
        self.buildings_data = {}
        self.existing_positions = []
        ci, cj = self.size // 2, self.size // 2

        # Apply Pattern
        if pattern_type == "Grid":
            self.city_map = apply_grid_pattern(self.city_map, self.size, spacing=pattern_param)
        elif pattern_type == "Radial":
            self.city_map = apply_radial_pattern(self.city_map, self.size, (ci, cj), ring_dist=pattern_param)
        elif pattern_type == "Fractal":
            self.city_map = apply_fractal_pattern(self.city_map, self.size, depth=pattern_param)
        elif pattern_type == "Polygonal":
            self.city_map = apply_voronoi_pattern(self.city_map, self.size, density=pattern_param)
        else:
            self.city_map = apply_default_pattern(self.city_map, (ci, cj))

        # Identify Lots
        potential_lots = []
        for i in range(2, self.size - 10, 2):
            for j in range(2, self.size - 6, 2):
                dist = np.sqrt((i - ci)**2 + (j - cj)**2)
                gravity = np.exp(-(dist / (self.size * 0.45))**2) 
                score = self.heightmap[i, j] * gravity
                potential_lots.append((score, i, j))
        potential_lots.sort(key=lambda x: x[0], reverse=True)

        placed = 0
        min_dist_sq = min_dist ** 2
        slope_threshold = 0.05 

        for _, base_i, base_j in potential_lots:
            if placed >= max_buildings: break

            # --- 1. JITTER (Random movement) ---
            # If not grid, move the house slightly so they aren't perfectly aligned
            cur_i, cur_j = base_i, base_j
            if pattern_type != "Grid":
                cur_i += random.randint(-2, 2)
                cur_j += random.randint(-2, 2)
                
            # Boundary check after jitter
            if cur_i < 1 or cur_i >= self.size - 10 or cur_j < 1 or cur_j >= self.size - 10:
                continue

            # --- 2. SELECT BUILDING TYPE ---
            # Pick a random template
            b_type = random.choice(self.building_templates)
            w, d = b_type["w"], b_type["d"]

            # --- 3. STEEPNESS CHECK ---
            if avoid_steep:
                if self.get_local_slope(cur_i, cur_j) > slope_threshold:
                    continue

            # --- 4. COLLISION CHECK (Variable Size) ---
            # Check area: from i-1 to i + width + 1, j-1 to j + depth + 1
            r_start, r_end = max(0, cur_i-1), min(self.size, cur_i + w + 1)
            c_start, c_end = max(0, cur_j-1), min(self.size, cur_j + d + 1)
            
            if np.any(self.city_map[r_start:r_end, c_start:c_end] != 0):
                continue

            # --- 5. DISTANCE CHECK ---
            too_close = False
            for ex, ey in self.existing_positions:
                # Euclidean distance check
                if (cur_i - ex)**2 + (cur_j - ey)**2 < min_dist_sq:
                    too_close = True
                    break
            if too_close: continue

            # --- PLACE BUILDING ---
            # Mark map with ID
            self.city_map[cur_i:cur_i+w, cur_j:cur_j+d] = placed + 1
            
            # Store Metadata
            base_height = float(self.heightmap[cur_i, cur_j] * 30)
            final_height = max(1.0, base_height * b_type["h_mod"]) # Towers get taller, huts shorter
            
            self.buildings_data[placed + 1] = {
                "pos": [int(cur_i), int(cur_j)],
                "size": [w, d], # Store dimensions
                "height": final_height,
                "type": b_type["name"]
            }
            self.existing_positions.append((cur_i, cur_j))
            
            # Connect Road (if Default pattern)
            if pattern_type == "Default":
                # Trace back to center or existing road
                curr_trace = cur_i - 1
                while curr_trace != ci and self.city_map[curr_trace, cur_j] >= 0:
                    self.city_map[curr_trace, cur_j] = -1
                    curr_trace += 1 if ci > curr_trace else -1
            
            placed += 1
            
        if pattern_type == "Fractal":
            self.prune_unused_roads()
            
        return self.city_map

    def get_3d_entities(self):
        entities = []
        for b_id, data in self.buildings_data.items():
            i, j = data["pos"]
            w, d = data["size"]
            h = data["height"]
            
            # Position is center of the box. 
            # i is row (x), j is col (z). 
            # If map is i=0..100, j=0..100
            
            entities.append({
                "type": "building",
                "model": data["type"],
                "position": [float(i + w/2), float(h / 2), float(j + d/2)], 
                "scale": [float(w), float(h), float(d)]
            })
            
        for i in range(self.size):
            for j in range(self.size):
                if self.city_map[i, j] == -1:
                    entities.append({
                        "type": "road",
                        "position": [float(i), 0.05, float(j)],
                        "scale": [1.0, 0.1, 1.0]
                    })
        return entities

# --- GUI LOGIC ---
def on_generate():
    try:
        pattern = pattern_var.get()
        b_count = int(count_entry.get())
        seed_val = int(seed_entry.get())
        min_dst = float(dist_entry.get())
        sharpness = float(sharp_entry.get())
        avoid_steep = slope_var.get()
        
        try:
            p_param = int(param_entry.get())
        except ValueError:
            p_param = 15
            
        generator.seed = seed_val
        map_data = generator.run_generation(pattern, b_count, p_param, min_dst, sharpness, avoid_steep)
        
        plt.close('all') 
        plt.figure(figsize=(7, 7))
        display = np.zeros(map_data.shape)
        display[map_data == -1] = 1 
        display[map_data > 0] = 2   
        
        plt.imshow(generator.heightmap, cmap='terrain', alpha=0.6)
        plt.imshow(display, cmap='ocean', alpha=0.5)
        plt.title(f"{pattern} | Sharpness: {sharpness} | Safe: {avoid_steep}")
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

def on_export():
    if not generator.buildings_data:
        messagebox.showwarning("Warning", "Generate a city first!")
        return
    data = generator.get_3d_entities()
    with open("city_3d_export.json", "w") as f:
        json.dump(data, f, indent=4)
    messagebox.showinfo("Success", "City exported to city_3d_export.json")

def update_param_label(event):
    sel = pattern_var.get()
    if sel == "Grid":
        param_label.config(text="Grid Spacing:")
        param_entry.delete(0, tk.END); param_entry.insert(0, "10"); param_entry.config(state='normal')
    elif sel == "Radial":
        param_label.config(text="Ring Dist:")
        param_entry.delete(0, tk.END); param_entry.insert(0, "12"); param_entry.config(state='normal')
    elif sel == "Fractal":
        param_label.config(text="Recursion Depth:")
        param_entry.delete(0, tk.END); param_entry.insert(0, "3"); param_entry.config(state='normal')
    elif sel == "Polygonal":
        param_label.config(text="Road Density (10-100):")
        param_entry.delete(0, tk.END); param_entry.insert(0, "20"); param_entry.config(state='normal')
    else:
        param_label.config(text="Setting (N/A):")
        param_entry.delete(0, tk.END); param_entry.config(state='disabled')

# --- TKINTER SETUP ---
root = tk.Tk()
root.title("CityGen Tool v9 - Varied Housing")
generator = CityGenerator(size=100)

frame_inputs = tk.Frame(root)
frame_inputs.pack(padx=10, pady=10)

# Pattern Select
tk.Label(frame_inputs, text="Pattern Type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
pattern_var = tk.StringVar(value="Default")
pattern_combo = ttk.Combobox(frame_inputs, textvariable=pattern_var, values=["Default", "Grid", "Radial", "Fractal", "Polygonal"], state="readonly")
pattern_combo.grid(row=0, column=1, padx=5, pady=5)
pattern_combo.bind("<<ComboboxSelected>>", update_param_label)

# Pattern Param
param_label = tk.Label(frame_inputs, text="Setting (N/A):")
param_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
param_entry = tk.Entry(frame_inputs)
param_entry.insert(0, "0"); param_entry.config(state='disabled')
param_entry.grid(row=1, column=1, padx=5, pady=5)

# Sharpness Input
tk.Label(frame_inputs, text="Sharpen Terrain:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
sharp_entry = tk.Entry(frame_inputs)
sharp_entry.insert(0, "1.5") 
sharp_entry.grid(row=2, column=1, padx=5, pady=5)

# Min Distance Input
tk.Label(frame_inputs, text="Min House Dist:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
dist_entry = tk.Entry(frame_inputs)
dist_entry.insert(0, "5.0") 
dist_entry.grid(row=3, column=1, padx=5, pady=5)

# Seed
tk.Label(frame_inputs, text="Seed:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
seed_entry = tk.Entry(frame_inputs)
seed_entry.insert(0, "42")
seed_entry.grid(row=4, column=1, padx=5, pady=5)

# Buildings
tk.Label(frame_inputs, text="Max Buildings:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
count_entry = tk.Entry(frame_inputs)
count_entry.insert(0, "200") 
count_entry.grid(row=5, column=1, padx=5, pady=5)

# Slope Checkbox
slope_var = tk.BooleanVar(value=True)
slope_check = tk.Checkbutton(frame_inputs, text="Avoid Steep Terrain", variable=slope_var)
slope_check.grid(row=6, column=0, columnspan=2, pady=5)

frame_btns = tk.Frame(root)
frame_btns.pack(pady=10)
tk.Button(frame_btns, text="Generate Map", command=on_generate, bg="#2ecc71", width=15).pack(side="left", padx=5)
tk.Button(frame_btns, text="Export JSON", command=on_export, bg="#3498db", fg="white", width=15).pack(side="left", padx=5)

root.mainloop()