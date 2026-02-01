import numpy as np
import noise
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter import ttk, messagebox
import patterns

class CityGenerator:
    def __init__(self, size=100, seed=42):
        self.size = size
        self.seed = seed
        self.heightmap = None
        self.city_map = None
        self.buildings_data = {}

    def generate_fractal(self):
        self.heightmap = np.zeros((self.size, self.size))
        centre = self.size / 2
        for i in range(self.size):
            for j in range(self.size):
                nx, ny = (i - centre) / 30.0, (j - centre) / 30.0
                val = sum((1.0 - abs(noise.pnoise2(nx*f, ny*f, base=self.seed))) * (0.5**k) 
                          for k, f in enumerate([1, 2, 4, 8]))
                self.heightmap[i,j] = val**2
        self.heightmap = (self.heightmap - self.heightmap.min()) / self.heightmap.max()

    def run_generation(self, pattern_type, max_buildings):
        self.generate_fractal()
        self.city_map = np.zeros((self.size, self.size), dtype=int)
        self.buildings_data = {}
        ci, cj = self.size // 2, self.size // 2

        if pattern_type == "Grid":
            self.city_map = patterns.apply_grid_pattern(self.city_map, self.size)
        elif pattern_type == "Radial":
            self.city_map = patterns.apply_radial_pattern(self.city_map, self.size, (ci, cj))
        else:
            self.city_map = patterns.apply_default_pattern(self.city_map, (ci, cj))

        potential_lots = []
        for i in range(2, self.size - 10, 2):
            for j in range(2, self.size - 6, 2):
                dist = np.sqrt((i - ci)**2 + (j - cj)**2)
                gravity = np.exp(-(dist / (self.size * 0.25))**2) # Kill edge-clustering
                score = self.heightmap[i, j] * gravity
                potential_lots.append((score, i, j))
        
        potential_lots.sort(key=lambda x: x[0], reverse=True)

        placed = 0
        for _, i, j in potential_lots:
            if placed >= max_buildings: break
            if np.all(self.city_map[i:i+8, j:j+4] == 0):
                self.city_map[i:i+8, j:j+4] = placed + 1
                self.buildings_data[placed + 1] = {
                    "pos": [int(i), int(j)],
                    "height": float(self.heightmap[i,j] * 30)
                }
                if pattern_type == "Default":
                    curr_i = i - 1
                    while curr_i != ci and self.city_map[curr_i, j] >= 0:
                        self.city_map[curr_i, j] = -1
                        curr_i += 1 if ci > curr_i else -1
                placed += 1
        return self.city_map

    def get_3d_entities(self):
        entities = []
        # Buildings
        for b_id, data in self.buildings_data.items():
            i, j = data["pos"]
            entities.append({
                "type": "building",
                "position": [float(i + 4), float(data["height"] / 2), float(j + 2)], # centreed pivot
                "scale": [8.0, float(data["height"]), 4.0]
            })
        # Roads
        for i in range(self.size):
            for j in range(self.size):
                if self.city_map[i, j] == -1:
                    entities.append({
                        "type": "road",
                        "position": [float(i), 0.05, float(j)],
                        "scale": [1.0, 0.1, 1.0]
                    })
        return entities

# GUI UI Logic
def on_generate():
    try:
        pattern = pattern_var.get()
        b_count = int(count_entry.get())
        generator.seed = int(seed_entry.get())
        map_data = generator.run_generation(pattern, b_count)
        
        plt.close('all') 
        plt.figure(figsize=(7, 7))
        display = np.zeros(map_data.shape)
        display[map_data == -1] = 1 
        display[map_data > 0] = 2   
        plt.imshow(display, cmap='ocean')
        plt.title(f"Seed: {generator.seed} | {pattern}")
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for Seed and Building Count")

def on_export():
    if not generator.buildings_data:
        messagebox.showwarning("Warning", "Generate a city first!")
        return
    data = generator.get_3d_entities()
    with open("city_3d_export.json", "w") as f:
        json.dump(data, f, indent=4)
    messagebox.showinfo("Success", "City exported to city_3d_export.json")

# Tkinter Setup
root = tk.Tk()
root.title("CityGen Hackathon Tool")
generator = CityGenerator(size=100)

# Pattern Select
tk.Label(root, text="Urban Pattern:").grid(row=0, column=0, padx=10, pady=5)
pattern_var = tk.StringVar(value="Default")
ttk.Combobox(root, textvariable=pattern_var, values=["Default", "Grid", "Radial"]).grid(row=0, column=1)

# Seed Input
tk.Label(root, text="Seed:").grid(row=1, column=0, padx=10, pady=5)
seed_entry = tk.Entry(root)
seed_entry.insert(0, "42")
seed_entry.grid(row=1, column=1)

# Count Input
tk.Label(root, text="Buildings:").grid(row=2, column=0, padx=10, pady=5)
count_entry = tk.Entry(root)
count_entry.insert(0, "40")
count_entry.grid(row=2, column=1)

# Buttons
tk.Button(root, text="Generate Map", command=on_generate, bg="#2ecc71").grid(row=3, column=0, pady=10, sticky="e")
tk.Button(root, text="Export JSON", command=on_export, bg="#3498db", fg="white").grid(row=3, column=1, pady=10, sticky="w")

root.mainloop()