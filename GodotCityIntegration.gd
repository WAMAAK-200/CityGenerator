extends Node3D

@export var building_prefab: PackedScene
@export var road_prefab: PackedScene
@export var json_path: String = "res://city_3d_export.json"

func _ready():
	generate_city()

func generate_city():
	if not FileAccess.file_exists(json_path):
		print("JSON file not found at: ", json_path)
		return

	var file = FileAccess.open(json_path, FileAccess.READ)
	var entities = JSON.parse_string(file.get_as_text())
	
	for entity in entities:
		var prefab = building_prefab if entity["type"] == "building" else road_prefab
		
		if prefab:
			var instance = prefab.instantiate()
			add_child(instance)
			
			var p = entity["position"]
			var s = entity["scale"]
			
			# Python exports centered coordinates, so we apply them directly
			instance.position = Vector3(p[0], p[1], p[2])
			instance.scale = Vector3(s[0], s[1], s[2])