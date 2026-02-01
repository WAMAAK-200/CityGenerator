using UnityEngine;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json; // Ensure you have Newtonsoft installed via Package Manager

public class CityLoader : MonoBehaviour {
    public GameObject buildingPrefab;
    public GameObject roadPrefab;
    public string jsonFileName = "city_3d_export.json";

    [System.Serializable]
    public class EntityData {
        public string type;
        public float[] position;
        public float[] scale;
    }

    void Start() {
        // Look in the project root or StreamingAssets
        string fullPath = Path.Combine(Application.dataPath, "../../" + jsonFileName); 
        if (!File.Exists(fullPath)) fullPath = Path.Combine(Application.streamingAssetsPath, jsonFileName);

        string jsonString = File.ReadAllText(fullPath);
        List<EntityData> cityData = JsonConvert.DeserializeObject<List<EntityData>>(jsonString);

        foreach (var entity in cityData) {
            Vector3 worldPos = new Vector3(entity.position[0], entity.position[1], entity.position[2]);
            Vector3 worldScale = new Vector3(entity.scale[0], entity.scale[1], entity.scale[2]);
            
            GameObject prefab = (entity.type == "building") ? buildingPrefab : roadPrefab;
            
            if (prefab != null) {
                GameObject instance = Instantiate(prefab, worldPos, Quaternion.identity, this.transform);
                instance.transform.localScale = worldScale;
            }
        }
    }
}