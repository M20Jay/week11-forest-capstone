import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GFW_API_KEY")
print(f"API key loaded: {api_key}")

headers = {
    "x-api-key": api_key,
    "Origin": "https://martin-mlops.com"
}
print(f"Headers built: {headers}")

countries = {
    "Kenya":       [33.9, -4.7, 41.9, 4.6],
    "Tanzania":    [29.3, -11.7, 40.3, -0.95],
    "Uganda":      [29.6, -1.5, 35.0, 4.2],
    "Rwanda":      [28.9, -2.9, 30.9, -1.0],
    "DRC":         [12.2, -13.3, 31.3, 5.4],
    "Burundi":     [29.0, -4.5, 30.8, -2.3],
    "South Sudan": [23.9, 3.5, 35.3, 12.2],
    "Somalia":     [41.0, -1.7, 51.1, 12.0],
}
def build_geometry(bbox):
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]]
    }

for country_name, bbox in countries.items():
    geometry = build_geometry(bbox)
    print(f"Geometry built for {country_name}")

sql = """
    SELECT
        umd_tree_cover_loss__year AS year,
        SUM(area__ha) AS loss_area_ha
    FROM results
    WHERE umd_tree_cover_density_2000__threshold = 30
      AND umd_tree_cover_loss__year >= 2001
    GROUP BY umd_tree_cover_loss__year
    ORDER BY umd_tree_cover_loss__year
"""

url = "https://data-api.globalforestwatch.org/dataset/umd_tree_cover_loss/v1.10/query/json"

for country_name, bbox in countries.items():
    geometry = build_geometry(bbox)
    print(f"\nProcessing {country_name}...")

    response = requests.post(
        url,
        headers=headers,
        json={"sql": sql, "geometry": geometry}
    )

    print(f"  Status code: {response.status_code}")

    if response.status_code == 200:
        filename = f"data/raw/{country_name.lower().replace(' ', '_')}_tree_cover_loss.json"
        with open(filename, "w") as f:
            json.dump(response.json(), f, indent=2)
        print(f"  Saved to {filename}")
    else:
        print(f"  Did not save {country_name} — request was not successful")
