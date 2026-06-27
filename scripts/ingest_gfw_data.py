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

geometry = {
    "type": "Polygon",
    "coordinates": [[
        [33.9, -4.7],
        [41.9, -4.7],
        [41.9, 4.6],
        [33.9, 4.6],
        [33.9, -4.7],
    ]]
}
print("Geometry built for Kenya bounding box")

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
print("SQL query built")

url = "https://data-api.globalforestwatch.org/dataset/umd_tree_cover_loss/v1.10/query/json"

response = requests.post(
    url,
    headers=headers,
    json={"sql": sql, "geometry": geometry}
)

print(f"Status code: {response.status_code}")
print(response.json())

if response.status_code == 200:
    with open("data/raw/kenya_tree_cover_loss.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    print("Saved raw response to data/raw/kenya_tree_cover_loss.json")
else:
    print("Did not save — request was not successful")

