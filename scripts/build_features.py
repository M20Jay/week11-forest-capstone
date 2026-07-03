# scripts/build_features.py
# Reads all 8 country JSON files, engineers features, saves to parquet

import json
import pandas as pd
from pathlib import Path
from src.forest.config import RAW_DATA_DIR, FEATURES_PATH
from src.forest.features import engineer_features

# Step 1 — Load all 8 country JSON files
name_fixes = {"Drc": "DRC"}
frames = []

for filepath in sorted(Path(RAW_DATA_DIR).glob("*_tree_cover_loss.json")):
    country_name = (
        filepath.stem
        .replace("_tree_cover_loss", "")
        .replace("_", " ")
        .title()
    )
    country_name = name_fixes.get(country_name, country_name)
    with open(filepath) as f:
        data = json.load(f)
    df_country = pd.DataFrame(data["data"])
    df_country["country"] = country_name
    frames.append(df_country)

df = pd.concat(frames, ignore_index=True)
print(f"Combined DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2 — Engineer features
df = engineer_features(df)
print(f"Feature DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(10))
print(f"\nColumns: {list(df.columns)}")
print(f"\nHigh risk distribution:\n{df['high_risk'].value_counts()}")

# Step 3 — Save to processed folder
Path(FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(FEATURES_PATH, index=False)
print(f"\nSaved features to {FEATURES_PATH}")
