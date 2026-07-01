import os
import json
import pandas as pd
from pathlib import Path

# Step 1 — Read all 8 country JSON files and combine into one DataFrame
raw_dir = Path("data/raw")
frames = []

name_fixes = {"Drc": "DRC"}

for filepath in sorted(raw_dir.glob("*_tree_cover_loss.json")):
    country_name = filepath.stem.replace("_tree_cover_loss", "").replace("_", " ").title()
    country_name = name_fixes.get(country_name, country_name)
    with open(filepath) as f:
        data = json.load(f)
    df_country = pd.DataFrame(data["data"])
    df_country["country"] = country_name
    frames.append(df_country)

df = pd.concat(frames, ignore_index=True)
print(f"Combined DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(10))

# Step 2 — Sort properly so rolling calculations work correctly
df = df.sort_values(["country", "year"]).reset_index(drop=True)

# Step 3 — Engineer features, calculated within each country separately
df["prev_year_loss"] = (
    df.groupby("country")["loss_area_ha"]
    .shift(1)
)

df["rolling_3yr_avg"] = (
    df.groupby("country")["loss_area_ha"]
    .transform(lambda x: x.rolling(3).mean())
)

df["yoy_change"] = (
    df.groupby("country")["loss_area_ha"]
    .pct_change()
)

# Step 4 — Add country area (km²) for normalisation
country_areas = {
    "Burundi":     27834,
    "DRC":         2344858,
    "Kenya":       580367,
    "Rwanda":      26338,
    "Somalia":     637657,
    "South Sudan": 644329,
    "Tanzania":    945087,
    "Uganda":      241551,
}
df["country_area_km2"] = df["country"].map(country_areas)
df["loss_per_km2"] = df["loss_area_ha"] / df["country_area_km2"]

# Step 5 — Create the label (what we're predicting)
df["high_risk"] = (
    df["loss_area_ha"] > df.groupby("country")["loss_area_ha"].transform("median")
).astype(int)

# Step 6 — Drop rows with NaN (first 2 years per country, from rolling/shift)
df = df.dropna().reset_index(drop=True)

print(f"\nFeature DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(10))
print(f"\nColumns: {list(df.columns)}")
print(f"\nHigh risk distribution:\n{df['high_risk'].value_counts()}")

# Step 7 — Save to processed folder
output_path = "data/processed/features.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved features to {output_path}")
