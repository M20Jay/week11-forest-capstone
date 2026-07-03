# src/forest/features.py
# Reusable feature engineering functions
# Imported by scripts/ingest_gfw_data.py and scripts/build_features.py

import pandas as pd
from src.forest.config import COUNTRY_AREAS


def build_geometry(bbox):
    """Build a GeoJSON Polygon from a [west, south, east, north] bounding box."""
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


def engineer_features(df):
    """
    Takes a combined DataFrame with columns [year, loss_area_ha, country]
    and returns a DataFrame with all engineered features and the high_risk label.
    Drops first 2 years per country (NaN from rolling/shift calculations).
    """
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Lag and rolling features — calculated within each country
    df["prev_year_loss"] = (
        df.groupby("country")["loss_area_ha"].shift(1)
    )
    df["rolling_3yr_avg"] = (
        df.groupby("country")["loss_area_ha"]
        .transform(lambda x: x.rolling(3).mean())
    )
    df["yoy_change"] = (
        df.groupby("country")["loss_area_ha"].pct_change()
    )

    # Country-level normalisation
    df["country_area_km2"] = df["country"].map(COUNTRY_AREAS)
    df["loss_per_km2"] = df["loss_area_ha"] / df["country_area_km2"]

    # Binary label — above each country's own median = high risk
    df["high_risk"] = (
        df["loss_area_ha"] > df.groupby("country")["loss_area_ha"]
        .transform("median")
    ).astype(int)

    # Drop NaN rows (first 2 years per country)
    df = df.dropna().reset_index(drop=True)

    return df
