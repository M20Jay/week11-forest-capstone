# tests/test_features.py
# Tests for src/forest/features.py and src/forest/config.py

import pytest
import pandas as pd
import numpy as np
from src.forest.features import build_geometry, engineer_features
from src.forest.config import COUNTRIES, COUNTRY_AREAS, FEATURE_COLUMNS, TARGET_COLUMN


# ── build_geometry tests ──────────────────────────────────────────

def test_build_geometry_returns_dict():
    bbox = [33.9, -4.7, 41.9, 4.6]
    result = build_geometry(bbox)
    assert isinstance(result, dict)

def test_build_geometry_type_is_polygon():
    bbox = [33.9, -4.7, 41.9, 4.6]
    result = build_geometry(bbox)
    assert result["type"] == "Polygon"

def test_build_geometry_closes_polygon():
    bbox = [33.9, -4.7, 41.9, 4.6]
    coords = build_geometry(bbox)["coordinates"][0]
    assert coords[0] == coords[-1], "First and last coordinate must match to close polygon"

def test_build_geometry_has_five_points():
    bbox = [33.9, -4.7, 41.9, 4.6]
    coords = build_geometry(bbox)["coordinates"][0]
    assert len(coords) == 5

def test_build_geometry_west_is_leftmost():
    west, south, east, north = 33.9, -4.7, 41.9, 4.6
    coords = build_geometry([west, south, east, north])["coordinates"][0]
    all_lons = [c[0] for c in coords]
    assert min(all_lons) == west

def test_build_geometry_drc_largest():
    kenya_bbox = COUNTRIES["Kenya"]
    drc_bbox   = COUNTRIES["DRC"]
    kenya_width = kenya_bbox[2] - kenya_bbox[0]
    drc_width   = drc_bbox[2]   - drc_bbox[0]
    assert drc_width > kenya_width, "DRC bounding box must be wider than Kenya"


# ── engineer_features tests ───────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal 2-country DataFrame for testing feature engineering."""
    rows = []
    for country in ["Kenya", "Rwanda"]:
        for year in range(2001, 2011):
            rows.append({
                "year": year,
                "loss_area_ha": float(1000 + year * 10),
                "country": country
            })
    return pd.DataFrame(rows)

def test_engineer_features_returns_dataframe(sample_df):
    result = engineer_features(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_engineer_features_correct_columns(sample_df):
    result = engineer_features(sample_df)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in result.columns, f"Missing column: {col}"

def test_engineer_features_no_nulls(sample_df):
    result = engineer_features(sample_df)
    assert result.isnull().sum().sum() == 0, "No NaN values should remain after dropna"

def test_engineer_features_drops_first_two_years(sample_df):
    result = engineer_features(sample_df)
    # 2 countries × 10 years = 20 rows, minus 2 NaN rows per country = 16
    assert len(result) == 16

def test_engineer_features_high_risk_is_binary(sample_df):
    result = engineer_features(sample_df)
    unique_values = set(result[TARGET_COLUMN].unique())
    assert unique_values == {0, 1}, "high_risk must contain only 0 and 1"

def test_engineer_features_country_area_mapped(sample_df):
    result = engineer_features(sample_df)
    kenya_area = result[result["country"] == "Kenya"]["country_area_km2"].unique()
    assert len(kenya_area) == 1
    assert kenya_area[0] == COUNTRY_AREAS["Kenya"]

def test_engineer_features_loss_per_km2_correct(sample_df):
    result = engineer_features(sample_df)
    row = result[result["country"] == "Kenya"].iloc[0]
    expected = row["loss_area_ha"] / COUNTRY_AREAS["Kenya"]
    assert abs(row["loss_per_km2"] - expected) < 1e-10


# ── config tests ──────────────────────────────────────────────────

def test_all_eac_countries_present():
    expected = {"Kenya", "Tanzania", "Uganda", "Rwanda",
                "DRC", "Burundi", "South Sudan", "Somalia"}
    assert set(COUNTRIES.keys()) == expected

def test_all_bounding_boxes_valid():
    for country, bbox in COUNTRIES.items():
        west, south, east, north = bbox
        assert west < east,  f"{country}: west must be less than east"
        assert south < north, f"{country}: south must be less than north"

def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 6

def test_target_column_name():
    assert TARGET_COLUMN == "high_risk"
