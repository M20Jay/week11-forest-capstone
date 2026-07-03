# src/forest/config.py
# Single source of truth for all shared constants
# Every script imports from here — never hardcode these values elsewhere

from pathlib import Path

# ── Directory paths ───────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR      = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR       = BASE_DIR / "outputs"
FEATURES_PATH     = PROCESSED_DATA_DIR / "features.parquet"
MODEL_PATH        = OUTPUTS_DIR / "deforestation_pipeline.pkl"
LOG_PATH          = OUTPUTS_DIR / "train.log"

# ── EAC member states — bounding boxes [west, south, east, north] ──
# Source: Natural Earth (via gist.github.com/graydon/11198540)
# Cross-checked: OpenStreetMap Nominatim
# Sanity-tested: DRC largest (19.1° × 18.7°), Rwanda/Burundi smallest
COUNTRIES = {
    "Kenya":       [33.9, -4.7,  41.9,  4.6],
    "Tanzania":    [29.3, -11.7, 40.3, -0.95],
    "Uganda":      [29.6, -1.5,  35.0,  4.2],
    "Rwanda":      [28.9, -2.9,  30.9, -1.0],
    "DRC":         [12.2, -13.3, 31.3,  5.4],
    "Burundi":     [29.0, -4.5,  30.8, -2.3],
    "South Sudan": [23.9,  3.5,  35.3, 12.2],
    "Somalia":     [41.0, -1.7,  51.1, 12.0],
}

# ── Country areas in km² (for loss_per_km2 normalisation) ─────────
COUNTRY_AREAS = {
    "Burundi":     27834,
    "DRC":         2344858,
    "Kenya":       580367,
    "Rwanda":      26338,
    "Somalia":     637657,
    "South Sudan": 644329,
    "Tanzania":    945087,
    "Uganda":      241551,
}

# ── GFW API ────────────────────────────────────────────────────────
GFW_URL = (
    "https://data-api.globalforestwatch.org"
    "/dataset/umd_tree_cover_loss/v1.10/query/json"
)
ORIGIN  = "https://martin-mlops.com"

# ── Feature schema ─────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "loss_area_ha",
    "prev_year_loss",
    "rolling_3yr_avg",
    "yoy_change",
    "country_area_km2",
    "loss_per_km2",
]
TARGET_COLUMN = "high_risk"

# ── Model training ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
MLFLOW_EXPERIMENT = "deforestation-risk-eac"
