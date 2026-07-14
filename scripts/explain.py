# scripts/explain.py
# SHAP explainability for deforestation risk model
# Explains WHY the model flagged specific countries as HIGH RISK
# Produces: shap_summary.png, shap_waterfall.png

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import logging

from sklearn.model_selection import train_test_split

from src.forest.config import (
    MODEL_PATH, FEATURES_PATH, OUTPUTS_DIR,
    FEATURE_COLUMNS, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE
)

# ── Logging setup ─────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "explain.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Load data and model ───────────────────────────────────────────
log.info("Loading features from %s", FEATURES_PATH)
df = pd.read_parquet(FEATURES_PATH)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

log.info("Loading pipeline from %s", MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)

# ── Extract model from pipeline ───────────────────────────────────
scaler        = pipeline.named_steps["scaler"]
model         = pipeline.named_steps["model"]
X_test_scaled = scaler.transform(X_test)

log.info("Model type: %s", type(model).__name__)

# ── Create SHAP explainer ─────────────────────────────────────────
log.info("Creating TreeExplainer...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_values_high_risk = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_high_risk = shap_values[:, :, 1]
else:
    shap_values_high_risk = shap_values

log.info("SHAP values shape: %s", shap_values_high_risk.shape)

# ── Summary plot — all predictions ───────────────────────────────
log.info("Generating SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_high_risk,
    X_test_scaled,
    feature_names=FEATURE_COLUMNS,
    show=False
)
plt.title("SHAP Summary — Deforestation Risk Model\nRed = pushes toward HIGH RISK, Blue = pushes toward LOW RISK")
plt.tight_layout()
summary_path = OUTPUTS_DIR / "shap_summary.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved SHAP summary plot to %s", summary_path)

# ── Find Kenya 2023 or fallback to first HIGH RISK ────────────────
log.info("Finding Kenya 2023 in test set...")
X_test_with_meta = X_test.copy()
X_test_with_meta["country"] = df.loc[X_test.index, "country"].values
X_test_with_meta["year"]    = df.loc[X_test.index, "year"].values

kenya_mask = (
    (X_test_with_meta["country"] == "Kenya") &
    (X_test_with_meta["year"] == 2023)
)

if kenya_mask.sum() > 0:
    kenya_idx = X_test_with_meta[kenya_mask].index[0]
    pos       = X_test.index.get_loc(kenya_idx)
    label     = "Kenya 2023"
    log.info("Found Kenya 2023 at test set position %d", pos)
else:
    log.warning("Kenya 2023 not in test set — using first HIGH RISK prediction instead")
    high_risk_positions = np.where(y_test.values == 1)[0]
    pos     = high_risk_positions[0]
    country = X_test_with_meta.iloc[pos]["country"]
    year    = int(X_test_with_meta.iloc[pos]["year"])
    label   = f"{country} {year}"
    log.info("Using %s at test position %d instead", label, pos)

# ── SHAP values for selected prediction ──────────────────────────
kenya_shap = shap_values_high_risk[pos]

log.info("SHAP values for %s:", label)
for feature, value in zip(FEATURE_COLUMNS, kenya_shap):
    direction = "toward HIGH RISK" if value > 0 else "toward LOW RISK"
    log.info("  %-20s: %+.4f  (%s)", feature, value, direction)

# ── Waterfall plot ────────────────────────────────────────────────
log.info("Generating waterfall plot for %s...", label)

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[1]

shap_explanation = shap.Explanation(
    values=kenya_shap,
    base_values=float(base_val),
    data=X_test_scaled[pos],
    feature_names=FEATURE_COLUMNS
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation, show=False)
plt.title(f"SHAP Waterfall — {label}\nWhy the model predicted HIGH RISK")
plt.tight_layout()
waterfall_path = OUTPUTS_DIR / f"shap_waterfall_{label.replace(' ', '_').lower()}.png"
plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved waterfall plot to %s", waterfall_path)

log.info("SHAP explanation complete.")
log.info("Outputs saved to %s", OUTPUTS_DIR)
