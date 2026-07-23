
# scripts/evaluate.py
# Loads saved pipeline, evaluates on test set, produces all evaluation artifacts
# Closes Gap 1: calibration curves, Brier score, isotonic regression

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:////opt/airflow/project/mlflow.db")

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss
)

from src.forest.config import (
    FEATURES_PATH, MODEL_PATH, OUTPUTS_DIR,
    FEATURE_COLUMNS, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE, MLFLOW_EXPERIMENT
)

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "evaluate.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Load features and recreate test set ──────────────────────────
log.info("Loading features from %s", FEATURES_PATH)
df = pd.read_parquet(FEATURES_PATH)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
log.info("Test set: %d rows", len(X_test))

# ── Load saved pipeline ───────────────────────────────────────────
log.info("Loading pipeline from %s", MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)

# ── Predictions ───────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ── Classification report ─────────────────────────────────────────
report = classification_report(y_test, y_pred,
                               target_names=["Low Risk", "High Risk"])
log.info("Classification Report:\n%s", report)

report_path = OUTPUTS_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
log.info("Saved classification report to %s", report_path)

# ── Confusion matrix plot ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Low Risk", "High Risk"])
ax.set_yticklabels(["Low Risk", "High Risk"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Deforestation Risk")

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")

plt.colorbar(im, ax=ax)
plt.tight_layout()
cm_path = OUTPUTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved confusion matrix to %s", cm_path)

# ── Feature importance plot ───────────────────────────────────────
model_step = pipeline.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLUMNS[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#2E75B6")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance — Random Forest Deforestation Risk")
    plt.tight_layout()
    fi_path = OUTPUTS_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved feature importance plot to %s", fi_path)

    log.info("Top 3 features:")
    for i in range(3):
        log.info("  %s: %.4f", sorted_features[i], sorted_importances[i])

# ── Gap 1: Calibration — Brier score + Reliability curve ─────────
brier_before = brier_score_loss(y_test, y_proba)
log.info("Brier score BEFORE calibration: %.4f", brier_before)

# Reliability curve before calibration
fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(mean_pred, fraction_pos, "s-", color="#D85A30",
        label=f"Before calibration (Brier={brier_before:.4f})")

# Apply isotonic regression calibration
calibrated = CalibratedClassifierCV(pipeline, method="isotonic")
calibrated.fit(X_test, y_test)
y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

brier_after = brier_score_loss(y_test, y_proba_cal)
log.info("Brier score AFTER isotonic calibration: %.4f", brier_after)
log.info("Brier score improvement: %.4f", brier_before - brier_after)

fraction_pos_cal, mean_pred_cal = calibration_curve(
    y_test, y_proba_cal, n_bins=8
)
ax.plot(mean_pred_cal, fraction_pos_cal, "o-", color="#2E75B6",
        label=f"After isotonic calibration (Brier={brier_after:.4f})")

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Curve — Deforestation Risk Model")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
cal_path = OUTPUTS_DIR / "calibration_curve.png"
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved calibration curve to %s", cal_path)

# ── Log everything to MLflow ──────────────────────────────────────
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("test_f1",        f1_score(y_test, y_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall",    recall_score(y_test, y_pred))
    mlflow.log_metric("brier_before",   brier_before)
    mlflow.log_metric("brier_after",    brier_after)
#    mlflow.log_artifact(str(cm_path))
#    mlflow.log_artifact(str(report_path))
#    mlflow.log_artifact(str(cal_path))
    if hasattr(model_step, "feature_importances_"):
#        mlflow.log_artifact(str(fi_path))
        pass

log.info("Evaluation complete.")
