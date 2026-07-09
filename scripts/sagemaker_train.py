# scripts/sagemaker_train.py
# SageMaker-aware wrapper around train.py
# Reads from /opt/ml/input/data/train/ (SageMaker standard)
# Saves model to /opt/ml/model/ (SageMaker standard)
# All ML logic identical to train.py

import os
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
    GridSearchCV, train_test_split
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── SageMaker paths ───────────────────────────────────────────────
# SageMaker automatically sets these environment variables
INPUT_DIR  = os.environ.get("SM_CHANNEL_TRAIN", "data/processed")
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "loss_area_ha", "prev_year_loss", "rolling_3yr_avg",
    "yoy_change", "country_area_km2", "loss_per_km2",
]
TARGET_COLUMN = "high_risk"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5

# ── Load features ─────────────────────────────────────────────────
features_path = os.path.join(INPUT_DIR, "features.parquet")
log.info("Loading features from %s", features_path)
df = pd.read_parquet(features_path)
log.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── Model comparison ──────────────────────────────────────────────
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        verbosity=0, eval_metric="logloss"),
}

best_f1 = 0
best_model_name = None

for model_name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1
    )
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=skf,
        method="predict_proba", n_jobs=-1
    )[:, 1]

    best_thresh, best_thresh_f1 = 0.5, 0
    for thresh in np.arange(0.2, 0.8, 0.05):
        y_thresh = (oof_proba >= thresh).astype(int)
        thresh_f1 = f1_score(y_train, y_thresh)
        if thresh_f1 > best_thresh_f1:
            best_thresh_f1 = thresh_f1
            best_thresh = thresh

    pipeline.fit(X_train, y_train)
    y_pred   = pipeline.predict(X_test)
    test_f1  = f1_score(y_test, y_pred)

    log.info("%s | CV F1: %.4f ± %.4f | Test F1: %.4f | Threshold: %.2f",
             model_name, cv_scores.mean(), cv_scores.std(),
             test_f1, best_thresh)

    if cv_scores.mean() > best_f1:
        best_f1 = cv_scores.mean()
        best_model_name = model_name

log.info("Best model: %s (CV F1: %.4f)", best_model_name, best_f1)

# ── Hyperparameter tuning ─────────────────────────────────────────
if best_model_name == "RandomForest":
    best_estimator = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth":    [3, 5, None],
        "model__min_samples_split": [2, 5],
    }
elif best_model_name == "GradientBoosting":
    best_estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "model__n_estimators":  [50, 100, 200],
        "model__max_depth":     [3, 5],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }
elif best_model_name == "LightGBM":
    best_estimator = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    param_grid = {
        "model__n_estimators":  [50, 100, 200],
        "model__max_depth":     [3, 5, -1],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }
else:
    best_estimator = xgb.XGBClassifier(
        random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss")
    param_grid = {
        "model__n_estimators":  [50, 100, 200],
        "model__max_depth":     [3, 5],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }

tuned_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_estimator)
])

grid_search = GridSearchCV(
    tuned_pipeline, param_grid, cv=skf,
    scoring="f1", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
final_pipeline = grid_search.best_estimator_

y_pred_final  = final_pipeline.predict(X_test)
final_test_f1 = f1_score(y_test, y_pred_final)

log.info("Best params: %s", grid_search.best_params_)
log.info("Tuned CV F1: %.4f", grid_search.best_score_)
log.info("Final Test F1: %.4f", final_test_f1)
log.info("\n%s", classification_report(y_test, y_pred_final))

# ── Save model to SageMaker output path ──────────────────────────
model_path = os.path.join(OUTPUT_DIR, "deforestation_pipeline.pkl")
joblib.dump(final_pipeline, model_path)
log.info("Model saved to %s", model_path)

# ── Verify ────────────────────────────────────────────────────────
loaded = joblib.load(model_path)
verify = loaded.predict(X_test[:5])
log.info("Verification predictions: %s", verify.tolist())
log.info("SageMaker training complete.")
