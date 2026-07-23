# scripts/train.py
# Trains deforestation risk classifier — compares 4 models, tunes best,
# logs everything to MLflow, saves fitted pipeline to outputs/

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import os
mlflow.set_tracking_uri("sqlite:////opt/airflow/project/mlflow.db")
mlflow.set_registry_uri("sqlite:////opt/airflow/project/mlflow.db")
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import lightgbm as lgb
import xgboost as xgb

from src.forest.config import (
    FEATURES_PATH, MODEL_PATH, LOG_PATH, OUTPUTS_DIR,
    FEATURE_COLUMNS, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, MLFLOW_EXPERIMENT
)

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Load features ─────────────────────────────────────────────────
log.info("Loading features from %s", FEATURES_PATH)
df = pd.read_parquet(FEATURES_PATH)
log.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ── Train / test split ────────────────────────────────────────────
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

mlflow.set_experiment(MLFLOW_EXPERIMENT)
best_f1 = 0
best_model_name = None

log.info("Starting model comparison across %d models", len(models))

for model_name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    with mlflow.start_run(run_name=model_name):
        # Cross-validated F1
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1
        )
        # OOF probabilities (Gap 2)
        oof_proba = cross_val_predict(
            pipeline, X_train, y_train, cv=skf,
            method="predict_proba", n_jobs=-1
        )[:, 1]

        # Threshold sweep (Gap 2)
        best_thresh = 0.5
        best_thresh_f1 = 0
        for thresh in np.arange(0.2, 0.8, 0.05):
            y_thresh = (oof_proba >= thresh).astype(int)
            thresh_f1 = f1_score(y_train, y_thresh)
            if thresh_f1 > best_thresh_f1:
                best_thresh_f1 = thresh_f1
                best_thresh = thresh

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_f1        = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall    = recall_score(y_test, y_pred)

        # Log params
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("optimal_threshold", round(best_thresh, 2))

        # Log metrics
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std",  cv_scores.std())
        mlflow.log_metric("test_f1",        test_f1)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall",    test_recall)
        mlflow.log_metric("oof_best_threshold", best_thresh)
        mlflow.log_metric("oof_best_f1",    best_thresh_f1)

        log.info(
            "%s | CV F1: %.4f ± %.4f | Test F1: %.4f | Best threshold: %.2f",
            model_name, cv_scores.mean(), cv_scores.std(),
            test_f1, best_thresh
        )

        if cv_scores.mean() > best_f1:
            best_f1 = cv_scores.mean()
            best_model_name = model_name

log.info("Best model: %s (CV F1: %.4f)", best_model_name, best_f1)

# ── Hyperparameter tuning on best model ──────────────────────────
log.info("Starting GridSearchCV on %s", best_model_name)

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
else:  # XGBoost
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

log.info("Best params: %s", grid_search.best_params_)
log.info("Best CV F1 after tuning: %.4f", grid_search.best_score_)

# ── Final evaluation ──────────────────────────────────────────────
y_pred_final  = final_pipeline.predict(X_test)
final_test_f1 = f1_score(y_test, y_pred_final)

log.info("Final Test F1: %.4f", final_test_f1)
log.info("\n%s", classification_report(y_test, y_pred_final))

# ── Log final run to MLflow ───────────────────────────────────────
with mlflow.start_run(run_name=f"{best_model_name}_tuned"):
    mlflow.log_param("model_type", f"{best_model_name}_tuned")
    mlflow.log_params({
        k.replace("model__", ""): v
        for k, v in grid_search.best_params_.items()
    })
    mlflow.log_metric("tuned_cv_f1", grid_search.best_score_)
    mlflow.log_metric("tuned_test_f1", final_test_f1)
    mlflow.log_metric("tuned_precision", precision_score(y_test, y_pred_final))
    mlflow.log_metric("tuned_recall",    recall_score(y_test, y_pred_final))
# #     mlflow.sklearn.log_model(final_pipeline, "model")

# ── Save pipeline ─────────────────────────────────────────────────
joblib.dump(final_pipeline, MODEL_PATH)
log.info("Pipeline saved to %s", MODEL_PATH)

# ── Verify saved pipeline loads correctly ─────────────────────────
loaded = joblib.load(MODEL_PATH)
verify_pred = loaded.predict(X_test[:5])
log.info("Verification — first 5 predictions: %s", verify_pred.tolist())
log.info("Training complete.")
