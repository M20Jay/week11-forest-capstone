# scripts/save_predictions.py
# Loads trained model, scores all 5 countries,
# saves predictions to S3 as parquet file

import logging
import boto3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from src.forest.config import (
    COUNTRIES, MODEL_PATH, OUTPUTS_DIR, FEATURES_PATH
)

# ── Setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = "martin-mlops-models"
S3_PREFIX = "week11/predictions"
LOCAL_PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.parquet"

# ── Load model ────────────────────────────────────────────────────
log.info("Loading model from %s", MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)
log.info("Model loaded successfully")

# ── Load features ─────────────────────────────────────────────────
log.info("Loading features from %s", FEATURES_PATH)
df = pd.read_parquet(FEATURES_PATH)
log.info("Loaded %d rows", len(df))

# ── Make predictions ──────────────────────────────────────────────
from src.forest.config import FEATURE_COLUMNS, TARGET_COLUMN

X = df[FEATURE_COLUMNS]
df["predicted_risk"]       = pipeline.predict(X)
df["predicted_probability"] = pipeline.predict_proba(X)[:, 1]
df["risk_level"] = pd.cut(
    df["predicted_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)
df["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

log.info("Predictions complete — %d rows scored", len(df))
log.info("Risk distribution:\n%s", df["risk_level"].value_counts().to_string())

# ── Save locally ──────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(LOCAL_PREDICTIONS_PATH, index=False)
log.info("Predictions saved locally to %s", LOCAL_PREDICTIONS_PATH)

# ── Upload to S3 ──────────────────────────────────────────────────
s3 = boto3.client("s3")

# Save latest predictions
s3_key_latest = f"{S3_PREFIX}/latest.parquet"
s3.upload_file(
    str(LOCAL_PREDICTIONS_PATH),
    S3_BUCKET,
    s3_key_latest
)
log.info("Uploaded to s3://%s/%s", S3_BUCKET, s3_key_latest)

# Save timestamped copy for history
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
s3_key_timestamped = f"{S3_PREFIX}/predictions_{timestamp}.parquet"
s3.upload_file(
    str(LOCAL_PREDICTIONS_PATH),
    S3_BUCKET,
    s3_key_timestamped
)
log.info("Uploaded timestamped copy to s3://%s/%s", S3_BUCKET, s3_key_timestamped)

log.info("S3 upload complete.")
