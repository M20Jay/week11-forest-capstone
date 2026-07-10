# scripts/serve.py
# FastAPI prediction endpoint for deforestation risk model
# Built by Martin James Ng'ang'a | MLOps Engineer | github.com/M20Jay
# Loads pipeline once at startup, serves predictions via HTTP
# Logs every prediction to PostgreSQL (UPSERT — idempotent)
# Exposes Prometheus metrics for Grafana monitoring

import logging
import time
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import psycopg2

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from src.forest.config import MODEL_PATH, FEATURE_COLUMNS, OUTPUTS_DIR

# ── Logging setup ─────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "serve.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────
PREDICTION_COUNT   = Counter(
    "deforestation_predictions_total",
    "Total number of predictions made"
)
HIGH_RISK_COUNT    = Counter(
    "deforestation_high_risk_total",
    "Total number of HIGH RISK predictions"
)
PREDICTION_LATENCY = Histogram(
    "deforestation_prediction_seconds",
    "Time taken to make a prediction in seconds"
)
HIGH_RISK_RATIO    = Gauge(
    "deforestation_high_risk_ratio",
    "Current ratio of high risk predictions (0.0 to 1.0)"
)

# ── Global state ──────────────────────────────────────────────────
pipeline          = None
total_predictions = 0
total_high_risk   = 0

# ── Pydantic request model — with validation ──────────────────────
class PredictionRequest(BaseModel):
    country:         str
    year:            int   = Field(ge=2000, le=2100,
                                   description="Year of observation (2000-2100)")
    loss_area_ha:    float = Field(ge=0,
                                   description="Tree cover loss in hectares")
    prev_year_loss:  float = Field(description="Previous year loss in hectares")
    rolling_3yr_avg: float = Field(ge=0,
                                   description="3-year rolling average loss in hectares")
    yoy_change:      float = Field(description="Year-on-year percentage change")
    country_area_km2:float = Field(gt=0,
                                   description="Country area in km2 — must be positive")
    loss_per_km2:    float = Field(ge=0,
                                   description="Loss normalized by country area")

# ── Pydantic response model ───────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:  int
    probability: float
    risk_level:  str
    country:     str
    year:        int

# ── PostgreSQL helpers ────────────────────────────────────────────
def get_db_connection():
    """
    Connect to PostgreSQL.
    Returns connection object if successful, None if unavailable.
    Graceful degradation — predictions still work without PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST",     "localhost"),
            port=os.getenv("DB_PORT",     "5432"),
            database=os.getenv("DB_NAME", "mlops"),
            user=os.getenv("DB_USER",     "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            connect_timeout=3
        )
        return conn
    except Exception as e:
        log.warning("PostgreSQL unavailable: %s", e)
        return None


def create_predictions_table():
    """
    Create the predictions table with CHECK constraints (ACID consistency).
    Safe to call on every startup — IF NOT EXISTS prevents duplication.
    """
    conn = get_db_connection()
    if conn is None:
        log.warning("Skipping table creation — PostgreSQL not available")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id            SERIAL PRIMARY KEY,
                    country       VARCHAR(50)  NOT NULL,
                    year          INTEGER      NOT NULL
                                  CHECK (year >= 2000 AND year <= 2100),
                    loss_area_ha  FLOAT        NOT NULL
                                  CHECK (loss_area_ha >= 0),
                    prediction    INTEGER      NOT NULL
                                  CHECK (prediction IN (0, 1)),
                    probability   FLOAT        NOT NULL
                                  CHECK (probability >= 0.0 AND probability <= 1.0),
                    risk_level    VARCHAR(10)  NOT NULL
                                  CHECK (risk_level IN ('HIGH', 'LOW')),
                    timestamp     TIMESTAMPTZ  DEFAULT NOW(),
                    UNIQUE (country, year)
                )
            """)
        conn.commit()
        log.info("Predictions table ready")
    except Exception as e:
        log.warning("Could not create predictions table: %s", e)
    finally:
        conn.close()


def log_prediction_to_db(
    request:     PredictionRequest,
    prediction:  int,
    probability: float,
    risk_level:  str
):
    """
    UPSERT prediction to PostgreSQL.
    Idempotent: running twice for same country+year updates the row,
    never creates a duplicate. Implements the ACID idempotency pattern.
    """
    conn = get_db_connection()
    if conn is None:
        return   # graceful degradation — skip logging, keep serving
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions
                    (country, year, loss_area_ha,
                     prediction, probability, risk_level, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (country, year)
                DO UPDATE SET
                    loss_area_ha = EXCLUDED.loss_area_ha,
                    prediction   = EXCLUDED.prediction,
                    probability  = EXCLUDED.probability,
                    risk_level   = EXCLUDED.risk_level,
                    timestamp    = EXCLUDED.timestamp
            """, (
                request.country,
                request.year,
                request.loss_area_ha,
                prediction,
                probability,
                risk_level
            ))
        conn.commit()
        log.info("Prediction logged to PostgreSQL: %s %d → %s",
                 request.country, request.year, risk_level)
    except Exception as e:
        log.warning("Could not log prediction to PostgreSQL: %s", e)
    finally:
        conn.close()


# ── App lifespan — load model ONCE at startup ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when server starts (before any requests).
    Loads pipeline.pkl into memory — stays there for server lifetime.
    Loading once = millisecond predictions. Loading per request = 0.5s delay each.
    """
    global pipeline
    log.info("=" * 60)
    log.info("Deforestation Risk API starting...")
    log.info("Loading model from %s", MODEL_PATH)
    pipeline = joblib.load(MODEL_PATH)
    log.info("Model loaded successfully")
    create_predictions_table()
    log.info("API ready to serve predictions")
    log.info("=" * 60)
    yield
    log.info("API shutting down")


# ── FastAPI application ───────────────────────────────────────────
app = FastAPI(
    title="Deforestation Risk API",
    description=(
        "Predicts deforestation risk for all 8 East African Community (EAC) "
        "countries using real satellite data from Global Forest Watch.\n\n"
        "Built by **Martin James Ng'ang'a** | MLOps Engineer | "
        "Nairobi, Kenya | github.com/M20Jay\n\n"
        "Week 11 Environmental Capstone — aligned with UNEP biodiversity "
        "monitoring mandate."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns API status and running prediction counts.
    Used by monitoring systems to verify the API is alive.
    """
    return {
        "status":            "ok",
        "model":             "loaded" if pipeline is not None else "not loaded",
        "total_predictions": total_predictions,
        "total_high_risk":   total_high_risk,
        "high_risk_ratio":   round(
            total_high_risk / total_predictions, 4
        ) if total_predictions > 0 else 0.0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.

    Accepts deforestation features for one country-year combination.
    Returns binary prediction (0=LOW, 1=HIGH), probability, and risk level.
    Every prediction is logged to PostgreSQL and recorded in Prometheus metrics.
    """
    global total_predictions, total_high_risk

    if pipeline is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded — server starting up")

    start_time = time.time()

    # Build feature DataFrame in exact column order the model expects
    features = pd.DataFrame([{
        "loss_area_ha":     request.loss_area_ha,
        "prev_year_loss":   request.prev_year_loss,
        "rolling_3yr_avg":  request.rolling_3yr_avg,
        "yoy_change":       request.yoy_change,
        "country_area_km2": request.country_area_km2,
        "loss_per_km2":     request.loss_per_km2,
    }])[FEATURE_COLUMNS]

    # Make prediction
    prediction  = int(pipeline.predict(features)[0])
    probability = float(pipeline.predict_proba(features)[0][1])
    risk_level  = "HIGH" if prediction == 1 else "LOW"

    # Update running counters
    total_predictions += 1
    if prediction == 1:
        total_high_risk += 1

    # Update Prometheus metrics
    PREDICTION_COUNT.inc()
    if prediction == 1:
        HIGH_RISK_COUNT.inc()
    PREDICTION_LATENCY.observe(time.time() - start_time)
    HIGH_RISK_RATIO.set(
        total_high_risk / total_predictions if total_predictions > 0 else 0
    )

    # Log to PostgreSQL (UPSERT — idempotent)
    log_prediction_to_db(request, prediction, probability, risk_level)

    log.info(
        "country=%s year=%d risk=%s probability=%.4f latency=%.3fs",
        request.country, request.year, risk_level,
        probability, time.time() - start_time
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
        country=request.country,
        year=request.year,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """
    Prometheus metrics endpoint.
    Scraped every 15 seconds by Prometheus.
    Grafana reads from Prometheus to build drift monitoring dashboards.
    """
    return generate_latest().decode("utf-8")
