# dags/forest_pipeline.py
# Note: Requires pip install in Airflow Docker container for full execution
# Run: docker exec airflow-docker-airflow-worker-1 pip install requests pandas numpy scikit-learn joblib lightgbm xgboost shap
# Airflow DAG — Deforestation Risk Pipeline
# Runs every Monday at 5am
# Orchestrates: ingest → features → train → evaluate → monitor

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ── Default arguments ─────────────────────────────────────────────
default_args = {
    "owner":            "martin",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ── Project path ──────────────────────────────────────────────────
PROJECT = "/opt/airflow/project"
PYTHON  = "/home/airflow/.local/bin/python3"

# ── DAG definition ────────────────────────────────────────────────
with DAG(
    dag_id="deforestation_risk_pipeline",
    default_args=default_args,
    description="Weekly deforestation risk pipeline — EAC 8 countries",
    schedule_interval="0 5 * * 1",  # every Monday at 5am
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["mlops", "environmental", "week11"],
) as dag:

    # Task 1 — Ingest fresh GFW satellite data
    ingest = BashOperator(
        task_id="ingest_gfw_data",
        bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/ingest_gfw_data.py",
    )

    # Task 2 — Build features from raw data
    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/build_features.py",
    )

    # Task 3 — Train model on latest data
    train = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/train.py",
    )

    # Task 4 — Evaluate model performance
    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/evaluate.py",
    )

    # Task 5 — Monitor for data drift
    monitor = BashOperator(
        task_id="monitor_drift",
        bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/monitor.py",
    )

    # ── Task dependencies ─────────────────────────────────────────
    ingest >> build_features >> train >> evaluate >> monitor
