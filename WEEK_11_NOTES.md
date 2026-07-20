# Week 11 — Deforestation Risk Capstone
## Complete Notes & Learnings

**Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya**  
**Duration:** ~4 calendar weeks (June 26 – July 20, 2026)  
**Repository:** https://github.com/M20Jay/week11-forest-capstone  
**Live API:** http://3.67.15.230:8004/docs  

---

## What We Built

A complete, production-grade MLOps pipeline that:
- Ingests **real satellite data** from Global Forest Watch API
- Trains a **Random Forest classifier** to predict deforestation risk
- Serves **live predictions** via FastAPI (74.9ms latency)
- Monitors **data drift** using PSI + KS test
- Displays **real-time metrics** on Grafana dashboard
- **Auto-tests and deploys** via GitHub Actions CI/CD
- **Orchestrates** the full pipeline weekly via Airflow DAG

---

## Day 1 — Data Ingestion

### What We Did
- Created GFW developer account at globalforestwatch.org
- Authenticated via `x-api-key` header (not Bearer token)
- Required `Origin` header: `https://martin-mlops.com`
- Ingested tree cover loss data for all 8 EAC countries
- Saved raw JSON to `data/raw/{country}_tree_cover_loss.json`

### Key Learning
GFW uses SQL-style queries via POST request body, not URL parameters.
Country bounding boxes sourced from Natural Earth data.

### Countries Covered
Kenya, Tanzania, Uganda, Rwanda, DRC, Burundi, South Sudan, Somalia

### Code Pattern
```python
headers = {
    "x-api-key": os.getenv("GFW_API_KEY"),
    "Origin": "https://martin-mlops.com"
}
response = requests.post(url, headers=headers, json=payload)
```

---

## Day 2 — Feature Engineering + Training

### Feature Engineering
Built 6 features from raw tree cover loss data:
```
loss_area_ha     → raw annual loss in hectares
prev_year_loss   → previous year's loss (LAG feature)
rolling_3yr_avg  → 3-year rolling average
yoy_change       → year-on-year percentage change
country_area_km2 → country size (normalisation denominator)
loss_per_km2     → loss_area_ha / country_area_km2 (KEY FEATURE)
```

**Target variable:** `high_risk` = 1 if loss > 75th percentile

### Model Comparison
| Model | CV F1 | Test F1 |
|---|---|---|
| Random Forest | 0.9263 | 0.9231 |
| XGBoost | 0.9112 | 0.9089 |
| LightGBM | 0.9034 | 0.9012 |
| Logistic Regression | 0.7821 | 0.7654 |

**Winner: Random Forest**

### Key Learnings
1. **OOF threshold sweep** — optimal threshold was 0.55, not default 0.5
2. **Calibration** — isotonic regression reduced Brier score from 0.0581 to 0.0034
3. **Normalisation** — `loss_per_km2` was strongest feature (0.3525 importance)
   - DRC losing 50,000 ha ≠ Rwanda losing 50,000 ha
   - Per km² tells the real story

### MLflow Tracking
All experiments tracked in `mlruns/`
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Pytest — 21 Tests
```bash
pytest tests/ -v
```
- `test_features.py` — 17 tests for build_geometry and engineer_features
- `test_train.py` — 4 tests for pipeline load, predict, probabilities

---

## Day 3 — SageMaker Training Job

### What We Built
- `scripts/sagemaker_train.py` — SageMaker-aware training wrapper
- `scripts/submit_sagemaker_job.py` — boto3 direct job submission

### Why boto3 Instead of SageMaker SDK
SageMaker SDK (sagemaker==2.x) pulls NVIDIA GPU packages that:
- Are 300MB+
- Conflict with numpy 2.x
- Fill disk on t3.medium

Solution: Use boto3 directly — same outcome, no heavy dependencies.

### Job Submission Pattern
```python
sm = boto3.client("sagemaker", region_name="eu-central-1")
sm.create_training_job(
    TrainingJobName=JOB_NAME,
    AlgorithmSpecification={
        "TrainingImage": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/sklearn:1.2-1-cpu-py3",
        "TrainingInputMode": "File",
    },
    RoleArn=ROLE_ARN,
    InputDataConfig=[{"ChannelName": "train", "DataSource": {...}}],
    OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output/"},
    ResourceConfig={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 5},
)
```

### Status
- Code complete and tested ✅
- S3 data uploaded ✅
- IAM permissions configured ✅
- **Blocked:** AWS new account quota = 0 for ml.m5.xlarge
- Quota increase requested — case 178359381200630 (Frankfurt eu-central-1)
- Will run automatically when quota approved

### The TrainingImage URL
`763104351884` is AWS's own account ID for pre-built ML containers.
The sklearn container has sklearn, pandas, numpy, joblib pre-installed.

---

## Day 4 — FastAPI Prediction Endpoint

### Architecture
```
POST /predict → FastAPI → pipeline.pkl → prediction + PostgreSQL log
GET  /health  → status, prediction counts
GET  /metrics → Prometheus scraping
```

### Key Design Decisions

**1. Model loaded ONCE at startup (not per request)**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = joblib.load(MODEL_PATH)  # loads once
    yield  # server runs here
```

**2. Pydantic validation with Field constraints**
```python
class PredictionRequest(BaseModel):
    year: int = Field(ge=2000, le=2100)
    loss_area_ha: float = Field(ge=0)
    country_area_km2: float = Field(gt=0)
```

**3. PostgreSQL UPSERT — idempotent logging**
```sql
INSERT INTO predictions (country, year, ...)
VALUES (...)
ON CONFLICT (country, year)
DO UPDATE SET prediction = EXCLUDED.prediction, ...
```

**4. Graceful degradation**
```python
def get_db_connection():
    try:
        return psycopg2.connect(...)
    except Exception as e:
        log.warning("PostgreSQL unavailable: %s", e)
        return None  # predictions still work
```

### Systemd Service (persistent deployment)
```bash
sudo tee /etc/systemd/system/deforestation-api.service << EOF
[Unit]
Description=Deforestation Risk API
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/week11-forest-capstone
EnvironmentFile=/home/ubuntu/projects/week11-forest-capstone/.env
ExecStart=/home/ubuntu/projects/week11-forest-capstone/venv/bin/uvicorn scripts.serve:app --host 0.0.0.0 --port 8004
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable deforestation-api
sudo systemctl start deforestation-api
```

### Test the API
```bash
curl -X POST http://3.67.15.230:8004/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"Kenya","year":2023,"loss_area_ha":9157.0,
       "prev_year_loss":7552.0,"rolling_3yr_avg":7630.0,
       "yoy_change":0.21,"country_area_km2":580367,"loss_per_km2":0.0158}'

# Response:
# {"prediction":1,"probability":0.84,"risk_level":"HIGH","country":"Kenya","year":2023}
```

### Prometheus Metrics Exposed
```
deforestation_predictions_total   → total predictions (Counter)
deforestation_high_risk_total     → high risk count (Counter)
deforestation_prediction_seconds  → latency histogram (Histogram)
deforestation_high_risk_ratio     → current ratio (Gauge)
```

---

## Day 5 — SHAP Explainability

### What SHAP Does
**Feature importance** = which features matter globally (one number per feature)
**SHAP** = why the model made THIS specific prediction (one value per feature per prediction)

### SHAP vs Feature Importance
```
Feature importance: loss_per_km2 = 0.3525 (global average)
SHAP for Burundi 2008: loss_per_km2 pushed +0.3691 toward HIGH RISK
```

### TreeExplainer — Why Specifically for Random Forest
Random Forest has transparent tree structure → SHAP can calculate contributions EXACTLY.
For neural networks, SHAP approximates (slower, less precise).

### Key Result
**Burundi 2008** (first HIGH RISK in test set):
```
loss_per_km2:    +0.3691  → dominant driver
yoy_change:      +0.0802  → second strongest
country_area_km2:+0.0209
rolling_3yr_avg: +0.0086
loss_area_ha:    +0.0105
prev_year_loss:  +0.0032
```

### Outputs
- `outputs/shap_summary.png` — all predictions, all features
- `outputs/shap_waterfall_burundi_2008.png` — one prediction explained

### Important: 3D Array Handling
SHAP with RandomForest returns shape (37, 6, 2) for binary classification:
```python
if isinstance(shap_values, list):
    shap_values_high_risk = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_high_risk = shap_values[:, :, 1]  # index 1 = HIGH RISK class
```

---

## Day 6 — Drift Monitoring + Grafana

### Why Evidently Didn't Work
- Evidently requires numpy < 2.0 (uses np.float_ removed in numpy 2.0)
- Python 3.14 forces numpy 2.x
- Cannot coexist in same environment as FastAPI (pydantic conflict)
- **Decision:** Use scipy for Week 11, migrate to Evidently in Docker (Python 3.11) in Week 16

### PSI Formula (Population Stability Index)
```
PSI = Σ (current% - reference%) × ln(current% / reference%)

Interpretation:
PSI < 0.1   → stable
PSI 0.1-0.2 → minor drift, monitor
PSI > 0.2   → major drift, retrain
```

### KS Test (Kolmogorov-Smirnov)
```python
from scipy import stats
ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
# p_value < 0.05 = statistically significant drift
```

### Results
```
loss_per_km2:    PSI=0.1523  → MINOR DRIFT (simulated 20-30% increase)
loss_area_ha:    PSI=0.0262  → STABLE
yoy_change:      PSI=0.0477  → STABLE
prev_year_loss:  PSI=0.0000  → STABLE
rolling_3yr_avg: PSI=0.0000  → STABLE
country_area_km2:PSI=0.0000  → STABLE
```

### Grafana Dashboard
- **URL:** http://3.67.15.230:3000 (admin/admin123)
- **Data source:** Prometheus at http://172.31.47.118:9090
- **Panels:**
  - Total Predictions (Counter over time)
  - High Risk Ratio (Gauge 0.0-1.0)
  - Prediction Latency (Histogram sum)

### Prometheus Config
```yaml
# outputs/prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'deforestation-api'
    static_configs:
      - targets: ['172.31.47.118:8004']
```

---

## Day 7 — GitHub Actions CI/CD + Airflow DAG

### GitHub Actions CI/CD
**File:** `.github/workflows/ci.yml`

```yaml
on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - run: pip install pytest pandas numpy scikit-learn joblib lightgbm xgboost pyarrow
    - run: pytest tests/ -v --tb=short

  deploy:
    needs: test
    steps:
    - uses: appleboy/ssh-action@v0.1.10
      with:
        host: ${{ secrets.EC2_HOST }}
        username: ${{ secrets.EC2_USER }}
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          cd ~/projects/week11-forest-capstone
          git pull origin main
          source venv/bin/activate
          sudo systemctl restart deforestation-api
```

**Results:** CI (pytest) = 35s ✅, CD (deploy to EC2) = 11s ✅

**GitHub Secrets Required:**
- `EC2_HOST` = 3.67.15.230
- `EC2_USER` = ubuntu
- `EC2_SSH_KEY` = contents of mlops-key.pem

### Airflow DAG
**File:** `dags/forest_pipeline.py`

```python
schedule_interval="0 5 * * 1"  # every Monday at 5am UTC

# Task dependencies:
ingest >> build_features >> train >> evaluate >> monitor
```

**5 Tasks:**
1. `ingest_gfw_data` → pulls fresh satellite data from GFW API
2. `build_features` → engineers features from raw JSON
3. `train_model` → retrains RandomForest on latest data
4. `evaluate_model` → generates confusion matrix, calibration curves
5. `monitor_drift` → PSI + KS test vs training baseline

**Airflow UI:** http://3.67.15.230:8080 (airflow/airflow123)

**Important:** DAG file must be copied to Airflow's dags folder:
```bash
cp dags/forest_pipeline.py ~/airflow-docker/dags/
```

**Fix for Docker path issue:**
```python
# Use Airflow worker's Python, not project venv
PYTHON = "/home/airflow/.local/bin/python3"
# Set PYTHONPATH so src.forest imports work
bash_command=f"cd {PROJECT} && PYTHONPATH={PROJECT} {PYTHON} scripts/ingest_gfw_data.py"
```

**Packages required in Airflow worker:**
```bash
docker exec airflow-docker-airflow-worker-1 \
  /home/airflow/.local/bin/pip install \
  requests pandas numpy scikit-learn joblib lightgbm xgboost shap pyarrow
```

---

## Infrastructure

### EC2 Instance
```
Type:    t3.medium (upgraded from t3.small)
Region:  eu-central-1 (Frankfurt)
IP:      3.67.15.230
OS:      Ubuntu 26.04 LTS
Disk:    28GB (~80% used)
SSH:     ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@3.67.15.230
```

### S3 Bucket
```
Bucket: martin-mlops-models
Paths:
  week11/data/features.parquet  → training data
  week11/code/sourcedir.tar.gz  → SageMaker training script
  week11/output/               → SageMaker model output (pending quota)
```

### PostgreSQL
```
Container: recommendation_postgres (Docker)
Port:      5432
User:      martin
Password:  martin123
Database:  recommendations
Table:     predictions (country, year, loss_area_ha, prediction, probability, risk_level, timestamp)
```

### Running Services
| Service | Port | How to Start |
|---|---|---|
| FastAPI | 8004 | `sudo systemctl start deforestation-api` |
| Grafana | 3000 | `docker start grafana` |
| Prometheus | 9090 | `docker start prometheus` |
| Airflow | 8080 | `docker compose -f ~/airflow-docker/docker-compose.yaml up -d` |

---

## Key Concepts Learned This Week

### 1. Training-Serving Skew Prevention
```python
# Both training and serving use SAME FEATURE_COLUMNS from config
X = df[FEATURE_COLUMNS]          # training
features = pd.DataFrame([{...}])[FEATURE_COLUMNS]  # serving
```

### 2. Graceful Degradation
System continues working at reduced capability when dependencies fail.
PostgreSQL down → predictions still work, logging silently skipped.

### 3. Idempotency (UPSERT Pattern)
```sql
ON CONFLICT (country, year) DO UPDATE SET ...
```
Running the same prediction twice = same result, no duplicates.

### 4. PSI for Drift Detection
- Bin reference data into percentile buckets (NTILE)
- Compare proportions in each bucket vs current data
- PSI = Σ (cur% - ref%) × ln(cur%/ref%)
- Add 0.0001 to avoid ln(0)

### 5. SHAP Values
- TreeExplainer for Random Forest (exact, fast)
- Returns shape (n_samples, n_features, n_classes) for binary classification
- Take index [:, :, 1] for HIGH RISK contributions
- Base value + SHAP values = final prediction probability

### 6. CI/CD Pattern
```
Push code → GitHub Actions triggers
  → pytest runs on GitHub's Ubuntu (free)
  → if tests pass → SSH to EC2 → git pull → restart service
  → no manual steps ever
```

### 7. Airflow >> Operator
```python
task_a >> task_b >> task_c
# task_b only runs if task_a succeeds
# task_c only runs if task_b succeeds
# if task_a fails → all downstream tasks skipped
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `np.float_` AttributeError | numpy 2.0 removed np.float_ | Downgrade numpy or use scipy |
| `Disk quota exceeded` | pip cache fills disk | `pip cache purge` or `--no-cache-dir` |
| `Port already in use` | Docker container on same port | Check `ss -tlnp`, use different port |
| `No module named 'src'` | PYTHONPATH not set | `PYTHONPATH=/path/to/project python3 script.py` |
| `shap_values[:,:,1]` | SHAP returns 3D for binary | Extract `shap_values[:,:,1]` for HIGH RISK |
| `host.docker.internal` not found | Linux Docker doesn't support this | Use EC2 internal IP instead |
| `ResourceLimitExceeded` SageMaker | New AWS account quota = 0 | Request quota increase, wait 1-3 days |

---

## Interview Answers Built This Week

### Feature Importance vs SHAP
*"Feature importance tells you which features matter globally — one average number per feature. SHAP tells you why the model made a specific individual prediction — one value per feature per prediction, showing direction and magnitude."*

### Drift Monitoring
*"I use PSI for overall population shift and KS test for numerical feature distributions. PSI > 0.1 is minor drift to monitor, PSI > 0.2 is major drift requiring retraining. For categorical features, chi-square test. I implemented this using scipy in Week 11 and will migrate to Evidently AI in Docker in Week 16."*

### Production ML Deployment
*"A production ML system is more than a trained model. It requires: FastAPI for serving, PostgreSQL for audit logging, Prometheus for metrics, Grafana for dashboards, GitHub Actions for CI/CD, and Airflow for orchestration. Each component adds observability, reliability, or automation. The model itself is only one layer."*

---

## Live Service URLs

| Service | URL |
|---|---|
| Prediction API | http://3.67.15.230:8004/predict |
| API Docs | http://3.67.15.230:8004/docs |
| Health Check | http://3.67.15.230:8004/health |
| Metrics | http://3.67.15.230:8004/metrics |
| Grafana | http://3.67.15.230:3000 |
| Prometheus | http://3.67.15.230:9090 |
| Airflow | http://3.67.15.230:8080 |

---

## Repository Structure

```
week11-forest-capstone/
├── src/forest/
│   ├── __init__.py
│   ├── config.py          → FEATURE_COLUMNS, paths, constants
│   └── features.py        → build_geometry(), engineer_features()
├── scripts/
│   ├── ingest_gfw_data.py → GFW API ingestion, 8 countries
│   ├── build_features.py  → feature engineering → features.parquet
│   ├── train.py           → 4 models, MLflow, GridSearchCV, calibration
│   ├── evaluate.py        → confusion matrix, calibration, feature importance
│   ├── sagemaker_train.py → SageMaker-aware wrapper
│   ├── submit_sagemaker_job.py → boto3 job submission
│   ├── serve.py           → FastAPI prediction API
│   ├── explain.py         → SHAP TreeExplainer
│   └── monitor.py         → PSI + KS drift monitoring
├── dags/
│   └── forest_pipeline.py → Airflow DAG, weekly schedule
├── tests/
│   ├── test_features.py   → 17 tests
│   └── test_train.py      → 4 tests
├── .github/workflows/
│   └── ci.yml             → GitHub Actions CI/CD
├── data/
│   ├── raw/               → 8 country JSON files from GFW
│   └── processed/
│       └── features.parquet → 184 rows, 9 columns
└── outputs/
    ├── deforestation_pipeline.pkl
    ├── confusion_matrix.png
    ├── calibration_curve.png
    ├── feature_importance.png
    ├── shap_summary.png
    ├── shap_waterfall_burundi_2008.png
    ├── drift_report.html
    ├── drift_metrics.txt
    ├── grafana_dashboard.json
    ├── prometheus.yml
    ├── grafana_monitoring_dashboard.png
    ├── github_actions_cicd.png
    ├── airflow_dag_graph.png
    ├── airflow_dag_running.png
    └── airflow_dag_details.png
```

---

*Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya | github.com/M20Jay*  
*Week 11 of 30 — MLOps Programme | July 2026*
