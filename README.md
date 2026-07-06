# Forest Capstone — Deforestation Risk Monitoring with Real Satellite Data

![Python](https://img.shields.io/badge/python-3.14-blue)
![Status](https://img.shields.io/badge/status-Day%202%20Complete-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-8%20EAC%20countries-green)
![Data Source](https://img.shields.io/badge/data-Global%20Forest%20Watch-green)
![AWS](https://img.shields.io/badge/AWS-EC2%20Frankfurt-orange)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![Week](https://img.shields.io/badge/Week%2011%20of%2015-MLOps%20Programme-lightgrey)

A production-style MLOps pipeline that ingests real, satellite-derived deforestation data from Global Forest Watch's authenticated Data API, building toward a deforestation risk classifier aligned with UNEP's biodiversity monitoring mandate.

🔗 Data Source → [Global Forest Watch Data API](https://data-api.globalforestwatch.org)

🔗 Live Pipeline → *Coming Day 4-5 — FastAPI endpoint on martin-mlops.com*

---

## The Problem This Solves

Deforestation monitoring across Africa has historically relied on periodic manual reports — by the time loss is reported, the damage is months old. Real-time, automated risk classification using satellite-derived data closes that gap, giving environmental bodies like UNEP a continuously updated picture instead of a stale snapshot.

> "Data that arrives after the forest is already gone isn't monitoring — it's an obituary."

---

## Geographic Scope

This pipeline covers all eight member states of the East African Community (EAC) — Kenya, Tanzania, Uganda, Rwanda, the Democratic Republic of Congo, Burundi, South Sudan, and Somalia. The EAC is an official regional economic and political bloc, not an arbitrary country selection, which gives this dataset a real, defensible geographic boundary.

Country bounding box coordinates were sourced from [Natural Earth](https://www.naturalearthdata.com/) public reference data (via a [community-compiled reference](https://gist.github.com/graydon/11198540)), cross-checked against [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/), and sanity-tested against known relative country sizes (DRC — Africa's 2nd-largest country — correctly produced the largest bounding box in the set) before use.

All 8 countries returned successful (HTTP 200) responses from the GFW API, including Somalia and South Sudan — two countries initially flagged as potential edge cases due to arid terrain and historically lower data infrastructure.

---

## Architecture (Day 2)

```
GFW Data API (data-api.globalforestwatch.org)
    ↓ authenticated via x-api-key + Origin header
scripts/ingest_gfw_data.py
    ↓ SQL-style query, looped across 8 EAC countries, 2001-2025
data/raw/{country}_tree_cover_loss.json (8 files)
    ↓
scripts/build_features.py
    ↓ rolling averages, lag features, normalisation, binary label
data/processed/features.parquet (184 rows x 9 columns)
    ↓
scripts/train.py
    ↓ 4-model comparison, OOF threshold sweep, GridSearchCV, MLflow
outputs/deforestation_pipeline.pkl
    ↓
scripts/evaluate.py
    ↓ confusion matrix, calibration curves, feature importance
outputs/ (confusion_matrix.png, calibration_curve.png, feature_importance.png)
```

*Best model: RandomForest — CV F1=0.9263 (tuned), Test F1=0.9231, Brier score improved 0.0581→0.0034 after isotonic calibration.*

---

## Key Findings (Day 2)

| Finding | Value |
|---|---|
| Best model | Random Forest (after 4-model comparison) |
| CV F1 (tuned) | 0.9263 |
| Test F1 | 0.9231 |
| Test Recall (High Risk) | 0.95 — catches 95% of genuine high-risk years |
| Top feature | `loss_per_km2` (0.3525) — normalisation was the right decision |
| Optimal threshold | 0.55 (found via OOF sweep, not default 0.5) |
| Brier score before calibration | 0.0581 |
| Brier score after isotonic calibration | 0.0034 |

---

## Progress

| Component | Status |
|---|---|
| GFW account + authenticated API access | ✅ Done (Day 1) |
| Real data ingestion — Kenya only | ✅ Done (Day 1) |
| Expanded to all 8 EAC countries | ✅ Done (Day 2) |
| Raw data persistence (`data/raw/`) | ✅ Done (Day 1) |
| Secure credential handling (`.env`, `.gitignore`) | ✅ Done (Day 1) |
| Reproducible environment (`requirements.txt`) | ✅ Done (Day 1) |
| `src/forest/` shared package | ✅ Done (Day 2) |
| Feature engineering — 184 rows, 9 columns | ✅ Done (Day 2) |
| Model comparison — 4 models, MLflow tracking | ✅ Done (Day 2) |
| Hyperparameter tuning — GridSearchCV | ✅ Done (Day 2) |
| OOF threshold sweep — Gap 2 closed | ✅ Done (Day 2) |
| Calibration curves + Brier score — Gap 1 closed | ✅ Done (Day 2) |
| pytest suite — 21 tests passing | ✅ Done (Day 2) |
| SageMaker Training Job | ⏳ Day 3 |
| SageMaker vs EC2 FastAPI comparison | ⏳ Day 4 |
| SHAP explainability | ⏳ Day 5 |
| Evidently AI + Grafana drift monitoring | ⏳ Day 6 |
| GitHub Actions CI/CD + Airflow DAG | ⏳ Day 7 |

---

## Project Structure

```
week11-forest-capstone/
├── src/forest/
│   ├── __init__.py          → makes src/forest a Python package
│   ├── config.py            → single source of truth for all constants
│   └── features.py          → reusable build_geometry() and engineer_features()
├── scripts/
│   ├── ingest_gfw_data.py   → GFW API ingestion across 8 EAC countries
│   ├── build_features.py    → feature engineering → features.parquet
│   ├── train.py             → 4-model comparison, MLflow, GridSearchCV
│   └── evaluate.py          → confusion matrix, calibration, feature importance
├── tests/
│   ├── test_features.py     → 17 tests: build_geometry, engineer_features
│   └── test_train.py        → 4 tests: pipeline load, predict, probabilities
├── dags/                    → Airflow DAG (Day 7)
├── data/
│   ├── raw/                 → 8 country JSON files from GFW API (provenance)
│   └── processed/           → features.parquet — 184 rows, 9 columns
├── outputs/                 → trained pipeline, evaluation plots, logs
├── pytest.ini               → test configuration
├── requirements.txt         → exact reproducible environment
└── .gitignore               → protects .env, venv/, secrets, mlruns/
```

---

## Running This Yourself

```bash
git clone https://github.com/M20Jay/week11-forest-capstone.git
cd week11-forest-capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your own GFW API key (get one at globalforestwatch.org/my-gfw)
echo "GFW_API_KEY=your_key_here" > .env

# Run the full pipeline
python3 scripts/ingest_gfw_data.py
python3 scripts/build_features.py
python3 scripts/train.py
python3 scripts/evaluate.py

# Run tests
pytest tests/ -v
```

---

## Why This Matters

Global Forest Watch data is widely used by environmental and conservation organizations for biodiversity and deforestation monitoring. This pipeline demonstrates the exact technical pattern — authenticated API integration, real satellite-derived data, reproducible engineering practice — needed for production environmental monitoring systems at scale.

---

*Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya | Week 11 of 15 — MLOps Programme*
