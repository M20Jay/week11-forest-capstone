# Forest Capstone — Deforestation Risk Monitoring with Real Satellite Data

![Python](https://img.shields.io/badge/python-3.14-blue)
![Status](https://img.shields.io/badge/status-Day%202%20of%207-yellow)
![Coverage](https://img.shields.io/badge/coverage-8%20EAC%20countries-green)
![Data Source](https://img.shields.io/badge/data-Global%20Forest%20Watch-green)
![AWS](https://img.shields.io/badge/AWS-EC2%20Frankfurt-orange)
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

Country bounding box coordinates were sourced from [Natural Earth](https://www.naturalearthdata.com/) public reference data, cross-checked against OpenStreetMap Nominatim, and sanity-tested against known relative country sizes (DRC — Africa's 2nd-largest country — correctly produced the largest bounding box in the set) before use.

All 8 countries returned successful (HTTP 200) responses from the GFW API, including Somalia and South Sudan — two countries initially flagged as potential edge cases due to arid terrain and historically lower data infrastructure.

---

## Architecture (Day 2)

```
GFW Data API (data-api.globalforestwatch.org)
↓ authenticated via x-api-key + Origin header
Python ingestion script (scripts/ingest_gfw_data.py)
↓ SQL-style query against umd_tree_cover_loss dataset
↓ looped across 8 EAC countries, canopy density ≥30%, 2001-2025
Raw JSON response, per country
↓ persisted for provenance
data/raw/{country}_tree_cover_loss.json
```

*Confirmed real output across all 8 countries (2001–2025) — verified directly from Hansen/UMD satellite data via GFW's live API, not synthetic or estimated figures.*

---

## Progress

| Component | Status |
|---|---|
| GFW account + authenticated API access | ✅ Done |
| Real data ingestion — Kenya only | ✅ Done (Day 1) |
| Expanded to all 8 EAC countries | ✅ Done (Day 2) |
| Raw data persistence (`data/raw/`) | ✅ Done |
| Secure credential handling (`.env`, `.gitignore`) | ✅ Done |
| Reproducible environment (`requirements.txt`) | ✅ Done |
| Feature engineering | ⏳ Day 2 (in progress) |
| Random Forest training | ⏳ Day 2 |
| SageMaker Training Job comparison | ⏳ Day 3-4 |
| SHAP explainability | ⏳ Day 5 |
| Evidently AI drift monitoring | ⏳ Day 6 |

---

## Project Structure

```
week11-forest-capstone/
├── src/forest/          → reusable package code (Day 2+)
├── dags/                → Airflow DAG for scheduled retraining (later)
├── scripts/
│   └── ingest_gfw_data.py   → GFW API ingestion across 8 EAC countries, auth, save logic
├── tests/               → pytest test suite (Day 2+)
├── data/
│   ├── raw/             → raw GFW API responses, per country (provenance)
│   └── processed/       → engineered features (Day 2+)
├── outputs/             → plots, model artifacts (later)
├── requirements.txt     → exact reproducible environment
└── .gitignore           → protects .env, venv/, secrets
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

python3 scripts/ingest_gfw_data.py
```

---

## Why This Matters

Global Forest Watch data is widely used by environmental and conservation organizations for biodiversity and deforestation monitoring. This pipeline demonstrates the exact technical pattern — authenticated API integration, real satellite-derived data, reproducible engineering practice — needed for production environmental monitoring systems at scale.

---

*Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya | Week 11 of 15 — MLOps Programme*
