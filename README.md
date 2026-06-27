# Forest Capstone — Deforestation Risk Monitoring with Real Satellite Data

![Python](https://img.shields.io/badge/python-3.14-blue)
![Status](https://img.shields.io/badge/status-Day%201%20of%207-yellow)
![Data Source](https://img.shields.io/badge/data-Global%20Forest%20Watch-green)
![AWS](https://img.shields.io/badge/AWS-EC2%20Frankfurt-orange)
![Week](https://img.shields.io/badge/Week%2011%20of%2015-MLOps%20Programme-lightgrey)

A production-style MLOps pipeline that ingests real, satellite-derived deforestation data from Global Forest Watch's authenticated Data API, building toward a deforestation risk classifier aligned with UNEP's biodiversity monitoring mandate.

🔗 Data Source → [Global Forest Watch Data API](https://data-api.globalforestwatch.org)
🔗 Programme → [github.com/M20Jay](https://github.com/M20Jay)

---

## The Problem This Solves

Deforestation monitoring across Africa has historically relied on periodic manual reports — by the time loss is reported, the damage is months old. Real-time, automated risk classification using satellite-derived data closes that gap, giving environmental bodies like UNEP a continuously updated picture instead of a stale snapshot.

> "Data that arrives after the forest is already gone isn't monitoring — it's an obituary."

---

## Architecture (Day 1)

```
GFW Data API (data-api.globalforestwatch.org)

↓ authenticated via x-api-key + Origin header

Python ingestion script (scripts/ingest_gfw_data.py)

↓ SQL-style query against umd_tree_cover_loss dataset

↓ filtered to Kenya bounding box, canopy density ≥30%, 2001-2025

Raw JSON response

↓ persisted for provenance

data/raw/kenya_tree_cover_loss.json
```
**Confirmed real output (2001–2025):** Kenya tree cover loss ranges from ~3,093 ha (2002) to a peak of ~9,157 ha (2020) — verified directly from Hansen/UMD satellite data via GFW's live API, not synthetic or estimated figures.
---

## Day 1 — What's Been Built

| Component | Status |
|---|---|
| GFW account + authenticated API access | ✅ Done |
| Real data ingestion (Kenya, 2001–2025) | ✅ Done |
| Raw data persistence (`data/raw/`) | ✅ Done |
| Secure credential handling (`.env`, `.gitignore`) | ✅ Done |
| Reproducible environment (`requirements.txt`) | ✅ Done |
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

│   └── ingest_gfw_data.py   → GFW API ingestion, auth, save logic

├── tests/               → pytest test suite (Day 2+)

├── data/

│   ├── raw/             → raw GFW API responses (provenance)

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

## Why This Matters for UNEP's Mandate

Global Forest Watch is GFW data UNEP itself references for environmental monitoring. This pipeline demonstrates the exact technical pattern — authenticated API integration, real satellite-derived data, reproducible engineering practice — needed for production biodiversity monitoring systems at scale.

---

*Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya | Week 11 of 15 — MLOps Programme*
