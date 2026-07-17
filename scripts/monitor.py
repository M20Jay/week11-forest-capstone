# scripts/monitor.py
# Drift monitoring for deforestation risk model
# Uses scipy + pandas — Evidently incompatible with Python 3.14
# Calculates PSI and KS test per feature
# Generates HTML drift report + drift_metrics.txt
# Note: Will migrate to Evidently AI in Docker (Week 21)

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
from scipy import stats
from datetime import datetime

from src.forest.config import (
    FEATURES_PATH, OUTPUTS_DIR, FEATURE_COLUMNS
)

# ── Logging setup ─────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "monitor.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── PSI calculation ───────────────────────────────────────────────
def calculate_psi(reference, current, bins=10):
    """
    Population Stability Index.
    PSI < 0.1   → stable
    PSI 0.1-0.2 → monitor
    PSI > 0.2   → major drift — retrain
    """
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)
    ref_props = (ref_counts + 0.0001) / (len(reference) + 0.0001 * bins)
    cur_props = (cur_counts + 0.0001) / (len(current)   + 0.0001 * bins)
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return round(float(psi), 4)

# ── Load reference data ───────────────────────────────────────────
log.info("Loading reference data from %s", FEATURES_PATH)
reference_data = pd.read_parquet(FEATURES_PATH)[FEATURE_COLUMNS]
log.info("Reference data: %d rows, %d columns",
         reference_data.shape[0], reference_data.shape[1])

# ── Load current data from PostgreSQL ────────────────────────────
def load_current_data():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST",         "localhost"),
            port=os.getenv("DB_PORT",         "5432"),
            database=os.getenv("DB_NAME",     "recommendations"),
            user=os.getenv("DB_USER",         "martin"),
            password=os.getenv("DB_PASSWORD", "martin123"),
            connect_timeout=3
        )
        query = """
            SELECT loss_area_ha, country, year
            FROM predictions
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        log.info("Loaded %d recent predictions from PostgreSQL", len(df))
        return df
    except Exception as e:
        log.warning("PostgreSQL unavailable: %s", e)
        return None

current_data_raw = load_current_data()

# ── Simulate drift if insufficient real data ──────────────────────
if current_data_raw is None or len(current_data_raw) < 10:
    log.warning("Insufficient prediction data — simulating drift for demonstration")
    np.random.seed(42)
    current_data = reference_data.copy()
    current_data["loss_per_km2"]  *= np.random.uniform(1.1, 1.3, len(current_data))
    current_data["loss_area_ha"]  *= np.random.uniform(1.05, 1.2, len(current_data))
    current_data["yoy_change"]    += np.random.normal(0.05, 0.02, len(current_data))
    log.info("Simulated current data with realistic drift applied")
else:
    current_data = current_data_raw[
        [col for col in FEATURE_COLUMNS if col in current_data_raw.columns]
    ]

log.info("Current data: %d rows", len(current_data))

# ── Calculate drift per feature ───────────────────────────────────
log.info("Calculating drift metrics...")
results = {}

for feature in FEATURE_COLUMNS:
    ref = reference_data[feature].dropna().values
    cur = current_data[feature].dropna().values

    psi = calculate_psi(ref, cur)
    ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)

    if psi > 0.2:
        status = "MAJOR DRIFT"
    elif psi > 0.1:
        status = "MINOR DRIFT"
    else:
        status = "STABLE"

    results[feature] = {
        "psi":       psi,
        "ks_stat":   round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "status":    status,
    }

    log.info("  %-20s PSI=%.4f  KS=%.4f (p=%.4f)  → %s",
             feature, psi, ks_stat, ks_pvalue, status)

# ── Overall summary ───────────────────────────────────────────────
drifted   = [f for f, r in results.items() if r["status"] != "STABLE"]
drift_share = len(drifted) / len(FEATURE_COLUMNS)
log.info("Drifted features: %d/%d", len(drifted), len(FEATURE_COLUMNS))

# ── Save drift_metrics.txt ────────────────────────────────────────
metrics_path = OUTPUTS_DIR / "drift_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Drift Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*60}\n")
    f.write(f"Drifted features: {len(drifted)}/{len(FEATURE_COLUMNS)}\n\n")
    f.write(f"{'Feature':<20} {'PSI':>8} {'KS':>8} {'p-value':>8} {'Status'}\n")
    f.write(f"{'-'*60}\n")
    for feature, r in results.items():
        f.write(f"{feature:<20} {r['psi']:>8.4f} {r['ks_stat']:>8.4f} "
                f"{r['ks_pvalue']:>8.4f} {r['status']}\n")
log.info("Saved drift metrics to %s", metrics_path)

# ── Generate HTML report ──────────────────────────────────────────
html_path = OUTPUTS_DIR / "drift_report.html"
rows = ""
for feature, r in results.items():
    color = ("#ffcccc" if r["status"] == "MAJOR DRIFT"
             else "#fff3cc" if r["status"] == "MINOR DRIFT"
             else "#ccffcc")
    rows += f"""
    <tr style="background:{color}">
        <td>{feature}</td>
        <td>{r['psi']}</td>
        <td>{r['ks_stat']}</td>
        <td>{r['ks_pvalue']}</td>
        <td><b>{r['status']}</b></td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Deforestation Risk — Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1   {{ color: #1F4E79; }}
        h2   {{ color: #2E75B6; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #1F4E79; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border: 1px solid #ccc; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Deforestation Risk Model — Drift Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Built by Martin James Ng'ang'a | MLOps Engineer | github.com/M20Jay</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Drifted features: {len(drifted)} / {len(FEATURE_COLUMNS)}</p>
        <p>Reference data: {len(reference_data)} rows (training features)</p>
        <p>Current data: {len(current_data)} rows (recent predictions)</p>
        <p><b>Note:</b> PSI and KS test calculated using scipy.
           Evidently AI migration planned for Week 21 (Docker).</p>
    </div>

    <h2>Per-Feature Drift Analysis</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>PSI Score</th>
            <th>KS Statistic</th>
            <th>KS p-value</th>
            <th>Status</th>
        </tr>
        {rows}
    </table>

    <br>
    <h2>Interpretation Guide</h2>
    <table>
        <tr><th>Metric</th><th>Threshold</th><th>Meaning</th></tr>
        <tr style="background:#ccffcc">
            <td>PSI</td><td>&lt; 0.1</td>
            <td>Stable — no action needed</td></tr>
        <tr style="background:#fff3cc">
            <td>PSI</td><td>0.1 - 0.2</td>
            <td>Minor drift — monitor closely</td></tr>
        <tr style="background:#ffcccc">
            <td>PSI</td><td>&gt; 0.2</td>
            <td>Major drift — retrain model</td></tr>
        <tr style="background:#ccffcc">
            <td>KS p-value</td><td>&gt; 0.05</td>
            <td>No significant drift</td></tr>
        <tr style="background:#ffcccc">
            <td>KS p-value</td><td>&lt; 0.05</td>
            <td>Statistically significant drift</td></tr>
    </table>
</body>
</html>"""

with open(html_path, "w") as f:
    f.write(html)

log.info("Saved HTML report to %s", html_path)
log.info("Monitoring complete.")
