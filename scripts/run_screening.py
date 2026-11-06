#!/usr/bin/env python3
"""SpaceGuard AI - End-to-End Conjunction Screening Workflow.

Performs orbital screening between active satellites and space debris,
extracts astrodynamic encounter features, classifies collision risk tiers
using ML models and physics baseline, and exports reports to outputs/.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.conjunction import ConjunctionEngine
from src.data_loader import CelesTrakLoader
from src.feature_engineering import AstrodynamicFeatureExtractor
from src.ml_model import ConjunctionRiskMLPipeline
from src.orbital_engine import OrbitalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScreeningWorkflow")


def main():
    parser = argparse.ArgumentParser(description="Run Conjunction Assessment Screening")
    parser.add_argument("--hours", type=float, default=24.0, help="Screening time horizon in hours")
    parser.add_argument("--step-sec", type=float, default=120.0, help="Propagation time step in seconds")
    parser.add_argument("--threshold-km", type=float, default=150.0, help="Spatial screening threshold in km")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "random_forest"], help="ML model")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "outputs"), help="Outputs destination")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ingest orbital catalogs
    loader = CelesTrakLoader()
    active_sats, _ = loader.fetch_group("active")
    stations, _ = loader.fetch_group("stations")
    primaries = active_sats + stations

    c_deb, _ = loader.fetch_group("cosmos-2251-debris")
    i_deb, _ = loader.fetch_group("iridium-33-debris")
    f_deb, _ = loader.fetch_group("fengyun-1c-debris")
    secondaries = c_deb + i_deb + f_deb

    logger.info("Loaded %d primary satellites and %d space debris objects.", len(primaries), len(secondaries))

    # Initialize orbital & screening engines
    orb_engine = OrbitalEngine()
    conj_engine = ConjunctionEngine(
        engine=orb_engine,
        screening_threshold_km=args.threshold_km,
        warning_threshold_km=50.0,
        critical_threshold_km=10.0,
    )

    # Initialize ML pipeline
    ml_pipe = ConjunctionRiskMLPipeline()
    ml_pipe.load_models()

    now_utc = datetime.now(timezone.utc)
    logger.info("Screening epoch: %s (Horizon: %.1f hours, Step: %.1fs)", now_utc.isoformat(), args.hours, args.step_sec)

    events = conj_engine.screen_catalog(
        primary_objects=primaries,
        secondary_objects=secondaries,
        start_dt=now_utc,
        duration_hours=args.hours,
        step_seconds=args.step_sec,
    )

    logger.info("Screening completed. Identified %d close approaches within %.1f km.", len(events), args.threshold_km)

    report_rows = []
    feat_extractor = AstrodynamicFeatureExtractor()

    for ev in events:
        feats = feat_extractor.extract_from_event(ev)
        ml_res = ml_pipe.predict(feats, model_type=args.model)

        row = ev.to_dict()
        row["ml_model_used"] = ml_res["model_used"]
        row["ml_predicted_risk"] = ml_res["predicted_risk"]
        row["ml_risk_probabilities"] = ml_res["probabilities"]
        row["physics_baseline_risk"] = ml_res["physics_baseline_risk"]
        row["ml_disclaimer"] = ml_res["disclaimer"]
        report_rows.append(row)

    # Export outputs
    json_path = out_dir / "conjunction_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2, default=str)

    if report_rows:
        df_out = pd.DataFrame(report_rows)
        # Drop large nested vectors for clean CSV view
        csv_df = df_out.drop(columns=["target_position_eci", "chaser_position_eci", "ml_risk_probabilities"], errors="ignore")
        csv_path = out_dir / "conjunction_report.csv"
        csv_df.to_csv(csv_path, index=False)
        logger.info("Exported conjunction assessment reports to %s and %s", json_path, csv_path)
    else:
        logger.info("No conjunctions detected within threshold. Summary report saved to %s", json_path)


if __name__ == "__main__":
    main()
