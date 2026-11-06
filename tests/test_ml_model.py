"""Tests for machine learning pipeline, feature engineering, and physics baseline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    AstrodynamicFeatureExtractor,
    SyntheticConjunctionGenerator,
)
from src.ml_model import ConjunctionRiskMLPipeline


def test_synthetic_conjunction_generator():
    gen = SyntheticConjunctionGenerator(random_seed=123)
    df = gen.generate_simulation_dataset(num_samples=100)

    assert len(df) == 100
    assert "ground_truth_risk" in df.columns
    assert set(df["ground_truth_risk"].unique()).issubset({"Low", "Warning", "Critical"})
    assert np.all(df["is_synthetic"] == True)
    assert np.all(df["data_nature"] == "SYNTHETIC_SIMULATION")
    assert all(col in df.columns for col in FEATURE_COLUMNS)


def test_physics_baseline_predict():
    # Critical test case: close miss, high speed, small radial sep
    crit_sample = {
        "miss_distance_km": 4.5,
        "relative_speed_km_s": 9.2,
        "radial_separation_km": 1.2,
    }
    assert AstrodynamicFeatureExtractor.physics_baseline_predict(crit_sample) == "Critical"

    # Warning test case: moderate miss, moderate radial sep
    warn_sample = {
        "miss_distance_km": 28.0,
        "relative_speed_km_s": 5.0,
        "radial_separation_km": 12.0,
    }
    assert AstrodynamicFeatureExtractor.physics_baseline_predict(warn_sample) == "Warning"

    # Low test case: large distance
    low_sample = {
        "miss_distance_km": 95.0,
        "relative_speed_km_s": 3.0,
        "radial_separation_km": 60.0,
    }
    assert AstrodynamicFeatureExtractor.physics_baseline_predict(low_sample) == "Low"


def test_ml_pipeline_train_evaluate_predict(tmp_path):
    gen = SyntheticConjunctionGenerator(random_seed=42)
    df = gen.generate_simulation_dataset(num_samples=250)

    pipe = ConjunctionRiskMLPipeline(models_dir=tmp_path / "models", random_state=42)
    summary = pipe.train_and_evaluate(df=df, test_size=0.3, save_artifacts=True)

    assert "Random Forest" in summary
    assert "Physics Baseline" in summary
    assert summary["Random Forest"]["accuracy"] >= 0.85

    # Check artifacts written to disk
    assert (tmp_path / "models" / "random_forest_model.joblib").exists()
    assert (tmp_path / "models" / "evaluation_report.json").exists()

    # Inference test
    sample_feat = {
        "miss_distance_km": 3.2,
        "relative_speed_km_s": 8.0,
        "radial_separation_km": 0.8,
        "intrack_separation_km": 2.1,
        "crosstrack_separation_km": 1.5,
        "relative_radial_vel_km_s": 1.2,
        "relative_intrack_vel_km_s": 7.0,
        "relative_crosstrack_vel_km_s": 2.0,
        "altitude_diff_km": 1.0,
        "mean_altitude_km": 500.0,
        "inclination_diff_deg": 12.0,
        "eccentricity_diff": 0.001,
        "energy_diff_km2_s2": 0.5,
        "time_to_tca_hours": 12.0,
    }

    pred_res = pipe.predict(sample_feat, model_type="random_forest")
    assert pred_res["predicted_risk"] in ["Critical", "Warning", "Low"]
    assert "probabilities" in pred_res
    assert sum(pred_res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert "disclaimer" in pred_res
