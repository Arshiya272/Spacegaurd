"""SpaceGuard AI - Machine Learning Pipeline Module.

Trains, evaluates, compares, and persists Random Forest and XGBoost classifiers
for conjunction risk categorization against a transparent physics-based baseline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import (
    FEATURE_COLUMNS,
    AstrodynamicFeatureExtractor,
    SyntheticConjunctionGenerator,
)

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["Low", "Warning", "Critical"]
LABEL_MAP = {"Low": 0, "Warning": 1, "Critical": 2}
REV_LABEL_MAP = {0: "Low", 1: "Warning", 2: "Critical"}


@dataclass
class ModelEvaluationResult:
    """Performance evaluation metrics for a risk screening model."""
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: List[List[int]]
    feature_importances: Dict[str, float]
    classification_report: Dict[str, Any]


class ConjunctionRiskMLPipeline:
    """Machine learning training, comparison, serialization, and inference pipeline."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ):
        if models_dir is None:
            self.models_dir = DEFAULT_PROJECT_ROOT / "models"
        else:
            self.models_dir = Path(models_dir) if Path(models_dir).is_absolute() else DEFAULT_PROJECT_ROOT / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model: Optional[Any] = None
        self.evaluation_results: Dict[str, ModelEvaluationResult] = {}
        self.feature_names = FEATURE_COLUMNS

    def train_and_evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.25,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """Train Random Forest and XGBoost models, evaluate vs physics baseline, and serialize."""
        # If no dataset supplied, generate validated synthetic scenario dataset
        if df is None:
            logger.info("Generating 3,000 synthetic simulation encounter scenarios...")
            gen = SyntheticConjunctionGenerator(random_seed=self.random_state)
            df = gen.generate_simulation_dataset(num_samples=3000)

        # Ensure ground truth target column
        if "ground_truth_risk" not in df.columns:
            raise ValueError("DataFrame must contain 'ground_truth_risk' column.")

        X = df[self.feature_names].copy()
        y_str = df["ground_truth_risk"].values
        y = np.array([LABEL_MAP[val] for val in y_str])

        # Stratified hold-out split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info("Dataset partition: %d train samples, %d test samples", len(X_train), len(X_test))

        # 1. Physics-based deterministic baseline on test set
        baseline_preds_str = AstrodynamicFeatureExtractor.physics_baseline_predict(X_test)
        y_baseline = np.array([LABEL_MAP[val] for val in baseline_preds_str])
        baseline_acc = float(accuracy_score(y_test, y_baseline))
        baseline_f1 = float(f1_score(y_test, y_baseline, average="macro"))
        baseline_cm = confusion_matrix(y_test, y_baseline).tolist()
        baseline_rep = classification_report(y_test, y_baseline, target_names=CLASS_NAMES, output_dict=True)

        self.evaluation_results["Physics Baseline"] = ModelEvaluationResult(
            model_name="Physics Baseline (Deterministic)",
            accuracy=baseline_acc,
            macro_f1=baseline_f1,
            weighted_f1=float(f1_score(y_test, y_baseline, average="weighted")),
            confusion_matrix=baseline_cm,
            feature_importances={k: 0.0 for k in self.feature_names},
            classification_report=baseline_rep,
        )

        # 2. Random Forest Classifier
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)

        rf_imp = dict(zip(self.feature_names, [round(float(v), 5) for v in self.rf_model.feature_importances_]))
        rf_imp_sorted = dict(sorted(rf_imp.items(), key=lambda item: item[1], reverse=True))

        self.evaluation_results["Random Forest"] = ModelEvaluationResult(
            model_name="Random Forest",
            accuracy=float(accuracy_score(y_test, y_pred_rf)),
            macro_f1=float(f1_score(y_test, y_pred_rf, average="macro")),
            weighted_f1=float(f1_score(y_test, y_pred_rf, average="weighted")),
            confusion_matrix=confusion_matrix(y_test, y_pred_rf).tolist(),
            feature_importances=rf_imp_sorted,
            classification_report=classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, output_dict=True),
        )

        # 3. XGBoost Classifier
        try:
            from xgboost import XGBClassifier
            self.xgb_model = XGBClassifier(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                random_state=self.random_state,
                eval_metric="mlogloss",
            )
            self.xgb_model.fit(X_train, y_train)
            y_pred_xgb = self.xgb_model.predict(X_test)

            xgb_imp = dict(zip(self.feature_names, [round(float(v), 5) for v in self.xgb_model.feature_importances_]))
            xgb_imp_sorted = dict(sorted(xgb_imp.items(), key=lambda item: item[1], reverse=True))

            self.evaluation_results["XGBoost"] = ModelEvaluationResult(
                model_name="XGBoost",
                accuracy=float(accuracy_score(y_test, y_pred_xgb)),
                macro_f1=float(f1_score(y_test, y_pred_xgb, average="macro")),
                weighted_f1=float(f1_score(y_test, y_pred_xgb, average="weighted")),
                confusion_matrix=confusion_matrix(y_test, y_pred_xgb).tolist(),
                feature_importances=xgb_imp_sorted,
                classification_report=classification_report(y_test, y_pred_xgb, target_names=CLASS_NAMES, output_dict=True),
            )
        except Exception as exc:
            logger.warning("XGBoost training skipped or encountered error: %s", exc)

        # Serialize if requested
        if save_artifacts:
            self.save_models()

        summary = {
            name: {
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
                "weighted_f1": res.weighted_f1,
            }
            for name, res in self.evaluation_results.items()
        }
        return summary

    def save_models(self) -> None:
        """Serialize models and performance report to models/ directory."""
        if self.rf_model is not None:
            joblib.dump(self.rf_model, self.models_dir / "random_forest_model.joblib")
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, self.models_dir / "xgboost_model.joblib")

        report = {
            name: {
                "model_name": res.model_name,
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
                "weighted_f1": res.weighted_f1,
                "confusion_matrix": res.confusion_matrix,
                "feature_importances": res.feature_importances,
                "classification_report": res.classification_report,
            }
            for name, res in self.evaluation_results.items()
        }

        with open(self.models_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved models and evaluation report to %s", self.models_dir)

    def load_models(self) -> bool:
        """Load pre-trained models and report from models/ directory."""
        rf_path = self.models_dir / "random_forest_model.joblib"
        xgb_path = self.models_dir / "xgboost_model.joblib"
        report_path = self.models_dir / "evaluation_report.json"

        loaded_any = False
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
            loaded_any = True
        if xgb_path.exists():
            try:
                self.xgb_model = joblib.load(xgb_path)
                loaded_any = True
            except Exception as e:
                logger.warning("Could not load XGBoost model: %s", e)

        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    rep_dict = json.load(f)
                for name, d in rep_dict.items():
                    self.evaluation_results[name] = ModelEvaluationResult(
                        model_name=d["model_name"],
                        accuracy=d["accuracy"],
                        macro_f1=d["macro_f1"],
                        weighted_f1=d["weighted_f1"],
                        confusion_matrix=d["confusion_matrix"],
                        feature_importances=d.get("feature_importances", {}),
                        classification_report=d.get("classification_report", {}),
                    )
            except Exception as e:
                logger.warning("Could not load evaluation report: %s", e)

        return loaded_any

    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, float]],
        model_type: str = "xgboost",
    ) -> Dict[str, Any]:
        """Classify risk for a single feature vector or DataFrame."""
        if isinstance(features, dict):
            df_feat = pd.DataFrame([features])[self.feature_names]
        else:
            df_feat = features[self.feature_names].copy()

        # Physics baseline
        physics_pred = AstrodynamicFeatureExtractor.physics_baseline_predict(df_feat)
        if isinstance(physics_pred, list):
            physics_pred_val = physics_pred[0]
        else:
            physics_pred_val = physics_pred

        # Select model
        active_model = self.xgb_model if (model_type.lower() == "xgboost" and self.xgb_model is not None) else self.rf_model

        if active_model is None:
            # Fallback to physics baseline if models not yet fitted
            return {
                "predicted_risk": physics_pred_val,
                "probabilities": {cls: (1.0 if cls == physics_pred_val else 0.0) for cls in CLASS_NAMES},
                "physics_baseline_risk": physics_pred_val,
                "model_used": "Physics Baseline (Fallback)",
                "disclaimer": (
                    "Deterministic astrodynamic screening rule. "
                    "Not an empirical collision probability."
                ),
            }

        pred_idx = int(active_model.predict(df_feat)[0])
        pred_label = REV_LABEL_MAP.get(pred_idx, "Unknown")

        probs = active_model.predict_proba(df_feat)[0]
        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

        return {
            "predicted_risk": pred_label,
            "probabilities": prob_dict,
            "physics_baseline_risk": physics_pred_val,
            "model_used": type(active_model).__name__,
            "disclaimer": (
                "ML screening classification reflects algorithmic prioritization. "
                "Does not represent statistical collision probability (Pc)."
            ),
        }
