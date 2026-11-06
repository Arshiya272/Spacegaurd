#!/usr/bin/env python3
"""SpaceGuard AI - Machine Learning Model Training Workflow.

Generates a validated synthetic conjunction simulation dataset, trains Random Forest
and XGBoost classifiers, benchmarks them against a deterministic physics baseline,
and serializes model weights and metrics to models/.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import SyntheticConjunctionGenerator
from src.ml_model import ConjunctionRiskMLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainWorkflow")


def main():
    parser = argparse.ArgumentParser(description="Train SpaceGuard AI Risk Classification Models")
    parser.add_argument("--samples", type=int, default=3000, help="Number of simulation encounter scenarios")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set fraction for hold-out evaluation")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Destination folder for serialized models",
    )
    parser.add_argument(
        "--data-out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "simulated_conjunction_dataset.csv"),
        help="Path to save generated synthetic training dataset",
    )
    args = parser.parse_args()

    logger.info("Initializing SpaceGuard AI ML Pipeline (Project Root: %s)...", PROJECT_ROOT)
    logger.info("Generating %d synthetic encounter scenarios (LEO/MEO physical distributions)...", args.samples)

    generator = SyntheticConjunctionGenerator(random_seed=42)
    df_sim = generator.generate_simulation_dataset(num_samples=args.samples)

    # Save synthetic training dataset with clear metadata
    out_path = Path(args.data_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sim.to_csv(out_path, index=False)
    logger.info("Saved synthetic simulation dataset to %s (%d rows)", out_path, len(df_sim))

    # Initialize and run pipeline
    pipeline = ConjunctionRiskMLPipeline(models_dir=args.models_dir, random_state=42)
    metrics_summary = pipeline.train_and_evaluate(df=df_sim, test_size=args.test_size, save_artifacts=True)

    logger.info("==================================================")
    logger.info("           MODEL EVALUATION BENCHMARK             ")
    logger.info("==================================================")
    for model_name, stats in metrics_summary.items():
        logger.info(
            "%-30s | Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
            model_name,
            stats["accuracy"],
            stats["macro_f1"],
            stats["weighted_f1"],
        )
    logger.info("==================================================")
    logger.info("Models and evaluation report successfully persisted to %s", args.models_dir)


if __name__ == "__main__":
    main()
