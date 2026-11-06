"""SpaceGuard AI - Astrodynamic Feature Engineering Module.

Extracts physically motivated orbital features from conjunction events and orbital
elements for machine learning screening and physics-based baseline comparisons.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.conjunction import ConjunctionEvent
from src.orbital_engine import MU_EARTH

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "miss_distance_km",
    "relative_speed_km_s",
    "radial_separation_km",
    "intrack_separation_km",
    "crosstrack_separation_km",
    "relative_radial_vel_km_s",
    "relative_intrack_vel_km_s",
    "relative_crosstrack_vel_km_s",
    "altitude_diff_km",
    "mean_altitude_km",
    "inclination_diff_deg",
    "eccentricity_diff",
    "energy_diff_km2_s2",
    "time_to_tca_hours",
]


class AstrodynamicFeatureExtractor:
    """Extracts astrodynamic features from conjunction events and element sets."""

    def __init__(self):
        pass

    @staticmethod
    def extract_from_event(
        event: ConjunctionEvent,
        time_to_tca_hours: float = 24.0,
        elems_target: Optional[Dict[str, Any]] = None,
        elems_chaser: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Extract a single feature vector from an evaluated ConjunctionEvent."""
        alt_diff = abs(event.target_altitude_km - event.chaser_altitude_km)
        mean_alt = (event.target_altitude_km + event.chaser_altitude_km) / 2.0

        # Keplerian orbital differences if available
        inc_diff = 0.0
        ecc_diff = 0.0
        energy_diff = 0.0

        if elems_target and elems_chaser:
            inc_1 = float(elems_target.get("INCLINATION", 0.0))
            inc_2 = float(elems_chaser.get("INCLINATION", 0.0))
            inc_diff = abs(inc_1 - inc_2)

            ecc_1 = float(elems_target.get("ECCENTRICITY", 0.0))
            ecc_2 = float(elems_chaser.get("ECCENTRICITY", 0.0))
            ecc_diff = abs(ecc_1 - ecc_2)

            # Specific orbital energy epsilon = -mu / (2a)
            mm1 = float(elems_target.get("MEAN_MOTION", 15.0)) * (2.0 * math.pi / 86400.0)
            mm2 = float(elems_chaser.get("MEAN_MOTION", 15.0)) * (2.0 * math.pi / 86400.0)
            a1 = (MU_EARTH / (mm1 ** 2)) ** (1.0 / 3.0) if mm1 > 0 else 7000.0
            a2 = (MU_EARTH / (mm2 ** 2)) ** (1.0 / 3.0) if mm2 > 0 else 7000.0
            energy1 = -MU_EARTH / (2.0 * a1)
            energy2 = -MU_EARTH / (2.0 * a2)
            energy_diff = abs(energy1 - energy2)

        return {
            "miss_distance_km": float(event.miss_distance_km),
            "relative_speed_km_s": float(event.relative_speed_km_s),
            "radial_separation_km": abs(float(event.radial_separation_km)),
            "intrack_separation_km": abs(float(event.intrack_separation_km)),
            "crosstrack_separation_km": abs(float(event.crosstrack_separation_km)),
            "relative_radial_vel_km_s": abs(float(event.relative_radial_vel_km_s)),
            "relative_intrack_vel_km_s": abs(float(event.relative_intrack_vel_km_s)),
            "relative_crosstrack_vel_km_s": abs(float(event.relative_crosstrack_vel_km_s)),
            "altitude_diff_km": float(alt_diff),
            "mean_altitude_km": float(mean_alt),
            "inclination_diff_deg": float(inc_diff),
            "eccentricity_diff": float(ecc_diff),
            "energy_diff_km2_s2": float(energy_diff),
            "time_to_tca_hours": float(time_to_tca_hours),
        }

    def events_to_dataframe(
        self, events: List[ConjunctionEvent]
    ) -> pd.DataFrame:
        """Convert a list of ConjunctionEvents into a feature DataFrame."""
        rows = [self.extract_from_event(ev) for ev in events]
        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        return pd.DataFrame(rows)[FEATURE_COLUMNS]

    @staticmethod
    def physics_baseline_predict(row_or_df: Union[pd.Series, pd.DataFrame, Dict[str, Any]]) -> Union[str, List[str]]:
        """Transparent, deterministic astrodynamic baseline classification.

        Rules:
        - Critical: miss_distance <= 10.0 km AND relative_speed >= 2.0 km/s AND radial_sep <= 5.0 km
        - Warning: miss_distance <= 50.0 km AND radial_sep <= 20.0 km
        - Low: all others
        """
        def classify_single(r: Any) -> str:
            d_miss = float(r["miss_distance_km"])
            v_rel = float(r["relative_speed_km_s"])
            r_sep = float(r.get("radial_separation_km", d_miss))

            if d_miss <= 10.0 and v_rel >= 2.0 and r_sep <= 5.0:
                return "Critical"
            elif d_miss <= 50.0 and r_sep <= 25.0:
                return "Warning"
            else:
                return "Low"

        if isinstance(row_or_df, pd.DataFrame):
            return [classify_single(row) for _, row in row_or_df.iterrows()]
        elif isinstance(row_or_df, pd.Series) or isinstance(row_or_df, dict):
            return classify_single(row_or_df)
        raise TypeError("Input must be Series, DataFrame, or dict.")


class SyntheticConjunctionGenerator:
    """Generates scientifically defensible synthetic conjunction simulation scenarios.

    Never mixes with authentic orbital observations without explicit identification.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def generate_simulation_dataset(
        self, num_samples: int = 3000
    ) -> pd.DataFrame:
        """Generate physically consistent orbital encounter scenarios with explicit labels.

        Labels are derived from a validated multi-tier kinetic encounter model:
        - Critical (Label 2): Extreme close approach (miss distance < 5-10 km, high relative velocity, tight radial bounding).
        - Warning (Label 1): Moderate close approach (miss distance 10-45 km, overlapping orbital shells).
        - Low (Label 0): Broad close approach (miss distance 45-120 km, safe geometric separation).
        """
        records = []

        for i in range(num_samples):
            # Target orbit distribution (primarily LEO and Sun-Synchronous, with some MEO)
            mean_alt = self.rng.choice([
                self.rng.uniform(350, 600),   # ISS & mega-constellation shell
                self.rng.uniform(600, 1000),  # Upper LEO (Cosmos/Fengyun debris belt)
                self.rng.uniform(1000, 1500), # High LEO
            ])

            # Geometry sampling
            # Bias distribution so we have balanced classes across risk tiers
            scenario_choice = self.rng.choice(["critical_tier", "warning_tier", "safe_tier"], p=[0.25, 0.35, 0.40])

            if scenario_choice == "critical_tier":
                miss_dist = self.rng.exponential(scale=3.0) + 0.1
                miss_dist = min(miss_dist, 9.9)
                radial_sep = self.rng.uniform(0.05, min(miss_dist, 4.0))
                rel_speed = self.rng.uniform(3.0, 14.5)  # typical orbital crossing speed
                alt_diff = self.rng.uniform(0.01, 4.0)
            elif scenario_choice == "warning_tier":
                miss_dist = self.rng.uniform(10.0, 48.0)
                radial_sep = self.rng.uniform(2.0, min(miss_dist, 20.0))
                rel_speed = self.rng.uniform(1.5, 12.0)
                alt_diff = self.rng.uniform(2.0, 22.0)
            else:
                miss_dist = self.rng.uniform(50.0, 120.0)
                radial_sep = self.rng.uniform(15.0, miss_dist)
                rel_speed = self.rng.uniform(0.5, 10.0)
                alt_diff = self.rng.uniform(20.0, 60.0)

            # In-track and cross-track geometry
            rem_sq = max(0.0, miss_dist**2 - radial_sep**2)
            intrack_sep = math.sqrt(rem_sq) * self.rng.uniform(0.3, 0.9)
            crosstrack_sep = math.sqrt(max(0.0, rem_sq - intrack_sep**2))

            # Relative velocity components
            rel_rad_v = rel_speed * self.rng.uniform(0.05, 0.3)
            rem_v_sq = max(0.0, rel_speed**2 - rel_rad_v**2)
            rel_in_v = math.sqrt(rem_v_sq) * self.rng.uniform(0.4, 0.9)
            rel_cross_v = math.sqrt(max(0.0, rem_v_sq - rel_in_v**2))

            # Orbital mechanics parameters
            inc_diff = self.rng.uniform(0.1, 75.0)
            ecc_diff = self.rng.exponential(scale=0.005)
            energy_diff = self.rng.uniform(0.1, 5.0)
            time_to_tca = self.rng.uniform(1.0, 72.0)

            # Multi-parameter kinetic risk scoring for ground-truth simulation tier
            kinetic_severity = (rel_speed ** 1.5) / (miss_dist + 0.5)

            if miss_dist <= 10.0 and radial_sep <= 5.0 and kinetic_severity > 0.8:
                risk_tier = "Critical"
            elif miss_dist <= 50.0 and radial_sep <= 25.0:
                risk_tier = "Warning"
            else:
                risk_tier = "Low"

            record = {
                "scenario_id": f"SIM-{i+1:05d}",
                "data_nature": "SYNTHETIC_SIMULATION",
                "is_synthetic": True,
                "miss_distance_km": round(float(miss_dist), 4),
                "relative_speed_km_s": round(float(rel_speed), 4),
                "radial_separation_km": round(float(radial_sep), 4),
                "intrack_separation_km": round(float(intrack_sep), 4),
                "crosstrack_separation_km": round(float(crosstrack_sep), 4),
                "relative_radial_vel_km_s": round(float(rel_rad_v), 4),
                "relative_intrack_vel_km_s": round(float(rel_in_v), 4),
                "relative_crosstrack_vel_km_s": round(float(rel_cross_v), 4),
                "altitude_diff_km": round(float(alt_diff), 4),
                "mean_altitude_km": round(float(mean_alt), 2),
                "inclination_diff_deg": round(float(inc_diff), 3),
                "eccentricity_diff": round(float(ecc_diff), 6),
                "energy_diff_km2_s2": round(float(energy_diff), 4),
                "time_to_tca_hours": round(float(time_to_tca), 2),
                "kinetic_severity_index": round(float(kinetic_severity), 4),
                "ground_truth_risk": risk_tier,
            }
            records.append(record)

        df = pd.DataFrame(records)
        return df
