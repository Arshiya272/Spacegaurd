"""SpaceGuard AI - Conjunction Screening and Close-Approach Engine.

Performs astrodynamic close-approach screening, Time of Closest Approach (TCA)
minimization, and Radial-InTrack-CrossTrack (RIC) frame relative decomposition.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.orbital_engine import (
    EARTH_RADIUS_KM,
    OrbitalElements,
    OrbitalEngine,
    StateVector,
)

logger = logging.getLogger(__name__)


@dataclass
class ConjunctionEvent:
    """Astrodynamic close-approach event description between two orbital bodies."""
    target_id: str
    target_name: str
    target_type: str
    chaser_id: str
    chaser_name: str
    chaser_type: str
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    # RIC (Radial, In-track, Cross-track) relative offsets at TCA
    radial_separation_km: float
    intrack_separation_km: float
    crosstrack_separation_km: float
    relative_radial_vel_km_s: float
    relative_intrack_vel_km_s: float
    relative_crosstrack_vel_km_s: float
    target_altitude_km: float
    chaser_altitude_km: float
    target_position_eci: List[float]
    chaser_position_eci: List[float]
    physics_risk_level: str  # 'CRITICAL', 'WARNING', 'MONITORING', 'LOW'
    screening_disclaimer: str = (
        "Deterministic geometric screening baseline. "
        "Does not represent statistical collision probability (Pc)."
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tca_iso"] = self.tca.isoformat()
        return data


class ConjunctionEngine:
    """Space debris close-approach screening and astrodynamic risk evaluation."""

    def __init__(
        self,
        engine: Optional[OrbitalEngine] = None,
        screening_threshold_km: float = 100.0,
        warning_threshold_km: float = 50.0,
        critical_threshold_km: float = 10.0,
    ):
        self.engine = engine or OrbitalEngine()
        self.screening_threshold_km = screening_threshold_km
        self.warning_threshold_km = warning_threshold_km
        self.critical_threshold_km = critical_threshold_km

    def check_orbit_overlap(
        self,
        elems_a: OrbitalElements,
        elems_b: OrbitalElements,
        buffer_km: Optional[float] = None,
    ) -> bool:
        """Fast O(1) filter to eliminate non-intersecting orbit shells.

        Returns False if orbits are strictly separated in radial altitude bounds.
        """
        buf = buffer_km if buffer_km is not None else self.screening_threshold_km

        # If A's perigee is far above B's apogee, or vice versa, they cannot encounter
        if elems_a.perigee_altitude_km > (elems_b.apogee_altitude_km + buf):
            return False
        if elems_b.perigee_altitude_km > (elems_a.apogee_altitude_km + buf):
            return False

        return True

    @staticmethod
    def compute_ric_frame(
        pos_target: np.ndarray, vel_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive orthonormal Radial-InTrack-CrossTrack (RIC / RTN) coordinate axes.

        - R (Radial): Unit vector along position vector (away from Earth center)
        - C (Cross-track): Unit vector along orbital angular momentum vector (r x v)
        - I (In-track): Unit vector completing right-handed triad (C x R)
        """
        r_norm = np.linalg.norm(pos_target)
        if r_norm < 1e-6:
            return np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])

        u_r = pos_target / r_norm

        h_vec = np.cross(pos_target, vel_target)
        h_norm = np.linalg.norm(h_vec)
        if h_norm < 1e-6:
            u_c = np.array([0, 0, 1])
        else:
            u_c = h_vec / h_norm

        u_i = np.cross(u_c, u_r)
        u_i = u_i / np.linalg.norm(u_i)

        return u_r, u_i, u_c

    def screen_pair(
        self,
        sat_a: Dict[str, Any],
        sat_b: Dict[str, Any],
        satrec_a: Any,
        satrec_b: Any,
        start_dt: datetime,
        duration_hours: float = 12.0,
        step_seconds: float = 60.0,
    ) -> Optional[ConjunctionEvent]:
        """Screen a specific pair of resident space objects across a time horizon.

        Performs coarse search followed by sub-second quadratic TCA refinement.
        """
        elems_a = self.engine.extract_orbital_elements(sat_a)
        elems_b = self.engine.extract_orbital_elements(sat_b)

        # 1. Fast altitude envelope pre-filtering
        if not self.check_orbit_overlap(elems_a, elems_b):
            return None

        # 2. Time-stepped propagation
        total_steps = int(math.ceil((duration_hours * 3600.0) / step_seconds))
        min_dist = float("inf")
        min_step_idx = -1
        best_dt = start_dt

        # Ensure start_dt has timezone
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        # Coarse scan
        for step in range(total_steps):
            current_dt = start_dt + timedelta(seconds=step * step_seconds)
            state_a = self.engine.propagate(satrec_a, current_dt, omm_fallback=sat_a)
            state_b = self.engine.propagate(satrec_b, current_dt, omm_fallback=sat_b)

            if not (state_a.is_valid and state_b.is_valid):
                continue

            delta_r = state_a.position_km - state_b.position_km
            dist = float(np.linalg.norm(delta_r))

            if dist < min_dist:
                min_dist = dist
                min_step_idx = step
                best_dt = current_dt

        # If coarse minimum exceeds screening envelope, skip
        if min_dist > self.screening_threshold_km or min_step_idx < 0:
            return None

        # 3. Fine sub-second TCA refinement via golden section search
        t_left = best_dt - timedelta(seconds=step_seconds)
        t_right = best_dt + timedelta(seconds=step_seconds)

        refined_tca, fine_dist = self._refine_tca(
            sat_a, sat_b, satrec_a, satrec_b, t_left, t_right
        )

        if fine_dist > self.screening_threshold_km:
            return None

        # Compute state vectors at precise TCA
        final_state_a = self.engine.propagate(satrec_a, refined_tca, omm_fallback=sat_a)
        final_state_b = self.engine.propagate(satrec_b, refined_tca, omm_fallback=sat_b)

        delta_r = final_state_a.position_km - final_state_b.position_km
        delta_v = final_state_a.velocity_km_s - final_state_b.velocity_km_s
        rel_speed = float(np.linalg.norm(delta_v))

        # RIC decomposition relative to Target (sat_a)
        u_r, u_i, u_c = self.compute_ric_frame(
            final_state_a.position_km, final_state_a.velocity_km_s
        )

        r_sep = float(np.dot(delta_r, u_r))
        i_sep = float(np.dot(delta_r, u_i))
        c_sep = float(np.dot(delta_r, u_c))

        r_vel = float(np.dot(delta_v, u_r))
        i_vel = float(np.dot(delta_v, u_i))
        c_vel = float(np.dot(delta_v, u_c))

        # Classify deterministic physical risk tier
        if fine_dist <= self.critical_threshold_km and rel_speed >= 2.0:
            risk = "CRITICAL"
        elif fine_dist <= self.warning_threshold_km:
            risk = "WARNING"
        elif fine_dist <= self.screening_threshold_km:
            risk = "MONITORING"
        else:
            risk = "LOW"

        return ConjunctionEvent(
            target_id=str(sat_a.get("NORAD_CAT_ID", sat_a.get("OBJECT_ID", "UNKNOWN"))),
            target_name=str(sat_a.get("OBJECT_NAME", "TARGET")),
            target_type=str(sat_a.get("OBJECT_TYPE", "ACTIVE")),
            chaser_id=str(sat_b.get("NORAD_CAT_ID", sat_b.get("OBJECT_ID", "UNKNOWN"))),
            chaser_name=str(sat_b.get("OBJECT_NAME", "DEBRIS")),
            chaser_type=str(sat_b.get("OBJECT_TYPE", "DEBRIS")),
            tca=refined_tca,
            miss_distance_km=fine_dist,
            relative_speed_km_s=rel_speed,
            radial_separation_km=r_sep,
            intrack_separation_km=i_sep,
            crosstrack_separation_km=c_sep,
            relative_radial_vel_km_s=r_vel,
            relative_intrack_vel_km_s=i_vel,
            relative_crosstrack_vel_km_s=c_vel,
            target_altitude_km=final_state_a.altitude_km,
            chaser_altitude_km=final_state_b.altitude_km,
            target_position_eci=final_state_a.position_km.tolist(),
            chaser_position_eci=final_state_b.position_km.tolist(),
            physics_risk_level=risk,
        )

    def _refine_tca(
        self,
        sat_a: Dict[str, Any],
        sat_b: Dict[str, Any],
        satrec_a: Any,
        satrec_b: Any,
        t_start: datetime,
        t_end: datetime,
        tol_seconds: float = 0.5,
    ) -> Tuple[datetime, float]:
        """Golden-section minimization to find true TCA to sub-second accuracy."""
        inv_phi = (math.sqrt(5) - 1) / 2  # ~0.6180339887

        a_sec = 0.0
        b_sec = (t_end - t_start).total_seconds()
        c_sec = b_sec - inv_phi * (b_sec - a_sec)
        d_sec = a_sec + inv_phi * (b_sec - a_sec)

        def eval_dist(offset_sec: float) -> float:
            t = t_start + timedelta(seconds=offset_sec)
            s_a = self.engine.propagate(satrec_a, t, omm_fallback=sat_a)
            s_b = self.engine.propagate(satrec_b, t, omm_fallback=sat_b)
            return float(np.linalg.norm(s_a.position_km - s_b.position_km))

        fc = eval_dist(c_sec)
        fd = eval_dist(d_sec)

        while (b_sec - a_sec) > tol_seconds:
            if fc < fd:
                b_sec = d_sec
                d_sec = c_sec
                fd = fc
                c_sec = b_sec - inv_phi * (b_sec - a_sec)
                fc = eval_dist(c_sec)
            else:
                a_sec = c_sec
                c_sec = d_sec
                fc = fd
                d_sec = a_sec + inv_phi * (b_sec - a_sec)
                fd = eval_dist(d_sec)

        best_sec = (a_sec + b_sec) / 2.0
        best_tca = t_start + timedelta(seconds=best_sec)
        final_dist = eval_dist(best_sec)
        return best_tca, final_dist

    def screen_catalog(
        self,
        primary_objects: List[Dict[str, Any]],
        secondary_objects: List[Dict[str, Any]],
        start_dt: datetime,
        duration_hours: float = 12.0,
        step_seconds: float = 120.0,
        max_events: int = 50,
    ) -> List[ConjunctionEvent]:
        """Screen candidate pairs between primary targets (e.g. satellites) and secondary (debris)."""
        events: List[ConjunctionEvent] = []

        # Build Satrecs in advance
        satrecs_a = [self.engine.omm_to_satrec(o) for o in primary_objects]
        satrecs_b = [self.engine.omm_to_satrec(o) for o in secondary_objects]

        logger.info(
            "Screening %d primary objects against %d secondary objects...",
            len(primary_objects),
            len(secondary_objects),
        )

        for i, (sat_a, rec_a) in enumerate(zip(primary_objects, satrecs_a)):
            for j, (sat_b, rec_b) in enumerate(zip(secondary_objects, satrecs_b)):
                # Avoid self-conjunction
                id_a = sat_a.get("NORAD_CAT_ID", sat_a.get("OBJECT_ID"))
                id_b = sat_b.get("NORAD_CAT_ID", sat_b.get("OBJECT_ID"))
                if id_a == id_b:
                    continue

                event = self.screen_pair(
                    sat_a,
                    sat_b,
                    rec_a,
                    rec_b,
                    start_dt=start_dt,
                    duration_hours=duration_hours,
                    step_seconds=step_seconds,
                )
                if event is not None:
                    events.append(event)
                    if len(events) >= max_events:
                        break

            if len(events) >= max_events:
                break

        # Sort by miss distance ascending (most dangerous first)
        events.sort(key=lambda ev: ev.miss_distance_km)
        return events
