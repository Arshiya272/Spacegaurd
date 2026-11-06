"""Tests for close-approach conjunction screening, TCA, and RIC frame decomposition."""

from datetime import datetime, timezone
import numpy as np
import pytest

from src.conjunction import ConjunctionEngine
from src.orbital_engine import OrbitalElements, OrbitalEngine


@pytest.fixture
def conj_engine():
    return ConjunctionEngine()


def test_orbit_overlap_filter_separated(conj_engine):
    # Sat A in LEO (400 km)
    elems_a = OrbitalElements(
        semi_major_axis_km=6778.0,
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        mean_motion_rev_day=15.5,
        orbital_period_minutes=92.0,
        perigee_altitude_km=395.0,
        apogee_altitude_km=405.0,
        bstar=0.0,
        orbit_regime="LEO",
    )
    # Sat B in GEO (35,786 km)
    elems_b = OrbitalElements(
        semi_major_axis_km=42164.0,
        eccentricity=0.0001,
        inclination_deg=0.1,
        raan_deg=0.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        mean_motion_rev_day=1.0,
        orbital_period_minutes=1436.0,
        perigee_altitude_km=35780.0,
        apogee_altitude_km=35790.0,
        bstar=0.0,
        orbit_regime="GEO",
    )
    # They are thousands of km apart, overlap check MUST return False
    assert conj_engine.check_orbit_overlap(elems_a, elems_b, buffer_km=100.0) is False


def test_orbit_overlap_filter_intersecting(conj_engine):
    # Both in similar LEO altitude band
    elems_a = OrbitalElements(
        semi_major_axis_km=7100.0,
        eccentricity=0.01,
        inclination_deg=98.0,
        raan_deg=10.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        mean_motion_rev_day=14.5,
        orbital_period_minutes=98.0,
        perigee_altitude_km=650.0,
        apogee_altitude_km=790.0,
        bstar=0.0,
        orbit_regime="LEO",
    )
    elems_b = OrbitalElements(
        semi_major_axis_km=7120.0,
        eccentricity=0.02,
        inclination_deg=74.0,
        raan_deg=25.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        mean_motion_rev_day=14.4,
        orbital_period_minutes=98.5,
        perigee_altitude_km=635.0,
        apogee_altitude_km=815.0,
        bstar=0.0,
        orbit_regime="LEO",
    )
    assert conj_engine.check_orbit_overlap(elems_a, elems_b, buffer_km=50.0) is True


def test_compute_ric_frame_orthonormality(conj_engine):
    # Arbitrary state vector in LEO
    pos = np.array([5000.0, 3000.0, 2500.0])
    vel = np.array([-2.5, 6.2, 3.1])

    u_r, u_i, u_c = conj_engine.compute_ric_frame(pos, vel)

    # Unit norms
    assert np.linalg.norm(u_r) == pytest.approx(1.0, abs=1e-7)
    assert np.linalg.norm(u_i) == pytest.approx(1.0, abs=1e-7)
    assert np.linalg.norm(u_c) == pytest.approx(1.0, abs=1e-7)

    # Mutual orthogonality (dot products == 0)
    assert np.dot(u_r, u_i) == pytest.approx(0.0, abs=1e-7)
    assert np.dot(u_r, u_c) == pytest.approx(0.0, abs=1e-7)
    assert np.dot(u_i, u_c) == pytest.approx(0.0, abs=1e-7)


def test_screen_pair_conjunction_detection(conj_engine):
    # Two identical orbit shells with slight phase offset
    now_utc = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
    sat_a = {
        "OBJECT_NAME": "PRIMARY_SAT",
        "OBJECT_ID": "2020-001A",
        "NORAD_CAT_ID": 50001,
        "EPOCH": "2026-11-05T12:00:00Z",
        "MEAN_MOTION": 15.0,
        "ECCENTRICITY": 0.001,
        "INCLINATION": 53.0,
        "RA_OF_ASC_NODE": 100.0,
        "ARG_OF_PERICENTER": 45.0,
        "MEAN_ANOMALY": 180.0,
        "OBJECT_TYPE": "ACTIVE",
    }
    sat_b = {
        "OBJECT_NAME": "CHASER_DEBRIS",
        "OBJECT_ID": "2020-001B",
        "NORAD_CAT_ID": 50002,
        "EPOCH": "2026-11-05T12:00:00Z",
        "MEAN_MOTION": 15.0,
        "ECCENTRICITY": 0.001,
        "INCLINATION": 53.05,
        "RA_OF_ASC_NODE": 100.05,
        "ARG_OF_PERICENTER": 45.0,
        "MEAN_ANOMALY": 180.1,  # Close approach near epoch
        "OBJECT_TYPE": "DEBRIS",
    }
    rec_a = conj_engine.engine.omm_to_satrec(sat_a)
    rec_b = conj_engine.engine.omm_to_satrec(sat_b)

    event = conj_engine.screen_pair(
        sat_a,
        sat_b,
        rec_a,
        rec_b,
        start_dt=now_utc,
        duration_hours=2.0,
        step_seconds=30.0,
    )

    assert event is not None
    assert event.miss_distance_km < 100.0
    assert event.relative_speed_km_s > 0.0
    assert event.target_id == "50001"
    assert event.chaser_id == "50002"
    assert event.physics_risk_level in ["CRITICAL", "WARNING", "MONITORING", "LOW"]
