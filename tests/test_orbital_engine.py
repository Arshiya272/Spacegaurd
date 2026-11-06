"""Tests for SGP4 orbital mechanics, coordinate systems, and state vectors."""

import math
from datetime import datetime, timezone
import numpy as np
import pytest

from src.orbital_engine import (
    EARTH_RADIUS_KM,
    MU_EARTH,
    OrbitalEngine,
    StateVector,
)


@pytest.fixture
def engine():
    return OrbitalEngine()


@pytest.fixture
def iss_omm():
    return {
        "OBJECT_NAME": "ISS (ZARYA)",
        "OBJECT_ID": "1998-067A",
        "NORAD_CAT_ID": 25544,
        "EPOCH": "2026-11-05T12:00:00Z",
        "MEAN_MOTION": 15.49842142,
        "ECCENTRICITY": 0.0004523,
        "INCLINATION": 51.6412,
        "RA_OF_ASC_NODE": 120.3541,
        "ARG_OF_PERICENTER": 75.1234,
        "MEAN_ANOMALY": 285.0123,
        "BSTAR": 0.00021432,
    }


def test_datetime_to_jd_standard_epoch(engine):
    # Standard J2000.0 epoch: 2000-01-01 12:00:00 UTC -> JD 2451545.0
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd_day, fr = engine.datetime_to_jd(dt)
    assert (jd_day + fr) == pytest.approx(2451545.0, abs=1e-5)


def test_gmst_calculation(engine):
    # At J2000.0, GMST ~ 280.46 degrees ~ 4.8949 radians
    jd, fr = 2451545.0, 0.0
    gmst = engine.gmst_from_jd(jd, fr)
    assert 0.0 <= gmst < (2.0 * math.pi)


def test_extract_orbital_elements(engine, iss_omm):
    elems = engine.extract_orbital_elements(iss_omm)
    assert elems.orbit_regime == "LEO"
    assert 350.0 <= elems.perigee_altitude_km <= 450.0
    assert 350.0 <= elems.apogee_altitude_km <= 450.0
    assert 90.0 <= elems.orbital_period_minutes <= 95.0
    assert elems.semi_major_axis_km == pytest.approx(EARTH_RADIUS_KM + 420.0, abs=50.0)


def test_sgp4_propagation_physical_validity(engine, iss_omm):
    satrec = engine.omm_to_satrec(iss_omm)
    assert satrec is not None

    prop_dt = datetime(2026, 11, 5, 13, 30, 0, tzinfo=timezone.utc)
    state = engine.propagate(satrec, prop_dt, omm_fallback=iss_omm)

    assert state.is_valid is True
    assert state.error_code == 0

    # LEO altitude should remain between 350 km and 500 km
    assert 350.0 <= state.altitude_km <= 500.0

    # Orbital speed in LEO should be approximately 7.6 - 7.8 km/s
    assert 7.4 <= state.speed_km_s <= 7.9

    # Position vector norm equals Earth radius + altitude
    pos_norm = np.linalg.norm(state.position_km)
    assert pos_norm == pytest.approx(EARTH_RADIUS_KM + state.altitude_km, abs=0.1)


def test_generate_orbit_path_shape(engine, iss_omm):
    satrec = engine.omm_to_satrec(iss_omm)
    start_dt = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
    path = engine.generate_orbit_path(satrec, start_dt, period_minutes=92.0, num_points=50, omm_fallback=iss_omm)

    assert len(path) > 40
    assert path.shape[1] == 3
    # Check that points are non-zero and in LEO range
    r_mags = np.linalg.norm(path, axis=1)
    assert np.all(r_mags > 6000.0)
    assert np.all(r_mags < 7500.0)
