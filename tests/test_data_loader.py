"""Tests for CelesTrak authentic orbital data ingestion and validation."""

import json
from pathlib import Path
import pytest

from src.data_loader import CelesTrakLoader, REQUIRED_OMM_FIELDS


def test_validate_catalog_valid_record():
    loader = CelesTrakLoader()
    sample_records = [
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": 25544,
            "EPOCH": "2026-11-05T12:00:00Z",
            "MEAN_MOTION": 15.498,
            "ECCENTRICITY": 0.00045,
            "INCLINATION": 51.64,
            "RA_OF_ASC_NODE": 120.35,
            "ARG_OF_PERICENTER": 75.12,
            "MEAN_ANOMALY": 285.01,
            "BSTAR": 0.00021,
        }
    ]
    valid_recs, stats = loader.validate_catalog(sample_records, default_type="PAYLOAD")
    assert len(valid_recs) == 1
    assert stats["valid_count"] == 1
    assert stats["discarded_count"] == 0
    assert valid_recs[0]["OBJECT_TYPE"] == "PAYLOAD"


def test_validate_catalog_rejects_unphysical_bounds():
    loader = CelesTrakLoader()
    invalid_records = [
        # Unphysical eccentricity >= 1.0
        {
            "OBJECT_NAME": "BAD_ECC",
            "EPOCH": "2026-11-05T12:00:00Z",
            "MEAN_MOTION": 15.0,
            "ECCENTRICITY": 1.25,
            "INCLINATION": 51.0,
            "RA_OF_ASC_NODE": 0.0,
            "ARG_OF_PERICENTER": 0.0,
            "MEAN_ANOMALY": 0.0,
        },
        # Unphysical negative mean motion
        {
            "OBJECT_NAME": "BAD_MM",
            "EPOCH": "2026-11-05T12:00:00Z",
            "MEAN_MOTION": -2.0,
            "ECCENTRICITY": 0.01,
            "INCLINATION": 51.0,
            "RA_OF_ASC_NODE": 0.0,
            "ARG_OF_PERICENTER": 0.0,
            "MEAN_ANOMALY": 0.0,
        },
        # Missing required field
        {
            "OBJECT_NAME": "MISSING_FIELDS",
            "MEAN_MOTION": 15.0,
        },
    ]
    valid_recs, stats = loader.validate_catalog(invalid_records)
    assert len(valid_recs) == 0
    assert stats["discarded_count"] == 3


def test_parse_tle_text():
    loader = CelesTrakLoader()
    raw_tle = """
ISS (ZARYA)
1 25544U 98067A   26309.85070602  .00014285  00000+0  25838-3 0  9993
2 25544  51.6412 120.3541 0004523  75.1234 285.0123 15.49842142450008
"""
    records = loader.parse_tle_text(raw_tle, object_type="PAYLOAD")
    assert len(records) == 1
    rec = records[0]
    assert rec["OBJECT_NAME"] == "ISS (ZARYA)"
    assert rec["NORAD_CAT_ID"] == 25544
    assert pytest.approx(rec["INCLINATION"], 0.001) == 51.6412
    assert pytest.approx(rec["MEAN_MOTION"], 0.001) == 15.49842142


def test_offline_cached_fallback(tmp_path):
    loader = CelesTrakLoader(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    # Pre-populate dummy cache
    dummy_file = tmp_path / "raw" / "stations.json"
    dummy_meta = tmp_path / "raw" / "stations.meta.json"
    dummy_recs = [{
        "OBJECT_NAME": "CACHED_SAT",
        "NORAD_CAT_ID": 11111,
        "EPOCH": "2026-11-05T12:00:00Z",
        "MEAN_MOTION": 15.0,
        "ECCENTRICITY": 0.001,
        "INCLINATION": 45.0,
        "RA_OF_ASC_NODE": 10.0,
        "ARG_OF_PERICENTER": 10.0,
        "MEAN_ANOMALY": 10.0,
    }]
    with open(dummy_file, "w") as f:
        json.dump(dummy_recs, f)
    with open(dummy_meta, "w") as f:
        json.dump({"source": "Local Unit Test Cache"}, f)

    recs, meta = loader.fetch_group("stations", force_refresh=False)
    assert len(recs) == 1
    assert recs[0]["OBJECT_NAME"] == "CACHED_SAT"
    assert meta["offline_mode"] is True
