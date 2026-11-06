"""SpaceGuard AI - Authentic Orbital Data Ingestion Module.

Handles automated retrieval, validation, and local caching of Orbit Mean-Elements
Message (OMM) datasets and Two-Line Elements (TLEs) from CelesTrak.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard CelesTrak Query Endpoints
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Validated CelesTrak Group identifiers
CATALOG_GROUPS = {
    "stations": {"name": "Space Stations", "group": "stations", "type": "PAYLOAD"},
    "active": {"name": "Active Satellites", "group": "active", "type": "ACTIVE"},
    "starlink": {"name": "Starlink Constellation", "group": "starlink", "type": "ACTIVE"},
    "cosmos-2251-debris": {"name": "Cosmos 2251 Debris", "special": "1999-025", "type": "DEBRIS"},
    "iridium-33-debris": {"name": "Iridium 33 Debris", "special": "1997-051", "type": "DEBRIS"},
    "fengyun-1c-debris": {"name": "Fengyun 1C Debris", "special": "1982-092", "type": "DEBRIS"},
}

REQUIRED_OMM_FIELDS = [
    "OBJECT_NAME",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
]


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

class CelesTrakLoader:
    """Client for retrieving, validating, and caching authentic orbital data from CelesTrak."""

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        request_timeout: int = 12,
    ):
        if raw_dir is None:
            self.raw_dir = DEFAULT_PROJECT_ROOT / "data" / "raw"
        else:
            self.raw_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else DEFAULT_PROJECT_ROOT / raw_dir

        if processed_dir is None:
            self.processed_dir = DEFAULT_PROJECT_ROOT / "data" / "processed"
        else:
            self.processed_dir = Path(processed_dir) if Path(processed_dir).is_absolute() else DEFAULT_PROJECT_ROOT / processed_dir

        self.request_timeout = request_timeout

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_group(
        self,
        group_key: str,
        fmt: str = "json",
        force_refresh: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fetch orbital elements for a specified CelesTrak catalog group.

        Args:
            group_key: Key in CATALOG_GROUPS (e.g. 'stations', 'cosmos-2251-debris')
            fmt: Format string ('json', 'csv', or 'tle')
            force_refresh: If True, bypass cache and query CelesTrak directly.

        Returns:
            Tuple of (records_list, metadata_dict)
        """
        group_info = CATALOG_GROUPS.get(group_key)
        if not group_info:
            raise ValueError(f"Unknown catalog group '{group_key}'. Choose from: {list(CATALOG_GROUPS.keys())}")

        cache_filename = f"{group_key}.{fmt}"
        cache_path = self.raw_dir / cache_filename
        meta_path = self.raw_dir / f"{group_key}.meta.json"

        # Check local cache first if refresh is not forced
        if not force_refresh and cache_path.exists():
            logger.info("Loading cached %s dataset from %s", group_key, cache_path)
            records, meta = self._load_from_cache(cache_path, meta_path)
            if records:
                meta["offline_mode"] = True
                meta["source"] = f"Local Cache ({cache_path.name})"
                return records, meta

        # Query CelesTrak
        params = {"FORMAT": fmt}
        if "special" in group_info:
            params["SPECIAL"] = group_info["special"]
        else:
            params["GROUP"] = group_info["group"]

        url = CELESTRAK_GP_URL
        metadata = {
            "group_key": group_key,
            "group_name": group_info["name"],
            "object_type": group_info["type"],
            "url": url,
            "params": params,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "format": fmt,
            "offline_mode": False,
        }

        try:
            logger.info("Requesting %s from CelesTrak (%s)...", group_key, params)
            resp = requests.get(url, params=params, timeout=self.request_timeout)
            resp.raise_for_status()

            if fmt.lower() == "json":
                raw_json = resp.json()
                if isinstance(raw_json, list):
                    records = raw_json
                elif isinstance(raw_json, dict):
                    records = [raw_json]
                else:
                    records = []
            elif fmt.lower() == "tle":
                records = self.parse_tle_text(resp.text, object_type=group_info["type"])
            else:
                records = []

            # Validate records
            valid_records, validation_stats = self.validate_catalog(records, default_type=group_info["type"])
            metadata.update(validation_stats)
            metadata["source"] = f"CelesTrak Online ({url})"

            # Save to raw cache
            self._save_to_cache(cache_path, meta_path, valid_records, metadata)
            logger.info("Successfully fetched %d valid records for %s", len(valid_records), group_key)
            return valid_records, metadata

        except Exception as exc:
            logger.warning("CelesTrak API request failed for %s: %s. Attempting offline fallback.", group_key, exc)
            if cache_path.exists():
                records, meta = self._load_from_cache(cache_path, meta_path)
                meta["offline_mode"] = True
                meta["fallback_reason"] = str(exc)
                meta["source"] = f"Local Cache (Fallback: {cache_path.name})"
                return records, meta

            logger.error("No local cache available for %s", group_key)
            return [], {
                **metadata,
                "offline_mode": True,
                "error": str(exc),
                "record_count": 0,
                "valid_count": 0,
            }

    def validate_catalog(
        self, records: List[Dict[str, Any]], default_type: str = "UNKNOWN"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate OMM records for physical plausibility and completeness."""
        valid_records: List[Dict[str, Any]] = []
        discarded_count = 0

        for r in records:
            if not isinstance(r, dict):
                discarded_count += 1
                continue

            # Ensure object type tag
            if "OBJECT_TYPE" not in r:
                r["OBJECT_TYPE"] = default_type

            # Check required fields
            has_required = all(k in r and r[k] is not None for k in REQUIRED_OMM_FIELDS)
            if not has_required:
                discarded_count += 1
                continue

            try:
                # Numerical bounds validation
                ecc = float(r["ECCENTRICITY"])
                inc = float(r["INCLINATION"])
                mm = float(r["MEAN_MOTION"])

                if not (0.0 <= ecc < 1.0):
                    discarded_count += 1
                    continue
                if not (0.0 <= inc <= 180.0):
                    discarded_count += 1
                    continue
                if mm <= 0.0 or mm > 30.0:  # Earth orbit physical bounds (~48 min period min)
                    discarded_count += 1
                    continue

                # Ensure NORAD catalog id exists
                if "NORAD_CAT_ID" not in r and "OBJECT_ID" in r:
                    r["NORAD_CAT_ID"] = r["OBJECT_ID"]

                valid_records.append(r)
            except (ValueError, TypeError):
                discarded_count += 1

        stats = {
            "total_received": len(records),
            "valid_count": len(valid_records),
            "discarded_count": discarded_count,
            "validation_passed": len(valid_records) > 0,
        }
        return valid_records, stats

    def parse_tle_text(self, text: str, object_type: str = "UNKNOWN") -> List[Dict[str, Any]]:
        """Parse raw TLE format text into standardized OMM-compatible dictionaries."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        records = []

        i = 0
        while i < len(lines):
            # 3-line format: Name, Line 1, Line 2
            if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                name = lines[i]
                l1 = lines[i + 1]
                l2 = lines[i + 2]
                i += 3
            # 2-line format: Line 1, Line 2
            elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                l1 = lines[i]
                l2 = lines[i + 1]
                name = f"OBJECT-{l1[2:7].strip()}"
                i += 2
            else:
                i += 1
                continue

            try:
                norad_id = int(l1[2:7].strip())
                int_des = l1[9:17].strip()
                epoch_year = int(l1[18:20].strip())
                epoch_days = float(l1[20:32].strip())
                bstar = float(f"{l1[53:59].strip()}e{l1[59:61].strip()}") if l1[53:59].strip() else 0.0

                inc = float(l2[8:16].strip())
                raan = float(l2[17:25].strip())
                ecc = float("0." + l2[26:33].strip())
                arg_per = float(l2[34:42].strip())
                mean_anom = float(l2[43:51].strip())
                mean_motion = float(l2[52:63].strip())

                # Approximate ISO epoch
                full_year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
                epoch_iso = f"{full_year}-01-01T00:00:00Z"

                record = {
                    "OBJECT_NAME": name,
                    "OBJECT_ID": int_des or str(norad_id),
                    "NORAD_CAT_ID": norad_id,
                    "OBJECT_TYPE": object_type,
                    "EPOCH": epoch_iso,
                    "MEAN_MOTION": mean_motion,
                    "ECCENTRICITY": ecc,
                    "INCLINATION": inc,
                    "RA_OF_ASC_NODE": raan,
                    "ARG_OF_PERICENTER": arg_per,
                    "MEAN_ANOMALY": mean_anom,
                    "BSTAR": bstar,
                    "TLE_LINE1": l1,
                    "TLE_LINE2": l2,
                }
                records.append(record)
            except Exception as e:
                logger.debug("Failed parsing TLE block: %s", e)

        return records

    def _save_to_cache(
        self,
        cache_path: Path,
        meta_path: Path,
        records: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        """Serialize data and metadata to local raw cache."""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            logger.error("Failed saving cache to %s: %s", cache_path, exc)

    def _load_from_cache(
        self, cache_path: Path, meta_path: Path
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data and metadata from local raw cache."""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            return records, meta
        except Exception as exc:
            logger.error("Failed reading cache from %s: %s", cache_path, exc)
            return [], {}
