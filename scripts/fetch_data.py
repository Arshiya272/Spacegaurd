#!/usr/bin/env python3
"""SpaceGuard AI - Authentic Orbital Data Ingestion Workflow.

Retrieves and validates OMM JSON datasets from CelesTrak for active satellites
and major debris catalog groups (Cosmos 2251, Iridium 33, Fengyun 1C).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CATALOG_GROUPS, CelesTrakLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FetchData")


def main():
    parser = argparse.ArgumentParser(description="Fetch authentic orbital data from CelesTrak")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(CATALOG_GROUPS.keys()),
        help=f"Catalog groups to fetch. Options: {list(CATALOG_GROUPS.keys())}",
    )
    parser.add_argument("--format", default="json", choices=["json", "tle", "csv"], help="Orbital element format")
    parser.add_argument("--refresh", action="store_true", help="Force refresh bypassing local raw cache")
    parser.add_argument(
        "--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="Directory to store raw files"
    )
    args = parser.parse_args()

    loader = CelesTrakLoader(raw_dir=args.raw_dir)
    total_valid = 0

    logger.info("Starting CelesTrak data collection for groups: %s", args.groups)
    for group_key in args.groups:
        try:
            records, meta = loader.fetch_group(group_key=group_key, fmt=args.format, force_refresh=args.refresh)
            status = "ONLINE FETCH" if not meta.get("offline_mode") else "OFFLINE CACHE"
            logger.info(
                "[%-18s] Source: %-25s | Mode: %-13s | Valid Records: %d",
                group_key,
                meta.get("source", "N/A")[:25],
                status,
                len(records),
            )
            total_valid += len(records)
        except Exception as exc:
            logger.error("Failed processing group %s: %s", group_key, exc)

    logger.info("Orbital data collection complete. Total resident space objects available: %d", total_valid)


if __name__ == "__main__":
    main()
