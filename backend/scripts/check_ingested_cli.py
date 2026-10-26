#!/usr/bin/env python
"""Check whether one or more source capture files have already been ingested.

Matches purely on the source file's sha256 (the same check the ingest pipeline
uses), so a file only counts as ingested if it is byte-identical to a previously
ingested one. Nothing is written -- this is a read-only lookup.

Usage:
    python backend/scripts/check_ingested_cli.py <path-to-capture-file> [<more files>...]

Exit code is 0 only if every given file is already ingested; 1 if any file is
new (or missing).
"""

import argparse
import sys
from pathlib import Path

from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.ingest import metadata_compute


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether raw thermal capture files have already been ingested."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Source .npz/.npz.zst/.h5 files")
    args = parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    all_ingested = True
    with get_session(engine) as session:
        for path in args.paths:
            if not path.is_file():
                print(f"MISSING    {path}", file=sys.stderr)
                all_ingested = False
                continue

            source_sha256 = metadata_compute.sha256_of_file(path)
            existing = repository.find_by_source_sha256(session, source_sha256)
            if existing is None:
                print(f"NEW        {path}")
                print(f"    sha256: {source_sha256}")
                all_ingested = False
                continue

            print(f"INGESTED   {path} -> dataset {existing.dataset_id}")
            print(f"    name:     {existing.name}")
            print(f"    source:   {existing.source_filename}")
            print(f"    ingested: {existing.ingested_at.isoformat()}")
            print(f"    sha256:   {source_sha256}")

    return 0 if all_ingested else 1


if __name__ == "__main__":
    raise SystemExit(main())
