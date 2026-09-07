#!/usr/bin/env python
"""Recompute auto-metadata (stats, thumbnails, extra-array inventory...)
for already-ingested datasets directly from their zarr stores -- no need to
re-upload/re-ingest just to backfill a newly added computed field.

Usage:
    python backend/scripts/recompute_metadata_cli.py                  # all datasets
    python backend/scripts/recompute_metadata_cli.py <dataset_id> ...  # specific ones
"""

import argparse
import sys

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.ingest.recompute import DatasetNotFoundError, recompute_all, recompute_metadata
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute auto-metadata for ingested datasets.")
    parser.add_argument(
        "dataset_ids", nargs="*", help="Specific dataset IDs to recompute (default: every dataset)"
    )
    args = parser.parse_args()

    settings = get_settings()
    storage = FilesystemDatasetStorage(settings.data_root)
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    exit_code = 0
    with get_session(engine) as session:
        if args.dataset_ids:
            for dataset_id in args.dataset_ids:
                try:
                    recompute_metadata(storage, dataset_id, session, stats_sample_frames=settings.stats_sample_frames)
                    print(f"OK      {dataset_id}")
                except DatasetNotFoundError:
                    print(f"FAILED  {dataset_id}: dataset not found", file=sys.stderr)
                    exit_code = 1
        else:
            succeeded, errors = recompute_all(storage, session, stats_sample_frames=settings.stats_sample_frames)
            print(f"OK      {succeeded} dataset(s) recomputed")
            for dataset_id, error in errors.items():
                print(f"FAILED  {dataset_id}: {error}", file=sys.stderr)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
