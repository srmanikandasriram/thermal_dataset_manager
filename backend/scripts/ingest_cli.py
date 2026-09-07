#!/usr/bin/env python
"""CLI wrapper around the ingest pipeline, usable before any HTTP API exists.

Usage:
    python backend/scripts/ingest_cli.py <path-to-capture-file> [<more files>...]
    python backend/scripts/ingest_cli.py --telemetry-rows 0 no_telemetry_capture.npz
"""

import argparse
import sys
from pathlib import Path

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.ingest.pipeline import DuplicateDatasetError, ingest_dataset
from thermal_dataset_manager.ingest.validation import IngestValidationError
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one or more raw thermal capture files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Source .npz/.npz.zst/.h5 files")
    parser.add_argument(
        "--telemetry-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Telemetry rows per frame, applied to every camera in every file. "
            "2 (default) = top 2 rows are telemetry; 0 = no telemetry rows; "
            "-2 = bottom 2 rows are telemetry. "
            "Overrides THERMAL_DEFAULT_TELEMETRY_ROW_COUNT for this run."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    telemetry_row_count = (
        args.telemetry_rows if args.telemetry_rows is not None else settings.default_telemetry_row_count
    )
    storage = FilesystemDatasetStorage(settings.data_root)
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    exit_code = 0
    with get_session(engine) as session:
        for path in args.paths:
            try:
                metadata = ingest_dataset(
                    path,
                    storage,
                    session,
                    default_telemetry_row_count=telemetry_row_count,
                    zarr_target_chunk_bytes=settings.zarr_target_chunk_bytes,
                    stats_sample_frames=settings.stats_sample_frames,
                )
            except IngestValidationError as e:
                print(f"FAILED  {path}:", file=sys.stderr)
                for err in e.errors:
                    print(f"    - {err}", file=sys.stderr)
                exit_code = 1
                continue
            except DuplicateDatasetError as e:
                print(f"DUPLICATE  {path}:", file=sys.stderr)
                print(f"    already ingested as dataset {e.existing_dataset_id}", file=sys.stderr)
                print(f"    name:     {e.existing_name}", file=sys.stderr)
                print(f"    source:   {e.existing_source_filename}", file=sys.stderr)
                print(f"    ingested: {e.existing_ingested_at.isoformat()}", file=sys.stderr)
                print(f"    sha256:   {e.source_sha256}", file=sys.stderr)
                exit_code = 1
                continue
            print(f"OK      {path} -> dataset {metadata.auto.dataset_id} ({metadata.auto.camera_count} camera(s))")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
