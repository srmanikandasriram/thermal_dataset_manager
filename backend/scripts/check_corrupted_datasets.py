#!/usr/bin/env python
"""Scan datasets_dir for orphaned or corrupted dataset directories.

A dataset directory is well-formed when it has a metadata.json (ingest
completed -- see FilesystemDatasetStorage.dataset_exists) and its raw.zarr
store opens cleanly with the expected camera_<n>/{counts,timestamps} arrays.

Flags:
  ORPHANED      directory exists under datasets/ but has no metadata.json --
                either an interrupted ingest (crashed before the pipeline
                finished) or a stale/abandoned write.
  INACCESSIBLE  a path under the dataset dir raised PermissionError/OSError
                while being read -- typically a partial write left by a
                different OS user on shared/network storage (e.g. sshfs
                where the remote uid doesn't match the local session).
  CORRUPTED     metadata.json exists but the zarr store fails to open, is
                missing a camera group, or is missing counts/timestamps.
  DB_ORPHAN     dataset_id is in the catalog DB but has no directory on disk.
  OK            metadata.json present, zarr store opens, arrays present.

Read-only: this script never writes, deletes, or chmods anything.

Usage:
    python backend/scripts/check_corrupted_datasets.py
"""

import sys

from sqlmodel import select

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.models.catalog import DatasetRecord
from thermal_dataset_manager.storage import layout
from thermal_dataset_manager.storage.zarr_store import camera_indices, open_group


def check_dataset(data_root, dataset_id: str) -> tuple[str, str]:
    """Returns (status, detail)."""
    dataset_dir = layout.dataset_dir(data_root, dataset_id)
    metadata_path = layout.metadata_path(data_root, dataset_id)
    zarr_path = layout.zarr_store_path(data_root, dataset_id)

    try:
        has_metadata = metadata_path.exists()
    except OSError as exc:
        return "INACCESSIBLE", f"could not stat metadata.json: {exc}"

    if not has_metadata:
        try:
            any(dataset_dir.iterdir())
        except OSError as exc:
            return "INACCESSIBLE", f"could not list {dataset_dir}: {exc}"
        return "ORPHANED", "no metadata.json (ingest never completed)"

    try:
        cam_indices = camera_indices(zarr_path)
        if not cam_indices:
            return "CORRUPTED", "raw.zarr has no camera_<n> groups"
        root = open_group(zarr_path)
        for cam_idx in cam_indices:
            cam_group = root[f"camera_{cam_idx}"]
            if "counts" not in cam_group.array_keys():
                return "CORRUPTED", f"camera_{cam_idx} missing 'counts' array"
            if "timestamps" not in cam_group.array_keys():
                return "CORRUPTED", f"camera_{cam_idx} missing 'timestamps' array"
            n_frames = cam_group["counts"].shape[0]
            if "telemetry" in cam_group.array_keys() and cam_group["telemetry"].shape[0] != n_frames:
                return "CORRUPTED", f"camera_{cam_idx} counts/telemetry frame-count mismatch"
            if cam_group["timestamps"].shape[0] != n_frames:
                return (
                    "CORRUPTED",
                    f"camera_{cam_idx} counts/timestamps frame-count mismatch",
                )
    except PermissionError as exc:
        return "INACCESSIBLE", f"permission error reading raw.zarr: {exc}"
    except OSError as exc:
        return "INACCESSIBLE", f"OS error reading raw.zarr: {exc}"
    except Exception as exc:  # noqa: BLE001 -- zarr/numcodecs raise assorted errors on bad stores
        return "CORRUPTED", f"{type(exc).__name__}: {exc}"

    return "OK", ""


def main() -> int:
    settings = get_settings()
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    data_root = settings.data_root
    try:
        on_disk_ids = sorted(p.name for p in settings.datasets_dir.iterdir() if p.is_dir())
    except OSError as exc:
        print(f"FATAL: could not list {settings.datasets_dir}: {exc}", file=sys.stderr)
        return 2

    with get_session(engine) as session:
        db_ids = {r.dataset_id for r in session.exec(select(DatasetRecord)).all()}

        problems = 0
        for dataset_id in on_disk_ids:
            status, detail = check_dataset(data_root, dataset_id)
            if status != "OK":
                problems += 1
                print(f"{status:13} {dataset_id}  {detail}")
            db_ids.discard(dataset_id)

        for dataset_id in sorted(db_ids):
            problems += 1
            print(f"{'DB_ORPHAN':13} {dataset_id}  in catalog DB but no directory on disk")

    if problems == 0:
        print(f"OK: all {len(on_disk_ids)} dataset directories are well-formed.")
        return 0

    print(f"\n{problems} problem(s) found out of {len(on_disk_ids)} dataset directories.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
