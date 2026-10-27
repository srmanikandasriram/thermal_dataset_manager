#!/usr/bin/env python
"""Compute the frame-content hash (used for content-based duplicate detection
at ingest) for datasets ingested before it existed, and report groups of
datasets that turn out to hold identical frames.

Reads every un-hashed dataset's zarr store once; safe to re-run (datasets that
already have a hash are skipped).

Usage:
    python backend/scripts/backfill_content_hash_cli.py
"""

from collections import defaultdict

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.catalog.repository import upsert_from_metadata
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.ingest.content_hash import stored_camera_content_sha256
from thermal_dataset_manager.storage import zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


def main() -> int:
    settings = get_settings()
    storage = FilesystemDatasetStorage(settings.data_root)
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    by_hash: dict[str, list[str]] = defaultdict(list)
    done = failed = 0
    with get_session(engine) as session:
        for dataset_id in storage.list_dataset_ids():
            metadata = storage.read_metadata(dataset_id)
            try:
                root = zarr_store.open_group(storage.zarr_store_path(dataset_id))
                changed = False
                for cam in metadata.auto.cameras:
                    if cam.content_sha256 is None:
                        group = root[f"camera_{cam.camera_index}"]
                        cam.content_sha256 = stored_camera_content_sha256(
                            group["counts"],
                            group["telemetry"] if "telemetry" in group.array_keys() else None,
                            cam.telemetry_row_count,
                        )
                        changed = True
                    by_hash[cam.content_sha256].append(f"{dataset_id} ({metadata.user.name})")
                if changed:
                    storage.write_metadata(dataset_id, metadata, locked=True)
                    upsert_from_metadata(session, metadata)
                    session.commit()
                    done += 1
                    print(f"OK      {dataset_id} {metadata.user.name}", flush=True)
            except Exception as e:  # noqa: BLE001 - one bad dataset shouldn't abort the run
                print(f"FAILED  {dataset_id}: {type(e).__name__}: {e}")
                failed += 1

    groups = [g for g in by_hash.values() if len(g) > 1]
    print(f"\nhashed {done} dataset(s), {failed} failed; {len(groups)} group(s) of identical frames")
    for g in groups:
        print("DUPLICATES:")
        for item in g:
            print(f"    {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
