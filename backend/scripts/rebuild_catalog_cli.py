#!/usr/bin/env python
"""Rebuild the SQLite search catalog from disk -- rescans every dataset's
metadata.json and repopulates the catalog tables from scratch. The catalog
is a derived index (never a source of truth), so this is always safe to
run, including as a fix if it ever ends up out of sync with what's
actually on disk (e.g. after a THERMAL_DATA_ROOT/THERMAL_CATALOG_DB
mismatch, or restoring from a backup).

Usage:
    python backend/scripts/rebuild_catalog_cli.py
"""

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.catalog.rebuild import rebuild_catalog
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


def main() -> int:
    settings = get_settings()
    storage = FilesystemDatasetStorage(settings.data_root)
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    with get_session(engine) as session:
        count = rebuild_catalog(storage, session)

    print(f"OK      {count} dataset(s) indexed from {settings.data_root}")
    print(f"        catalog: {settings.catalog_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
