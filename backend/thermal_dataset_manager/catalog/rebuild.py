"""Rebuild the SQLite catalog from the per-dataset metadata.json files on
disk. The catalog is a derived index, never a source of truth -- if it's
deleted, corrupted, or out of sync, this restores it fully.
"""

from sqlalchemy import text
from sqlmodel import Session

from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.storage.protocol import DatasetStorage


def rebuild_catalog(storage: DatasetStorage, session: Session) -> int:
    session.exec(text("DELETE FROM datasettag"))
    session.exec(text("DELETE FROM tag"))
    session.exec(text("DELETE FROM datasetrecord"))
    session.exec(text("DELETE FROM dataset_fts"))
    session.commit()

    count = 0
    for dataset_id in storage.list_dataset_ids():
        try:
            metadata = storage.read_metadata(dataset_id)
        except (FileNotFoundError, ValueError):
            continue
        repository.upsert_from_metadata(session, metadata)
        count += 1
    return count
