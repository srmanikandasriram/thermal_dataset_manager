from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS dataset_fts USING fts5(
    dataset_id UNINDEXED,
    name,
    description,
    notes,
    tags_concat
);
"""


def make_engine(catalog_db_url: str):
    connect_args = {"check_same_thread": False} if catalog_db_url.startswith("sqlite") else {}
    return create_engine(catalog_db_url, connect_args=connect_args)


# Columns added to DatasetRecord after its first release. SQLModel's
# create_all() only creates missing *tables*, never missing columns, so an
# existing catalog.sqlite needs these added by hand. The catalog is a derived
# index (safe to delete and rebuild), so this is a convenience, not a real
# migration system -- non-SQLite backends would need their own ALTER.
_DATASETRECORD_ADDED_COLUMNS = {
    "source_sha256": "VARCHAR NOT NULL DEFAULT ''",
    "camera_index": "INTEGER",
    "capture_id": "VARCHAR",
    "content_sha256": "VARCHAR",
}


def _add_missing_columns(conn) -> None:
    existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(datasetrecord)")}
    for name, decl in _DATASETRECORD_ADDED_COLUMNS.items():
        if name not in existing:
            conn.exec_driver_sql(f"ALTER TABLE datasetrecord ADD COLUMN {name} {decl}")


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(_FTS_DDL))
        if engine.dialect.name == "sqlite":
            _add_missing_columns(conn)


def get_session(engine) -> Session:
    return Session(engine)
