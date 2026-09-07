from datetime import datetime

from sqlalchemy import func, text
from sqlmodel import Session, select

from thermal_dataset_manager.models.catalog import DatasetRecord, DatasetTag, Tag
from thermal_dataset_manager.models.metadata import DatasetMetadata


def _get_or_create_tag(session: Session, name: str) -> Tag:
    tag = session.exec(select(Tag).where(Tag.name == name)).first()
    if tag is None:
        tag = Tag(name=name)
        session.add(tag)
        session.flush()
    return tag


def upsert_from_metadata(session: Session, metadata: DatasetMetadata) -> None:
    dataset_id = metadata.auto.dataset_id
    name = metadata.user.name or metadata.auto.source_filename
    total_frame_count = sum(cam.frame_count for cam in metadata.auto.cameras)
    durations = [cam.duration_seconds for cam in metadata.auto.cameras if cam.duration_seconds is not None]

    record = session.get(DatasetRecord, dataset_id)
    if record is None:
        record = DatasetRecord(dataset_id=dataset_id, name=name, ingested_at=metadata.auto.ingested_at)
        session.add(record)

    record.name = name
    record.description = metadata.user.description
    record.notes = metadata.user.notes
    record.ingested_at = metadata.auto.ingested_at
    record.source_filename = metadata.auto.source_filename
    record.source_sha256 = metadata.auto.source_sha256
    record.camera_count = metadata.auto.camera_count
    record.total_frame_count = total_frame_count
    record.max_duration_seconds = max(durations) if durations else None
    record.capture_location = metadata.user.capture_location
    record.scene_category = metadata.user.scene_category

    # Re-derive tag associations from scratch.
    session.exec(
        text("DELETE FROM datasettag WHERE dataset_id = :dataset_id"),
        params={"dataset_id": dataset_id},
    )
    for tag_name in metadata.user.tags:
        tag = _get_or_create_tag(session, tag_name)
        session.add(DatasetTag(dataset_id=dataset_id, tag_id=tag.id))

    # Refresh the FTS shadow row.
    session.exec(text("DELETE FROM dataset_fts WHERE dataset_id = :dataset_id"), params={"dataset_id": dataset_id})
    session.exec(
        text(
            "INSERT INTO dataset_fts (dataset_id, name, description, notes, tags_concat) "
            "VALUES (:dataset_id, :name, :description, :notes, :tags_concat)"
        ),
        params={
            "dataset_id": dataset_id,
            "name": name,
            "description": metadata.user.description,
            "notes": metadata.user.notes,
            "tags_concat": " ".join(metadata.user.tags),
        },
    )
    session.commit()


def delete(session: Session, dataset_id: str) -> None:
    session.exec(text("DELETE FROM datasettag WHERE dataset_id = :dataset_id"), params={"dataset_id": dataset_id})
    session.exec(text("DELETE FROM dataset_fts WHERE dataset_id = :dataset_id"), params={"dataset_id": dataset_id})
    record = session.get(DatasetRecord, dataset_id)
    if record is not None:
        session.delete(record)
    session.commit()


def get(session: Session, dataset_id: str) -> DatasetRecord | None:
    return session.get(DatasetRecord, dataset_id)


def find_by_source_sha256(session: Session, source_sha256: str) -> DatasetRecord | None:
    """The existing dataset ingested from a byte-identical source file, if any.

    Used for duplicate detection at ingest time. Returns the oldest match if
    (somehow) more than one exists.
    """
    if not source_sha256:
        return None
    return session.exec(
        select(DatasetRecord)
        .where(DatasetRecord.source_sha256 == source_sha256)
        .order_by(DatasetRecord.ingested_at.asc())
    ).first()


def tags_for_dataset(session: Session, dataset_id: str) -> list[str]:
    rows = session.exec(
        select(Tag.name).join(DatasetTag, DatasetTag.tag_id == Tag.id).where(DatasetTag.dataset_id == dataset_id)
    ).all()
    return list(rows)


def _filtered_query(
    session: Session,
    *,
    q: str | None,
    tags: list[str] | None,
    date_from: datetime | None,
    date_to: datetime | None,
):
    """Build the `SELECT ... FROM datasetrecord WHERE <filters>` shared by
    `search` and `count` -- no ordering, offset or limit. Returns None when
    a filter matches nothing (so callers can short-circuit)."""
    candidate_ids: set[str] | None = None

    if q:
        rows = session.exec(
            text("SELECT dataset_id FROM dataset_fts WHERE dataset_fts MATCH :q"), params={"q": q}
        ).all()
        candidate_ids = {r[0] if isinstance(r, tuple) else r.dataset_id for r in rows}

    if tags:
        for tag_name in tags:
            ids_with_tag = set(
                session.exec(
                    select(DatasetTag.dataset_id)
                    .join(Tag, Tag.id == DatasetTag.tag_id)
                    .where(Tag.name == tag_name)
                ).all()
            )
            candidate_ids = ids_with_tag if candidate_ids is None else candidate_ids & ids_with_tag

    statement = select(DatasetRecord)
    if candidate_ids is not None:
        if not candidate_ids:
            return None
        statement = statement.where(DatasetRecord.dataset_id.in_(candidate_ids))
    if date_from is not None:
        statement = statement.where(DatasetRecord.ingested_at >= date_from)
    if date_to is not None:
        statement = statement.where(DatasetRecord.ingested_at <= date_to)
    return statement


def search(
    session: Session,
    *,
    q: str | None = None,
    tags: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DatasetRecord]:
    statement = _filtered_query(session, q=q, tags=tags, date_from=date_from, date_to=date_to)
    if statement is None:
        return []
    statement = statement.order_by(DatasetRecord.ingested_at.desc()).offset(offset).limit(limit)
    return list(session.exec(statement).all())


def count(
    session: Session,
    *,
    q: str | None = None,
    tags: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> int:
    """Total number of datasets matching the same filters `search` uses,
    ignoring limit/offset -- for paginated listing UIs."""
    statement = _filtered_query(session, q=q, tags=tags, date_from=date_from, date_to=date_to)
    if statement is None:
        return 0
    count_stmt = select(func.count()).select_from(statement.order_by(None).subquery())
    return int(session.exec(count_stmt).one())
