from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from thermal_dataset_manager.api.deps import get_app_settings, get_db_session, get_storage
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.ingest.recompute import DatasetNotFoundError, recompute_metadata
from thermal_dataset_manager.models.metadata import DatasetMetadata, UserMetadataUpdate
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


SortField = Literal[
    "name",
    "description",
    "ingested_at",
    "source_filename",
    "capture_id",
    "camera_count",
    "total_frame_count",
    "max_duration_seconds",
    "scene_category",
    "capture_location",
]


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    description: str
    tags: list[str]
    ingested_at: datetime
    source_filename: str
    capture_id: str | None
    camera_count: int
    total_frame_count: int
    max_duration_seconds: float | None
    scene_category: str | None
    capture_location: str | None
    thumbnail_url: str


class DatasetListResponse(BaseModel):
    items: list[DatasetSummary]
    limit: int
    offset: int
    total: int  # datasets matching the query, ignoring limit/offset


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    q: str | None = None,
    tags: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    capture_id: str | None = None,
    sort: SortField = "ingested_at",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
) -> DatasetListResponse:
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    records = repository.search(
        session,
        q=q,
        tags=tag_list,
        date_from=date_from,
        date_to=date_to,
        capture_id=capture_id,
        sort_by=sort,
        descending=order == "desc",
        limit=limit,
        offset=offset,
    )
    total = repository.count(session, q=q, tags=tag_list, date_from=date_from, date_to=date_to, capture_id=capture_id)
    items = [
        DatasetSummary(
            dataset_id=r.dataset_id,
            name=r.name,
            description=r.description,
            tags=repository.tags_for_dataset(session, r.dataset_id),
            ingested_at=r.ingested_at,
            source_filename=r.source_filename,
            capture_id=r.capture_id,
            camera_count=r.camera_count,
            total_frame_count=r.total_frame_count,
            max_duration_seconds=r.max_duration_seconds,
            scene_category=r.scene_category,
            capture_location=r.capture_location,
            thumbnail_url=f"/api/datasets/{r.dataset_id}/cameras/{r.camera_index or 0}/thumbnail.png",
        )
        for r in records
    ]
    return DatasetListResponse(items=items, limit=limit, offset=offset, total=total)


@router.get("/{dataset_id}", response_model=DatasetMetadata)
def get_dataset(
    dataset_id: str,
    storage: FilesystemDatasetStorage = Depends(get_storage),
) -> DatasetMetadata:
    if not storage.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return storage.read_metadata(dataset_id)


@router.patch("/{dataset_id}/metadata", response_model=DatasetMetadata)
def update_dataset_metadata(
    dataset_id: str,
    update: UserMetadataUpdate,
    storage: FilesystemDatasetStorage = Depends(get_storage),
    session: Session = Depends(get_db_session),
) -> DatasetMetadata:
    if not storage.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    metadata = storage.read_metadata(dataset_id)
    update_fields = update.model_dump(exclude_unset=True)
    metadata.user = metadata.user.model_copy(update=update_fields)

    storage.write_metadata(dataset_id, metadata, locked=True)
    repository.upsert_from_metadata(session, metadata)
    return metadata


@router.post("/{dataset_id}/recompute-metadata", response_model=DatasetMetadata)
def recompute_dataset_metadata(
    dataset_id: str,
    storage: FilesystemDatasetStorage = Depends(get_storage),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> DatasetMetadata:
    """Recompute this dataset's auto-metadata (stats, thumbnails...) from its already-ingested zarr store. Useful after adding
    new computed fields, or to retry if background computation after ingest
    failed -- never re-reads the original source file.
    """
    try:
        return recompute_metadata(storage, dataset_id, session, stats_sample_frames=settings.stats_sample_frames)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
