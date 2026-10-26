from datetime import datetime

from sqlmodel import Field, SQLModel


class DatasetRecord(SQLModel, table=True):
    """Derived, rebuildable projection of a dataset's metadata.json for search/listing.

    Never a source of truth — see catalog.rebuild.rebuild_catalog().
    """

    dataset_id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    notes: str = ""
    ingested_at: datetime = Field(index=True)
    source_filename: str = ""
    source_sha256: str = Field(default="", index=True)
    camera_index: int | None = None
    capture_id: str | None = Field(default=None, index=True)
    camera_count: int = 1
    total_frame_count: int = 0
    max_duration_seconds: float | None = None
    capture_location: str | None = None
    scene_category: str | None = None


class Tag(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)


class DatasetTag(SQLModel, table=True):
    dataset_id: str = Field(foreign_key="datasetrecord.dataset_id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)
