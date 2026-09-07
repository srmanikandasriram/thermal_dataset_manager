from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from thermal_dataset_manager.api.deps import get_app_settings, get_db_session, get_storage
from thermal_dataset_manager.catalog.rebuild import rebuild_catalog
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.ingest.recompute import recompute_all
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

router = APIRouter(prefix="/api/admin", tags=["admin"])


class RebuildResponse(BaseModel):
    datasets_indexed: int


@router.post("/rebuild-catalog", response_model=RebuildResponse)
def rebuild_catalog_endpoint(
    storage: FilesystemDatasetStorage = Depends(get_storage),
    session: Session = Depends(get_db_session),
) -> RebuildResponse:
    count = rebuild_catalog(storage, session)
    return RebuildResponse(datasets_indexed=count)


class RecomputeAllResponse(BaseModel):
    succeeded: int
    errors: dict[str, str]


@router.post("/recompute-all-metadata", response_model=RecomputeAllResponse)
def recompute_all_metadata_endpoint(
    storage: FilesystemDatasetStorage = Depends(get_storage),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> RecomputeAllResponse:
    """Recompute auto-metadata for every dataset. Runs synchronously --
    intended for occasional use (e.g. after extending what gets computed),
    not a hot path; call per-dataset recompute for a single dataset instead.
    """
    succeeded, errors = recompute_all(storage, session, stats_sample_frames=settings.stats_sample_frames)
    return RecomputeAllResponse(succeeded=succeeded, errors=errors)
