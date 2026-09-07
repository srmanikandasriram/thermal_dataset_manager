import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from thermal_dataset_manager.api import admin, datasets, frames, ingest
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Backstop for orphaned staged uploads (interrupted transfers, or
        # reviews the user never confirmed) that survived a server restart.
        storage = FilesystemDatasetStorage(settings.data_root)
        removed = storage.sweep_stale_staging(settings.staging_max_age_seconds)
        if removed:
            logger.info("Swept %d stale staged upload(s) on startup", removed)
        yield

    app = FastAPI(title="Thermal Dataset Manager", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets.router)
    app.include_router(ingest.router)
    app.include_router(frames.router)
    app.include_router(admin.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
