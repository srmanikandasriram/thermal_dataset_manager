from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from thermal_dataset_manager.api import admin, datasets, frames
from thermal_dataset_manager.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="Thermal Dataset Manager")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets.router)
    app.include_router(frames.router)
    app.include_router(admin.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
