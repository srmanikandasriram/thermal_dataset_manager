from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="THERMAL_", env_file=".env", extra="ignore")

    data_root: Path = Path("./datasets")
    # None means "derive from data_root" (see _default_catalog_db below) --
    # NOT a fixed CWD-relative path. THERMAL_CATALOG_DB set explicitly
    # always wins; only unset THERMAL_CATALOG_DB derives it from
    # data_root, so the two can never silently point at different places
    # just because the server was started from a different directory.
    catalog_db: str | None = None
    cors_origins: list[str] = ["http://localhost:5173"]

    # Target uncompressed bytes per zarr chunk along the time axis.
    zarr_target_chunk_bytes: int = 24 * 1024 * 1024

    # Default telemetry-row count per frame when the ingest request doesn't
    # set one per camera. Signed: 2 = top 2 rows are telemetry, 0 = none,
    # -2 = bottom 2 rows are telemetry.
    default_telemetry_row_count: int = 2

    # Number of frames sampled (evenly spaced) when computing count
    # statistics at ingest time, to keep ingest fast for long captures.
    stats_sample_frames: int = 64

    # Staged uploads (post-upload, pre-confirm) older than this are swept up
    # as orphaned -- covers interrupted transfers and abandoned reviews that
    # never reach /confirm.
    staging_max_age_hours: float = 24.0

    @property
    def staging_max_age_seconds(self) -> float:
        return self.staging_max_age_hours * 3600.0

    @property
    def staging_dir(self) -> Path:
        return self.data_root / "staging"

    @property
    def datasets_dir(self) -> Path:
        return self.data_root / "datasets"

    @model_validator(mode="after")
    def _default_catalog_db(self) -> "Settings":
        if self.catalog_db is None:
            self.catalog_db = f"sqlite:///{self.data_root / 'catalog.sqlite'}"
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    return settings
