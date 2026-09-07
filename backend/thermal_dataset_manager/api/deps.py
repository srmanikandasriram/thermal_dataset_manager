from collections.abc import Generator
from functools import lru_cache

from sqlmodel import Session

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import Settings, get_settings
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


@lru_cache
def get_engine():
    settings = get_settings()
    engine = make_engine(settings.catalog_db)
    init_db(engine)
    return engine


def get_db_session() -> Generator[Session, None, None]:
    with get_session(get_engine()) as session:
        yield session


def get_storage() -> FilesystemDatasetStorage:
    settings = get_settings()
    return FilesystemDatasetStorage(settings.data_root)


def get_app_settings() -> Settings:
    return get_settings()
