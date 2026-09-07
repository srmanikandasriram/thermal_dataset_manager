"""Local-filesystem implementation of DatasetStorage.

Immutability model: after ingest, a dataset's raw.zarr/ store and
manifest.json are chmod'd read-only. metadata.json stays writable through
the app (chmod-writable -> atomic write -> chmod-readonly) so the metadata
PATCH endpoint can update the user-editable half; the pipeline is the only
code that ever writes the zarr store or manifest.
"""

import json
import os
import shutil
import stat
import time
from pathlib import Path

from thermal_dataset_manager.models.metadata import DatasetMetadata
from thermal_dataset_manager.storage import layout

_FILE_RO = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_FILE_RW = _FILE_RO | stat.S_IWUSR
_DIR_RO = _FILE_RO | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
_DIR_RW = _DIR_RO | stat.S_IWUSR


def _chmod_tree(path: Path, *, writable: bool) -> None:
    file_mode = _FILE_RW if writable else _FILE_RO
    dir_mode = _DIR_RW if writable else _DIR_RO
    if path.is_dir():
        os.chmod(path, dir_mode)
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chmod(Path(root) / d, dir_mode)
            for f in files:
                os.chmod(Path(root) / f, file_mode)
    elif path.exists():
        os.chmod(path, file_mode)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str))
    tmp_path.replace(path)


class FilesystemDatasetStorage:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    def dataset_dir(self, dataset_id: str) -> Path:
        return layout.dataset_dir(self.data_root, dataset_id)

    def zarr_store_path(self, dataset_id: str) -> Path:
        return layout.zarr_store_path(self.data_root, dataset_id)

    def list_dataset_ids(self) -> list[str]:
        datasets_dir = self.data_root / "datasets"
        if not datasets_dir.exists():
            return []
        return sorted(p.name for p in datasets_dir.iterdir() if p.is_dir())

    def dataset_exists(self, dataset_id: str) -> bool:
        return layout.metadata_path(self.data_root, dataset_id).exists()

    def read_metadata(self, dataset_id: str) -> DatasetMetadata:
        path = layout.metadata_path(self.data_root, dataset_id)
        return DatasetMetadata.model_validate_json(path.read_text())

    def write_metadata(self, dataset_id: str, metadata: DatasetMetadata, *, locked: bool) -> None:
        path = layout.metadata_path(self.data_root, dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if locked and path.exists():
            os.chmod(path, _FILE_RW)
        _atomic_write_json(path, json.loads(metadata.model_dump_json()))
        if locked:
            os.chmod(path, _FILE_RO)

    def write_manifest(self, dataset_id: str, manifest: dict) -> None:
        path = layout.manifest_path(self.data_root, dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, manifest)

    def lock_dataset(self, dataset_id: str) -> None:
        zarr_path = self.zarr_store_path(dataset_id)
        manifest = layout.manifest_path(self.data_root, dataset_id)
        metadata = layout.metadata_path(self.data_root, dataset_id)
        _chmod_tree(zarr_path, writable=False)
        _chmod_tree(manifest, writable=False)
        _chmod_tree(metadata, writable=False)

    def new_staging_dir(self, upload_token: str) -> Path:
        path = layout.staging_dir(self.data_root, upload_token)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove_staging_dir(self, upload_token: str) -> None:
        path = layout.staging_dir(self.data_root, upload_token)
        if path.exists():
            shutil.rmtree(path)

    def sweep_stale_staging(self, max_age_seconds: float) -> int:
        """Delete staged-upload directories older than max_age_seconds.

        Covers uploads that never reached /confirm: interrupted transfers
        (the write loop's own error handling should catch most of these,
        but not a server crash mid-write) and abandoned reviews where the
        user never clicked "Ingest". Safe to call repeatedly; returns the
        number of directories removed.
        """
        staging_root = self.data_root / "staging"
        if not staging_root.exists():
            return 0
        now = time.time()
        removed = 0
        for entry in staging_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                age = now - entry.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > max_age_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed
