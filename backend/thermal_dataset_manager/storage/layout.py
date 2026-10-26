"""Path-building helpers for the on-disk dataset layout.

    <data_root>/
        datasets/<dataset_id>/
            raw.zarr/
            metadata.json
            manifest.json
            previews/camera_<n>/thumbnail.png
            previews/camera_<n>/frames/<index>.png
"""

from pathlib import Path


def dataset_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "datasets" / dataset_id


def zarr_store_path(data_root: Path, dataset_id: str) -> Path:
    return dataset_dir(data_root, dataset_id) / "raw.zarr"


def metadata_path(data_root: Path, dataset_id: str) -> Path:
    return dataset_dir(data_root, dataset_id) / "metadata.json"


def manifest_path(data_root: Path, dataset_id: str) -> Path:
    return dataset_dir(data_root, dataset_id) / "manifest.json"


def previews_dir(data_root: Path, dataset_id: str, camera_index: int) -> Path:
    return dataset_dir(data_root, dataset_id) / "previews" / f"camera_{camera_index}"


def thumbnail_path(data_root: Path, dataset_id: str, camera_index: int) -> Path:
    return previews_dir(data_root, dataset_id, camera_index) / "thumbnail.png"


def frame_preview_path(data_root: Path, dataset_id: str, camera_index: int, frame_index: int) -> Path:
    return previews_dir(data_root, dataset_id, camera_index) / "frames" / f"{frame_index:06d}.png"
