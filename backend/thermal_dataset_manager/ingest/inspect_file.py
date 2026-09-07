"""Inspect a staged source file's raw keys before committing to ingest.

Lets the caller (the API/UI) show the user what's actually in the file and
either accept an auto-detected camera mapping (via the key registry) or
build one manually -- useful when a capture uses a naming convention the
registry doesn't recognize yet.
"""

from dataclasses import dataclass
from pathlib import Path

from thermal_dataset_manager.ingest.key_registry import CameraKeyPair, KeyRegistryError, discover_camera_keys
from thermal_dataset_manager.ingest.readers.common import load_array_map
from thermal_dataset_manager.models.metadata import SourceFormat


@dataclass
class KeyInfo:
    key: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class FileInspection:
    keys: list[KeyInfo]
    auto_pairs: list[CameraKeyPair] | None
    auto_mapping_error: str | None


def inspect_source_file(path: Path, source_format: SourceFormat) -> FileInspection:
    array_map = load_array_map(path, source_format)
    keys = [KeyInfo(key=k, shape=tuple(v.shape), dtype=str(v.dtype)) for k, v in array_map.items()]

    try:
        auto_pairs = discover_camera_keys(list(array_map.keys()))
        auto_mapping_error = None
    except KeyRegistryError as e:
        auto_pairs = None
        auto_mapping_error = str(e)

    return FileInspection(keys=keys, auto_pairs=auto_pairs, auto_mapping_error=auto_mapping_error)
