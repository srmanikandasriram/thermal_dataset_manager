import io
from pathlib import Path

import h5py
import numpy as np
import pytest
import zstandard
from sqlmodel import Session

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

FRAME_HEIGHT_TOTAL = 12  # 2 telemetry rows + 10 image rows
FRAME_WIDTH = 16
TELEMETRY_ROWS = 2


def _synthetic_frames(n_frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=(n_frames, FRAME_HEIGHT_TOTAL, FRAME_WIDTH), dtype=np.uint16)


def _synthetic_timestamps(n_frames: int, start: float = 1_700_000_000.0, fps: float = 30.0) -> np.ndarray:
    return start + np.arange(n_frames, dtype=np.float64) / fps


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    root.mkdir()
    return root


@pytest.fixture
def storage(data_root: Path) -> FilesystemDatasetStorage:
    return FilesystemDatasetStorage(data_root)


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    engine = make_engine(f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    init_db(engine)
    with get_session(engine) as session:
        yield session


def write_single_camera_npz(path: Path, n_frames: int = 20, seed: int = 0) -> None:
    frames = _synthetic_frames(n_frames, seed=seed)
    timestamps = _synthetic_timestamps(n_frames)
    np.savez(path, raw_thr_frames=frames, raw_thr_tstamps=timestamps)


def write_single_camera_npz_zst(path: Path, n_frames: int = 20) -> None:
    frames = _synthetic_frames(n_frames)
    timestamps = _synthetic_timestamps(n_frames)
    buf = io.BytesIO()
    np.savez(buf, raw_thr_frames=frames, raw_thr_tstamps=timestamps)
    buf.seek(0)
    compressor = zstandard.ZstdCompressor()
    with open(path, "wb") as fh:
        compressor.copy_stream(buf, fh)


def write_multi_camera_npz(path: Path, n_frames: int = 15, n_cameras: int = 2) -> None:
    arrays = {}
    for cam_idx in range(n_cameras):
        arrays[f"raw_thr_frames_{cam_idx}"] = _synthetic_frames(n_frames, seed=cam_idx)
        arrays[f"raw_thr_tstamps_{cam_idx}"] = _synthetic_timestamps(n_frames)
    np.savez(path, **arrays)


def write_single_camera_h5(path: Path, n_frames: int = 20) -> None:
    frames = _synthetic_frames(n_frames)
    timestamps = _synthetic_timestamps(n_frames)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw_thr_frames", data=frames)
        f.create_dataset("raw_thr_tstamps", data=timestamps)


@pytest.fixture
def sample_npz(tmp_path: Path) -> Path:
    path = tmp_path / "capture.npz"
    write_single_camera_npz(path)
    return path
