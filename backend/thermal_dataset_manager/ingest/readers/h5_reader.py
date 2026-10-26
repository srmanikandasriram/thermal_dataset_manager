from pathlib import Path

import h5py
import numpy as np

from thermal_dataset_manager.ingest.readers.source import ArrayInfo


class H5Source:
    """An .h5 file opened lazily: top-level dataset shapes/dtypes are read up
    front, array data only on `read`."""

    def __init__(self, path: Path) -> None:
        self._file = h5py.File(path, "r")
        self.infos: dict[str, ArrayInfo] = {
            key: ArrayInfo(shape=tuple(ds.shape), dtype=ds.dtype)
            for key, ds in self._file.items()
            if isinstance(ds, h5py.Dataset)
        }

    def read(self, key: str) -> np.ndarray:
        return self._file[key][()]

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "H5Source":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
