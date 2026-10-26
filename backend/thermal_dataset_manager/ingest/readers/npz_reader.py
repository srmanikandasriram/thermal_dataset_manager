import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import zstandard

from thermal_dataset_manager.ingest.readers.source import ArrayInfo

ZST_MAGIC = b"\x28\xb5\x2f\xfd"

_NPY_SUFFIX = ".npy"


def _read_header(fp: BinaryIO) -> ArrayInfo:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError(f"Unsupported .npy format version {version}")
    return ArrayInfo(shape=tuple(shape), dtype=dtype)


class NpzSource:
    """An .npz (or zstd-compressed .npz) opened lazily: array headers are read
    up front, array data only on `read`. A .npz.zst is decompressed to an
    anonymous temporary file rather than into memory, since a zip needs a
    seekable file.
    """

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as fh:
            is_zst = fh.read(4) == ZST_MAGIC
        if is_zst:
            self._file: BinaryIO = tempfile.TemporaryFile()
            with open(path, "rb") as src:
                zstandard.ZstdDecompressor().copy_stream(src, self._file)
            self._file.seek(0)
        else:
            self._file = open(path, "rb")
        try:
            self._zip = zipfile.ZipFile(self._file)
            self._members = {
                name[: -len(_NPY_SUFFIX)]: name for name in self._zip.namelist() if name.endswith(_NPY_SUFFIX)
            }
            self.infos: dict[str, ArrayInfo] = {}
            for key, member in self._members.items():
                with self._zip.open(member) as fp:
                    self.infos[key] = _read_header(fp)
        except Exception:
            self._file.close()
            raise

    def read(self, key: str) -> np.ndarray:
        with self._zip.open(self._members[key]) as fp:
            return np.lib.format.read_array(fp, allow_pickle=False)

    def close(self) -> None:
        self._zip.close()
        self._file.close()

    def __enter__(self) -> "NpzSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
