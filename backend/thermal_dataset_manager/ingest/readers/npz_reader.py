import struct
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import zstandard

from thermal_dataset_manager.ingest.readers.source import ArrayInfo

ZST_MAGIC = b"\x28\xb5\x2f\xfd"

_NPY_SUFFIX = ".npy"

# Arrays at least this big are memory-mapped straight out of an uncompressed
# archive instead of being read into memory, so a capture larger than RAM can
# still be ingested (everything downstream works in slices).
_MEMMAP_MIN_BYTES = 256 * 1024 * 1024
_LOCAL_HEADER_SIZE = 30


def _read_npy_header(fp: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    """(shape, fortran_order, dtype); leaves `fp` at the start of the array data."""
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError(f"Unsupported .npy format version {version}")
    return tuple(shape), fortran, dtype


def _read_header(fp: BinaryIO) -> ArrayInfo:
    shape, _fortran, dtype = _read_npy_header(fp)
    return ArrayInfo(shape=shape, dtype=dtype)


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
        memmapped = self._memmap(key)
        if memmapped is not None:
            return memmapped
        with self._zip.open(self._members[key]) as fp:
            return np.lib.format.read_array(fp, allow_pickle=False)

    def _memmap(self, key: str) -> np.ndarray | None:
        """A read-only memory map of a large array stored uncompressed in the
        archive, or None when it is small, compressed or Fortran-ordered (those
        are read normally)."""
        info = self._zip.getinfo(self._members[key])
        header = self.infos[key]
        n_bytes = int(np.prod(header.shape, dtype=np.int64)) * header.dtype.itemsize
        if info.compress_type != zipfile.ZIP_STORED or n_bytes < _MEMMAP_MIN_BYTES:
            return None
        # The member's data starts after its local header, whose name/extra
        # lengths can differ from the central directory's.
        self._file.seek(info.header_offset)
        local = self._file.read(_LOCAL_HEADER_SIZE)
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        self._file.seek(info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len)
        shape, fortran, dtype = _read_npy_header(self._file)
        if fortran:
            return None
        return np.memmap(self._file, dtype=dtype, mode="r", offset=self._file.tell(), shape=shape)

    def close(self) -> None:
        self._zip.close()
        self._file.close()

    def __enter__(self) -> "NpzSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
