import io
from pathlib import Path

import numpy as np
import zstandard

ZST_MAGIC = b"\x28\xb5\x2f\xfd"


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == ZST_MAGIC:
        decompressor = zstandard.ZstdDecompressor()
        buf = io.BytesIO()
        with open(path, "rb") as fh:
            decompressor.copy_stream(fh, buf)
        buf.seek(0)
        return np.load(buf)
    return np.load(path)


def load_array_map(path: Path) -> dict[str, np.ndarray]:
    """Read every array in an .npz/.npz.zst file into a flat name -> ndarray map."""
    npz = _load_npz(path)
    try:
        return {key: npz[key] for key in npz.files}
    finally:
        npz.close()
