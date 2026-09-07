from pathlib import Path

import h5py
import numpy as np


def load_array_map(path: Path) -> dict[str, np.ndarray]:
    """Read every dataset in an .h5 file into a flat name -> ndarray map."""
    with h5py.File(path, "r") as f:
        return {key: f[key][()] for key in f if isinstance(f[key], h5py.Dataset)}
