"""Lazy, header-first access to the arrays in a source capture file.

Opening a source reports every array's shape and dtype without reading any
array data; `read` then loads one array at a time. The pipeline uses this to
decide which arrays are worth loading (camera frames and timestamps) before
touching the (potentially huge) rest of the file.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: np.dtype


class ArraySource(Protocol):
    infos: dict[str, ArrayInfo]

    def read(self, key: str) -> np.ndarray: ...

    def close(self) -> None: ...

    def __enter__(self) -> "ArraySource": ...

    def __exit__(self, *exc: object) -> None: ...
