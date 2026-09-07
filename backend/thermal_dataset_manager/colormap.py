"""Shared colormap application, used for both ingest-time thumbnails and
on-demand frame rendering, so scrubbing and thumbnails always agree.

Preview normalization uses CLAHE (contrast-limited adaptive histogram
equalization) rather than a flat linear min/max stretch -- it brings out
local detail in thermal frames much better than a single global clip.
"""

from functools import lru_cache

import matplotlib
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist

DEFAULT_CMAP = "gray"

# skimage default; higher allows more contrast (and noise) within each tile.
_CLAHE_CLIP_LIMIT = 0.01

# skimage's default nbins=256 quantizes the whole stretched image before
# per-tile equalization runs -- fine for 8-bit photos, but it crushes local
# detail in raw thermal data where a small hot/cold outlier region can make
# the rest of the scene's variation collapse into a handful of bins before
# CLAHE even gets a chance to redistribute it. Cost scales close to linearly
# with nbins (full uint16 precision, 65536, measured ~170ms/frame vs ~50ms
# here on a 512x640 frame) with no quality plateau, so this is a deliberate
# speed/quality middle ground rather than "as much precision as possible."
_CLAHE_NBINS = 8192


@lru_cache(maxsize=8)
def _lut(cmap_name: str) -> np.ndarray:
    cmap = matplotlib.colormaps[cmap_name]
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype("uint8")


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization. Returns a float64
    array in [0, 1].

    The image is first stretched to its own min/max before equalizing --
    equalize_adapthist treats its input as already spanning [0, 1] (via
    img_as_float's dtype-range assumption), and raw sensor counts rarely
    fill the full uint16 range, so skipping this would compress contrast.
    """
    img_min, img_max = float(image.min()), float(image.max())
    span = max(img_max - img_min, 1e-6)
    stretched = (image.astype(np.float64) - img_min) / span
    return equalize_adapthist(stretched, clip_limit=_CLAHE_CLIP_LIMIT, nbins=_CLAHE_NBINS)


def apply_colormap(image: np.ndarray, cmap_name: str = DEFAULT_CMAP) -> Image.Image:
    lut = _lut(cmap_name)
    equalized = apply_clahe(image)
    indices = np.clip(equalized * 255.0, 0, 255).astype("uint8")
    rgb = lut[indices]
    return Image.fromarray(rgb, mode="RGB")
