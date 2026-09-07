import numpy as np

from thermal_dataset_manager.colormap import apply_clahe, apply_colormap


def test_apply_colormap_output_shape_and_mode():
    image = np.linspace(0, 1000, 12 * 16, dtype=np.uint16).reshape(12, 16)
    png = apply_colormap(image)
    assert png.mode == "RGB"
    assert png.size == (16, 12)


def test_apply_colormap_handles_degenerate_range():
    image = np.full((4, 4), 500, dtype=np.uint16)
    png = apply_colormap(image)
    assert png.size == (4, 4)


def test_apply_colormap_is_grayscale_by_default():
    image = np.linspace(0, 60000, 32 * 32, dtype=np.uint16).reshape(32, 32)
    png = apply_colormap(image)
    arr = np.array(png)
    assert (arr[..., 0] == arr[..., 1]).all()
    assert (arr[..., 1] == arr[..., 2]).all()


def test_apply_clahe_output_range_and_shape():
    image = np.linspace(0, 60000, 32 * 32, dtype=np.uint16).reshape(32, 32)
    equalized = apply_clahe(image)
    assert equalized.shape == image.shape
    assert equalized.min() >= 0.0
    assert equalized.max() <= 1.0


def test_apply_clahe_enhances_low_contrast_region():
    """CLAHE should pull out local structure a flat linear stretch would
    flatten -- e.g. a subtle gradient sitting on top of a much brighter
    uniform region.
    """
    image = np.full((64, 64), 40000, dtype=np.uint16)
    image[:32, :32] = np.linspace(100, 200, 32 * 32, dtype=np.uint16).reshape(32, 32)

    equalized = apply_clahe(image)
    low_contrast_region = equalized[:32, :32]
    # After CLAHE, the subtle gradient should span a meaningfully wider
    # range than its tiny slice (100-200 out of 0-65535) of the raw data.
    # A flat linear stretch (or CLAHE at the old default nbins=256) leaves
    # this at 0.0 -- any real separation confirms local detail survived.
    assert low_contrast_region.max() - low_contrast_region.min() > 0.05
