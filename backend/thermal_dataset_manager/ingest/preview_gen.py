from pathlib import Path

from thermal_dataset_manager.colormap import apply_colormap
from thermal_dataset_manager.models.metadata import CameraStreamMetadata
from thermal_dataset_manager.storage import layout, zarr_store


def generate_thumbnail(
    data_root: Path,
    dataset_id: str,
    store_path: Path,
    camera: CameraStreamMetadata,
) -> None:
    """Render a representative frame (~50% through the capture) as a
    colormapped PNG thumbnail for the browse grid.
    """
    if camera.frame_count == 0:
        return
    frame_index = camera.frame_count // 2
    image = zarr_store.read_frame_image(store_path, camera.camera_index, frame_index)
    png = apply_colormap(image)
    out_path = layout.thumbnail_path(data_root, dataset_id, camera.camera_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png.save(out_path)
