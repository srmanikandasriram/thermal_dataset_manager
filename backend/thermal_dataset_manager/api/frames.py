import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from thermal_dataset_manager.api.deps import get_storage
from thermal_dataset_manager.colormap import apply_colormap
from thermal_dataset_manager.ingest.preview_gen import generate_thumbnail
from thermal_dataset_manager.storage import layout, zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

router = APIRouter(prefix="/api/datasets", tags=["frames"])


def _camera_metadata_or_404(storage: FilesystemDatasetStorage, dataset_id: str, camera_index: int):
    if not storage.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    metadata = storage.read_metadata(dataset_id)
    for cam in metadata.auto.cameras:
        if cam.camera_index == camera_index:
            return cam
    raise HTTPException(status_code=404, detail="Camera not found")


@router.get("/{dataset_id}/cameras/{camera_index}/thumbnail.png")
def get_thumbnail(
    dataset_id: str,
    camera_index: int,
    storage: FilesystemDatasetStorage = Depends(get_storage),
) -> FileResponse:
    cam = _camera_metadata_or_404(storage, dataset_id, camera_index)
    path = layout.thumbnail_path(storage.data_root, dataset_id, camera_index)
    if not path.exists():
        # Previews are a derived cache -- regenerate on demand if missing
        # (e.g. deleted to pick up a colormap change), same as frames below.
        store_path = storage.zarr_store_path(dataset_id)
        generate_thumbnail(storage.data_root, dataset_id, store_path, cam)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(path, media_type="image/png")


@router.get("/{dataset_id}/cameras/{camera_index}/frames/{frame_index}.png")
def get_frame(
    dataset_id: str,
    camera_index: int,
    frame_index: int,
    storage: FilesystemDatasetStorage = Depends(get_storage),
) -> Response:
    cam = _camera_metadata_or_404(storage, dataset_id, camera_index)
    if frame_index < 0 or frame_index >= cam.frame_count:
        raise HTTPException(status_code=404, detail="Frame index out of range")

    cache_path = layout.frame_preview_path(storage.data_root, dataset_id, camera_index, frame_index)
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")

    store_path = storage.zarr_store_path(dataset_id)
    image = zarr_store.read_frame_image(store_path, camera_index, frame_index)
    png = apply_colormap(image)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    png.save(cache_path)

    buf = io.BytesIO()
    png.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
