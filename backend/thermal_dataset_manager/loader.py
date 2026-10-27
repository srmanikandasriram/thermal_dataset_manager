"""Programmatic access to the dataset library, for downstream processing.

Works straight off the data root on disk (no server needed): the SQLite
catalog does the searching, the zarr stores are opened lazily, and nothing is
read until you ask for frames.

    from thermal_dataset_manager.loader import Library

    lib = Library()                                   # uses THERMAL_DATA_ROOT
    videos = lib.find(tags=["bunny"], min_frames=500, pipeline_disabled=["temporal_filter"])
    for v in videos:
        print(v.name, v.n_frames, v.shape, v.fps)

    v = videos[0]
    clip = v.frames(slice(100, 200))                  # (100, H, W) uint16, image rows only
    clip = v.frames(time=(2.0, 4.0))                  # by seconds since the first frame
    pick = v.frames([0, 50, 99])                      # arbitrary frame indices
    raw = v.frames(slice(0, 10), raw=True)            # telemetry rows re-attached
    for batch in v.iter_batches(32): ...              # bounded memory, whole video
    tele = v.parse_telemetry(0)                       # decoded Boson+ telemetry of frame 0

    siblings = lib.capture(v.capture_id)              # all cameras of one capture

A `Video` is one camera stream (one dataset).
"""

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.catalog.db import get_session, make_engine
from thermal_dataset_manager.config import Settings, get_settings
from thermal_dataset_manager.ingest.telemetry import BosonTelemetry, parse_telemetry_rows
from thermal_dataset_manager.models.metadata import CameraStreamMetadata, DatasetMetadata
from thermal_dataset_manager.storage import zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

FrameSelection = slice | int | Sequence[int] | np.ndarray | None


class Video:
    """One thermal video (a single camera stream). Cheap to hold: the zarr
    store is only opened, and frames only read, when first needed."""

    def __init__(self, storage: FilesystemDatasetStorage, metadata: DatasetMetadata) -> None:
        self._storage = storage
        self.metadata = metadata
        self.camera: CameraStreamMetadata = metadata.auto.cameras[0]
        self._group = None
        self._timestamps: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"Video({self.dataset_id}, {self.name!r}, {self.n_frames} x {self.shape})"

    # -- identity / description -------------------------------------------------

    @property
    def dataset_id(self) -> str:
        return self.metadata.auto.dataset_id

    @property
    def name(self) -> str:
        return self.metadata.user.name

    @property
    def capture_id(self) -> str | None:
        return self.metadata.user.capture_id

    @property
    def tags(self) -> list[str]:
        return self.metadata.user.tags

    @property
    def n_frames(self) -> int:
        return self.camera.frame_count

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the image rows."""
        return (self.camera.frame_height, self.camera.frame_width)

    @property
    def fps(self) -> float | None:
        return self.camera.fps_estimate

    @property
    def path(self) -> Path:
        return self._storage.zarr_store_path(self.dataset_id)

    # -- data -------------------------------------------------------------------

    def _cam_group(self):
        if self._group is None:
            root = zarr_store.open_group(self.path)
            self._group = root[f"camera_{self.camera.camera_index}"]
        return self._group

    @property
    def counts(self):
        """The lazy zarr array of image frames, shape (n_frames, H, W) uint16.
        Slice it yourself for full control; nothing is read until you do."""
        return self._cam_group()["counts"]

    @property
    def has_telemetry(self) -> bool:
        return self.camera.telemetry_row_count != 0

    @property
    def timestamps(self) -> np.ndarray:
        """Per-frame timestamps in seconds, float64 (absolute UTC epoch or
        relative, per `camera.timestamp_kind`)."""
        if self._timestamps is None:
            self._timestamps = self._cam_group()["timestamps"][:]
        return self._timestamps

    def time_slice(self, start: float | None = None, end: float | None = None) -> slice:
        """Frame slice covering [start, end) seconds since the first frame."""
        t = self.timestamps - self.timestamps[0]
        lo = 0 if start is None else int(np.searchsorted(t, start, side="left"))
        hi = len(t) if end is None else int(np.searchsorted(t, end, side="left"))
        return slice(lo, hi)

    def _resolve(self, selection: FrameSelection, time: tuple[float | None, float | None] | None):
        if time is not None:
            if selection is not None:
                raise ValueError("pass either a frame selection or time=, not both")
            return self.time_slice(*time)
        return slice(None) if selection is None else selection

    @staticmethod
    def _read(array, selection):
        if isinstance(selection, (slice, int, np.integer)):
            return array[selection]
        idx = np.asarray(selection, dtype=int)
        # zarr orthogonal indexing needs sorted unique-ish indices to be efficient;
        # keep the caller's order by sorting for the read and un-sorting after.
        order = np.argsort(idx, kind="stable")
        data = array.get_orthogonal_selection((idx[order], slice(None), slice(None)))
        return data[np.argsort(order, kind="stable")]

    def frames(
        self,
        selection: FrameSelection = None,
        *,
        time: tuple[float | None, float | None] | None = None,
        raw: bool = False,
    ) -> np.ndarray:
        """Load frames as uint16, shape (n, H, W) (or (H, W) for an int).

        `selection`: a slice, one index, a list/array of indices, or None for
        every frame. `time=(start, end)` selects by seconds since the first
        frame instead. `raw=True` re-attaches the telemetry rows where they
        sat in the source frame, giving (n, H + |R|, W).
        """
        sel = self._resolve(selection, time)
        image = self._read(self.counts, sel)
        if not raw or not self.has_telemetry:
            return image
        tele = self._read(self._cam_group()["telemetry"], sel)
        parts = [tele, image] if self.camera.telemetry_row_count > 0 else [image, tele]
        return np.concatenate(parts, axis=-2)

    def iter_batches(
        self, batch_size: int, selection: slice | None = None, *, raw: bool = False
    ) -> Iterator[np.ndarray]:
        """Yield consecutive batches of at most `batch_size` frames, so a long
        video can be processed with bounded memory."""
        start, stop, step = (selection or slice(None)).indices(self.n_frames)
        for lo in range(start, stop, step * batch_size):
            yield self.frames(slice(lo, min(lo + step * batch_size, stop), step), raw=raw)

    def telemetry_rows(self, selection: FrameSelection = None) -> np.ndarray:
        """The stripped telemetry rows, shape (n, |R|, W) uint16; empty
        second axis when the stream has none."""
        sel = self._resolve(selection, None)
        if not self.has_telemetry:
            n = len(range(*sel.indices(self.n_frames))) if isinstance(sel, slice) else np.size(sel)
            return np.zeros((n, 0, self.camera.frame_width), dtype="uint16")
        return self._read(self._cam_group()["telemetry"], sel)

    def parse_telemetry(self, frame_index: int) -> BosonTelemetry | None:
        """Decode one frame's Boson+ telemetry (temperatures, frame counter,
        ...). None when the stream has no valid telemetry."""
        if not self.has_telemetry:
            return None
        return parse_telemetry_rows(self.telemetry_rows(frame_index))


class Library:
    def __init__(self, data_root: Path | str | None = None) -> None:
        settings = Settings(data_root=Path(data_root)) if data_root is not None else get_settings()
        self.storage = FilesystemDatasetStorage(settings.data_root)
        self._engine = make_engine(settings.catalog_db)

    def get(self, dataset_id: str) -> Video:
        return Video(self.storage, self.storage.read_metadata(dataset_id))

    def capture(self, capture_id: str) -> list[Video]:
        """Every video sharing a capture id (e.g. the cameras of one file)."""
        return self.find(capture_id=capture_id)

    def find(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        capture_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        scene_category: str | None = None,
        capture_location: str | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        min_duration: float | None = None,
        shape: tuple[int, int] | None = None,
        has_telemetry: bool | None = None,
        gain_mode: str | None = None,
        pipeline_enabled: Sequence[str] = (),
        pipeline_disabled: Sequence[str] = (),
        where: Callable[[DatasetMetadata], bool] | None = None,
        limit: int | None = None,
    ) -> list[Video]:
        """Videos matching every given filter.

        `q`, `tags`, `capture_id`, `date_from/to` are searched in the catalog
        (`q` is a full-text query over name, description, notes and tags).
        The rest are checked against each candidate's metadata: `shape` is
        (height, width) of the image rows; `pipeline_enabled`/`_disabled` name
        Boson processing stages (e.g. "temporal_filter") that must be on / off
        in the capture's telemetry, so they exclude videos without telemetry;
        `where` is any predicate on the full `DatasetMetadata`.
        """
        with get_session(self._engine) as session:
            records = repository.search(
                session,
                q=q,
                tags=tags,
                capture_id=capture_id,
                date_from=date_from,
                date_to=date_to,
                sort_by="name",
                descending=False,
                limit=1_000_000,
            )
        videos: list[Video] = []
        for record in records:
            md = self.storage.read_metadata(record.dataset_id)
            cam = md.auto.cameras[0]
            stages = set(cam.telemetry.pipeline_enabled or []) if cam.telemetry else None
            checks = [
                scene_category is None or md.user.scene_category == scene_category,
                capture_location is None or md.user.capture_location == capture_location,
                min_frames is None or cam.frame_count >= min_frames,
                max_frames is None or cam.frame_count <= max_frames,
                min_duration is None or (cam.duration_seconds or 0) >= min_duration,
                shape is None or (cam.frame_height, cam.frame_width) == tuple(shape),
                has_telemetry is None or (cam.telemetry is not None) == has_telemetry,
                gain_mode is None or (cam.telemetry is not None and cam.telemetry.gain_mode == gain_mode),
                not pipeline_enabled or (stages is not None and set(pipeline_enabled) <= stages),
                not pipeline_disabled or (stages is not None and not (set(pipeline_disabled) & stages)),
                where is None or where(md),
            ]
            if all(checks):
                videos.append(Video(self.storage, md))
                if limit is not None and len(videos) >= limit:
                    break
        return videos
