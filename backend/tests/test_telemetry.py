import os

import numpy as np
import pytest

from thermal_dataset_manager.ingest.metadata_compute import compute_camera_metadata
from thermal_dataset_manager.ingest.telemetry import (
    TelemetryParseError,
    parse_frame_telemetry,
    parse_telemetry_line,
)

WIDTH = 640


def _put_u16(row: np.ndarray, word: int, value: int) -> None:
    row[word] = value & 0xFFFF


def _put_u32(row: np.ndarray, word: int, value: int) -> None:
    # most-significant 16-bit word first (see telemetry module docstring)
    row[word] = (value >> 16) & 0xFFFF
    row[word + 1] = value & 0xFFFF


def _put_u32_le(row: np.ndarray, word: int, value: int) -> None:
    # least-significant 16-bit word first (pipeline enable bits)
    row[word] = value & 0xFFFF
    row[word + 1] = (value >> 16) & 0xFFFF


def _put_ascii(row: np.ndarray, word: int, n_words: int, text: str) -> None:
    buf = text.encode("ascii").ljust(n_words * 2, b"\x00")
    row[word : word + n_words] = np.frombuffer(buf, dtype="<u2").byteswap()


def _synthetic_row(**overrides) -> np.ndarray:
    row = np.zeros(WIDTH, dtype="<u2")
    _put_u16(row, 0, overrides.get("revision", 3))
    _put_u32(row, 1, overrides.get("camera_serial", 275171))
    _put_u32(row, 3, overrides.get("sensor_serial", 5002640))
    _put_ascii(row, 5, 10, overrides.get("part_number", "22640A024-6IAAX"))
    _put_u16(row, 28, overrides.get("frame_rate_hz", 60))
    # status bits: FFC complete (0b11), gain mode low (0b001 << 2), FFC desired (bit 5)
    _put_u16(row, 38, overrides.get("status", 0b11 | (1 << 2) | (1 << 5)))
    _put_u32(row, 42, overrides.get("frame_counter", 5014889))
    _put_u16(row, 46, overrides.get("fpa_counts", 26033))
    _put_u16(row, 47, overrides.get("camera_temp_k10", 3076))
    _put_u16(row, 48, overrides.get("camera_temp_last_ffc_k10", 3076))
    _put_u32_le(row, 55, overrides.get("pipeline_bits", 0b1010000111))
    _put_u32(row, 81, overrides.get("core_temp_c1000", 43848) & 0xFFFFFFFF)
    for i, v in enumerate((0x5A5A, 0xA55A, 0xA5A5)):
        _put_u16(row, 89 + i, v)
    _put_u32(row, 140, overrides.get("timestamp_ms", 83583332))
    _put_u16(row, 156, overrides.get("external_sync", 0))
    return row


def test_parses_core_fields_from_synthetic_row():
    t = parse_telemetry_line(_synthetic_row())

    assert t.revision == 3
    assert t.camera_serial == 275171
    assert t.sensor_serial == 5002640
    assert t.part_number == "22640A024-6IAAX"
    assert t.frame_rate_hz == 60
    assert t.frame_counter == 5014889
    assert t.timestamp_ms == 83583332
    assert t.fpa_counts == 26033
    assert t.camera_temp_k == pytest.approx(307.6)
    assert t.camera_temp_c == pytest.approx(34.45)
    assert t.core_temp_c == pytest.approx(43.848)
    assert t.toggling_pattern_ok


def test_status_and_pipeline_bits_decode():
    t = parse_telemetry_line(_synthetic_row())
    assert t.ffc_state == "complete"
    assert t.gain_mode == "low"
    assert t.ffc_desired is True
    assert t.table_switch_desired is False
    assert t.raw_pipeline_bits == 0b1010000111
    # bits 0,1,2,7,9 set
    assert t.pipeline_enabled == {
        "ffc_offset": True,
        "gain": True,
        "temp_compensation": True,
        "averager": False,
        "temporal_filter": False,
        "scnr": False,
        "ssn": False,
        "bpr": True,
        "sffc": True,
        "srnr": False,
        "lfsr": False,
    }


def test_negative_core_temp_is_signed():
    row = _synthetic_row(core_temp_c1000=(-5123) & 0xFFFFFFFF)
    assert parse_telemetry_line(row).core_temp_c == pytest.approx(-5.123)


def test_short_row_rejected():
    with pytest.raises(TelemetryParseError):
        parse_telemetry_line(np.zeros(100, dtype="<u2"))


def test_parse_frame_picks_valid_row_and_skips_padding():
    # Two telemetry rows on top; only the first carries the structure.
    telem = _synthetic_row()
    padding = np.zeros(WIDTH, dtype="<u2")
    image = np.full((512, WIDTH), 1234, dtype="<u2")
    frame = np.vstack([telem, padding, image])

    t = parse_frame_telemetry(frame, telemetry_row_count=2)
    assert t is not None
    assert t.telemetry_row_index == 0
    assert t.frame_counter == 5014889


def test_parse_frame_telemetry_bottom_rows():
    telem = _synthetic_row(frame_counter=42)
    padding = np.zeros(WIDTH, dtype="<u2")
    image = np.full((512, WIDTH), 7, dtype="<u2")
    frame = np.vstack([image, telem, padding])

    t = parse_frame_telemetry(frame, telemetry_row_count=-2)
    assert t is not None and t.frame_counter == 42


def test_parse_frame_telemetry_none_when_disabled_or_absent():
    frame = np.zeros((514, WIDTH), dtype="<u2")
    assert parse_frame_telemetry(frame, telemetry_row_count=0) is None
    # rows present but all-zero -> not a valid telemetry line
    assert parse_frame_telemetry(frame, telemetry_row_count=2) is None


def _frame_stack(n_frames: int, **row_overrides) -> np.ndarray:
    """Frame stack with two top telemetry rows (row 0 real, row 1 padding)."""
    image = np.full((10, WIDTH), 4321, dtype="<u2")
    padding = np.zeros(WIDTH, dtype="<u2")
    frames = [
        np.vstack(
            [
                _synthetic_row(frame_counter=5_000_000 + i, timestamp_ms=1000 + 17 * i, **row_overrides),
                padding,
                image,
            ]
        )
        for i in range(n_frames)
    ]
    return np.stack(frames)


def _split(raw: np.ndarray, rows: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Split raw frames the way ingest does: (image rows, telemetry rows)."""
    return raw[:, rows:, :], raw[:, :rows, :]


def test_compute_camera_metadata_embeds_static_telemetry():
    frames, telemetry = _split(_frame_stack(12))
    timestamps = 1_700_000_000.0 + np.arange(12) / 60.0

    meta = compute_camera_metadata(
        camera_index=0, camera_label="A", frames=frames, telemetry=telemetry, timestamps=timestamps,
        telemetry_row_count=2, timestamp_kind="absolute_utc", stats_sample_frames=4,
    )

    t = meta.telemetry
    assert t is not None
    assert t.telemetry_revision == 3
    assert t.part_number == "22640A024-6IAAX"
    assert t.camera_serial == 275171
    assert t.frame_rate_hz == 60
    assert t.gain_mode == "low"
    assert t.pipeline_enabled == ["bpr", "ffc_offset", "gain", "sffc", "temp_compensation"]
    assert t.frames_sampled == 8  # _TELEMETRY_SAMPLE_FRAMES


def test_static_telemetry_absent_without_telemetry_rows():
    frames = np.full((5, 10, WIDTH), 7, dtype="<u2")
    meta = compute_camera_metadata(
        camera_index=0, camera_label=None, frames=frames, telemetry=None,
        timestamps=np.arange(5, dtype=float), telemetry_row_count=0,
        timestamp_kind="relative", stats_sample_frames=4,
    )
    assert meta.telemetry is None


def test_static_telemetry_drops_fields_that_disagree(monkeypatch):
    # two frames with different serials -> serial is dropped, revision kept
    a = _frame_stack(1, camera_serial=111)
    b = _frame_stack(1, camera_serial=222)
    frames, telemetry = _split(np.concatenate([a, b, a, b]))
    meta = compute_camera_metadata(
        camera_index=0, camera_label=None, frames=frames, telemetry=telemetry,
        timestamps=np.arange(len(frames), dtype=float) / 60.0,
        telemetry_row_count=2, timestamp_kind="relative", stats_sample_frames=4,
    )
    assert meta.telemetry is not None
    assert meta.telemetry.camera_serial is None
    assert meta.telemetry.telemetry_revision == 3


# --- Optional integration check against a real ingested dataset -----------------
# Set THERMAL_TELEMETRY_TEST_ZARR to a raw.zarr path to exercise the parser on
# real Boson+ frames (skipped by default so the suite stays hermetic).

_REAL_ZARR = os.environ.get("THERMAL_TELEMETRY_TEST_ZARR")


@pytest.mark.skipif(not _REAL_ZARR, reason="THERMAL_TELEMETRY_TEST_ZARR not set")
def test_real_dataset_telemetry_is_consistent():
    import zarr

    root = zarr.open_group(store=zarr.DirectoryStore(_REAL_ZARR), mode="r")
    cam = root["camera_0"]
    trc = int(cam.attrs["telemetry_row_count"])
    counts = cam["counts"]

    parsed = [parse_frame_telemetry(counts[i], trc) for i in range(10)]
    assert all(p is not None for p in parsed)
    assert all(p.toggling_pattern_ok for p in parsed)
    assert all(p.revision in (1, 2, 3) for p in parsed)
    # frame counter is a rolling per-frame counter
    deltas = np.diff([p.frame_counter for p in parsed])
    assert set(deltas.tolist()) == {1}
    # camera temperature is a sane physical value
    assert all(250.0 < p.camera_temp_k < 350.0 for p in parsed)
    # pipeline bits decode to a plausible always-on core (offset + gain correction)
    assert all(p.pipeline_enabled["ffc_offset"] and p.pipeline_enabled["gain"] for p in parsed)


def test_ingest_strips_telemetry_and_records_metadata(tmp_path, storage, db_session):
    from thermal_dataset_manager.ingest.pipeline import ingest_dataset
    from thermal_dataset_manager.storage import zarr_store

    raw = _frame_stack(12)
    src = tmp_path / "boson.npz"
    np.savez(src, raw_thr_frames=raw, raw_thr_tstamps=1_700_000_000.0 + np.arange(12) / 60.0)

    [meta] = ingest_dataset(
        src, storage, db_session,
        default_telemetry_row_count=2, zarr_target_chunk_bytes=1024 * 1024, stats_sample_frames=4,
    )

    cam = meta.auto.cameras[0]
    assert cam.telemetry is not None
    assert cam.telemetry.camera_serial == 275171
    assert cam.telemetry.gain_mode == "low"

    store_path = storage.zarr_store_path(meta.auto.dataset_id)
    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 0, 5), raw[5, 2:, :])
    np.testing.assert_array_equal(zarr_store.read_frame_telemetry(store_path, 0, 5), raw[5, :2, :])
