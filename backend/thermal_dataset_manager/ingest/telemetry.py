"""Parser for the Teledyne FLIR Boson+ telemetry line.

Layout is per "Table 5: Telemetry Line Encoding" in the Boson+ Product
Datasheet (Rev 114, section 6.7). The telemetry line is one image row -- for
the 16-bit pre-AGC tap these captures use, that is ``frame_width`` uint16
words. When telemetry is enabled the camera adds *two* extra rows; only the
first carries the Table 5 structure (the second is zero padding), so this
module scans the telemetry rows and parses the first one that validates.

Word-order conventions, established empirically against ingested captures
(and consistent with the datasheet's toggling-pattern check at bytes
178-183):

* 16-bit scalar fields  -> the uint16 word as-is.
* 32-bit fields split over two words -> most-significant word first, i.e.
  ``(row[w] << 16) | row[w + 1]`` (frame counter, timestamp). The "pipeline
  enable bits" field is the known exception -- it is least-significant word
  first; see ``_u32_le``.
* ASCII fields          -> the two chars in each word are swapped relative
  to numpy's little-endian read, so byte-swap before decoding.

Datasheet offsets are given in bytes; every field starts on an even byte, so
``word_index = byte_offset // 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thermal_dataset_manager.ingest.types import telemetry_row_slice

# Highest word index touched by Table 5 is 250 (spotmeterMaxColumn, byte 500).
_MIN_TELEMETRY_WORDS = 251

_TOGGLING_PATTERN = (0x5A5A, 0xA55A, 0xA5A5)
_TOGGLING_WORD = 89  # byte 178

_FFC_STATE = {0: "never_started", 1: "imminent", 2: "in_progress", 3: "complete"}
_GAIN_MODE = {0: "high", 1: "low", 2: "automatic"}

# "Pipeline enable bits" (byte 110). Bits not listed here are reserved.
_PIPELINE_BITS = {
    0: "ffc_offset",
    1: "gain",
    2: "temp_compensation",
    3: "averager",
    4: "temporal_filter",
    5: "scnr",
    6: "ssn",  # a.k.a. SPNR
    7: "bpr",
    9: "sffc",
    13: "srnr",
    17: "lfsr",
}

_KELVIN_ZERO_C = 273.15


class TelemetryParseError(ValueError):
    pass


def _u16(row: np.ndarray, word: int) -> int:
    return int(row[word])


def _u32(row: np.ndarray, word: int) -> int:
    """32-bit unsigned, most-significant 16-bit word first.

    This is the word order the frame counter and millisecond timestamp use
    (verified empirically: +1 per frame, +~16.7 ms per frame).
    """
    return (int(row[word]) << 16) | int(row[word + 1])


def _u32_le(row: np.ndarray, word: int) -> int:
    """32-bit unsigned, least-significant 16-bit word first.

    The "pipeline enable bits" field uses this order, not `_u32`'s -- with
    `_u32` its assigned bits land on reserved positions and its reserved
    positions light up, whereas this way the low word carries exactly the
    documented stage bits (stable across all inspected captures). FLIR's
    telemetry is not internally consistent about 32-bit word order.
    """
    return int(row[word]) | (int(row[word + 1]) << 16)


def _i32(row: np.ndarray, word: int) -> int:
    value = _u32(row, word)
    return value - (1 << 32) if value >= (1 << 31) else value


def _ascii(row: np.ndarray, word: int, n_words: int) -> str:
    raw = row[word : word + n_words].byteswap().tobytes()
    return raw.split(b"\x00", 1)[0].decode("ascii", "replace").strip()


@dataclass(frozen=True)
class BosonTelemetry:
    """Decoded Boson+ telemetry line. Temperatures are in the units named;
    ``raw_status_bits`` / ``raw_pipeline_bits`` are kept for callers that want
    fields this class does not break out."""

    revision: int
    camera_serial: int
    sensor_serial: int
    part_number: str
    software_revision: tuple[int, int, int]  # (major, minor, patch) words, as stored

    frame_rate_hz: int
    frame_counter: int
    frame_counter_last_ffc: int
    timestamp_ms: int

    fpa_counts: int
    fpa_temp_k: float | None  # datasheet gives no Tfpa conversion; None unless known
    camera_temp_k: float
    camera_temp_c: float
    camera_temp_last_ffc_k: float
    core_temp_c: float

    ffc_state: str
    gain_mode: str
    ffc_desired: bool
    table_switch_desired: bool
    low_power_state: bool
    overtemp_state: bool
    overtemp_event_count: int

    mean_in_roi: int
    min_in_roi: int
    max_in_roi: int
    roi: tuple[int, int, int, int]

    external_sync_slave: bool
    valid_image: int
    current_nuc_table: int
    desired_nuc_table: int

    raw_status_bits: int
    raw_pipeline_bits: int
    pipeline_enabled: dict[str, bool]  # stage name -> enabled (see _PIPELINE_BITS)
    toggling_pattern: tuple[int, int, int]

    spotmeter_mean_counts: int
    spotmeter_min_counts: int
    spotmeter_max_counts: int

    telemetry_row_index: int = field(default=0)

    @property
    def toggling_pattern_ok(self) -> bool:
        return self.toggling_pattern == _TOGGLING_PATTERN


def _row_looks_like_telemetry(row: np.ndarray) -> bool:
    if row.shape[0] < _MIN_TELEMETRY_WORDS:
        return False
    revision = _u16(row, 0)
    pattern = tuple(int(v) for v in row[_TOGGLING_WORD : _TOGGLING_WORD + 3])
    return revision in (1, 2, 3) and pattern == _TOGGLING_PATTERN


def parse_telemetry_line(row: np.ndarray) -> BosonTelemetry:
    """Decode a single telemetry row (a 1-D uint16 array of at least 251 words)."""
    row = np.ascontiguousarray(row, dtype="<u2")
    if row.ndim != 1:
        raise TelemetryParseError(f"expected a 1-D row, got shape {row.shape}")
    if row.shape[0] < _MIN_TELEMETRY_WORDS:
        raise TelemetryParseError(
            f"row has {row.shape[0]} words, need at least {_MIN_TELEMETRY_WORDS}"
        )

    status = _u16(row, 38)
    pipeline_bits = _u32_le(row, 55)
    camera_temp_k = _u16(row, 47) / 10.0

    return BosonTelemetry(
        revision=_u16(row, 0),
        camera_serial=_u32(row, 1),
        sensor_serial=_u32(row, 3),
        part_number=_ascii(row, 5, 10),
        software_revision=(_u32(row, 22), _u32(row, 24), _u32(row, 26)),
        frame_rate_hz=_u16(row, 28),
        frame_counter=_u32(row, 42),
        frame_counter_last_ffc=_u32(row, 44),
        timestamp_ms=_u32(row, 140),
        fpa_counts=_u16(row, 46),
        fpa_temp_k=None,
        camera_temp_k=camera_temp_k,
        camera_temp_c=camera_temp_k - _KELVIN_ZERO_C,
        camera_temp_last_ffc_k=_u16(row, 48) / 10.0,
        core_temp_c=_i32(row, 81) / 1000.0,
        ffc_state=_FFC_STATE.get(status & 0b11, "reserved"),
        gain_mode=_GAIN_MODE.get((status >> 2) & 0b111, "reserved"),
        ffc_desired=bool((status >> 5) & 1),
        table_switch_desired=bool((status >> 6) & 1),
        low_power_state=bool((status >> 7) & 1),
        overtemp_state=bool((status >> 8) & 1),
        overtemp_event_count=_u32(row, 83),
        mean_in_roi=_u16(row, 33),
        min_in_roi=_u16(row, 34),
        max_in_roi=_u16(row, 35),
        roi=(_u16(row, 29), _u16(row, 30), _u16(row, 31), _u16(row, 32)),
        external_sync_slave=bool(_u16(row, 156)),
        valid_image=_u16(row, 191),
        current_nuc_table=_u16(row, 79),
        desired_nuc_table=_u16(row, 80),
        raw_status_bits=status,
        raw_pipeline_bits=pipeline_bits,
        pipeline_enabled={
            name: bool((pipeline_bits >> bit) & 1) for bit, name in _PIPELINE_BITS.items()
        },
        toggling_pattern=(_u16(row, 89), _u16(row, 90), _u16(row, 91)),
        spotmeter_mean_counts=_u16(row, 243),
        spotmeter_min_counts=_u16(row, 245),
        spotmeter_max_counts=_u16(row, 248),
    )


def parse_telemetry_rows(rows: np.ndarray) -> BosonTelemetry | None:
    """Parse the telemetry rows ``(R, W)`` of one frame.

    Returns None when none of the rows validate as a Boson+ telemetry line
    (including when ``R == 0``, i.e. telemetry is disabled).
    """
    for i, row in enumerate(rows):
        if _row_looks_like_telemetry(row):
            parsed = parse_telemetry_line(row)
            object.__setattr__(parsed, "telemetry_row_index", i)
            return parsed
    return None


def telemetry_rows(raw_frame: np.ndarray, telemetry_row_count: int) -> np.ndarray:
    """The telemetry rows of a raw (unstripped) frame, in top-to-bottom order.

    ``telemetry_row_count`` follows the project convention: positive N -> the
    top N rows, negative N -> the bottom |N| rows, 0 -> none.
    """
    return raw_frame[telemetry_row_slice(telemetry_row_count)]


def parse_frame_telemetry(
    raw_frame: np.ndarray, telemetry_row_count: int
) -> BosonTelemetry | None:
    """Parse the telemetry embedded in one raw (unstripped) frame ``(H, W)``.

    Returns None when telemetry is disabled (``telemetry_row_count == 0``) or
    when none of the telemetry rows validate as a Boson+ telemetry line.
    """
    return parse_telemetry_rows(telemetry_rows(raw_frame, telemetry_row_count))
