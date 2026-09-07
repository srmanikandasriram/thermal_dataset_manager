"""Extendable registry mapping source-file array/dataset key names to camera streams.

Each entry is a (frames_pattern, timestamps_pattern) pair of regexes with a
named group `suffix` for the optional multi-camera suffix (e.g. `_0`, `_1`,
`_A`, `_B`). To support a new naming convention, append a pattern pair here
-- no other pipeline code needs to change.

Camera suffixes are not assumed to be numeric or contiguous. Known letter
suffixes are normalized via `_SUFFIX_ALIASES` (`_A` -> `_0`, `_B` -> `_1`)
so a letter-labeled capture is indistinguishable from a numeric one
downstream. Cameras are then assigned a stable `camera_index` (0, 1, 2, ...)
in sorted-suffix order; the (normalized) suffix is kept as `camera_label`.
"""

import re
from dataclasses import dataclass

KEY_PATTERNS: list[tuple[re.Pattern[str], re.Pattern[str]]] = [
    (
        re.compile(r"^raw_thr_frames(?:_(?P<suffix>\w+))?$"),
        re.compile(r"^raw_thr_tstamps(?:_(?P<suffix>\w+))?$"),
    ),
]

# Suffix used internally for a capture with no multi-camera suffix at all.
_NO_SUFFIX = ""

# Some captures label cameras with letters instead of indices. Normalize those
# to the canonical numeric suffix so a `_A`/`_B` capture is indistinguishable
# from a `_0`/`_1` one downstream. Matched case-insensitively.
_SUFFIX_ALIASES: dict[str, str] = {
    "a": "0",
    "b": "1",
}


def _normalize_suffix(suffix: str) -> str:
    return _SUFFIX_ALIASES.get(suffix.lower(), suffix)


@dataclass(frozen=True)
class CameraKeyPair:
    camera_index: int
    camera_label: str | None
    frames_key: str
    timestamps_key: str


class KeyRegistryError(ValueError):
    pass


def discover_camera_keys(keys: list[str]) -> list[CameraKeyPair]:
    """Find and pair up (frames, timestamps) keys for each camera stream present.

    Cameras are matched by their suffix (absent suffix == the sole camera,
    labeled None). Raises KeyRegistryError if a frames key has no matching
    timestamps key (or vice versa), or if no recognized keys are found at all.
    """
    key_set = set(keys)
    frames_by_suffix: dict[str, str] = {}
    timestamps_by_suffix: dict[str, str] = {}

    def _record(target: dict[str, str], suffix: str, key: str) -> None:
        norm = _normalize_suffix(suffix) if suffix else _NO_SUFFIX
        if norm in target and target[norm] != key:
            raise KeyRegistryError(
                f"keys '{target[norm]}' and '{key}' both map to camera suffix '{norm}'"
            )
        target[norm] = key

    for frames_pattern, timestamps_pattern in KEY_PATTERNS:
        for key in key_set:
            m = frames_pattern.match(key)
            if m:
                _record(frames_by_suffix, m.group("suffix") or "", key)
        for key in key_set:
            m = timestamps_pattern.match(key)
            if m:
                _record(timestamps_by_suffix, m.group("suffix") or "", key)

    if not frames_by_suffix and not timestamps_by_suffix:
        raise KeyRegistryError(
            f"No recognized frame/timestamp keys found among: {sorted(key_set)}"
        )

    all_suffixes = sorted(set(frames_by_suffix) | set(timestamps_by_suffix))
    pairs: list[CameraKeyPair] = []
    problems: list[str] = []
    for camera_index, suffix in enumerate(all_suffixes):
        frames_key = frames_by_suffix.get(suffix)
        timestamps_key = timestamps_by_suffix.get(suffix)
        label = suffix or None
        if frames_key is None:
            problems.append(
                f"camera '{label}': found timestamps key '{timestamps_key}' but no matching frames key"
            )
            continue
        if timestamps_key is None:
            problems.append(
                f"camera '{label}': found frames key '{frames_key}' but no matching timestamps key"
            )
            continue
        pairs.append(
            CameraKeyPair(
                camera_index=camera_index,
                camera_label=label,
                frames_key=frames_key,
                timestamps_key=timestamps_key,
            )
        )

    if problems:
        raise KeyRegistryError("; ".join(problems))

    return pairs
