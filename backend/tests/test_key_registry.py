import pytest

from thermal_dataset_manager.ingest.key_registry import KeyRegistryError, discover_camera_keys


def test_single_camera_no_suffix():
    pairs = discover_camera_keys(["raw_thr_frames", "raw_thr_tstamps"])
    assert len(pairs) == 1
    assert pairs[0].camera_index == 0
    assert pairs[0].frames_key == "raw_thr_frames"
    assert pairs[0].timestamps_key == "raw_thr_tstamps"


def test_multi_camera_suffixed():
    pairs = discover_camera_keys(
        ["raw_thr_frames_0", "raw_thr_tstamps_0", "raw_thr_frames_1", "raw_thr_tstamps_1"]
    )
    assert {p.camera_index for p in pairs} == {0, 1}


def test_multi_camera_letter_suffixed():
    """_A/_B are normalized to the canonical _0/_1 labels, but the original
    keys are still used to read the arrays."""
    pairs = discover_camera_keys(
        [
            "raw_thr_frames_A",
            "raw_thr_tstamps_A",
            "raw_thr_frames_B",
            "raw_thr_tstamps_B",
            "thr_cam_timestamp_offset_A",  # unrelated key, should be ignored
        ]
    )
    assert {p.camera_label for p in pairs} == {"0", "1"}
    assert {p.camera_index for p in pairs} == {0, 1}
    a = next(p for p in pairs if p.camera_label == "0")
    assert a.frames_key == "raw_thr_frames_A"
    assert a.timestamps_key == "raw_thr_tstamps_A"


def test_letter_and_numeric_suffix_collision_raises():
    with pytest.raises(KeyRegistryError):
        discover_camera_keys(
            [
                "raw_thr_frames_A",
                "raw_thr_tstamps_A",
                "raw_thr_frames_0",
                "raw_thr_tstamps_0",
            ]
        )


def test_unrecognized_keys_raise():
    with pytest.raises(KeyRegistryError):
        discover_camera_keys(["something_else", "unrelated"])


def test_unmatched_frames_key_raises():
    with pytest.raises(KeyRegistryError):
        discover_camera_keys(["raw_thr_frames_0", "raw_thr_tstamps_1"])
