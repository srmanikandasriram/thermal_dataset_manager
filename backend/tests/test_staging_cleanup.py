import os
import time

from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


def test_sweep_removes_only_stale_staging_dirs(tmp_path):
    storage = FilesystemDatasetStorage(tmp_path)

    fresh = storage.new_staging_dir("fresh-token")
    (fresh / "capture.npz").write_bytes(b"data")

    stale = storage.new_staging_dir("stale-token")
    (stale / "capture.npz").write_bytes(b"data")
    # Backdate the stale dir's mtime so it looks like it was created a while ago.
    old_time = time.time() - 3600
    os.utime(stale, (old_time, old_time))

    removed = storage.sweep_stale_staging(max_age_seconds=1800)

    assert removed == 1
    assert fresh.exists()
    assert not stale.exists()


def test_sweep_on_missing_staging_root_is_noop(tmp_path):
    storage = FilesystemDatasetStorage(tmp_path / "does-not-exist")
    assert storage.sweep_stale_staging(max_age_seconds=100) == 0


def test_sweep_ignores_non_directory_entries(tmp_path):
    storage = FilesystemDatasetStorage(tmp_path)
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    stray_file = staging_root / "stray.txt"
    stray_file.write_text("oops")
    old_time = time.time() - 3600
    os.utime(stray_file, (old_time, old_time))

    removed = storage.sweep_stale_staging(max_age_seconds=1)

    assert removed == 0
    assert stray_file.exists()
