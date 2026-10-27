#!/usr/bin/env python
"""CLI wrapper around the ingest pipeline, usable before any HTTP API exists.

Usage:
    python backend/scripts/ingest_cli.py <path-to-capture-file> [<more files>...]
    python backend/scripts/ingest_cli.py --telemetry-rows 0 no_telemetry_capture.npz
    python backend/scripts/ingest_cli.py --capture run42 cam_a.npz cam_b.npz
    python backend/scripts/ingest_cli.py /path/to/captures/        # whole folder, recursively
    python backend/scripts/ingest_cli.py --name-from-path 2 /data/captures/   # 'run1_take3_thermal_video'
    python backend/scripts/ingest_cli.py --move-ingested-to /data/ingested /data/to_be_ingested

A multi-camera file is split into one dataset per camera.

--move-ingested-to DIR moves each source file into DIR once it has been ingested
successfully, keeping its path relative to the folder (or file's folder) it was
given from. Files that are skipped or fail stay where they are, so what is left
behind is exactly what still needs attention. Add --move-duplicates to also move
files whose content is already ingested.
"""

import argparse
import shutil
import sys
from pathlib import Path

from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import get_settings
from thermal_dataset_manager.ingest.key_registry import KeyRegistryError
from thermal_dataset_manager.ingest.pipeline import (
    DuplicateDatasetError,
    UnsupportedSourceFormatError,
    ingest_dataset,
)
from thermal_dataset_manager.ingest.validation import IngestValidationError
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


SOURCE_SUFFIXES = (".npz.zst", ".npz", ".h5", ".hdf5")


def expand_paths(paths: list[Path], exclude_dir: Path | None = None) -> list[tuple[Path, Path]]:
    """(file, base) pairs. Files are kept as given (base = their folder);
    directories are searched recursively for supported source files (sorted,
    so runs are reproducible; base = that directory). Anything under
    `exclude_dir` (e.g. the --move-ingested-to destination) is left out."""
    expanded: list[tuple[Path, Path]] = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                p for p in path.rglob("*") if p.is_file() and p.name.lower().endswith(SOURCE_SUFFIXES)
            )
            expanded.extend((p, path) for p in found)
        else:
            expanded.append((path, path.parent))
    if exclude_dir is not None:
        expanded = [(p, b) for p, b in expanded if exclude_dir not in p.resolve().parents]
    return expanded


def name_from_path(path: Path, n_parents: int) -> str:
    """'<grandparent>_<parent>_<stem>' style name: the last `n_parents` folder
    names above the file, then the file name without its format extension."""
    stem = path.name
    for suffix in SOURCE_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return "_".join([*path.resolve().parent.parts[-n_parents:], stem] if n_parents > 0 else [stem])


def move_source(path: Path, base: Path, dest_root: Path) -> bool:
    """Move an ingested source file to dest_root/<path relative to base>.
    Never overwrites; returns False (after saying why) if it could not move."""
    target = dest_root / path.relative_to(base)
    if target.exists():
        print(f"NOT MOVED  {path}: {target} already exists", file=sys.stderr)
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    except OSError as e:
        print(f"NOT MOVED  {path}: {e}", file=sys.stderr)
        return False
    print(f"MOVED   {path} -> {target}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one or more raw thermal capture files.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=(
            "Source .npz/.npz.zst/.h5 files, or directories to search recursively for them "
            "(other file types are ignored)"
        ),
    )
    parser.add_argument(
        "--telemetry-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Telemetry rows per frame, applied to every camera in every file. "
            "2 (default) = top 2 rows are telemetry; 0 = no telemetry rows; "
            "-2 = bottom 2 rows are telemetry. "
            "Overrides THERMAL_DEFAULT_TELEMETRY_ROW_COUNT for this run."
        ),
    )
    parser.add_argument(
        "--capture",
        default=None,
        metavar="ID",
        help=(
            "Capture id to give every dataset ingested in this run, to group related files "
            "(e.g. one single-camera file per camera). Datasets split from one multi-camera "
            "file are grouped automatically under a generated id when this is omitted."
        ),
    )
    parser.add_argument(
        "--name-from-path",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Name each dataset after its file's path instead of its file name: the N parent "
            "folder names plus the file name (no extension), joined with '_'. E.g. N=2 turns "
            ".../run1/take3/thermal_video.npz into 'run1_take3_thermal_video'."
        ),
    )
    parser.add_argument(
        "--move-ingested-to",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "After a file is ingested successfully, move it to DIR (keeping its path relative to "
            "the folder it was found under). Skipped and failed files are left in place."
        ),
    )
    parser.add_argument(
        "--move-duplicates",
        action="store_true",
        help="With --move-ingested-to, also move files that were already ingested (duplicates).",
    )
    args = parser.parse_args()
    if args.move_duplicates and args.move_ingested_to is None:
        parser.error("--move-duplicates requires --move-ingested-to")
    move_to = args.move_ingested_to.resolve() if args.move_ingested_to is not None else None

    settings = get_settings()
    telemetry_row_count = (
        args.telemetry_rows if args.telemetry_rows is not None else settings.default_telemetry_row_count
    )
    storage = FilesystemDatasetStorage(settings.data_root)
    engine = make_engine(settings.catalog_db)
    init_db(engine)

    exit_code = 0
    with get_session(engine) as session:
        for path, base in expand_paths(args.paths, exclude_dir=move_to):
            try:
                datasets = ingest_dataset(
                    path,
                    storage,
                    session,
                    default_telemetry_row_count=telemetry_row_count,
                    zarr_target_chunk_bytes=settings.zarr_target_chunk_bytes,
                    stats_sample_frames=settings.stats_sample_frames,
                    capture_id=args.capture,
                    name=name_from_path(path, args.name_from_path) if args.name_from_path is not None else None,
                )
            except IngestValidationError as e:
                print(f"FAILED  {path}:", file=sys.stderr)
                for err in e.errors:
                    print(f"    - {err}", file=sys.stderr)
                exit_code = 1
                continue
            except DuplicateDatasetError as e:
                print(f"DUPLICATE  {path}:", file=sys.stderr)
                print(f"    already ingested as dataset {e.existing_dataset_id}", file=sys.stderr)
                print(f"    name:     {e.existing_name}", file=sys.stderr)
                print(f"    source:   {e.existing_source_filename}", file=sys.stderr)
                print(f"    ingested: {e.existing_ingested_at.isoformat()}", file=sys.stderr)
                print(f"    sha256:   {e.source_sha256}", file=sys.stderr)
                # Not a failure: re-running over a folder is expected to hit these.
                if move_to is not None and args.move_duplicates and not move_source(path, base, move_to):
                    exit_code = 1
                continue
            except (KeyRegistryError, UnsupportedSourceFormatError) as e:
                print(f"SKIPPED  {path}: {e}", file=sys.stderr)
                exit_code = 1
                continue
            except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort a batch run
                print(f"FAILED  {path}: {type(e).__name__}: {e}", file=sys.stderr)
                exit_code = 1
                continue
            for metadata in datasets:
                capture = f", capture {metadata.user.capture_id}" if metadata.user.capture_id else ""
                print(
                    f"OK      {path} -> dataset {metadata.auto.dataset_id} ({metadata.user.name}{capture})",
                    flush=True,
                )
            if move_to is not None and not move_source(path, base, move_to):
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
