#!/usr/bin/env python
"""Delete cached preview frames (previews/camera_<n>/frames/) for every dataset,
keeping the thumbnails. Preview frames are re-rendered on demand when viewed,
so this is always safe.

Usage:
    python backend/scripts/clear_previews_cli.py            # clear
    python backend/scripts/clear_previews_cli.py --dry-run  # just report
"""

import argparse
import shutil

from thermal_dataset_manager.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete cached preview frames, keeping thumbnails.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted.")
    args = parser.parse_args()

    datasets_dir = get_settings().data_root / "datasets"
    dirs = sorted(datasets_dir.glob("*/previews/camera_*/frames"))
    files = size = 0
    for d in dirs:
        for f in d.rglob("*"):
            if f.is_file():
                files += 1
                size += f.stat().st_size
        if not args.dry_run:
            shutil.rmtree(d)

    verb = "would delete" if args.dry_run else "deleted"
    print(f"OK      {verb} {files} preview frame(s), {size / 1e6:.1f} MB, from {len(dirs)} camera(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
