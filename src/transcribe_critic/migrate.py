"""Migration utility for renaming legacy whisper_* files to unified asr_* naming.

Usage:
    transcribe-critic-migrate <directory> [--dry-run] [--recursive]
"""

import argparse
import sys
from pathlib import Path

from transcribe_critic.shared import LEGACY_WHISPER_MERGED_TXT, ASR_MERGED_TXT


def migrate_directory(directory: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Rename legacy whisper_*.txt/.json files to asr_* naming in a single directory.

    Returns list of (old_path, new_path) for files renamed (or that would be renamed
    in dry-run mode).
    """
    renames = []

    # Rename whisper_{model}.txt/.json → asr_{model}.txt/.json
    for pattern in ("whisper_*.txt", "whisper_*.json"):
        for old in sorted(directory.glob(pattern)):
            stem = old.stem  # e.g. "whisper_medium"
            if "merged" in stem:
                continue  # handled separately
            model = stem.removeprefix("whisper_")
            new = old.parent / f"asr_{model}{old.suffix}"
            if new.exists():
                print(f"  SKIP {old.name} → {new.name} (target already exists)")
                continue
            renames.append((old, new))

    # Rename whisper_merged.txt → asr_merged.txt
    legacy_merged = directory / LEGACY_WHISPER_MERGED_TXT
    if legacy_merged.exists():
        new_merged = directory / ASR_MERGED_TXT
        if new_merged.exists():
            print(f"  SKIP {legacy_merged.name} → {new_merged.name} (target already exists)")
        else:
            renames.append((legacy_merged, new_merged))

    # Execute renames (or just report in dry-run)
    for old, new in renames:
        if dry_run:
            print(f"  [dry-run] {old.name} → {new.name}")
        else:
            old.rename(new)
            print(f"  {old.name} → {new.name}")

    return renames


def main():
    parser = argparse.ArgumentParser(
        prog="transcribe-critic-migrate",
        description="Rename legacy whisper_* transcript files to unified asr_* naming.",
    )
    parser.add_argument("directory", type=Path,
                        help="Output directory (or parent of multiple output dirs with --recursive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be renamed without doing it")
    parser.add_argument("--recursive", action="store_true",
                        help="Process all subdirectories, not just the given directory")

    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    if args.recursive:
        dirs = sorted(d for d in args.directory.iterdir() if d.is_dir())
    else:
        dirs = [args.directory]

    total = 0
    for d in dirs:
        # Only process directories that have whisper_* files
        whisper_files = list(d.glob("whisper_*.txt")) + list(d.glob("whisper_*.json"))
        if not whisper_files:
            continue
        print(f"{d.name}/")
        renames = migrate_directory(d, dry_run=args.dry_run)
        total += len(renames)

    if total == 0:
        print("No legacy whisper_* files found.")
    else:
        verb = "would rename" if args.dry_run else "renamed"
        print(f"\n{verb.capitalize()} {total} file(s).")


if __name__ == "__main__":
    main()
