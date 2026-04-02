"""Migration utility for renaming legacy whisper_* files to unified asr_* naming.

Usage:
    transcribe-critic-migrate <directory> [--dry-run] [--recursive]
"""

import argparse
import sys
from pathlib import Path

from transcribe_critic.shared import ASR_MERGED_TXT

# Legacy filename constant — only needed by this migration tool
LEGACY_WHISPER_MERGED_TXT = "whisper_merged.txt"


def _plan_rename(old: Path, new: Path, renames: list) -> None:
    """Add a rename pair, or print SKIP if target exists."""
    if new.exists():
        print(f"  SKIP {old.name} → {new.name} (target already exists)")
    else:
        renames.append((old, new))


def migrate_directory(directory: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Rename legacy whisper_*.txt/.json files to asr_* naming in a single directory.

    Returns list of (old_path, new_path) for files renamed (or that would be renamed
    in dry-run mode).
    """
    renames = []

    # Rename whisper_{model}.txt/.json → asr_{model}.txt/.json
    for txt in sorted(directory.glob("whisper_*.txt")):
        model = txt.stem.removeprefix("whisper_")
        if "merged" in model:
            continue  # handled separately
        _plan_rename(txt, directory / f"asr_{model}.txt", renames)
        legacy_json = directory / f"whisper_{model}.json"
        if legacy_json.exists():
            _plan_rename(legacy_json, directory / f"asr_{model}.json", renames)

    # Rename whisper_merged.txt → asr_merged.txt
    legacy_merged = directory / LEGACY_WHISPER_MERGED_TXT
    if legacy_merged.exists():
        _plan_rename(legacy_merged, directory / ASR_MERGED_TXT, renames)

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
