#!/usr/bin/env python3
"""Compare baseline (whole-text) vs sentence-aligned wdiff on a transcript pair.

Usage:
    python scripts/compare_alignment.py /path/to/output_dir [--pad-words N]

Expects asr_parakeet.{txt,json} and asr_distil-large-v3.{txt,json} in the
output directory (or whichever models are present).  Reports diffs side by
side and highlights differences between the two approaches.
"""

import argparse
import json
import sys
from pathlib import Path

from transcribe_critic.shared import SpeechConfig, get_model_quality_rank
from transcribe_critic.transcription import (
    _filter_trivial_diffs,
    _load_segments_with_offsets,
    _pairwise_diffs_by_sentence,
    _parse_wdiff_diffs,
)


def _discover_model_pair(output_dir: Path) -> tuple[str, str]:
    """Find the two models to compare (highest-ranked = base)."""
    models = []
    for txt in sorted(output_dir.glob("asr_*.txt")):
        name = txt.stem.removeprefix("asr_")
        json_path = output_dir / f"asr_{name}.json"
        if json_path.exists() and name != "merged":
            models.append(name)
    if len(models) < 2:
        print(f"Error: need at least 2 models with JSON, found: {models}", file=sys.stderr)
        sys.exit(1)
    models.sort(key=get_model_quality_rank, reverse=True)
    return models[0], models[1]  # base, other


def _fmt_diff(d: dict, max_width: int = 40) -> str:
    a = d["a_text"][:max_width] if d["a_text"] else "(empty)"
    b = d["b_text"][:max_width] if d["b_text"] else "(empty)"
    return f'b_pos={d["b_pos"]:>4} {d["type"]:12s}: "{a}" vs "{b}"'


def compare(output_dir: Path, pad_words: int = 5) -> dict:
    """Run comparison and return results dict."""
    config = SpeechConfig(url="local", output_dir=output_dir)

    base_model, other_model = _discover_model_pair(output_dir)
    base_text = (output_dir / f"asr_{base_model}.txt").read_text()
    other_text = (output_dir / f"asr_{other_model}.txt").read_text()
    base_json = output_dir / f"asr_{base_model}.json"
    other_json = output_dir / f"asr_{other_model}.json"
    base_segs = _load_segments_with_offsets(base_json)

    print(f"Models: base={base_model} ({len(base_text.split())} words), "
          f"other={other_model} ({len(other_text.split())} words)")
    print(f"Base segments: {len(base_segs)}, pad_words: {pad_words}")
    print()

    # Baseline
    bl = _filter_trivial_diffs(_parse_wdiff_diffs(other_text, base_text, config))
    print(f"=== BASELINE (whole-text): {len(bl)} diffs ===")
    for i, d in enumerate(bl):
        print(f"  [{i:3d}] {_fmt_diff(d)}")

    # Sentence-aligned
    sa = _filter_trivial_diffs(
        _pairwise_diffs_by_sentence(base_segs, other_json, config, pad_words=pad_words))
    print(f"\n=== SENTENCE-ALIGNED: {len(sa)} diffs ===")
    for i, d in enumerate(sa):
        print(f"  [{i:3d}] {_fmt_diff(d)}")

    # Compare
    bl_set = {(d["b_pos"], d["type"], d["a_text"], d["b_text"]) for d in bl}
    sa_set = {(d["b_pos"], d["type"], d["a_text"], d["b_text"]) for d in sa}
    common = bl_set & sa_set
    only_bl = sorted(bl_set - sa_set)
    only_sa = sorted(sa_set - bl_set)

    print(f"\n=== COMPARISON ===")
    print(f"Common: {len(common)}, Only baseline: {len(only_bl)}, "
          f"Only sentence-aligned: {len(only_sa)}")

    if only_bl:
        print("\nOnly in baseline:")
        for d in only_bl:
            print(f"  b_pos={d[0]:>4} {d[1]:12s}: \"{d[2][:40]}\" vs \"{d[3][:40]}\"")
    if only_sa:
        print("\nOnly in sentence-aligned:")
        for d in only_sa:
            print(f"  b_pos={d[0]:>4} {d[1]:12s}: \"{d[2][:40]}\" vs \"{d[3][:40]}\"")

    if not only_bl and not only_sa:
        print("\nResult: IDENTICAL diffs")

    return {
        "base_model": base_model,
        "other_model": other_model,
        "base_words": len(base_text.split()),
        "base_segments": len(base_segs),
        "baseline_diffs": len(bl),
        "sentence_aligned_diffs": len(sa),
        "common": len(common),
        "only_baseline": len(only_bl),
        "only_sentence_aligned": len(only_sa),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Directory with asr_*.txt and asr_*.json files")
    parser.add_argument("--pad-words", type=int, default=5,
                        help="Boundary padding words (default: 5)")
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: {args.output_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    compare(args.output_dir, pad_words=args.pad_words)


if __name__ == "__main__":
    main()
