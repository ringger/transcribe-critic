#!/usr/bin/env python3
"""Prototype: confidence-weighted ensemble selection.

At each disagreement between ASR models, look up per-word confidence
scores from the available JSON outputs. Low confidence at a disagreement
position → trust the other model's reading.

Currently uses:
- Whisper per-word `probability` (0-1, already in asr_*.json)
- Parakeet per-token `confidence` (0-1, requires re-transcription to capture)

Usage:
    python scripts/prototype_confidence.py <run_dir> [--threshold 0.9] [--max-diffs 20]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transcribe_critic.shared import ALL_MODELS, get_model_quality_rank
from transcribe_critic.transcription import (
    _apply_resolutions,
    _filter_trivial_diffs,
    _merge_pairwise_diffs,
    _parse_wdiff_diffs,
)


def load_word_confidence(json_path: Path) -> list[dict]:
    """Load per-word confidence from an asr_*.json file.

    Handles both Whisper format (word.probability) and merged subword tokens.
    Returns list of {word, start, end, confidence} dicts.
    """
    with open(json_path) as f:
        data = json.load(f)

    words = []
    for seg in data.get("segments", []):
        seg_logprob = seg.get("avg_logprob", None)

        if seg.get("words"):
            for w in seg["words"]:
                raw = w.get("word", "")
                if not raw:
                    continue
                conf = w.get("probability", w.get("confidence", None))

                if raw.startswith(" ") or not words:
                    words.append({
                        "word": raw.strip(),
                        "start": w["start"],
                        "end": w["end"],
                        "confidence": conf,
                        "seg_logprob": seg_logprob,
                    })
                else:
                    # Merge subword token
                    words[-1]["word"] += raw
                    words[-1]["end"] = w["end"]
                    # Use minimum confidence across subwords
                    if conf is not None and words[-1]["confidence"] is not None:
                        words[-1]["confidence"] = min(words[-1]["confidence"], conf)
                    words[-1]["seg_logprob"] = seg_logprob
        else:
            # No word-level timestamps — use segment level
            seg_words = seg.get("text", "").split()
            for sw in seg_words:
                words.append({
                    "word": sw,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "confidence": None,
                    "seg_logprob": seg_logprob,
                })
    return words


def normalize(text: str) -> str:
    """Normalize for comparison."""
    import re
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def align_word_positions(model_text: str, word_index: list[dict]) -> dict[int, int]:
    """Map word positions in model_text to indices in word_index.

    Returns dict: text_word_pos → word_index_pos.
    Uses sequential matching with normalized comparison.
    """
    text_words = model_text.split()
    mapping = {}
    idx_pos = 0
    for tw_pos, tw in enumerate(text_words):
        tw_norm = normalize(tw)
        # Find next matching word in index
        while idx_pos < len(word_index):
            iw_norm = normalize(word_index[idx_pos]["word"])
            if tw_norm == iw_norm or iw_norm.startswith(tw_norm) or tw_norm.startswith(iw_norm):
                mapping[tw_pos] = idx_pos
                idx_pos += 1
                break
            idx_pos += 1
    return mapping


def get_confidence_at_position(word_index: list[dict], pos_map: dict[int, int],
                                word_pos: int, word_len: int) -> float | None:
    """Get minimum confidence across words at the given position range."""
    confidences = []
    for wp in range(word_pos, word_pos + max(word_len, 1)):
        idx = pos_map.get(wp)
        if idx is not None and idx < len(word_index):
            conf = word_index[idx]["confidence"]
            if conf is not None:
                confidences.append(conf)
    return min(confidences) if confidences else None


def main():
    parser = argparse.ArgumentParser(
        description="Confidence-weighted ensemble selection"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Confidence threshold below which we distrust a reading (default: 0.9)")
    parser.add_argument("--max-diffs", type=int, default=0,
                        help="Max diffs to process (0=all)")

    args = parser.parse_args()
    run_dir = args.run_dir
    threshold = args.threshold

    # Discover models with JSON
    models = {}
    for txt_path in sorted(run_dir.glob("asr_*.txt")):
        name = txt_path.stem.replace("asr_", "")
        if name in ("merged", "realigned"):
            continue
        json_path = run_dir / f"asr_{name}.json"
        if json_path.exists() and name in ALL_MODELS:
            models[name] = {
                "txt": txt_path,
                "json": json_path,
                "text": txt_path.read_text().strip(),
            }

    if len(models) < 2:
        print(f"Need at least 2 models, found {len(models)}")
        sys.exit(1)

    base_model = max(models.keys(), key=get_model_quality_rank)
    other_models = [m for m in models if m != base_model]
    print(f"Models: {', '.join(models.keys())}  (base: {base_model})")

    # Load confidence indexes
    print("\nLoading confidence data...")
    conf_indexes = {}
    conf_available = {}
    for name, info in models.items():
        idx = load_word_confidence(info["json"])
        conf_indexes[name] = idx
        has_conf = sum(1 for w in idx if w["confidence"] is not None)
        conf_available[name] = has_conf > 0
        print(f"  {name}: {len(idx)} words, "
              f"{'confidence available' if has_conf else 'NO confidence data'} "
              f"({has_conf}/{len(idx)} words)")

    # Build position mappings
    pos_maps = {}
    for name, info in models.items():
        pos_maps[name] = align_word_positions(info["text"], conf_indexes[name])

    # Find disagreements
    class _Cfg:
        verbose = False
        output_dir = run_dir

    pairwise = []
    for om in other_models:
        diffs = _parse_wdiff_diffs(models[om]["text"], models[base_model]["text"], _Cfg())
        diffs = _filter_trivial_diffs(diffs)
        pairwise.append((om, diffs))

    if len(other_models) > 1:
        all_diffs = _merge_pairwise_diffs(pairwise, base_model, list(models.keys()))
    else:
        all_diffs = pairwise[0][1] if pairwise else []
    all_diffs.sort(key=lambda d: d["b_pos"])

    total = len(all_diffs)
    if args.max_diffs and total > args.max_diffs:
        all_diffs = all_diffs[:args.max_diffs]
    print(f"\n{total} disagreements, processing {len(all_diffs)}")
    print(f"Threshold: {threshold}")
    print("=" * 80)

    # Process each diff
    results = []
    kept = 0
    changed = 0
    conf_decided = 0

    for i, diff in enumerate(all_diffs):
        b_pos = diff["b_pos"]
        b_len = max(diff["b_len"], 1)

        if "readings" in diff:
            readings = diff["readings"]
        else:
            readings = {
                diff.get("other_model", other_models[0]): diff["a_text"] or "",
                base_model: diff["b_text"] or "",
            }

        base_reading = readings[base_model]

        # Get confidence for each model's reading at this position
        model_conf = {}
        for model_name, reading_text in readings.items():
            if not reading_text:
                model_conf[model_name] = None
                continue
            # For the base model, use b_pos directly
            if model_name == base_model:
                conf = get_confidence_at_position(
                    conf_indexes[model_name], pos_maps[model_name],
                    b_pos, b_len)
            else:
                # For other models, use a_pos
                a_pos = diff.get("a_pos", b_pos)
                a_len = diff.get("a_len", b_len)
                conf = get_confidence_at_position(
                    conf_indexes[model_name], pos_maps[model_name],
                    a_pos, a_len)
            model_conf[model_name] = conf

        # Decision logic — use all available confidence signals:
        # 1. If base has high confidence → keep base
        # 2. If base has low confidence AND alt has high → change
        # 3. If base has no confidence but alt has high AND multiple models
        #    agree on the alt reading → change
        # 4. Otherwise → keep base
        base_conf = model_conf.get(base_model)
        chosen_text = base_reading
        reason = "no confidence data"

        # Find best alternative: highest confidence among non-base models
        best_alt = None
        best_alt_conf = None
        for m, conf in model_conf.items():
            if m == base_model:
                continue
            if conf is not None and (best_alt_conf is None or conf > best_alt_conf):
                best_alt = m
                best_alt_conf = conf

        # Count how many models share each distinct reading
        base_norm = normalize(base_reading)
        alt_reading = readings.get(best_alt, "") if best_alt else ""
        alt_count = sum(1 for m, r in readings.items()
                        if normalize(r) == normalize(alt_reading)) if alt_reading else 0
        base_count = sum(1 for m, r in readings.items()
                         if normalize(r) == base_norm)

        if base_conf is not None and base_conf >= threshold:
            reason = f"base conf={base_conf:.3f} (high)"
        elif base_conf is not None and base_conf < threshold:
            if best_alt and best_alt_conf is not None and best_alt_conf >= threshold:
                chosen_text = alt_reading
                reason = f"base conf={base_conf:.3f} LOW, {best_alt} conf={best_alt_conf:.3f} HIGH"
                conf_decided += 1
            elif best_alt and best_alt_conf is not None and best_alt_conf > base_conf:
                chosen_text = alt_reading
                reason = f"base conf={base_conf:.3f}, {best_alt} conf={best_alt_conf:.3f} (higher)"
                conf_decided += 1
            else:
                reason = f"base conf={base_conf:.3f} low, no better alternative"
        elif base_conf is None and best_alt_conf is not None:
            # No base confidence — use alt confidence + model count
            if best_alt_conf >= threshold and alt_count > base_count:
                chosen_text = alt_reading
                reason = f"base conf=N/A, {best_alt} conf={best_alt_conf:.3f} HIGH + majority ({alt_count} vs {base_count})"
                conf_decided += 1
            elif best_alt_conf >= threshold:
                reason = f"base conf=N/A, {best_alt} conf={best_alt_conf:.3f} high but base has majority ({base_count} vs {alt_count})"
            else:
                reason = f"base conf=N/A, {best_alt} conf={best_alt_conf:.3f} (not high enough)"

        is_changed = normalize(chosen_text) != normalize(base_reading)
        if is_changed:
            changed += 1
        else:
            kept += 1

        action = "CHANGE →" if is_changed else "keep"
        confs_str = ", ".join(f"{m}={c:.3f}" if c else f"{m}=N/A"
                              for m, c in model_conf.items())
        print(f"\n[{i+1:4d}] word {b_pos}: {action} \"{chosen_text}\"")
        print(f"       {confs_str}")
        print(f"       {reason}")

        results.append({
            "b_pos": b_pos,
            "base_reading": base_reading,
            "chosen_text": chosen_text,
            "changed": is_changed,
            "confidences": {m: c for m, c in model_conf.items()},
            "reason": reason,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Kept base reading:            {kept}/{len(results)}")
    print(f"Changed to alt reading:       {changed}/{len(results)}")
    print(f"Decided by confidence:        {conf_decided}/{len(results)}")

    # Build realigned transcript
    base_words = models[base_model]["text"].split()
    resolutions = {}
    for r, diff in zip(results, all_diffs):
        text = r["chosen_text"]
        if not text:
            text = "(omit)"
        resolutions[id(diff)] = text

    realigned_text = _apply_resolutions(base_words, all_diffs, resolutions)
    realigned_path = run_dir / "asr_conf_realigned.txt"
    realigned_path.write_text(realigned_text)
    print(f"\nRealigned transcript: {realigned_path}")
    print(f"  ({len(realigned_text.split())} words, base had {len(base_words)})")

    # Save results
    output_path = run_dir / "confidence_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON: {output_path}")

    print(f"\nTo score: transcribe-critic-eval score <manifest.json> --run-dir {run_dir.parent}")


if __name__ == "__main__":
    main()
