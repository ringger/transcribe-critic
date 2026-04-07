#!/usr/bin/env python3
"""Prototype: re-recognize audio segments around ensemble disagreements.

Given an existing ensemble run directory (with asr_*.txt and asr_*.json files),
this script:
  1. Detects disagreements between ASR transcripts (reusing pipeline diffing)
  2. Maps each disagreement to a time span via word-level timestamps
  3. Extracts broad audio snippets around each disagreement region
  4. Re-transcribes each snippet with the available models
  5. Compares re-transcription results to the original readings

The idea: if a model reproduces the same text on a focused, context-rich segment,
that's higher confidence. Whisper's initial_prompt parameter lets us prime the
decoder with surrounding agreed-upon text.

Usage:
    python scripts/prototype_realign.py <run_dir> [--context-secs 15] [--dry-run] [--max-diffs 5]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add src to path so we can reuse pipeline code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transcribe_critic.shared import ALL_MODELS, get_model_quality_rank
from transcribe_critic.transcription import (
    _apply_resolutions,
    _filter_trivial_diffs,
    _merge_pairwise_diffs,
    _normalize_for_comparison,
    _parse_wdiff_diffs,
)


# ---------------------------------------------------------------------------
# Word-position → timestamp mapping
# ---------------------------------------------------------------------------

def build_word_timestamp_index(json_path: Path) -> list[dict]:
    """Build a flat list of {word, start, end} from an asr_*.json file.

    Returns one entry per *whole word* in transcript order.  Subword tokens
    (e.g., parakeet's character-level timestamps) are merged: tokens that
    don't start with a space are joined to the preceding token.  If the JSON
    has no word-level timestamps, falls back to segment-level (assigns
    segment time span to each word in the segment).
    """
    with open(json_path) as f:
        data = json.load(f)

    words = []
    for seg in data.get("segments", []):
        if seg.get("words"):
            # Merge subword tokens into whole words.
            # Convention: a token starting with a space begins a new word.
            for w in seg["words"]:
                raw = w.get("word", "")
                if not raw:
                    continue
                if raw.startswith(" ") or not words:
                    # New word
                    words.append({
                        "word": raw.strip(),
                        "start": w["start"],
                        "end": w["end"],
                    })
                else:
                    # Continuation of previous word
                    words[-1]["word"] += raw
                    words[-1]["end"] = w["end"]
        else:
            # Fallback: spread segment time evenly across words
            seg_words = seg.get("text", "").split()
            if not seg_words:
                continue
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            dur = (seg_end - seg_start) / max(len(seg_words), 1)
            for i, sw in enumerate(seg_words):
                words.append({
                    "word": sw,
                    "start": seg_start + i * dur,
                    "end": seg_start + (i + 1) * dur,
                })
    return words


def word_pos_to_time_span(word_index: list[dict], pos: int, length: int,
                          context_secs: float) -> tuple[float, float]:
    """Map a word position range to an audio time span with context padding.

    Returns (start_sec, end_sec) with context_secs padding on each side,
    clamped to [0, last_word_end].
    """
    if not word_index:
        return 0.0, 0.0

    # Clamp to valid range
    pos = max(0, min(pos, len(word_index) - 1))
    end_pos = max(0, min(pos + max(length, 1) - 1, len(word_index) - 1))

    start = word_index[pos]["start"]
    end = word_index[end_pos]["end"]

    # Add context padding
    start = max(0.0, start - context_secs)
    end = min(word_index[-1]["end"], end + context_secs)

    return start, end


# ---------------------------------------------------------------------------
# Audio segment extraction
# ---------------------------------------------------------------------------

def extract_audio_segment(audio_path: Path, start_sec: float, end_sec: float,
                          output_path: Path) -> None:
    """Extract an audio segment using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ss", str(start_sec), "-to", str(end_sec),
         "-ar", "16000", "-ac", "1", str(output_path)],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# Re-transcription (batched: load model once, transcribe many segments)
# ---------------------------------------------------------------------------

WHISPER_MODEL_MAP = {
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "large": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}


def load_model(model_name: str):
    """Load an ASR model, returning a (backend, model_object) tuple."""
    entry = ALL_MODELS.get(model_name)
    if not entry:
        raise ValueError(f"Unknown model: {model_name}")

    backend = entry["backend"]
    if backend == "whisper":
        # mlx_whisper doesn't have a persistent model object — it loads
        # from hf_repo each call but caches internally. Return the repo id.
        hf_id = WHISPER_MODEL_MAP.get(model_name,
                                       f"mlx-community/whisper-{model_name}-mlx")
        return backend, hf_id
    elif backend == "parakeet_mlx":
        from parakeet_mlx import from_pretrained
        return backend, from_pretrained(entry["hf_id"])
    elif backend == "mlx_audio":
        from mlx_audio.stt import load
        return backend, load(entry["hf_id"])
    raise ValueError(f"Unsupported backend: {backend}")


def transcribe_segment(backend: str, model_obj, segment_path: Path,
                       prompt_context: str = "") -> str:
    """Transcribe a single segment using a pre-loaded model."""
    if backend == "whisper":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            str(segment_path),
            path_or_hf_repo=model_obj,  # hf_id string
            initial_prompt=prompt_context or None,
            word_timestamps=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.2,
            compression_ratio_threshold=2.0,
            hallucination_silence_threshold=3.0,
        )
        return result.get("text", "").strip()
    elif backend == "parakeet_mlx":
        result = model_obj.transcribe(str(segment_path))
        return result.text.strip()
    elif backend == "mlx_audio":
        result = model_obj.generate(str(segment_path))
        return result.text.strip()
    return ""


def batch_retranscribe(model_name: str, segments: list[dict]) -> dict[int, str]:
    """Load a model once and re-transcribe all segments.

    Args:
        model_name: Name of the ASR model.
        segments: List of dicts with keys: "index", "path", "prompt_context".

    Returns:
        dict mapping segment index → transcribed text.
    """
    if not segments:
        return {}

    print(f"\n  Loading {model_name}...")
    try:
        backend, model_obj = load_model(model_name)
    except Exception as e:
        print(f"  FAILED to load {model_name}: {e}")
        return {s["index"]: None for s in segments}

    results = {}
    for seg in segments:
        try:
            prompt = seg.get("prompt_context", "") if backend == "whisper" else ""
            text = transcribe_segment(backend, model_obj, seg["path"], prompt)
            results[seg["index"]] = text
        except Exception as e:
            print(f"  Segment {seg['index']} FAILED: {e}")
            results[seg["index"]] = None
    print(f"  {model_name}: transcribed {sum(1 for v in results.values() if v is not None)}/{len(segments)} segments")
    return results


# ---------------------------------------------------------------------------
# Context extraction (for Whisper initial_prompt)
# ---------------------------------------------------------------------------

def get_surrounding_context(word_index: list[dict], pos: int, length: int,
                            context_words: int = 50) -> tuple[str, str]:
    """Extract agreed-upon text before and after a disagreement region.

    Returns (prefix_context, suffix_context) — the text surrounding the
    disagreement that can be used as Whisper's initial_prompt.
    """
    # Words before the diff
    pre_start = max(0, pos - context_words)
    prefix = " ".join(w["word"] for w in word_index[pre_start:pos])

    # Words after the diff
    post_end = min(len(word_index), pos + length + context_words)
    suffix = " ".join(w["word"] for w in word_index[pos + length:post_end])

    return prefix, suffix


# ---------------------------------------------------------------------------
# Fuzzy matching for comparing re-transcription to original readings
# ---------------------------------------------------------------------------

def normalize_for_match(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def word_overlap_score(candidate: str, reference: str) -> float:
    """Compute word-level overlap between candidate and reference.

    Returns fraction of reference words found in candidate (recall-oriented).
    """
    if not reference.strip():
        return 1.0 if not candidate.strip() else 0.0

    ref_words = normalize_for_match(reference).split()
    cand_words = normalize_for_match(candidate).split()

    if not ref_words:
        return 1.0

    matches = sum(1 for w in ref_words if w in cand_words)
    return matches / len(ref_words)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def find_available_models(run_dir: Path) -> dict[str, dict]:
    """Discover which models have both txt and json outputs in the run dir."""
    models = {}
    for txt_path in sorted(run_dir.glob("asr_*.txt")):
        name = txt_path.stem.replace("asr_", "")
        if name == "merged":
            continue
        json_path = run_dir / f"asr_{name}.json"
        if json_path.exists() and name in ALL_MODELS:
            models[name] = {
                "txt": txt_path,
                "json": json_path,
                "text": txt_path.read_text().strip(),
            }
    return models




def run_realignment(run_dir: Path, audio_path: Path, context_secs: float,
                    max_diffs: int, dry_run: bool) -> None:
    """Main prototype workflow."""
    print(f"Run directory: {run_dir}")
    print(f"Audio: {audio_path}")
    print(f"Context padding: {context_secs}s each side")

    print()

    # 1. Discover available models
    models = find_available_models(run_dir)
    if len(models) < 2:
        print(f"Need at least 2 models, found {len(models)}: {list(models.keys())}")
        return

    print(f"Found {len(models)} models: {', '.join(models.keys())}")

    # 2. Determine base model (highest quality rank)
    base_model = max(models.keys(), key=get_model_quality_rank)
    other_models = [m for m in models if m != base_model]
    print(f"Base model: {base_model}")
    print()

    # 3. Build word-timestamp index for base model
    base_word_index = build_word_timestamp_index(models[base_model]["json"])
    print(f"Base model has {len(base_word_index)} timestamped words")

    # Also build indexes for other models (for their readings)
    model_word_indexes = {base_model: base_word_index}
    for m in other_models:
        model_word_indexes[m] = build_word_timestamp_index(models[m]["json"])

    # 4. Find disagreements (reuse pipeline diffing)
    # Build a lightweight SpeechConfig stand-in
    class _MinimalConfig:
        verbose = False
        output_dir = run_dir

    config = _MinimalConfig()

    pairwise_diffs = []
    for other_model in other_models:
        diffs = _parse_wdiff_diffs(
            models[other_model]["text"],  # text_a
            models[base_model]["text"],   # text_b
            config,
        )
        diffs = _filter_trivial_diffs(diffs)
        pairwise_diffs.append((other_model, diffs))
        print(f"  {other_model} vs {base_model}: {len(diffs)} meaningful diffs")

    # Merge into multi-way diffs (deduplicates same-position diffs)
    if len(other_models) > 1:
        all_diffs = _merge_pairwise_diffs(
            pairwise_diffs, base_model, list(models.keys()),
        )
        pairwise_total = sum(len(d) for _, d in pairwise_diffs)
        print(f"  Merged {pairwise_total} pairwise diffs into {len(all_diffs)} multi-way diffs")
    else:
        all_diffs = pairwise_diffs[0][1] if pairwise_diffs else []

    if not all_diffs:
        print("\nNo disagreements found!")
        return

    # Sort by position
    all_diffs.sort(key=lambda d: d["b_pos"])
    if max_diffs and len(all_diffs) > max_diffs:
        print(f"\nLimiting to first {max_diffs} of {len(all_diffs)} diffs")
        all_diffs = all_diffs[:max_diffs]

    print(f"\nProcessing {len(all_diffs)} disagreements...")
    print("=" * 80)

    # --- Phase 1: Prepare all diffs (readings, time spans, audio segments) ---
    tmp_dir = Path(tempfile.mkdtemp(prefix="realign_"))
    diff_info = []  # per-diff metadata
    # model_name → list of {"index": i, "path": Path, "prompt_context": str}
    model_segments = {}

    for i, diff in enumerate(all_diffs):
        b_pos = diff["b_pos"]
        b_len = max(diff["b_len"], 1)

        if "readings" in diff:
            readings = diff["readings"]
        else:
            readings = {
                diff.get("other_model", other_models[0]): diff["a_text"] or "(omit)",
                base_model: diff["b_text"] or "(omit)",
            }

        base_reading = readings[base_model]
        disagreeing = {m: r for m, r in readings.items()
                       if m != base_model and normalize_for_match(r) != normalize_for_match(base_reading)}
        involved_models = list(disagreeing.keys()) + [base_model]

        start_sec, end_sec = word_pos_to_time_span(
            base_word_index, b_pos, b_len, context_secs
        )

        print(f"\n--- Diff {i+1}/{len(all_diffs)} at word {b_pos} ({diff['type']}) ---")
        for m in involved_models:
            print(f"  {m:>20s}: \"{readings[m]}\"")
        print(f"  Audio span: {start_sec:.1f}s – {end_sec:.1f}s ({end_sec - start_sec:.1f}s)")

        if dry_run:
            prefix, suffix = get_surrounding_context(
                base_word_index, b_pos, b_len
            )
            if prefix:
                print(f"  Context before: ...{prefix[-80:]}")
            if suffix:
                print(f"  Context after:  {suffix[:80]}...")

        info = {
            "index": i, "b_pos": b_pos, "b_len": b_len,
            "readings": readings, "base_reading": base_reading,
            "involved_models": involved_models,
            "time_span": (start_sec, end_sec),
        }

        if not dry_run:
            segment_path = tmp_dir / f"segment_{i:03d}.wav"
            extract_audio_segment(audio_path, start_sec, end_sec, segment_path)
            info["segment_path"] = segment_path

            prefix, _ = get_surrounding_context(
                base_word_index, b_pos, b_len
            )
            whisper_prompt = prefix[-200:] if prefix else ""
            info["whisper_prompt"] = whisper_prompt

            for model in involved_models:
                model_segments.setdefault(model, []).append({
                    "index": i,
                    "path": segment_path,
                    "prompt_context": whisper_prompt,
                })

        diff_info.append(info)

    if dry_run:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Dry run complete. {len(all_diffs)} diffs would be re-transcribed.")
        unique_models = set()
        for di in diff_info:
            unique_models.update(di["involved_models"])
        print(f"Models to load: {len(unique_models)} ({', '.join(sorted(unique_models))})")
        # Cleanup empty tmp dir
        tmp_dir.rmdir()
        return

    # --- Phase 2: Batch re-transcription (load each model once) ---
    print(f"\n{'=' * 80}")
    print(f"BATCH RE-TRANSCRIPTION ({len(model_segments)} models)")
    print("=" * 80)

    # re_results[diff_index][model_name] = transcribed text
    all_re_results = {}
    for model_name in sorted(model_segments.keys()):
        segs = model_segments[model_name]
        print(f"\n  {model_name}: {len(segs)} segments")
        batch_results = batch_retranscribe(model_name, segs)
        for idx, text in batch_results.items():
            all_re_results.setdefault(idx, {})[model_name] = text

    # --- Phase 3: Score each diff ---
    print(f"\n{'=' * 80}")
    print("SCORING")
    print("=" * 80)

    results = []
    for info in diff_info:
        i = info["index"]
        readings = info["readings"]
        base_reading = info["base_reading"]
        re_results = all_re_results.get(i, {})

        print(f"\n--- Diff {i+1}/{len(all_diffs)} at word {info['b_pos']} ---")
        for m in info["involved_models"]:
            re_text = re_results.get(m)
            if re_text is not None:
                print(f"  {m:>20s}: orig=\"{readings[m]}\"  re=\"{re_text[:80]}{'...' if len(re_text) > 80 else ''}\"")
            else:
                print(f"  {m:>20s}: orig=\"{readings[m]}\"  re=FAILED")

        # Score by distinct text readings, not by model.
        # Group models that produced the same reading, then count how many
        # re-transcriptions support each distinct reading.
        # Deduplicate readings by normalized text
        distinct_readings = {}  # norm_text → {"text": original, "models": [model_names]}
        for model_name, reading_text in readings.items():
            norm = normalize_for_match(reading_text) if reading_text else ""
            if norm not in distinct_readings:
                distinct_readings[norm] = {"text": reading_text, "models": []}
            distinct_readings[norm]["models"].append(model_name)

        # For each distinct reading, find which re-transcriptions support it
        text_support = {}  # norm_text → {"text", "models", "supporters", "cross_confirmed"}
        for norm, info in distinct_readings.items():
            text = info["text"]
            if not text or text == "(omit)":
                text_support[norm] = {**info, "supporters": [], "cross_confirmed": False}
                continue
            supporters = []
            for re_model, re_text in re_results.items():
                if re_text is None:
                    continue
                if word_overlap_score(re_text, text) >= 0.8:
                    supporters.append(re_model)
            # Cross-confirmed = supported by a model that originally had a different reading
            cross = [s for s in supporters if s not in info["models"]]
            text_support[norm] = {
                **info, "supporters": supporters,
                "cross_confirmed": len(cross) > 0,
            }

        # Pick winning text:
        # 1. Cross-confirmed readings (strongest signal)
        # 2. Total supporter count (re-transcription evidence)
        # 3. Original model count (majority vote)
        # 4. Tie-break: prefer base reading
        base_norm = normalize_for_match(base_reading)
        candidates = sorted(
            text_support.values(),
            key=lambda x: (
                x["cross_confirmed"],       # cross-confirmation first
                len(x["supporters"]),        # then total re-transcription support
                len(x["models"]),            # then original model count
                base_norm == normalize_for_match(x["text"]),  # tie-break: prefer base
            ),
            reverse=True,
        )
        chosen_text = candidates[0]["text"] if candidates else base_reading
        changed = normalize_for_match(chosen_text) != base_norm

        # Report
        for norm, info in text_support.items():
            models_str = "+".join(info["models"])
            cross = [s for s in info["supporters"] if s not in info["models"]]
            support_str = f"supported by {info['supporters']}" if info["supporters"] else "no support"
            if cross:
                print(f"  ★ \"{info['text']}\" ({models_str}): {support_str} — cross-confirmed by {cross}")
            elif info["supporters"]:
                print(f"    \"{info['text']}\" ({models_str}): {support_str} (self only)")
            else:
                print(f"    \"{info['text']}\" ({models_str}): {support_str}")

        action = "CHANGE →" if changed else "keep"
        print(f"  → {action} \"{chosen_text}\"")

        # Map back to a "winner" model for compatibility (pick first model with this text)
        winner = next(
            (m for m, r in readings.items()
             if normalize_for_match(r) == normalize_for_match(chosen_text)),
            base_model,
        )

        results.append({
            "diff_index": i,
            "b_pos": b_pos,
            "type": diff["type"],
            "base_model": base_model,
            "base_reading": base_reading,
            "chosen_text": chosen_text,
            "changed": changed,
            "cross_confirmed": candidates[0]["cross_confirmed"] if candidates else False,
            "readings": readings,
            "time_span": (start_sec, end_sec),
            "re_transcriptions": re_results,
            "winner": winner,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    kept = sum(1 for r in results if not r.get("changed", False))
    changed = sum(1 for r in results if r.get("changed", False))
    cross_confirmed = sum(1 for r in results if r.get("cross_confirmed", False))
    print(f"Kept base reading:                   {kept}/{len(results)}")
    print(f"Changed to different reading:        {changed}/{len(results)}")
    print(f"Cross-model confirmed:               {cross_confirmed}/{len(results)}")

    # Build realigned transcript by applying chosen readings to base text
    base_words = models[base_model]["text"].split()
    resolutions = {}
    for r, diff in zip(results, all_diffs):
        text = r["chosen_text"]
        if not text:
            text = "(omit)"
        resolutions[id(diff)] = text

    realigned_text = _apply_resolutions(base_words, all_diffs, resolutions)
    realigned_path = run_dir / "asr_realigned.txt"
    realigned_path.write_text(realigned_text)
    print(f"\nRealigned transcript: {realigned_path}")
    print(f"  ({len(realigned_text.split())} words, base had {len(base_words)})")

    # Save results JSON
    output_path = run_dir / "realignment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON: {output_path}")

    print(f"\nTo score with eval framework:")
    print(f"  transcribe-critic-eval score <manifest.json> --run-dir {run_dir.parent}")

    # Cleanup
    for p in tmp_dir.glob("*.wav"):
        p.unlink()
    tmp_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(
        description="Re-recognize audio segments around ensemble disagreements"
    )
    parser.add_argument("run_dir", type=Path,
                        help="Ensemble run directory with asr_*.txt and asr_*.json files")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Path to source audio (default: auto-detect from run_dir)")
    parser.add_argument("--context-secs", type=float, default=15.0,
                        help="Seconds of audio context on each side of disagreement (default: 15)")
    parser.add_argument("--max-diffs", type=int, default=10,
                        help="Maximum number of diffs to process (default: 10, 0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without re-transcribing")

    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory")
        sys.exit(1)

    # Auto-detect audio file
    audio_path = args.audio
    if not audio_path:
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.webm"]:
            candidates = list(args.run_dir.glob(ext))
            if candidates:
                audio_path = candidates[0]
                break
    if not audio_path and not args.dry_run:
        print("Error: No audio file found. Use --audio to specify.")
        sys.exit(1)

    run_realignment(
        run_dir=args.run_dir,
        audio_path=audio_path,
        context_secs=args.context_secs,
        max_diffs=args.max_diffs or 0,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
