#!/usr/bin/env python3
"""Re-apply saved Kev decisions under confidence thresholds, without new calls.

For each threshold t, a Kev decision overrides the base (parakeet) reading only
when Kev's confidence >= t; otherwise the base reading is kept. Writes
asr_merged_kev_t<t>.txt variants into a new run dir (with baseline copies) so
`transcribe-critic-eval score --hypothesis all` scores them all.

Usage:
    python kev_threshold_sweep.py --run-dir runs/rev16-kev9b --out-dir runs/rev16-kev9b-sweep \
        --thresholds 0.1,0.2,0.3,0.5 --files 3,4,9
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transcribe_critic.shared import SpeechConfig, get_model_quality_rank
from transcribe_critic.transcription import (
    _apply_resolutions,
    _cluster_diffs,
    _filter_trivial_diffs,
    _merge_pairwise_diffs,
    _parse_wdiff_diffs,
    collapse_repetition_loops,
)

OMIT = "(omit)"


def rebuild_diffs(src: Path, models: list[str]):
    config = SpeechConfig(url="", output_dir=src)
    transcripts = {m: collapse_repetition_loops((src / f"asr_{m}.txt").read_text())[0]
                   for m in models}
    base_model = max(transcripts, key=get_model_quality_rank)
    base_words = transcripts[base_model].split()
    others = [m for m in models if m != base_model]
    pairwise = [(m, _filter_trivial_diffs(_parse_wdiff_diffs(transcripts[m], transcripts[base_model], config)))
                for m in others]
    all_diffs = _merge_pairwise_diffs(pairwise, base_model, models)
    clusters = _cluster_diffs(all_diffs, len(base_words),
                              context_words=SpeechConfig.merge_diff_context_words,
                              max_cluster_diffs=SpeechConfig.merge_max_diffs_per_call)
    return base_model, base_words, all_diffs, clusters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--files", default="3,4,9")
    ap.add_argument("--models", default="parakeet,qwen3-asr,distil-large-v3")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    models = args.models.split(",")
    thresholds = [float(t) for t in args.thresholds.split(",")]

    for f in args.files.split(","):
        src = args.run_dir / f"file{f}"
        out = args.out_dir / f"file{f}"
        out.mkdir(parents=True, exist_ok=True)
        for m in models:
            shutil.copyfile(src / f"asr_{m}.txt", out / f"asr_{m}.txt")
        shutil.copyfile(src / "asr_merged_kev.txt", out / "asr_merged_kev.txt")

        base_model, base_words, all_diffs, clusters = rebuild_diffs(src, models)
        recs = [json.loads(l) for l in open(src / "kev_diffs.jsonl")]
        ordered = [d for c in clusters for d in c]
        assert len(recs) == len(ordered), (len(recs), len(ordered))
        for r, d in zip(recs, ordered):
            assert (r["b_pos"], r["b_len"]) == (d["b_pos"], d["b_len"])

        stats = {}
        for t in thresholds:
            resolutions = {}
            overrides = 0
            for r, d in zip(recs, ordered):
                base_reading = d["readings"][base_model] or OMIT
                conf = r["kev_confidence"]
                chosen = r["kev_choice"]
                if chosen != base_reading and (conf is None or conf < t):
                    chosen = base_reading
                overrides += int(chosen != base_reading)
                resolutions[id(d)] = chosen
            merged = _apply_resolutions(base_words, all_diffs, resolutions)
            name = f"asr_merged_kev_t{t:g}.txt"
            (out / name).write_text(merged)
            stats[f"{t:g}"] = {"overrides": overrides, "n_diffs": len(recs)}
        json.dump(stats, open(out / "sweep_stats.json", "w"), indent=2)
        print(f"file{f}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
