#!/usr/bin/env python3
"""Adjudicate transcribe-critic ASR disagreements with a Kev (System One) server.

Reuses transcribe_critic's own diff extraction, clustering, and resolution
application so the only thing that changes is the decision-maker: instead of
an LLM answering "1. A / 2. B" per cluster, each disagreement becomes a Kev
Choice question over the distinct readings, asked against the same cluster
context the LLM prompt would have shown.

Usage:
    python kev_adjudicate.py --src-dir <run>/file4 --out-dir runs/rev16-kev9b/file4 \
        --models parakeet,qwen3-asr,distil-large-v3 [--kev-url http://127.0.0.1:8009]

Outputs (in --out-dir):
    asr_merged_kev.txt   merged transcript (Kev decisions applied to the base)
    asr_<model>.txt      copies of the input transcripts, so the eval scorer
                         scores the baselines alongside
    kev_diffs.jsonl      one record per disagreement: readings, Kev
                         probabilities, chosen reading, and the reference
                         adjudicator's choice when a checkpoint dir is given
    kev_summary.json     counts, agreement rate, timing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from transcribe_critic.shared import SpeechConfig, get_model_quality_rank
from transcribe_critic.transcription import (
    _apply_resolutions,
    _build_cluster_prompt,
    _cluster_diffs,
    _filter_trivial_diffs,
    _merge_pairwise_diffs,
    _parse_wdiff_diffs,
    collapse_repetition_loops,
)

OMIT = "(omit)"
INSTRUCTIONS = (
    "Several automatic speech recognizers transcribed the same recording and "
    "disagree about the words at marker [{n}] in the transcript excerpt. "
    "Choose the reading that is most likely what the speaker actually said, "
    "judging by grammar, meaning, and the surrounding context. "
    "When more than one recognizer produced the same reading, that agreement is "
    "a useful signal but not decisive."
)


def kev_call(url: str, state, questions: dict) -> dict:
    body = json.dumps({"state": state, "model": "kev-latest",
                       "questions": questions}).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def reading_options(readings: dict[str, str]) -> dict[str, str]:
    """Distinct readings -> Kev choice criteria (option name -> description)."""
    counts: dict[str, int] = {}
    for text in readings.values():
        key = text if text else OMIT
        counts[key] = counts.get(key, 0) + 1
    n = len(readings)
    options = {}
    for key, c in counts.items():
        who = f"{c} of {n} recognizers"
        if key == OMIT:
            options[key] = f"No words here: {who} heard nothing at this point"
        else:
            options[key] = f"Transcribed by {who}"
    return options


def letter_for(readings: dict[str, str], chosen: str) -> str:
    """Map a chosen reading back to the first model letter that produced it."""
    for j, text in enumerate(readings.values()):
        if (text if text else OMIT) == chosen:
            return chr(ord("A") + j)
    return chr(ord("A") + len(readings) - 1)


def load_reference_choices(ckpt_dir: Path | None, n_clusters: int) -> list[list[str] | None]:
    """Load per-cluster letter choices from a transcribe-critic checkpoint dir."""
    if ckpt_dir is None:
        return [None] * n_clusters
    out = []
    for i in range(n_clusters):
        p = ckpt_dir / f"cluster_{i:03d}.json"
        if p.exists():
            out.append(json.load(open(p)).get("choices"))
        else:
            out.append(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", required=True,
                    help="comma-separated model short names, e.g. parakeet,qwen3-asr,distil-large-v3")
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009")
    ap.add_argument("--ref-checkpoints", type=Path, default=None,
                    help="ensemble_chunks dir from a prior LLM run, for per-diff agreement")
    ap.add_argument("--context-words", type=int, default=SpeechConfig.merge_diff_context_words)
    ap.add_argument("--max-diffs", type=int, default=SpeechConfig.merge_max_diffs_per_call)
    ap.add_argument("--limit-clusters", type=int, default=None, help="debug: stop after N clusters")
    args = ap.parse_args()

    models = args.models.split(",")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = SpeechConfig(url="", output_dir=args.out_dir)

    transcripts = {}
    loops_total = 0
    for m in models:
        src = args.src_dir / f"asr_{m}.txt"
        shutil.copyfile(src, args.out_dir / src.name)
        text, loops = collapse_repetition_loops(src.read_text())
        loops_total += len(loops)
        transcripts[m] = text

    base_model = max(transcripts, key=get_model_quality_rank)
    base_text = transcripts[base_model]
    base_words = base_text.split()
    others = [m for m in models if m != base_model]
    print(f"base={base_model} others={others} loops_collapsed={loops_total}")

    pairwise = []
    for m in others:
        diffs = _filter_trivial_diffs(_parse_wdiff_diffs(transcripts[m], base_text, config))
        print(f"  {m} vs {base_model}: {len(diffs)} meaningful diffs")
        pairwise.append((m, diffs))
    all_diffs = (_merge_pairwise_diffs(pairwise, base_model, models)
                 if len(others) > 1 else pairwise[0][1])
    clusters = _cluster_diffs(all_diffs, len(base_words),
                              context_words=args.context_words,
                              max_cluster_diffs=args.max_diffs)
    n_diffs = sum(len(c) for c in clusters)
    print(f"  {n_diffs} multi-way diffs in {len(clusters)} clusters")

    ref_choices = load_reference_choices(args.ref_checkpoints, len(clusters))
    if args.ref_checkpoints:
        n_ref = sum(1 for c in ref_choices if c)
        print(f"  reference checkpoints found for {n_ref}/{len(clusters)} clusters")

    resolutions: dict[int, str] = {}
    agree = compared = 0
    t0 = time.time()
    diffs_out = open(args.out_dir / "kev_diffs.jsonl", "w")

    for ci, cluster in enumerate(clusters):
        if args.limit_clusters is not None and ci >= args.limit_clusters:
            break
        prompt = _build_cluster_prompt(cluster, base_words, context_words=args.context_words)
        # Context line is the first block of the prompt, between the two headers.
        context_line = prompt.split("CONTEXT:\n", 1)[1].split("\n\nDISAGREEMENTS:", 1)[0]

        questions = {}
        single = {}
        for i, d in enumerate(cluster, 1):
            readings = d["readings"] if "readings" in d else {
                "a": d["a_text"], base_model: d["b_text"]}
            opts = reading_options(readings)
            if len(opts) == 1:
                single[i] = next(iter(opts))
                continue
            questions[f"q{i}"] = {
                "type": "choice",
                "instructions": INSTRUCTIONS.format(n=i),
                "criteria": opts,
            }

        answers = {}
        latency = None
        if questions:
            resp = kev_call(args.kev_url, {"transcript_excerpt": context_line}, questions)
            answers = resp["answers"]
            latency = resp.get("latency_ms")

        cluster_letters = []
        for i, d in enumerate(cluster, 1):
            readings = d["readings"] if "readings" in d else {
                "a": d["a_text"], base_model: d["b_text"]}
            if i in single:
                chosen, probs, conf = single[i], None, None
            else:
                a = answers[f"q{i}"]
                chosen, probs, conf = a["choice"], a["probabilities"], a["confidence"]
            resolutions[id(d)] = chosen
            letter = letter_for(readings, chosen)
            cluster_letters.append(letter)

            ref_letter = None
            rc = ref_choices[ci]
            if rc and i - 1 < len(rc) and rc[i - 1]:
                ref_letter = rc[i - 1]
                ref_text = list(readings.values())[ord(ref_letter) - ord("A")] or OMIT
                compared += 1
                agree += int(ref_text == chosen)
            diffs_out.write(json.dumps({
                "cluster": ci, "n": i, "b_pos": d["b_pos"], "b_len": d["b_len"],
                "readings": readings, "kev_choice": chosen, "kev_letter": letter,
                "kev_probs": probs, "kev_confidence": conf,
                "ref_letter": ref_letter,
                "ref_choice": (list(readings.values())[ord(ref_letter) - ord("A")] or OMIT)
                if ref_letter else None,
            }) + "\n")
        diffs_out.flush()
        if ci % 10 == 0:
            el = time.time() - t0
            print(f"  cluster {ci + 1}/{len(clusters)} ({len(cluster)} diffs, "
                  f"{latency} ms) elapsed {el:.0f}s", flush=True)

    diffs_out.close()
    merged = _apply_resolutions(base_words, all_diffs, resolutions)
    (args.out_dir / "asr_merged_kev.txt").write_text(merged)
    elapsed = time.time() - t0
    summary = {
        "src_dir": str(args.src_dir), "models": models, "base_model": base_model,
        "n_diffs": n_diffs, "n_clusters": len(clusters), "n_resolved": len(resolutions),
        "ref_compared": compared, "ref_agree": agree,
        "ref_agreement": (agree / compared) if compared else None,
        "elapsed_s": round(elapsed, 1),
    }
    json.dump(summary, open(args.out_dir / "kev_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
