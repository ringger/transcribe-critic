# Kev-9B as the transcribe-critic adjudicator (2026-09-23)

Brought here on 2026-10-18 from the sibling `jev` project (its `EXPERIMENTS.md`
§1, at jev commit b45e770), where the Kev decision-model experiments were run.
Everything this experiment produced lives in this folder: the scripts
(`kev_adjudicate.py`, `kev_threshold_sweep.py`, `run_rev16.sh`), the outputs in
`runs/`, the logs in `logs/` and `logs-run_rev16-kev9b.log`, and the numbers in
`results.json`. Run the scripts from this folder; they import
`transcribe_critic` and need a Kev server, whose clone and launcher stay in
`jev` (`jev/vendor/kev`, see `jev/TASKS.md` #26).

**Question.** Can a System One decision model replace the LLM that resolves
ASR disagreements in transcribe-critic's cross-architecture ensemble?

**Setup.**
- Model: `jaredpalmer/kev-9b` (Qwen3.5-9B-Base + LoRA + pointer head), served
  locally via MLX in bf16 on an M4 Max, calibration temperature 2.30.
- Data: Rev16 files 3, 4, 9 (the same three used for every prior
  transcribe-critic ensemble experiment). Files 3 and 4 are two-hour episodes
  of a vernacular podcast ("The Read"); the reference is verbatim informal
  speech, which is what makes this set hard for every adjudicator.
- Witnesses: existing `asr_parakeet` (base), `asr_qwen3-asr`,
  `asr_distil-large-v3` transcripts from `eval-runs/rev16-asr-backends`.
- Method: `kev_adjudicate.py` reuses transcribe-critic's wdiff diff extraction,
  multi-way merge, clustering (30 context words, ≤50 diffs per cluster) and
  `_apply_resolutions`. Each cluster's context line is the Kev `state`; each
  disagreement is a Choice question whose options are the distinct readings
  (identical readings collapsed, described as "transcribed by N of 3
  recognizers"; empty reading = "(omit)").
- Scoring: `transcribe-critic-eval score`, plain WER against the Rev16 STM
  references.

**Throughput.** 3,549 disagreements in 507 clusters, 14 minutes total on the
laptop (0.4 to 9 s per cluster; roughly 0.24 s per decision). No API cost.

**Result: neutral.** Kev-9B zero-shot matches the earlier local
qwen2.5:14b adjudicator (24.8%) and does not beat parakeet alone.

| Variant | File 3 | File 4 | File 9 | Avg WER |
|---|---|---|---|---|
| parakeet alone | 27.3 | 26.5 | 20.4 | **24.7** |
| Kev-9B merge (all decisions) | 27.5 | 26.5 | 20.8 | 24.9 |
| Kev-9B, override only if confidence ≥ 0.2 | 27.4 | 26.4 | 20.5 | 24.8 |
| Kev-9B, override only if confidence ≥ 0.5 | 27.3 | 26.4 | 20.4 | 24.7 |
| qwen3-asr alone | 28.0 | 26.9 | 21.1 | 25.3 |
| distil-large-v3 alone | 29.6 | 27.6 | 21.6 | 26.3 |
| prior: local qwen2.5:14b 3-way merge | 27.5 | 26.3 | 20.6 | 24.8 |

**Diagnostics.**
- Kev's probabilities sit near uniform: mean Choice confidence 0.25; on
  two-way items the winning reading averages p = 0.62, on three-way items
  0.48. It is not discriminating on this task.
- Agreement with the prior local adjudicator is 45 to 49 percent per file,
  close to chance for two- and three-way choices.
- The confidence-threshold sweep converges monotonically to parakeet as the
  threshold rises: Kev's overrides are slightly net-harmful at every level,
  so there is no confident subset worth keeping.
- Many diffs are punctuation, hyphenation, or contraction variants that WER
  normalization ignores, so the effective decision set is smaller than 3,549.

**Interpretation.** Kev was trained on ticket routing, sentiment, NLI-style
policy decisions, and similar; "which of these near-identical readings did a
speaker of vernacular English actually say" is out of distribution, and
the surrounding context alone rarely settles it. The dataset's difficulty
(informal register, long form, verbatim reference) means the per-diff
signal available from text alone is thin for any adjudicator, which is also
why Sonnet only bought a few tenths of a point in the earlier experiments.

**Next steps (not yet done).**
1. Per-diff oracle: align the Rev16 reference to the base transcript and
   label which reading is correct at each diff. This gives the ceiling for
   any adjudicator and turns each diff into a labeled Kev training example.
2. Fine-tune Kev-4B/9B with `--init_from` on oracle labels from the other 13
   Rev16 files, evaluate on 3/4/9. Kev's README reports large gains from a
   few hundred in-domain labels.
3. Add acoustic confidence to the state (transcribe-critic already has
   `--confidence`), since text context alone is the limiting signal.

Artifacts: `runs/rev16-kev9b/` (merges, `kev_diffs.jsonl` per file,
`results.md`), `runs/rev16-kev9b-sweep/`, logs in `logs/`.

