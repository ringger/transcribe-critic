# Whisper Ensemble Experiment Results

## Dataset

Three files from Rev16 evaluated against reference transcripts using meeteval WER.

## Current Best Results

| File | parakeet | 3-way cross-arch | qwen3-asr | distil-large-v3 | 3-way Whisper |
|------|----------|------------------|-----------|-----------------|---------------|
| 3 | **27.3%** | 27.5% | 28.0% | 29.6% | 29.4% |
| 4 | 26.5% | **26.3%** | 26.9% | 27.6% | 28.5% |
| 9 | **20.4%** | 20.6% | 21.1% | 21.6% | 21.7% |
| **Avg** | **24.7%** | **24.8%** | **25.3%** | **26.3%** | **26.5%** |

Best single model: parakeet (24.7%). Best ensemble: 3-way cross-architecture parakeet+qwen3-asr+distil-large-v3 (24.8%, local qwen2.5:14b adjudicator). The cross-arch ensemble essentially matches parakeet solo, with the ensemble winning on file 4.

All results use anti-hallucination flags. Cross-arch ensemble uses local adjudicator (qwen2.5:14b).

## Experiment History

### 1–9. Architecture and Prompt Evolution

Started with chunk-rewrite (LLM rewrites 500-word chunks), which introduced errors in uncontested regions. Refactored to targeted diff resolution: wdiff finds disagreements, cluster nearby diffs, LLM resolves only the disagreements. Tested with local qwen2.5:7b and 14b — both degraded results due to format leakage (7b) or poor choices (14b).

Key milestones:
- **Targeted diff resolution** replaced chunk-rewrite (exp 2)
- **`_clean_resolution()`** strips LLM format artifacts (exp 3)
- **A/B choice format** replaced text-echo — eliminates invented words, 100% parse rate (exp 10)
- **Bug fixes**: context indexed by wrong positions, clustering sort mismatch (exp 9)

| Variant | File 3 WER | Parse Rate | Notes |
|---------|-----------|-----------|-------|
| Chunk-rewrite | 30.9% | — | Boundary duplication, uncontested errors |
| 7b + text-echo | 29.1% | ~55% | Parsing failures masked as conservatism |
| 14b + text-echo | 31.3% | 98% | More intervention ≠ better |
| 14b + A/B format | 29.3% | 100% | Format solved, model quality the bottleneck |

### 10. Claude Sonnet API

Switched to Claude Sonnet 4 via API. Added checkpointing and retry-without-context for content refusals.

| File | medium | 14b merged | Sonnet merged | Gap to medium |
|------|--------|------------|---------------|---------------|
| 3 | 28.9% | 29.3% | **28.3%** | **-0.6pp** |
| 4 | 27.9% | 28.0% | **27.6%** | **-0.3pp** |
| 9 | 24.8% | 24.9% | **24.8%** | **0.0pp** |
| **Avg** | **27.2%** | **27.4%** | **26.9%** | **-0.3pp** |

**Key finding:** Model quality was the bottleneck. The diff resolution architecture and A/B format were necessary but not sufficient — a capable adjudicator was also required.

### 11. Whisper Large — Catastrophic Hallucination

Whisper large (mlx-whisper) on file 3 produced "The unremarkable." repeated 7,479 times (97% WER). Root cause: `condition_on_previous_text=True` (default) creates a feedback loop. Large models are most susceptible.

### 12. Anti-Hallucination Flags

Added flags to all Whisper runs: `condition_on_previous_text=False`, `no_speech_threshold=0.2`, `compression_ratio_threshold=2.0`, `hallucination_silence_threshold=3.0`.

| Metric | Before flags | After flags | Change |
|--------|-------------|-------------|--------|
| Medium avg | 27.2% | 27.7% | +0.5pp |
| 2-way merged avg | 26.9% | 26.6% | -0.3pp |
| Gap (merged vs medium) | -0.3pp | -1.1pp | -0.8pp |

The flags changed individual WERs slightly, but the ensemble gap widened from -0.3pp to -1.1pp. The flags appear to produce transcripts that differ in more meaningful ways, giving the adjudicator better signal.

### 13. distil-large-v3

Replaced whisper large with `mlx-community/distil-whisper-large-v3` — a distilled model 6x faster than large-v3, within 1% WER, and specifically optimized to reduce hallucinations. No catastrophic failures (only minor hallucination loops of 8–10 words, caught by existing collapsing logic).

**Standalone quality:** distil-large-v3 (26.3% avg) beats medium (27.7%) by 1.4pp and even beats the 2-way small+medium ensemble (26.6%) by 0.3pp.

**3-way ensemble (small + medium + distil-large-v3):**

| File | distil-large-v3 | 2-way merged | 3-way merged |
|------|----------------|--------------|--------------|
| 3 | 29.6% | **28.5%** | 29.4% |
| 4 | **27.6%** | 28.2% | 28.5% |
| 9 | **21.6%** | 23.1% | 21.7% |
| **Avg** | **26.3%** | 26.6% | 26.5% |

The 3-way ensemble doesn't improve over distil-large-v3 alone. On file 4, merging dragged 27.6% up to 28.5% by incorporating medium and small's worse readings. When one model is clearly better, ensembling with weaker models dilutes quality.

## Key Lessons

1. **Constrain the output format.** A/B choice eliminates format leakage and invented words.
2. **Model quality matters most.** Same architecture, same prompt — Sonnet succeeds where local models fail.
3. **Ensembling helps when models are comparable.** 2-way small+medium ensemble beats medium by 1.1pp. But ensembling a strong model with weaker ones can hurt.
4. **Anti-hallucination flags are essential.** They prevent catastrophic failures and improve ensemble signal.
5. **Parakeet is the best single model.** 24.7% avg WER — faster than Whisper, better accuracy, word-level timestamps.
6. **Always do error analysis.** The cross-arch ensemble initially scored 32.3% — worse than any individual model. Rather than concluding the approach doesn't work, error analysis revealed a base selection bug. After the fix: 24.8%.
7. **Leaderboard WER doesn't transfer to all domains.** Granite scored 5.52% on clean benchmarks but 105% on podcasts. Parakeet (6.05% leaderboard) beat Granite by 80pp on real data.
8. **Base model selection is critical.** The best model must be the base for wdiff merging. Corrections flow inward from weaker models — starting from a weaker base amplifies errors.

### 14. Non-Whisper ASR Backends (2026-03-30)

The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (Aug 2025) shows Whisper is no longer competitive on standard benchmarks. The top open models achieve ~5-6% avg WER vs Whisper large-v3's ~9-10%. Three models with MLX support were added to the pipeline:

| Model | Leaderboard WER | Params | Backend | Architecture |
|-------|----------------|--------|---------|-------------|
| ibm-granite/granite-4.0-1b-speech | 5.52% | 1B | mlx-audio | Conformer + LLM decoder |
| Qwen/Qwen3-ASR-1.7B | 5.76% | 1.7B | mlx-audio | LLM-based encoder-decoder |
| nvidia/parakeet-tdt-0.6b-v2 | 6.05% | 0.6B | parakeet-mlx | FastConformer-TDT |

**Hypothesis:** Architecturally diverse models should provide better ensemble diversity than combining Whisper sizes, since the 3-way Whisper ensemble (26.5%) failed to improve over distil-large-v3 alone (26.3%).

**Implementation notes:**
- `--asr-models parakeet,qwen3-asr,granite-speech` CLI flag added alongside `--whisper-models`
- ASR outputs named `asr_{model}.txt/.json` (vs `whisper_{model}.txt/.json`)
- Granite-speech requires manual 240s audio chunking (no built-in long-audio support in mlx-audio); exhibited hallucination loops (caught by existing collapse logic)
- Parakeet handles long audio natively (120s chunks), produces word-level timestamps
- Qwen3-ASR handles long audio natively (1200s chunks), chunk-level timestamps only

**Rev16 results (files 3, 4, 9):**

| File | distil-large-v3 | parakeet | qwen3-asr | granite-speech | whisper_merged (3-way) |
|------|----------------|----------|-----------|----------------|----------------------|
| 3 | 29.6% | **27.3%** | empty | 93.1% | 29.4% |
| 4 | 27.6% | **26.5%** | empty | 110.2% | 28.5% |
| 9 | 21.6% | **20.4%** | empty | 112.1% | 21.7% |
| **Avg** | **26.3%** | **24.7%** | — | 105.2% | 26.5% |

**Key findings:**
- **Parakeet is the new best single model at 24.7% avg WER** — 1.6pp better than distil-large-v3 (26.3%) and better than the 3-way Whisper ensemble (26.5%). Wins on all 3 files. Also the fastest: ~1 min for 2h audio.
- **Qwen3-ASR at 25.3%** — required two bug fixes (WAV conversion for long MP3s, max_tokens=32768, chunk_duration=300s) before producing usable output.
- **3-way cross-arch ensemble (parakeet + qwen3-asr + distil-large-v3) at 24.8%** — essentially matches parakeet solo. See experiment 15 below for the error analysis that found the base selection bug.
- **Granite-speech is unusable on podcast audio.** Despite 5.52% WER on clean benchmarks, it produced 801/775/311 hallucination loops per file (57K/48K/18K words removed by collapse). Even after collapse, WER is 93-112%. The model has no built-in anti-hallucination controls, and the chunking overlap may compound the problem.

**Leaderboard WER vs podcast WER:** The Open ASR Leaderboard benchmarks use clean, segmented audio (LibriSpeech, Earnings-22, etc.). Podcast audio is harder: informal speech, overlapping speakers, background music, varied recording quality. Granite's 5.52% leaderboard WER vs 105% podcast WER shows that benchmark performance does not transfer to all domains.

### 15. Cross-Architecture Ensemble Error Analysis (2026-03-31)

Initial 3-way ensemble (parakeet + qwen3-asr + distil-large-v3) scored **32.3% avg WER** with local adjudicator — worse than any individual model. Error analysis identified the cause:

**Symptom:** Garbled name lists, doubled words, misplaced phrases in merged output.

**Investigation:**
1. Checked normalization — already working; wdiff produced only 2 real diffs in first 300 normalized words between parakeet and qwen3
2. Compared parakeet (best, 20.4%) to merged output (25.4%) for file9: 778 substitutions, 873 insertions
3. Categorized diffs: 383 punctuation-only, 386 real word changes — the adjudicator was accepting too many changes
4. Noticed merge used qwen3-asr as base (quality_rank=8) rather than parakeet (rank=7 at the time)
5. **Found the bug:** `_resolve_whisper_diffs` re-selected base model using old `MODEL_SIZES` list (Whisper-only), ignoring the quality-ranked selection made by `_ensemble_whisper_transcripts`. The two code paths disagreed.

**Fix:** Made `_resolve_whisper_diffs` use `get_model_quality_rank()` consistently. Updated quality ranks to reflect Rev16 results (parakeet=9, qwen3=8) rather than leaderboard scores.

**Results after fix:**

| Variant | File 3 | File 4 | File 9 | Avg WER |
|---------|--------|--------|--------|---------|
| asr_parakeet (solo) | 27.3% | 26.5% | **20.4%** | **24.7%** |
| 3-way merged (local, fixed) | 27.5% | **26.3%** | 20.6% | **24.8%** |
| asr_qwen3-asr (solo) | 28.0% | 26.9% | 21.1% | 25.3% |
| whisper_distil-large-v3 (solo) | 29.6% | 27.6% | 21.6% | 26.3% |
| 3-way merged (local, before fix) | 35.7% | 35.7% | 25.4% | 32.3% |

The bug fix improved the ensemble from **32.3% to 24.8%** (7.5pp). The ensemble now matches parakeet solo, with the ensemble winning on file 4 (26.3% vs 26.5%).

**Lesson:** Always do error analysis before concluding an approach doesn't work. The initial 32.3% result appeared to show that cross-architecture ensembling degrades quality. The real cause was a pre-existing base selection bug exposed by the new models — the wdiff merging approach was working correctly once the right model was used as base.

## Next Steps

- Score new ASR models and mixed ensembles on Rev16
- Try 2-way ensembles: distil-large-v3 + parakeet, distil-large-v3 + granite-speech
- Expand eval to more Rev16 files for statistical significance
- Test with a local adjudicator closer to Sonnet quality (e.g., Llama 3.3 70B)

## Environment

- Hardware: M4 Max, 64GB RAM
- LLM (local): Ollama qwen2.5:14b (9GB)
- LLM (API): Claude Sonnet 4 (claude-sonnet-4-20250514)
- Whisper models: small, medium, distil-large-v3 (mlx-community/distil-whisper-large-v3)
- ASR models: parakeet-tdt-0.6b-v2, Qwen3-ASR-1.7B-8bit, granite-4.0-1b-speech-8bit
- Scoring: meeteval WER
