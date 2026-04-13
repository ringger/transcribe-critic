# Experiment Results

## Rev16 Baseline WER by Model (3 files, 2026-03-31)

| Model | File 3 | File 4 | File 9 | Avg | Wtd Avg |
|-------|--------|--------|--------|-----|---------|
| parakeet | 27.3% | 22.6% | 20.4% | 23.4% | 24.7% |
| qwen3-asr | 28.0% | 22.8% | 21.1% | 24.0% | 25.3% |
| distil-large-v3 | 29.6% | 23.3% | 21.6% | 24.8% | 26.3% |

Source: `eval-runs/rev16-asr-backends/results.md`

---

## Hypothesis Reassessment Experiments (2026-04-06)

All experiments on Rev16, comparing against parakeet standalone baseline.

### Re-recognition (scripts/prototype_realign.py)

Re-transcribe audio segments around disagreements and compare results.

**File 3 (The Read, 2h12m, AAVE):** 20-diff sample
- Models almost always reproduce their own reading (self-confirmation is noise)
- Cross-model confirmation (model changes its mind) is rare but reliable when it occurs
- With 15s context padding, 1542 segments merge down to 9 (~whole file) — pointless
- WER: 27.3% (identical to parakeet baseline)
- **Verdict:** Not worth the compute

### Confidence-Weighted Selection (scripts/prototype_confidence.py)

Use per-word confidence scores at disagreement positions to pick the better reading.

**Confidence availability:**
| Model | Signal | Source |
|-------|--------|--------|
| Parakeet | Per-token confidence (0-1, entropy-based) | `AlignedToken.confidence`, now saved in JSON |
| Whisper (distil-large-v3) | Per-word probability (0-1) | Already in JSON |
| Qwen3-ASR | None | No confidence exposed |

**Key finding:** Low confidence (< 0.95) correctly flags uncertain positions — every wrong reading in our sample showed low confidence. But confidence tells you *when* to doubt, not *what* to believe.

**File 3 (2h12m):** All 1542 diffs, threshold=0.9
- 85 changes made
- WER: 27.5% (+0.2% worse than parakeet 27.3%)

**File 9 (Podcasts in Color, 50m):** All 332 diffs
- 14 changes made
- WER: 20.4% (tied with parakeet)
- Threshold sweep: 0.85 → 20.5%, 0.90 → 20.4%, 0.95 → 20.4%

### Confidence-Augmented LLM Adjudication

Add `[conf=X%]` annotations to ensemble prompt so the LLM can weigh acoustic certainty alongside linguistic plausibility. Enabled via `--confidence` CLI flag.

**File 9 results:**

| Adjudicator | Confidence | WER | vs Parakeet (20.4%) |
|-------------|-----------|-----|---------------------|
| Local (qwen2.5:14b) | No | 20.7% | +0.3% worse |
| Local (qwen2.5:14b) | Yes | 20.6% | +0.2% worse |
| API (Claude Sonnet) | No | 20.4% | tied |
| API (Claude Sonnet) | Yes | 20.4% | tied |

- Confidence helped local adjudicator marginally (20.7% → 20.6%)
- No effect on API adjudicator (already making good decisions from context)
- API consistently better than local regardless of confidence

### Forced Alignment

**Qwen3-ForcedAligner-0.6B:** Tested on "wedding" vs "wife" disagreement. Assigns identical timestamps and confidence (0.6738) to both correct and incorrect text. It is a timestamp localization model, not a hypothesis scorer. **Not useful.**

**Parakeet TDT forced alignment:** Architecturally feasible (joint network exposes logits) but blocked on SentencePiece tokenizer extraction.

### New Model: cohere-transcribe

Best WER on Open ASR Leaderboard (5.42%). Added to `ALL_MODELS` with gated HF access.

**File 9 results:**
| Model | WER |
|-------|-----|
| parakeet | 20.4% |
| qwen3-asr | 21.1% |
| distil-large-v3 | 21.6% |
| cohere | 23.3% |

Leaderboard benchmarks (clean, read speech) do not predict podcast performance. Same pattern as granite-speech (5.52% leaderboard, 105% on podcasts). The mlx-community 8-bit quantized version has a conv1d shape bug; original fp32 model works.

---

## Sentence-Level Alignment (2026-04-13)

Prototype behind `--sentence-align` flag. Splits transcripts into sentences using ASR JSON timestamps before wdiff, with 5-word boundary padding to prevent edge artifacts. Comparison script: `scripts/compare_alignment.py`.

### Diff comparison: parakeet vs distil-large-v3

| File | Words | Baseline diffs | SA diffs | Common | Only BL | Only SA |
|------|-------|----------------|----------|--------|---------|---------|
| 30s fixture | 73 | 4 | 4 | 4 | 0 | 0 |
| file14 (3min) | 397 | 22 | 23 | 20 | 2 | 3 |
| 10min extract | 1,369 | 20 | 20 | 20 | 0 | 0 |
| file9 (50min) | 9,872 | 188 | 198 | 182 | 6 | 16 |
| file3 (132min) | 23,916 | 1,002 | 1,017 | 962 | 40 | 55 |
| file4 (126min) | 23,554 | 1,087 | 1,109 | 1,057 | 30 | 52 |

~96% diff overlap on long files. SA-only diffs are mostly filler words ("Like,", "Yeah.") that whole-text LCS absorbs differently. No spurious boundary artifacts with padding.

### Qwen3-ASR segment granularity issue

Qwen3-ASR produces only ~10 segments for a 50min file (~1000 words each). Pairing with parakeet's 624 fine-grained sentences caused 2-3x diff explosion (560 SA diffs vs 203 baseline). Sentence alignment needs reasonably granular segments from both models.

### Assessment

Sentence-level alignment produces near-identical diffs to whole-text alignment on all files tested. The original motivation — preventing alignment cascade on long audio (e.g., name-list garbling) — was not triggered on any Rev16 podcast, even at 132 min / 24K words. Whole-text wdiff handled these files without cascading.

The prototype works correctly and doesn't regress, but we have not yet found audio where it demonstrably helps. The cascade problem may be rare with the parakeet + distil-large-v3 pair, or may require specific content types (e.g., dense name lists) not present in Rev16.

**Decision:** keep behind `--sentence-align` flag, do not make default. The flag is available if cascade problems surface on specific audio. Worth revisiting if a triggering case is found.

### Test clip sources

- 30s: `tests/fixtures/tao_30s.mp3` (project fixture)
- 3min: Rev16 file14 — "Coming Soon: Season 2"
- 10min: First 10min of Rev16 file27 — "What We Own is Sacred Because We Are Sacred" (ffmpeg trim)
- 50min/132min/126min: Rev16 files 9/3/4 (pre-existing transcripts in `eval-runs/rev16-asr-backends/`)

---

## Key Takeaways

1. **Parakeet standalone is hard to beat on podcast audio.** No ensemble method tested improves over parakeet's 20.4-27.3% WER across Rev16 files.

2. **Confidence is a valid uncertainty signal** but insufficient for decision-making without knowing what to trust instead.

3. **API adjudicator (Claude Sonnet) outperforms local (qwen2.5:14b)** consistently, but neither beats parakeet solo.

4. **Leaderboard WER doesn't predict podcast performance.** Both cohere (5.42% leaderboard → 23.3% podcast) and granite (5.52% → 105%) show large gaps.

5. **Remaining untested approach:** Prompting Whisper with alternate transcripts at disagreement points (targeted hypothesis testing via `initial_prompt`).
