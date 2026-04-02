# Backlog

## Architecture / Refactoring

- ~~**Remove Whisper-centric naming throughout codebase.**~~ ✅ Done. Unified `ALL_MODELS` registry, `--models` CLI flag (old flags deprecated), `asr_*.txt` file naming with backward-compat read-both, `transcribe-critic-migrate` utility. See commit history.

- ~~**Ensemble should respect requested models, not discover from disk.**~~ ✅ Done. `_ensemble_asr_transcripts()` now filters `data.asr_transcripts` to only models in `config.models`, with a log message for skipped models.

## Cross-Architecture Ensemble Improvements

The wdiff-based ensemble degrades (31-32% WER) when mixing architecturally diverse models, even though individual models score well (24-26%). Root cause: wdiff alignment breaks on structurally different text (punctuation, capitalization, word boundaries, name list ordering).

- ~~**Normalize before diffing.**~~ ✅ Done. `_normalize_for_comparison()` in merge.py strips punctuation, lowercases, normalizes whitespace before wdiff alignment.

- **Timestamp-guided alignment.** Align by time window (e.g., 10s segments) using word timestamps from parakeet/whisper instead of LCS. Prevents name-list garbling. More principled but bigger lift.

- **Sentence-level-first alignment.** Split into sentences first (using parakeet's sentence boundaries), align sentences across models, then wdiff within matched sentences. Keeps damage local.

- **Confidence-weighted selection.** Use parakeet's per-word confidence scores. Take parakeet's word unless confidence is low AND another model disagrees. No LLM needed.

- **Revisit LLM-based chunk merge.** Give LLM full parallel texts in small chunks. Failed with Whisper-only (exp 1-9) but stronger models + more diverse signal might work now.

## Model Issues

- **Granite-speech is unusable on podcast audio.** 105% WER due to massive hallucination (801 loops on one 2h file). Leaderboard WER (5.52%) doesn't transfer. Consider removing from registry or flagging as experimental.

- **Qwen3-ASR MP3 truncation.** mlx-audio silently truncates long MP3 files. Workaround in place (WAV conversion via `_ensure_wav()`). Should file upstream bug.

- **Qwen3-ASR token limit.** Default `max_tokens=8192` insufficient for 5-min chunks. Workaround in place (`max_tokens=32768`). Should file upstream bug or contribute fix.

## Evaluation

- Expand Rev16 eval to more files for statistical significance
- Test 2-way ensembles: parakeet + distil-large-v3, parakeet + qwen3-asr
- Test with a local adjudicator closer to Sonnet quality (e.g., Llama 3.3 70B, Qwen 2.5 72B)
