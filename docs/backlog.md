# Backlog

See `docs/experiments.md` for results of completed experiments.

## Cross-Architecture Ensemble Improvements

- **Timestamp-guided alignment.** Align by time window (e.g., 10s segments) using word timestamps from parakeet/whisper instead of LCS. Prevents name-list garbling. More principled but bigger lift.

- **Sentence-level-first alignment.** Split into sentences first (using parakeet's sentence boundaries), align sentences across models, then wdiff within matched sentences. Keeps damage local.

- **Revisit LLM-based chunk merge.** Give LLM full parallel texts in small chunks. Failed with Whisper-only (exp 1-9) but stronger models + more diverse signal might work now.

- **Ripple diff resolution.** Resolve diffs sequentially within clusters, re-running wdiff after each resolution so subsequent diffs have correct positions and updated context. Initial attempt using position-shift patching failed (corrupted text). Needs a ground-up design that re-diffs after each applied resolution rather than adjusting offsets.

## Model Issues

- **Granite-speech is unusable on podcast audio.** Consider removing from registry or flagging as experimental.

- **Qwen3-ASR MP3 truncation.** mlx-audio silently truncates long MP3 files. Workaround in place (`_ensure_wav()`). Should file upstream bug.

- **Qwen3-ASR token limit.** Default `max_tokens=8192` insufficient for 5-min chunks. Workaround in place (`max_tokens=32768`). Should file upstream bug or contribute fix.

## Hypothesis Reassessment at Disagreement Points

- **Parakeet TDT forced alignment.** TDT decoder exposes logits via joint network — architecturally feasible to score candidate token sequences against encoder output. Blocked on SentencePiece tokenizer extraction (not bundled in HF cache or .nemo archive).

- **Prompt Whisper with alternate transcripts.** Use `initial_prompt` to bias Whisper toward each candidate reading. If prompting with "wedding" produces "wedding" but prompting with "wife" also produces "wedding", strong evidence for "wedding." Cheap and uses existing API.

## Evaluation

- **Expand Rev16 eval to more files.** Current results (3 files) are too few for statistical significance — e.g., parakeet 24.7% vs ensemble 24.8% could be noise. Add more Rev16 files to the manifest and re-run.

- **Test 2-way ensembles.** Currently only tested 3-way (parakeet+qwen3-asr+distil-large-v3). A 2-way ensemble matching 3-way quality would save compute. Test: parakeet+distil-large-v3, parakeet+qwen3-asr.

- **Test stronger local adjudicator.** Local adjudicator gave neutral ensemble results. Try Llama 3.3 70B or Qwen 2.5 72B to close the gap with Claude Sonnet adjudication without API cost.
