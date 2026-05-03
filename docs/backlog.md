# Backlog

See `docs/experiments.md` for results of completed experiments.

## Cross-Architecture Ensemble Improvements

- **Timestamp-guided alignment.** Align by time window (e.g., 10s segments) using word timestamps from parakeet/whisper instead of LCS. Prevents name-list garbling. More principled but bigger lift.

- **Sentence-level-first alignment.** Prototype implemented behind `--sentence-align` flag with boundary padding (5 words). Eval on Rev16 (50min–132min) shows 96% diff overlap with baseline; no spurious boundary artifacts. Remaining differences are mostly filler words ("Like,", "Yeah.") that whole-text LCS absorbs differently. Still needs end-to-end WER comparison to confirm it helps on cascade-prone audio.
  - *Limitation*: requires reasonably granular segments from both models. Qwen3-ASR produces only ~10 segments for a 50min file (~1000 words each), causing 2-3x diff explosion when paired with parakeet's fine-grained sentences. Sentence alignment should fall back to whole-text for models with coarse segments.
  - *Complementary diff cancellation*: detect insertion+deletion of the same word at adjacent sentence boundaries and cancel them out. Not yet needed — boundary padding handles the cases seen so far.

- **Revisit LLM-based chunk merge.** Give LLM full parallel texts in small chunks. Failed with Whisper-only (exp 1-9) but stronger models + more diverse signal might work now.

- **Ripple diff resolution.** Resolve diffs sequentially within clusters, re-running wdiff after each resolution so subsequent diffs have correct positions and updated context. Initial attempt using position-shift patching failed (corrupted text). Needs a ground-up design that re-diffs after each applied resolution rather than adjusting offsets.

- **Symbol/word duplication in adjudication.** When one model emits "%" and another spells out "percent", the adjudicator sometimes keeps both, producing "252% percent". Seen on Instagram reel run (parakeet+qwen3-asr+distil-large-v3, Claude Sonnet adjudicator) — every `%` token got a duplicated " percent" suffix. Likely the same class of bug applies to other symbol/word pairs ($/dollars, &/and, #/number). Fix could be normalization-aware diff merging or a stronger adjudicator instruction to pick one form.

## Model Issues

- **Qwen3-ASR MP3 truncation.** mlx-audio silently truncates long MP3 files. Workaround in place (`_ensure_wav()`). Should file upstream bug.

- **Qwen3-ASR token limit.** Default `max_tokens=8192` insufficient for 5-min chunks. Workaround in place (`max_tokens=32768`). Should file upstream bug or contribute fix.

## Hypothesis Reassessment at Disagreement Points

- **Parakeet TDT forced alignment.** TDT decoder exposes logits via joint network — architecturally feasible to score candidate token sequences against encoder output. Blocked on SentencePiece tokenizer extraction (not bundled in HF cache or .nemo archive).

- **Prompt Whisper with alternate transcripts.** Use `initial_prompt` to bias Whisper toward each candidate reading. If prompting with "wedding" produces "wedding" but prompting with "wife" also produces "wedding", strong evidence for "wedding." Cheap and uses existing API.

## Output / Presentation

- **TTS summary narration.** Read the generated summary aloud as an mp3 alongside the transcript. Two modes:
  - *Out-of-the-box voice* (simpler, no consent issue): pick a stock voice from the TTS engine. Good default; works for any content including multi-speaker.
  - *Voice-cloned* from source audio (personal-use only due to consent): extract a 30–60s clean sample (parakeet word timestamps can help pick a contiguous span) and clone the speaker's voice. Best fit for single-speaker content (podcasts, monologues like Nate Jones).
  - Candidate TTS: ElevenLabs (cloud, best quality, both modes), XTTS-v2 / F5-TTS / Kokoro (local, Apple Silicon). Slots in as a new pipeline stage after `analysis`: summary text → TTS → mp3.

## Evaluation

- **Expand Rev16 eval to more files.** Current results (3 files) are too few for statistical significance — e.g., parakeet 24.7% vs ensemble 24.8% could be noise. Add more Rev16 files to the manifest and re-run.

- **Test 2-way ensembles.** Currently only tested 3-way (parakeet+qwen3-asr+distil-large-v3). A 2-way ensemble matching 3-way quality would save compute. Test: parakeet+distil-large-v3, parakeet+qwen3-asr.

- **Test stronger local adjudicator.** Local adjudicator gave neutral ensemble results. Try Llama 3.3 70B or Qwen 2.5 72B to close the gap with Claude Sonnet adjudication without API cost.
