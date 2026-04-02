# Transcript Source Quality Notes

## YouTube Auto-Captions

YouTube generates automatic captions using Google's Cloud Speech-to-Text API, an
ASR system built on deep neural networks. The system identifies phonemes from the
audio, assembles them into words, then applies NLP for context, grammar, and
punctuation.

### Accuracy

- University of Minnesota research found YouTube auto-captions typically achieve
  **60-70% accuracy** overall.
- Accuracy varies significantly by content type. Clean studio audio with a single
  native English speaker can be much higher; noisy recordings, multiple speakers,
  accents, or technical jargon push accuracy much lower.
- YouTube's ASR has improved over time: neural networks (2015), transformer models
  (2019), LLM-based context (2022).

### Known Limitations

- **Technical terminology**: Domain-specific terms are frequently mangled
  (e.g., "dienophile" → "dinah file", "dyna file", etc.)
- **Proper nouns**: Names of people, companies, and products are often wrong
- **Filler words**: "um", false starts, and repetitions are transcribed literally
- **Punctuation**: Older videos may lack punctuation entirely; newer ones are
  better but still imperfect
- **No speaker labels**: Auto-captions don't identify speakers

### Creator-Uploaded vs Auto-Generated

YouTube captions may be auto-generated or manually uploaded by the creator.
The pipeline downloads whatever is available via yt-dlp. Creator-uploaded
captions are typically much higher quality but less common. The VTT file
metadata doesn't always make it obvious which type you have.

## ASR Models

The pipeline runs multiple ASR models from diverse architectures and ensembles
them via LLM adjudication. The default models are `distil-large-v3`, `parakeet`,
and `qwen3-asr`. See [ensemble-experiments.md](ensemble-experiments.md) for
comparative WER data.

### Model Characteristics

| Model | Backend | Rev16 WER | Speed | Notes |
|-------|---------|-----------|-------|-------|
| parakeet | parakeet-mlx | **24.7%** | Fast | Best single model; word-level timestamps |
| qwen3-asr | mlx-audio | 25.3% | Moderate | Chunk-level timestamps |
| distil-large-v3 | mlx-whisper | 26.3% | Fast | Best Whisper model; 6x faster than large-v3 |
| small | mlx-whisper | 28.9% | Fast | Retired from defaults |
| medium | mlx-whisper | 27.7% | Moderate | Retired from defaults |
| large-v3 | mlx-whisper | — | Slow | Catastrophic hallucination on long audio (2h+); not used |
| granite-speech | mlx-audio | 105% | Slow | Unusable on podcasts (hallucination loops) |

### Known Limitations

- **Whisper hallucination on silence**: Without anti-hallucination flags, Whisper
  can generate repetitive phantom text during silent passages. The pipeline
  applies flags to all Whisper models: `condition_on_previous_text=False`,
  `no_speech_threshold=0.2`, `compression_ratio_threshold=2.0`,
  `hallucination_silence_threshold=3.0`
- **Whisper large-v3**: Especially prone to hallucination on 2h+ recordings
- **Granite-speech**: Massive hallucination loops on podcast audio despite
  strong leaderboard WER (5.52%). Leaderboard scores don't transfer to podcasts.
- **Qwen3-ASR**: mlx-audio silently truncates long MP3 files; the pipeline
  converts to WAV first as a workaround

## Multi-Source Merge

The pipeline's merge step aligns YouTube captions and ASR output via wdiff,
then uses an LLM to resolve differences chunk by chunk. The `analysis.md` output
reports per-source coverage and retention for each run.

In practice, when audio quality is high (single speaker, studio recording),
YouTube and ASR agree closely (97-99% overlap). The merge adds the most
value on noisy or technical content where the sources diverge significantly.

## References

- [Use automatic captioning - YouTube Help](https://support.google.com/youtube/answer/6373554?hl=en)
- [The Technology Behind YouTube's Auto-Captioning System - DEV Community](https://dev.to/activejack/the-technology-behind-youtubes-auto-captioning-system-1o45)
- [Fixing YouTube's automatic captioning with AI - Tech, Chem, Scott](https://blog.hartleygroup.org/2025/12/05/fixing-youtubes-automatic-captioning-with-ai/)
