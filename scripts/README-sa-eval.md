# Sentence Alignment Evaluation Notes

## Test clips

| Clip | Duration | Source | Result |
|------|----------|--------|--------|
| 30s | 30s | `tests/fixtures/tao_30s.mp3` (project fixture) | Identical diffs (4/4 match) |
| 3min | 3min | Rev16 file14 — "Coming Soon: Season 2" (`eval-data/rev16/audio/14.wav`) | 20/22 common, 2 minor alignment differences at b_pos=139 and 252 |
| 10min | 10min | First 10min of Rev16 file27 — "What We Own is Sacred Because We Are Sacred" (`eval-data/rev16/audio/27.wav`, trimmed with ffmpeg) | Identical diffs (20/20 match) |

## Observations

- Boundary padding (5 words) eliminates spurious diffs from sentence boundary misalignment.
- On 30s and 10min clips, sentence-aligned diffs are identical to baseline.
- On 3min clip (file14), 2 of 22 diffs differ slightly in scope (sentence-aligned captures wider substitution span at b_pos=139, and splits a deletion differently at b_pos=252). Neither is clearly better or worse.
- No spurious boundary artifacts observed with padding enabled.

## Scripts

- `scripts/compare_alignment.py` — compare baseline vs sentence-aligned diffs on any output dir
