#!/usr/bin/env bash
# Adjudicate Rev16 files 3, 4, 9 with the local Kev server, then score every
# hypothesis (Kev merge + copied baselines) against the Rev16 references.
# Usage: ./run_rev16.sh [run-name]   (default: rev16-kev9b)
set -euo pipefail
cd "$(dirname "$0")"

RUN=${1:-rev16-kev9b}
TC=/Users/ringger/Projects/transcribe-critic
SRC=$TC/eval-runs/rev16-asr-backends
OUT=runs/$RUN
MODELS=parakeet,qwen3-asr,distil-large-v3
mkdir -p "$OUT" logs

for f in 3 4 9; do
  python3 kev_adjudicate.py \
    --src-dir "$SRC/file$f" --out-dir "$OUT/file$f" --models "$MODELS" \
    --ref-checkpoints "$SRC/file$f/ensemble_chunks_local_v2" \
    2>&1 | tee "logs/$RUN-file$f.log"
done

transcribe-critic-eval score "$TC/eval-data/rev16/manifest.json" \
  --run-dir "$OUT" --file-ids 3,4,9 --hypothesis all --metrics wer \
  --output "$OUT/results.md" 2>&1 | tee "logs/$RUN-score.log"
