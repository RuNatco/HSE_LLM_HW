#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${1:-exp9}"
case "$NAME" in
  exp1) BATCH=4; LR=5e-5; SCHEDULER=linear ;;
  exp2) BATCH=8; LR=1e-4; SCHEDULER=cosine ;;
  exp3) BATCH=8; LR=2e-4; SCHEDULER=cosine ;;
  exp8) BATCH=8; LR=3e-4; SCHEDULER=cosine ;;
  exp9) BATCH=8; LR=2.5e-4; SCHEDULER=cosine ;;
  exp10) BATCH=8; LR=2.5e-4; SCHEDULER=linear ;;
  *) echo "Choose: exp1, exp2, exp3, exp8, exp9, exp10" >&2; exit 2 ;;
esac
mkdir -p logs
docker run --rm --gpus all --shm-size=32g \
  -v "$PWD":/app -w /app \
  -e TRAIN_OUTPUT_DIR="./output_dir/repeat-${NAME}" \
  -e TRAIN_BATCH_SIZE="$BATCH" -e TRAIN_GRAD_ACCUM=1 \
  -e TRAIN_LR="$LR" -e TRAIN_LR_SCHEDULER="$SCHEDULER" \
  -e TRAIN_WARMUP_STEPS=200 -e TRAIN_OPTIM=adamw_torch_fused \
  -e TRAIN_BF16=true -e TRAIN_TF32=true -e TRAIN_TORCH_COMPILE=false \
  llm-hw1 python your_solution.py train 2>&1 | tee "logs/repeat-${NAME}.log"
