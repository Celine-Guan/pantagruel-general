#!/usr/bin/env bash
set -euo pipefail

# Configurable paths (edit as needed)
ROOT_DIR="/home/qguan/pantagruel/Pantagruel-eval/Word-sense-disambiguation/verbs"
DATA_DIR="/home/qguan/pantagruel/datasets/FSE-output"  # directory produced by prepare_data.py (contains train/, test/, targets)
OUTPUT_DIR_ROOT="/home/qguan/pantagruel/Pantagruel-eval/Word-sense-disambiguation/vsd-output"
SCORE_DIR_ROOT="/home/qguan/pantagruel/Pantagruel-eval/Word-sense-disambiguation/vsd-output_score"

# Runtime settings
PADDING=80
BATCHSIZE=32
DEVICE=0                 # GPU id for transformers; use -1 for CPU
CUDA_VISIBLE_DEVICES=0   # set which GPU to expose
HF_TOKEN=

# Model list
MODELS=(
  "PantagrueLLM/Text_Base_FR_OSCAR"
  "PantagrueLLM/Text_Base_FR_croissant"
)

# Ensure output roots exist
mkdir -p "$OUTPUT_DIR_ROOT"
mkdir -p "$SCORE_DIR_ROOT"

cd "$ROOT_DIR"

for MODEL in "${MODELS[@]}"; do
  # Create a filesystem-safe experiment name from model id
  EXP_NAME=$(echo "$MODEL" | sed 's|/|_|g; s|:|_|g; s|\.|_|g')

  echo "Running VSD for model: $MODEL (exp: $EXP_NAME)"

  # Output file bases are derived from exp_name by flue_vsd.py; we pass directories
  OUT_DIR="$OUTPUT_DIR_ROOT"
  SCORE_DIR="$SCORE_DIR_ROOT"
  SCORE_FILE="$SCORE_DIR/${EXP_NAME}.csv"
  mkdir -p "$OUT_DIR" "$SCORE_DIR"

  # Run experiment
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python flue_vsd.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --padding "$PADDING" \
    --batchsize "$BATCHSIZE" \
    --device "$DEVICE" \
    --output "$OUT_DIR" \
    --output_score "$SCORE_FILE"
    --hf_token "$HF_TOKEN"

  echo "Completed: $MODEL"
  echo "Vectors in: $OUT_DIR/${EXP_NAME}.train.vecs and $OUT_DIR/${EXP_NAME}.test.vecs"
  echo "Score in:   $SCORE_FILE"
  echo "---------------------------------------------"
done

echo "All models completed."


