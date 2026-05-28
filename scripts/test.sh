#!/bin/bash
# Evaluate a trained checkpoint on NH-HAZE (real data, OOD test).
#
# Usage:
#   bash scripts/test.sh <checkpoint_dir> [guidance_scale] [num_inference_steps]

set -e

CHECKPOINT="${1:?Usage: $0 <checkpoint_dir> [guidance_scale] [num_inference_steps]}"
GUIDANCE_SCALE="${2:-3.5}"
NUM_STEPS="${3:-28}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

TRANSFORMER_LORA="$CHECKPOINT/transformer_lora"
QWEN_LORA="$CHECKPOINT/qwen_lora"

QWEN_KWARG=""
if [ -d "$QWEN_LORA" ]; then
    QWEN_KWARG=", qwen_lora_path='$QWEN_LORA'"
fi

uv run python -c "
from src.dehaze_lora.validate import validate
validate(
    val_metadata_path='data/processed/NH-HAZE/metadata.jsonl',
    transformer_lora_path='$TRANSFORMER_LORA'${QWEN_KWARG},
    output_dir='$CHECKPOINT/nhhaze_eval',
    device='cuda:0',
    guidance_scale=$GUIDANCE_SCALE,
    num_inference_steps=$NUM_STEPS,
)
"
