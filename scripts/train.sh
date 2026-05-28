#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
uv run python -m src.dehaze_lora.train --config configs/config.yaml "$@"
