#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
uv run python -c "
from src.dehaze_lora.preprocess import preprocess_reside, preprocess_nhhaze
from pathlib import Path

preprocess_reside(
    Path('data/RESIDE/ITS'),
    Path('data/RESIDE/processed'),
    split='train'
)
preprocess_nhhaze(
    Path('data/NH-HAZE'),
    Path('data/NH-HAZE/processed'),
)
"
