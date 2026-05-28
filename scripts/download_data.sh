#!/bin/bash
set -e
# Download and preprocess dehazing datasets (RESIDE ITS + NH-HAZE).
# Prerequisites: uv sync --extra dev, Kaggle API key at ~/.kaggle/kaggle.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA="$PROJECT_DIR/data"
RAW="$DATA/raw"

mkdir -p "$RAW"

echo "=== 1/4 Downloading NH-HAZE (330 MB) ==="
if [ ! -d "$RAW/NH-HAZE/hazy" ]; then
    NH_URL="https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip"
    TMP_ZIP="$DATA/NH-HAZE.zip"
    wget -q --show-progress -O "$TMP_ZIP" "$NH_URL"
    TMP_DIR="$DATA/nh_haze_tmp"
    unzip -qo "$TMP_ZIP" -d "$TMP_DIR"
    mkdir -p "$RAW/NH-HAZE/hazy" "$RAW/NH-HAZE/GT"
    # NH-HAZE zip is flat: 01_hazy.png, 01_GT.png, ...
    for f in "$TMP_DIR"/NH-HAZE/*_hazy.png; do
        name=$(basename "$f" | sed 's/_hazy//')
        mv "$f" "$RAW/NH-HAZE/hazy/$name"
    done
    for f in "$TMP_DIR"/NH-HAZE/*_GT.png; do
        name=$(basename "$f" | sed 's/_GT//')
        mv "$f" "$RAW/NH-HAZE/GT/$name"
    done
    rm -rf "$TMP_ZIP" "$TMP_DIR"
    echo "NH-HAZE: $(ls "$RAW/NH-HAZE/hazy" | wc -l) pairs"
else
    echo "NH-HAZE already exists, skipping"
fi

echo ""
echo "=== 2/4 Downloading RESIDE ITS (~5 GB via Kaggle) ==="
if [ ! -d "$RAW/RESIDE_ITS/clear" ]; then
    uv run kaggle datasets download balraj98/indoor-training-set-its-residestandard \
        -p "$DATA/reside_its_tmp"
    unzip -qo "$DATA/reside_its_tmp/indoor-training-set-its-residestandard.zip" \
        -d "$RAW/RESIDE_ITS"
    rm -rf "$DATA/reside_its_tmp"
    echo "RESIDE ITS: $(ls "$RAW/RESIDE_ITS/clear" | wc -l) clear, $(ls "$RAW/RESIDE_ITS/hazy" | wc -l) hazy"
else
    echo "RESIDE ITS already exists, skipping"
fi

echo ""
echo "=== 3/4 Preprocessing ==="
uv run python -m src.dehaze_lora.preprocess

echo ""
echo "=== 4/4 Done ==="
echo "Train metadata: $DATA/processed/RESIDE/metadata_train.jsonl"
echo "Val metadata:   $DATA/processed/NH-HAZE/metadata.jsonl"
