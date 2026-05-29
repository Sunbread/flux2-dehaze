from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image
from torchvision import transforms

from .dataset import DEHAZE_PROMPT
from .types import MetadataItem, PathInput


def resize_and_save(
    src: PathInput | Image.Image,
    dst_path: Path,
    target_size: int = 512,
) -> None:
    if isinstance(src, Image.Image):
        img = src.convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    img_pil = transforms.ToPILImage()(transform(img))
    img_pil.save(dst_path)


def _process_pair(
    args: tuple[Path, Path, Path, Path, int, str],
) -> MetadataItem:
    """Worker function: process one hazy+clear pair. Takes paths, returns metadata dict."""
    hazy_src, clear_src, hazy_dst, clear_dst, target_size, pair_name = args
    resize_and_save(hazy_src, hazy_dst, target_size)
    resize_and_save(clear_src, clear_dst, target_size)
    return {
        "image": str(hazy_dst),
        "gt": str(clear_dst),
        "caption": DEHAZE_PROMPT,
    }


def preprocess_reside(input_dir: Path, output_dir: Path, split: str = "train") -> None:
    hazy_dir = input_dir / "hazy"
    clear_dir = input_dir / "clear"
    list_file = input_dir / "list" / f"{split}_list.txt"

    output_hazy = output_dir / "hazy"
    output_clear = output_dir / "clear"
    output_hazy.mkdir(parents=True, exist_ok=True)
    output_clear.mkdir(parents=True, exist_ok=True)

    metadata = []
    with open(list_file) as f:
        for idx, line in enumerate(f):
            hazy_name = line.strip()
            clear_name = hazy_name.replace("_b_", "_GT_")

            hazy_path = hazy_dir / hazy_name
            clear_path = clear_dir / clear_name

            if not clear_path.exists():
                clear_path = clear_dir / hazy_name.replace(".jpg", "_GT.jpg")

            if not hazy_path.exists() or not clear_path.exists():
                continue

            hazy_img = Image.open(hazy_path).convert("RGB")
            clear_img = Image.open(clear_path).convert("RGB")

            new_name = f"{split}_{idx:05d}.png"

            resize_and_save(hazy_img, output_hazy / new_name)
            resize_and_save(clear_img, output_clear / new_name)

            metadata.append({
                "image": str(output_hazy / new_name),
                "gt": str(output_clear / new_name),
                "caption": DEHAZE_PROMPT,
            })

    meta_path = output_dir / f"metadata_{split}.jsonl"
    with open(meta_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"Preprocessed {len(metadata)} pairs -> {meta_path}")


def preprocess_reside_standard(
    input_dir: Path,
    output_dir: Path,
    split: str = "train",
    workers: Optional[int] = None,
) -> None:
    """Preprocess RESIDE-Standard format using multiprocessing."""
    csv_path = input_dir / "metadata.csv"

    output_hazy = output_dir / "hazy"
    output_clear = output_dir / "clear"
    output_hazy.mkdir(parents=True, exist_ok=True)
    output_clear.mkdir(parents=True, exist_ok=True)

    # Gather all tasks first
    tasks = []
    pair_idx = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            clear_rel = row["clear_image_path"].strip()
            hazy_str = row["hazy_image_paths"].strip()

            clear_src = input_dir / clear_rel
            if not clear_src.exists():
                continue

            hazy_list = _parse_path_list(hazy_str)
            for hazy_rel in hazy_list:
                hazy_src = input_dir / hazy_rel.strip()
                if not hazy_src.exists():
                    continue
                new_name = f"{split}_{pair_idx:05d}.png"
                tasks.append((
                    hazy_src, clear_src,
                    output_hazy / new_name, output_clear / new_name,
                    512, new_name,
                ))
                pair_idx += 1

    if workers is None:
        workers = min(os.cpu_count() or 1, len(tasks))

    print(f"Processing {len(tasks)} pairs with {workers} workers...")
    metadata = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(_process_pair, tasks)):
            metadata.append(result)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(tasks)} pairs done")

    meta_path = output_dir / f"metadata_{split}.jsonl"
    with open(meta_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"Preprocessed {len(metadata)} pairs (RESIDE-Standard) -> {meta_path}")


def _parse_path_list(raw: str) -> list[str]:
    """Parse a string like "['hazy/1_1.png', 'hazy/1_2.png']" into a list of paths."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        return [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]
    return [raw]


def preprocess_nhhaze(
    input_dir: Path,
    output_dir: Path,
    workers: Optional[int] = None,
) -> None:
    hazy_dir = input_dir / "hazy"
    gt_dir = input_dir / "GT"

    output_hazy = output_dir / "hazy"
    output_gt = output_dir / "GT"
    output_hazy.mkdir(parents=True, exist_ok=True)
    output_gt.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, hazy_path in enumerate(sorted(hazy_dir.glob("*.png"))):
        gt_path = gt_dir / hazy_path.name
        if not gt_path.exists():
            continue
        new_name = f"nhhaze_{idx:03d}.png"
        tasks.append((
            hazy_path, gt_path,
            output_hazy / new_name, output_gt / new_name,
            512, new_name,
        ))

    if workers is None:
        workers = min(os.cpu_count() or 1, len(tasks))

    print(f"Processing {len(tasks)} NH-HAZE pairs with {workers} workers...")
    metadata = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_process_pair, tasks):
            metadata.append(result)

    meta_path = output_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"Preprocessed {len(metadata)} NH-HAZE pairs -> {meta_path}")


if __name__ == "__main__":
    preprocess_reside_standard(
        Path("data/raw/RESIDE_ITS"), Path("data/processed/RESIDE"), split="train",
    )
    preprocess_nhhaze(
        Path("data/raw/NH-HAZE"), Path("data/processed/NH-HAZE"),
    )