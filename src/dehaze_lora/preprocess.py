from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

from .dataset import DEHAZE_PROMPT  # shared with dataset module


def resize_and_save(src_img: Image.Image, dst_path: Path, target_size: int = 512):
    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    img_tensor = transform(src_img)
    img_pil = transforms.ToPILImage()(img_tensor)
    img_pil.save(dst_path)


def preprocess_reside(input_dir: Path, output_dir: Path, split: str = "train"):
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


def preprocess_nhhaze(input_dir: Path, output_dir: Path):
    hazy_dir = input_dir / "hazy"
    gt_dir = input_dir / "GT"

    output_hazy = output_dir / "hazy"
    output_gt = output_dir / "GT"
    output_hazy.mkdir(parents=True, exist_ok=True)
    output_gt.mkdir(parents=True, exist_ok=True)

    metadata = []
    for idx, hazy_path in enumerate(sorted(hazy_dir.glob("*.png"))):
        gt_path = gt_dir / hazy_path.name
        if not gt_path.exists():
            continue

        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        new_name = f"nhhaze_{idx:03d}.png"
        resize_and_save(hazy_img, output_hazy / new_name)
        resize_and_save(gt_img, output_gt / new_name)

        metadata.append({
            "image": str(output_hazy / new_name),
            "gt": str(output_gt / new_name),
            "caption": DEHAZE_PROMPT,
        })

    meta_path = output_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"Preprocessed {len(metadata)} NH-HAZE pairs -> {meta_path}")


if __name__ == "__main__":
    preprocess_reside(
        Path("data/RESIDE/ITS"), Path("data/RESIDE/processed"), split="train",
    )
    preprocess_nhhaze(
        Path("data/NH-HAZE"), Path("data/NH-HAZE/processed"),
    )
