import json
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

DEHAZE_PROMPT = (
    "Dehaze this image naturally. Remove fog, mist, smog, and gray atmospheric "
    "haze while preserving the original composition, subjects, perspective, "
    "camera angle, scene identity, and lighting direction. Restore clear "
    "visibility, realistic contrast, natural colors, crisp edges, and distant "
    "details. Keep the result photorealistic and clean, without over-sharpening, "
    "over-saturation, halos, artificial HDR, or changing any objects."
)


class DehazeDataset(Dataset):
    """
    Dehazing training dataset.

    90% samples use dehaze prompt, 10% caption set to "" (CFG unconditional branch).
    Empty caption is later wrapped by encode_prompt's chat template.
    """

    def __init__(
        self,
        metadata_path: str,
        caption_dropout_rate: float = 0.1,
        target_size: int = 512,
        dropout_seed: int = 0,
    ):
        self.metadata = [json.loads(l) for l in open(metadata_path)]
        self.caption_dropout_rate = caption_dropout_rate
        self.target_size = target_size
        self.dropout_seed = dropout_seed
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]

        hazy = Image.open(item["image"]).convert("RGB")
        gt = Image.open(item["gt"]).convert("RGB")

        sz = self.target_size
        hazy = transforms.Resize(
            (sz, sz), interpolation=transforms.InterpolationMode.BICUBIC
        )(hazy)
        gt = transforms.Resize(
            (sz, sz), interpolation=transforms.InterpolationMode.BICUBIC
        )(gt)

        hazy_tensor = self.transform(hazy)
        gt_tensor = self.transform(gt)

        caption = DEHAZE_PROMPT
        if random.Random(self.dropout_seed + idx).random() < self.caption_dropout_rate:
            caption = ""

        return {"hazy": hazy_tensor, "gt": gt_tensor, "caption": caption}


class DehazeValDataset(Dataset):
    """Validation set; uses per-sample caption from metadata, falling back to DEHAZE_PROMPT."""

    def __init__(self, metadata_path: str, target_size: int = 512):
        self.metadata = [json.loads(l) for l in open(metadata_path)]
        self.target_size = target_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        hazy = Image.open(item["image"]).convert("RGB")
        gt = Image.open(item["gt"]).convert("RGB")

        sz = self.target_size
        hazy = transforms.Resize(
            (sz, sz), interpolation=transforms.InterpolationMode.BICUBIC
        )(hazy)
        gt = transforms.Resize(
            (sz, sz), interpolation=transforms.InterpolationMode.BICUBIC
        )(gt)

        return {
            "hazy": self.transform(hazy),
            "gt": self.transform(gt),
            "caption": item.get("caption", DEHAZE_PROMPT),
        }
