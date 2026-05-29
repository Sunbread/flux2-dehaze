"""Shared type definitions for the dehaze_lora package.

Tightened type boundaries: TypedDicts replace bare dict for data contracts;
TypeAlias for path inputs prevents str/Path ambiguity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, TypeAlias

import torch

PathInput: TypeAlias = str | Path | os.PathLike[str]


class MetadataItem(TypedDict):
    image: str
    gt: str
    caption: NotRequired[str]


class DatasetItem(TypedDict):
    hazy: torch.Tensor
    gt: torch.Tensor
    caption: str


class ValidationImageOutput(TypedDict):
    hazy: torch.Tensor
    gt: torch.Tensor
    cond: torch.Tensor
    reconstruction: torch.Tensor
    cfg: torch.Tensor


class ValidationBatchResult(TypedDict):
    images: list[ValidationImageOutput]
    psnr: list[float]
    ssim: list[float]


class MetricsDict(TypedDict):
    mean_psnr: float
    mean_ssim: float


class DenoiseOutput(TypedDict):
    cond: torch.Tensor
    reconstruction: torch.Tensor
    cfg: torch.Tensor


LoraTarget: TypeAlias = Literal["transformer", "qwen", "both"]


class ModelDict(TypedDict):
    vae: Any
    transformer: Any
    scheduler: Any
    text_encoder: Any
    tokenizer: Any
