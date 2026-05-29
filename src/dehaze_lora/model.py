from __future__ import annotations

from typing import Any

import torch
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
)
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from peft import LoraConfig, get_peft_model, PeftModel

from .types import ModelDict

QWEN_LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRANSFORMER_LORA_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj"]

# Flux2 Klein uses Qwen3 hidden states from these layers, concatenated
QWEN3_HIDDEN_STATES_LAYERS = (9, 18, 27)


def _inject_lora(
    model: torch.nn.Module,
    rank: int,
    alpha: int,
    target_modules: list[str],
) -> PeftModel:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        inference_mode=False,
        lora_dropout=0.0,
    )
    return get_peft_model(model, lora_config)


def load_models(
    model_name: str = "black-forest-labs/FLUX.2-klein-base-9B",
    lora_target: str = "both",
    lora_rank: int = 16,
    lora_alpha: int = 8,
    transformer_device: str = "cuda:0",
    qwen_device: str = "cuda:1",
    gradient_checkpointing: bool = False,
) -> ModelDict:
    # VAE (always on transformer device, always frozen)
    vae = AutoencoderKLFlux2.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.requires_grad_(False)
    vae.to(transformer_device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    # Transformer on its own device
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled on transformer")

    if lora_target in ("transformer", "both"):
        transformer = _inject_lora(
            transformer,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=TRANSFORMER_LORA_MODULES,
        )
        print(f"Transformer LoRA injected: rank={lora_rank}, alpha={lora_alpha}")

    transformer.to(transformer_device)

    # Qwen3 on its own device
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.requires_grad_(False)

    if lora_target in ("qwen", "both"):
        text_encoder = _inject_lora(
            text_encoder,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=QWEN_LORA_MODULES,
        )
        print(f"Qwen3 LoRA injected: rank={lora_rank}, alpha={lora_alpha}")

    text_encoder.to(qwen_device)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_name, subfolder="tokenizer"
    )

    return {
        "vae": vae,
        "transformer": transformer,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }


def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    """Generate 4D RoPE position coordinates for text tokens.

    These are NOT vocabulary token IDs — they are (T,H,W,L) spatial
    coordinates fed into the RoPE position embedder. T/H/W are zero
    for text, L is 0..seq_len-1.

    torch.arange(1) produces int64 [0], keeping the cartesian product
    in int64 (text_ids convention in Flux2)."""
    B, L, _ = x.shape
    B = int(B)
    seq_len = int(L)
    out_ids: list[torch.Tensor] = []
    for i in range(B):
        t = torch.arange(1, device=x.device)
        h = torch.arange(1, device=x.device)
        w = torch.arange(1, device=x.device)
        l = torch.arange(seq_len, device=x.device)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """VAE-level patchify: 32ch 8x-down → 128ch 16x-down (2x2 patches)."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(B, C * 4, H // 2, W // 2)
    return latents


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse of _patchify_latents: 128ch 16x-down → 32ch 8x-down."""
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(B, C // 4, H * 2, W * 2)
    return latents


def patchify_and_make_ids(
    latent: torch.Tensor,
    patch_size: int = 1,
    index: float = 0.0,
    axes_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transformer-level patchify: BCHW → B(HW)C, plus RoPE position coordinates.

    Position IDs are (T,H,W,L) spatial coordinates for the RoPE embedder
    — NOT vocabulary token IDs. They tell the model where each image
    patch sits in the 4D grid.

    Flux2: patch_size=1, axes_dim=4.
    """
    import torch.nn.functional as F

    patch_size = int(patch_size)
    axes_dim = int(axes_dim)
    index = float(index)

    bs, c, h, w = latent.shape
    ph = pw = patch_size

    h_pad = ((h + ph - 1) // ph) * ph
    w_pad = ((w + pw - 1) // pw) * pw
    if h != h_pad or w != w_pad:
        latent = F.pad(latent, (0, w_pad - w, 0, h_pad - h))

    h_len = h_pad // ph
    w_len = w_pad // pw

    tokens = latent.reshape(bs, c, h_len, ph, w_len, pw)
    tokens = tokens.permute(0, 2, 4, 1, 3, 5)
    tokens = tokens.reshape(bs, h_len * w_len, c * ph * pw)

    ids = torch.zeros(1, h_len, w_len, axes_dim, device=latent.device, dtype=torch.float32)
    ids[:, :, :, 0] = index
    ids[:, :, :, 1] = torch.linspace(0, h_len - 1, steps=h_len, device=latent.device, dtype=torch.float32).unsqueeze(1)
    ids[:, :, :, 2] = torch.linspace(0, w_len - 1, steps=w_len, device=latent.device, dtype=torch.float32).unsqueeze(0)
    ids = ids.reshape(1, h_len * w_len, axes_dim).expand(bs, -1, -1)

    return tokens, ids


def unpatchify(
    tokens: torch.Tensor,
    h_orig: int,
    w_orig: int,
    patch_size: int = 1,
) -> torch.Tensor:
    """Reverse of patchify_and_make_ids: B(HW)C → BCHW."""
    h_orig = int(h_orig)
    w_orig = int(w_orig)
    patch_size = int(patch_size)

    bs, n, c = tokens.shape
    ph = pw = patch_size

    h_pad = ((h_orig + ph - 1) // ph) * ph
    w_pad = ((w_orig + pw - 1) // pw) * pw
    h_len = h_pad // ph
    w_len = w_pad // pw

    c_per = c // (ph * pw)
    tokens = tokens.reshape(bs, h_len, w_len, c_per, ph, pw)
    tokens = tokens.permute(0, 3, 1, 4, 2, 5)
    latent = tokens.reshape(bs, c_per, h_pad, w_pad)

    return latent[:, :, :h_orig, :w_orig]


def encode_vae_image(vae, image: torch.Tensor) -> torch.Tensor:
    """
    Encode image → patchified + BN-normalized latent.

    Flux2 VAE: 32ch, 8x downscale → patchify → 128ch, 16x downscale.
    BN normalization uses the VAE's internal BatchNorm running stats —
    required by the transformer which expects zero-mean unit-variance latents.

    VAE expects input in [-1, 1]; image is normalized from [0, 1].

    Uses latent_dist.mode() (deterministic mean, matching diffusers/comfy/
    simpletuner). VAE is frozen — no reason to sample noise from the posterior.
    """
    image = image * 2.0 - 1.0  # [0, 1] → [-1, 1]
    latents = vae.encode(image).latent_dist.mode()
    latents = _patchify_latents(latents)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = (latents - bn_mean) / bn_std

    return latents


def decode_vae_image(vae, latents: torch.Tensor) -> torch.Tensor:
    """Reverse of encode_vae_image: denormalize → unpatchify → VAE decode."""
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    latents = _unpatchify_latents(latents)
    image = vae.decode(latents).sample
    image = (image + 1.0) / 2.0  # [-1, 1] → [0, 1]
    return image


def encode_prompts(
    text_encoder: Any,
    tokenizer: Any,
    prompts: list[str],
    max_seq_len: int = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch-encode prompts → (prompt_embeds, text_ids).

    Applies Qwen3 chat template, tokenizes with padding, runs a single
    batched forward pass, and concatenates hidden states from layers
    (9, 18, 27) → joint_attention_dim (12288).
    """
    max_seq_len = int(max_seq_len)
    device = str(device)

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]

    tokens = tokenizer(
        texts,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    out = torch.stack(
        [output.hidden_states[k] for k in QWEN3_HIDDEN_STATES_LAYERS], dim=1
    )
    B, num_layers, L, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        B, L, num_layers * hidden_dim
    )

    text_ids = _prepare_text_ids(prompt_embeds)
    return prompt_embeds, text_ids
