import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json

import os
from pathlib import Path

from .model import (
    encode_prompts, encode_vae_image, decode_vae_image,
    patchify_and_make_ids, unpatchify,
)
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu


def load_inference_models(
    model_name="black-forest-labs/FLUX.2-klein-base-9B",
    transformer_lora_path=None,
    qwen_lora_path=None,
    device="cuda",
):
    vae = AutoencoderKLFlux2.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if transformer_lora_path:
        transformer = PeftModel.from_pretrained(transformer, transformer_lora_path)
        print(f"Loaded transformer LoRA from {transformer_lora_path}")
    transformer.requires_grad_(False)
    transformer.to(device)
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    return vae, transformer, scheduler


@torch.no_grad()
def _denoise_all_modes(
    transformer,
    scheduler,
    vae,
    hazy_latent,          # BCHW, already VAE-encoded + patchified + BN
    cond_embeds,          # (B, L, 12288)
    cond_text_ids,        # (B, L, 4)
    uncond_embeds,        # (B, L, 12288)
    uncond_text_ids,      # (B, L, 4)
    *,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    device: str = "cuda",
    noise_seed: int = 42,
) -> dict:
    """Run 3 denoising trajectories from shared noise: cond, reconstruction, CFG.

    Returns dict with keys 'cond', 'reconstruction', 'cfg' -- each a decoded
    RGB tensor of shape (B, 3, H', W').
    """
    B = hazy_latent.shape[0]
    ph = hazy_latent.shape[2]
    pw = hazy_latent.shape[3]

    transformer_config = transformer.config
    patch_size = getattr(transformer_config, "patch_size", 1)

    # Shared reference tokens
    ref_tokens, ref_ids = patchify_and_make_ids(
        hazy_latent, patch_size=patch_size, index=10.0,
    )

    # Shared starting noise
    image_seq_len = hazy_latent.shape[2] * hazy_latent.shape[3]
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    rng = torch.Generator(device).manual_seed(noise_seed)
    z0 = torch.randn_like(hazy_latent, generator=rng)

    def _fresh_scheduler():
        from diffusers import FlowMatchEulerDiscreteScheduler
        sched = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
        sched.set_timesteps(num_inference_steps, mu=mu)
        return sched

    # --- Mode 1: Conditional generation (prompt only, no CFG) ---
    sched = _fresh_scheduler()
    z_cond = z0.clone()
    for t in tqdm(sched.timesteps, desc="Cond", leave=False):
        timestep = (
            t.float().unsqueeze(0).expand(B)
            .to(device=device, dtype=torch.bfloat16)
            / scheduler.config.num_train_timesteps
        )
        noisy_tokens, noisy_ids = patchify_and_make_ids(
            z_cond, patch_size=patch_size, index=0.0,
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v = transformer(
                hidden_states=torch.cat([noisy_tokens, ref_tokens], dim=1),
                encoder_hidden_states=cond_embeds,
                timestep=timestep,
                img_ids=torch.cat([noisy_ids, ref_ids], dim=1),
                txt_ids=cond_text_ids,
                return_dict=False,
            )[0]
        v = v[:, : noisy_tokens.shape[1], :]
        v = unpatchify(v, ph, pw, patch_size=patch_size)
        z_cond = sched.step(v, t, z_cond).prev_sample
    cond_img = decode_vae_image(vae, z_cond)

    # --- Mode 2: Reconstruction (uncond only) ---
    sched = _fresh_scheduler()
    z_recon = z0.clone()
    for t in tqdm(sched.timesteps, desc="Recon", leave=False):
        timestep = (
            t.float().unsqueeze(0).expand(B)
            .to(device=device, dtype=torch.bfloat16)
            / scheduler.config.num_train_timesteps
        )
        noisy_tokens, noisy_ids = patchify_and_make_ids(
            z_recon, patch_size=patch_size, index=0.0,
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v = transformer(
                hidden_states=torch.cat([noisy_tokens, ref_tokens], dim=1),
                encoder_hidden_states=uncond_embeds,
                timestep=timestep,
                img_ids=torch.cat([noisy_ids, ref_ids], dim=1),
                txt_ids=uncond_text_ids,
                return_dict=False,
            )[0]
        v = v[:, : noisy_tokens.shape[1], :]
        v = unpatchify(v, ph, pw, patch_size=patch_size)
        z_recon = sched.step(v, t, z_recon).prev_sample
    recon_img = decode_vae_image(vae, z_recon)

    # --- Mode 3: CFG ---
    sched = _fresh_scheduler()
    z_cfg = z0.clone()
    for t in tqdm(sched.timesteps, desc="CFG", leave=False):
        timestep = (
            t.float().unsqueeze(0).expand(B)
            .to(device=device, dtype=torch.bfloat16)
            / scheduler.config.num_train_timesteps
        )
        noisy_tokens, noisy_ids = patchify_and_make_ids(
            z_cfg, patch_size=patch_size, index=0.0,
        )
        img_hidden = torch.cat([noisy_tokens, ref_tokens], dim=1)
        img_ids_combined = torch.cat([noisy_ids, ref_ids], dim=1)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v_cond = transformer(
                hidden_states=img_hidden,
                encoder_hidden_states=cond_embeds,
                timestep=timestep,
                img_ids=img_ids_combined,
                txt_ids=cond_text_ids,
                return_dict=False,
            )[0]
        v_cond = v_cond[:, : noisy_tokens.shape[1], :]
        v_cond = unpatchify(v_cond, ph, pw, patch_size=patch_size)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v_uncond = transformer(
                hidden_states=img_hidden,
                encoder_hidden_states=uncond_embeds,
                timestep=timestep,
                img_ids=img_ids_combined,
                txt_ids=uncond_text_ids,
                return_dict=False,
            )[0]
        v_uncond = v_uncond[:, : noisy_tokens.shape[1], :]
        v_uncond = unpatchify(v_uncond, ph, pw, patch_size=patch_size)

        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        z_cfg = sched.step(v_cfg, t, z_cfg).prev_sample
    cfg_img = decode_vae_image(vae, z_cfg)

    return {"cond": cond_img, "reconstruction": recon_img, "cfg": cfg_img}


@torch.no_grad()
def run_validation_batch(
    vae,
    transformer,
    scheduler,
    text_encoder,
    tokenizer,
    val_subset: list,         # list of metadata dicts (typically 4 items)
    *,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    max_seq_len: int = 512,
    transformer_device: str = "cuda:0",
    qwen_device: str = "cuda:1",
    seed: int = 42,
) -> dict:
    """Run 3-mode validation on a batch of images.

    Args:
        val_subset: List of metadata dicts with 'image', 'gt', 'caption' keys.
        seed: Validation noise seed (deterministic per step).

    Returns:
        dict with keys: 'images' (list of dicts with hazy/gt/cond/recon/cfg tensors),
        'psnr' (list of floats), 'ssim' (list of floats).
    """
    from .dataset import DEHAZE_PROMPT
    from PIL import Image
    from torchvision import transforms

    bsz = len(val_subset)
    if bsz == 0:
        return {"images": [], "psnr": [], "ssim": []}

    # Load and resize images
    hazy_imgs = []
    gt_imgs = []
    target_size = 512
    for item in val_subset:
        hazy_pil = Image.open(item["image"]).convert("RGB")
        gt_pil = Image.open(item["gt"]).convert("RGB")
        hazy_tensor = transforms.ToTensor()(
            transforms.Resize(
                (target_size, target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )(hazy_pil)
        )
        gt_tensor = transforms.ToTensor()(
            transforms.Resize(
                (target_size, target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )(gt_pil)
        )
        hazy_imgs.append(hazy_tensor)
        gt_imgs.append(gt_tensor)

    hazy_batch = torch.stack(hazy_imgs).to(transformer_device, dtype=torch.bfloat16)
    gt_batch = torch.stack(gt_imgs)

    # VAE encode all hazy images (batched, once)
    hazy_latent = encode_vae_image(vae, hazy_batch)

    # Text encode cond + uncond (once per batch, same prompt for all images)
    ce, cti = encode_prompts(
        text_encoder, tokenizer,
        [DEHAZE_PROMPT] * bsz,
        max_seq_len, qwen_device, torch.bfloat16,
    )
    ue, uti = encode_prompts(
        text_encoder, tokenizer,
        [""] * bsz,
        max_seq_len, qwen_device, torch.bfloat16,
    )
    ce = ce.to(transformer_device)
    cti = cti.to(transformer_device)
    ue = ue.to(transformer_device)
    uti = uti.to(transformer_device)

    # Run 3-mode denoising
    results = _denoise_all_modes(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        hazy_latent=hazy_latent,
        cond_embeds=ce,
        cond_text_ids=cti,
        uncond_embeds=ue,
        uncond_text_ids=uti,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=transformer_device,
        noise_seed=seed,
    )

    # Compute PSNR/SSIM for CFG output vs GT
    psnr_list, ssim_list = [], []
    image_outputs = []
    for i in range(bsz):
        cfg_np = results["cfg"][i].permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()
        gt_np = gt_batch[i].permute(1, 2, 0).numpy()

        psnr_list.append(psnr(gt_np, cfg_np, data_range=1.0))
        ssim_list.append(ssim(gt_np, cfg_np, data_range=1.0, channel_axis=2))

        image_outputs.append({
            "hazy": hazy_batch[i].cpu(),
            "gt": gt_batch[i],
            "cond": results["cond"][i].cpu(),
            "reconstruction": results["reconstruction"][i].cpu(),
            "cfg": results["cfg"][i].cpu(),
        })

    return {"images": image_outputs, "psnr": psnr_list, "ssim": ssim_list}


def validate(
    val_metadata_path: str,
    model_name: str = "black-forest-labs/FLUX.2-klein-base-9B",
    transformer_lora_path: str = None,
    qwen_lora_path: str = None,
    output_dir: str = "outputs/eval",
    device: str = "cuda",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    batch_size: int = 4,
):
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

    all_metadata = [json.loads(l) for l in open(val_metadata_path)]
    if not all_metadata:
        raise ValueError(f"Validation dataset is empty: {val_metadata_path}")

    print(f"Loaded {len(all_metadata)} validation samples")

    vae, transformer, scheduler = load_inference_models(
        model_name, transformer_lora_path, qwen_lora_path, device,
    )

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval()

    if qwen_lora_path:
        text_encoder = PeftModel.from_pretrained(text_encoder, qwen_lora_path)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_name, subfolder="tokenizer",
    )

    os.makedirs(output_dir, exist_ok=True)

    all_psnr = []
    all_ssim = []

    for i in tqdm(range(0, len(all_metadata), batch_size), desc="Validating"):
        batch = all_metadata[i:i + batch_size]
        results = run_validation_batch(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            val_subset=batch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            transformer_device=device,
            qwen_device=device,
            seed=42 + i // batch_size,
        )
        all_psnr.extend(results["psnr"])
        all_ssim.extend(results["ssim"])

    results_dict = {
        "mean_psnr": float(np.mean(all_psnr)),
        "mean_ssim": float(np.mean(all_ssim)),
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(
        f"Mean PSNR: {results_dict['mean_psnr']:.2f} dB | "
        f"Mean SSIM: {results_dict['mean_ssim']:.4f}"
    )
    return results_dict
