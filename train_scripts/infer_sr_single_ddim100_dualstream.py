import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v12
from diffusion.model.nets.semantic_adapter import CLIPSemanticAdapter
from diffusion.model.nets.dual_lora import apply_dual_lora, set_dual_lora_scales
from diffusion.model.nets.adapter_cond import mask_adapter_cond


def randn_like_with_generator(tensor, generator):
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)


def get_lq_init_latents(z_lr, scheduler, steps, generator, strength, dtype):
    strength = float(max(0.0, min(1.0, strength)))
    scheduler.set_timesteps(steps, device=z_lr.device)
    timesteps = scheduler.timesteps
    start_index = int(round(strength * (len(timesteps) - 1)))
    start_index = min(max(start_index, 0), len(timesteps) - 1)
    t_start = timesteps[start_index]
    noise = randn_like_with_generator(z_lr, generator)
    if hasattr(scheduler, "add_noise"):
        latents = scheduler.add_noise(z_lr, noise, t_start)
    else:
        latents = z_lr + noise
    return latents.to(dtype=dtype), timesteps[start_index:]




def build_adapter_struct_input(lr_small_m11: torch.Tensor) -> torch.Tensor:
    return lr_small_m11.float().clamp(-1.0, 1.0)

def _load_pixart_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
    saved = set(saved_trainable.keys())
    model_keys = set(pixart.state_dict().keys())
    curr = pixart.state_dict()
    loaded, skipped_shape, missing_in_model = 0, 0, 0
    for k in sorted(saved):
        if k not in model_keys:
            missing_in_model += 1
            continue
        ckpt_t = saved_trainable[k]
        if tuple(ckpt_t.shape) == tuple(curr[k].shape):
            curr[k] = ckpt_t.to(dtype=curr[k].dtype)
            loaded += 1
        else:
            skipped_shape += 1
    pixart.load_state_dict(curr, strict=False)
    print(f"[{context}] pixart subset load: loaded={loaded}, model_miss={missing_in_model}, shape_skip={skipped_shape}, saved_total={len(saved)}")


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if (device == "cuda") else torch.float32

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    layer_cfg = ckpt.get("layer_config", {}) if isinstance(ckpt, dict) else {}
    anchor_layers = list(layer_cfg.get("anchor_layers", [2, 4, 6, 8]))

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        anchor_layers=anchor_layers,
        semantic_layers=[24, 25, 26, 27],
    ).to(device)

    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    load_state_dict_shape_compatible(pixart, base, context="base-pretrain")
    pixart.enable_sr_conditioning_layers(
        pixel_layers=list(layer_cfg.get("pixel_layers", anchor_layers)),
        lr_conv_layers=list(layer_cfg.get("lr_conv_layers", anchor_layers)),
        semantic_layers=list(layer_cfg.get("semantic_layers", [24, 25, 26, 27])),
        init_gate=-4.0,
    )
    sem0 = int(layer_cfg.get("semantic_layers", [24, 25, 26, 27])[0])
    sem_std = float(pixart.blocks[sem0].cross_attn.out_proj.weight.detach().float().std().item())
    print(f"[semantic-enable-check] layer={sem0} out_proj.weight.std={sem_std:.6e}")
    if sem_std <= 1e-8:
        raise RuntimeError("semantic layer out_proj.weight.std() is zero after enable; call order is wrong.")

    adapter = build_adapter_v12(
        in_channels=3,
        hidden_size=1152,
    ).to(device).float()
    sem_adapter = CLIPSemanticAdapter(
        encoder_name_or_path=args.semantic_encoder_name_or_path,
        hidden_size=1152,
        num_prompt_tokens=16,
    ).to(device).eval()

    saved_trainable = ckpt.get("pixart_trainable", {})

    has_dual_lora = any(
        ("pixel_lora_A" in k) or ("pixel_lora_B" in k) or ("semantic_lora_A" in k) or ("semantic_lora_B" in k)
        for k in saved_trainable.keys()
    )
    dual_lora = bool(layer_cfg.get("dual_lora", False))
    if dual_lora or has_dual_lora:
        apply_dual_lora(
            pixart,
            pixel_rank=int(layer_cfg.get("pixel_lora_rank", args.lora_rank)),
            semantic_rank=int(layer_cfg.get("semantic_lora_rank", args.lora_rank)),
            pixel_alpha=float(layer_cfg.get("pixel_lora_alpha", args.lora_alpha)),
            semantic_alpha=float(layer_cfg.get("semantic_lora_alpha", args.lora_alpha)),
        )
        set_dual_lora_scales(pixart, lambda_pix=float(args.lambda_pix), lambda_sem=float(args.lambda_sem))

    _load_pixart_subset_compatible(pixart, saved_trainable, context="infer")
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    sem_adapter_sd = ckpt.get("sem_adapter", None)
    if not bool(args.disable_semantic_branch):
        if isinstance(sem_adapter_sd, dict) and len(sem_adapter_sd) > 0:
            required_sem_keys = ("proj.weight", "proj.bias", "out_scale")
            missing_sem_keys = [k for k in required_sem_keys if k not in sem_adapter_sd]
            if missing_sem_keys:
                raise RuntimeError(f"Checkpoint sem_adapter missing required keys in infer: {missing_sem_keys}")
            sem_adapter.load_state_dict(sem_adapter_sd, strict=False)
        else:
            raise RuntimeError("Checkpoint missing sem_adapter while semantic branch is enabled in infer.")

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    hard_prompt_pack = torch.load(args.hard_prompt_embed_path, map_location="cpu")
    data_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }

    pixart.eval()
    adapter.eval()
    sem_adapter.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )

    lr_pil = Image.open(args.lr_image).convert("RGB")
    lr_pil = lr_pil.resize((args.crop_size, args.crop_size), Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    lr_in = norm(to_tensor(lr_pil)).unsqueeze(0).to(device)
    if args.input_is_lr_small:
        lr_small = lr_in
        lr = F.interpolate(lr_small, size=(args.crop_size, args.crop_size), mode="bicubic", align_corners=False, antialias=True)
    else:
        lr = lr_in
        lr_small = F.interpolate(lr, size=(args.crop_size // 4, args.crop_size // 4), mode="bicubic", align_corners=False, antialias=True)

    z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    if args.use_lq_init:
        latents, run_timesteps = get_lq_init_latents(
            z_lr.to(compute_dtype), scheduler, args.steps, gen, args.lq_init_strength, compute_dtype
        )
    else:
        scheduler.set_timesteps(args.steps, device=device)
        latents = randn_like_with_generator(z_lr.to(compute_dtype), gen)
        run_timesteps = scheduler.timesteps

    adapter_in = build_adapter_struct_input(lr_small).to(device=device, dtype=torch.float32)
    aug_level = torch.zeros((latents.shape[0],), device=device, dtype=compute_dtype)

    for t in run_timesteps:
        t_b = torch.tensor([t], device=device).expand(latents.shape[0])
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        cond = adapter(adapter_in, t_embed=t_embed.float())
        if not bool(args.disable_semantic_branch):
            cond["sem_tokens"] = sem_adapter(((adapter_in.clamp(-1, 1) + 1.0) * 0.5).float())
        if "ip_tokens" not in cond and "cond_tokens" not in cond:
            raise KeyError("adapter output missing ip_tokens/cond_tokens")
        if "local_entry_tokens" not in cond and "cond_map" not in cond:
            raise KeyError("adapter output missing local_entry_tokens/cond_map")
        with torch.autocast(device_type="cuda", dtype=compute_dtype) if device == "cuda" else torch.no_grad():
            drop_uncond = torch.ones(latents.shape[0], device=device, dtype=torch.long)
            drop_cond = torch.zeros(latents.shape[0], device=device, dtype=torch.long)
            model_in = latents.to(compute_dtype)
            y_cond = hard_prompt_pack["y"].to(device).repeat(latents.shape[0], 1, 1, 1)
            mask_cond = hard_prompt_pack["mask"].to(device).repeat(latents.shape[0], 1, 1, 1)
            y_uncond = null_pack["y"].to(device).repeat(latents.shape[0], 1, 1, 1)
            mask_uncond = null_pack["mask"].to(device).repeat(latents.shape[0], 1, 1, 1)
            if args.cfg_scale == 1.0:
                out = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y_cond,
                    aug_level=aug_level,
                    mask=mask_cond,
                    data_info=data_info,
                    adapter_cond=cond,
                    force_drop_ids=drop_cond,
                    sft_strength=args.sft_strength,
                )
            else:
                cond_zero = mask_adapter_cond(cond, torch.zeros((latents.shape[0],), device=device))
                out_uncond = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y_uncond,
                    aug_level=aug_level,
                    mask=mask_uncond,
                    data_info=data_info,
                    adapter_cond=cond_zero,
                    force_drop_ids=drop_uncond,
                    sft_strength=args.sft_strength,
                )
                out_cond = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y_cond,
                    aug_level=aug_level,
                    mask=mask_cond,
                    data_info=data_info,
                    adapter_cond=cond,
                    force_drop_ids=drop_cond,
                    sft_strength=args.sft_strength,
                )
                out = out_uncond + args.cfg_scale * (out_cond - out_uncond)

            if out.shape[1] != 4:
                raise RuntimeError(f"Expected 4-channel output, got {out.shape[1]}")
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    pred01 = ((pred[0].detach().cpu().float() + 1.0) * 0.5).clamp(0.0, 1.0)
    pred_pil = transforms.ToPILImage()(pred01)

    out_path = Path(args.out_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_pil.save(out_path)
    print(f"✅ Saved SR image to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image SR inference (DDIM, dualstream, validation-aligned settings)")
    parser.add_argument("--lr-image", type=str, required=True)
    parser.add_argument("--out-image", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--pixart-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--null-t5-embed-path", type=str, required=True)
    parser.add_argument("--hard-prompt-embed-path", type=str, required=True)
    parser.add_argument("--semantic-encoder-name-or-path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use-lq-init", dest="use_lq_init", action="store_true")
    parser.add_argument("--no-lq-init", dest="use_lq_init", action="store_false")
    parser.set_defaults(use_lq_init=True)
    parser.add_argument("--lq-init-strength", type=float, default=0.3)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sft_strength", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lambda-pix", type=float, default=1.0)
    parser.add_argument("--lambda-sem", type=float, default=1.0)
    parser.add_argument("--disable-semantic-branch", action="store_true")
    parser.add_argument("--input_is_lr_small", type=lambda x: str(x).lower() in ("1","true","yes","y"), default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
