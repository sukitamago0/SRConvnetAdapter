import os
import sys
import csv
import json
import math
import glob
import random
import argparse
import re
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import lpips
from diffusers import AutoencoderKL, DDIMScheduler

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v12

USE_ADAPTIVE_TEXT_PROMPT = True
ADAPTIVE_PROMPT_CACHE_ROOT = os.getenv("ADAPTIVE_PROMPT_CACHE_ROOT", "")



_SOBEL_X = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)


def _to_luma01(img_m11: torch.Tensor) -> torch.Tensor:
    img01 = (img_m11.float() + 1.0) * 0.5
    r = img01[:, 0:1]; g = img01[:, 1:2]; b = img01[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)


def _harris_response(gray01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    kx = _SOBEL_X.to(device=gray01.device)
    ky = _SOBEL_Y.to(device=gray01.device)
    ix = F.conv2d(gray01, kx, padding=1)
    iy = F.conv2d(gray01, ky, padding=1)
    ixx = F.avg_pool2d(ix * ix, kernel_size=3, stride=1, padding=1)
    iyy = F.avg_pool2d(iy * iy, kernel_size=3, stride=1, padding=1)
    ixy = F.avg_pool2d(ix * iy, kernel_size=3, stride=1, padding=1)
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    r = det - 0.04 * trace * trace
    r_flat = r.flatten(1)
    r_min = r_flat.min(dim=1, keepdim=True)[0][:, :, None]
    r_max = r_flat.max(dim=1, keepdim=True)[0][:, :, None]
    return ((r - r_min) / (r_max - r_min + eps)).clamp(0.0, 1.0)


def build_component_masks_from_hr(hr_m11: torch.Tensor, corner_q: float = 0.95, edge_q: float = 0.80):
    gray = _to_luma01(hr_m11)
    harris = _harris_response(gray)
    grad_x = F.conv2d(gray, _SOBEL_X.to(device=gray.device), padding=1)
    grad_y = F.conv2d(gray, _SOBEL_Y.to(device=gray.device), padding=1)
    grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
    b = gray.shape[0]
    c_th = torch.quantile(harris.flatten(1), corner_q, dim=1, keepdim=True)
    e_th = torch.quantile(grad_mag.flatten(1), edge_q, dim=1, keepdim=True)
    m_corner = (harris >= c_th.view(b, 1, 1, 1)).float()
    m_edge = (grad_mag >= e_th.view(b, 1, 1, 1)).float() * (1.0 - m_corner)
    m_flat = 1.0 - (m_edge + m_corner).clamp(0.0, 1.0)
    m_flat = F.avg_pool2d(m_flat, kernel_size=3, stride=1, padding=1)
    m_edge = F.avg_pool2d(m_edge, kernel_size=3, stride=1, padding=1)
    m_corner = F.avg_pool2d(m_corner, kernel_size=3, stride=1, padding=1)
    norm = (m_flat + m_edge + m_corner).clamp_min(1e-6)
    return m_flat / norm, m_edge / norm, m_corner / norm

def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def reorder_to_y_and_shave(pred01: torch.Tensor, hr01: torch.Tensor, crop_border: int = 4):
    py = rgb01_to_y01(pred01).clamp(0.0, 1.0)
    hy = rgb01_to_y01(hr01).clamp(0.0, 1.0)
    if crop_border > 0:
        py = py[..., crop_border:-crop_border, crop_border:-crop_border]
        hy = hy[..., crop_border:-crop_border, crop_border:-crop_border]
    return py, hy


def calculate_psnr_y(pred01: torch.Tensor, hr01: torch.Tensor, crop_border: int = 4):
    py, hy = reorder_to_y_and_shave(pred01, hr01, crop_border=crop_border)
    mse = torch.mean((py - hy) ** 2)
    return float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())


def _ssim_per_channel(img1: torch.Tensor, img2: torch.Tensor):
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    kernel = torch.tensor(
        [[0.0001, 0.0007, 0.0022, 0.0039, 0.0044, 0.0039, 0.0022, 0.0007, 0.0001],
         [0.0007, 0.0050, 0.0148, 0.0262, 0.0297, 0.0262, 0.0148, 0.0050, 0.0007],
         [0.0022, 0.0148, 0.0439, 0.0779, 0.0885, 0.0779, 0.0439, 0.0148, 0.0022],
         [0.0039, 0.0262, 0.0779, 0.1382, 0.1570, 0.1382, 0.0779, 0.0262, 0.0039],
         [0.0044, 0.0297, 0.0885, 0.1570, 0.1784, 0.1570, 0.0885, 0.0297, 0.0044],
         [0.0039, 0.0262, 0.0779, 0.1382, 0.1570, 0.1382, 0.0779, 0.0262, 0.0039],
         [0.0022, 0.0148, 0.0439, 0.0779, 0.0885, 0.0779, 0.0439, 0.0148, 0.0022],
         [0.0007, 0.0050, 0.0148, 0.0262, 0.0297, 0.0262, 0.0148, 0.0050, 0.0007],
         [0.0001, 0.0007, 0.0022, 0.0039, 0.0044, 0.0039, 0.0022, 0.0007, 0.0001]],
        dtype=img1.dtype, device=img1.device
    ).view(1, 1, 9, 9)
    mu1 = F.conv2d(img1, kernel, padding=4)
    mu2 = F.conv2d(img2, kernel, padding=4)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=4) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=4) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=4) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return ssim_map.mean()


def calculate_ssim_y(pred01: torch.Tensor, hr01: torch.Tensor, crop_border: int = 4):
    py, hy = reorder_to_y_and_shave(pred01, hr01, crop_border=crop_border)
    return float(_ssim_per_channel(py, hy).item())


def randn_like_with_generator(tensor, generator):
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)


def load_prompt_pack(path: str, pack_name: str = "prompt") -> dict:
    pack = torch.load(path, map_location="cpu")
    legacy_converted = False
    if "y" not in pack:
        if "hidden" in pack:
            pack["y"] = pack["hidden"].unsqueeze(1)
            legacy_converted = True
        else:
            raise KeyError(f"Invalid {pack_name} prompt pack: missing both 'y' and legacy 'hidden'")
    if "mask" not in pack:
        if "attention_mask" in pack:
            pack["mask"] = pack["attention_mask"].unsqueeze(1).unsqueeze(1)
            legacy_converted = True
        else:
            raise KeyError(f"Invalid {pack_name} prompt pack: missing both 'mask' and legacy 'attention_mask'")
    if pack["y"].dim() != 4:
        raise RuntimeError(f"Invalid {pack_name} prompt pack: y must be 4D, got shape={tuple(pack['y'].shape)}")
    if pack["mask"].dim() != 4:
        raise RuntimeError(f"Invalid {pack_name} prompt pack: mask must be 4D, got shape={tuple(pack['mask'].shape)}")
    pack["_legacy_converted"] = bool(legacy_converted)
    return pack


def make_sample_key(path: str) -> str:
    p = os.path.splitext(str(path))[0].replace("\\", "/").strip("/")
    return p.replace("/", "__")


def load_adaptive_prompt_batch(sample_keys, cache_root: str, cache_mem: dict):
    packs = []
    hits = 0
    tok_counts = []
    nonpad_ratios = []
    for key in sample_keys:
        pack = cache_mem.get(key, None)
        if pack is None:
            pth = os.path.join(cache_root, f"{key}.pth")
            if os.path.exists(pth):
                pack = load_prompt_pack(pth, pack_name="adaptive")
                cache_mem[key] = pack
                hits += 1
        else:
            hits += 1
        packs.append(pack)
        if pack is not None:
            mask = pack["mask"].detach().float()
            tok_counts.append(float(mask.sum().item()))
            nonpad_ratios.append(float(mask.mean().item()))
    hit_rate = float(hits / max(1, len(sample_keys)))
    avg_tok_count = float(sum(tok_counts) / max(1, len(tok_counts))) if len(tok_counts) > 0 else 0.0
    avg_nonpad_ratio = float(sum(nonpad_ratios) / max(1, len(nonpad_ratios))) if len(nonpad_ratios) > 0 else 0.0
    return packs, hit_rate, avg_tok_count, avg_nonpad_ratio


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


def center_crop_aligned_pair(lr_pil: Image.Image, hr_pil: Image.Image, scale: int = 4):
    wl, hl = lr_pil.size
    wh, hh = hr_pil.size
    h2 = min(hh, hl * scale)
    w2 = min(wh, wl * scale)
    h2 = (h2 // scale) * scale
    w2 = (w2 // scale) * scale
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Invalid aligned size with LR={lr_pil.size}, HR={hr_pil.size}")
    hr_top = (hh - h2) // 2
    hr_left = (wh - w2) // 2
    lr_h2 = h2 // scale
    lr_w2 = w2 // scale
    lr_top = (hl - lr_h2) // 2
    lr_left = (wl - lr_w2) // 2
    hr_aligned = TF.crop(hr_pil, hr_top, hr_left, h2, w2)
    lr_aligned = TF.crop(lr_pil, lr_top, lr_left, lr_h2, lr_w2)
    return lr_aligned, hr_aligned


def build_adapter_struct_input(lr_small_m11: torch.Tensor) -> torch.Tensor:
    return lr_small_m11.float().clamp(-1.0, 1.0)


def load_state_dict_shape_compatible(model: nn.Module, state_dict: dict, context: str = "load"):
    curr = model.state_dict()
    filt = {}
    skipped = []
    for k, v in state_dict.items():
        if k in curr and tuple(v.shape) == tuple(curr[k].shape):
            filt[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filt, strict=False)
    print(f"[{context}] compatible load: loaded={len(filt)}, skipped_shape_or_missing={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}")
    if len(skipped) > 0:
        print(f"[{context}] skipped examples: {skipped[:5]}")
    return missing, unexpected, skipped


def infer_lora_rank_from_state_dict(state_dict: dict):
    """
    Infer LoRA rank from saved lora_A / lora_B tensors.
    Returns int or None.
    """
    if not isinstance(state_dict, dict):
        return None

    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        if "lora_A" in k and v.ndim == 2:
            return int(v.shape[0])
        if "lora_B" in k and v.ndim == 2:
            return int(v.shape[1])
    return None



def _block_id_from_name(name: str):
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _lora_target_kind(module_name: str):
    if ("attn.qkv" in module_name) or ("attn.proj" in module_name):
        return "attn"
    return None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)


def apply_lora(model, lora_rank: int = 4, lora_alpha: float = 4.0):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        block_id = _block_id_from_name(name)
        if block_id is None:
            continue
        kind = _lora_target_kind(name)
        if kind is None:
            continue
        if not (0 <= block_id <= 27):
            continue
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        child = name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, int(lora_rank), alpha=float(lora_alpha)))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={int(lora_rank)}, alpha={float(lora_alpha)}).")


class RealSRValPairedDataset(Dataset):
    # copied from train_sigma_sr_vpred_dualstream.py validation behavior
    def __init__(self, roots, crop_size=512):
        self.crop_size = int(crop_size)
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.pairs = []

        for root in roots:
            root = str(root)
            if not os.path.isdir(root):
                continue
            for hr_path in sorted(glob.glob(os.path.join(root, "*_HR.png"))):
                lr_path = hr_path.replace("_HR.png", "_LR4.png")
                if os.path.exists(lr_path):
                    self.pairs.append((hr_path, lr_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No RealSR validation pairs found under roots={roots}. Expected '*_HR.png' and '*_LR4.png'."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        lr_pil = Image.open(lr_path).convert("RGB")

        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
        lr_crop = TF.center_crop(lr_aligned, (self.crop_size // 4, self.crop_size // 4))
        hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
        lr_up = TF.resize(
            lr_crop,
            (self.crop_size, self.crop_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up))
        lr_small_tensor = self.norm(self.to_tensor(lr_crop))
        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "lr_small": lr_small_tensor,
            "hr_path": hr_path,
            "lr_path": lr_path,
            "image_name": Path(hr_path).name,
            "sample_key": make_sample_key(hr_path),
        }


class DRealSRPairedDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, crop_size: int = 512):
        self.hr_dir = str(hr_dir)
        self.lr_dir = str(lr_dir)
        self.crop_size = int(crop_size)
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.pairs = []

        if not os.path.isdir(self.hr_dir):
            raise FileNotFoundError(f"DRealSR HR dir not found: {self.hr_dir}")
        if not os.path.isdir(self.lr_dir):
            raise FileNotFoundError(f"DRealSR LR dir not found: {self.lr_dir}")

        hr_paths = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        for hr_path in hr_paths:
            name = Path(hr_path).name
            if not name.endswith("_x4.png"):
                continue
            lr_name = name.replace("_x4.png", "_x1.png")
            lr_path = os.path.join(self.lr_dir, lr_name)
            if not os.path.exists(lr_path):
                print(f"⚠️ DRealSR pair missing LR, skipped: hr={hr_path}, expected_lr={lr_path}")
                continue
            self.pairs.append((hr_path, lr_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No DRealSR pairs found from hr_dir={self.hr_dir}, lr_dir={self.lr_dir} with *_x4.png -> *_x1.png mapping"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        lr_pil = Image.open(lr_path).convert("RGB")

        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
        lr_crop = TF.center_crop(lr_aligned, (self.crop_size // 4, self.crop_size // 4))
        hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
        lr_up = TF.resize(
            lr_crop,
            (self.crop_size, self.crop_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up))
        lr_small_tensor = self.norm(self.to_tensor(lr_crop))
        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "lr_small": lr_small_tensor,
            "hr_path": hr_path,
            "lr_path": lr_path,
            "image_name": Path(hr_path).name,
            "sample_key": make_sample_key(hr_path),
        }


class MetricSuite:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='vgg').to("cpu").eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

        self.dists_fn = None
        self.iqa = {}

        try:
            from DISTS_pytorch import DISTS
            self.dists_fn = DISTS().to("cpu").eval()
            for p in self.dists_fn.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print(f"⚠️ DISTS unavailable, will output NaN: {e}")

        try:
            import pyiqa
            for name in ["maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign"]:
                try:
                    self.iqa[name] = pyiqa.create_metric(name, device="cpu")
                except Exception as e_metric:
                    print(f"⚠️ pyiqa metric '{name}' unavailable, will output NaN: {e_metric}")
        except Exception as e:
            print(f"⚠️ pyiqa unavailable, no-reference IQA metrics will output NaN: {e}")

    @torch.no_grad()
    def compute(self, pred_m11: torch.Tensor, hr_m11: torch.Tensor):
        pred_cpu = pred_m11.detach().to("cpu", dtype=torch.float32)
        hr_cpu = hr_m11.detach().to("cpu", dtype=torch.float32)
        pred01 = (pred_cpu + 1.0) / 2.0
        hr01 = (hr_cpu + 1.0) / 2.0

        out = {
            "psnr": calculate_psnr_y(pred01, hr01, crop_border=4),
            "ssim": calculate_ssim_y(pred01, hr01, crop_border=4),
            "lpips": float(self.lpips_fn(pred_cpu, hr_cpu).mean().item()),
            "dists": float("nan"),
            "maniqa": float("nan"),
            "musiq": float("nan"),
            "clipiqa": float("nan"),
            "liqe": float("nan"),
            "topiq_nr": float("nan"),
            "qalign": float("nan"),
        }

        if self.dists_fn is not None:
            try:
                out["dists"] = float(self.dists_fn(pred01, hr01).mean().item())
            except Exception as e:
                print(f"⚠️ DISTS compute failed, writing NaN: {e}")

        pred01_clamp = pred01.clamp(0.0, 1.0)
        for name in ["maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign"]:
            fn = self.iqa.get(name, None)
            if fn is None:
                continue
            try:
                val = fn(pred01_clamp)
                out[name] = float(val.detach().float().mean().item())
            except Exception as e:
                print(f"⚠️ {name} compute failed, writing NaN: {e}")

        return out

    @torch.no_grad()
    def compute_component(self, pred_m11: torch.Tensor, hr_m11: torch.Tensor, mask: torch.Tensor):
        pred_cpu = pred_m11.detach().to("cpu", dtype=torch.float32)
        hr_cpu = hr_m11.detach().to("cpu", dtype=torch.float32)
        mask_cpu = mask.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)

        pred01 = (pred_cpu + 1.0) / 2.0
        hr01 = (hr_cpu + 1.0) / 2.0
        py = rgb01_to_y01(pred01)[..., 4:-4, 4:-4]
        hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
        my = mask_cpu[..., 4:-4, 4:-4]

        mse = ((my * (py - hy)) ** 2).sum() / (my.sum().clamp_min(1e-6))
        psnr_v = float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())

        mu_x = (my * py).sum() / my.sum().clamp_min(1e-6)
        mu_y = (my * hy).sum() / my.sum().clamp_min(1e-6)
        vx = (my * (py - mu_x) ** 2).sum() / my.sum().clamp_min(1e-6)
        vy = (my * (hy - mu_y) ** 2).sum() / my.sum().clamp_min(1e-6)
        cxy = (my * (py - mu_x) * (hy - mu_y)).sum() / my.sum().clamp_min(1e-6)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_v = float((((2 * mu_x * mu_y + c1) * (2 * cxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (vx + vy + c2))).item())

        pred_masked = pred_cpu * mask_cpu
        hr_masked = hr_cpu * mask_cpu
        lpips_v = float(self.lpips_fn(pred_masked, hr_masked).mean().item())

        return {"psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v}


def build_model_and_assets(args, device, compute_dtype):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if "adapter" not in ckpt:
        raise KeyError("Checkpoint must contain key: adapter")

    pixart_state = ckpt.get("pixart_trainable", ckpt.get("pixart_keep", None))
    if pixart_state is None:
        raise KeyError("Checkpoint must contain pixart_keep or pixart_trainable")

    layer_cfg = ckpt.get("layer_config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(layer_cfg, dict):
        layer_cfg = {}

    cfg = ckpt.get("config_snapshot", {}) if isinstance(ckpt, dict) else {}

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        anchor_layers=list(layer_cfg.get("anchor_layers", [0, 1, 2, 3, 4, 5, 6, 7])),
        semantic_layers=list(layer_cfg.get("semantic_layers", [24, 25, 26, 27])),
    ).to(device)

    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        load_state_dict_shape_compatible(pixart, base, context="base-pretrain")

    has_lora = any(("lora_A" in k) or ("lora_B" in k) for k in pixart_state.keys())

    lora_rank = ckpt.get("lora_rank", None)
    if lora_rank is None:
        lora_rank = cfg.get("lora_rank", None)
    if lora_rank is None and has_lora:
        lora_rank = infer_lora_rank_from_state_dict(pixart_state)
    if lora_rank is None:
        lora_rank = 4
    lora_rank = int(lora_rank)

    lora_alpha = ckpt.get("lora_alpha", None)
    if lora_alpha is None:
        lora_alpha = cfg.get("lora_alpha", None)
    if lora_alpha is None and has_lora:
        lora_alpha = float(lora_rank)
    if lora_alpha is None:
        lora_alpha = 4.0
    lora_alpha = float(lora_alpha)

    print(f"[Eval-Model] has_lora={has_lora} lora_rank={lora_rank} lora_alpha={lora_alpha}")
    if has_lora:
        apply_lora(pixart, lora_rank=lora_rank, lora_alpha=lora_alpha)

    load_state_dict_shape_compatible(pixart, pixart_state, context="eval")

    loaded_lora_keys = []
    skipped_lora_keys = []

    for k, v in pixart_state.items():
        if ("lora_A" in k) or ("lora_B" in k):
            curr = pixart.state_dict()
            if k in curr and tuple(curr[k].shape) == tuple(v.shape):
                loaded_lora_keys.append(k)
            else:
                skipped_lora_keys.append(k)

    print(
        f"[LoRA Eval] requested rank={lora_rank}, alpha={lora_alpha}, "
        f"saved_lora_tensors={len([k for k in pixart_state if ('lora_A' in k) or ('lora_B' in k)])}, "
        f"shape_compatible={len(loaded_lora_keys)}, skipped={len(skipped_lora_keys)}"
    )

    if has_lora and len(loaded_lora_keys) == 0:
        raise RuntimeError(
            "Checkpoint contains LoRA weights, but none of them are shape-compatible with the eval model. "
            "This usually means the eval script rebuilt LoRA with the wrong rank/alpha."
        )

    adapter = build_adapter_v12(
        in_channels=3,
        hidden_size=1152,
    ).to(device).float()
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    sem_adapter = None

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = load_prompt_pack(args.null_t5_embed_path, pack_name="null")
    print(f"[null-pack] loaded from: {args.null_t5_embed_path}")
    print(f"[null-pack] y shape: {tuple(null_pack['y'].shape)}")
    print(f"[null-pack] mask shape: {tuple(null_pack['mask'].shape)}")
    print(f"[null-pack] legacy converted: {bool(null_pack.get('_legacy_converted', False))}")
    print(f"[adaptive-prompt] enabled={USE_ADAPTIVE_TEXT_PROMPT} root={ADAPTIVE_PROMPT_CACHE_ROOT}")
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )

    pixart.eval()
    adapter.eval()
    return pixart, adapter, sem_adapter, vae, null_pack, scheduler


@torch.no_grad()
def run_ddim_predict(pixart, adapter, sem_adapter, vae, null_pack, scheduler, batch, args, device, compute_dtype, gen, prompt_cache_mem):
    hr = batch["hr"].to(device)
    lr = batch["lr"].to(device)
    lr_small = batch["lr_small"].to(device)

    z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

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
    data_info = {
        "img_hw": torch.tensor([[float(args.crop_size), float(args.crop_size)]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }
    sample_keys = batch.get("sample_key", None)
    if isinstance(sample_keys, str):
        sample_keys = [sample_keys]
    elif sample_keys is None:
        fallback_path = batch["hr_path"][0] if isinstance(batch["hr_path"], list) else batch["hr_path"]
        sample_keys = [make_sample_key(fallback_path)]
    prompt_cache_hit_rate = 0.0
    avg_prompt_token_count = 0.0
    avg_prompt_nonpad_ratio = 0.0
    text_cond_delta_vals = []

    for t in run_timesteps:
        t_b = torch.tensor([t], device=device).expand(latents.shape[0])
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
            cond = adapter(adapter_in, t_embed=t_embed.float())
            if "cond_map" not in cond or "cond_tokens" not in cond:
                raise KeyError("adapter output must contain both 'cond_map' and 'cond_tokens'")
            # conditional prompt source = adaptive cache (if enabled); unconditional prompt source = null prompt.
            y_cond = null_pack["y"].to(device).repeat(latents.shape[0], 1, 1, 1)
            mask_cond = null_pack["mask"].to(device).repeat(latents.shape[0], 1, 1, 1)
            if USE_ADAPTIVE_TEXT_PROMPT and ADAPTIVE_PROMPT_CACHE_ROOT:
                prompt_packs, prompt_cache_hit_rate, avg_prompt_token_count, avg_prompt_nonpad_ratio = load_adaptive_prompt_batch(
                    sample_keys, ADAPTIVE_PROMPT_CACHE_ROOT, prompt_cache_mem
                )
                if not all(p is not None for p in prompt_packs):
                    num_missing = int(sum(1 for p in prompt_packs if p is None))
                    raise RuntimeError(f"Adaptive prompt cache miss during full-eval: missing={num_missing}/{len(prompt_packs)}")
                y_cond = torch.cat([p["y"] for p in prompt_packs], dim=0).to(device)
                mask_cond = torch.cat([p["mask"] for p in prompt_packs], dim=0).to(device)
            y_uncond = null_pack["y"].to(device).repeat(latents.shape[0], 1, 1, 1)
            mask_uncond = null_pack["mask"].to(device).repeat(latents.shape[0], 1, 1, 1)
            drop_cond = torch.zeros(latents.shape[0], device=device, dtype=torch.long)
            drop_uncond = torch.ones(latents.shape[0], device=device, dtype=torch.long)
            if args.cfg_scale == 1.0:
                out = pixart(
                    x=latents.to(compute_dtype),
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
                    x=latents.to(compute_dtype),
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
                    x=latents.to(compute_dtype),
                    timestep=t_b,
                    y=y_cond,
                    aug_level=aug_level,
                    mask=mask_cond,
                    data_info=data_info,
                    adapter_cond=cond,
                    force_drop_ids=drop_cond,
                    sft_strength=args.sft_strength,
                )
                out_text_null = pixart(
                    x=latents.to(compute_dtype),
                    timestep=t_b,
                    y=y_uncond,
                    aug_level=aug_level,
                    mask=mask_uncond,
                    data_info=data_info,
                    adapter_cond=cond,
                    force_drop_ids=drop_cond,
                    sft_strength=args.sft_strength,
                )
                text_cond_delta_vals.append(float((out_cond - out_text_null).detach().abs().mean().item()))
                out = out_uncond + args.cfg_scale * (out_cond - out_uncond)
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    image_stats = getattr(pixart, "_last_image_cond_stats", {}) or {}
    sft_stats = getattr(pixart, "_last_sft_stats", {}) or {}
    debug_stats = {
        "cond_map_std": float(cond["cond_map"].detach().float().std().item()),
        "lr_cond_token_std": float(cond["cond_tokens"].detach().float().std().item()),
        "lr_cross_text_ctx_std": float(image_stats.get("lr_cross_text_ctx_std", 0.0)),
        "lr_cross_img_delta_std": float(image_stats.get("lr_cross_img_delta_std", 0.0)),
        "avg_image_alpha": float(image_stats.get("avg_image_alpha", 0.0)),
        "local_entry_gate": float(sft_stats.get("local_entry_gate", 0.0)),
        "text_cond_delta": float(sum(text_cond_delta_vals) / max(1, len(text_cond_delta_vals))),
        "prompt_cache_hit_rate": float(prompt_cache_hit_rate),
        "avg_prompt_token_count": float(avg_prompt_token_count),
        "avg_prompt_nonpad_ratio": float(avg_prompt_nonpad_ratio),
    }
    return pred, hr, lr, debug_stats


def tensor_m11_to_pil(x: torch.Tensor):
    x = x.detach().float().cpu().clamp(-1, 1)
    x01 = (x + 1.0) * 0.5
    return TF.to_pil_image(x01)


def save_triptych(lr_m11, hr_m11, pred_m11, path, steps):
    lr_img = tensor_m11_to_pil(lr_m11[0])
    hr_img = tensor_m11_to_pil(hr_m11[0])
    pr_img = tensor_m11_to_pil(pred_m11[0])
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(lr_img); plt.title("Input LR"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(hr_img); plt.title("GT"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(pr_img); plt.title(f"Pred @{steps}"); plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_component_visualization(gt_m11, pred_m11, masks, path, steps):
    gt_img = tensor_m11_to_pil(gt_m11[0])
    pred_img = tensor_m11_to_pil(pred_m11[0])
    m_flat, m_edge, m_corner = masks
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); plt.imshow(gt_img); plt.title("GT"); plt.axis("off")
    plt.subplot(2, 3, 2); plt.imshow(pred_img); plt.title(f"Pred @{steps}"); plt.axis("off")
    plt.subplot(2, 3, 3); plt.axis("off")
    plt.subplot(2, 3, 4); plt.imshow(m_flat[0, 0].detach().cpu(), cmap='gray'); plt.title("M_flat"); plt.axis("off")
    plt.subplot(2, 3, 5); plt.imshow(m_edge[0, 0].detach().cpu(), cmap='gray'); plt.title("M_edge"); plt.axis("off")
    plt.subplot(2, 3, 6); plt.imshow(m_corner[0, 0].detach().cpu(), cmap='gray'); plt.title("M_corner"); plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def nanmean(xs):
    vals = [float(v) for v in xs if v is not None and not math.isnan(float(v))]
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


def evaluate_dataset(dataset_name: str, loader, args, metric_suite, pixart, adapter, sem_adapter, vae, null_pack, scheduler, device, compute_dtype):
    base_out = Path(args.output_dir) / dataset_name
    preds_dir = base_out / "preds"
    trip_dir = base_out / "triptychs"
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"[Eval] Writing outputs to: {base_out}")
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)
    if args.save_triptychs:
        trip_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    prompt_cache_mem = {}

    rows = []
    pbar = tqdm(loader, desc=f"Eval[{dataset_name}]@{args.steps}")
    for idx, batch in enumerate(pbar):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        pred, hr, lr, debug_stats = run_ddim_predict(pixart, adapter, sem_adapter, vae, null_pack, scheduler, batch, args, device, compute_dtype, gen, prompt_cache_mem)
        m = metric_suite.compute(pred, hr)
        m_flat, m_edge, m_corner = build_component_masks_from_hr(hr)
        m_flat_c = metric_suite.compute_component(pred, hr, m_flat)
        m_edge_c = metric_suite.compute_component(pred, hr, m_edge)
        m_corner_c = metric_suite.compute_component(pred, hr, m_corner)

        hr_path = batch["hr_path"][0] if isinstance(batch["hr_path"], list) else batch["hr_path"]
        lr_path = batch["lr_path"][0] if isinstance(batch["lr_path"], list) else batch["lr_path"]
        image_name = batch["image_name"][0] if isinstance(batch["image_name"], list) else batch["image_name"]
        stem = Path(image_name).stem

        pred_path = preds_dir / f"{stem}_steps{args.steps}.png"
        if args.save_preds:
            tensor_m11_to_pil(pred[0]).save(pred_path)

        if args.save_triptychs and (not args.paper_only) and (idx % 5 == 0):
            tri_path = trip_dir / f"{idx:04d}_{stem}_steps{args.steps}.png"
            save_triptych(lr, hr, pred, str(tri_path), args.steps)
            comp_path = trip_dir / f"{idx:04d}_{stem}_comp_steps{args.steps}.png"
            save_component_visualization(hr, pred, (m_flat, m_edge, m_corner), str(comp_path), args.steps)

        row = {
            "dataset": dataset_name,
            "image_name": image_name,
            "hr_path": str(hr_path),
            "lr_path": str(lr_path),
            "pred_path": str(pred_path),
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "lpips": m["lpips"],
            "dists": m["dists"],
            "maniqa": m["maniqa"],
            "musiq": m["musiq"],
            "clipiqa": m["clipiqa"],
            "liqe": m["liqe"],
            "topiq_nr": m["topiq_nr"],
            "qalign": m["qalign"],
            "flat_psnr": m_flat_c["psnr"],
            "flat_ssim": m_flat_c["ssim"],
            "flat_lpips": m_flat_c["lpips"],
            "edge_psnr": m_edge_c["psnr"],
            "edge_ssim": m_edge_c["ssim"],
            "edge_lpips": m_edge_c["lpips"],
            "corner_psnr": m_corner_c["psnr"],
            "corner_ssim": m_corner_c["ssim"],
            "corner_lpips": m_corner_c["lpips"],
            "cond_map_std": debug_stats["cond_map_std"],
            "sem_tok_std": debug_stats["sem_tok_std"],
            "sem_out_std": debug_stats["sem_out_std"],
            "sem_out_scale": debug_stats["sem_out_scale"],
            "text_cond_delta": debug_stats["text_cond_delta"],
            "prompt_cache_hit_rate": debug_stats["prompt_cache_hit_rate"],
            "avg_prompt_token_count": debug_stats["avg_prompt_token_count"],
            "avg_prompt_nonpad_ratio": debug_stats["avg_prompt_nonpad_ratio"],
        }
        rows.append(row)

        pbar.set_postfix({
            "psnr": f"{row['psnr']:.2f}",
            "lpips": f"{row['lpips']:.4f}",
            "c_lp": f"{row['corner_lpips']:.4f}",
            "cond": f"{row['cond_map_std']:.3f}",
            "txtΔ": f"{row['text_cond_delta']:.4f}",
            "hit": f"{row['prompt_cache_hit_rate']:.2f}",
        })

    csv_path = base_out / "per_image_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "image_name", "hr_path", "lr_path", "pred_path",
        "psnr", "ssim", "lpips", "dists", "maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign",
        "flat_psnr", "flat_ssim", "flat_lpips",
        "edge_psnr", "edge_ssim", "edge_lpips",
        "corner_psnr", "corner_ssim", "corner_lpips",
        "cond_map_std", "sem_tok_std", "sem_out_std", "sem_out_scale",
        "text_cond_delta", "prompt_cache_hit_rate", "avg_prompt_token_count", "avg_prompt_nonpad_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    paper_fieldnames = [
        "dataset", "image_name",
        "psnr", "ssim", "lpips", "dists",
        "maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign"
    ]
    paper_csv_path = base_out / "paper_metrics.csv"
    paper_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paper_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, float("nan")) for k in paper_fieldnames})

    protocol = {
        "dataset_crop": "center crop, LR 128x128 / HR 512x512",
        "psnr_ssim": "Y channel, border shave 4",
        "lpips": "RGB [-1,1]",
        "dists": "RGB [0,1]",
        "nr_iqa": ["MANIQA", "MUSIQ", "CLIPIQA", "LIQE", "TOPIQ_NR", "QALIGN"],
    }
    summary = {
        "dataset": dataset_name,
        "num_images": len(rows),
        "steps": int(args.steps),
        "use_lq_init": bool(args.use_lq_init),
        "lq_init_strength": float(args.lq_init_strength),
        "protocol": protocol,
        "mean": {
            "psnr": nanmean([r["psnr"] for r in rows]),
            "ssim": nanmean([r["ssim"] for r in rows]),
            "lpips": nanmean([r["lpips"] for r in rows]),
            "dists": nanmean([r["dists"] for r in rows]),
            "maniqa": nanmean([r["maniqa"] for r in rows]),
            "musiq": nanmean([r["musiq"] for r in rows]),
            "clipiqa": nanmean([r["clipiqa"] for r in rows]),
            "liqe": nanmean([r["liqe"] for r in rows]),
            "topiq_nr": nanmean([r["topiq_nr"] for r in rows]),
            "qalign": nanmean([r["qalign"] for r in rows]),
        },
        "analysis_mean": {
            "flat_psnr": nanmean([r["flat_psnr"] for r in rows]),
            "flat_ssim": nanmean([r["flat_ssim"] for r in rows]),
            "flat_lpips": nanmean([r["flat_lpips"] for r in rows]),
            "edge_psnr": nanmean([r["edge_psnr"] for r in rows]),
            "edge_ssim": nanmean([r["edge_ssim"] for r in rows]),
            "edge_lpips": nanmean([r["edge_lpips"] for r in rows]),
            "corner_psnr": nanmean([r["corner_psnr"] for r in rows]),
            "corner_ssim": nanmean([r["corner_ssim"] for r in rows]),
            "corner_lpips": nanmean([r["corner_lpips"] for r in rows]),
        }
    }
    paper_summary = {
        "dataset": dataset_name,
        "num_images": len(rows),
        "steps": int(args.steps),
        "protocol": protocol,
        "mean": {
            "psnr": summary["mean"]["psnr"],
            "ssim": summary["mean"]["ssim"],
            "lpips": summary["mean"]["lpips"],
            "dists": summary["mean"]["dists"],
            "maniqa": summary["mean"]["maniqa"],
            "musiq": summary["mean"]["musiq"],
            "clipiqa": summary["mean"]["clipiqa"],
            "liqe": summary["mean"]["liqe"],
            "topiq_nr": summary["mean"]["topiq_nr"],
            "qalign": summary["mean"]["qalign"],
        },
    }

    summary_path = base_out / "summary.json"
    summary_txt_path = base_out / "summary.txt"
    paper_summary_json_path = base_out / "paper_summary.json"
    paper_summary_csv = base_out / "paper_summary.csv"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))
    paper_summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_summary_json_path, "w", encoding="utf-8") as f:
        json.dump(paper_summary, f, indent=2, ensure_ascii=False)
    paper_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "num_images", "steps", "psnr", "ssim", "lpips", "dists", "maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign"])
        writer.writeheader()
        writer.writerow({
            "dataset": dataset_name,
            "num_images": len(rows),
            "steps": int(args.steps),
            **paper_summary["mean"],
        })

    print(f"✅ [{dataset_name}] wrote: {csv_path}")
    print(f"✅ [{dataset_name}] wrote: {paper_csv_path}")
    print(f"✅ [{dataset_name}] wrote: {paper_summary_csv}")
    label = "RealSR" if dataset_name.lower() == "realsr" else ("DRealSR" if dataset_name.lower() == "drealsr" else dataset_name)
    print(
        f"✅ [{label}] PSNR={summary['mean']['psnr']:.3f} SSIM={summary['mean']['ssim']:.4f} "
        f"LPIPS={summary['mean']['lpips']:.4f} DISTS={summary['mean']['dists']:.4f} "
        f"MANIQA={summary['mean']['maniqa']:.4f} MUSIQ={summary['mean']['musiq']:.4f} CLIPIQA={summary['mean']['clipiqa']:.4f}"
    )
    return paper_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Full SR evaluation on RealSR and DRealSR")
    parser.add_argument("--dataset", type=str, default="both", choices=["realsr", "drealsr", "both"])

    parser.add_argument("--pixart_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae")
    parser.add_argument("--null_t5_embed_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/null_t5_embed_sigma_300.pth")
    parser.add_argument("--semantic_encoder_name_or_path", type=str, default="openai/clip-vit-large-patch14")

    parser.add_argument("--realsr_roots", type=str, nargs="*", default=["/data/RealSR/Nikon/Test/4", "/data/RealSR/Canon/Test/4"])
    parser.add_argument("--drealsr_hr_dir", type=str, default="/data/DRealSR/HR")
    parser.add_argument("--drealsr_lr_dir", type=str, default="/data/DRealSR/LR")

    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use_lq_init", dest="use_lq_init", action="store_true")
    parser.add_argument("--no-use_lq_init", dest="use_lq_init", action="store_false")
    parser.set_defaults(use_lq_init=True)
    parser.add_argument("--lq_init_strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sft_strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="/home/hello/HJT/SRConvnetAdapter/experiments_results")
    parser.add_argument("--save_preds", dest="save_preds", action="store_true")
    parser.add_argument("--no-save_preds", dest="save_preds", action="store_false")
    parser.set_defaults(save_preds=False)

    parser.add_argument("--save_triptychs", dest="save_triptychs", action="store_true")
    parser.add_argument("--no-save_triptychs", dest="save_triptychs", action="store_false")
    parser.set_defaults(save_triptychs=False)

    parser.add_argument("--paper_only", dest="paper_only", action="store_true")
    parser.add_argument("--no-paper_only", dest="paper_only", action="store_false")
    parser.set_defaults(paper_only=False)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pixart, adapter, sem_adapter, vae, null_pack, scheduler = build_model_and_assets(args, device, compute_dtype)
    metric_suite = MetricSuite()

    paper_rows = []
    if args.dataset in ("realsr", "both"):
        realsr_ds = RealSRValPairedDataset(roots=args.realsr_roots, crop_size=args.crop_size)
        realsr_loader = DataLoader(realsr_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        paper_rows.append(evaluate_dataset("realsr", realsr_loader, args, metric_suite, pixart, adapter, sem_adapter, vae, null_pack, scheduler, device, compute_dtype))

    if args.dataset in ("drealsr", "both"):
        drealsr_ds = DRealSRPairedDataset(args.drealsr_hr_dir, args.drealsr_lr_dir, crop_size=args.crop_size)
        drealsr_loader = DataLoader(drealsr_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        paper_rows.append(evaluate_dataset("drealsr", drealsr_loader, args, metric_suite, pixart, adapter, sem_adapter, vae, null_pack, scheduler, device, compute_dtype))

    if args.dataset == "both" and len(paper_rows) > 0:
        ckpt_name = Path(args.ckpt_path).stem
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        paper_table_path = Path(args.output_dir) / "paper_table.csv"
        with open(paper_table_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["dataset", "ckpt_name", "psnr", "ssim", "lpips", "dists", "maniqa", "musiq", "clipiqa", "liqe", "topiq_nr", "qalign"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in paper_rows:
                writer.writerow({
                    "dataset": row["dataset"],
                    "ckpt_name": ckpt_name,
                    **row["mean"],
                })
        print(f"✅ wrote combined paper table: {paper_table_path}")


if __name__ == "__main__":
    main()
