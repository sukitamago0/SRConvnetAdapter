# /home/hello/HJT/DiTSR/experiments/train_4090_auto_v8.py
# DiTSR v8 Training Script (Final Corrected Version)
# ------------------------------------------------------------------
# Fixes included:
# 1. [Optim] Fixed parameter grouping (Mutually Exclusive).
# 2. [Process] Unlocked x_embedder learning rate (1e-4).
# 3. [Process] Delayed LPIPS ramp for structure-first convergence (Stage 1).
# 4. [Structure] Fixed NameError by ensuring Dataset classes are defined.
# ------------------------------------------------------------------

import os
import sys
from pathlib import Path

# Recommended allocator setup for fragmentation control on 24G cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

# ================= 1. Path Setup =================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import random
import math
import re
import resource
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import hashlib
import shutil
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

# [Import V8 Model]
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v12
from diffusion.model.nets.semantic_adapter import CLIPSemanticAdapter
from diffusion.model.utils import set_grad_checkpoint
from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor
from utils.prompt_key_utils import make_sample_key

BASE_PIXART_SHA256 = None
LAST_TRAIN_LOG = {}

ACTIVE_PIXART_KEY_FRAGMENTS = (
    "final_layer",
    "sft_cond_reduce",
    "sft_layers",
    "sft_alpha",
    "tasr_time_gates",
    "hybrid_cross_attn",
    "semantic_alpha",
    "lora_A",
    "lora_B",
)
FP32_SAVE_KEY_FRAGMENTS = ACTIVE_PIXART_KEY_FRAGMENTS
FINAL_LAYER_KEYWORD = "final_layer"
TRAIN_ADAPTER_IN_STAGE_A = True

def get_required_active_key_fragments_for_model(model: nn.Module):
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    required = []
    for frag in ACTIVE_PIXART_KEY_FRAGMENTS:
        if any(frag in name for name in trainable_names):
            required.append(frag)
    return tuple(required)

# ================= 2. Hyper-parameters =================
# paper table must use eval_sr_realsr_drealsr_full.py (full evaluation script).
# LR is early structural condition only.
# semantic/text are late-detail semantic conditions.
# no late LR detail control.
# early band is for LR-derived structure anchoring only.
# late band is for semantic/detail guidance only.
# mid band remains largely prior-driven.
# fine details should be generated mainly by semantic guidance + pretrained PixArt prior.
TRAIN_DF2K_HR_DIR = "/data/DF2K/DF2K_train_HR"
TRAIN_DF2K_LR_DIR = "/data/DF2K/DF2K_train_LR_unknown"
TRAIN_REALSR_DIRS = [
    "/data/RealSR/Canon/Train/4",
    "/data/RealSR/Nikon/Train/4",
]
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR"
VAL_LR_DIR_CANDIDATES = [
    "/data/DF2K/DF2K_valid_LR_unknown/X4",
    "/data/DF2K/DF2K_valid_LR_bicubic/X4",
]
VAL_LR_DIR = next((p for p in VAL_LR_DIR_CANDIDATES if os.path.exists(p)), None)
REALSR_VAL_ROOTS = [
    "/data/RealSR/Nikon/Test/4",
    "/data/RealSR/Canon/Test/4",
]

PRETRAINED_ROOT = os.getenv("DTSR_PRETRAINED_ROOT", "/home/hello/HJT/PixArt-sigma/output/pretrained_models")
PIXART_PATH = os.path.join(PRETRAINED_ROOT, "PixArt-Sigma-XL-2-512-MS.pth")
DIFFUSERS_ROOT = os.path.join(PRETRAINED_ROOT, "pixart_sigma_sdxlvae_T5_diffusers")
VAE_PATH = os.path.join(DIFFUSERS_ROOT, "vae")
NULL_T5_EMBED_PATH = os.path.join(PRETRAINED_ROOT, "null_t5_embed_sigma_300.pth")

OUT_BASE = os.getenv("DTSR_OUT_BASE", os.path.join(PROJECT_ROOT, "experiments_results"))
INIT_CKPT_PATH = os.getenv("DTSR_INIT_CKPT", "")  # optional bootstrap ckpt (weights only)
OUT_DIR = os.path.join(OUT_BASE, "td-resp-final_lora16")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")

DEVICE = "cuda"
COMPUTE_DTYPE = torch.bfloat16
SEED = 3407
DETERMINISTIC = True
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "0") == "1"
FAST_TRAIN_STEPS = int(os.getenv("FAST_TRAIN_STEPS", "10"))
FAST_VAL_BATCHES = int(os.getenv("FAST_VAL_BATCHES", "2"))
FAST_VAL_STEPS = int(os.getenv("FAST_VAL_STEPS", "10"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "0"))

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16 
NUM_WORKERS = 4
VAL_NUM_WORKERS = 0
TRAIN_PROMPT_CACHE_MAX_ITEMS = 0
VAL_PROMPT_CACHE_MAX_ITEMS = 128
GUARD_FORWARD_EVERY = 50

LR_BASE = 1e-5 
LORA_RANK = 16
LORA_ALPHA = 16
SEMANTIC_ENCODER_NAME_OR_PATH = os.getenv("SEMANTIC_ENCODER_NAME_OR_PATH", "openai/clip-vit-large-patch14")
USE_SEMANTIC_BRANCH = True
ALLOW_SEM_ADAPTER_NONSTRICT_RESUME = False
USE_FIXED_HARD_PROMPT = False
USE_ADAPTIVE_TEXT_PROMPT = True
STRICT_ADAPTIVE_PROMPT = True
ADAPTIVE_PROMPT_CACHE_ROOT = os.getenv("ADAPTIVE_PROMPT_CACHE_ROOT", "")
TRAIN_PIXART_X_EMBEDDER = False  # S2D: keep backbone patch embedder frozen for clean attribution
SPARSE_INJECT_RATIO = 1.0
INJECTION_CUTOFF_LAYER = 28
INJECTION_STRATEGY = "three_stage_sr"
INJECT_R_END = 0.1
INJECT_S_MIN = 0.1
INJECT_S_MAX = 1.0
INJECT_INIT_P = 2.0
# anchor_layers = early structure-only band
ANCHOR_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7]
# semantic_layers = late semantic-only band
SEMANTIC_LAYERS = [24, 25, 26, 27]

L1_BASE_WEIGHT = 0.25
GW_ALPHA = 4.0
COMP_CORNER_Q = 0.95
COMP_EDGE_Q = 0.80

VAL_STEPS_LIST = [50]
BEST_VAL_STEPS = 50
KEEP_TOPK = 1
VAL_MODE = "realsr"
VAL_PACK_DIR = os.path.join(PROJECT_ROOT, "valpacks", "df2k_train_like_50_seed3407")
VAL_PACK_LR_DIR_NAME = "lq512"
TRAIN_DEG_MODE = "highorder"
CFG_SCALE = 1.0

# [V8 Change] Default to LQ-Init for validation
USE_LQ_INIT = True 
LQ_INIT_STRENGTH = 0.3
USE_TALA_TRAIN = True
TALA_TRAIN_ACTIVE_MODE = "high_t"
TALA_TRAIN_ACTIVE_T_MIN = 600
TALA_TRAIN_BLEND_POW = 1.0
TALA_TRAIN_MAX_RATIO = 0.30
TALA_TRAIN_DETACH_LR_LATENT = True
# this is a high-noise training blend regime inspired by LQ-init semantics,
# not an exact scheduler-level mapping.
TALA_TRAIN_HIGH_T_MODE = True
USE_TIMESTEP_AWARE_PERCEPTUAL = True
PERCEP_T_MAX = 350
PERCEP_DECODE_MAX_SAMPLES = 2
LPIPS_LATE_WEIGHT = 0.15
USE_LEGACY_PATCH_LPIPS = False

INIT_NOISE_STD = 0.0
USE_ADAPTER_CFDROPOUT = True
COND_DROP_PROB = 0.10
FORCE_DROP_TEXT = True  # validation-time text drop behavior
# Phase2: Drop only concat-LR branch (adapter still sees normal/augmented LR)
CONCAT_LR_DROP_ENABLED = False
CONCAT_LR_DROP_SCHEDULE = [
    (0, 0.0),
    (1000, 0.0),
    (4000, 0.4),
    (12000, 0.4),
    (20000, 0.1),
]
CONCAT_LR_DROP_NO_RESCALE = True
INJECT_SCALE_REG_LAMBDA = 1e-4
PIXEL_LOSS_T_MIN = 200
LATENT_L1_T_MIN = 200

# LPIPS perceptual curriculum (single-script, no separate stage script)
USE_LPIPS_PERCEP = True
LPIPS_START_EPOCH = 8
LPIPS_RAMP_END_EPOCH = 25
LPIPS_WEIGHT_MAX = 0.25
LPIPS_T_MAX = 800

# As LPIPS ramps in, old structure losses ramp down
LATENT_L1_WEIGHT_START = 0.08
LATENT_L1_WEIGHT_END = 0.0

LR_CONS_WEIGHT_START = 0.0
LR_CONS_WEIGHT_END = 0.0

GW_WEIGHT_START = 0.0
GW_WEIGHT_END = 0.0

# For this experiment, require sufficient structure quality before switching to LPIPS-first checkpointing
CKPT_SELECT_MODE = "psnr_gate_then_lpips"
CKPT_SELECT_PSNR_GATE = 25.5
BEST_PSNR_PATH = os.path.join(CKPT_DIR, "best_psnr.pth")
START_FROM_BASE_ONLY = True

USE_LR_CONSISTENCY = True 
USE_NOISE_CONSISTENCY = False

VAE_TILING = True
DEG_OPS = ["blur", "resize", "noise", "jpeg"]
P_TWO_STAGE = 0.35
RESIZE_SCALE_RANGE = (0.3, 1.8)
NOISE_RANGE = (0.0, 0.05)
BLUR_KERNELS = [7, 9, 11, 13, 15, 21]
JPEG_QUALITY_RANGE = (30, 95)
RESIZE_INTERP_MODES = [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR, transforms.InterpolationMode.BICUBIC]


# VNext training controls (script-versioned, no ENV override)
USE_EMA = False
EMA_DECAY = 0.999
EMA_TRACK_SET = "small"  # small | all
EMA_VALIDATE = False  # S2D main experiment: disable EMA/EMA-validate for clean comparison
EMA_VALIDATE_EVERY = 5  # run EMA validation every N validation triggers

T_SAMPLE_MODE = "power"   # uniform | power | two_stage
T_SAMPLE_POWER = 2.5
T_SAMPLE_MIN = 0
T_SAMPLE_MAX = 999
T_TWO_STAGE_SWITCH = 15000

INJECT_REG_WARMUP_END = 0
INJECT_REG_RAMP_END = 1

LR_CONSIST_WEIGHT_MAX = 0.1
LR_CONSIST_WARMUP = 0
LR_CONSIST_RAMP = 2400

# Conservative KV-compress to reduce attention memory with minimal quality impact.
KV_COMPRESS_ENABLE = False
KV_COMPRESS_SCALE = 2
KV_COMPRESS_LAYERS = list(range(20, 28))  # late blocks only

# Staged unfreeze for resume from old checkpoint
TRAIN_STAGE = "single"
AUTO_STAGE_ENABLED = False
ENABLE_SELECTIVE_TUNING = False
ENABLE_LORA = True

# Phase0 safety: check zero-impact regression before training
RUN_PHASE0_REGRESSION_TEST = False
PHASE0_MAX_PSNR_DROP = 0.02
PHASE0_MAX_LPIPS_RISE = 0.005
PHASE0_NUM_SAMPLES = 4

# Keep legacy injection mostly shallow when enabling dual-stream

# ================= 3. Logic Functions =================



def _linear_ramp(epoch_1based: int, start_epoch: int, end_epoch: int) -> float:
    if epoch_1based < start_epoch:
        return 0.0
    if epoch_1based >= end_epoch:
        return 1.0
    return float(epoch_1based - start_epoch) / float(max(1, end_epoch - start_epoch))


def get_loss_weights(epoch_1based: int):
    struct_decay = _linear_ramp(epoch_1based, start_epoch=8, end_epoch=25)
    latent_l1_w = LATENT_L1_WEIGHT_START + (LATENT_L1_WEIGHT_END - LATENT_L1_WEIGHT_START) * struct_decay
    lr_cons_w = LR_CONS_WEIGHT_START + (LR_CONS_WEIGHT_END - LR_CONS_WEIGHT_START) * struct_decay
    gw_w = GW_WEIGHT_START + (GW_WEIGHT_END - GW_WEIGHT_START) * struct_decay
    lpips_w = 0.0
    if USE_LPIPS_PERCEP:
        lpips_w = LPIPS_WEIGHT_MAX * _linear_ramp(epoch_1based, start_epoch=LPIPS_START_EPOCH, end_epoch=LPIPS_RAMP_END_EPOCH)

    return {
        "mse": 1.0,
        "latent_l1": float(latent_l1_w),
        "lr_cons": float(lr_cons_w),
        "gw": float(gw_w),
        "lpips": float(lpips_w),
    }


def get_sft_strength(epoch_1based: int) -> float:
    return 1.0


def get_inject_reg_weight(epoch_1based: int) -> float:
    if epoch_1based < 20:
        return 1.0
    if epoch_1based < 25:
        p = (epoch_1based - 20) / max(1, 25 - 20)
        return float(1.0 - 0.95 * p)
    return 0.05


def get_fixed_loss_weights(epoch_1based: int = 1):
    return get_loss_weights(epoch_1based)


def gan_bce_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def inject_reg_lambda(step: int) -> float:
    if step < INJECT_REG_WARMUP_END:
        return INJECT_SCALE_REG_LAMBDA
    if step < INJECT_REG_RAMP_END:
        p = (step - INJECT_REG_WARMUP_END) / max(1, (INJECT_REG_RAMP_END - INJECT_REG_WARMUP_END))
        return INJECT_SCALE_REG_LAMBDA * (1.0 - p)
    return 0.0


def get_concat_lr_drop_p(global_step: int, stage: str) -> float:
    del stage
    if (not CONCAT_LR_DROP_ENABLED) or (CONCAT_LR_DROP_SCHEDULE is None) or (len(CONCAT_LR_DROP_SCHEDULE) == 0):
        return 0.0
    s = int(global_step)
    sched = sorted([(int(k), float(v)) for k, v in CONCAT_LR_DROP_SCHEDULE], key=lambda x: x[0])
    if s <= sched[0][0]:
        return float(sched[0][1])
    for (a_step, a_p), (b_step, b_p) in zip(sched[:-1], sched[1:]):
        if a_step <= s <= b_step:
            if b_step == a_step:
                return float(b_p)
            t = (s - a_step) / float(b_step - a_step)
            return float(a_p + (b_p - a_p) * t)
    return float(sched[-1][1])


def sample_t(batch: int, device: str, step: int) -> torch.Tensor:
    tmin = int(max(0, T_SAMPLE_MIN))
    tmax = int(min(999, T_SAMPLE_MAX))
    if tmax < tmin:
        tmax = tmin

    mode = T_SAMPLE_MODE.lower()
    if mode == 'uniform':
        return torch.randint(tmin, tmax + 1, (batch,), device=device).long()
    if mode == 'power':
        u = torch.rand((batch,), device=device)
        span = float(max(1, tmax - tmin))
        t = torch.floor((u ** float(T_SAMPLE_POWER)) * span + tmin)
        return t.clamp(tmin, tmax).long()
    if mode == 'two_stage':
        if step < T_TWO_STAGE_SWITCH:
            u = torch.rand((batch,), device=device)
            span = float(max(1, tmax - tmin))
            t = torch.floor((u ** float(T_SAMPLE_POWER)) * span + tmin)
            return t.clamp(tmin, tmax).long()
        return torch.randint(tmin, tmax + 1, (batch,), device=device).long()
    raise ValueError(f"Unknown T_SAMPLE_MODE: {T_SAMPLE_MODE}")

def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481*r + 128.553*g + 24.966*b) / 255.0

# ----------------- Edge-guided perceptual regularizers (GT-driven) -----------------
# Goal: (1) match gradients where GT has edges, (2) suppress high-frequency hallucinations where GT is flat/defocused.
_SOBEL_X = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_LAPLACE = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

def _to_luma01(img_m11: torch.Tensor) -> torch.Tensor:
    # img in [-1,1], returns luma in [0,1], shape [B,1,H,W], float32
    img01 = (img_m11.float() + 1.0) * 0.5
    r = img01[:, 0:1]; g = img01[:, 1:2]; b = img01[:, 2:3]
    luma = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return luma.clamp(0.0, 1.0)

@torch.cuda.amp.autocast(enabled=False)
def _harris_response(gray01: torch.Tensor, k: float = 0.04, eps: float = 1e-6) -> torch.Tensor:
    kx = _SOBEL_X.to(device=gray01.device)
    ky = _SOBEL_Y.to(device=gray01.device)
    ix = F.conv2d(gray01, kx, padding=1)
    iy = F.conv2d(gray01, ky, padding=1)
    ixx = F.avg_pool2d(ix * ix, kernel_size=3, stride=1, padding=1)
    iyy = F.avg_pool2d(iy * iy, kernel_size=3, stride=1, padding=1)
    ixy = F.avg_pool2d(ix * iy, kernel_size=3, stride=1, padding=1)
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    r = det - k * trace * trace
    r_min = r.flatten(1).min(dim=1, keepdim=True)[0][:, :, None]
    r_max = r.flatten(1).max(dim=1, keepdim=True)[0][:, :, None]
    return ((r - r_min) / (r_max - r_min + eps)).clamp(0.0, 1.0)


@torch.cuda.amp.autocast(enabled=False)
def build_component_masks_from_hr(hr_m11: torch.Tensor):
    gray = _to_luma01(hr_m11)
    harris = _harris_response(gray)
    grad_x = F.conv2d(gray, _SOBEL_X.to(device=gray.device), padding=1)
    grad_y = F.conv2d(gray, _SOBEL_Y.to(device=gray.device), padding=1)
    grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    b = gray.shape[0]
    corner_q = torch.quantile(harris.flatten(1), COMP_CORNER_Q, dim=1, keepdim=True)
    edge_q = torch.quantile(grad_mag.flatten(1), COMP_EDGE_Q, dim=1, keepdim=True)
    m_corner = (harris >= corner_q.view(b, 1, 1, 1)).float()
    m_edge = (grad_mag >= edge_q.view(b, 1, 1, 1)).float() * (1.0 - m_corner)
    m_flat = 1.0 - (m_edge + m_corner).clamp(0.0, 1.0)

    m_flat = F.avg_pool2d(m_flat, kernel_size=3, stride=1, padding=1)
    m_edge = F.avg_pool2d(m_edge, kernel_size=3, stride=1, padding=1)
    m_corner = F.avg_pool2d(m_corner, kernel_size=3, stride=1, padding=1)
    norm = (m_flat + m_edge + m_corner).clamp_min(1e-6)
    return m_flat / norm, m_edge / norm, m_corner / norm


@torch.cuda.amp.autocast(enabled=False)
def gradient_weighted_loss(pred_m11: torch.Tensor, hr_m11: torch.Tensor, alpha: float = 4.0):
    pred_y = _to_luma01(pred_m11)
    hr_y = _to_luma01(hr_m11)
    kx = _SOBEL_X.to(device=pred_m11.device)
    ky = _SOBEL_Y.to(device=pred_m11.device)
    gx_p = F.conv2d(pred_y, kx, padding=1)
    gy_p = F.conv2d(pred_y, ky, padding=1)
    gx_h = F.conv2d(hr_y, kx, padding=1)
    gy_h = F.conv2d(hr_y, ky, padding=1)
    dx = (gx_p - gx_h).abs()
    dy = (gy_p - gy_h).abs()
    dgw = (1.0 + alpha * dx) * (1.0 + alpha * dy)
    return F.l1_loss(dgw * pred_m11.float(), dgw * hr_m11.float())
# -------------------------------------------------------------------------------

@torch.cuda.amp.autocast(enabled=False)
def build_adapter_struct_input(lr_small_m11: torch.Tensor) -> torch.Tensor:
    # Keep as real lr_small tensor; no resize/back-projection from lr_up.
    return lr_small_m11.float().clamp(-1.0, 1.0)


@torch.cuda.amp.autocast(enabled=False)
def structure_consistency_loss(pred_m11: torch.Tensor, lr_m11: torch.Tensor) -> torch.Tensor:
    # Compare downsampled structure (Sobel/Laplacian) and add a light low-frequency luma consistency.
    pred_lr = F.interpolate(pred_m11.float(), size=lr_m11.shape[-2:], mode='bilinear', align_corners=False)
    p = _to_luma01(pred_lr)
    l = _to_luma01(lr_m11.float())
    kx = _SOBEL_X.to(device=p.device)
    ky = _SOBEL_Y.to(device=p.device)
    kl = _LAPLACE.to(device=p.device)
    pgx = F.conv2d(p, kx, padding=1); pgy = F.conv2d(p, ky, padding=1)
    lgx = F.conv2d(l, kx, padding=1); lgy = F.conv2d(l, ky, padding=1)
    pl = F.conv2d(p, kl, padding=1)
    ll = F.conv2d(l, kl, padding=1)
    loss_sobel = (pgx - lgx).abs().mean() + (pgy - lgy).abs().mean()
    loss_lap = (pl - ll).abs().mean()
    # Low-frequency data-consistency (gray/luma domain).
    p_low = F.avg_pool2d(p, kernel_size=5, stride=1, padding=2)
    l_low = F.avg_pool2d(l, kernel_size=5, stride=1, padding=2)
    loss_lowfreq = F.l1_loss(p_low, l_low)
    return 0.4 * loss_sobel + 0.4 * loss_lap + 0.2 * loss_lowfreq


@torch.cuda.amp.autocast(enabled=False)
def perceptual_lpips_loss(lpips_fn: nn.Module, pred_m11: torch.Tensor, target_m11: torch.Tensor) -> torch.Tensor:
    # inputs are in [-1, 1]
    return lpips_fn(pred_m11.float(), target_m11.float()).mean()


def compute_component_metrics(pred_m11: torch.Tensor, hr_m11: torch.Tensor, mask: torch.Tensor, lpips_fn_cpu: nn.Module):
    pred_cpu = pred_m11.detach().to("cpu", dtype=torch.float32)
    hr_cpu = hr_m11.detach().to("cpu", dtype=torch.float32)
    mask_cpu = mask.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    pred01 = (pred_cpu + 1.0) / 2.0
    hr01 = (hr_cpu + 1.0) / 2.0
    py = rgb01_to_y01(pred01)[..., 4:-4, 4:-4]
    hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
    my = mask_cpu[..., 4:-4, 4:-4]
    mse = ((my * (py - hy)) ** 2).sum() / my.sum().clamp_min(1e-6)
    psnr_v = float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())
    mu_x = (my * py).sum() / my.sum().clamp_min(1e-6)
    mu_y = (my * hy).sum() / my.sum().clamp_min(1e-6)
    vx = (my * (py - mu_x) ** 2).sum() / my.sum().clamp_min(1e-6)
    vy = (my * (hy - mu_y) ** 2).sum() / my.sum().clamp_min(1e-6)
    cxy = (my * (py - mu_x) * (hy - mu_y)).sum() / my.sum().clamp_min(1e-6)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_v = float((((2 * mu_x * mu_y + c1) * (2 * cxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (vx + vy + c2))).item())
    lpips_v = float(lpips_fn_cpu(pred_cpu * mask_cpu, hr_cpu * mask_cpu).mean().item())
    return psnr_v, ssim_v, lpips_v


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

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


def load_adaptive_prompt_batch(sample_keys, cache_root: str, cache_mem, max_cache_items: int = 0):
    packs = []
    hits = 0
    num_missing = 0
    tok_counts = []
    nonpad_ratios = []
    use_cache = (cache_mem is not None) and (max_cache_items > 0)
    for key in sample_keys:
        pack = None
        if use_cache and key in cache_mem:
            pack = cache_mem[key]
            cache_mem.move_to_end(key)
            hits += 1
        else:
            pth = os.path.join(cache_root, f"{key}.pth")
            if os.path.exists(pth):
                pack = load_prompt_pack(pth, pack_name="adaptive")
                if use_cache:
                    cache_mem[key] = pack
                    cache_mem.move_to_end(key)
                    while len(cache_mem) > max_cache_items:
                        cache_mem.popitem(last=False)
        packs.append(pack)
        if pack is None:
            num_missing += 1
        if pack is not None:
            mask = pack["mask"].detach().float()
            tok_counts.append(float(mask.sum().item()))
            nonpad_ratios.append(float(mask.mean().item()))
    hit_rate = float(hits / max(1, len(sample_keys))) if use_cache else 0.0
    avg_tok_count = float(np.mean(tok_counts)) if len(tok_counts) > 0 else 0.0
    avg_nonpad_ratio = float(np.mean(nonpad_ratios)) if len(nonpad_ratios) > 0 else 0.0
    return packs, hit_rate, avg_tok_count, avg_nonpad_ratio, int(num_missing)


def _decode_vae_sample_checkpointed(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents with non-reentrant activation checkpoint to reduce peak memory."""
    def _decode_fn(z):
        return vae.decode(z).sample
    return torch.utils.checkpoint.checkpoint(_decode_fn, latents, use_reentrant=False)


def _maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_rss_gb():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0 / 1024.0

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


def get_tala_train_interval(diffusion):
    num_train_timesteps = int(getattr(diffusion, "num_timesteps", 1000))
    t_max = max(0, num_train_timesteps - 1)
    if TALA_TRAIN_HIGH_T_MODE:
        min_t = int(TALA_TRAIN_ACTIVE_T_MIN)
    else:
        min_t = int(round(float(LQ_INIT_STRENGTH) * float(t_max)))
    min_t = max(0, min(t_max, int(min_t)))
    return min_t, t_max


def build_tala_train_latent_and_target(
    zh,
    zl,
    timesteps,
    noise,
    diffusion,
    lq_init_strength,
    high_t_mode,
    active_t_min,
    blend_pow,
    max_ratio,
    detach_lr_latent,
    use_tala=True,
):
    zt_hr = diffusion.q_sample(zh, timesteps, noise)
    alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, timesteps, zh.shape)
    sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, zh.shape)
    if not use_tala:
        target_v = alpha_t * noise - sigma_t * zh.float()
        ratio = torch.zeros((zh.shape[0],), device=zh.device, dtype=zh.dtype)
        return zt_hr, target_v, alpha_t, sigma_t, ratio, {
            "tala_mode": "disabled",
            "tala_t_min": 0,
            "tala_t_max": 0,
            "tala_effective_eps_std": float(noise.detach().float().std().item()),
        }

    zl_src = zl.detach() if detach_lr_latent else zl
    zt_lr = diffusion.q_sample(zl_src, timesteps, noise)
    t_min, t_max = get_tala_train_interval(diffusion)
    if high_t_mode:
        t_min = max(0, min(t_max, int(active_t_min)))
    else:
        t_min = max(t_min, int(round(float(lq_init_strength) * float(t_max))))
    if str(TALA_TRAIN_ACTIVE_MODE).lower() != "high_t":
        raise RuntimeError(f"Unsupported TALA_TRAIN_ACTIVE_MODE={TALA_TRAIN_ACTIVE_MODE}, only 'high_t' is allowed.")
    tau = ((timesteps.float() - float(t_min)) / max(1.0, float(t_max - t_min))).clamp(0.0, 1.0)
    ratio_cap = min(float(max_ratio), float(lq_init_strength))
    ratio = ratio_cap * tau.pow(float(blend_pow))
    ratio = ratio * (timesteps >= int(t_min)).float()
    ratio_map = ratio.view(-1, 1, 1, 1).to(dtype=zt_hr.dtype, device=zt_hr.device)
    zt_mix = (1.0 - ratio_map) * zt_hr + ratio_map * zt_lr
    eps_eff = (zt_mix - alpha_t * zh) / sigma_t.clamp_min(1e-6)
    target_v = alpha_t * eps_eff - sigma_t * zh.float()
    return zt_mix, target_v, alpha_t, sigma_t, ratio, {
        "tala_mode": "high_t_blend_regime",
        "tala_t_min": int(t_min),
        "tala_t_max": int(t_max),
        "tala_effective_eps_std": float(eps_eff.detach().float().std().item()),
    }


def assert_strict_tala_configuration():
    if not USE_TALA_TRAIN:
        return
    with open(__file__, "r", encoding="utf-8") as f:
        script_text = f.read()
    banned_tokens = ("apply_" + "tala_" + "train_" + "blend(", "TALA_" + "TRAIN_" + "MAX_" + "TIMESTEP")
    bad = [tok for tok in banned_tokens if tok in script_text]
    if bad:
        raise RuntimeError(f"Strict TALA violation: banned legacy tokens still present: {bad}")

def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    if cond is None:
        return None
    if not torch.is_tensor(keep_mask):
        keep_mask = torch.tensor(keep_mask)

    def _find_device_dtype(x):
        if torch.is_tensor(x):
            return x.device, x.dtype
        if isinstance(x, dict):
            for v in x.values():
                found = _find_device_dtype(v)
                if found is not None:
                    return found
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None:
                    return found
        return None

    found = _find_device_dtype(cond)
    if found is None:
        return cond

    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)

    def _mask_tensor(x: torch.Tensor):
        m = keep_mask
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)

    def _mask_obj(x):
        if torch.is_tensor(x):
            return _mask_tensor(x)
        if isinstance(x, dict):
            return {k: _mask_obj(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_mask_obj(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_mask_obj(v) for v in x)
        return x

    return _mask_obj(cond)



def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): sha.update(chunk)
    return sha.hexdigest()


def assert_finite_tensor(name: str, x: torch.Tensor, stats: dict):
    if x is None:
        return
    if not torch.is_tensor(x):
        return
    if torch.isfinite(x).all():
        return
    stat_items = " | ".join([f"{k}={v}" for k, v in stats.items()])
    raise RuntimeError(f"[NaNGuard] non-finite detected in {name}. {stat_items}")

def _should_keep_fp32_on_save(param_name: str) -> bool:
    return any(tag in param_name for tag in FP32_SAVE_KEY_FRAGMENTS)

def collect_trainable_state_dict(model: nn.Module):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        tensor = param.detach().cpu()
        if _should_keep_fp32_on_save(name): tensor = tensor.float()
        state[name] = tensor
    return state

def validate_active_trainable_state_keys(trainable_sd: dict, required_fragments):
    keys = list(trainable_sd.keys())
    missing = []
    counts = {}
    for frag in required_fragments:
        c = sum(1 for k in keys if frag in k)
        counts[frag] = c
        if c == 0: missing.append(frag)
    if missing:
        raise RuntimeError("active trainable checkpoint validation failed: " + ", ".join(missing))
    return counts

def compute_injection_scale_reg(model: nn.Module, lambda_reg: float = 1e-4):
    if lambda_reg <= 0 or not hasattr(model, "injection_scales") or not hasattr(model, "injection_layers"):
        return torch.tensor(0.0, device=DEVICE)
    depth = max(1, len(getattr(model, "blocks", [])) - 1)
    reg = torch.tensor(0.0, device=DEVICE)
    for i, p_scale in enumerate(model.injection_scales):
        lid = int(model.injection_layers[i])
        u = float(lid) / float(depth)
        reg = reg + (u * u) * (F.softplus(p_scale) ** 2).mean()
    return reg * float(lambda_reg)

class ParamEMA:
    def __init__(self, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

    def register(self, named_params_iterable):
        for name, p in named_params_iterable:
            self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, named_params_iterable):
        d = self.decay
        one_minus_d = 1.0 - d
        for name, p in named_params_iterable:
            if name not in self.shadow:
                self.shadow[name] = p.detach().float().clone()
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=one_minus_d)

    @torch.no_grad()
    def apply(self, named_params_iterable):
        self.backup = {}
        for name, p in named_params_iterable:
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, named_params_iterable):
        for name, p in named_params_iterable:
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def collect_ema_named_params(pixart: nn.Module, adapter: nn.Module, mode: str = "small"):
    names = (
        "final_layer",
        "sft_cond_reduce",
        "sft_layers",
        "gcsa2",
        "gcsa3",
        "gcsa4",
        "lora_A",
        "lora_B",
        "x_embedder",
        "aug_embedder",
    )
    out = []
    for n, p in adapter.named_parameters():
        if p.requires_grad:
            out.append((f"adapter.{n}", p))
    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if mode == "all":
            out.append((f"pixart.{n}", p))
        elif any(k in n for k in names):
            out.append((f"pixart.{n}", p))
    return out


def log_injection_scale_stats(model: nn.Module, prefix: str = "[InjectScale]"):
    if not hasattr(model, "injection_scales"):
        return
    vals = []
    for p in model.injection_scales:
        vals.append(float(F.softplus(p.detach().float()).mean().item()))
    if len(vals) == 0:
        return
    k = min(5, len(vals))
    front_mean = float(np.mean(vals[:k]))
    back_mean = float(np.mean(vals[-k:]))
    print(f"{prefix} front{k}_mean={front_mean:.4f} back{k}_mean={back_mean:.4f} min={min(vals):.4f} max={max(vals):.4f} (softplus)")

def get_config_snapshot():
    return {
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "lora_rank": int(LORA_RANK),
        "lora_alpha": float(LORA_ALPHA),
        "lr_latent_noise_std": INIT_NOISE_STD,
        "loss_weights_mode": "fixed_v_latent_l1_lr_cons_gw",
        "adapter_type": "SRConvNetLSAAdapterV12",
        "control_type": "structure_early_band + adaptive_text + image_semantic + tala_train_consistency",
        "adapter_backbone": "V12_with_official_SMFANet_FMB",
        "adapter_token_head": "structure-only cond_map head",
        "internal_control_type": "disabled",
        "internal_control_layers": [],
        "semantic_prompt_branch": "enabled",
        "semantic_prompt_source": "CLIP + IP-Adapter Resampler + decoupled cross-attn",
        "semantic_prompt_tokens": 16,
        "use_semantic_branch": bool(USE_SEMANTIC_BRANCH),
        "use_native_text_path": True,
        "native_text_is_null": True,
        "use_adaptive_text_prompt": bool(USE_ADAPTIVE_TEXT_PROMPT),
        "strict_adaptive_prompt": bool(STRICT_ADAPTIVE_PROMPT),
        "adaptive_prompt_cache_root": ADAPTIVE_PROMPT_CACHE_ROOT,
        "use_tala_train": bool(USE_TALA_TRAIN),
        "tala_train_active_mode": str(TALA_TRAIN_ACTIVE_MODE),
        "tala_train_active_t_min": int(TALA_TRAIN_ACTIVE_T_MIN),
        "tala_train_blend_pow": float(TALA_TRAIN_BLEND_POW),
        "tala_train_max_ratio": float(TALA_TRAIN_MAX_RATIO),
        "tala_train_detach_lr_latent": bool(TALA_TRAIN_DETACH_LR_LATENT),
        "tala_train_high_t_mode": bool(TALA_TRAIN_HIGH_T_MODE),
        "use_timestep_aware_perceptual": bool(USE_TIMESTEP_AWARE_PERCEPTUAL),
        "percep_t_max": int(PERCEP_T_MAX),
        "percep_decode_max_samples": int(PERCEP_DECODE_MAX_SAMPLES),
        "lpips_late_weight": float(LPIPS_LATE_WEIGHT),
        "use_legacy_patch_lpips": bool(USE_LEGACY_PATCH_LPIPS),
        "semantic_fusion": "parallel_decoupled",
        "lr_control_role": "structure_only",
        "anchor_layers": list(ANCHOR_LAYERS),
        "semantic_layers": list(SEMANTIC_LAYERS),
        "tala_lite": False,
        "control_integration": "disabled",
        "seed": SEED,
        "lpips_enabled": bool(USE_LPIPS_PERCEP),
        "latent_l1_start": LATENT_L1_WEIGHT_START,
        "latent_l1_end": LATENT_L1_WEIGHT_END,
        "lr_cons_start": LR_CONS_WEIGHT_START,
        "lr_cons_end": LR_CONS_WEIGHT_END,
        "gw_start": GW_WEIGHT_START,
        "gw_end": GW_WEIGHT_END,
        "ckpt_select_mode": CKPT_SELECT_MODE,
        "ckpt_select_psnr_gate": CKPT_SELECT_PSNR_GATE,
    }


def validate_schedule_alignment():
    # Single-stage training: no stage schedule validation required.
    return


def validate_s2d_decoupling():
    return

# ================= 4. Data Pipeline =================
class DegradationPipeline:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.blur_kernels = BLUR_KERNELS
        self.blur_sigma_range = (0.2, 2.0)
        self.aniso_sigma_range = (0.2, 2.5)
        self.aniso_theta_range = (0.0, math.pi)
        self.noise_range = NOISE_RANGE
        self.downscale_factor = 0.25 

    def _sample_uniform(self, low, high, generator):
        if generator is None: return float(random.uniform(low, high))
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def _sample_int(self, low, high, generator):
        if generator is None: return int(random.randint(low, high))
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def _sample_choice(self, choices, generator):
        if generator is None: return random.choice(choices)
        idx = int(torch.randint(0, len(choices), (1,), generator=generator).item())
        return choices[idx]

    def _build_aniso_kernel(self, k, sigma_x, sigma_y, theta, device, dtype):
        ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        c, s = math.cos(theta), math.sin(theta)
        x_rot = c * xx + s * yy
        y_rot = -s * xx + c * yy
        kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_aniso_blur(self, img, k, sigma_x, sigma_y, theta):
        kernel = self._build_aniso_kernel(k, sigma_x, sigma_y, theta, img.device, img.dtype)
        kernel = kernel.view(1, 1, k, k)
        weight = kernel.repeat(img.shape[0], 1, 1, 1)
        img = img.unsqueeze(0)
        img = F.conv2d(img, weight, padding=k // 2, groups=img.shape[1])
        return img.squeeze(0)

    def _apply_jpeg(self, img, quality):
        img = img.detach().to(torch.float32)
        img_np = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out = TF.to_tensor(out).to(img.device, dtype=img.dtype)
        return out

    def _shuffle_ops(self, generator):
        ops = list(DEG_OPS)
        if generator is None: random.shuffle(ops)
        else:
            for i in range(len(ops) - 1, 0, -1):
                j = int(torch.randint(0, i + 1, (1,), generator=generator).item())
                ops[i], ops[j] = ops[j], ops[i]
        return ops

    def _sample_stage_params(self, generator):
        blur_applied = bool(self._sample_uniform(0.0, 1.0, generator) < 0.9)
        blur_is_aniso = bool(self._sample_uniform(0.0, 1.0, generator) < 0.5)
        if blur_applied:
            k_size = self._sample_choice(self.blur_kernels, generator)
            if blur_is_aniso:
                sigma_x = self._sample_uniform(*self.aniso_sigma_range, generator)
                sigma_y = self._sample_uniform(*self.aniso_sigma_range, generator)
                theta = self._sample_uniform(*self.aniso_theta_range, generator)
                sigma = 0.0
            else:
                sigma = self._sample_uniform(*self.blur_sigma_range, generator)
                sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        else:
            k_size = 0; sigma = 0.0; sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        resize_scale = self._sample_uniform(*RESIZE_SCALE_RANGE, generator)
        resize_interp = self._sample_choice(RESIZE_INTERP_MODES, generator)
        resize_interp_idx = RESIZE_INTERP_MODES.index(resize_interp)
        noise_std = self._sample_uniform(*self.noise_range, generator)
        jpeg_quality = self._sample_int(*JPEG_QUALITY_RANGE, generator)
        return {
            "blur_applied": blur_applied,
            "k_size": k_size,
            "sigma": sigma, "sigma_x": sigma_x, "sigma_y": sigma_y, "theta": theta,
            "resize_scale": resize_scale,
            "resize_interp_idx": resize_interp_idx, "resize_interp": resize_interp,
            "noise_std": noise_std,
            "jpeg_quality": jpeg_quality,
        }

    def __call__(self, hr_tensor, return_meta: bool = False, meta=None, generator=None):
        img = (hr_tensor + 1.0) * 0.5
        if meta is None:
            use_two_stage = bool(self._sample_uniform(0.0, 1.0, generator) < P_TWO_STAGE)
            ops_stage1 = self._shuffle_ops(generator)
            ops_stage2 = self._shuffle_ops(generator) if use_two_stage else []
            stage1 = self._sample_stage_params(generator)
            stage2 = self._sample_stage_params(generator) if use_two_stage else None
        else:
            use_two_stage = bool(int(meta.get("use_two_stage", torch.tensor(0)).item()))
            ops_stage1 = [op for op in str(meta.get("ops_stage1", ",".join(DEG_OPS))).split(",") if op]
            ops_stage2 = [op for op in str(meta.get("ops_stage2", "")).split(",") if op] if use_two_stage else []
            stage1 = {
                "blur_applied": bool(int(meta["stage1_blur_applied"].item())),
                "k_size": int(meta["stage1_k_size"].item()),
                "sigma": float(meta["stage1_sigma"].item()),
                "sigma_x": float(meta["stage1_sigma_x"].item()),
                "sigma_y": float(meta["stage1_sigma_y"].item()),
                "theta": float(meta["stage1_theta"].item()),
                "resize_scale": float(meta["stage1_resize_scale"].item()),
                "resize_interp_idx": int(meta["stage1_resize_interp"].item()),
                "resize_interp": RESIZE_INTERP_MODES[int(meta["stage1_resize_interp"].item())],
                "noise_std": float(meta["stage1_noise_std"].item()),
                "jpeg_quality": int(meta["stage1_jpeg_quality"].item()),
                "noise": meta.get("stage1_noise", None),
            }
            stage2 = None
            if use_two_stage:
                stage2 = {
                    "blur_applied": bool(int(meta["stage2_blur_applied"].item())),
                    "k_size": int(meta["stage2_k_size"].item()),
                    "sigma": float(meta["stage2_sigma"].item()),
                    "sigma_x": float(meta["stage2_sigma_x"].item()),
                    "sigma_y": float(meta["stage2_sigma_y"].item()),
                    "theta": float(meta["stage2_theta"].item()),
                    "resize_scale": float(meta["stage2_resize_scale"].item()),
                    "resize_interp_idx": int(meta["stage2_resize_interp"].item()),
                    "resize_interp": RESIZE_INTERP_MODES[int(meta["stage2_resize_interp"].item())],
                    "noise_std": float(meta["stage2_noise_std"].item()),
                    "jpeg_quality": int(meta["stage2_jpeg_quality"].item()),
                    "noise": meta.get("stage2_noise", None),
                }

        def apply_ops(img_in, ops, params):
            out = img_in
            stage_noise = None
            for op in ops:
                if op == "blur" and params["blur_applied"]:
                    if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                        out = self._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                    else: out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
                elif op == "resize":
                    mid_h = max(1, int(round(self.crop_size * params["resize_scale"])))
                    mid_w = max(1, int(round(self.crop_size * params["resize_scale"])))
                    out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
                elif op == "noise":
                    if params["noise_std"] > 0:
                        if meta is None:
                            if generator is None: noise = torch.randn_like(out)
                            else: noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
                        else:
                            noise = params.get("noise")
                            if noise is None: noise = torch.zeros_like(out)
                            else: noise = noise.to(out.device, dtype=out.dtype)
                        stage_noise = noise
                        out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
                    else: stage_noise = torch.zeros_like(out)
                elif op == "jpeg": out = self._apply_jpeg(out, params["jpeg_quality"])
            if stage_noise is None: stage_noise = torch.zeros_like(out)
            return out, stage_noise

        lr_small, stage1_noise = apply_ops(img, ops_stage1, stage1)
        stage2_noise = torch.zeros_like(lr_small)
        if use_two_stage: lr_small, stage2_noise = apply_ops(lr_small, ops_stage2, stage2)

        down_h = int(self.crop_size * self.downscale_factor)
        down_w = int(self.crop_size * self.downscale_factor)
        lr_small = TF.resize(lr_small, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = TF.resize(lr_small, [self.crop_size, self.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = (lr_out * 2.0 - 1.0).clamp(-1.0, 1.0)

        # FIXED: Strictly return 2 values
        if return_meta:
            meta_out = {
                "stage1_blur_applied": torch.tensor(int(stage1["blur_applied"]), dtype=torch.int64),
                "stage1_k_size": torch.tensor(int(stage1["k_size"]), dtype=torch.int64),
                "stage1_sigma": torch.tensor(float(stage1["sigma"]), dtype=torch.float32),
                "stage1_sigma_x": torch.tensor(float(stage1["sigma_x"]), dtype=torch.float32),
                "stage1_sigma_y": torch.tensor(float(stage1["sigma_y"]), dtype=torch.float32),
                "stage1_theta": torch.tensor(float(stage1["theta"]), dtype=torch.float32),
                "stage1_noise_std": torch.tensor(float(stage1["noise_std"]), dtype=torch.float32),
                "stage1_noise": stage1_noise.detach().cpu().float(),
                "stage1_resize_scale": torch.tensor(float(stage1["resize_scale"]), dtype=torch.float32),
                "stage1_resize_interp": torch.tensor(int(stage1["resize_interp_idx"]), dtype=torch.int64),
                "stage1_jpeg_quality": torch.tensor(int(stage1["jpeg_quality"]), dtype=torch.int64),
                "stage2_blur_applied": torch.tensor(int(stage2["blur_applied"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_k_size": torch.tensor(int(stage2["k_size"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_sigma": torch.tensor(float(stage2["sigma"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_x": torch.tensor(float(stage2["sigma_x"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_y": torch.tensor(float(stage2["sigma_y"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_theta": torch.tensor(float(stage2["theta"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise_std": torch.tensor(float(stage2["noise_std"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise": stage2_noise.detach().cpu().float(),
                "stage2_resize_scale": torch.tensor(float(stage2["resize_scale"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_resize_interp": torch.tensor(int(stage2["resize_interp_idx"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_jpeg_quality": torch.tensor(int(stage2["jpeg_quality"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "use_two_stage": torch.tensor(int(use_two_stage), dtype=torch.int64),
                "ops_stage1": ",".join(ops_stage1),
                "ops_stage2": ",".join(ops_stage2),
                "down_h": torch.tensor(int(down_h), dtype=torch.int64),
                "down_w": torch.tensor(int(down_w), dtype=torch.int64),
            }
            return lr_out, meta_out
        return lr_out

# ================= 6. Datasets (Correctly Placed BEFORE Main) =================
def _scan_images(root):
    root_p = Path(root)
    if not root_p.exists():
        return []
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        out.extend(root_p.rglob(ext))
    return sorted(out)

def _normalize_pair_stem(stem: str) -> str:
    key = stem.lower()
    for tok in ("_hr", "_lr4", "_lr", "x4"):
        key = key.replace(tok, "")
    return key

def build_paired_file_list(df2k_hr_root: str, df2k_lr_root: str, realsr_roots):
    pairs = []
    df2k_hr = _scan_images(df2k_hr_root)
    if df2k_hr and os.path.exists(df2k_lr_root):
        lr_map = {}
        for p in _scan_images(df2k_lr_root):
            lr_map.setdefault(_normalize_pair_stem(p.stem), []).append(str(p))
        for hr_path in df2k_hr:
            key = _normalize_pair_stem(hr_path.stem)
            lr_cands = lr_map.get(key, [])
            if not lr_cands:
                continue
            best_lr = sorted(lr_cands, key=lambda x: ("lr4" not in x.lower() and "x4" not in x.lower(), len(x)))[0]
            pairs.append((best_lr, str(hr_path)))

    for root in realsr_roots:
        for hr_path in _scan_images(root):
            stem = hr_path.stem
            if "_hr" not in stem.lower():
                continue
            lr_name = stem.replace("_HR", "_LR4").replace("_hr", "_lr4") + hr_path.suffix
            lr_path = hr_path.with_name(lr_name)
            if lr_path.exists():
                pairs.append((str(lr_path), str(hr_path)))
    return sorted(pairs)

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

class DF2K_Online_Dataset(Dataset):
    def __init__(self, crop_size=512, is_train=True, scale=4):
        self.pairs = build_paired_file_list(TRAIN_DF2K_HR_DIR, TRAIN_DF2K_LR_DIR, TRAIN_REALSR_DIRS)
        if len(self.pairs) == 0:
            raise RuntimeError("No LR/HR pairs found from DF2K/RealSR training roots.")
        self.crop_size = crop_size
        self.lr_patch = crop_size // scale
        self.scale = scale
        self.is_train = is_train
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0

    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def _make_generator(self, idx: int):
        gen = torch.Generator(); seed = SEED + (self.epoch * 1_000_000) + int(idx); gen.manual_seed(seed)
        return gen
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        try:
            lr_path, hr_path = self.pairs[idx]
            lr_pil = Image.open(lr_path).convert("RGB")
            hr_pil = Image.open(hr_path).convert("RGB")
        except: return self.__getitem__((idx + 1) % len(self))
        gen = None
        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=self.scale)

        if self.is_train:
            gen = self._make_generator(idx)
            w_l, h_l = lr_aligned.size
            if h_l < self.lr_patch or w_l < self.lr_patch:
                lr_aligned = TF.resize(lr_aligned, (self.lr_patch, self.lr_patch), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                hr_aligned = TF.resize(hr_aligned, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                x = 0
                y = 0
            else:
                max_x = w_l - self.lr_patch
                max_y = h_l - self.lr_patch
                x = int(torch.randint(0, max_x + 1, (1,), generator=gen).item())
                y = int(torch.randint(0, max_y + 1, (1,), generator=gen).item())
            lr_crop = TF.crop(lr_aligned, y, x, self.lr_patch, self.lr_patch)
            hr_crop = TF.crop(hr_aligned, y * self.scale, x * self.scale, self.crop_size, self.crop_size)
            if torch.rand(1, generator=gen).item() < 0.5:
                lr_crop = TF.hflip(lr_crop)
                hr_crop = TF.hflip(hr_crop)
            k = int(torch.randint(0, 4, (1,), generator=gen).item())
            if k:
                angle = 90 * k
                lr_crop = TF.rotate(lr_crop, angle)
                hr_crop = TF.rotate(hr_crop, angle)
        else:
            lr_crop = TF.center_crop(lr_aligned, (self.lr_patch, self.lr_patch))
            hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))

        lr_up = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up))
        lr_small_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": hr_path, "sample_key": make_sample_key(lr_path)}

class DF2K_Val_Fixed_Dataset(Dataset):
    def __init__(self, hr_root, lr_root=None, crop_size=512):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.lr_root = lr_root; self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3); self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        lr_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path); lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
                lr_small_pil = TF.center_crop(lr_aligned, (self.crop_size//4, self.crop_size//4))
                hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
        if lr_crop is None:
            hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
            w, h = hr_crop.size
            lr_small_pil = hr_crop.resize((w//4, h//4), Image.BICUBIC)
        lr_up_pil = lr_small_pil.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up_pil))
        lr_small_tensor = self.norm(self.to_tensor(lr_small_pil))
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": hr_path, "sample_key": make_sample_key(lr_p if self.lr_root and os.path.exists(lr_p) else hr_path)}

class DF2K_Val_Degraded_Dataset(Dataset):
    def __init__(self, hr_root, crop_size=512, seed=3407, deg_mode="highorder"):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor(); self.pipeline = DegradationPipeline(crop_size)
        self.seed = int(seed); self.deg_mode = deg_mode
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        if self.deg_mode == "bicubic":
            lr_small01 = TF.resize((hr_tensor + 1.0) * 0.5, (self.crop_size // 4, self.crop_size // 4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_up01 = TF.resize(lr_small01, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = (lr_up01 * 2.0 - 1.0).clamp(-1.0, 1.0)
            lr_small_tensor = (lr_small01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            gen = torch.Generator(); gen.manual_seed(self.seed + idx)
            lr_tensor, _ = self.pipeline(hr_tensor, return_meta=True, generator=gen)
            # fallback when pipeline only outputs lr_up
            lr_small_tensor = F.interpolate(lr_tensor.unsqueeze(0), size=(self.crop_size // 4, self.crop_size // 4), mode="bicubic", align_corners=False, antialias=True).squeeze(0).clamp(-1.0, 1.0)
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": hr_path, "sample_key": make_sample_key(hr_path)}

class ValPackDataset(Dataset):
    def __init__(self, pack_dir: str, lr_dir_name: str = "lq512", crop_size: int = 512):
        self.pack_dir = Path(pack_dir); self.hr_dir = self.pack_dir / "gt512"; self.lr_dir = self.pack_dir / lr_dir_name
        if not self.hr_dir.is_dir(): raise FileNotFoundError(f"gt512 dir not found: {self.hr_dir}")
        if not self.lr_dir.is_dir(): raise FileNotFoundError(f"LR dir not found: {self.lr_dir}")
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor(); self.hr_paths = sorted(list(self.hr_dir.glob("*.png")))
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; name = hr_path.stem; lr_path = self.lr_dir / f"{name}.png"
        if not lr_path.is_file(): raise FileNotFoundError(f"LR image missing: {lr_path}")
        hr_pil = Image.open(hr_path).convert("RGB"); lr_pil = Image.open(lr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size)); lr_crop = TF.center_crop(lr_pil, (self.crop_size, self.crop_size))
        if lr_crop.size != (self.crop_size, self.crop_size):
            lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        hr_tensor = self.norm(self.to_tensor(hr_crop)); lr_tensor = self.norm(self.to_tensor(lr_crop))
        lr_small_tensor = F.interpolate(lr_tensor.unsqueeze(0), size=(self.crop_size // 4, self.crop_size // 4), mode="bicubic", align_corners=False, antialias=True).squeeze(0).clamp(-1.0, 1.0)
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": str(hr_path), "sample_key": make_sample_key(str(lr_path))}

class RealSR_Val_Paired_Dataset(Dataset):
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
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": hr_path, "sample_key": make_sample_key(lr_path)}


# ================= 7. LoRA =================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device); self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None: self.base.bias.requires_grad = False
    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)

def _block_id_from_name(name: str):
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _lora_target_kind(module_name: str):
    if ("attn.qkv" in module_name) or ("attn.proj" in module_name):
        return "attn"
    return None


def apply_lora(model):
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
        rank = int(LORA_RANK)
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        child = name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, rank, alpha=float(LORA_ALPHA)))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={LORA_RANK}, alpha={LORA_ALPHA}).")


def configure_pixart_trainable_params(pixart: nn.Module, train_x_embedder: bool = False):
    total_trainable_before = sum(1 for _, p in pixart.named_parameters() if p.requires_grad)
    anchor_ids = sorted([int(i) for i in getattr(pixart, "anchor_layers", ANCHOR_LAYERS)])
    for name, p in pixart.named_parameters():
        p.requires_grad = False

        if (
            "final_layer" in name
            or "lora_A" in name
            or "lora_B" in name
            or "sft_cond_reduce" in name
        ):
            p.requires_grad = True
        if any(f"sft_alpha.{i}" in name for i in anchor_ids):
            p.requires_grad = True
        if any(f"sft_layers.{i}." in name for i in anchor_ids):
            p.requires_grad = True
        if any(f"tasr_time_gates.{i}." in name for i in anchor_ids):
            p.requires_grad = True
        if ("hybrid_cross_attn" in name) or ("semantic_alpha" in name):
            p.requires_grad = True

    if not train_x_embedder:
        for n, p in pixart.named_parameters():
            if "x_embedder" in n and p.requires_grad:
                p.requires_grad_(False)

    if ENABLE_LORA:
        for n, p in pixart.named_parameters():
            if ".base.weight" in n or ".base.bias" in n:
                p.requires_grad_(False)
                continue
            if ("lora_A" in n) or ("lora_B" in n):
                bid = _block_id_from_name(n)
                kind = _lora_target_kind(n)
                p.requires_grad_(bool(bid is not None and 0 <= bid <= 27 and kind == "attn"))

    total_trainable_after = sum(1 for _, p in pixart.named_parameters() if p.requires_grad)
    if total_trainable_after == 0:
        raise RuntimeError("No PixArt trainable parameters selected after configuration.")
    trainable_sft_layer_ids = sorted(list({int(n.split("sft_layers.")[1].split(".")[0]) for n, p in pixart.named_parameters() if p.requires_grad and "sft_layers." in n}))
    trainable_sft_alpha_ids = sorted(list({int(n.split("sft_alpha.")[1].split(".")[0]) for n, p in pixart.named_parameters() if p.requires_grad and "sft_alpha." in n}))
    trainable_tasr_gate_ids = sorted(list({int(n.split("tasr_time_gates.")[1].split(".")[0]) for n, p in pixart.named_parameters() if p.requires_grad and "tasr_time_gates." in n}))
    outside_anchor = sorted(list(set(trainable_sft_layer_ids) - set(anchor_ids)))
    print(f"[SFT-Trainable] anchor_layers={anchor_ids}")
    print(f"[SFT-Trainable] actual_trainable_sft_ids={trainable_sft_layer_ids}")
    print(f"[SFT-Trainable] actual_trainable_sft_alpha_ids={trainable_sft_alpha_ids}")
    print(f"[SFT-Trainable] actual_trainable_tasr_gate_ids={trainable_tasr_gate_ids}")
    print(f"[SFT-Trainable] outside_anchor={outside_anchor}")
    if outside_anchor:
        raise RuntimeError(f"SFT trainable mismatch: found trainable SFT layers not in anchor_layers: {outside_anchor}")
    if trainable_sft_layer_ids != anchor_ids:
        raise RuntimeError(f"SFT trainable mismatch: trainable_sft_ids={trainable_sft_layer_ids}, anchor_layers={anchor_ids}")
    if trainable_sft_alpha_ids != anchor_ids:
        raise RuntimeError(f"SFT alpha mismatch: trainable_sft_alpha_ids={trainable_sft_alpha_ids}, anchor_layers={anchor_ids}")
    if trainable_tasr_gate_ids != anchor_ids:
        raise RuntimeError(f"TASR gate mismatch: trainable_tasr_gate_ids={trainable_tasr_gate_ids}, anchor_layers={anchor_ids}")
    print(f"✅ PixArt trainable configured(single-stage): before={total_trainable_before}, after={total_trainable_after}")


def compute_save_keys_for_stages(pixart: nn.Module, train_x_embedder: bool = True):
    return set()


def collect_state_dict_by_keys(model: nn.Module, keys):
    state = model.state_dict()
    return {k: state[k].detach().float().cpu() for k in sorted(keys) if k in state}


def collect_sem_adapter_trainable_state_dict(sem_adapter: nn.Module):
    full = sem_adapter.state_dict()
    keep = {}
    for k, v in full.items():
        if (
            k.startswith("resampler.")
            or k.startswith("proj_norm.")
            or k.startswith("proj.")
            or k == "out_scale"
        ):
            keep[k] = v.detach().float().cpu()
    return keep


def is_allowed_sem_adapter_missing(missing, unexpected):
    missing = set(missing)
    unexpected = set(unexpected)
    if len(unexpected) > 0:
        return False
    allowed = {"out_scale"}
    for k in list(missing):
        if k in allowed:
            continue
        if k.startswith("image_encoder."):
            continue
        return False
    return True


def should_run_proxy_validation(epoch_1based: int) -> bool:
    if epoch_1based < 15:
        return (epoch_1based % 5) == 0
    if epoch_1based < 25:
        return ((epoch_1based - 15) % 2) == 0
    return True


def build_optimizer_and_clippables(pixart: nn.Module, adapter: nn.Module, sem_adapter: nn.Module):
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    lora_params, structure_params, semantic_hpa_params = [], [], []
    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if ("lora_A" in n) or ("lora_B" in n):
            lora_params.append(p)
            continue
        if ("hybrid_cross_attn" in n) or ("semantic_alpha" in n):
            semantic_hpa_params.append(p)
            continue
        structure_params.append(p)

    sem_adapter_params = [p for n, p in sem_adapter.named_parameters() if p.requires_grad and (("resampler" in n) or ("proj_norm" in n) or ("proj." in n) or ("out_scale" in n))]

    optim_groups = []
    if len(adapter_params) + len(structure_params) > 0:
        optim_groups.append({"params": adapter_params + structure_params, "lr": 3e-4, "weight_decay": 0.01})
    if len(semantic_hpa_params) + len(sem_adapter_params) > 0:
        optim_groups.append({"params": semantic_hpa_params + sem_adapter_params, "lr": 1e-4, "weight_decay": 0.01})
    if len(lora_params) > 0:
        optim_groups.append({"params": lora_params, "lr": 1e-4, "weight_decay": 0.01})

    if len(optim_groups) == 0:
        raise RuntimeError("No optimizer groups built; check stage trainable settings.")

    optimizer = torch.optim.AdamW(optim_groups)
    params_to_clip = adapter_params + structure_params + semantic_hpa_params + sem_adapter_params + lora_params

    pixart_trainable = [p for p in pixart.parameters() if p.requires_grad]
    grouped = structure_params + lora_params + semantic_hpa_params
    if len({id(p) for p in grouped}) != len(grouped):
        raise RuntimeError("Optimizer grouping has duplicate PixArt params across groups.")
    if {id(p) for p in grouped} != {id(p) for p in pixart_trainable}:
        raise RuntimeError("Optimizer grouping does not exactly cover PixArt trainable params.")

    group_counts = {
        "structure_group_pixart": len(structure_params),
        "lora": len(lora_params),
        "semantic_hpa_group_pixart": len(semantic_hpa_params),
        "sem_adapter_group": len(sem_adapter_params),
        "adapter": len(adapter_params),
    }
    return optimizer, params_to_clip, group_counts


def log_critical_path_gradients(step: int, pixart: nn.Module, adapter: nn.Module):
    if step > 200 or (step % 50 != 0):
        return
    pix_named = dict(pixart.named_parameters())
    ad_named = dict(adapter.named_parameters())
    watched = [
        ("pixart.final_layer.linear.weight", pix_named.get("final_layer.linear.weight", None)),
        ("pixart.sft_layers.2.scale_conv1.weight", pix_named.get("sft_layers.2.scale_conv1.weight", None)),
        ("pixart.sft_alpha.2", pix_named.get("sft_alpha.2", None)),
        ("adapter.stage1.0.smfa.linear_0.weight", ad_named.get("stage1.0.smfa.linear_0.weight", None)),
        ("adapter.stage3.0.pcfn.conv_0.weight", ad_named.get("stage3.0.pcfn.conv_0.weight", None)),
        ("adapter.to32.1.weight", ad_named.get("to32.1.weight", None)),
    ]
    msg = [f"[GradSanity][step={step}]"]
    warnings = []
    for name, p in watched:
        if p is None:
            msg.append(f"{name}=N/A")
            continue
        w_norm = float(p.detach().float().norm().item())
        g_norm = 0.0 if p.grad is None else float(p.grad.detach().float().norm().item())
        msg.append(f"{name}: g={g_norm:.3e}, w={w_norm:.3e}")
        if g_norm < 1e-12:
            warnings.append(name)
    print(" | ".join(msg))
    if warnings:
        print(f"⚠️ [GradSanity] near-zero grad on critical paths: {warnings}")


# ================= 8. Checkpointing =================
def should_keep_ckpt(psnr_v, lpips_v):
    psnr_ok = math.isfinite(psnr_v)
    if not psnr_ok:
        return (999, float("inf"), float("inf"))
    lpips_ok = math.isfinite(lpips_v)
    lp = float(lpips_v) if lpips_ok else float("inf")
    p = float(psnr_v)
    mode = str(CKPT_SELECT_MODE).lower()
    gate = float(CKPT_SELECT_PSNR_GATE)
    if mode == "psnr_gate_then_lpips":
        gate_passed = p >= gate
        if gate_passed:
            return (0, lp, -p, 1)
        return (0, -p, lp, 0)
    if mode == "lpips_first":
        return (0, lp, -p, int(p >= gate))
    return (0, -p, lp, int(p >= gate))

def atomic_torch_save(state, path):
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp); os.replace(tmp, path); return True, "zip"
    except Exception as e_zip:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        try:
            torch.save(state, tmp, _use_new_zipfile_serialization=False); os.replace(tmp, path); return True, f"legacy ({e_zip})"
        except Exception as e_old:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            return False, f"zip_error={e_zip}; legacy_error={e_old}"

def save_smart(
    epoch,
    global_step,
    pixart,
    adapter,
    sem_adapter,
    optimizer,
    best_records,
    metrics,
    dl_gen,
    ema=None,
    keep_keys=None,
    eval_source: str = "raw",
    eval_steps: int = 50,
    eval_tag: str = "",
    export_eval_weights: bool = True,
    ema_named_params=None,
    last_val_psnr: float = -1.0,
):
    global BASE_PIXART_SHA256
    eval_source = str(eval_source).lower()
    psnr_v, ssim_v, lpips_v = metrics
    rank = should_keep_ckpt(psnr_v, lpips_v)
    if not isinstance(rank, tuple) or len(rank) < 2:
        raise RuntimeError(f"should_keep_ckpt must return a tuple(rank>=2), got {rank}")
    priority = int(rank[0])
    score = tuple(rank[1:])
    gate_passed = bool(float(psnr_v) >= float(CKPT_SELECT_PSNR_GATE)) if math.isfinite(psnr_v) else False
    print(f"[CKPT-RANK] ckpt_priority_mode={CKPT_SELECT_MODE} psnr_gate={CKPT_SELECT_PSNR_GATE:.3f} gate_passed={int(gate_passed)} rank={rank}")

    current_record = {
        "path": None,
        "epoch": epoch,
        "priority": priority,
        "score": score,
        "psnr": psnr_v,
        "lpips": lpips_v,
        "eval_source": eval_source,
        "eval_steps": int(eval_steps),
    }

    prev_entry = best_records[0] if len(best_records) > 0 else None
    if isinstance(prev_entry, dict) and "global_best_record" in prev_entry:
        prev_global_best = prev_entry.get("global_best_record", None)
        prev_aux = dict(prev_entry.get("best_aux_meta", {}))
    elif isinstance(prev_entry, dict):
        prev_global_best = dict(prev_entry)
        prev_aux = {
            "best_psnr": float(prev_entry.get("best_psnr", -float("inf"))),
            "best_psnr_step": int(prev_entry.get("best_psnr_step", -1)),
        }
    else:
        prev_global_best = None
        prev_aux = {}

    # keep a single global best proxy checkpoint by should_keep_ckpt rule.
    prev_best = prev_global_best
    prev_priority = int(prev_best['priority']) if prev_best is not None and 'priority' in prev_best else None
    prev_score = prev_best.get('score', (float("inf"),)) if prev_best is not None else None
    if not isinstance(prev_score, tuple):
        prev_score = (prev_score,)
    save_as_best = (
        prev_best is None
        or (priority < prev_priority)
        or (priority == prev_priority and score < prev_score)
    )

    source_tag = eval_source if eval_source in ("raw", "ema") else "raw"
    best_ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    current_record['path'] = best_ckpt_path

    if BASE_PIXART_SHA256 is None and os.path.exists(PIXART_PATH):
        try:
            BASE_PIXART_SHA256 = file_sha256(PIXART_PATH)
        except Exception as e:
            print(f"⚠️ Base PixArt hash failed (non-fatal): {e}")
            BASE_PIXART_SHA256 = None

    keep_keys = set(keep_keys or set())

    def _build_resume_state_snapshot():
        pixart_sd = collect_trainable_state_dict(pixart)
        required_frags = get_required_active_key_fragments_for_model(pixart)
        active_key_counts = validate_active_trainable_state_keys(pixart_sd, required_frags)
        lora_key_count = sum(("lora_A" in k or "lora_B" in k) for k in pixart_sd.keys())
        if ENABLE_LORA and lora_key_count == 0:
            raise RuntimeError("LoRA is enabled but no LoRA keys found in pixart_trainable.")
        if lora_key_count == 0:
            print("ℹ️ LoRA save check skipped (LoRA disabled or no trainable LoRA tensors).")
        else:
            print(f"✅ LoRA save check: {lora_key_count} tensors")
        print("✅ active save check:", ", ".join([f"{k}={v}" for k, v in active_key_counts.items()]))

        adapter_sd = {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()}
        sem_adapter_sd = collect_sem_adapter_trainable_state_dict(sem_adapter)
        adapter_token_head_counts = {
            "adapter.to32": int(sum(1 for k in adapter_sd.keys() if k.startswith("to32."))),
        }
        print("✅ adapter token head save check:", ", ".join([f"{k}={v}" for k, v in adapter_token_head_counts.items()]))
        state = {
            "epoch": epoch,
            "step": global_step,
            "lora_rank": int(LORA_RANK),
            "lora_alpha": float(LORA_ALPHA),
            "adapter": adapter_sd,
            "sem_adapter": sem_adapter_sd,
            "adapter_token_head_counts": adapter_token_head_counts,
            "optimizer": optimizer.state_dict(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dl_gen_state": dl_gen.get_state(),
            "pixart_trainable": pixart_sd,
            "best_records": None,
            "config_snapshot": get_config_snapshot(),
            "base_pixart_sha256": BASE_PIXART_SHA256,
            "env_info": {"torch": torch.__version__, "numpy": np.__version__},
            "ema_state": ({k: v.detach().cpu().float() for k, v in ema.shadow.items()} if ema is not None else None),
            "checkpoint_role": "last",
            "last_val_psnr": float(last_val_psnr),
            "best_eval_source": source_tag,
            "best_eval_steps": int(eval_steps),
            "best_eval_metrics": {"psnr": float(psnr_v), "ssim": float(ssim_v), "lpips": float(lpips_v)},
            "best_eval_tag": str(eval_tag),
            "train_runtime_log": dict(LAST_TRAIN_LOG),
            "layer_config": {
                "anchor_layers": sorted(list(getattr(pixart, "anchor_layers", ANCHOR_LAYERS))),
                "semantic_layers": list(getattr(pixart, "semantic_layers", SEMANTIC_LAYERS)),
                "use_tala_train": bool(USE_TALA_TRAIN),
                "tala_active_mode": str(TALA_TRAIN_ACTIVE_MODE),
                "tala_active_t_min": int(TALA_TRAIN_ACTIVE_T_MIN),
                "tala_train_blend_pow": float(TALA_TRAIN_BLEND_POW),
                "tala_max_ratio": float(TALA_TRAIN_MAX_RATIO),
                "tala_train_detach_lr_latent": bool(TALA_TRAIN_DETACH_LR_LATENT),
                "use_adaptive_text_prompt": bool(USE_ADAPTIVE_TEXT_PROMPT),
                "ckpt_select_mode": str(CKPT_SELECT_MODE),
                "ckpt_select_psnr_gate": float(CKPT_SELECT_PSNR_GATE),
                "gate_passed_when_saved": bool(gate_passed),
            },
        }
        return state

    def _build_eval_state_snapshot(checkpoint_role: str):
        pixart_sd = collect_trainable_state_dict(pixart)
        required_frags = get_required_active_key_fragments_for_model(pixart)
        active_key_counts = validate_active_trainable_state_keys(pixart_sd, required_frags)
        lora_key_count = sum(("lora_A" in k or "lora_B" in k) for k in pixart_sd.keys())
        if ENABLE_LORA and lora_key_count == 0:
            raise RuntimeError("LoRA is enabled but no LoRA keys found in pixart_trainable.")
        if lora_key_count == 0:
            print("ℹ️ LoRA save check skipped (LoRA disabled or no trainable LoRA tensors).")
        else:
            print(f"✅ LoRA save check: {lora_key_count} tensors")
        print("✅ active save check:", ", ".join([f"{k}={v}" for k, v in active_key_counts.items()]))

        adapter_sd = {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()}
        sem_adapter_sd = collect_sem_adapter_trainable_state_dict(sem_adapter)
        adapter_token_head_counts = {
            "adapter.to32": int(sum(1 for k in adapter_sd.keys() if k.startswith("to32."))),
        }
        print("✅ adapter token head save check:", ", ".join([f"{k}={v}" for k, v in adapter_token_head_counts.items()]))
        state = {
            "epoch": epoch,
            "step": global_step,
            "lora_rank": int(LORA_RANK),
            "lora_alpha": float(LORA_ALPHA),
            "adapter": adapter_sd,
            "sem_adapter": sem_adapter_sd,
            "adapter_token_head_counts": adapter_token_head_counts,
            "pixart_trainable": pixart_sd,
            "best_records": None,
            "config_snapshot": get_config_snapshot(),
            "base_pixart_sha256": BASE_PIXART_SHA256,
            "env_info": {"torch": torch.__version__, "numpy": np.__version__},
            "checkpoint_role": str(checkpoint_role),
            "last_val_psnr": float(last_val_psnr),
            "best_eval_source": source_tag,
            "best_eval_steps": int(eval_steps),
            "best_eval_metrics": {"psnr": float(psnr_v), "ssim": float(ssim_v), "lpips": float(lpips_v)},
            "best_eval_tag": str(eval_tag),
            "train_runtime_log": dict(LAST_TRAIN_LOG),
            "layer_config": {
                "anchor_layers": sorted(list(getattr(pixart, "anchor_layers", ANCHOR_LAYERS))),
                "semantic_layers": list(getattr(pixart, "semantic_layers", SEMANTIC_LAYERS)),
                "use_tala_train": bool(USE_TALA_TRAIN),
                "tala_active_mode": str(TALA_TRAIN_ACTIVE_MODE),
                "tala_active_t_min": int(TALA_TRAIN_ACTIVE_T_MIN),
                "tala_train_blend_pow": float(TALA_TRAIN_BLEND_POW),
                "tala_max_ratio": float(TALA_TRAIN_MAX_RATIO),
                "tala_train_detach_lr_latent": bool(TALA_TRAIN_DETACH_LR_LATENT),
                "use_adaptive_text_prompt": bool(USE_ADAPTIVE_TEXT_PROMPT),
                "ckpt_select_mode": str(CKPT_SELECT_MODE),
                "ckpt_select_psnr_gate": float(CKPT_SELECT_PSNR_GATE),
                "gate_passed_when_saved": bool(gate_passed),
            },
        }
        return state

    new_global_best = dict(current_record) if save_as_best else (dict(prev_global_best) if prev_global_best is not None else dict(current_record))

    best_psnr = float(prev_aux.get("best_psnr", -float("inf")))
    if math.isfinite(psnr_v) and psnr_v > best_psnr:
        best_psnr = float(psnr_v)
        best_psnr_step = int(global_step)
    else:
        best_psnr_step = int(prev_aux.get("best_psnr_step", global_step if best_psnr > -float("inf") else -1))

    new_aux_meta = {
        "best_psnr": float(best_psnr),
        "best_psnr_step": int(best_psnr_step),
    }
    next_best_records = [{
        "global_best_record": new_global_best,
        "best_aux_meta": new_aux_meta,
    }]

    # Always save full training-state checkpoint for resume
    state_last = _build_resume_state_snapshot()
    state_last["best_records"] = next_best_records
    last_path = LAST_CKPT_PATH
    ok_last, msg_last = atomic_torch_save(state_last, last_path)
    if ok_last:
        print(f"💾 Saved last checkpoint to {last_path} [{msg_last}]")
    else:
        print(f"❌ Failed to save last.pth: {msg_last}")

    best_saved = False
    if save_as_best:
        state_best_train = _build_eval_state_snapshot(checkpoint_role="best")
        state_best_train["best_records"] = next_best_records
        ok_train, msg_train = atomic_torch_save(state_best_train, best_ckpt_path)
        if ok_train:
            print(f"🏆 Updated best proxy checkpoint: {os.path.basename(best_ckpt_path)} [{msg_train}] source={source_tag}")
            best_saved = True
        else:
            print(f"❌ Failed to save best proxy checkpoint: {msg_train}")

    # Disabled to reduce CPU memory pressure during checkpoint save.
    # if math.isfinite(psnr_v) and psnr_v >= new_aux_meta["best_psnr"]:
    #     state_psnr = _build_eval_state_snapshot(checkpoint_role="best_psnr")
    #     state_psnr["best_records"] = next_best_records
    #     atomic_torch_save(state_psnr, BEST_PSNR_PATH)
    print("[CKPT-SAVE] last=resume_full best=eval_light best_psnr=disabled")

    # remove legacy rolling-best artifacts from older runs to keep CKPT_DIR clean
    if best_saved:
        for fname in os.listdir(CKPT_DIR):
            if fname.startswith("best_train_") or fname.startswith("best_eval_"):
                stale = os.path.join(CKPT_DIR, fname)
                if os.path.isfile(stale):
                    try:
                        os.remove(stale)
                    except Exception:
                        pass
    return next_best_records


def save_last_resume_only(
    epoch,
    global_step,
    pixart,
    adapter,
    sem_adapter,
    optimizer,
    best_records,
    dl_gen,
    ema=None,
    ema_named_params=None,
    last_val_psnr: float = -1.0,
):
    adapter_sd = {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()}
    sem_adapter_sd = collect_sem_adapter_trainable_state_dict(sem_adapter)
    pixart_sd = collect_trainable_state_dict(pixart)
    state_last = {
        "epoch": epoch,
        "step": global_step,
        "lora_rank": int(LORA_RANK),
        "lora_alpha": float(LORA_ALPHA),
        "adapter": adapter_sd,
        "sem_adapter": sem_adapter_sd,
        "adapter_token_head_counts": {
            "adapter.to32": int(sum(1 for k in adapter_sd.keys() if k.startswith("to32."))),
        },
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "dl_gen_state": dl_gen.get_state(),
        "pixart_trainable": pixart_sd,
        "best_records": best_records,
        "config_snapshot": get_config_snapshot(),
        "base_pixart_sha256": BASE_PIXART_SHA256,
        "env_info": {"torch": torch.__version__, "numpy": np.__version__},
        "ema_state": ({k: v.detach().cpu().float() for k, v in ema.shadow.items()} if ema is not None else None),
        "checkpoint_role": "last",
        "last_val_psnr": float(last_val_psnr),
        "best_eval_source": "raw",
        "best_eval_steps": int(BEST_VAL_STEPS),
        "best_eval_metrics": None,
        "best_eval_tag": "skip_proxy_val",
        "train_runtime_log": dict(LAST_TRAIN_LOG),
        "layer_config": {
            "anchor_layers": sorted(list(getattr(pixart, "anchor_layers", ANCHOR_LAYERS))),
            "semantic_layers": list(getattr(pixart, "semantic_layers", SEMANTIC_LAYERS)),
            "use_tala_train": bool(USE_TALA_TRAIN),
            "tala_active_mode": str(TALA_TRAIN_ACTIVE_MODE),
            "tala_active_t_min": int(TALA_TRAIN_ACTIVE_T_MIN),
            "tala_train_blend_pow": float(TALA_TRAIN_BLEND_POW),
            "tala_max_ratio": float(TALA_TRAIN_MAX_RATIO),
            "tala_train_detach_lr_latent": bool(TALA_TRAIN_DETACH_LR_LATENT),
            "use_adaptive_text_prompt": bool(USE_ADAPTIVE_TEXT_PROMPT),
            "ckpt_select_mode": str(CKPT_SELECT_MODE),
            "ckpt_select_psnr_gate": float(CKPT_SELECT_PSNR_GATE),
            "gate_passed_when_saved": False,
        },
    }
    ok_last, msg_last = atomic_torch_save(state_last, LAST_CKPT_PATH)
    if ok_last:
        print(f"💾 Saved last checkpoint to {LAST_CKPT_PATH} [{msg_last}] (resume-only)")
    else:
        print(f"❌ Failed to save last.pth (resume-only): {msg_last}")


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
def _load_pixart_trainable_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
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

def _build_limited_val_loader(val_loader, num_samples: int):
    if num_samples <= 0:
        return val_loader
    n = min(num_samples, len(val_loader.dataset))
    subset = torch.utils.data.Subset(val_loader.dataset, list(range(n)))
    return DataLoader(subset, batch_size=1, shuffle=False)


def init_from_ckpt_weights_only(pixart, adapter, ckpt_path: str):
    if not ckpt_path:
        return False
    if not os.path.exists(ckpt_path):
        print(f"ℹ️ INIT_CKPT_PATH not found, skip bootstrap: {ckpt_path}")
        return False
    print(f"📦 Bootstrapping model weights from {ckpt_path} (weights-only, no optimizer)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_trainable = ckpt.get("pixart_trainable", ckpt.get("pixart_keep", {}))
    _load_pixart_trainable_subset_compatible(pixart, saved_trainable, context="init")
    adapter_sd = ckpt.get("adapter", None)
    if isinstance(adapter_sd, dict):
        try:
            adapter.load_state_dict(adapter_sd, strict=True)
            print("✅ Adapter bootstrap load succeeded.")
        except Exception as e:
            print(f"⚠️ Adapter bootstrap load skipped due to mismatch: {e}")
    return True


def resume(pixart, adapter, sem_adapter, optimizer, dl_gen, ema=None, ema_named_params=None):
    if not os.path.exists(LAST_CKPT_PATH):
        return 0, 0, [], None, -1.0
    print(f"📥 Resuming from {LAST_CKPT_PATH}...")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    ckpt_role = str(ckpt.get("checkpoint_role", "last"))
    if ckpt_role not in ("last", "best", "best_train"):
        raise RuntimeError(
            f"Checkpoint role '{ckpt_role}' is not resume-capable. "
            "Use last.pth or best.pth for resume."
        )
    saved_trainable = ckpt.get("pixart_trainable", ckpt.get("pixart_keep", {}))
    required_frags = get_required_active_key_fragments_for_model(pixart)
    missing_required = [frag for frag in required_frags if not any(frag in k for k in saved_trainable.keys())]
    if missing_required:
        print("⚠️ Checkpoint missing some currently-trainable fragments (stage-switch expected): " + ", ".join(missing_required))

    adapter_sd = ckpt.get("adapter", {})
    try:
        adapter.load_state_dict(adapter_sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Adapter strict load failed during resume: {e}") from e
    sem_adapter_sd = ckpt.get("sem_adapter", ckpt.get("sem_prompt", None))
    if isinstance(sem_adapter_sd, dict):
        missing, unexpected = sem_adapter.load_state_dict(sem_adapter_sd, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            if is_allowed_sem_adapter_missing(missing, unexpected):
                print("ℹ️ sem_adapter trainable-only checkpoint accepted (missing frozen image_encoder and/or out_scale).")
            else:
                msg = f"sem_adapter resume non-strict load: missing={len(missing)} unexpected={len(unexpected)}"
                if ALLOW_SEM_ADAPTER_NONSTRICT_RESUME:
                    print(f"⚠️ {msg}")
                else:
                    raise RuntimeError(msg)
    else:
        print("ℹ️ sem_adapter state not found in checkpoint; continue with fresh semantic prompt branch.")

    _load_pixart_trainable_subset_compatible(pixart, saved_trainable, context="resume")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"⚠️ Optimizer state restore skipped due to mismatch: {e}")
    rs = ckpt.get("rng_state", None)
    if rs is not None:
        try:
            if rs.get("torch") is not None: torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None: torch.cuda.set_rng_state_all(rs["cuda"])
            if rs.get("numpy") is not None: np.random.set_state(rs["numpy"])
            if rs.get("python") is not None: random.setstate(rs["python"])
        except Exception as e: print(f"⚠️ RNG restore failed (non-fatal): {e}")
    dl_state = ckpt.get("dl_gen_state", None)
    if dl_state is not None:
        try: dl_gen.set_state(dl_state)
        except Exception as e: print(f"⚠️ DataLoader generator restore failed (non-fatal): {e}")
    if ema is not None:
        ema_sd = ckpt.get("ema_state", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            dev_map = {}
            if ema_named_params is not None:
                dev_map = {name: p.device for name, p in ema_named_params}
            restored = {}
            for k, v in ema_sd.items():
                if k not in dev_map:
                    continue
                restored[k] = v.float().to(device=dev_map[k])
            ema.shadow = restored
            print(f"✅ EMA restored: {len(ema.shadow)} tensors")
        else:
            print("ℹ️ EMA state not found in checkpoint; proceeding without EMA restore.")
    last_val_psnr = float(ckpt.get("last_val_psnr", -1.0))
    return ckpt["epoch"]+1, ckpt["step"], ckpt.get("best_records", []), None, last_val_psnr

# ================= 9. Validation =================
@torch.no_grad()
def validate(epoch, pixart, adapter, sem_adapter, vae, val_loader, null_pack, data_info, lpips_fn_val_cpu, val_tag: str = "raw"):
    tag = str(val_tag).lower()
    print(f"🔎 Validating Epoch {epoch+1} [{tag}]...")
    print(
        f"[VAL-CONFIG] use_tala_train={int(USE_TALA_TRAIN)} "
        f"tala_active_mode={TALA_TRAIN_ACTIVE_MODE} "
        f"tala_active_t_min={int(TALA_TRAIN_ACTIVE_T_MIN)} "
        f"tala_max_ratio={float(TALA_TRAIN_MAX_RATIO):.3f} "
        f"tala_train_blend_pow={float(TALA_TRAIN_BLEND_POW):.3f} "
        f"tala_train_detach_lr_latent={bool(TALA_TRAIN_DETACH_LR_LATENT)}"
    )
    pixart.eval(); adapter.eval(); sem_adapter.eval()
    results = {}
    val_gen = torch.Generator(device=DEVICE); val_gen.manual_seed(SEED)
    
    # [V8 Change] Validation uses V-Prediction scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
        clip_sample=False, prediction_type="v_prediction", set_alpha_to_one=False,
    )

    # Validation LPIPS must stay on CPU to reduce GPU memory peak.
    try:
        _lpips_dev = next(lpips_fn_val_cpu.parameters()).device
        if _lpips_dev.type != "cpu":
            raise RuntimeError(f"lpips_fn_val_cpu must be on CPU, got {_lpips_dev}")
    except StopIteration:
        pass
    
    steps_list = [FAST_VAL_STEPS] if FAST_DEV_RUN else VAL_STEPS_LIST
    for steps in steps_list:
        scheduler.set_timesteps(steps, device=DEVICE)
        psnrs, ssims, lpipss = [], [], []; vis_done = False
        flat_psnrs, flat_ssims, flat_lpipss = [], [], []
        edge_psnrs, edge_ssims, edge_lpipss = [], [], []
        corner_psnrs, corner_ssims, corner_lpipss = [], [], []
        cond_deltas = []
        text_cond_deltas = []
        adapter_map_stds = []
        cond_map_stds = []
        sem_tok_stds, sem_out_stds, sem_alphas = [], [], []
        sem_pre_stds, sem_post_stds, sem_out_scales = [], [], []
        tasr_gate_means, sft_alpha_means = [], []
        prompt_hit_rates, prompt_tok_counts, prompt_nonpad_ratios = [], [], []
        prompt_cache_mem = OrderedDict()
        for batch in tqdm(val_loader, desc=f"Val@{steps}"):
            hr = batch["hr"].to(DEVICE); lr = batch["lr"].to(DEVICE)
            sample_keys = batch.get("sample_key", None)
            if isinstance(sample_keys, str):
                sample_keys = [sample_keys]
            elif sample_keys is None:
                fallback_path = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
                sample_keys = [make_sample_key(fallback_path)]
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
            z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
            
            if USE_LQ_INIT: latents, run_timesteps = get_lq_init_latents(z_lr.to(COMPUTE_DTYPE), scheduler, steps, val_gen, LQ_INIT_STRENGTH, COMPUTE_DTYPE)
            else: latents = randn_like_with_generator(z_hr, val_gen); run_timesteps = scheduler.timesteps
            
            lr_small = batch["lr_small"].to(DEVICE, dtype=COMPUTE_DTYPE)
            aug_level = torch.zeros((latents.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE)
            
            for t in run_timesteps:
                t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
                with torch.no_grad():
                    t_embed = pixart.t_embedder(t_b.to(dtype=COMPUTE_DTYPE))
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    cond = adapter(lr_small, t_embed=t_embed)
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    drop_uncond = torch.ones(latents.shape[0], device=DEVICE, dtype=torch.long)
                    drop_cond = torch.zeros(latents.shape[0], device=DEVICE, dtype=torch.long)
                    model_in = latents.to(COMPUTE_DTYPE)
                    cond_zero = mask_adapter_cond(cond, torch.zeros((latents.shape[0],), device=DEVICE))
                    lr_01_for_semantics = ((batch["lr"].to(DEVICE, dtype=COMPUTE_DTYPE)) + 1.0) * 0.5
                    sem_tokens = sem_adapter(lr_01_for_semantics)
                    sem_adapter_stats = getattr(sem_adapter, "_last_sem_adapter_stats", {}) or {}
                    y_cond = null_pack["y"].to(DEVICE).repeat(latents.shape[0], 1, 1, 1)
                    y_mask_cond = null_pack["mask"].to(DEVICE).repeat(latents.shape[0], 1, 1, 1)
                    if USE_ADAPTIVE_TEXT_PROMPT and ADAPTIVE_PROMPT_CACHE_ROOT:
                        prompt_packs, hit_rate, tok_cnt, nonpad_ratio, _num_missing_prompt_packs = load_adaptive_prompt_batch(
                            sample_keys,
                            ADAPTIVE_PROMPT_CACHE_ROOT,
                            prompt_cache_mem,
                            max_cache_items=VAL_PROMPT_CACHE_MAX_ITEMS,
                        )
                        prompt_hit_rates.append(hit_rate)
                        prompt_tok_counts.append(tok_cnt)
                        prompt_nonpad_ratios.append(nonpad_ratio)
                        if not all(p is not None for p in prompt_packs):
                            if STRICT_ADAPTIVE_PROMPT:
                                missing_keys = [str(k) for k, p in zip(sample_keys, prompt_packs) if p is None]
                                raise RuntimeError(f"missing adaptive prompt pack for {missing_keys}")
                        y_cond = torch.cat([p["y"] for p in prompt_packs], dim=0).to(DEVICE)
                        y_mask_cond = torch.cat([p["mask"] for p in prompt_packs], dim=0).to(DEVICE)
                    y_uncond = null_pack["y"].to(DEVICE).repeat(latents.shape[0], 1, 1, 1)
                    y_mask_uncond = null_pack["mask"].to(DEVICE).repeat(latents.shape[0], 1, 1, 1)
                    out_uncond = pixart(x=model_in, timestep=t_b, y=y_uncond, aug_level=aug_level, mask=y_mask_uncond, data_info=data_info, adapter_cond=cond_zero, semantic_tokens=None, force_drop_ids=drop_uncond, sft_strength=get_sft_strength(epoch + 1))
                    out_cond = pixart(x=model_in, timestep=t_b, y=y_cond, aug_level=aug_level, mask=y_mask_cond, data_info=data_info, adapter_cond=cond, semantic_tokens=sem_tokens, force_drop_ids=drop_cond, sft_strength=get_sft_strength(epoch + 1))
                    out_text_null = pixart(x=model_in, timestep=t_b, y=y_uncond, aug_level=aug_level, mask=y_mask_uncond, data_info=data_info, adapter_cond=cond, semantic_tokens=sem_tokens, force_drop_ids=drop_cond, sft_strength=get_sft_strength(epoch + 1))
                    if out_uncond.shape[1] != 4 or out_cond.shape[1] != 4:
                        raise RuntimeError(f"Expected 4-channel CFG outputs, got uncond={out_uncond.shape[1]}, cond={out_cond.shape[1]}")
                    cond_deltas.append(float((out_cond - out_uncond).detach().abs().mean().item()))
                    text_cond_deltas.append(float((out_cond - out_text_null).detach().abs().mean().item()))
                    adapter_map_stds.append(float(cond["cond_map"].detach().float().std().item()))
                    cond_map_stds.append(float(cond["cond_map"].detach().float().std().item()))
                    sem_tok_stds.append(float(sem_tokens.detach().float().std().item()))
                    sem_pre_stds.append(float(sem_adapter_stats.get("sem_tokens_preproj_std", 0.0)))
                    sem_post_stds.append(float(sem_adapter_stats.get("sem_tokens_postproj_std", 0.0)))
                    sem_out_scales.append(float(sem_adapter_stats.get("sem_out_scale", 1.0)))
                    sem_stats = getattr(pixart, "_last_semantic_stats", {}) or {}
                    sem_out_stds.append(float(sem_stats.get("semantic_out_std", 0.0)))
                    sem_alphas.append(float(sem_stats.get("semantic_alpha_value", 0.0)))

                    sft_stats = getattr(pixart, "_last_sft_stats", {}) or {}
                    tasr_gate_means.append(float(sft_stats.get("tasr_gate_mean", 0.0)))
                    sft_alpha_means.append(float(sft_stats.get("alpha_mean", 0.0)))

                    if CFG_SCALE == 1.0:
                        out = out_cond
                    else:
                        out = out_uncond + CFG_SCALE * (out_cond - out_uncond)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample
            pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
            p01 = (pred + 1) / 2; h01 = (hr + 1) / 2
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]; hy = rgb01_to_y01(h01)[..., 4:-4, 4:-4]
            if "psnr" in globals():
                psnrs.append(psnr(py, hy, data_range=1.0).item()); ssims.append(ssim(py, hy, data_range=1.0).item())

            pred_cpu = pred.detach().to("cpu", dtype=torch.float32)
            hr_cpu = hr.detach().to("cpu", dtype=torch.float32)
            lpipss.append(lpips_fn_val_cpu(pred_cpu, hr_cpu).mean().item())
            del pred_cpu, hr_cpu

            m_flat, m_edge, m_corner = build_component_masks_from_hr(hr)
            fp, fs, fl = compute_component_metrics(pred, hr, m_flat, lpips_fn_val_cpu)
            ep, es, el = compute_component_metrics(pred, hr, m_edge, lpips_fn_val_cpu)
            cp, cs, cl = compute_component_metrics(pred, hr, m_corner, lpips_fn_val_cpu)
            flat_psnrs.append(fp); flat_ssims.append(fs); flat_lpipss.append(fl)
            edge_psnrs.append(ep); edge_ssims.append(es); edge_lpipss.append(el)
            corner_psnrs.append(cp); corner_ssims.append(cs); corner_lpipss.append(cl)

            if not vis_done:
                save_path = os.path.join(VIS_DIR, f"epoch{epoch+1:03d}_{tag}_steps{steps}.png")
                lr_np = (lr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                hr_np = (hr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                pr_np = (pred[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                plt.figure(figsize=(18,8))
                plt.subplot(2,3,1); plt.imshow(np.clip(lr_np, 0, 1)); plt.title("Input LR"); plt.axis("off")
                plt.subplot(2,3,2); plt.imshow(np.clip(hr_np, 0, 1)); plt.title("GT"); plt.axis("off")
                plt.subplot(2,3,3); plt.imshow(np.clip(pr_np, 0, 1)); plt.title(f"Pred @{steps}"); plt.axis("off")
                plt.subplot(2,3,4); plt.imshow(m_flat[0,0].detach().cpu(), cmap='gray'); plt.title("M_flat"); plt.axis("off")
                plt.subplot(2,3,5); plt.imshow(m_edge[0,0].detach().cpu(), cmap='gray'); plt.title("M_edge"); plt.axis("off")
                plt.subplot(2,3,6); plt.imshow(m_corner[0,0].detach().cpu(), cmap='gray'); plt.title("M_corner"); plt.axis("off")
                plt.savefig(save_path, bbox_inches="tight"); plt.close(); vis_done = True
            if FAST_DEV_RUN and len(psnrs) >= FAST_VAL_BATCHES: break
        res = (float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss)))
        results[int(steps)] = res
        cdelta = float(np.mean(cond_deltas)) if len(cond_deltas) > 0 else 0.0
        text_cdelta = float(np.mean(text_cond_deltas)) if len(text_cond_deltas) > 0 else 0.0
        prompt_hit_rate = float(np.mean(prompt_hit_rates)) if len(prompt_hit_rates) > 0 else 0.0
        avg_prompt_token_count = float(np.mean(prompt_tok_counts)) if len(prompt_tok_counts) > 0 else 0.0
        avg_prompt_nonpad_ratio = float(np.mean(prompt_nonpad_ratios)) if len(prompt_nonpad_ratios) > 0 else 0.0
        adapter_map_std = float(np.mean(adapter_map_stds)) if len(adapter_map_stds) > 0 else 0.0
        cond_map_std = float(np.mean(cond_map_stds)) if len(cond_map_stds) > 0 else 0.0
        sem_tok_std = float(np.mean(sem_tok_stds)) if len(sem_tok_stds) > 0 else 0.0
        sem_pre_std = float(np.mean(sem_pre_stds)) if len(sem_pre_stds) > 0 else 0.0
        sem_post_std = float(np.mean(sem_post_stds)) if len(sem_post_stds) > 0 else 0.0
        sem_out_scale = float(np.mean(sem_out_scales)) if len(sem_out_scales) > 0 else 1.0
        sem_out_std = float(np.mean(sem_out_stds)) if len(sem_out_stds) > 0 else 0.0
        sem_alpha = float(np.mean(sem_alphas)) if len(sem_alphas) > 0 else 0.0
        tasr_gate_mean_val = float(np.mean(tasr_gate_means)) if len(tasr_gate_means) > 0 else 0.0
        sft_alpha_mean = float(np.mean(sft_alpha_means)) if len(sft_alpha_means) > 0 else 0.0
        msg = (
            f"[VAL-PROXY@{steps}][{tag}] Ep{epoch+1}: proxy_psnr={res[0]:.2f} | proxy_ssim={res[1]:.4f} | proxy_lpips={res[2]:.4f} | "
            f"CONDΔ={cdelta:.5f} | text_cond_delta={text_cdelta:.5f} | prompt_cache_hit_rate={prompt_hit_rate:.3f} | avg_prompt_token_count={avg_prompt_token_count:.2f} | avg_prompt_nonpad_ratio={avg_prompt_nonpad_ratio:.3f} | ad_map_std={adapter_map_std:.4f} | cond_map_std={cond_map_std:.4f} | sem_tok_std={sem_tok_std:.4f} | sem_pre_std={sem_pre_std:.4f} | sem_post_std={sem_post_std:.4f} | sem_out_scale={sem_out_scale:.4f} | sem_out_std={sem_out_std:.4f} | sem_alpha={sem_alpha:.4f} | sem_K={16} | "
            f"tasr_gate_mean={tasr_gate_mean_val:.4f} | sft_alpha_mean={sft_alpha_mean:.4f} | "
            f"flat_lpips={np.mean(flat_lpipss):.4f} | edge_lpips={np.mean(edge_lpipss):.4f} | "
            f"corner_psnr={np.mean(corner_psnrs):.2f} | corner_lpips={np.mean(corner_lpipss):.4f}"
        )
        print(msg)
    pixart.train(); adapter.train(); sem_adapter.train()
    return results

# ================= 10. Main =================
def main():
    global LAST_TRAIN_LOG
    seed_everything(SEED); dl_gen = torch.Generator(); dl_gen.manual_seed(SEED)
    validate_schedule_alignment()
    validate_s2d_decoupling()
    print(f"[CUDA Allocator] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    required_paths = [PIXART_PATH, VAE_PATH, NULL_T5_EMBED_PATH]
    for pth in required_paths:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required pretrained path missing: {pth}")

    train_ds = DF2K_Online_Dataset(crop_size=512, is_train=True, scale=4)
    if VAL_MODE == "realsr":
        val_ds = RealSR_Val_Paired_Dataset(REALSR_VAL_ROOTS, crop_size=512)
        print(f"[VAL] mode=realsr roots={REALSR_VAL_ROOTS} pairs={len(val_ds)}")
    elif VAL_MODE == "valpack":
        val_ds = ValPackDataset(VAL_PACK_DIR, lr_dir_name=VAL_PACK_LR_DIR_NAME, crop_size=512)
        print(f"[VAL] mode=valpack path={VAL_PACK_DIR}/{VAL_PACK_LR_DIR_NAME}")
    elif VAL_MODE == "train_like":
        val_ds = DF2K_Val_Degraded_Dataset(VAL_HR_DIR, crop_size=512, seed=SEED, deg_mode=TRAIN_DEG_MODE)
        print(f"[VAL] mode=train_like deg_mode={TRAIN_DEG_MODE}")
    elif VAL_MODE == "lr_dir" and VAL_LR_DIR is not None:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=VAL_LR_DIR, crop_size=512)
        print(f"[VAL] mode=lr_dir lr_root={VAL_LR_DIR}")
    else:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=None, crop_size=512)
        print("[VAL] mode=fallback_bicubic_from_hr (no paired VAL_LR_DIR found)")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
        persistent_workers=False,
        prefetch_factor=1 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        persistent_workers=False,
        prefetch_factor=1 if VAL_NUM_WORKERS > 0 else None,
    )
    if USE_ADAPTIVE_TEXT_PROMPT:
        if (not ADAPTIVE_PROMPT_CACHE_ROOT) or (not os.path.isdir(ADAPTIVE_PROMPT_CACHE_ROOT)):
            raise FileNotFoundError(
                "Adaptive prompt cache root missing. Please run tools/prepare_adaptive_text_cache.py first."
            )
        cache_files = sorted(glob.glob(os.path.join(ADAPTIVE_PROMPT_CACHE_ROOT, "**", "*.pth"), recursive=True))
        if len(cache_files) == 0:
            raise RuntimeError(
                "Adaptive prompt cache root has no .pth files. Please run tools/prepare_adaptive_text_cache.py first."
            )
        if hasattr(train_ds, "pairs") and len(train_ds.pairs) > 0:
            ex_lr = train_ds.pairs[0][0]
            ex_key = make_sample_key(ex_lr)
            ex_exists = os.path.exists(os.path.join(ADAPTIVE_PROMPT_CACHE_ROOT, f"{ex_key}.pth"))
        else:
            ex_key = "N/A"
            ex_exists = False
        print(f"[adaptive-prompt] root = {ADAPTIVE_PROMPT_CACHE_ROOT}")
        print(f"[adaptive-prompt] cache files found = {len(cache_files)}")
        print(f"[adaptive-prompt] example sample key = {ex_key}")
        print(f"[adaptive-prompt] example cache exists = {ex_exists}")

    kv_cfg = {
        "sampling": None,
        "scale_factor": int(KV_COMPRESS_SCALE),
        "kv_compress_layer": list(KV_COMPRESS_LAYERS),
    } if KV_COMPRESS_ENABLE else None

    pixart = PixArtSigmaSR_XL_2(
        input_size=64, in_channels=4, out_channels=4,
        force_null_caption=False,
        anchor_layers=list(ANCHOR_LAYERS),
        semantic_layers=list(SEMANTIC_LAYERS),
        kv_compress_config=kv_cfg,
    ).to(DEVICE)
    set_grad_checkpoint(pixart, use_fp32_attention=False, gc_step=1)
    if KV_COMPRESS_ENABLE:
        print(f"[KV-Compress] enabled scale={KV_COMPRESS_SCALE} layers={KV_COMPRESS_LAYERS}")
    else:
        print("[KV-Compress] disabled")
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base: base = base["state_dict"]
    if "pos_embed" in base: del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        missing, unexpected, skipped = load_state_dict_shape_compatible(pixart, base, context="base-pretrain")
        print(f"[Load] missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
    if ENABLE_LORA:
        apply_lora(pixart)
    else:
        print("ℹ️ LoRA disabled.")
    pixart.train()

    adapter = build_adapter_v12(
        in_channels=3,
        hidden_size=1152,
    ).to(DEVICE).train()
    sem_adapter = CLIPSemanticAdapter(
        encoder_name_or_path=SEMANTIC_ENCODER_NAME_OR_PATH,
        hidden_size=1152,
        num_prompt_tokens=16,
        clip_input_res=224,
    ).to(DEVICE).train()
    sem_adapter.image_encoder.eval()
    for p in sem_adapter.image_encoder.parameters():
        p.requires_grad_(False)
    save_plan_keys = compute_save_keys_for_stages(pixart, train_x_embedder=TRAIN_PIXART_X_EMBEDDER)
    ever_keys = set(save_plan_keys)
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()
    if VAE_TILING and hasattr(vae, "enable_tiling"): vae.enable_tiling()

    lpips_fn_val_cpu = lpips.LPIPS(net='vgg').to("cpu").eval()
    lpips_fn_train = lpips.LPIPS(net='vgg').to(DEVICE).eval() if USE_LPIPS_PERCEP else None
    for p in vae.parameters(): p.requires_grad_(False)
    for p in lpips_fn_val_cpu.parameters(): p.requires_grad_(False)
    if lpips_fn_train is not None:
        for p in lpips_fn_train.parameters(): p.requires_grad_(False)

    if not os.path.exists(NULL_T5_EMBED_PATH):
        raise FileNotFoundError(f"Null T5 embed not found: {NULL_T5_EMBED_PATH}")
    null_pack = load_prompt_pack(NULL_T5_EMBED_PATH, pack_name="null")
    y_null = null_pack["y"].to(DEVICE)
    mask_null = null_pack["mask"].to(DEVICE)
    if y_null.ndim != 4:
        raise RuntimeError(f"Invalid y shape from offline null T5 embed: {tuple(y_null.shape)} (expected [1,1,L,C])")
    print(f"[null-pack] loaded from: {NULL_T5_EMBED_PATH}")
    print(f"[null-pack] y shape: {tuple(y_null.shape)}")
    print(f"[null-pack] mask shape: {tuple(mask_null.shape)}")
    print(f"[null-pack] legacy converted: {bool(null_pack.get('_legacy_converted', False))}")
    if USE_ADAPTIVE_TEXT_PROMPT:
        print(f"[adaptive-prompt] enabled root={ADAPTIVE_PROMPT_CACHE_ROOT}")
    else:
        print("[adaptive-prompt] disabled")
    prompt_cache_mem = None
    d_info = {"img_hw": torch.tensor([[512.,512.]]).to(DEVICE), "aspect_ratio": torch.tensor([1.]).to(DEVICE)}

    # Single-stage optimizer is built once.

    # [V8 Change] Switch to V-Prediction logic manually in loop (since IDDPM is epsilon based)
    # We will manually calculate v_target and loss.
    # Note: IDDPM class is kept for schedule utils, but we bypass its loss function.
    diffusion = IDDPM(str(1000))
    assert_strict_tala_configuration()

    ema = None
    ema_named_params = []
    if USE_EMA:
        ema = ParamEMA(decay=EMA_DECAY)
        ema_named_params = collect_ema_named_params(pixart, adapter, mode=EMA_TRACK_SET)
        ema.register(ema_named_params)
        print(f"✅ EMA tracking enabled: decay={EMA_DECAY}, track_set={EMA_TRACK_SET}, tensors={len(ema_named_params)}")

    print(f"📏 Validation steps: {VAL_STEPS_LIST}")
    print(
        f"[RESP-SUMMARY] anchor_layers={list(ANCHOR_LAYERS)} semantic_layers={list(SEMANTIC_LAYERS)} "
        f"USE_ADAPTIVE_TEXT_PROMPT={bool(USE_ADAPTIVE_TEXT_PROMPT)} USE_SEMANTIC_BRANCH={bool(USE_SEMANTIC_BRANCH)} "
        f"USE_TALA_TRAIN={bool(USE_TALA_TRAIN)} USE_LQ_INIT={bool(USE_LQ_INIT)} "
        f"LORA_RANK={int(LORA_RANK)} LORA_ALPHA={int(LORA_ALPHA)}"
    )
    print('[GATE-MODE] gate_mode = "pure_tasr_learned" | no_handcrafted_time_prior = True')
    print('[GATE-SOURCE] gate_time_source = "pure_timestep_only"')
    print('[SEMANTIC-SCHEDULE] semantic_time_schedule = "disabled"')
    print('[HPA-MODE] hpa_mode = "head_space_residual_injection"')
    print('[PACKED-FALLBACK] packed_text_fallback = "per_sample_safe"')
    print(f"[LPIPS-PATH] use_legacy_patch_lpips = {bool(USE_LEGACY_PATCH_LPIPS)}")
    print(
        f"[PROMPT-STRICT] USE_ADAPTIVE_TEXT_PROMPT={bool(USE_ADAPTIVE_TEXT_PROMPT)} "
        f"STRICT_ADAPTIVE_PROMPT={bool(STRICT_ADAPTIVE_PROMPT)}"
    )
    print(
        f"[TALA-STRICT] USE_LQ_INIT={bool(USE_LQ_INIT)} LQ_INIT_STRENGTH={float(LQ_INIT_STRENGTH):.3f} "
        f"TALA_TRAIN_HIGH_T_MODE={bool(TALA_TRAIN_HIGH_T_MODE)} TALA_TRAIN_ACTIVE_T_MIN={int(TALA_TRAIN_ACTIVE_T_MIN)} "
        f"TALA_TRAIN_MAX_RATIO={float(TALA_TRAIN_MAX_RATIO):.3f}"
    )
    print(f"[RUN-MODE] anchor_layers={list(ANCHOR_LAYERS)} use_tala_train={bool(USE_TALA_TRAIN)} use_hpa={bool(USE_SEMANTIC_BRANCH)} lora_rank={int(LORA_RANK)} lora_alpha={int(LORA_ALPHA)}")
    print(
        f"[PERCEP-CONFIG] USE_TIMESTEP_AWARE_PERCEPTUAL={bool(USE_TIMESTEP_AWARE_PERCEPTUAL)} "
        f"PERCEP_T_MAX={int(PERCEP_T_MAX)} PERCEP_DECODE_MAX_SAMPLES={int(PERCEP_DECODE_MAX_SAMPLES)} "
        f"LPIPS_LATE_WEIGHT={float(LPIPS_LATE_WEIGHT):.3f}"
    )
    if USE_ADAPTIVE_TEXT_PROMPT:
        print("[PROMPT-SOURCE] conditional prompt source = adaptive cache")
    else:
        print("[PROMPT-SOURCE] conditional prompt source = null prompt")
    print("[PROMPT-SOURCE] unconditional prompt source = null prompt")
    print(f"[LORA] rank={int(LORA_RANK)} alpha={int(LORA_ALPHA)}")
    print(f"📏 EMA tracking: {'enabled' if USE_EMA else 'disabled'}")
    print(f"📏 EMA validation frequency: every {EMA_VALIDATE_EVERY} validations")
    print("📏 Raw validation frequency: every validation trigger")
    print("[VAL-SCHEDULE] epochs 1-14: every 5 | epochs 15-24: every 2 | epochs 25+: every epoch")
    print(f"[PROMPT-CACHE] train_max_items={TRAIN_PROMPT_CACHE_MAX_ITEMS} val_max_items={VAL_PROMPT_CACHE_MAX_ITEMS}")
    print(f"[DATALOADER] train_num_workers={NUM_WORKERS} val_num_workers={VAL_NUM_WORKERS}")
    print(f"[GUARD-FWD] extra_guard_forward_every={GUARD_FORWARD_EVERY}")

    if START_FROM_BASE_ONLY:
        print("✅ START_FROM_BASE_ONLY=True: skip bootstrap/resume checkpoints; train from PixArt base.")
    elif not os.path.exists(LAST_CKPT_PATH) and INIT_CKPT_PATH:
        init_from_ckpt_weights_only(pixart, adapter, INIT_CKPT_PATH)

    configure_pixart_trainable_params(pixart, train_x_embedder=TRAIN_PIXART_X_EMBEDDER)
    for p in adapter.parameters():
        p.requires_grad_(True)
    for n, p in sem_adapter.named_parameters():
        if ("resampler" in n) or ("proj_norm" in n) or ("proj." in n):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    ever_keys.update({n for n, p in pixart.named_parameters() if p.requires_grad})
    validation_count = 0
    optimizer, params_to_clip, group_counts = build_optimizer_and_clippables(pixart, adapter, sem_adapter)
    print(f"✅ Optim groups(single-stage): {group_counts}")
    _maybe_empty_cuda_cache()

    if START_FROM_BASE_ONLY:
        ep_start, step, best, last_val_psnr = 0, 0, [], -1.0
    else:
        ep_start, step, best, _, last_val_psnr = resume(
            pixart, adapter, sem_adapter, optimizer, dl_gen, ema=ema, ema_named_params=ema_named_params
        )
    if ema is not None:
        ema_named_params = collect_ema_named_params(pixart, adapter, mode=EMA_TRACK_SET)
        ema.register(ema_named_params)

    print("🚀 DiT-SR V8 Training Started (V-Pred, Aug, Copy-Init).")
    max_steps = MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else (FAST_TRAIN_STEPS if FAST_DEV_RUN else None)
    for epoch in range(ep_start, 1000):
        if max_steps is not None and step >= max_steps: break
        train_ds.set_epoch(epoch)
        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Ep{epoch+1}")
        accum_micro_steps = 0
        last_cond_delta_curr = float("nan")
        last_text_cond_delta_curr = float("nan")
        reached_max_steps = False
        for i, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                reached_max_steps = True
                break

            hr = batch['hr'].to(DEVICE); lr = batch['lr'].to(DEVICE); lr_small_b = batch['lr_small']
            sample_keys = batch.get("sample_key", None)
            if isinstance(sample_keys, str):
                sample_keys = [sample_keys]
            elif sample_keys is None:
                fallback_path = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
                sample_keys = [make_sample_key(fallback_path)]
            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
                zl = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

            # [V8 Logic] V-Prediction Training
            t = sample_t(zh.shape[0], DEVICE, step)
            noise = torch.randn_like(zh)
            if USE_TALA_TRAIN:
                zt_in, target_v, alpha_t, sigma_t, tala_ratio, tala_stats = build_tala_train_latent_and_target(
                    zh=zh,
                    zl=zl,
                    timesteps=t,
                    noise=noise,
                    diffusion=diffusion,
                    lq_init_strength=LQ_INIT_STRENGTH,
                    high_t_mode=TALA_TRAIN_HIGH_T_MODE,
                    active_t_min=TALA_TRAIN_ACTIVE_T_MIN,
                    blend_pow=TALA_TRAIN_BLEND_POW,
                    max_ratio=TALA_TRAIN_MAX_RATIO,
                    detach_lr_latent=TALA_TRAIN_DETACH_LR_LATENT,
                    use_tala=True,
                )
            else:
                zt_in = diffusion.q_sample(zh, t, noise)
                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()
                tala_ratio = torch.zeros((zh.shape[0],), device=zh.device, dtype=zh.dtype)
                tala_stats = {
                    "tala_mode": "disabled",
                    "tala_t_min": 0,
                    "tala_t_max": 0,
                    "tala_effective_eps_std": float(noise.detach().float().std().item()),
                }
            
            aug_level_emb = torch.zeros((zh.shape[0],), device=DEVICE, dtype=torch.float32)

            adapter_in = lr_small_b.to(DEVICE)
            adapter_in = adapter_in.to(dtype=COMPUTE_DTYPE)
            with torch.no_grad():
                t_embed = pixart.t_embedder(t.to(dtype=COMPUTE_DTYPE))
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                cond = adapter(adapter_in, t_embed=t_embed) # time-aware adapter conditioning
            cond_in = cond
            cond_drop_prob = float(COND_DROP_PROB)
            if USE_ADAPTER_CFDROPOUT and cond_drop_prob > 0:
                keep = (torch.rand((zt_in.shape[0],), device=DEVICE) >= cond_drop_prob).float()
                cond_in = mask_adapter_cond(cond, keep)

            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                # semantic adapter is the only late semantic/detail guidance
                lr_01_for_semantics = (lr.to(COMPUTE_DTYPE) + 1.0) * 0.5
                sem_tokens = sem_adapter(lr_01_for_semantics)
                sem_adapter_stats = getattr(sem_adapter, "_last_sem_adapter_stats", {}) or {}
                # conditional prompt source = adaptive cache (if enabled); unconditional prompt source = null prompt.
                y_cond = y_null.repeat(zt_in.shape[0], 1, 1, 1)
                y_mask = mask_null.repeat(zt_in.shape[0], 1, 1, 1)
                prompt_cache_hit_rate = 0.0
                num_missing_prompt_packs = 0
                avg_prompt_token_count = 0.0
                avg_prompt_nonpad_ratio = 0.0
                if USE_ADAPTIVE_TEXT_PROMPT and ADAPTIVE_PROMPT_CACHE_ROOT:
                    prompt_packs, prompt_cache_hit_rate, avg_prompt_token_count, avg_prompt_nonpad_ratio, num_missing_prompt_packs = load_adaptive_prompt_batch(
                        sample_keys,
                        ADAPTIVE_PROMPT_CACHE_ROOT,
                        prompt_cache_mem,
                        max_cache_items=TRAIN_PROMPT_CACHE_MAX_ITEMS,
                    )
                    if not all(p is not None for p in prompt_packs):
                        if STRICT_ADAPTIVE_PROMPT:
                            missing_keys = [str(k) for k, p in zip(sample_keys, prompt_packs) if p is None]
                            raise RuntimeError(f"missing adaptive prompt pack for {missing_keys}")
                    else:
                        y_cond = torch.cat([p["y"] for p in prompt_packs], dim=0).to(DEVICE)
                        y_mask = torch.cat([p["mask"] for p in prompt_packs], dim=0).to(DEVICE)
                drop_cond = torch.zeros(zt_in.shape[0], device=DEVICE, dtype=torch.long)
                sft_strength = get_sft_strength(epoch + 1)
                kwargs = dict(
                    x=zt_in,
                    timestep=t,
                    y=y_cond,
                    aug_level=aug_level_emb,
                    mask=y_mask,
                    data_info=d_info,
                    adapter_cond=cond_in,
                    semantic_tokens=sem_tokens,
                    sft_strength=sft_strength,
                )
                kwargs["force_drop_ids"] = drop_cond

                out = pixart(**kwargs)
                if out.shape[1] != 4:
                    raise RuntimeError(f"Expected 4-channel model output, got {out.shape[1]} channels")
                model_pred = out.float()
                sem_stats = getattr(pixart, "_last_semantic_stats", {}) or {}
                tala_ratio_mean = float(tala_ratio.detach().float().mean().item())
                tala_ratio_max = float(tala_ratio.detach().float().max().item())
                tala_active_frac = float((tala_ratio.detach().float() > 0).float().mean().item())
                guard_stats = {
                    "sem_tok_std": float(sem_tokens.detach().float().std().item()),
                    "sem_pre_std": float(sem_adapter_stats.get("sem_tokens_preproj_std", 0.0)),
                    "sem_post_std": float(sem_adapter_stats.get("sem_tokens_postproj_std", 0.0)),
                    "sem_out_scale": float(sem_adapter_stats.get("sem_out_scale", 1.0)),
                    "semantic_out_std": float(sem_stats.get("semantic_out_std", 0.0)),
                    "semantic_alpha_value": float(sem_stats.get("semantic_alpha_value", 0.0)),
                    "hpa_text_ctx_std": float(sem_stats.get("hpa_text_ctx_std", 0.0)),
                    "hpa_img_delta_std": float(sem_stats.get("hpa_img_delta_std", 0.0)),
                    "semantic_block_ids_active": str(list(sem_stats.get("semantic_block_ids_active", []))),
                    "ad_map_std": float(cond_in["cond_map"].detach().float().std().item()),
                    "cond_map_std": float(cond_in["cond_map"].detach().float().std().item()),
                    "sft_strength": float(sft_strength),
                    "t_min": int(t.min().item()),
                    "t_max": int(t.max().item()),
                    "t_mean": float(t.float().mean().item()),
                    "tala_on": int(USE_TALA_TRAIN),
                    "tala_ratio_mean": float(tala_ratio_mean),
                    "tala_ratio_max": float(tala_ratio_max),
                    "tala_active_frac": float(tala_active_frac),
                    "tala_t_min_batch": int(t.min().item()),
                    "tala_t_max_batch": int(t.max().item()),
                    "tala_mode": str(tala_stats.get("tala_mode", "unknown")),
                    "tala_t_min": int(tala_stats.get("tala_t_min", 0)),
                    "tala_t_max": int(tala_stats.get("tala_t_max", 0)),
                    "tala_effective_eps_std": float(tala_stats.get("tala_effective_eps_std", 0.0)),
                }
                assert_finite_tensor("sem_tokens", sem_tokens, guard_stats)
                assert_finite_tensor("cond_map", cond_in["cond_map"], guard_stats)
                assert_finite_tensor("model_pred", model_pred, guard_stats)
                run_guard_forward = ((i % GUARD_FORWARD_EVERY) == 0) or ((step % GUARD_FORWARD_EVERY) == 0)
                if run_guard_forward:
                    drop_uncond_guard = torch.ones(zt_in.shape[0], device=DEVICE, dtype=torch.long)
                    cond_zero_guard = mask_adapter_cond(cond_in, torch.zeros((zt_in.shape[0],), device=DEVICE))
                    y_uncond_guard = null_pack["y"].to(DEVICE).repeat(zt_in.shape[0], 1, 1, 1)
                    y_mask_uncond_guard = null_pack["mask"].to(DEVICE).repeat(zt_in.shape[0], 1, 1, 1)
                    with torch.no_grad():
                        out_uncond_guard = pixart(
                            x=zt_in.detach(),
                            timestep=t,
                            y=y_uncond_guard,
                            aug_level=aug_level_emb,
                            mask=y_mask_uncond_guard,
                            data_info=d_info,
                            adapter_cond=cond_zero_guard,
                            semantic_tokens=None,
                            force_drop_ids=drop_uncond_guard,
                            sft_strength=sft_strength,
                        ).float()
                        out_text_null_guard = pixart(
                            x=zt_in.detach(),
                            timestep=t,
                            y=y_uncond_guard,
                            aug_level=aug_level_emb,
                            mask=y_mask_uncond_guard,
                            data_info=d_info,
                            adapter_cond=cond_in,
                            semantic_tokens=sem_tokens.detach(),
                            force_drop_ids=drop_cond,
                            sft_strength=sft_strength,
                        ).float()
                    assert_finite_tensor("out_uncond", out_uncond_guard, guard_stats)
                    cond_delta_curr = float((model_pred.detach() - out_uncond_guard).abs().mean().item())
                    text_cond_delta_curr = float((model_pred.detach() - out_text_null_guard).abs().mean().item())
                    last_cond_delta_curr = cond_delta_curr
                    last_text_cond_delta_curr = text_cond_delta_curr
                else:
                    cond_delta_curr = float(last_cond_delta_curr)
                    text_cond_delta_curr = float(last_text_cond_delta_curr)

                # [V8 Logic] Min-SNR-Gamma Weighting (per-sample, shape [B])
                alpha_s = alpha_t[:, 0, 0, 0]
                sigma_s = sigma_t[:, 0, 0, 0]
                snr = (alpha_s ** 2) / (sigma_s ** 2)
                gamma = 5.0
                min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
                loss_weights = min_snr_gamma / snr
                loss_v = (F.mse_loss(model_pred, target_v, reduction='none').mean(dim=[1, 2, 3]) * loss_weights).mean()

                # Reconstruct x0 for other losses (x0 = alpha * zt_in - sigma * v)
                z0 = alpha_t * zt_in.float() - sigma_t * model_pred

                latent_l1_per = torch.mean(torch.abs(z0 - zh.float()), dim=[1, 2, 3])
                latent_l1_mask = (t >= int(LATENT_L1_T_MIN)).float()
                if float(latent_l1_mask.sum().item()) > 0:
                    loss_latent_l1 = (latent_l1_per * latent_l1_mask).sum() / latent_l1_mask.sum().clamp_min(1.0)
                else:
                    loss_latent_l1 = torch.zeros((), device=DEVICE, dtype=z0.dtype)

                w = get_loss_weights(epoch + 1)
                
                # Calculate patch-space losses
                loss_lr_cons = torch.tensor(0.0, device=DEVICE)
                loss_gw = torch.tensor(0.0, device=DEVICE)
                loss_lpips = torch.tensor(0.0, device=DEVICE)
                lpips_num_samples = 0

                need_patch_loss = (
                    (w.get('lr_cons', 0.0) > 0)
                    or (w.get('gw', 0.0) > 0)
                    or (USE_LEGACY_PATCH_LPIPS and (w.get('lpips', 0.0) > 0))
                )

                patch_t_mask = (t >= int(PIXEL_LOSS_T_MIN))
                allow_by_stage = True
                patch_loss_num_samples = int(patch_t_mask.sum().item()) if allow_by_stage else 0
                calc_patch_loss = need_patch_loss and allow_by_stage and (patch_loss_num_samples > 0)

                if calc_patch_loss:
                    active_idx = torch.nonzero(patch_t_mask, as_tuple=False).squeeze(1)
                    z0_sel = z0.index_select(0, active_idx)
                    img_p_valid = _decode_vae_sample_checkpointed(vae, z0_sel / vae.config.scaling_factor).clamp(-1, 1)
                    img_t_valid = hr.index_select(0, active_idx).clamp(-1, 1)
                    lr_patch = lr.index_select(0, active_idx).clamp(-1, 1)

                    if w.get('lr_cons', 0.0) > 0:
                        loss_lr_cons = structure_consistency_loss(img_p_valid, lr_patch)

                    if w.get('gw', 0.0) > 0:
                        loss_gw = gradient_weighted_loss(img_p_valid, img_t_valid, alpha=GW_ALPHA)

                    if USE_LEGACY_PATCH_LPIPS and (w.get('lpips', 0.0) > 0) and USE_LPIPS_PERCEP:
                        lpips_t_mask = (t.index_select(0, active_idx) <= int(LPIPS_T_MAX))
                        if float(lpips_t_mask.sum().item()) > 0:
                            lp_idx = torch.nonzero(lpips_t_mask, as_tuple=False).squeeze(1)
                            lpips_num_samples = int(lp_idx.numel())
                            loss_lpips = perceptual_lpips_loss(
                                lpips_fn_train,
                                img_p_valid.index_select(0, lp_idx),
                                img_t_valid.index_select(0, lp_idx),
                            )
                else:
                    patch_loss_num_samples = 0

                late_lpips = torch.tensor(0.0, device=DEVICE)
                late_lpips_num_samples = 0
                late_lpips_active_frac = float((t <= int(PERCEP_T_MAX)).float().mean().item())
                if USE_TIMESTEP_AWARE_PERCEPTUAL and USE_LPIPS_PERCEP and (lpips_fn_train is not None):
                    late_mask = (t <= int(PERCEP_T_MAX))
                    if bool(late_mask.any().item()):
                        late_indices = torch.nonzero(late_mask, as_tuple=False).squeeze(1)
                        if late_indices.numel() > int(PERCEP_DECODE_MAX_SAMPLES):
                            late_indices = late_indices[: int(PERCEP_DECODE_MAX_SAMPLES)]
                        z0_sel = z0.index_select(0, late_indices)
                        hr_sel = hr.index_select(0, late_indices).clamp(-1, 1)
                        pred_img = _decode_vae_sample_checkpointed(vae, z0_sel / vae.config.scaling_factor).clamp(-1, 1)
                        late_lpips = perceptual_lpips_loss(lpips_fn_train, pred_img, hr_sel)
                        late_lpips_num_samples = int(late_indices.numel())

                inject_reg_w = get_inject_reg_weight(epoch + 1)
                inject_reg = compute_injection_scale_reg(pixart, inject_reg_lambda(step) * inject_reg_w)
                loss = (
                    loss_v
                    + w['latent_l1'] * loss_latent_l1
                    + w.get('lr_cons', 0.0) * loss_lr_cons
                    + w.get('gw', 0.0) * loss_gw
                    + (w.get('lpips', 0.0) * loss_lpips if USE_LEGACY_PATCH_LPIPS else 0.0)
                    + float(LPIPS_LATE_WEIGHT) * late_lpips
                    + inject_reg
                ) / GRAD_ACCUM_STEPS
                assert_finite_tensor("loss", loss, guard_stats)

            loss.backward()
            accum_micro_steps += 1

            if accum_micro_steps == GRAD_ACCUM_STEPS:
                log_critical_path_gradients(step + 1, pixart, adapter)
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if ema is not None:
                    ema.update(ema_named_params)
                accum_micro_steps = 0

            if i % 10 == 0:
                sft_stats = getattr(pixart, "_last_sft_stats", {}) or {}
                tasr_gate_mean = float(sft_stats.get("tasr_gate_mean", 0.0))
                tasr_gate_min = float(sft_stats.get("tasr_gate_min", 0.0))
                tasr_gate_max = float(sft_stats.get("tasr_gate_max", 0.0))
                sft_delta_std = float(sft_stats.get("sft_delta_std", 0.0))
                alpha_mean_sft = float(sft_stats.get("alpha_mean", 0.0))
                alpha_min = float(sft_stats.get("alpha_min", 0.0))
                alpha_max = float(sft_stats.get("alpha_max", 0.0))
                sft_strength_logged = float(sft_stats.get("sft_strength", sft_strength))
                sem_stats = getattr(pixart, "_last_semantic_stats", {}) or {}
                sem_adapter_stats = getattr(sem_adapter, "_last_sem_adapter_stats", {}) or {}
                sem_tok_std = float(sem_tokens.detach().float().std().item())
                sem_pre_std = float(sem_adapter_stats.get("sem_tokens_preproj_std", 0.0))
                sem_post_std = float(sem_adapter_stats.get("sem_tokens_postproj_std", 0.0))
                sem_out_scale = float(sem_adapter_stats.get("sem_out_scale", 1.0))
                sem_out_std = float(sem_stats.get("semantic_out_std", 0.0))
                sem_alpha = float(sem_stats.get("semantic_alpha_value", 0.0))
                hpa_text_ctx_std = float(sem_stats.get("hpa_text_ctx_std", 0.0))
                hpa_img_delta_std = float(sem_stats.get("hpa_img_delta_std", 0.0))
                adapter_map_std = float(cond_in["cond_map"].detach().float().std().item())
                cond_map_std = float(cond_in["cond_map"].detach().float().std().item())
                cond_delta = float(cond_delta_curr)
                text_cond_delta = float(text_cond_delta_curr)
                tala_on = int(USE_TALA_TRAIN)
                cpu_rss_gb = float(get_rss_gb()) if (step % 100 == 0) else float("nan")
                if step % 100 == 0:
                    print(f"[MEM] step={step} cpu_rss_gb={cpu_rss_gb:.3f}")
                log_dict = {
                    'v_loss': f"{loss_v:.3f}",
                    'lat_l1': f"{loss_latent_l1:.3f}",
                    'l1_tmin': f"{LATENT_L1_T_MIN}",
                    'gw': f"{loss_gw.item():.3f}",
                    'lp': f"{loss_lpips.item():.3f}",
                    'lp_n': f"{lpips_num_samples}",
                    'w_lat': f"{w.get('latent_l1', 0.0):.3f}",
                    'w_gw': f"{w.get('gw', 0.0):.3f}",
                    'ireg': f"{inject_reg.item():.4f}",
                    'ireg_w': f"{inject_reg_w:.3f}",
                    'lr_cons': f"{loss_lr_cons.item():.3f}",
                    'px_n': f"{patch_loss_num_samples}",
                    'w_lr': f"{w.get('lr_cons', 0.0):.3f}",
                    'lp_w': f"{w.get('lpips', 0.0):.3f}",
                    'lp_legacy': f"{int(USE_LEGACY_PATCH_LPIPS)}",
                    'late_lp': f"{late_lpips.item():.3f}",
                    'late_n': f"{late_lpips_num_samples}",
                    'late_af': f"{late_lpips_active_frac:.3f}",
                    'sft_s': f"{sft_strength_logged:.3f}",
                    'proxy_psnr': f"{last_val_psnr:.2f}",
                    'tasr_gate': f"{tasr_gate_mean:.3f}[{tasr_gate_min:.3f},{tasr_gate_max:.3f}]",
                    'sft_dstd': f"{sft_delta_std:.3f}",
                    'sft_a': f"{alpha_mean_sft:.3f}[{alpha_min:.3f},{alpha_max:.3f}]",
                    'sem_tok_std': f"{sem_tok_std:.4f}",
                    'sem_pre_std': f"{sem_pre_std:.4f}",
                    'sem_post_std': f"{sem_post_std:.4f}",
                    'sem_out_scale': f"{sem_out_scale:.4f}",
                    'sem_out_std': f"{sem_out_std:.4f}",
                    'sem_alpha': f"{sem_alpha:.4f}",
                    'hpa_tctx': f"{hpa_text_ctx_std:.4f}",
                    'hpa_idelta': f"{hpa_img_delta_std:.4f}",
                    'cond_delta': f"{cond_delta:.5f}",
                    'text_cond_delta': f"{text_cond_delta:.5f}",
                    'guard_fwd': f"{int(run_guard_forward)}",
                    'a_hit': f"{prompt_cache_hit_rate:.3f}",
                    'p_miss': f"{int(num_missing_prompt_packs)}",
                    'p_tok': f"{avg_prompt_token_count:.1f}",
                    'a_valid': f"{avg_prompt_nonpad_ratio:.3f}",
                    'tala_on': f"{tala_on}",
                    'tala_r': f"{tala_ratio_mean:.3f}/{tala_ratio_max:.3f}",
                    'tala_af': f"{tala_active_frac:.3f}",
                    'tala_t': f"{int(t.min().item())}/{int(t.max().item())}",
                    'tala_eps': f"{float(tala_stats.get('tala_effective_eps_std', 0.0)):.3f}",
                    'sem_K': f"{16}",
                    'ad_map_std': f"{adapter_map_std:.4f}",
                    'cond_map_std': f"{cond_map_std:.4f}",
                    'cpu_rss_gb': f"{cpu_rss_gb:.3f}" if math.isfinite(cpu_rss_gb) else "nan",
                }
                log_dict["sft_strength"] = f"{sft_strength_logged:.3f}"
                pbar.set_postfix(log_dict)
                LAST_TRAIN_LOG = {
                    "sft_strength": float(sft_strength_logged),
                    "lpips_train_weight": float(w.get('lpips', 0.0)),
                    "use_legacy_patch_lpips": bool(USE_LEGACY_PATCH_LPIPS),
                    "lpips_num_samples": int(lpips_num_samples),
                    "late_lpips": float(late_lpips.item()),
                    "late_lpips_num_samples": int(late_lpips_num_samples),
                    "late_lpips_active_frac": float(late_lpips_active_frac),
                    "lr_cons_weight": float(w.get('lr_cons', 0.0)),
                    "inject_reg_weight": float(inject_reg_w),
                    "tasr_gate_mean": tasr_gate_mean,
                    "alpha_mean": alpha_mean_sft,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "sem_tok_std": sem_tok_std,
                    "sem_pre_std": sem_pre_std,
                    "sem_post_std": sem_post_std,
                    "sem_out_scale": sem_out_scale,
                    "sem_out_std": sem_out_std,
                    "sem_alpha": sem_alpha,
                    "semantic_alpha_value": sem_alpha,
                    "tasr_gate_min": tasr_gate_min,
                    "tasr_gate_max": tasr_gate_max,
                    "sft_delta_std": sft_delta_std,
                    "hpa_text_ctx_std": hpa_text_ctx_std,
                    "hpa_img_delta_std": hpa_img_delta_std,
                    "cond_delta": cond_delta,
                    "text_cond_delta": text_cond_delta,
                    "run_guard_forward": int(run_guard_forward),
                    "adaptive_prompt_hit_rate": float(prompt_cache_hit_rate),
                    "num_missing_prompt_packs": int(num_missing_prompt_packs),
                    "avg_prompt_token_count": float(avg_prompt_token_count),
                    "avg_cond_prompt_valid_ratio": float(avg_prompt_nonpad_ratio),
                    "tala_on": int(USE_TALA_TRAIN),
                    "tala_ratio_mean": float(tala_ratio_mean),
                    "tala_ratio_max": float(tala_ratio_max),
                    "tala_active_frac": float(tala_active_frac),
                    "tala_t_min_batch": int(t.min().item()),
                    "tala_t_max_batch": int(t.max().item()),
                    "tala_effective_eps_std": float(tala_stats.get("tala_effective_eps_std", 0.0)),
                    "sem_K": 16,
                    "adapter_map_std": adapter_map_std,
                    "cond_map_std": cond_map_std,
                    "cpu_rss_gb": (float(cpu_rss_gb) if math.isfinite(cpu_rss_gb) else float("nan")),
                    "train_prompt_cache_max_items": int(TRAIN_PROMPT_CACHE_MAX_ITEMS),
                    "val_prompt_cache_max_items": int(VAL_PROMPT_CACHE_MAX_ITEMS),
                }

        if accum_micro_steps > 0 and not reached_max_steps:
            log_critical_path_gradients(step + 1, pixart, adapter)
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if ema is not None:
                ema.update(ema_named_params)
            accum_micro_steps = 0

        epoch_1based = epoch + 1
        log_injection_scale_stats(pixart, prefix=f"[InjectScale][Ep{epoch_1based}]")
        run_proxy_val = should_run_proxy_validation(epoch_1based)
        print(f"[VAL-SCHEDULE] epoch={epoch_1based} run_proxy_val={run_proxy_val}")

        if run_proxy_val:
            validation_count += 1
            print(f"🔎 [VAL] Raw weights (validation #{validation_count})")
            val_raw = validate(epoch, pixart, adapter, sem_adapter, vae, val_loader, null_pack, d_info, lpips_fn_val_cpu, val_tag="raw")

            run_ema_val = (
                (ema is not None)
                and EMA_VALIDATE
                and (EMA_VALIDATE_EVERY > 0)
                and (validation_count % EMA_VALIDATE_EVERY == 0)
            )
            val_ema = None
            if run_ema_val:
                print(f"🔎 [VAL] EMA weights (validation #{validation_count})")
                ema.apply(ema_named_params)
                val_ema = validate(epoch, pixart, adapter, sem_adapter, vae, val_loader, null_pack, d_info, lpips_fn_val_cpu, val_tag="ema")
                ema.restore(ema_named_params)

            raw_metrics = val_raw[int(BEST_VAL_STEPS)] if int(BEST_VAL_STEPS) in val_raw else next(iter(val_raw.values()))
            best_eval_source_this_round = "raw"
            best_eval_metrics_this_round = raw_metrics

            if val_ema is not None:
                ema_metrics = val_ema[int(BEST_VAL_STEPS)] if int(BEST_VAL_STEPS) in val_ema else next(iter(val_ema.values()))
                r = raw_metrics
                e = ema_metrics
                print(f"[VAL-CMP@{BEST_VAL_STEPS}] raw: proxy_psnr={r[0]:.2f} proxy_ssim={r[1]:.4f} proxy_lpips={r[2]:.4f} | "
                      f"ema: proxy_psnr={e[0]:.2f} proxy_ssim={e[1]:.4f} proxy_lpips={e[2]:.4f}")
                if should_keep_ckpt(e[0], e[2]) < should_keep_ckpt(r[0], r[2]):
                    best_eval_source_this_round = "ema"
                    best_eval_metrics_this_round = ema_metrics

            print(
                f"[VAL-BEST@{BEST_VAL_STEPS}] source={best_eval_source_this_round} "
                f"proxy_psnr={best_eval_metrics_this_round[0]:.2f} "
                f"proxy_ssim={best_eval_metrics_this_round[1]:.4f} "
                f"proxy_lpips={best_eval_metrics_this_round[2]:.4f}"
            )

            print(f"[SingleStage] proxy_psnr={float(best_eval_metrics_this_round[0]):.3f}")
            last_val_psnr = float(raw_metrics[0])

            best = save_smart(
                epoch,
                step,
                pixart,
                adapter,
                sem_adapter,
                optimizer,
                best,
                best_eval_metrics_this_round,
                dl_gen,
                ema=ema,
                keep_keys=ever_keys,
                eval_source=best_eval_source_this_round,
                eval_steps=int(BEST_VAL_STEPS),
                eval_tag=f"val{validation_count}",
                export_eval_weights=True,
                ema_named_params=ema_named_params,
                last_val_psnr=last_val_psnr,
            )
        else:
            print(f"⏭️ Skip proxy validation at epoch {epoch_1based}")
            save_last_resume_only(
                epoch=epoch,
                global_step=step,
                pixart=pixart,
                adapter=adapter,
                sem_adapter=sem_adapter,
                optimizer=optimizer,
                best_records=best,
                dl_gen=dl_gen,
                ema=ema,
                ema_named_params=ema_named_params,
                last_val_psnr=last_val_psnr,
            )

if __name__ == "__main__":
    main()
