import os
import sys
import csv
import json
import math
import glob
import random
import argparse
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
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v7


def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


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
            for name in ["maniqa", "musiq", "clipiqa"]:
                try:
                    self.iqa[name] = pyiqa.create_metric(name, device="cpu")
                except Exception as e_metric:
                    print(f"⚠️ pyiqa metric '{name}' unavailable, will output NaN: {e_metric}")
        except Exception as e:
            print(f"⚠️ pyiqa unavailable, MANIQA/MUSIQ/CLIPIQA will output NaN: {e}")

    @torch.no_grad()
    def compute(self, pred_m11: torch.Tensor, hr_m11: torch.Tensor):
        pred_cpu = pred_m11.detach().to("cpu", dtype=torch.float32)
        hr_cpu = hr_m11.detach().to("cpu", dtype=torch.float32)
        pred01 = (pred_cpu + 1.0) / 2.0
        hr01 = (hr_cpu + 1.0) / 2.0

        py = rgb01_to_y01(pred01)[..., 4:-4, 4:-4]
        hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]

        out = {
            "psnr": float(psnr(py, hy, data_range=1.0).item()),
            "ssim": float(ssim(py, hy, data_range=1.0).item()),
            "lpips": float(self.lpips_fn(pred_cpu, hr_cpu).mean().item()),
            "dists": float("nan"),
            "maniqa": float("nan"),
            "musiq": float("nan"),
            "clipiqa": float("nan"),
        }

        if self.dists_fn is not None:
            try:
                out["dists"] = float(self.dists_fn(pred_cpu, hr_cpu).mean().item())
            except Exception as e:
                print(f"⚠️ DISTS compute failed, writing NaN: {e}")

        pred01_clamp = pred01.clamp(0.0, 1.0)
        for name in ["maniqa", "musiq", "clipiqa"]:
            fn = self.iqa.get(name, None)
            if fn is None:
                continue
            try:
                val = fn(pred01_clamp)
                out[name] = float(val.detach().float().mean().item())
            except Exception as e:
                print(f"⚠️ {name} compute failed, writing NaN: {e}")

        return out


def build_model_and_assets(args, device, compute_dtype):
    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        sparse_inject_ratio=1.0,
        dualstream_enabled=False,
        cross_attn_start_layer=16,
        dual_num_heads=16,
    ).to(device)

    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
        if hasattr(pixart, "init_lr_embedder_from_x_embedder"):
            pixart.init_lr_embedder_from_x_embedder()
    else:
        pixart.load_state_dict(base, strict=False)

    adapter = build_adapter_v7(
        in_channels=4,
        hidden_size=1152,
        injection_layers_map=getattr(pixart, "injection_layer_to_level", getattr(pixart, "injection_layers", None)),
    ).to(device).float()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if "pixart_trainable" not in ckpt or "adapter" not in ckpt:
        raise KeyError("Checkpoint must contain keys: pixart_trainable and adapter")

    load_state_dict_shape_compatible(pixart, ckpt.get("pixart_keep", ckpt["pixart_trainable"]), context="eval")
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    if "y" not in null_pack:
        raise KeyError("Null T5 embed file missing key 'y'")
    y_embed = null_pack["y"].to(device)

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
    return pixart, adapter, vae, y_embed, scheduler


@torch.no_grad()
def run_ddim_predict(pixart, adapter, vae, y_embed, scheduler, batch, args, device, compute_dtype, gen):
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

    for t in run_timesteps:
        t_b = torch.tensor([t], device=device).expand(latents.shape[0])
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
            cond = adapter(adapter_in, t_embed=t_embed.float())
            out = pixart(
                x=latents.to(compute_dtype),
                timestep=t_b,
                y=y_embed,
                aug_level=aug_level,
                mask=None,
                data_info=data_info,
                adapter_cond=cond,
                force_drop_ids=torch.ones(latents.shape[0], device=device),
            )
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    return pred, hr, lr


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


def nanmean(xs):
    vals = [float(v) for v in xs if v is not None and not math.isnan(float(v))]
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


def evaluate_dataset(dataset_name: str, loader, args, metric_suite, pixart, adapter, vae, y_embed, scheduler, device, compute_dtype):
    base_out = Path(args.output_dir) / dataset_name
    preds_dir = base_out / "preds"
    trip_dir = base_out / "triptychs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    trip_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    rows = []
    pbar = tqdm(loader, desc=f"Eval[{dataset_name}]@{args.steps}")
    for idx, batch in enumerate(pbar):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        pred, hr, lr = run_ddim_predict(pixart, adapter, vae, y_embed, scheduler, batch, args, device, compute_dtype, gen)
        m = metric_suite.compute(pred, hr)

        hr_path = batch["hr_path"][0] if isinstance(batch["hr_path"], list) else batch["hr_path"]
        lr_path = batch["lr_path"][0] if isinstance(batch["lr_path"], list) else batch["lr_path"]
        image_name = batch["image_name"][0] if isinstance(batch["image_name"], list) else batch["image_name"]
        stem = Path(image_name).stem

        pred_path = preds_dir / f"{stem}_steps{args.steps}.png"
        if args.save_preds:
            tensor_m11_to_pil(pred[0]).save(pred_path)

        if args.save_triptychs and (idx % 5 == 0):
            tri_path = trip_dir / f"{idx:04d}_{stem}_steps{args.steps}.png"
            save_triptych(lr, hr, pred, str(tri_path), args.steps)

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
        }
        rows.append(row)

        pbar.set_postfix({
            "psnr": f"{row['psnr']:.2f}",
            "ssim": f"{row['ssim']:.4f}",
            "lpips": f"{row['lpips']:.4f}",
        })

    csv_path = base_out / "per_image_metrics.csv"
    fieldnames = [
        "dataset", "image_name", "hr_path", "lr_path", "pred_path",
        "psnr", "ssim", "lpips", "dists", "maniqa", "musiq", "clipiqa"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": dataset_name,
        "num_images": len(rows),
        "steps": int(args.steps),
        "use_lq_init": bool(args.use_lq_init),
        "lq_init_strength": float(args.lq_init_strength),
        "mean": {
            "psnr": nanmean([r["psnr"] for r in rows]),
            "ssim": nanmean([r["ssim"] for r in rows]),
            "lpips": nanmean([r["lpips"] for r in rows]),
            "dists": nanmean([r["dists"] for r in rows]),
            "maniqa": nanmean([r["maniqa"] for r in rows]),
            "musiq": nanmean([r["musiq"] for r in rows]),
            "clipiqa": nanmean([r["clipiqa"] for r in rows]),
        }
    }

    with open(base_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(base_out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"✅ [{dataset_name}] wrote: {csv_path}")
    print(f"✅ [{dataset_name}] mean: {summary['mean']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Full SR evaluation on RealSR and DRealSR")
    parser.add_argument("--dataset", type=str, default="both", choices=["realsr", "drealsr", "both"])

    parser.add_argument("--pixart_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae")
    parser.add_argument("--null_t5_embed_path", type=str, default="/home/hello/HJT/PixArt-sigma/output/pretrained_models/null_t5_embed_sigma_300.pth")

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

    parser.add_argument("--output_dir", type=str, default="/home/hello/HJT/SRConvnetAdapter/experiments_results")
    parser.add_argument("--save_preds", action="store_true", default=True)
    parser.add_argument("--save_triptychs", action="store_true", default=True)
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

    pixart, adapter, vae, y_embed, scheduler = build_model_and_assets(args, device, compute_dtype)
    metric_suite = MetricSuite()

    if args.dataset in ("realsr", "both"):
        realsr_ds = RealSRValPairedDataset(roots=args.realsr_roots, crop_size=args.crop_size)
        realsr_loader = DataLoader(realsr_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        evaluate_dataset("realsr", realsr_loader, args, metric_suite, pixart, adapter, vae, y_embed, scheduler, device, compute_dtype)

    if args.dataset in ("drealsr", "both"):
        drealsr_ds = DRealSRPairedDataset(args.drealsr_hr_dir, args.drealsr_lr_dir, crop_size=args.crop_size)
        drealsr_loader = DataLoader(drealsr_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        evaluate_dataset("drealsr", drealsr_loader, args, metric_suite, pixart, adapter, vae, y_embed, scheduler, device, compute_dtype)


if __name__ == "__main__":
    main()
