#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline RAM tagging -> prompt text jsonl.
"""

import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image
import torch

from utils.prompt_key_utils import make_sample_key


def iter_lr_images(lr_roots):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for root in lr_roots:
        if not os.path.isdir(root):
            continue
        for p in sorted(Path(root).rglob("*")):
            if p.suffix.lower() not in exts:
                continue
            # only keep LR-like files/paths
            s = str(p).lower()
            if ("_lr" in s) or ("lr" in s) or ("x1" in s) or ("lq" in s):
                yield str(p)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_roots", nargs="+", required=True, help="LR image root dirs")
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--ram_pretrained", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_ram(args):
    try:
        from ram.models import ram
        from ram import inference_ram
        from ram import get_transform
    except Exception as e:
        raise RuntimeError(
            "RAM package not available. Please install the official RAM package with ram/get_transform/inference_ram."
        ) from e
    model = ram(pretrained=args.ram_pretrained, image_size=args.image_size, vit="swin_l")
    model.eval().to(args.device)
    transform = get_transform(image_size=args.image_size)
    return model, inference_ram, transform


@torch.no_grad()
def main():
    args = parse_args()
    paths = list(dict.fromkeys(iter_lr_images(args.lr_roots)))
    if len(paths) == 0:
        raise RuntimeError("No LR images found under --lr_roots")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    model, inference_ram, transform = load_ram(args)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(args.device)
            tags_str = inference_ram(x, model)[0]
            tags = [t.strip() for t in str(tags_str).split("|") if t.strip()]
            prompt_text = f"a realistic photo of {', '.join(tags)}" if len(tags) > 0 else "a realistic photo of image"
            rec = {"image_key": make_sample_key(p), "tags": tags, "prompt_text": prompt_text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ wrote {len(paths)} records to: {args.output_jsonl}")
    print(f"✅ first key: {make_sample_key(paths[0])}")


if __name__ == "__main__":
    main()

