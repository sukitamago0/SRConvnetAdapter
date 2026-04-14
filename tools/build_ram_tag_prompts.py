#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline RAM tagging -> prompt text cache builder.

Output format (jsonl):
{
  "image_key": "...",
  "tags": ["building", "window", "tree"],
  "prompt_text": "a realistic photo of building, window, tree"
}
"""

import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T


def make_sample_key(path: str) -> str:
    p = os.path.splitext(str(path))[0].replace("\\", "/").strip("/")
    return p.replace("/", "__")


def iter_images(inputs):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for item in inputs:
        if os.path.isdir(item):
            for p in sorted(Path(item).rglob("*")):
                if p.suffix.lower() in exts:
                    yield str(p)
        elif any(ch in item for ch in ["*", "?", "["]):
            for p in sorted(glob.glob(item)):
                if os.path.splitext(p)[1].lower() in exts:
                    yield p
        elif os.path.isfile(item):
            if os.path.splitext(item)[1].lower() in exts:
                yield item
            else:
                with open(item, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line


def load_ram_model(device: str):
    try:
        from ram.models import ram
        from ram import inference_ram
    except Exception as e:
        raise RuntimeError(
            "RAM package not found. Please install a RAM implementation that exposes "
            "`ram.models.ram` and `ram.inference_ram`."
        ) from e
    model = ram(pretrained=True, image_size=384, vit="swin_l")
    model.eval().to(device)
    return model, inference_ram


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="LR image dirs/globs/list-files")
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    paths = list(dict.fromkeys(iter_images(args.inputs)))
    if len(paths) == 0:
        raise RuntimeError("No images found from --inputs")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    model, inference_ram = load_ram_model(args.device)
    tfm = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(args.device)
            tags_str = inference_ram(x, model)[0]
            tags = [t.strip() for t in str(tags_str).split("|") if t.strip()]
            prompt_text = f"a realistic photo of {', '.join(tags)}" if len(tags) > 0 else "a realistic photo of image"
            rec = {
                "image_key": make_sample_key(p),
                "tags": tags,
                "prompt_text": prompt_text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ wrote {len(paths)} prompt records to: {args.output_jsonl}")


if __name__ == "__main__":
    main()

