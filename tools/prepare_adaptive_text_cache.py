#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command adaptive text cache preparation:
1) scan LR images
2) RAM tags -> prompt jsonl
3) T5 encode -> per-image .pth cache
"""

import argparse
import glob
import os
import subprocess


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_roots", nargs="+", required=True)
    ap.add_argument("--ram_pretrained", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--text_encoder_path", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--tmp_jsonl", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.tmp_jsonl)), exist_ok=True)

    run([
        "python", "tools/build_ram_tag_prompts.py",
        "--lr_roots", *args.lr_roots,
        "--output_jsonl", args.tmp_jsonl,
        "--ram_pretrained", args.ram_pretrained,
        "--device", args.device,
    ])
    run([
        "python", "tools/build_sigma_prompt_cache.py",
        "--prompt_jsonl", args.tmp_jsonl,
        "--tokenizer_path", args.tokenizer_path,
        "--text_encoder_path", args.text_encoder_path,
        "--output_root", args.output_root,
        "--device", args.device,
    ])

    with open(args.tmp_jsonl, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    cache_files = sorted(glob.glob(os.path.join(args.output_root, "**", "*.pth"), recursive=True))
    print(f"✅ total images: {len(keys)}")
    print(f"✅ cache files: {len(cache_files)}")
    print("✅ first keys:")
    for line in keys[:5]:
        # best-effort preview
        if "\"image_key\"" in line:
            print("  ", line[:160])
        else:
            print("  ", line)
    print("💡 hit-check suggestion:")
    print("   run training once and verify [adaptive-prompt] example cache exists = True")


if __name__ == "__main__":
    main()
