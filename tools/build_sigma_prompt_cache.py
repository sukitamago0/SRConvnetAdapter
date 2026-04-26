#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build offline PixArt-Sigma T5 prompt cache from jsonl prompt records.

Input jsonl format (from build_ram_tag_prompts.py):
  {"image_key":"...", "prompt_text":"..."}

Per-sample output pack (.pth):
  y, mask, hidden, attention_mask, meta
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5EncoderModel


def str_to_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_jsonl", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--text_encoder_path", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=300)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, local_files_only=True).eval().to(args.device)
    save_dtype = str_to_dtype(args.save_dtype)

    n = 0
    with open(args.prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec["image_key"]
            prompt_text = rec.get("prompt_text", "").strip()
            toks = tokenizer(
                prompt_text,
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = toks.input_ids.to(args.device)
            attention_mask = toks.attention_mask.to(args.device)
            hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]  # [1,L,C]
            y = hidden.unsqueeze(1)  # [1,1,L,C]
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [1,1,1,L]
            pack = {
                "y": y.detach().cpu().to(save_dtype),
                "mask": mask.detach().cpu(),
                "hidden": hidden.detach().cpu().to(save_dtype),
                "attention_mask": attention_mask.detach().cpu(),
                "meta": {
                    "image_key": key,
                    "prompt_text": prompt_text,
                    "max_length": int(args.max_length),
                    "tokenizer_path": args.tokenizer_path,
                    "text_encoder_path": args.text_encoder_path,
                    "y_shape": tuple(y.shape),
                    "mask_shape": tuple(mask.shape),
                },
            }
            out_path = Path(args.output_root) / f"{key}.pth"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pack, str(out_path))
            n += 1
    print(f"✅ wrote {n} prompt cache files to: {args.output_root}")


if __name__ == "__main__":
    main()

