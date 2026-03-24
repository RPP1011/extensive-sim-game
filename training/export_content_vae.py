#!/usr/bin/env python3
"""Export ContentVAE weights to JSON for Rust inference.

Usage:
    uv run --with torch --find-links https://download.pytorch.org/whl/cu124 \
        python3 training/export_content_vae.py \
        --checkpoint generated/content_vae/content_vae_best.pt \
        --output generated/content_vae_weights.json
"""

import argparse
import json
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="generated/content_vae/content_vae_best.pt")
    parser.add_argument("--output", "-o", default="generated/content_vae_weights.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    state = ckpt["model_state"]

    # Build JSON structure matching what Rust will load
    weights = {
        "config": config,
        "epoch": ckpt["epoch"],
        "val_loss": float(ckpt["val_loss"]),
    }

    # Export each parameter as a flat list
    params = {}
    for key, tensor in state.items():
        params[key] = {
            "shape": list(tensor.shape),
            "data": tensor.flatten().tolist(),
        }

    weights["params"] = params

    with open(args.output, "w") as f:
        json.dump(weights, f)

    size_mb = len(json.dumps(weights)) / 1_000_000
    print(f"Exported {len(params)} parameters ({sum(t.numel() for t in state.values()):,} values)")
    print(f"Config: {config}")
    print(f"Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
