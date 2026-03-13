#!/usr/bin/env python3
"""Benchmark V4 inference latency vs batch size.

Measures forward pass time for AbilityActorCriticV4 at various batch sizes
to determine whether GPU batched dispatch (IMPALA-style) is worth it vs
per-episode CPU inference.

Usage:
    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/bench_v4_inference.py \
        --pretrained generated/actor_critic_v4_full_unfrozen.pt
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AbilityActorCriticV4, MAX_ABILITIES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
MAX_ENTITIES = 14  # typical max in HvH


def bench_forward(model, batch_size, n_entities=MAX_ENTITIES, n_iters=200, warmup=20):
    """Benchmark forward pass at given batch size."""
    # Synthetic data matching real distribution
    ent_feat = torch.randn(batch_size, n_entities, ENTITY_DIM, device=DEVICE)
    ent_types = torch.randint(0, 3, (batch_size, n_entities), device=DEVICE)
    thr_feat = torch.zeros(batch_size, 1, THREAT_DIM, device=DEVICE)
    ent_mask = torch.zeros(batch_size, n_entities, dtype=torch.bool, device=DEVICE)
    thr_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE)
    pos_feat = torch.zeros(batch_size, 1, POSITION_DIM, device=DEVICE)
    pos_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE)
    ability_cls = [None] * MAX_ABILITIES

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                  ability_cls, pos_feat, pos_mask)

        torch.cuda.synchronize() if DEVICE == "cuda" else None

        # Timed iterations
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                  ability_cls, pos_feat, pos_mask)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        elapsed = time.perf_counter() - t0

    ms_per_batch = (elapsed / n_iters) * 1000
    ms_per_sample = ms_per_batch / batch_size
    samples_per_sec = batch_size * n_iters / elapsed

    return ms_per_batch, ms_per_sample, samples_per_sec


def bench_cpu_forward(model_cpu, n_entities=MAX_ENTITIES, n_iters=1000, warmup=50):
    """Benchmark single-sample CPU inference (simulates Rust inference)."""
    ent_feat = torch.randn(1, n_entities, ENTITY_DIM)
    ent_types = torch.randint(0, 3, (1, n_entities))
    thr_feat = torch.zeros(1, 1, THREAT_DIM)
    ent_mask = torch.zeros(1, n_entities, dtype=torch.bool)
    thr_mask = torch.ones(1, 1, dtype=torch.bool)
    pos_feat = torch.zeros(1, 1, POSITION_DIM)
    pos_mask = torch.ones(1, 1, dtype=torch.bool)
    ability_cls = [None] * MAX_ABILITIES

    model_cpu.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model_cpu(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                      ability_cls, pos_feat, pos_mask)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            model_cpu(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                      ability_cls, pos_feat, pos_mask)
        elapsed = time.perf_counter() - t0

    us_per_sample = (elapsed / n_iters) * 1e6
    return us_per_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="generated/actor_critic_v4_full_unfrozen.pt")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    args = p.parse_args()

    from tokenizer import AbilityTokenizer
    tok = AbilityTokenizer()

    # Build model
    model = AbilityActorCriticV4(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=4,
        external_cls_dim=128,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    if args.pretrained and Path(args.pretrained).exists():
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights from {args.pretrained}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params, d_model={args.d_model}")
    print(f"Device: {DEVICE}")
    print()

    # CPU benchmark (simulates Rust single-sample inference)
    model_cpu = model.cpu()
    cpu_us = bench_cpu_forward(model_cpu)
    print(f"CPU single-sample: {cpu_us:.1f} µs/sample ({1e6/cpu_us:.0f} samples/sec)")
    print()

    # GPU benchmarks at various batch sizes
    model_gpu = model.to(DEVICE)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"{'Batch':>6} {'ms/batch':>10} {'µs/sample':>11} {'samples/s':>12} {'vs CPU':>8}")
    print("-" * 55)

    for bs in batch_sizes:
        ms_batch, ms_sample, sps = bench_forward(model_gpu, bs)
        us_sample = ms_sample * 1000
        speedup = cpu_us / us_sample
        print(f"{bs:>6} {ms_batch:>10.3f} {us_sample:>11.1f} {sps:>12,.0f} {speedup:>7.1f}x")

    print()
    print("'vs CPU' = speedup over single-sample CPU inference")
    print("GPU wins when vs CPU > 1.0 AND batch collection latency is acceptable")


if __name__ == "__main__":
    main()
