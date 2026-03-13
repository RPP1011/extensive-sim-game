#!/usr/bin/env python3
"""Benchmark V4 GPU throughput for IMPALA planning.

Measures:
1. Forward-only throughput (actor inference)
2. Forward+backward throughput (learner updates)
3. Interleaved: inference batches between training steps
4. Sim-side: how fast can Rust generate states?

Usage:
    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/bench_v4_throughput.py
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
from model import AbilityActorCriticV4, MAX_ABILITIES, NUM_MOVE_DIRS, NUM_COMBAT_TYPES

DEVICE = "cuda"
ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
MAX_ENTITIES = 14


def make_batch(batch_size):
    return {
        "ent_feat": torch.randn(batch_size, MAX_ENTITIES, ENTITY_DIM, device=DEVICE),
        "ent_types": torch.randint(0, 3, (batch_size, MAX_ENTITIES), device=DEVICE),
        "thr_feat": torch.zeros(batch_size, 1, THREAT_DIM, device=DEVICE),
        "ent_mask": torch.zeros(batch_size, MAX_ENTITIES, dtype=torch.bool, device=DEVICE),
        "thr_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE),
        "pos_feat": torch.zeros(batch_size, 1, POSITION_DIM, device=DEVICE),
        "pos_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE),
        "ability_cls": [None] * MAX_ABILITIES,
    }


def forward(model, b):
    return model(
        b["ent_feat"], b["ent_types"], b["thr_feat"],
        b["ent_mask"], b["thr_mask"], b["ability_cls"],
        b["pos_feat"], b["pos_mask"],
    )


def main():
    from tokenizer import AbilityTokenizer
    tok = AbilityTokenizer()

    model = AbilityActorCriticV4(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=4,
        external_cls_dim=128,
        d_model=32, d_ff=64, n_layers=4, n_heads=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # === 1. Forward-only throughput (actor inference) ===
    print("=== Forward-only (actor inference) ===")
    print(f"{'Batch':>6} {'ms/batch':>10} {'batches/s':>10} {'samples/s':>12}")
    print("-" * 45)

    for bs in [32, 64, 128, 256, 512]:
        b = make_batch(bs)
        model.eval()
        with torch.no_grad():
            # warmup
            for _ in range(50):
                forward(model, b)
            torch.cuda.synchronize()

            n = 500
            t0 = time.perf_counter()
            for _ in range(n):
                forward(model, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        ms_batch = elapsed / n * 1000
        bps = n / elapsed
        sps = bs * n / elapsed
        print(f"{bs:>6} {ms_batch:>10.3f} {bps:>10.0f} {sps:>12,.0f}")

    # === 2. Forward+backward throughput (learner) ===
    print()
    print("=== Forward+backward (learner training step) ===")
    print(f"{'Batch':>6} {'ms/step':>10} {'steps/s':>10} {'samples/s':>12}")
    print("-" * 45)

    for bs in [32, 64, 128, 256, 512]:
        b = make_batch(bs)
        move_tgt = torch.randint(0, NUM_MOVE_DIRS, (bs,), device=DEVICE)
        combat_tgt = torch.randint(0, NUM_COMBAT_TYPES, (bs,), device=DEVICE)

        model.train()
        # warmup
        for _ in range(20):
            output, _ = forward(model, b)
            loss = F.cross_entropy(output["move_logits"], move_tgt) + \
                   F.cross_entropy(output["combat_logits"], combat_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            output, _ = forward(model, b)
            loss = F.cross_entropy(output["move_logits"], move_tgt) + \
                   F.cross_entropy(output["combat_logits"], combat_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_step = elapsed / n * 1000
        sps_step = n / elapsed
        sps_sample = bs * n / elapsed
        print(f"{bs:>6} {ms_step:>10.3f} {sps_step:>10.0f} {sps_sample:>12,.0f}")

    # === 3. Interleaved: K inference batches per 1 training step ===
    print()
    print("=== Interleaved: inference batches between training steps ===")
    print("(Simulates IMPALA: actors need inference, learner needs training)")
    print()

    train_bs = 256
    infer_bs = 64
    b_train = make_batch(train_bs)
    b_infer = make_batch(infer_bs)
    move_tgt = torch.randint(0, NUM_MOVE_DIRS, (train_bs,), device=DEVICE)
    combat_tgt = torch.randint(0, NUM_COMBAT_TYPES, (train_bs,), device=DEVICE)

    print(f"Train batch={train_bs}, Inference batch={infer_bs}")
    print(f"{'Infer/Train':>12} {'ms/cycle':>10} {'cycles/s':>10} {'infer_sps':>12} {'train_sps':>12}")
    print("-" * 60)

    for k in [1, 2, 4, 8, 16]:
        # warmup
        for _ in range(10):
            model.eval()
            with torch.no_grad():
                for _ in range(k):
                    forward(model, b_infer)
            model.train()
            output, _ = forward(model, b_train)
            loss = F.cross_entropy(output["move_logits"], move_tgt) + \
                   F.cross_entropy(output["combat_logits"], combat_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            model.eval()
            with torch.no_grad():
                for _ in range(k):
                    forward(model, b_infer)
            model.train()
            output, _ = forward(model, b_train)
            loss = F.cross_entropy(output["move_logits"], move_tgt) + \
                   F.cross_entropy(output["combat_logits"], combat_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_cycle = elapsed / n * 1000
        cps = n / elapsed
        infer_sps = k * infer_bs * n / elapsed
        train_sps = train_bs * n / elapsed
        print(f"{k:>12} {ms_cycle:>10.3f} {cps:>10.0f} {infer_sps:>12,.0f} {train_sps:>12,.0f}")

    # === 4. Max sustained throughput estimate ===
    print()
    print("=== Sustained throughput estimate ===")

    # Measure pure forward at bs=64 (typical IMPALA inference batch)
    b64 = make_batch(64)
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            forward(model, b64)
        torch.cuda.synchronize()
        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            forward(model, b64)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) / n * 1000

    # Measure fwd+bwd at bs=256
    b256 = make_batch(256)
    mt = torch.randint(0, NUM_MOVE_DIRS, (256,), device=DEVICE)
    ct = torch.randint(0, NUM_COMBAT_TYPES, (256,), device=DEVICE)
    model.train()
    for _ in range(20):
        o, _ = forward(model, b256)
        l = F.cross_entropy(o["move_logits"], mt) + F.cross_entropy(o["combat_logits"], ct)
        optimizer.zero_grad(); l.backward(); optimizer.step()
    torch.cuda.synchronize()
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        o, _ = forward(model, b256)
        l = F.cross_entropy(o["move_logits"], mt) + F.cross_entropy(o["combat_logits"], ct)
        optimizer.zero_grad(); l.backward(); optimizer.step()
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) / n * 1000

    print(f"  Forward (bs=64): {fwd_ms:.3f} ms → {64/fwd_ms*1000:.0f} inferences/sec")
    print(f"  Train step (bs=256): {train_ms:.3f} ms → {256/train_ms*1000:.0f} samples/sec")
    print(f"  If 50% GPU for inference, 50% for training:")
    print(f"    Inference: {64/fwd_ms*1000*0.5:.0f} samples/sec")
    print(f"    Training:  {256/train_ms*1000*0.5:.0f} samples/sec")
    print()
    print("  Sim-side requirement: each rayon thread produces ~30-40 decisions/episode")
    print("  At 100ms/episode with 16 threads: ~160 eps/s × 35 decisions = ~5,600 decisions/sec")
    print(f"  GPU can serve: {64/fwd_ms*1000:.0f} inferences/sec (inference only)")
    print(f"  Bottleneck: {'GPU' if 64/fwd_ms*1000 < 5600 else 'Sim'}")


if __name__ == "__main__":
    main()
