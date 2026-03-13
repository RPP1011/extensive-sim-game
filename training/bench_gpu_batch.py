#!/usr/bin/env python3
"""Benchmark GPU inference at various batch sizes to find optimal throughput."""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AbilityActorCriticV4, MAX_ABILITIES, NUM_COMBAT_TYPES
from tokenizer import AbilityTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ENTITIES = 20
ENTITY_DIM = 30
MAX_THREATS = 4
THREAT_DIM = 8
MAX_POSITIONS = 4
POSITION_DIM = 8


def bench(model, batch_sizes, warmup=5, iters=50):
    model.eval()
    print(f"{'Batch':>7} | {'Infer ms':>9} | {'inf/sec':>9} | {'Parse+Infer':>11}")
    print("-" * 50)

    for bs in batch_sizes:
        # Synthetic inputs
        ent_feat = torch.randn(bs, MAX_ENTITIES, ENTITY_DIM, device=DEVICE)
        ent_types = torch.zeros(bs, MAX_ENTITIES, dtype=torch.long, device=DEVICE)
        ent_mask = torch.zeros(bs, MAX_ENTITIES, dtype=torch.bool, device=DEVICE)
        thr_feat = torch.randn(bs, MAX_THREATS, THREAT_DIM, device=DEVICE)
        thr_mask = torch.zeros(bs, MAX_THREATS, dtype=torch.bool, device=DEVICE)
        pos_feat = torch.randn(bs, MAX_POSITIONS, POSITION_DIM, device=DEVICE)
        pos_mask = torch.zeros(bs, MAX_POSITIONS, dtype=torch.bool, device=DEVICE)
        ability_cls = [None] * MAX_ABILITIES

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                      ability_cls, pos_feat, pos_mask)
        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output, value = model(ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                                      ability_cls, pos_feat, pos_mask)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        throughput = bs / (avg_ms / 1000)

        # Simulate parse overhead (~0.5ms baseline + 0.002ms per sample)
        parse_ms = 0.5 + bs * 0.002
        total_ms = avg_ms + parse_ms
        eff_throughput = bs / (total_ms / 1000)

        print(f"{bs:>7} | {avg_ms:>8.2f}ms | {throughput:>8.0f} | {eff_throughput:>8.0f} eff")

        if avg_ms > 5000:
            print("  (skipping larger, too slow)")
            break


def main():
    tok = AbilityTokenizer()
    model = AbilityActorCriticV4(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=4,
        external_cls_dim=128,
        d_model=32, d_ff=64,
        n_layers=4, n_heads=4,
    ).to(DEVICE)

    ckpt = torch.load("generated/actor_critic_v4_full_unfrozen.pt",
                       map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total:,} params on {DEVICE}")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    bench(model, batch_sizes)


if __name__ == "__main__":
    main()
