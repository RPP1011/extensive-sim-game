#!/usr/bin/env python3
"""Export CLS embeddings for all known abilities to a registry JSON.

The registry maps ability names → 128-dim CLS embeddings, with a model hash
so Rust can verify the embeddings match the current model weights.

Also exports the behavioral outcome normalization stats (mean/std per dim)
so the runtime can denormalize predictions.

Usage:
    uv run --with numpy --with torch training/export_embedding_registry.py \
        generated/ability_transformer_pretrained_v6.pt \
        --ability-data generated/ability_dataset_curated.npz \
        --behavioral-data dataset/ability_profiles.npz \
        -o generated/ability_embedding_registry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerMLM
from tokenizer import AbilityTokenizer


def model_hash(state_dict: dict) -> str:
    """Compute a short hash of model weights for version tracking."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        h.update(key.encode())
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description="Export CLS embedding registry")
    p.add_argument("checkpoint", help="Pretrained .pt checkpoint")
    p.add_argument("--ability-data", required=True,
                   help="Pre-tokenized ability dataset (.npz with token_ids + texts)")
    p.add_argument("--behavioral-data", default=None,
                   help="ability_profiles.npz for outcome normalization stats")
    p.add_argument("-o", "--output", default="generated/ability_embedding_registry.json")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)

    # Load model
    model = AbilityTransformerMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
        cls_id=tokenizer.cls_id,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Filter to transformer + mlm_head keys only
    model_keys = {k: v for k, v in state.items()
                  if k.startswith(("transformer.", "mlm_head."))}
    model.load_state_dict(model_keys, strict=False)
    model.eval()

    mhash = model_hash(model_keys)
    print(f"Model hash: {mhash}")

    # Load ability texts from dataset
    data = np.load(args.ability_data, allow_pickle=True)
    if "texts" in data:
        texts = list(data["texts"])
    elif "dsl_texts" in data:
        blob = data["dsl_texts"].tobytes().decode("utf-8")
        texts = [t.strip() for t in blob.split("---SEPARATOR---") if t.strip()]
    else:
        raise ValueError("No texts found in ability dataset")

    # Deduplicate by ability name — only keep first occurrence (original, not augmented)
    seen = {}
    unique_texts = []
    for text in texts:
        name = _extract_name(text)
        if name and name not in seen:
            seen[name] = len(unique_texts)
            unique_texts.append(text)

    print(f"Unique abilities: {len(unique_texts)} (from {len(texts)} total)")

    # Batch encode all abilities
    embeddings = {}
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(unique_texts), batch_size):
            batch_texts = unique_texts[i:i + batch_size]
            batch_ids = []
            batch_names = []
            max_len = 0
            for text in batch_texts:
                ids = tokenizer.encode(text, add_cls=True)
                batch_ids.append(ids)
                max_len = max(max_len, len(ids))
                batch_names.append(_extract_name(text))

            # Pad to max_len
            padded = []
            masks = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [tokenizer.pad_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            cls = model.transformer.cls_embedding(input_ids, attention_mask)

            for j, name in enumerate(batch_names):
                if name:
                    embeddings[name] = cls[j].cpu().numpy().tolist()

    print(f"Exported {len(embeddings)} embeddings (d={args.d_model})")

    # Build registry
    registry = {
        "model_hash": mhash,
        "d_model": args.d_model,
        "n_abilities": len(embeddings),
        "embeddings": embeddings,
    }

    # Add behavioral normalization stats if available
    if args.behavioral_data:
        beh = np.load(args.behavioral_data, allow_pickle=True)
        outcome = beh["outcome"]
        registry["outcome_mean"] = outcome.mean(axis=0).tolist()
        registry["outcome_std"] = outcome.std(axis=0).clip(min=1e-6).tolist()
        print(f"Added outcome normalization stats ({outcome.shape[1]} dims)")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(registry, f)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Registry saved to {out_path} ({size_mb:.1f} MB)")


def _extract_name(text: str) -> str | None:
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ability ") or line.startswith("passive "):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].rstrip("{").strip()
    return None


if __name__ == "__main__":
    main()
