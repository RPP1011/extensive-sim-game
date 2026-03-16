#!/usr/bin/env python3
"""Export V5 actor-critic weights to JSON for Rust inference.

V5 extends V4 with:
  - d_model=128, 8 heads (was 32/4)
  - Entity features: 34-dim (30 base + 4 spatial summary)
  - Threat features: 10-dim (8 base + kind + LOS)
  - 6 type embeddings (adds aggregate type=5)
  - Aggregate token projection (16 → d_model)
  - No external_cls_proj when CLS dim matches d_model

Note: The latent interface is GPU-only and not exported to JSON.
Rust inference uses the entity encoder + cross-attention + decision heads + CfC temporal cell.

Usage:
    uv run --with numpy --with torch training/export_actor_critic_v5.py \
        generated/actor_critic_v5.pt \
        -o generated/actor_critic_weights_v5.json \
        --external-cls-dim 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityActorCriticV5, MAX_ABILITIES
from tokenizer import AbilityTokenizer


def export_linear(sd: dict, prefix: str) -> dict:
    w = sd[f"{prefix}.weight"].cpu().numpy().tolist()
    b = sd[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_in_proj(sd: dict, prefix: str) -> dict:
    w = sd[f"{prefix}.in_proj_weight"].cpu().numpy().tolist()
    b = sd[f"{prefix}.in_proj_bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_layer_norm(sd: dict, prefix: str) -> dict:
    gamma = sd[f"{prefix}.weight"].cpu().numpy().tolist()
    beta = sd[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"gamma": gamma, "beta": beta}


def main():
    p = argparse.ArgumentParser(description="Export V5 actor-critic weights to JSON")
    p.add_argument("checkpoint", help="V5 actor-critic checkpoint (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_weights_v5.json")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--external-cls-dim", type=int, default=128)
    p.add_argument("--h-dim", type=int, default=64)
    p.add_argument("--n-latents", type=int, default=12)
    p.add_argument("--n-latent-blocks", type=int, default=2)
    args = p.parse_args()

    tok = AbilityTokenizer(max_length=args.max_seq_len)

    model = AbilityActorCriticV5(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        h_dim=args.h_dim,
        n_latents=args.n_latents,
        n_latent_blocks=args.n_latent_blocks,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    sd = model.state_dict()

    export = {
        "format": "actor_critic_v5",
        "architecture": {
            "vocab_size": tok.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "pad_id": tok.pad_id,
            "cls_id": tok.cls_id,
        },
        "token_embedding": sd["transformer.token_emb.weight"].cpu().numpy().tolist(),
        "position_embedding": sd["transformer.pos_emb.weight"].cpu().numpy().tolist(),
        "output_norm": export_layer_norm(sd, "transformer.out_norm"),
        "layers": [],
    }

    # Ability transformer layers
    for i in range(args.n_layers):
        prefix = f"transformer.encoder.layers.{i}"
        export["layers"].append({
            "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
            "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
            "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
            "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
            "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
            "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
        })

    # Entity encoder V5
    ee_layers = []
    for i in range(args.entity_encoder_layers):
        prefix = f"entity_encoder.encoder.layers.{i}"
        ee_layers.append({
            "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
            "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
            "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
            "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
            "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
            "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
        })
    ee_export = {
        "entity_proj": export_linear(sd, "entity_encoder.entity_proj"),
        "zone_proj": export_linear(sd, "entity_encoder.zone_proj"),
        "agg_proj": export_linear(sd, "entity_encoder.agg_proj"),
        "type_emb": sd["entity_encoder.type_emb.weight"].cpu().numpy().tolist(),
        "input_norm": export_layer_norm(sd, "entity_encoder.input_norm"),
        "layers": ee_layers,
        "out_norm": export_layer_norm(sd, "entity_encoder.out_norm"),
    }
    export["entity_encoder_v5"] = ee_export

    # Cross-attention
    export["cross_attn"] = {
        "attn_in_proj": export_in_proj(sd, "cross_attn.cross_attn"),
        "attn_out_proj": export_linear(sd, "cross_attn.cross_attn.out_proj"),
        "norm_q": export_layer_norm(sd, "cross_attn.norm_q"),
        "norm_kv": export_layer_norm(sd, "cross_attn.norm_kv"),
        "ff_linear1": export_linear(sd, "cross_attn.ff.0"),
        "ff_linear2": export_linear(sd, "cross_attn.ff.2"),
        "norm_ff": export_layer_norm(sd, "cross_attn.norm_ff"),
    }

    # External CLS projection (only if external_cls_dim != d_model)
    if model.external_cls_proj is not None:
        export["external_cls_proj"] = export_linear(sd, "external_cls_proj")

    # CfC temporal cell
    export["temporal_cell"] = {
        "f_gate": export_linear(sd, "temporal_cell.f_gate"),
        "h_gate": export_linear(sd, "temporal_cell.h_gate"),
        "t_a": export_linear(sd, "temporal_cell.t_a"),
        "t_b": export_linear(sd, "temporal_cell.t_b"),
        "proj": export_linear(sd, "temporal_cell.proj"),
        "h_dim": args.h_dim,
    }

    # Move head (9-way directional)
    export["move_head"] = {
        "linear1": export_linear(sd, "move_head.0"),
        "linear2": export_linear(sd, "move_head.2"),
    }

    # Combat pointer head
    export["combat_head"] = {
        "combat_type_head": {
            "linear1": export_linear(sd, "combat_head.combat_type_head.0"),
            "linear2": export_linear(sd, "combat_head.combat_type_head.2"),
        },
        "pointer_key": export_linear(sd, "combat_head.pointer_key"),
        "attack_query": export_linear(sd, "combat_head.attack_query"),
        "ability_queries": [
            export_linear(sd, f"combat_head.ability_queries.{i}")
            for i in range(MAX_ABILITIES)
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f)

    # Count params (excluding GPU-only components: latent interface, value head)
    gpu_only_prefixes = ("latent_interface.", "value_head.")
    n_exported = sum(
        p.numel() for name, p in model.named_parameters()
        if not any(name.startswith(pfx) for pfx in gpu_only_prefixes)
    )
    n_total = sum(p.numel() for p in model.parameters())
    size_kb = out.stat().st_size / 1024
    print(f"Exported {n_exported:,} params to {out} ({size_kb:.0f} KB)")
    print(f"  (Skipped {n_total - n_exported:,} GPU-only params: latent interface, value head)")


if __name__ == "__main__":
    main()
