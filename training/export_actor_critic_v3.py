#!/usr/bin/env python3
"""Export V3 actor-critic weights to JSON for Rust inference.

V3 uses EntityEncoderV3 (entity + threat + position projections, 5 type embeddings)
and PointerHead (action type + pointer distributions).

Usage:
    uv run --with numpy --with torch training/export_actor_critic_v3.py \
        generated/actor_critic_v3.pt \
        -o generated/actor_critic_weights_v3.json \
        --d-model 32 --d-ff 64 --n-layers 4 --n-heads 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityActorCriticV3, MAX_ABILITIES
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
    p = argparse.ArgumentParser(description="Export V3 actor-critic weights to JSON")
    p.add_argument("checkpoint", help="V3 actor-critic checkpoint (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_weights_v3.json")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--external-cls-dim", type=int, default=0,
                   help="External CLS dimension (e.g. 128 for behavioral embeddings)")
    args = p.parse_args()

    tok = AbilityTokenizer(max_length=args.max_seq_len)

    model = AbilityActorCriticV3(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    sd = model.state_dict()

    export = {
        "format": "actor_critic_v3",
        "architecture": {
            "vocab_size": tok.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "game_state_dim": 0,
            "num_base_actions": 6,  # kept for compat but unused in V3
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

    # V3 entity encoder (entity + threat + position projections)
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
    export["entity_encoder_v3"] = {
        "entity_proj": export_linear(sd, "entity_encoder.entity_proj"),
        "threat_proj": export_linear(sd, "entity_encoder.threat_proj"),
        "position_proj": export_linear(sd, "entity_encoder.position_proj"),
        "type_emb": sd["entity_encoder.type_emb.weight"].cpu().numpy().tolist(),
        "input_norm": export_layer_norm(sd, "entity_encoder.input_norm"),
        "self_attn_layers": ee_layers,
        "out_norm": export_layer_norm(sd, "entity_encoder.out_norm"),
    }

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

    # External CLS projection (behavioral embeddings 128→d_model)
    if model.external_cls_proj is not None:
        export["external_cls_proj"] = export_linear(sd, "external_cls_proj")

    # Pointer head
    export["pointer_head"] = {
        "action_type_head": {
            "linear1": export_linear(sd, "pointer_head.action_type_head.0"),
            "linear2": export_linear(sd, "pointer_head.action_type_head.2"),
        },
        "pointer_key": export_linear(sd, "pointer_head.pointer_key"),
        "attack_query": export_linear(sd, "pointer_head.attack_query"),
        "move_query": export_linear(sd, "pointer_head.move_query"),
        "ability_queries": [
            export_linear(sd, f"pointer_head.ability_queries.{i}")
            for i in range(MAX_ABILITIES)
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f)

    size_kb = out.stat().st_size / 1024
    n_params = sum(p.numel() for p in model.parameters())
    n_value = sum(p.numel() for p in model.value_head.parameters())
    print(f"Exported {n_params - n_value:,} actor params to {out} ({size_kb:.0f} KB)")
    print(f"  (Skipped {n_value:,} value head params — training only)")


if __name__ == "__main__":
    main()
