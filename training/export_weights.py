#!/usr/bin/env python3
"""Export trained ability transformer weights to JSON for Rust inference.

Exports the transformer encoder + decision head in a flat JSON format
matching the project's existing weight loading conventions (row-major
flattened arrays).

Usage:
    uv run --with numpy --with torch training/export_weights.py \
        generated/ability_transformer_decision.pt \
        -o generated/ability_transformer_weights.json \
        --vocab-size 252
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerDecision
from tokenizer import AbilityTokenizer


def export_linear(state_dict: dict, prefix: str) -> dict:
    """Export a nn.Linear layer as {w: [[...]], b: [...]}."""
    w = state_dict[f"{prefix}.weight"].cpu().numpy().tolist()  # [out, in]
    b = state_dict[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_in_proj(state_dict: dict, prefix: str) -> dict:
    """Export nn.MultiheadAttention in_proj (combined QKV) weights.

    PyTorch stores these as in_proj_weight / in_proj_bias (underscore, not dot).
    """
    w = state_dict[f"{prefix}.in_proj_weight"].cpu().numpy().tolist()  # [3*d, d]
    b = state_dict[f"{prefix}.in_proj_bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_layer_norm(state_dict: dict, prefix: str) -> dict:
    """Export a nn.LayerNorm as {gamma: [...], beta: [...]}."""
    gamma = state_dict[f"{prefix}.weight"].cpu().numpy().tolist()
    beta = state_dict[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"gamma": gamma, "beta": beta}


def export_phase1(checkpoint_path: str, output_path: str, args):
    """Export Phase 1 transformer encoder only (no cross-attention or decision head)."""
    from model import AbilityTransformerMLM

    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)
    vocab_size = args.vocab_size or tokenizer.vocab_size

    model = AbilityTransformerMLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Filter to model keys only (exclude hint_head/recon_decoder)
    model_keys = {k: v for k, v in state.items() if not k.startswith(("hint_head.", "recon_decoder."))}
    model.load_state_dict(model_keys, strict=False)
    sd = model.state_dict()

    export = {
        "architecture": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "pad_id": tokenizer.pad_id,
            "cls_id": tokenizer.cls_id,
        },
        "token_embedding": sd["transformer.token_emb.weight"].cpu().numpy().tolist(),
        "position_embedding": sd["transformer.pos_emb.weight"].cpu().numpy().tolist(),
        "output_norm": export_layer_norm(sd, "transformer.out_norm"),
        "layers": [],
    }

    for i in range(args.n_layers):
        prefix = f"transformer.encoder.layers.{i}"
        layer = {
            "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
            "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
            "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
            "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
            "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
            "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
        }
        export["layers"].append(layer)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f)

    size_kb = out.stat().st_size / 1024
    n_params = sum(p.numel() for p in model.transformer.parameters())
    print(f"Exported Phase 1 encoder: {n_params:,} params to {out} ({size_kb:.0f} KB)")


def export_transformer(checkpoint_path: str, output_path: str, args):
    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)

    # Reconstruct model to get state dict keys
    model = AbilityTransformerDecision(
        vocab_size=args.vocab_size or tokenizer.vocab_size,
        game_state_dim=args.game_state_dim,
        n_targets=args.n_targets,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    sd = model.state_dict()

    export = {
        "architecture": {
            "vocab_size": args.vocab_size or tokenizer.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "game_state_dim": args.game_state_dim,
            "n_targets": args.n_targets,
            "pad_id": tokenizer.pad_id,
            "cls_id": tokenizer.cls_id,
        },
        # Embeddings
        "token_embedding": sd["transformer.token_emb.weight"].cpu().numpy().tolist(),
        "position_embedding": sd["transformer.pos_emb.weight"].cpu().numpy().tolist(),
        # Output layer norm
        "output_norm": export_layer_norm(sd, "transformer.out_norm"),
        # Transformer encoder layers
        "layers": [],
    }

    # Export each transformer layer
    for i in range(args.n_layers):
        prefix = f"transformer.encoder.layers.{i}"
        layer = {
            # Self-attention (in_proj combines Q, K, V)
            "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
            "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
            # Feed-forward
            "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
            "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
            # Layer norms (pre-norm)
            "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
            "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
        }
        export["layers"].append(layer)

    # Cross-attention (entity encoder + cross-attention block)
    if args.game_state_dim > 0:
        ee_layers = []
        for i in range(args.n_layers):
            prefix = f"entity_encoder.encoder.layers.{i}"
            ee_layers.append({
                "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
                "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
                "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
                "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
                "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
                "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
            })
        export["entity_encoder"] = {
            "proj": export_linear(sd, "entity_encoder.proj"),
            "type_emb": sd["entity_encoder.type_emb.weight"].cpu().numpy().tolist(),
            "input_norm": export_layer_norm(sd, "entity_encoder.input_norm"),
            "self_attn_layers": ee_layers,
            "out_norm": export_layer_norm(sd, "entity_encoder.out_norm"),
        }
        export["cross_attn"] = {
            "attn_in_proj": export_in_proj(sd, "cross_attn.cross_attn"),
            "attn_out_proj": export_linear(sd, "cross_attn.cross_attn.out_proj"),
            "norm_q": export_layer_norm(sd, "cross_attn.norm_q"),
            "norm_kv": export_layer_norm(sd, "cross_attn.norm_kv"),
            "ff_linear1": export_linear(sd, "cross_attn.ff.0"),
            "ff_linear2": export_linear(sd, "cross_attn.ff.2"),
            "norm_ff": export_layer_norm(sd, "cross_attn.norm_ff"),
        }

    export["decision_head"] = {
        "urgency": {
            "linear1": export_linear(sd, "decision_head.urgency.0"),
            "linear2": export_linear(sd, "decision_head.urgency.2"),
        },
        "target": {
            "linear1": export_linear(sd, "decision_head.target.0"),
            "linear2": export_linear(sd, "decision_head.target.2"),
        },
    }

    # Write JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f)

    size_kb = out.stat().st_size / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Exported {n_params:,} params to {out} ({size_kb:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description="Export transformer weights to JSON")
    p.add_argument("checkpoint", help="PyTorch checkpoint (.pt)")
    p.add_argument("-o", "--output", default="generated/ability_transformer_weights.json")

    p.add_argument("--phase1", action="store_true", help="Export Phase 1 encoder only (no decision head)")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--game-state-dim", type=int, default=0)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=256)

    args = p.parse_args()
    if args.phase1:
        export_phase1(args.checkpoint, args.output, args)
    else:
        export_transformer(args.checkpoint, args.output, args)


if __name__ == "__main__":
    main()
