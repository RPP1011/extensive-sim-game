#!/usr/bin/env python3
"""Convert next-state JSONL + ability registry to .npz format.

Reads:
  - nextstate_v2.jsonl: per-tick snapshots (entities, threats, positions)
  - nextstate_v2.registry.jsonl: per-scenario ability DSL text per unit

Pre-computes frozen ability transformer [CLS] embeddings and stores them
as a compact lookup table (LUT). Per-sample arrays index into the LUT
by (scenario, unit_id), avoiding multi-GB duplication.

Output arrays:
  - Standard: ent_feat, ent_types, ent_mask, ent_unit_ids, thr_feat, thr_mask,
              pos_feat, pos_mask, ticks, scenario_ids
  - Ability LUT: abl_lut (n_entries, MAX_ABILITIES, D_MODEL)
                 abl_lut_counts (n_entries,) — abilities per LUT entry
                 abl_ent_idx (n_samples, MAX_ENTS) — LUT index per entity slot (-1 = none)

Usage:
    uv run --with numpy --with torch training/convert_nextstate.py \
        generated/nextstate_v2.jsonl \
        --ability-weights generated/ability_transformer_pretrained_v4.pt \
        -o generated/nextstate_v2.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import AbilityTokenizer
from model import AbilityTransformerMLM

MAX_ENTS = 16
MAX_THR = 8
MAX_POS = 8
ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
MAX_ABILITIES = 9
D_MODEL = 32


def build_ability_lut(
    registry_path: Path,
    weights_path: Path | None,
    tokenizer: AbilityTokenizer,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, int], int]]:
    """Build compact ability embedding LUT from registry.

    Returns:
      lut_embs: (n_entries, MAX_ABILITIES, D_MODEL) float32
      lut_counts: (n_entries,) int8 — abilities per entry
      key_to_idx: (scenario, unit_id) → LUT index
    """
    with open(registry_path) as f:
        registry = [json.loads(line.strip()) for line in f]

    # Collect all unique DSL texts
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    # (scenario, unit_id) → list of text indices
    unit_abilities: dict[tuple[str, int], list[int]] = {}

    for r in registry:
        scenario = r["scenario"]
        for entry in r["entries"]:
            uid = entry["unit_id"]
            text_indices = []
            for dsl_text in entry["abilities"]:
                if dsl_text not in text_to_idx:
                    text_to_idx[dsl_text] = len(all_texts)
                    all_texts.append(dsl_text)
                text_indices.append(text_to_idx[dsl_text])
            if text_indices:
                unit_abilities[(scenario, uid)] = text_indices

    if not all_texts or weights_path is None:
        if not all_texts:
            print("No abilities found in registry")
        else:
            print("No ability weights provided, using zero embeddings")
        # Return empty LUT
        lut = np.zeros((1, MAX_ABILITIES, D_MODEL), dtype=np.float32)
        counts = np.zeros(1, dtype=np.int8)
        return lut, counts, {}

    # Run frozen ability transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights_path, map_location=device, weights_only=True)

    vocab_size = state["transformer.token_emb.weight"].shape[0]
    d_model = state["transformer.token_emb.weight"].shape[1]
    n_layers = sum(1 for k in state if "encoder.layers" in k and "self_attn.in_proj_weight" in k)
    d_ff = state["transformer.encoder.layers.0.linear1.weight"].shape[0]

    print(f"Loading AbilityTransformerMLM: vocab={vocab_size}, d={d_model}, "
          f"layers={n_layers}, d_ff={d_ff}")

    mlm = AbilityTransformerMLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=4,
        n_layers=n_layers, d_ff=d_ff,
    ).to(device)
    mlm.load_state_dict(state)
    mlm.eval()
    transformer = mlm.transformer

    print(f"Computing [CLS] embeddings for {len(all_texts)} unique abilities...")
    input_ids_list, mask_list = tokenizer.batch_encode(all_texts, add_cls=True)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_list, dtype=torch.long, device=device)

    all_cls = []
    with torch.no_grad():
        for start in range(0, len(input_ids), 64):
            end = min(start + 64, len(input_ids))
            cls = transformer.cls_embedding(input_ids[start:end], attention_mask[start:end])
            all_cls.append(cls.cpu().numpy())
    cls_np = np.concatenate(all_cls, axis=0)  # (n_unique_texts, d_model)
    print(f"  {cls_np.shape[0]} embeddings computed")

    # Build LUT: one entry per unique (scenario, unit_id)
    n_entries = len(unit_abilities)
    lut = np.zeros((n_entries, MAX_ABILITIES, d_model), dtype=np.float32)
    counts = np.zeros(n_entries, dtype=np.int8)
    key_to_idx: dict[tuple[str, int], int] = {}

    for i, ((sc, uid), text_idxs) in enumerate(unit_abilities.items()):
        key_to_idx[(sc, uid)] = i
        na = min(len(text_idxs), MAX_ABILITIES)
        counts[i] = na
        for j in range(na):
            lut[i, j] = cls_np[text_idxs[j]]

    print(f"  LUT: {n_entries} entries, {lut.nbytes / 1e3:.1f} KB")
    return lut, counts, key_to_idx


def main():
    p = argparse.ArgumentParser(description="Convert nextstate JSONL to npz")
    p.add_argument("input", help="Path to nextstate .jsonl")
    p.add_argument("-o", "--output", default=None, help="Output .npz path")
    p.add_argument("--ability-weights", default=None,
                   help="Path to pretrained ability transformer .pt (for CLS embeddings)")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".npz")
    registry_path = input_path.with_suffix(".registry.jsonl")

    tokenizer = AbilityTokenizer()

    if registry_path.exists():
        abl_lut, abl_lut_counts, key_to_idx = build_ability_lut(
            registry_path,
            Path(args.ability_weights) if args.ability_weights else None,
            tokenizer,
        )
    else:
        print(f"No registry at {registry_path}")
        abl_lut = np.zeros((1, MAX_ABILITIES, D_MODEL), dtype=np.float32)
        abl_lut_counts = np.zeros(1, dtype=np.int8)
        key_to_idx = {}

    # Build scenario name → list of (unit_id) for LUT lookup
    # We need scenario names from the registry
    scenario_names: dict[int, str] = {}  # scenario_id → name

    # Count lines
    print(f"Counting samples in {input_path}...")
    n = 0
    with open(input_path) as f:
        for _ in f:
            n += 1
    print(f"  {n} samples")

    # Allocate arrays
    ent_feat = np.zeros((n, MAX_ENTS, ENTITY_DIM), dtype=np.float32)
    ent_types = np.zeros((n, MAX_ENTS), dtype=np.int8)
    ent_mask = np.ones((n, MAX_ENTS), dtype=np.bool_)
    ent_unit_ids = np.full((n, MAX_ENTS), -1, dtype=np.int32)
    thr_feat = np.zeros((n, MAX_THR, THREAT_DIM), dtype=np.float32)
    thr_mask = np.ones((n, MAX_THR), dtype=np.bool_)
    pos_feat = np.zeros((n, MAX_POS, POSITION_DIM), dtype=np.float32)
    pos_mask = np.ones((n, MAX_POS), dtype=np.bool_)
    ticks = np.zeros(n, dtype=np.int32)
    scenario_ids = np.zeros(n, dtype=np.int32)
    # Per-entity LUT index (-1 = no abilities)
    abl_ent_idx = np.full((n, MAX_ENTS), -1, dtype=np.int32)

    scenario_map: dict[str, int] = {}

    print("Converting...")
    with open(input_path) as f:
        for i, line in enumerate(f):
            if i % 50000 == 0 and i > 0:
                print(f"  {i}/{n}...")
            s = json.loads(line)

            ents = s["entities"]
            types = s["entity_types"]
            uids = s["entity_unit_ids"]
            ne = min(len(ents), MAX_ENTS)
            for j in range(ne):
                ent_feat[i, j] = ents[j]
                ent_types[i, j] = types[j]
                ent_mask[i, j] = False
                ent_unit_ids[i, j] = uids[j]

            threats = s.get("threats", [])
            nt = min(len(threats), MAX_THR)
            for j in range(nt):
                thr_feat[i, j] = threats[j]
                thr_mask[i, j] = False

            positions = s.get("positions", [])
            np_ = min(len(positions), MAX_POS)
            for j in range(np_):
                pos_feat[i, j] = positions[j]
                pos_mask[i, j] = False

            ticks[i] = s["tick"]

            sc = s.get("scenario", "unknown")
            if sc not in scenario_map:
                scenario_map[sc] = len(scenario_map)
            scenario_ids[i] = scenario_map[sc]

            # Map entity slots to LUT indices
            for j in range(ne):
                key = (sc, uids[j])
                lut_idx = key_to_idx.get(key, -1)
                abl_ent_idx[i, j] = lut_idx

    has_abilities = (abl_ent_idx >= 0).any(axis=1).sum()

    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        ent_feat=ent_feat,
        ent_types=ent_types,
        ent_mask=ent_mask,
        ent_unit_ids=ent_unit_ids,
        thr_feat=thr_feat,
        thr_mask=thr_mask,
        pos_feat=pos_feat,
        pos_mask=pos_mask,
        ticks=ticks,
        scenario_ids=scenario_ids,
        abl_lut=abl_lut,
        abl_lut_counts=abl_lut_counts,
        abl_ent_idx=abl_ent_idx,
    )

    print(f"\nDone: {n} samples, {len(scenario_map)} scenarios")
    print(f"  {has_abilities}/{n} samples have ability data ({has_abilities/n*100:.1f}%)")
    print(f"  LUT: {len(abl_lut)} entries × {MAX_ABILITIES} abilities × {D_MODEL}d")
    print(f"  Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
