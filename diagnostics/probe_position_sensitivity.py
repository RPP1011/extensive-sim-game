#!/usr/bin/env python3
"""Probe entity encoder sensitivity to positional features.

Creates synthetic game states with a unit at different positions relative to
a zone threat, and checks if the encoder embeddings change meaningfully.

Tests:
  1. Self-entity position sweep (features 5,6) — does embedding change?
  2. Distance-from-enemy sweep (feature 7) — spatial awareness
  3. Threat dx/dy sweep — can encoder see where threats are?
  4. Zone count sweep (feature 10) — hostile zone awareness
  5. Full scenario: unit at safe vs danger positions near a zone
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from model import AbilityActorCriticV2
from tokenizer import AbilityTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTITY_DIM = 30
THREAT_DIM = 8


def make_entity(
    hp_pct=0.8, pos_x=0.0, pos_y=0.0, dist_from_caster=0.0,
    hostile_zones=0, move_speed=2.5, attack_range=5.0, auto_dps=15.0,
    exists=1.0,
):
    """Create a 30-dim entity feature vector."""
    f = [0.0] * ENTITY_DIM
    f[0] = hp_pct
    f[5] = pos_x / 20.0
    f[6] = pos_y / 20.0
    f[7] = dist_from_caster / 10.0
    f[10] = hostile_zones / 3.0
    f[12] = auto_dps / 30.0
    f[13] = attack_range / 10.0
    f[27] = move_speed / 5.0
    f[29] = exists
    return f


def make_threat(dx=0.0, dy=0.0, radius=3.0, time_ms=500.0, damage_ratio=0.5, exists=1.0):
    """Create an 8-dim threat feature vector."""
    dist = (dx**2 + dy**2) ** 0.5
    return [
        dx / 10.0, dy / 10.0, dist / 10.0,
        radius / 5.0, time_ms / 2000.0, damage_ratio,
        0.0,  # no CC
        exists,
    ]


def make_state(
    self_entity, enemy_entity=None, threats=None,
):
    """Build tensors for a single game state (batch=1)."""
    entities = [self_entity]
    types = [0]  # self

    if enemy_entity:
        entities.append(enemy_entity)
        types.append(1)  # enemy

    ent_feat = torch.tensor([entities], dtype=torch.float, device=DEVICE)
    ent_types = torch.tensor([types], dtype=torch.long, device=DEVICE)
    ent_mask = torch.zeros(1, len(entities), dtype=torch.bool, device=DEVICE)

    if threats:
        thr_feat = torch.tensor([threats], dtype=torch.float, device=DEVICE)
        thr_mask = torch.zeros(1, len(threats), dtype=torch.bool, device=DEVICE)
    else:
        thr_feat = torch.zeros(1, 1, THREAT_DIM, device=DEVICE)
        thr_mask = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)

    return ent_feat, ent_types, thr_feat, ent_mask, thr_mask


def get_pooled_embedding(model, *state_args):
    """Get the pooled entity embedding (same as what feeds base_head/value_head)."""
    with torch.no_grad():
        ent_feat, ent_types, thr_feat, ent_mask, thr_mask = state_args
        # Forward through entity encoder only
        tokens, full_mask = model.entity_encoder(
            ent_feat, ent_types, thr_feat, ent_mask, thr_mask
        )
        # Pool: mean of non-masked tokens
        valid = (~full_mask).unsqueeze(-1).float()
        pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0)


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def l2_dist(a, b):
    return (a - b).norm().item()


def main():
    tok = AbilityTokenizer()
    model = AbilityActorCriticV2(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=4,
        d_model=64, d_ff=128, n_layers=4, n_heads=4,
    ).to(DEVICE)

    # Load warmstart checkpoint
    ckpt = "generated/actor_critic_v2_curriculum.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt}\n")

    # ── Test 1: Position X sweep ──────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Self position X sweep (no threats)")
    print("=" * 60)
    baseline_emb = None
    for x in [-10, -5, -2, 0, 2, 5, 10]:
        ent = make_entity(pos_x=x, pos_y=0)
        emb = get_pooled_embedding(model, *make_state(ent))
        if baseline_emb is None:
            baseline_emb = emb
            print(f"  x={x:+3d}: (baseline) norm={emb.norm():.4f}")
        else:
            print(f"  x={x:+3d}: cos={cosine_sim(emb, baseline_emb):.4f}  L2={l2_dist(emb, baseline_emb):.4f}")

    # ── Test 2: Distance from enemy sweep ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("TEST 2: Distance from enemy sweep (enemy at origin)")
    print("=" * 60)
    baseline_emb = None
    for d in [1, 2, 3, 4, 5, 6, 8, 10]:
        self_ent = make_entity(pos_x=d, pos_y=0, dist_from_caster=d)
        enemy_ent = make_entity(pos_x=0, pos_y=0, exists=1.0, hostile_zones=0)
        emb = get_pooled_embedding(model, *make_state(self_ent, enemy_ent))
        if baseline_emb is None:
            baseline_emb = emb
            print(f"  d={d:2d}: (baseline)")
        else:
            print(f"  d={d:2d}: cos={cosine_sim(emb, baseline_emb):.4f}  L2={l2_dist(emb, baseline_emb):.4f}")

    # ── Test 3: Threat position sweep ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TEST 3: Threat dx sweep (zone at varying offsets from unit)")
    print("=" * 60)
    baseline_emb = None
    for dx in [-5, -3, -1, 0, 1, 3, 5]:
        self_ent = make_entity(pos_x=0, pos_y=0, hostile_zones=1)
        threat = make_threat(dx=dx, dy=0, radius=3.0, damage_ratio=0.7)
        emb = get_pooled_embedding(model, *make_state(self_ent, threats=[threat]))
        if baseline_emb is None:
            baseline_emb = emb
            print(f"  dx={dx:+2d}: (baseline)")
        else:
            print(f"  dx={dx:+2d}: cos={cosine_sim(emb, baseline_emb):.4f}  L2={l2_dist(emb, baseline_emb):.4f}")

    # ── Test 4: No threat vs threat present ───────────────────────────
    print(f"\n{'=' * 60}")
    print("TEST 4: No threat vs threat present")
    print("=" * 60)
    self_ent = make_entity(pos_x=0, pos_y=0)
    no_threat_emb = get_pooled_embedding(model, *make_state(self_ent))

    self_ent_z = make_entity(pos_x=0, pos_y=0, hostile_zones=1)
    threat = make_threat(dx=0, dy=0, radius=3.0, damage_ratio=0.7)
    with_threat_emb = get_pooled_embedding(model, *make_state(self_ent_z, threats=[threat]))

    far_threat = make_threat(dx=8, dy=0, radius=3.0, damage_ratio=0.7)
    far_threat_emb = get_pooled_embedding(model, *make_state(self_ent_z, threats=[far_threat]))

    print(f"  no_threat vs in_zone:  cos={cosine_sim(no_threat_emb, with_threat_emb):.4f}  L2={l2_dist(no_threat_emb, with_threat_emb):.4f}")
    print(f"  no_threat vs far_zone: cos={cosine_sim(no_threat_emb, far_threat_emb):.4f}  L2={l2_dist(no_threat_emb, far_threat_emb):.4f}")
    print(f"  in_zone vs far_zone:   cos={cosine_sim(with_threat_emb, far_threat_emb):.4f}  L2={l2_dist(with_threat_emb, far_threat_emb):.4f}")

    # ── Test 5: Zone caster scenario — safe band vs danger ────────────
    print(f"\n{'=' * 60}")
    print("TEST 5: Zone caster scenario — position relative to danger")
    print("  Stationary enemy at (0,0) with r=3.0 zone")
    print("  Ranger attack range = 5.0")
    print("=" * 60)
    baseline_emb = None
    labels = {1: "IN_ZONE (lethal)", 2: "EDGE (risky)", 3: "CLOSE_SAFE",
              4: "MAX_RANGE (ideal)", 5: "SAFE_FAR", 7: "TOO_FAR", 10: "WAY_OUT"}
    for d in [1, 2, 3, 4, 5, 7, 10]:
        # Self at distance d from enemy
        in_zone = d < 3.0
        self_ent = make_entity(
            pos_x=d, pos_y=0, dist_from_caster=d,
            hostile_zones=1 if in_zone else 0,
        )
        enemy_ent = make_entity(pos_x=0, pos_y=0, exists=1.0, hp_pct=1.0)

        # Threat: zone centered on enemy (0,0), so dx=-d from self perspective
        threat = make_threat(dx=-d, dy=0, radius=3.0, damage_ratio=0.7 if in_zone else 0.0)
        threats = [threat] if in_zone else []

        emb = get_pooled_embedding(model, *make_state(self_ent, enemy_ent, threats=threats))
        label = labels.get(d, "")
        if baseline_emb is None:
            baseline_emb = emb
            print(f"  d={d:2d} {label:20s}: (baseline)")
        else:
            print(f"  d={d:2d} {label:20s}: cos={cosine_sim(emb, baseline_emb):.4f}  L2={l2_dist(emb, baseline_emb):.4f}")

    # ── Test 6: Value head predictions at different positions ─────────
    print(f"\n{'=' * 60}")
    print("TEST 6: Value head V(s) at different positions")
    print("  (Higher = model thinks this position is better)")
    print("=" * 60)
    for d in [1, 2, 3, 4, 5, 7, 10]:
        in_zone = d < 3.0
        self_ent = make_entity(
            pos_x=d, pos_y=0, dist_from_caster=d,
            hostile_zones=1 if in_zone else 0,
        )
        enemy_ent = make_entity(pos_x=0, pos_y=0, exists=1.0, hp_pct=1.0)
        threat = make_threat(dx=-d, dy=0, radius=3.0, damage_ratio=0.7 if in_zone else 0.0)
        threats = [threat] if in_zone else []

        state_args = make_state(self_ent, enemy_ent, threats=threats)
        with torch.no_grad():
            v = model.forward_value(*state_args).item()

        label = labels.get(d, "")
        print(f"  d={d:2d} {label:20s}: V(s) = {v:+.4f}")

    # ── Test 7: Policy logits at different positions ──────────────────
    print(f"\n{'=' * 60}")
    print("TEST 7: Policy action probabilities at different positions")
    print("  Actions: 0=atk_near 1=atk_weak 2=atk_focus 11=move_toward 12=move_away 13=hold")
    print("=" * 60)
    action_names = {0: "atk_near", 1: "atk_weak", 2: "atk_focus",
                    11: "mv_toward", 12: "mv_away", 13: "hold"}
    for d in [1, 2, 3, 4, 5, 7, 10]:
        in_zone = d < 3.0
        self_ent = make_entity(
            pos_x=d, pos_y=0, dist_from_caster=d,
            hostile_zones=1 if in_zone else 0,
        )
        enemy_ent = make_entity(pos_x=0, pos_y=0, exists=1.0, hp_pct=1.0)
        threat = make_threat(dx=-d, dy=0, radius=3.0, damage_ratio=0.7 if in_zone else 0.0)
        threats = [threat] if in_zone else []

        state_args = make_state(self_ent, enemy_ent, threats=threats)
        with torch.no_grad():
            logits, _ = model(*state_args, [None] * 8)
            # Mask ability slots (no abilities in this test)
            mask = torch.tensor([[True]*3 + [False]*8 + [True]*3], device=DEVICE)
            logits = logits.masked_fill(~mask, -1e9)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        label = labels.get(d, "")
        parts = []
        for a_idx, a_name in sorted(action_names.items()):
            parts.append(f"{a_name}={probs[a_idx]:.3f}")
        print(f"  d={d:2d} {label:20s}: {' '.join(parts)}")


if __name__ == "__main__":
    main()
