#!/usr/bin/env python3
"""Probe CLS embeddings for compositional structure in ability DSL.

Tests whether the 32-dim transformer CLS embedding captures compositional
patterns that the flat 80-dim property vector (and autoencoder) cannot.

Compositional labels are extracted directly from DSL text:
- Conditional effects (when clauses)
- Multi-effect combinations (damage+heal, damage+cc, etc.)
- Delivery hooks (on_hit, on_arrival, on_complete)
- Scaling interactions (+ X% stat)
- Effect ordering / nesting depth

Usage:
    uv run --with numpy --with torch --with scikit-learn training/probe_compositionality.py \
        --checkpoint generated/ability_transformer_decision_v2.pt \
        --abilities generated/ability_dataset/ \
        [--max-abilities 10000]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerDecision
from tokenizer import AbilityTokenizer


# ---------------------------------------------------------------------------
# Compositional label extraction from raw DSL text
# ---------------------------------------------------------------------------

EFFECT_TYPES = {
    "damage", "heal", "shield", "stun", "slow", "root", "silence", "fear",
    "taunt", "knockback", "pull", "dash", "blink", "buff", "debuff", "duel",
    "summon", "dispel", "reflect", "lifesteal", "damage_modify", "self_damage",
    "execute", "blind", "resurrect", "overheal_shield", "absorb_to_heal",
    "shield_steal", "status_clone", "immunity", "detonate", "status_transfer",
    "death_mark", "polymorph", "banish", "confuse", "charm", "stealth",
    "leash", "link", "redirect", "rewind", "cooldown_modify", "apply_stacks",
    "obstacle", "suppress", "grounded", "projectile_block", "attach",
}

CC_EFFECTS = {"stun", "slow", "root", "silence", "fear", "taunt", "knockback",
              "pull", "polymorph", "banish", "confuse", "charm", "suppress",
              "grounded", "blind"}

DAMAGE_EFFECTS = {"damage", "self_damage", "execute", "detonate", "death_mark"}
HEAL_EFFECTS = {"heal", "lifesteal", "resurrect", "absorb_to_heal", "overheal_shield"}
MOBILITY_EFFECTS = {"dash", "blink", "knockback", "pull", "swap"}
DEFENSIVE_EFFECTS = {"shield", "immunity", "reflect", "projectile_block", "stealth"}


def extract_compositional_labels(dsl: str) -> dict[str, int] | None:
    """Extract compositional structure labels from ability DSL text.

    Returns None if the DSL cannot be parsed (skip this sample).
    """
    labels = {}

    # Strip comments
    lines = [l.split("//")[0].rstrip() for l in dsl.split("\n")]
    text = "\n".join(lines)

    # --- Effect inventory ---
    # Find all effect keywords that appear as effect lines (not in header)
    # Effects appear after the header block (after blank line or first effect)
    effect_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_body = True
            continue
        if in_body and stripped and not stripped.startswith("}"):
            effect_lines.append(stripped)

    found_effects = set()
    for line in effect_lines:
        first_word = line.split()[0] if line.split() else ""
        # Strip "when" prefix from conditional effects
        if first_word in EFFECT_TYPES:
            found_effects.add(first_word)
        # Also check deliver hooks
        for eff in EFFECT_TYPES:
            if re.search(rf'\b{eff}\b', line):
                found_effects.add(eff)

    has_damage = bool(found_effects & DAMAGE_EFFECTS)
    has_heal = bool(found_effects & HEAL_EFFECTS)
    has_cc = bool(found_effects & CC_EFFECTS)
    has_mobility = bool(found_effects & MOBILITY_EFFECTS)
    has_defensive = bool(found_effects & DEFENSIVE_EFFECTS)

    n_categories = sum([has_damage, has_heal, has_cc, has_mobility, has_defensive])

    # --- Compositionality indicators ---

    # 1. Has condition (when clause)
    labels["has_condition"] = int("when " in text)

    # 2. Has else branch
    labels["has_else"] = int(bool(re.search(r'\belse\b', text)))

    # 3. Number of distinct effect categories (0-5)
    labels["n_effect_categories"] = n_categories

    # 4. Multi-category combination classes
    if has_damage and has_heal:
        labels["combo_class"] = 1  # damage + heal
    elif has_damage and has_cc:
        labels["combo_class"] = 2  # damage + CC
    elif has_heal and has_cc:
        labels["combo_class"] = 3  # heal + CC
    elif has_damage and has_defensive:
        labels["combo_class"] = 4  # damage + defense
    elif n_categories >= 2:
        labels["combo_class"] = 5  # other multi
    elif has_damage:
        labels["combo_class"] = 6  # pure damage
    elif has_heal:
        labels["combo_class"] = 7  # pure heal
    elif has_cc:
        labels["combo_class"] = 8  # pure CC
    else:
        labels["combo_class"] = 0  # utility/other

    # 5. Has delivery mechanism (non-instant)
    labels["has_delivery"] = int(bool(re.search(r'\bdeliver\b', text)))

    # 6. Has delivery hooks (on_hit, on_arrival, on_complete)
    labels["has_hooks"] = int(bool(re.search(r'\bon_hit\b|\bon_arrival\b|\bon_complete\b', text)))

    # 7. Has scaling (+ X% stat)
    labels["has_scaling"] = int(bool(re.search(r'\+\s*\d+%\s*\w+', text)))

    # 8. Has area effect
    labels["has_area"] = int(bool(re.search(r'\bin\s+(circle|cone|line|ring|spread)\b', text)))

    # 9. Nesting depth (count brace depth)
    max_depth = 0
    depth = 0
    for ch in text:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth -= 1
    labels["nesting_depth"] = min(max_depth, 4)  # cap at 4

    # 10. Interaction class: condition + multi-effect (the truly compositional ones)
    labels["is_compositional"] = int(labels["has_condition"] and n_categories >= 2)

    # 11. Number of effect lines (raw count)
    n_effect_lines = sum(1 for line in effect_lines
                         if any(e in line for e in EFFECT_TYPES))
    labels["n_effect_lines"] = min(n_effect_lines, 6)  # cap

    return labels


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def load_transformer(checkpoint_path: str, tokenizer: AbilityTokenizer) -> AbilityTransformerDecision:
    """Load Phase 2 transformer from checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Infer architecture from state dict
    token_emb = state["transformer.token_emb.weight"]
    vocab_size, d_model = token_emb.shape
    pos_emb = state["transformer.pos_emb.weight"]
    max_seq_len = pos_emb.shape[0]

    # Count layers
    n_layers = 0
    while f"transformer.encoder.layers.{n_layers}.self_attn.in_proj_weight" in state:
        n_layers += 1

    # Infer n_heads and d_ff from weight shapes
    ff_w = state["transformer.encoder.layers.0.linear1.weight"]
    d_ff = ff_w.shape[0]
    attn_w = state["transformer.encoder.layers.0.self_attn.in_proj_weight"]
    # in_proj is [3*d_model, d_model], n_heads = d_model / head_dim
    # We can't infer n_heads from weights alone, but d_model=32, likely 4 heads
    n_heads = max(1, d_model // 8)  # head_dim=8

    model = AbilityTransformerDecision(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        game_state_dim=0,  # We only need the transformer encoder, not cross-attn
        n_targets=3,
    )

    # Load only transformer weights (ignore cross-attn/entity/decision head)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_cls_embeddings(
    model: AbilityTransformerDecision,
    tokenizer: AbilityTokenizer,
    dsl_texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Tokenize and encode abilities, return CLS embeddings."""
    all_cls = []
    for i in range(0, len(dsl_texts), batch_size):
        batch = dsl_texts[i:i + batch_size]
        encoded = [tokenizer.encode(t) for t in batch]
        max_len = max(len(e) for e in encoded)
        # Pad
        padded = [e + [tokenizer.pad_id] * (max_len - len(e)) for e in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        mask = (input_ids != tokenizer.pad_id).float()
        cls = model.transformer.cls_embedding(input_ids, mask)
        all_cls.append(cls.numpy())
    return np.concatenate(all_cls, axis=0)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def run_probes(embeddings: np.ndarray, labels: dict[str, np.ndarray], name: str):
    """Run linear probes and report results."""
    from sklearn.linear_model import LogisticRegression

    n = len(embeddings)
    # Random 80/20 split
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    print(f"\n=== Linear Probes: {name} ===")
    print(f"{'Probe':<24} {'Train':>6} {'Val':>6} {'Classes':>8} {'Majority':>9}")
    print("-" * 60)

    results = {}
    for probe_name, y in sorted(labels.items()):
        n_classes = len(np.unique(y))
        if n_classes < 2:
            continue

        X_tr, y_tr = embeddings[train_idx], y[train_idx]
        X_va, y_va = embeddings[val_idx], y[val_idx]

        # Majority baseline
        _, counts = np.unique(y_tr, return_counts=True)
        majority = counts.max() / counts.sum()

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        val_acc = clf.score(X_va, y_va)

        results[probe_name] = {
            "train": train_acc, "val": val_acc,
            "majority": majority, "n_classes": n_classes,
            "lift": val_acc - majority,
        }
        print(f"{probe_name:<24} {train_acc:>5.1%} {val_acc:>5.1%} "
              f"{n_classes:>8} {majority:>8.1%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Phase 2 transformer .pt")
    parser.add_argument("--abilities", required=True, help="Directory of .ability files")
    parser.add_argument("--max-abilities", type=int, default=10000)
    parser.add_argument("--encoder", default="generated/ability_encoder.json",
                        help="Autoencoder weights for comparison")
    parser.add_argument("--encoder-data", default="generated/ability_encoder_data.json",
                        help="Ability encoder data (for property extraction)")
    args = parser.parse_args()

    tokenizer = AbilityTokenizer()
    ability_dir = Path(args.abilities)

    # --- Load abilities ---
    print("Loading ability DSL files...")
    ability_files = sorted(ability_dir.glob("*.ability"))[:args.max_abilities]
    dsl_texts = []
    all_labels = []
    for f in ability_files:
        text = f.read_text()
        labels = extract_compositional_labels(text)
        if labels is None:
            continue
        dsl_texts.append(text)
        all_labels.append(labels)

    print(f"Loaded {len(dsl_texts)} abilities")

    # Convert labels to arrays
    label_names = list(all_labels[0].keys())
    label_arrays = {
        name: np.array([l[name] for l in all_labels])
        for name in label_names
    }

    # Print label distributions
    print("\n--- Label distributions ---")
    for name, arr in sorted(label_arrays.items()):
        vals, counts = np.unique(arr, return_counts=True)
        dist = ", ".join(f"{v}:{c}" for v, c in zip(vals, counts))
        print(f"  {name}: {dist}")

    # --- Extract transformer CLS embeddings ---
    print(f"\nLoading transformer from {args.checkpoint}...")
    model = load_transformer(args.checkpoint, tokenizer)
    print(f"Extracting CLS embeddings for {len(dsl_texts)} abilities...")
    cls_emb = extract_cls_embeddings(model, tokenizer, dsl_texts)
    print(f"CLS shape: {cls_emb.shape}")

    # --- Probe transformer CLS ---
    cls_results = run_probes(cls_emb, label_arrays, "Transformer CLS (d=32)")

    # --- For comparison: extract properties from DSL and run autoencoder ---
    # We approximate the 80-dim property vector from DSL text using simple regex
    # extraction. This won't be perfect but captures what the flat vector can represent.
    # The key insight: compositional labels that BOTH embeddings predict well
    # aren't truly compositional. Only labels where CLS >> autoencoder matter.
    encoder_path = Path(args.encoder)
    if encoder_path.exists():
        print("\n--- Autoencoder comparison ---")
        print("Extracting approximate property vectors from DSL...")
        props = extract_approx_properties(dsl_texts)
        print(f"Property vectors shape: {props.shape}")

        import json
        w = json.load(open(str(encoder_path)))
        enc = w["encoder"]
        w1 = np.array(enc["w1"])
        b1 = np.array(enc["b1"])
        w2 = np.array(enc["w2"])
        b2 = np.array(enc["b2"])
        h = np.maximum(0, props @ w1 + b1)
        ae_emb = h @ w2 + b2
        ae_emb = ae_emb / (np.linalg.norm(ae_emb, axis=1, keepdims=True) + 1e-8)

        ae_results = run_probes(ae_emb, label_arrays, "Autoencoder (80→32)")

        # Also probe raw 80-dim props as upper bound
        prop_results = run_probes(props, label_arrays, "Raw Props (80-dim, upper bound)")

        # --- Comparison ---
        print("\n" + "=" * 70)
        print("  COMPOSITIONALITY COMPARISON")
        print("=" * 70)
        print(f"{'Probe':<24} {'CLS':>6} {'AE':>6} {'Props':>6} {'CLS-AE':>7}")
        print("-" * 52)
        for name in sorted(cls_results.keys()):
            cls_v = cls_results[name]["val"]
            ae_v = ae_results.get(name, {}).get("val", float("nan"))
            pr_v = prop_results.get(name, {}).get("val", float("nan"))
            diff = cls_v - ae_v
            marker = " <<<" if diff > 0.05 else ""
            print(f"{name:<24} {cls_v:>5.1%} {ae_v:>5.1%} {pr_v:>5.1%} {diff:>+6.1%}{marker}")

        print("\n'<<<' = CLS significantly better (>5pp), indicates compositional info")
    else:
        print(f"\nAutoencoder weights not found at {encoder_path}, skipping comparison.")


def extract_approx_properties(dsl_texts: list[str]) -> np.ndarray:
    """Extract approximate 80-dim property vectors from DSL text.

    This replicates the Rust extract_ability_properties() logic in Python,
    reading header properties and summarizing effects from the DSL text.
    """
    props_list = []
    for dsl in dsl_texts:
        f = np.zeros(80, dtype=np.float32)

        # [0:8] Targeting one-hot
        tgt_map = {"enemy": 0, "ally": 1, "self": 2, "self_aoe": 3,
                    "ground": 4, "direction": 5, "vector": 6, "global": 7,
                    "target_enemy": 0, "target_ally": 1, "self_cast": 2,
                    "ground_target": 4}
        m = re.search(r'target:\s*(\w+)', dsl)
        if m:
            tgt = m.group(1)
            idx = tgt_map.get(tgt, 0)
            f[idx] = 1.0

        # [8] range/10
        m = re.search(r'range:\s*([\d.]+)', dsl)
        if m:
            f[8] = float(m.group(1)) / 10.0

        # [9] cooldown/20000
        m = re.search(r'cooldown:\s*(\d+)(ms|s)', dsl)
        if m:
            val = int(m.group(1))
            if m.group(2) == 's':
                val *= 1000
            f[9] = val / 20000.0

        # [10] cast_time/2000
        m = re.search(r'cast:\s*(\d+)(ms|s)', dsl)
        if m:
            val = int(m.group(1))
            if m.group(2) == 's':
                val *= 1000
            f[10] = val / 2000.0

        # [11] cost/30
        m = re.search(r'cost:\s*(\d+)', dsl)
        if m:
            f[11] = int(m.group(1)) / 30.0

        # [14:21] Delivery one-hot
        del_map = {"projectile": 1, "channel": 2, "zone": 3, "tether": 4,
                    "trap": 5, "chain": 6}
        m = re.search(r'deliver\s+(\w+)', dsl)
        if m:
            idx = del_map.get(m.group(1), 0)
            f[14 + idx] = 1.0
        else:
            f[14] = 1.0  # instant

        # [32:38] AI hint one-hot
        hint_map = {"damage": 0, "heal": 1, "cc": 2, "crowd_control": 2,
                     "defense": 3, "utility": 4}
        m = re.search(r'hint:\s*(\w+)', dsl)
        if m:
            idx = hint_map.get(m.group(1), 5)
            f[32 + idx] = 1.0

        # [41] has damage (total damage / 200)
        dmg_vals = re.findall(r'\bdamage\s+(\d+)', dsl)
        if dmg_vals:
            f[41] = sum(int(v) for v in dmg_vals) / 200.0

        # [45] has heal (total heal / 200)
        heal_vals = re.findall(r'\bheal\s+(\d+)', dsl)
        if heal_vals:
            f[45] = sum(int(v) for v in heal_vals) / 200.0

        # [48:63] CC effects — set flags for each CC type found
        cc_list = ["stun", "slow", "root", "silence", "fear", "taunt",
                    "knockback", "pull", "polymorph", "banish", "confuse",
                    "charm", "suppress", "grounded", "blind"]
        for ci, cc in enumerate(cc_list):
            if re.search(rf'\b{cc}\b', dsl):
                f[48 + ci] = 1.0

        # [74] AoE radius
        m = re.search(r'in\s+(?:circle|cone|line|ring|spread)\(([\d.]+)', dsl)
        if m:
            f[74] = float(m.group(1)) / 5.0

        props_list.append(f)

    return np.array(props_list)


if __name__ == "__main__":
    main()
