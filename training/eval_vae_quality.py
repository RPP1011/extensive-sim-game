#!/usr/bin/env python3
"""Evaluate VAE and VQ-VAE reconstruction quality using correct slot layout.

Slot layout (142 dims):
  [0:3]    output_type (active/passive/class)
  [3:11]   targeting (8 one-hot)
  [11:15]  range, cooldown, cast, cost
  [15:20]  hint (5 one-hot)
  [20:27]  delivery type (7 one-hot)
  [27:33]  delivery params (6)
  [33:37]  has_charges, has_toggle, has_recast, unstoppable
  [37:42]  charge/toggle/recast params (5)
  [42:67]  effect slot 0 (25 dims)
  [67:92]  effect slot 1 (25 dims)
  [92:117] effect slot 2 (25 dims)
  [117:142] effect slot 3 (25 dims)

Each effect slot (25 dims):
  [0:17]   effect type (17 one-hot)
  [17]     param / 155
  [18]     duration / 10000
  [19:24]  area (5 dims)
  [24]     condition flag

Effect types (17):
  0=damage, 1=heal, 2=shield, 3=dot, 4=hot, 5=slow, 6=root,
  7=stun, 8=silence, 9=knockback, 10=pull, 11=dash, 12=blink,
  13=stealth, 14=buff, 15=debuff, 16=summon

Usage:
    uv run --with torch --with numpy python training/eval_vae_quality.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Constants ---
SLOT_DIM = 142
EFFECT_OFFSET = 42
EFFECT_SLOT_DIM = 25
NUM_EFFECT_SLOTS = 4
NUM_EFFECT_TYPES = 17
NUM_ARCHETYPES = 19

EFFECT_NAMES = [
    "damage", "heal", "shield", "dot", "hot", "slow", "root",
    "stun", "silence", "knockback", "pull", "dash", "blink",
    "stealth", "buff", "debuff", "summon"
]

TARGETING_NAMES = [
    "enemy", "ally", "self", "self_aoe", "ground", "direction", "enemy_aoe", "ally_aoe"
]

DELIVERY_NAMES = [
    "instant", "projectile", "aoe", "cone", "line", "channel", "self_buff"
]


def decode_effect_type(slot_25):
    """Return (type_name, param, duration) from a 25-dim effect slot, or None if empty."""
    type_vec = slot_25[:NUM_EFFECT_TYPES]
    if type_vec.max() < 0.3:
        return None
    idx = type_vec.argmax()
    param = slot_25[17] * 155
    duration = slot_25[18] * 10000
    return EFFECT_NAMES[idx], param, duration


def decode_ability(raw_slots):
    """Decode a raw (unstandardized) slot vector into a human-readable dict."""
    result = {}
    # Targeting
    tgt = raw_slots[3:11]
    if tgt.max() > 0.3:
        result["targeting"] = TARGETING_NAMES[tgt.argmax()]
    # Range/CD/Cast/Cost
    result["range"] = raw_slots[11] * 10
    result["cooldown"] = raw_slots[12] * 30000
    result["cast_time"] = raw_slots[13] * 3000
    # Delivery
    deliv = raw_slots[20:27]
    if deliv.max() > 0.3:
        result["delivery"] = DELIVERY_NAMES[deliv.argmax()]
    # Effects
    effects = []
    for i in range(NUM_EFFECT_SLOTS):
        off = EFFECT_OFFSET + i * EFFECT_SLOT_DIM
        eff = decode_effect_type(raw_slots[off:off + EFFECT_SLOT_DIM])
        if eff:
            effects.append(eff)
    result["effects"] = effects
    return result


def effect_distribution(slots_raw, label="Data"):
    """Print effect type distribution across all abilities."""
    n = len(slots_raw)
    print(f"\n=== {label}: Effect Distribution ({n} abilities) ===")
    for slot_i in range(NUM_EFFECT_SLOTS):
        off = EFFECT_OFFSET + slot_i * EFFECT_SLOT_DIM
        counts = Counter()
        used = 0
        for row in slots_raw:
            eff = decode_effect_type(row[off:off + EFFECT_SLOT_DIM])
            if eff:
                counts[eff[0]] += 1
                used += 1
        pct_used = used / n * 100
        print(f"  Slot {slot_i} ({pct_used:.1f}% used):", end="")
        if used > 0:
            top = counts.most_common(5)
            parts = [f"{name}={c/n*100:.1f}%" for name, c in top]
            print(f" {', '.join(parts)}")
        else:
            print(" empty")


def categorical_accuracy(orig_raw, recon_raw, name=""):
    """Measure how often reconstructed one-hots match original."""
    n = len(orig_raw)
    results = {}

    # Targeting accuracy
    orig_tgt = orig_raw[:, 3:11].argmax(axis=1)
    recon_tgt = recon_raw[:, 3:11].argmax(axis=1)
    results["targeting"] = (orig_tgt == recon_tgt).mean()

    # Delivery accuracy
    orig_del = orig_raw[:, 20:27].argmax(axis=1)
    recon_del = recon_raw[:, 20:27].argmax(axis=1)
    results["delivery"] = (orig_del == recon_del).mean()

    # Effect type accuracy (per slot)
    for slot_i in range(NUM_EFFECT_SLOTS):
        off = EFFECT_OFFSET + slot_i * EFFECT_SLOT_DIM
        # Only measure where original has an effect
        orig_types = orig_raw[:, off:off + NUM_EFFECT_TYPES]
        recon_types = recon_raw[:, off:off + NUM_EFFECT_TYPES]
        has_effect = orig_types.max(axis=1) > 0.3
        if has_effect.sum() > 0:
            orig_idx = orig_types[has_effect].argmax(axis=1)
            recon_idx = recon_types[has_effect].argmax(axis=1)
            results[f"effect_{slot_i}_type"] = (orig_idx == recon_idx).mean()
            results[f"effect_{slot_i}_n"] = int(has_effect.sum())

    # Continuous fields: range, cooldown
    range_mae = np.abs(orig_raw[:, 11] - recon_raw[:, 11]).mean() * 10
    cd_mae = np.abs(orig_raw[:, 12] - recon_raw[:, 12]).mean() * 30000
    results["range_mae"] = range_mae
    results["cooldown_mae"] = cd_mae

    # Effect param MAE (slot 0)
    off0 = EFFECT_OFFSET
    has_eff0 = orig_raw[:, off0:off0 + NUM_EFFECT_TYPES].max(axis=1) > 0.3
    if has_eff0.sum() > 0:
        results["eff0_param_mae"] = np.abs(
            orig_raw[has_eff0, off0 + 17] - recon_raw[has_eff0, off0 + 17]
        ).mean() * 155

    print(f"\n=== {name}Categorical Accuracy ===")
    for k, v in results.items():
        if isinstance(v, float):
            if "mae" in k.lower():
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v*100:.1f}%")
        else:
            print(f"  {k}: {v}")


# --- Load Models ---

def load_vae(weights_path):
    """Load VAE from exported JSON weights."""
    with open(weights_path) as f:
        w = json.load(f)

    latent_dim = w['_latent_dim']
    slot_dim = w['_slot_dim']
    means = np.array(w['_scaler_means'])
    stds = np.array(w['_scaler_stds'])

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(slot_dim + NUM_ARCHETYPES, 256)
            self.enc2 = nn.Linear(256, 256)
            self.enc3 = nn.Linear(256, 128)
            self.enc_mu = nn.Linear(128, latent_dim)
            self.enc_logvar = nn.Linear(128, latent_dim)
            self.dec1 = nn.Linear(latent_dim + NUM_ARCHETYPES, 128)
            self.dec2 = nn.Linear(128, 256)
            self.dec3 = nn.Linear(256, 256)
            self.dec_out = nn.Linear(256, slot_dim)

        def encode(self, x, arch):
            h = torch.cat([x, arch], dim=-1)
            h = F.relu(self.enc1(h))
            h = F.relu(self.enc2(h))
            h = F.relu(self.enc3(h))
            return self.enc_mu(h), self.enc_logvar(h)

        def decode(self, z, arch):
            h = torch.cat([z, arch], dim=-1)
            h = F.relu(self.dec1(h))
            h = F.relu(self.dec2(h))
            h = F.relu(self.dec3(h))
            return self.dec_out(h)

        def forward(self, x, arch):
            mu, logvar = self.encode(x, arch)
            return self.decode(mu, arch), mu  # use mu for eval (no sampling)

    model = VAE()
    state = {}
    for name, param in model.named_parameters():
        if name in w:
            state[name] = torch.tensor(w[name], dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model, means, stds


def load_vqvae(weights_path):
    """Load VQ-VAE from exported JSON weights."""
    with open(weights_path) as f:
        w = json.load(f)

    num_codes = w['_num_codes']
    code_dim = w['_code_dim']
    slot_dim = w['_slot_dim']
    means = np.array(w['_scaler_means'])
    stds = np.array(w['_scaler_stds'])
    codebook = torch.tensor(w['vq.codebook'], dtype=torch.float32)

    class VQVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(slot_dim + NUM_ARCHETYPES, 256)
            self.enc2 = nn.Linear(256, 256)
            self.enc3 = nn.Linear(256, code_dim)
            self.dec1 = nn.Linear(code_dim + NUM_ARCHETYPES, 256)
            self.dec2 = nn.Linear(256, 256)
            self.dec3 = nn.Linear(256, slot_dim)
            self.register_buffer('codebook', codebook)

        def encode(self, x, arch):
            h = torch.cat([x, arch], dim=-1)
            h = F.relu(self.enc1(h))
            h = F.relu(self.enc2(h))
            return self.enc3(h)

        def quantize(self, z_e):
            dist = z_e.pow(2).sum(1, keepdim=True) - 2 * z_e @ self.codebook.T + self.codebook.pow(2).sum(1)
            indices = dist.argmin(dim=1)
            return self.codebook[indices], indices

        def decode(self, z_q, arch):
            h = torch.cat([z_q, arch], dim=-1)
            h = F.relu(self.dec1(h))
            h = F.relu(self.dec2(h))
            return self.dec3(h)

        def forward(self, x, arch):
            z_e = self.encode(x, arch)
            z_q, indices = self.quantize(z_e)
            return self.decode(z_q, arch), indices

    model = VQVAE()
    state = {}
    for name, param in model.named_parameters():
        if name in w:
            state[name] = torch.tensor(w[name], dtype=torch.float32)
    model.load_state_dict(state, strict=False)
    model.vq_codebook = codebook  # already set via buffer
    model.eval()
    return model, means, stds


def main():
    dataset_path = Path("generated/ability_dataset.npz")
    if not dataset_path.exists():
        print("ERROR: Dataset not found.")
        sys.exit(1)

    data = np.load(dataset_path)
    slots_raw = data['slots']
    archetypes = data['archetypes']
    print(f"Dataset: {len(slots_raw)} abilities, {SLOT_DIM} dims")

    # Show training data distribution
    effect_distribution(slots_raw, "Training Data")

    # Prepare tensors
    means = slots_raw.mean(axis=0)
    stds = slots_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    slots_std = (slots_raw - means) / stds
    slots_t = torch.tensor(slots_std, dtype=torch.float32)
    onehot = torch.zeros(len(archetypes), NUM_ARCHETYPES)
    onehot.scatter_(1, torch.tensor(archetypes, dtype=torch.long).unsqueeze(1), 1.0)

    # Use a fixed subset for evaluation
    np.random.seed(42)
    eval_idx = np.random.choice(len(slots_raw), size=min(2000, len(slots_raw)), replace=False)
    eval_slots = slots_t[eval_idx]
    eval_arch = onehot[eval_idx]
    eval_raw = slots_raw[eval_idx]

    # --- Evaluate VAE ---
    vae_path = Path("generated/ability_vae_weights.json")
    if vae_path.exists():
        print("\n" + "=" * 60)
        print("VAE Evaluation")
        print("=" * 60)
        model, m, s = load_vae(vae_path)
        with torch.no_grad():
            recon_std, _ = model(eval_slots, eval_arch)
            recon_raw = recon_std.numpy() * stds + means

        mse_std = F.mse_loss(recon_std, eval_slots).item()
        mse_raw = np.mean((recon_raw - eval_raw) ** 2)
        print(f"  MSE (standardized): {mse_std:.4f}")
        print(f"  MSE (raw): {mse_raw:.4f}")

        effect_distribution(recon_raw, "VAE Reconstructed")
        categorical_accuracy(eval_raw, recon_raw, "VAE ")

        # Show 5 examples
        print("\n=== VAE: 5 Sample Comparisons ===")
        for i in range(5):
            orig = decode_ability(eval_raw[i])
            rec = decode_ability(recon_raw[i])
            print(f"  Sample {i}:")
            print(f"    orig: {orig['effects']}, tgt={orig.get('targeting','?')}, del={orig.get('delivery','?')}")
            print(f"    recon: {rec['effects']}, tgt={rec.get('targeting','?')}, del={rec.get('delivery','?')}")

        # Sample from prior
        print("\n=== VAE: 10 Random Samples (from prior) ===")
        z_rand = torch.randn(10, model.enc_mu.out_features)
        arch_rand = torch.zeros(10, NUM_ARCHETYPES)
        for i in range(10):
            arch_rand[i, i % NUM_ARCHETYPES] = 1.0
        with torch.no_grad():
            sampled_std = model.decode(z_rand, arch_rand)
            sampled_raw = sampled_std.numpy() * stds + means
        for i in range(10):
            ab = decode_ability(sampled_raw[i])
            print(f"  Sample {i}: {ab['effects']}, tgt={ab.get('targeting','?')}, del={ab.get('delivery','?')}")
    else:
        print(f"\nVAE weights not found at {vae_path}")

    # --- Evaluate VQ-VAE ---
    vqvae_path = Path("generated/ability_vqvae_weights.json")
    if vqvae_path.exists():
        print("\n" + "=" * 60)
        print("VQ-VAE Evaluation")
        print("=" * 60)
        model, m, s = load_vqvae(vqvae_path)
        with torch.no_grad():
            recon_std, indices = model(eval_slots, eval_arch)
            recon_raw = recon_std.numpy() * stds + means

        mse_std = F.mse_loss(recon_std, eval_slots).item()
        mse_raw = np.mean((recon_raw - eval_raw) ** 2)
        print(f"  MSE (standardized): {mse_std:.4f}")
        print(f"  MSE (raw): {mse_raw:.4f}")

        from collections import Counter
        code_counts = Counter(indices.tolist())
        print(f"  Active codes: {len(code_counts)}/{model.codebook.shape[0]}")
        print(f"  Top 5 codes: {code_counts.most_common(5)}")

        effect_distribution(recon_raw, "VQ-VAE Reconstructed")
        categorical_accuracy(eval_raw, recon_raw, "VQ-VAE ")

        print("\n=== VQ-VAE: 5 Sample Comparisons ===")
        for i in range(5):
            orig = decode_ability(eval_raw[i])
            rec = decode_ability(recon_raw[i])
            print(f"  Sample {i} (code={indices[i].item()}):")
            print(f"    orig: {orig['effects']}, tgt={orig.get('targeting','?')}, del={orig.get('delivery','?')}")
            print(f"    recon: {rec['effects']}, tgt={rec.get('targeting','?')}, del={rec.get('delivery','?')}")
    else:
        print(f"\nVQ-VAE weights not found at {vqvae_path}")


if __name__ == "__main__":
    main()
