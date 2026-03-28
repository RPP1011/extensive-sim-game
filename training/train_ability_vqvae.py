#!/usr/bin/env python3
"""Train a VQ-VAE on the ability slot dataset.

VQ-VAE uses discrete codebook vectors instead of continuous latent space.
This naturally produces sparse, sharp outputs — each codebook entry
corresponds to a specific ability archetype rather than a blurry average.

Usage:
    uv run --with torch --with numpy python training/train_ability_vqvae.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SLOT_DIM = 142
NUM_CODES = 512           # codebook size — each code = one ability archetype
CODE_DIM = 64             # dimension of each codebook vector
NUM_ARCHETYPES = 19
HIDDEN_DIM = 256
BATCH_SIZE = 256
EPOCHS = 300
LR = 3e-4
COMMIT_COST = 0.25        # commitment loss weight (β in VQ-VAE paper)
CODEBOOK_EMA_DECAY = 0.99 # EMA update for codebook (more stable than gradient)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Vector Quantizer (EMA update, straight-through estimator)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, ema_decay=0.99, commit_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commit_cost = commit_cost
        self.ema_decay = ema_decay

        # Codebook: initialized from N(0, 1)
        self.register_buffer('codebook', torch.randn(num_codes, code_dim))
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', self.codebook.clone())

    def forward(self, z_e):
        """Quantize encoder output to nearest codebook vector.

        Args:
            z_e: [B, code_dim] encoder output

        Returns:
            z_q: [B, code_dim] quantized (straight-through)
            indices: [B] codebook indices
            vq_loss: commitment + codebook loss
        """
        # Find nearest codebook vector
        # distances: [B, num_codes]
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ self.codebook.T
            + self.codebook.pow(2).sum(dim=1, keepdim=True).T
        )
        indices = distances.argmin(dim=1)  # [B]

        # Quantize: look up codebook vectors
        z_q = self.codebook[indices]  # [B, code_dim]

        # EMA codebook update (no gradients through codebook)
        if self.training:
            with torch.no_grad():
                # Count how many inputs map to each code
                one_hot = F.one_hot(indices, self.num_codes).float()  # [B, K]
                counts = one_hot.sum(dim=0)  # [K]
                weights = one_hot.T @ z_e  # [K, D]

                self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
                self.ema_weight.mul_(self.ema_decay).add_(weights, alpha=1 - self.ema_decay)

                # Laplace smoothing to avoid dead codes
                n = self.ema_count.sum()
                smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.codebook.copy_(self.ema_weight / smoothed.unsqueeze(1))

        # Commitment loss: encourage encoder to commit to codebook
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, self.commit_cost * commitment_loss

    def lookup(self, indices):
        """Look up codebook vectors by index."""
        return self.codebook[indices]


# ---------------------------------------------------------------------------
# VQ-VAE Model
# ---------------------------------------------------------------------------

class AbilityVQVAE(nn.Module):
    """VQ-VAE for ability slot vectors.

    Encoder: slots + archetype → code_dim continuous
    Quantizer: continuous → nearest codebook vector (discrete)
    Decoder: codebook vector + archetype → slots
    """

    def __init__(self):
        super().__init__()
        input_dim = SLOT_DIM + NUM_ARCHETYPES

        # Encoder
        self.enc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.enc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.enc3 = nn.Linear(HIDDEN_DIM, CODE_DIM)

        # Vector quantizer
        self.vq = VectorQuantizer(NUM_CODES, CODE_DIM, CODEBOOK_EMA_DECAY, COMMIT_COST)

        # Decoder
        decoder_input = CODE_DIM + NUM_ARCHETYPES
        self.dec1 = nn.Linear(decoder_input, HIDDEN_DIM)
        self.dec2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dec3 = nn.Linear(HIDDEN_DIM, SLOT_DIM)

    def encode(self, x, archetype_onehot):
        h = torch.cat([x, archetype_onehot], dim=-1)
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        return self.enc3(h)  # no activation — raw code_dim vector

    def decode(self, z_q, archetype_onehot):
        h = torch.cat([z_q, archetype_onehot], dim=-1)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x, archetype_onehot):
        z_e = self.encode(x, archetype_onehot)
        z_q, indices, vq_loss = self.vq(z_e)
        recon = self.decode(z_q, archetype_onehot)
        return recon, indices, vq_loss

    def decode_from_index(self, indices, archetype_onehot):
        """Decode from codebook indices (for generation/editor)."""
        z_q = self.vq.lookup(indices)
        return self.decode(z_q, archetype_onehot)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"=== Ability VQ-VAE Training ===")
    print(f"Device: {DEVICE}")
    print(f"Slots: {SLOT_DIM}, Codes: {NUM_CODES}, Code dim: {CODE_DIM}")
    print(f"Hidden: {HIDDEN_DIM}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Commit cost: {COMMIT_COST}, EMA decay: {CODEBOOK_EMA_DECAY}")
    print()

    dataset_path = Path("generated/ability_dataset.npz")
    if not dataset_path.exists():
        print("ERROR: Dataset not found.")
        sys.exit(1)

    data = np.load(dataset_path)
    slots_raw = data['slots']
    archetypes = torch.tensor(data['archetypes'], dtype=torch.long)

    # Standardize
    means = slots_raw.mean(axis=0)
    stds = slots_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    slots = torch.tensor((slots_raw - means) / stds, dtype=torch.float32)

    onehot = torch.zeros(len(archetypes), NUM_ARCHETYPES)
    onehot.scatter_(1, archetypes.unsqueeze(1), 1.0)

    print(f"Loaded {len(slots)} abilities")

    # Split
    n = len(slots)
    n_val = n // 10
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_dl = DataLoader(
        TensorDataset(slots[train_idx], onehot[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(
        TensorDataset(slots[val_idx], onehot[val_idx]),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = AbilityVQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,} + {NUM_CODES * CODE_DIM:,} codebook = {params + NUM_CODES * CODE_DIM:,}")
    print()

    best_val = float('inf')

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_recon = 0
        train_vq = 0
        train_steps = 0
        code_usage = torch.zeros(NUM_CODES, device=DEVICE)

        for batch_slots, batch_arch in train_dl:
            batch_slots = batch_slots.to(DEVICE)
            batch_arch = batch_arch.to(DEVICE)

            recon, indices, vq_loss = model(batch_slots, batch_arch)
            recon_loss = F.mse_loss(recon, batch_slots)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_recon += recon_loss.item()
            train_vq += vq_loss.item()
            train_steps += 1

            # Track code usage
            for idx in indices:
                code_usage[idx] += 1

        scheduler.step()

        # Validate
        model.eval()
        val_recon = 0
        val_vq = 0
        val_steps = 0

        with torch.no_grad():
            for batch_slots, batch_arch in val_dl:
                batch_slots = batch_slots.to(DEVICE)
                batch_arch = batch_arch.to(DEVICE)

                recon, indices, vq_loss = model(batch_slots, batch_arch)
                recon_loss = F.mse_loss(recon, batch_slots)

                val_recon += recon_loss.item()
                val_vq += vq_loss.item()
                val_steps += 1

        avg_recon = val_recon / val_steps
        avg_vq = val_vq / val_steps
        avg_total = avg_recon + avg_vq
        active_codes = (code_usage > 0).sum().item()
        dt = time.time() - t0

        if avg_total < best_val:
            best_val = avg_total
            torch.save(model.state_dict(), "generated/ability_vqvae_best.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{EPOCHS} | recon={avg_recon:.4f} vq={avg_vq:.4f} "
                  f"total={avg_total:.4f} codes={active_codes}/{NUM_CODES} "
                  f"lr={scheduler.get_last_lr()[0]:.1e} | {dt:.1f}s")

    print()
    print(f"Best val loss: {best_val:.4f}")

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    model.load_state_dict(torch.load("generated/ability_vqvae_best.pt", weights_only=True))
    model.eval()

    print()
    print("=== Reconstruction Quality ===")
    with torch.no_grad():
        sample = slots[val_idx[:10]].to(DEVICE)
        sample_arch = onehot[val_idx[:10]].to(DEVICE)
        recon, indices, _ = model(sample, sample_arch)

        for i in range(5):
            orig = sample[i].cpu().numpy()
            rec = recon[i].cpu().numpy()
            mse = np.mean((orig - rec) ** 2)
            # Unstandardize to check effects
            orig_raw = orig * stds + means
            rec_raw = rec * stds + means
            print(f"  Sample {i}: MSE={mse:.4f}, code={indices[i].item()}")

            checks = [(29, 'damage', 200), (59, 'heal', 200), (60, 'shield', 200),
                       (61, 'slow', 100), (63, 'stun', 3000), (54, 'stealth', 5000), (84, 'dash', 8)]
            orig_fx = []
            rec_fx = []
            for slot, name, scale in checks:
                ov = orig_raw[slot] * scale
                rv = rec_raw[slot] * scale
                if abs(ov) > 5:
                    orig_fx.append(f'{name}={ov:.0f}')
                if abs(rv) > 5:
                    rec_fx.append(f'{name}={rv:.0f}')
            print(f"    orig: {orig_fx}")
            print(f"    recon: {rec_fx}")

    # Codebook analysis
    print()
    print("=== Codebook Analysis ===")
    with torch.no_grad():
        # Encode all training data and count code usage
        all_indices = []
        for batch_slots, batch_arch in train_dl:
            z_e = model.encode(batch_slots.to(DEVICE), batch_arch.to(DEVICE))
            _, indices, _ = model.vq(z_e)
            all_indices.extend(indices.cpu().tolist())

        from collections import Counter
        code_counts = Counter(all_indices)
        active = len(code_counts)
        print(f"  Active codes: {active}/{NUM_CODES} ({active/NUM_CODES*100:.0f}%)")
        top10 = code_counts.most_common(10)
        print(f"  Top 10 codes: {[(c, n) for c, n in top10]}")
        bottom = [c for c, n in code_counts.most_common() if n < 10]
        print(f"  Rare codes (<10 uses): {len(bottom)}")

        # Decode each of the top 10 codes and show what ability they represent
        print()
        print("=== Top 10 Code Archetypes ===")
        for code_idx, count in top10:
            # Decode with berserker archetype to see the ability
            arch_oh = torch.zeros(1, NUM_ARCHETYPES, device=DEVICE)
            arch_oh[0, 3] = 1.0  # berserker
            z_q = model.vq.codebook[code_idx:code_idx+1]
            decoded = model.decode(z_q, arch_oh)[0].cpu().numpy()
            decoded_raw = decoded * stds + means

            effects = []
            for slot, name, scale in checks:
                v = decoded_raw[slot] * scale
                if abs(v) > 5:
                    effects.append(f'{name}={v:.0f}')

            targets = ['enemy', 'ally', 'self', 'self_aoe', 'ground', 'direction']
            max_t = max(range(6), key=lambda j: decoded_raw[1 + j])
            target = targets[max_t]

            print(f"  Code {code_idx:>3} ({count:>5} uses): target={target:>10} | {effects}")

    # Export weights
    export_path = "generated/ability_vqvae_weights.json"
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    # Codebook (buffer, not parameter)
    weights['vq.codebook'] = model.vq.codebook.cpu().numpy().tolist()
    weights['_scaler_means'] = means.tolist()
    weights['_scaler_stds'] = stds.tolist()
    weights['_num_codes'] = NUM_CODES
    weights['_code_dim'] = CODE_DIM
    weights['_slot_dim'] = SLOT_DIM

    with open(export_path, 'w') as f:
        json.dump(weights, f)
    print(f"\nExported to {export_path}")
    print("Done!")


if __name__ == "__main__":
    train()
