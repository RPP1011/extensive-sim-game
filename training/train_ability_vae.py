#!/usr/bin/env python3
"""Train a VAE on the ability slot dataset to learn a latent ability space.

Usage:
    uv run --with torch --with numpy python training/train_ability_vae.py

The VAE learns to compress 142-dim ability slot vectors into a 32-dim
latent space. The latent space can then be used for:
- Ability editor (sliders on latent dims → decode to DSL)
- Interpolation between abilities
- Conditional generation (archetype-conditioned)
- Similarity search
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
LATENT_DIM = 64           # PCA shows need ~69 for 90% variance
HIDDEN_DIM = 256           # wider hidden for 64-dim latent
NUM_ARCHETYPES = 19
BATCH_SIZE = 256
EPOCHS = 300
LR = 3e-4                 # lower LR for stability
KL_WEIGHT_MAX = 0.005     # much lower β — prioritize reconstruction
KL_WARMUP_EPOCHS = 50     # slow warmup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STANDARDIZE = True         # per-dimension standardization

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AbilityVAE(nn.Module):
    """Conditional VAE for ability slot vectors.

    Encoder: slots (142) + archetype (19 one-hot) → hidden → μ, σ (64 each)
    Decoder: z (64) + archetype (19 one-hot) → hidden → slots (142)

    Uses 3 hidden layers for better nonlinear reconstruction.
    """

    def __init__(self):
        super().__init__()
        input_dim = SLOT_DIM + NUM_ARCHETYPES

        # Encoder (3 layers)
        self.enc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.enc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.enc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.enc_mu = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.enc_logvar = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)

        # Decoder (3 layers)
        decoder_input = LATENT_DIM + NUM_ARCHETYPES
        self.dec1 = nn.Linear(decoder_input, HIDDEN_DIM // 2)
        self.dec2 = nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM)
        self.dec3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dec_out = nn.Linear(HIDDEN_DIM, SLOT_DIM)

    def encode(self, x, archetype_onehot):
        h = torch.cat([x, archetype_onehot], dim=-1)
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, archetype_onehot):
        h = torch.cat([z, archetype_onehot], dim=-1)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        return self.dec_out(h)

    def forward(self, x, archetype_onehot):
        mu, logvar = self.encode(x, archetype_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, archetype_onehot)
        return recon, mu, logvar

    def sample(self, archetype_idx, n=1, device="cpu"):
        """Sample n abilities for a given archetype from the prior."""
        z = torch.randn(n, LATENT_DIM, device=device)
        onehot = torch.zeros(n, NUM_ARCHETYPES, device=device)
        onehot[:, archetype_idx] = 1.0
        with torch.no_grad():
            return self.decode(z, onehot)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(recon, target, mu, logvar, kl_weight):
    # Reconstruction: MSE (data is standardized so all dims equally weighted)
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # Sparsity penalty: penalize reconstruction being nonzero where target is zero
    # (in standardized space, zero = the mean, so check target near zero)
    zero_mask = (target.abs() < 0.1).float()
    sparsity_loss = (recon.abs() * zero_mask).mean() * 0.1

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + sparsity_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path):
    """Load ability dataset from npz file with per-dimension standardization."""
    data = np.load(path)
    slots_raw = data['slots']
    archetypes = torch.tensor(data['archetypes'], dtype=torch.long)
    levels = torch.tensor(data['levels'], dtype=torch.float32)

    if STANDARDIZE:
        # Per-dimension standardization so all effects get equal weight
        means = slots_raw.mean(axis=0)
        stds = slots_raw.std(axis=0)
        stds[stds < 1e-6] = 1.0  # avoid division by zero for constant dims
        slots_standardized = (slots_raw - means) / stds
        slots = torch.tensor(slots_standardized, dtype=torch.float32)
        # Save scaler params for inference
        scaler_params = {'means': means.tolist(), 'stds': stds.tolist()}
    else:
        slots = torch.tensor(slots_raw, dtype=torch.float32)
        scaler_params = None

    onehot = torch.zeros(len(archetypes), NUM_ARCHETYPES)
    onehot.scatter_(1, archetypes.unsqueeze(1), 1.0)

    return slots, onehot, levels, scaler_params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"=== Ability VAE Training ===")
    print(f"Device: {DEVICE}")
    print(f"Slot dim: {SLOT_DIM}, Latent dim: {LATENT_DIM}, Hidden: {HIDDEN_DIM}")
    print(f"Archetypes: {NUM_ARCHETYPES}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"KL weight: 0→{KL_WEIGHT_MAX} over {KL_WARMUP_EPOCHS} epochs")
    print()

    # Load data
    dataset_path = Path("generated/ability_dataset.npz")
    if not dataset_path.exists():
        print("ERROR: Dataset not found. Run:")
        print("  cargo run --bin xtask -- synth-abilities --count 100000 --seed 2026 > generated/ability_dataset_slots.jsonl")
        print("Then convert to npz with the analysis script.")
        sys.exit(1)

    slots, archetypes, levels, scaler_params = load_dataset(dataset_path)
    print(f"Loaded {len(slots)} abilities")
    if scaler_params:
        print(f"Standardized: per-dimension mean/std normalization")

    # Train/val split (90/10)
    n = len(slots)
    n_val = n // 10
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = TensorDataset(slots[train_idx], archetypes[train_idx])
    val_ds = TensorDataset(slots[val_idx], archetypes[val_idx])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print()

    # Model
    model = AbilityVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        # KL warmup
        kl_weight = min(KL_WEIGHT_MAX, KL_WEIGHT_MAX * epoch / max(KL_WARMUP_EPOCHS, 1))

        # Train
        model.train()
        train_total = 0
        train_recon = 0
        train_kl = 0
        train_steps = 0

        for batch_slots, batch_arch in train_dl:
            batch_slots = batch_slots.to(DEVICE)
            batch_arch = batch_arch.to(DEVICE)

            recon, mu, logvar = model(batch_slots, batch_arch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch_slots, mu, logvar, kl_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            train_steps += 1

        scheduler.step()

        # Validate
        model.eval()
        val_total = 0
        val_recon = 0
        val_kl = 0
        val_steps = 0

        with torch.no_grad():
            for batch_slots, batch_arch in val_dl:
                batch_slots = batch_slots.to(DEVICE)
                batch_arch = batch_arch.to(DEVICE)

                recon, mu, logvar = model(batch_slots, batch_arch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch_slots, mu, logvar, kl_weight)

                val_total += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                val_steps += 1

        avg_train = train_total / train_steps
        avg_val = val_total / val_steps
        avg_recon = val_recon / val_steps
        avg_kl = val_kl / val_steps
        dt = time.time() - t0

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), "generated/ability_vae_best.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{EPOCHS} | train={avg_train:.4f} val={avg_val:.4f} "
                  f"recon={avg_recon:.4f} kl={avg_kl:.4f} β={kl_weight:.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.1e} | {dt:.1f}s")

    print()
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

    # ---------------------------------------------------------------------------
    # Evaluation: sample and reconstruct
    # ---------------------------------------------------------------------------
    model.load_state_dict(torch.load("generated/ability_vae_best.pt", weights_only=True))
    model.eval()

    print()
    print("=== Reconstruction Quality ===")

    with torch.no_grad():
        # Take first 10 val samples
        sample_slots = slots[val_idx[:10]].to(DEVICE)
        sample_arch = archetypes[val_idx[:10]].to(DEVICE)

        recon, mu, logvar = model(sample_slots, sample_arch)
        mse = F.mse_loss(recon, sample_slots, reduction='none').mean(dim=1)

        for i in range(min(5, len(mse))):
            print(f"  Sample {i}: MSE={mse[i].item():.4f}, "
                  f"nonzero_in={int((sample_slots[i] != 0).sum())}, "
                  f"nonzero_out={int((recon[i].abs() > 0.1).sum())}")

    print()
    print("=== Latent Space Quality ===")

    with torch.no_grad():
        # Encode all val data
        all_mu = []
        for batch_slots, batch_arch in val_dl:
            mu, _ = model.encode(batch_slots.to(DEVICE), batch_arch.to(DEVICE))
            all_mu.append(mu.cpu())
        all_mu = torch.cat(all_mu, dim=0)

        print(f"  Latent μ stats: mean={all_mu.mean():.3f}, std={all_mu.std():.3f}")
        print(f"  Per-dim std: min={all_mu.std(dim=0).min():.3f}, max={all_mu.std(dim=0).max():.3f}")

        # Check if latent space is used (not collapsed)
        active_dims = (all_mu.std(dim=0) > 0.1).sum().item()
        print(f"  Active latent dims (std > 0.1): {active_dims}/{LATENT_DIM}")

    # Export weights as JSON for Rust inference
    export_path = "generated/ability_vae_weights.json"
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    # Include scaler params for de-standardization in the editor
    if scaler_params:
        weights['_scaler_means'] = scaler_params['means']
        weights['_scaler_stds'] = scaler_params['stds']
    weights['_latent_dim'] = LATENT_DIM
    weights['_slot_dim'] = SLOT_DIM
    with open(export_path, 'w') as f:
        json.dump(weights, f)
    print(f"\nExported weights to {export_path}")

    print("\nDone!")


if __name__ == "__main__":
    train()
