#!/usr/bin/env python3
"""Train grammar-guided VAE for content generation.

Architecture (v2):
  Encoder: 124-dim → 4-layer residual MLP → μ, σ → z (latent)
  Decoder: z → 4-layer factored heads for ability slots (142) + class slots (75)
  Free-bits KL: minimum λ per latent dimension to prevent posterior collapse
  Serializer: deterministic slot→DSL (in Rust, not trained)

Usage:
    uv run --with numpy --with torch --find-links https://download.pytorch.org/whl/cu124 \
        python3 training/train_content_vae.py \
        --data generated/vae_training_data.npz \
        --class-data generated/vae_training_class.npz \
        --epochs 200 --latent-dim 32 --hidden-dim 512 --lr 3e-4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual MLP block with LayerNorm."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ContentVAE(nn.Module):
    """Conditional VAE with input skip-connections to decoder.

    The decoder receives concat(z, x) so it can use the input context directly.
    This prevents mode collapse where the decoder ignores z and the input.
    z captures the "style" (which of the valid abilities for this context),
    while x provides the "what kind" (archetype, level, trigger).
    """

    def __init__(self, input_dim=124, latent_dim=32, hidden_dim=512,
                 ability_slot_dim=142, class_slot_dim=75,
                 n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ability_slot_dim = ability_slot_dim
        self.class_slot_dim = class_slot_dim

        # Encoder: input → proj → N residual blocks → μ, log_σ²
        self.enc_proj = nn.Linear(input_dim, hidden_dim)
        self.enc_blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Content type head: concat(z, x) → P(ability | class)
        dec_input_dim = latent_dim + input_dim  # z + x skip connection
        self.content_type_head = nn.Sequential(
            nn.Linear(dec_input_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        # Ability decoder: concat(z, x) → proj → N residual blocks → slots
        self.ab_proj = nn.Linear(dec_input_dim, hidden_dim)
        self.ab_blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.ab_norm = nn.LayerNorm(hidden_dim)
        self.ab_head = nn.Linear(hidden_dim, ability_slot_dim)

        # Class decoder: concat(z, x) → proj → N residual blocks → slots
        self.cl_proj = nn.Linear(dec_input_dim, hidden_dim)
        self.cl_blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.cl_norm = nn.LayerNorm(hidden_dim)
        self.cl_head = nn.Linear(hidden_dim, class_slot_dim)

    def encode(self, x):
        h = F.gelu(self.enc_proj(x))
        h = self.enc_blocks(h)
        h = self.enc_norm(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x):
        # Concatenate latent + input for conditional decoding
        zx = torch.cat([z, x], dim=-1)

        # Ability head
        ha = F.gelu(self.ab_proj(zx))
        ha = self.ab_blocks(ha)
        ha = self.ab_norm(ha)
        ability_slots = self.ab_head(ha)

        # Class head
        hc = F.gelu(self.cl_proj(zx))
        hc = self.cl_blocks(hc)
        hc = self.cl_norm(hc)
        class_slots = self.cl_head(hc)

        ct_logits = self.content_type_head(zx)
        return ability_slots, class_slots, ct_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        ability_slots, class_slots, ct_logits = self.decode(z, x)
        return ability_slots, class_slots, ct_logits, mu, logvar


def free_bits_kl(mu, logvar, free_bits=0.1):
    """KL with free-bits: each latent dim must contribute at least λ nats.
    Prevents posterior collapse by ensuring the encoder uses the latent space."""
    # Per-dimension KL: 0.5 * (μ² + σ² - 1 - log σ²)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (batch, latent)
    # Clamp each dim's mean KL to at least free_bits
    kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (latent,)
    kl_clamped = torch.clamp(kl_per_dim_mean, min=free_bits)
    return kl_clamped.sum()


# Categorical slot ranges in the 142-dim ability vector
# These use one-hot encoding and need cross-entropy, not MSE
ABILITY_CATEGORICAL_RANGES = [
    (0, 3),     # output_type: active/passive/class
    (3, 11),    # targeting: 8 types
    (15, 20),   # hint: 5 types
    (20, 27),   # delivery type: 7 types
]
# Per-effect categorical ranges (4 effects × 25 dims starting at 42)
for eff_i in range(4):
    base = 42 + eff_i * 25
    ABILITY_CATEGORICAL_RANGES.append((base, base + 17))       # effect type: 17 categories
    ABILITY_CATEGORICAL_RANGES.append((base + 19, base + 24))  # area shape: 5 types

# Class categorical ranges in the 75-dim class vector
CLASS_CATEGORICAL_RANGES = [
    (5, 21),    # tags multi-hot (16)  — keep as MSE/BCE since multi-hot
    (21, 32),   # scaling source: 11 types
]


def vae_loss(ability_pred, class_pred, ct_logits, mu, logvar,
             ability_target, class_target, ct_target,
             beta=1.0, free_bits=0.1):
    """Combined loss with CE on categoricals, MSE on continuous slots."""
    n = mu.size(0)
    ct_loss = F.cross_entropy(ct_logits, ct_target)

    is_ability = (ct_target == 0).float()
    is_class = (ct_target == 1).float()

    # --- Ability loss: CE on categoricals + MSE on continuous ---
    ab_ce_loss = torch.tensor(0.0, device=mu.device)
    ab_mse_loss = torch.tensor(0.0, device=mu.device)
    n_ab = is_ability.sum().clamp(min=1)

    if n_ab > 0:
        ab_mask = is_ability.bool()
        ab_p = ability_pred[ab_mask]
        ab_t = ability_target[ab_mask]

        # Cross-entropy on each categorical range with label smoothing
        categorical_dims = set()
        for start, end in ABILITY_CATEGORICAL_RANGES:
            target_idx = ab_t[:, start:end].argmax(dim=1)
            logits = ab_p[:, start:end]
            ab_ce_loss = ab_ce_loss + F.cross_entropy(logits, target_idx, label_smoothing=0.1)
            for d in range(start, end):
                categorical_dims.add(d)

        # MSE on continuous dims only
        continuous_mask = torch.ones(ability_pred.shape[1], dtype=torch.bool, device=mu.device)
        for d in categorical_dims:
            continuous_mask[d] = False
        ab_mse_loss = F.mse_loss(ab_p[:, continuous_mask], ab_t[:, continuous_mask])

    # --- Class loss: MSE (class slots are mostly continuous) ---
    cl_loss = torch.tensor(0.0, device=mu.device)
    n_cl = is_class.sum().clamp(min=1)
    if n_cl > 0:
        cl_mask = is_class.bool()
        cl_p = class_pred[cl_mask]
        cl_t = class_target[cl_mask]

        # CE on scaling source
        for start, end in CLASS_CATEGORICAL_RANGES:
            if end <= cl_t.shape[1]:
                target_idx = cl_t[:, start:end].argmax(dim=1)
                logits = cl_p[:, start:end]
                cl_loss = cl_loss + F.cross_entropy(logits, target_idx)

        # MSE on all class dims (categoricals are also fine with MSE for multi-hot)
        cl_loss = cl_loss + F.mse_loss(cl_p, cl_t)

    recon_loss = ab_ce_loss + ab_mse_loss * 10.0 + cl_loss  # weight MSE to balance with CE
    kl_loss = free_bits_kl(mu, logvar, free_bits)

    total = recon_loss + ct_loss + beta * kl_loss
    return total, recon_loss, kl_loss, ct_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"Loading {args.data}...")
    data = np.load(args.data)
    inputs = torch.from_numpy(data["inputs"])
    slots_ability = torch.from_numpy(data["slots_ability"])
    slots_class = torch.from_numpy(data["slots_class"])
    content_types = torch.from_numpy(data["content_types"]).long()

    # Optionally load and merge class data
    if args.class_data and os.path.exists(args.class_data):
        print(f"Loading class data from {args.class_data}...")
        cdata = np.load(args.class_data)
        inputs = torch.cat([inputs, torch.from_numpy(cdata["inputs"])], 0)
        slots_ability = torch.cat([slots_ability, torch.from_numpy(cdata["slots_ability"])], 0)
        slots_class = torch.cat([slots_class, torch.from_numpy(cdata["slots_class"])], 0)
        content_types = torch.cat([content_types, torch.from_numpy(cdata["content_types"]).long()], 0)

    n = inputs.shape[0]
    n_ab = (content_types == 0).sum().item()
    n_cl = (content_types == 1).sum().item()
    print(f"Samples: {n:,} (abilities: {n_ab:,}, classes: {n_cl:,})")
    print(f"Input dim: {inputs.shape[1]}, Ability slots: {slots_ability.shape[1]}, Class slots: {slots_class.shape[1]}")

    # Drop dead slots for better gradient signal
    ab_std = slots_ability.std(dim=0)
    cl_std = slots_class.std(dim=0)
    ab_active = (ab_std > 0.001).sum().item()
    cl_active = (cl_std > 0.001).sum().item()
    print(f"Active ability slots: {ab_active}/{slots_ability.shape[1]}, class slots: {cl_active}/{slots_class.shape[1]}")

    # Train/val split
    perm = torch.randperm(n)
    val_n = max(2000, n // 10)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_ds = TensorDataset(inputs[train_idx], slots_ability[train_idx],
                             slots_class[train_idx], content_types[train_idx])
    val_ds = TensorDataset(inputs[val_idx], slots_ability[val_idx],
                           slots_class[val_idx], content_types[val_idx])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ContentVAE(
        input_dim=inputs.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        ability_slot_dim=slots_ability.shape[1],
        class_slot_dim=slots_class.shape[1],
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({args.layers} layers, hidden={args.hidden_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Beta annealing
    beta_target = args.beta
    beta_warmup = args.epochs // 5

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        beta = min(beta_target, beta_target * epoch / max(1, beta_warmup))

        # Train
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_ct = 0.0
        train_batches = 0

        for batch in train_dl:
            x, ab_target, cl_target, ct_target = [b.to(device, non_blocking=True) for b in batch]

            ab_pred, cl_pred, ct_logits, mu, logvar = model(x)
            loss, recon, kl, ct_loss = vae_loss(
                ab_pred, cl_pred, ct_logits, mu, logvar,
                ab_target, cl_target, ct_target,
                beta, args.free_bits,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
            train_ct += ct_loss.item()
            train_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_ct_correct = 0
        val_ct_total = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                x, ab_target, cl_target, ct_target = [b.to(device, non_blocking=True) for b in batch]
                ab_pred, cl_pred, ct_logits, mu, logvar = model(x)
                loss, recon, kl, ct_loss = vae_loss(
                    ab_pred, cl_pred, ct_logits, mu, logvar,
                    ab_target, cl_target, ct_target,
                    beta, args.free_bits,
                )
                val_total += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
                val_ct_correct += (ct_logits.argmax(1) == ct_target).sum().item()
                val_ct_total += ct_target.size(0)
                val_batches += 1

        train_total /= max(1, train_batches)
        train_recon /= max(1, train_batches)
        train_kl /= max(1, train_batches)
        train_ct /= max(1, train_batches)
        val_total /= max(1, val_batches)
        val_recon /= max(1, val_batches)
        val_kl /= max(1, val_batches)
        val_ct_acc = val_ct_correct / max(1, val_ct_total)

        history.append({
            "epoch": epoch, "beta": beta,
            "train_loss": train_total, "train_recon": train_recon,
            "train_kl": train_kl, "train_ct": train_ct,
            "val_loss": val_total, "val_recon": val_recon,
            "val_kl": val_kl, "val_ct_acc": val_ct_acc,
        })

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"[{epoch:4d}/{args.epochs}] "
                  f"loss={train_total:.4f} recon={train_recon:.4f} kl={train_kl:.4f} "
                  f"ct={train_ct:.4f} | "
                  f"val={val_total:.4f} recon={val_recon:.4f} kl={val_kl:.4f} ct_acc={val_ct_acc:.3f} "
                  f"β={beta:.3f} lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best + early stopping
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_dim": inputs.shape[1],
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "ability_slot_dim": slots_ability.shape[1],
                    "class_slot_dim": slots_class.shape[1],
                    "n_layers": args.layers,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "val_loss": val_total,
            }, os.path.join(args.output_dir, "content_vae_best.pt"))
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Save final + history
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "input_dim": inputs.shape[1],
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "ability_slot_dim": slots_ability.shape[1],
            "class_slot_dim": slots_class.shape[1],
            "n_layers": args.layers,
            "dropout": args.dropout,
        },
        "epoch": args.epochs,
        "val_loss": val_total,
    }, os.path.join(args.output_dir, "content_vae_final.pt"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train content generation VAE")
    parser.add_argument("--data", default="generated/vae_training_data.npz")
    parser.add_argument("--class-data", default=None, help="Additional NPZ with class samples")
    parser.add_argument("--output-dir", default="generated/content_vae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5, help="KL weight (annealed from 0)")
    parser.add_argument("--free-bits", type=float, default=0.1, help="Min KL per latent dim (nats)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (0=disabled)")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
