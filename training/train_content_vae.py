#!/usr/bin/env python3
"""Train grammar-guided VAE for content generation.

Architecture:
  Encoder: 124-dim input → MLP → μ, σ → z (latent)
  Decoder: z → factored heads for ability slots (142) + class slots (75)
  Serializer: deterministic slot→DSL (in Rust, not trained)

Loss: reconstruction (MSE on continuous slots, BCE on categorical slots) + β·KL

Usage:
    uv run --with numpy --with torch python3 training/train_content_vae.py \
        --data generated/vae_training_data.npz \
        --epochs 200 \
        --latent-dim 32 \
        --lr 1e-3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContentVAE(nn.Module):
    """Grammar-guided VAE with factored decoder heads."""

    def __init__(self, input_dim=124, latent_dim=32, hidden_dim=256,
                 ability_slot_dim=142, class_slot_dim=75):
        super().__init__()
        self.latent_dim = latent_dim
        self.ability_slot_dim = ability_slot_dim
        self.class_slot_dim = class_slot_dim

        # Encoder: input → hidden → μ, log_σ²
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Content type head: z → P(ability | class)
        self.content_type_head = nn.Linear(latent_dim, 2)

        # Ability decoder: z → ability slots
        self.ability_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ability_slot_dim),
        )

        # Class decoder: z → class slots
        self.class_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_slot_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        ability_slots = self.ability_decoder(z)
        class_slots = self.class_decoder(z)
        content_type_logits = self.content_type_head(z)
        return ability_slots, class_slots, content_type_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        ability_slots, class_slots, ct_logits = self.decode(z)
        return ability_slots, class_slots, ct_logits, mu, logvar


def vae_loss(ability_pred, class_pred, ct_logits, mu, logvar,
             ability_target, class_target, ct_target, beta=1.0):
    """Combined loss: reconstruction + KL divergence."""

    # Content type classification loss
    ct_loss = F.cross_entropy(ct_logits, ct_target)

    # Reconstruction loss: MSE on the relevant slots only
    # Mask by content type: abilities use ability slots, classes use class slots
    is_ability = (ct_target == 0).float().unsqueeze(1)
    is_class = (ct_target == 1).float().unsqueeze(1)

    ability_recon = F.mse_loss(ability_pred * is_ability, ability_target * is_ability, reduction='sum')
    class_recon = F.mse_loss(class_pred * is_class, class_target * is_class, reduction='sum')

    n = mu.size(0)
    recon_loss = (ability_recon + class_recon) / n

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n

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

    n = inputs.shape[0]
    print(f"Samples: {n:,} (abilities: {(content_types==0).sum():,}, classes: {(content_types==1).sum():,})")
    print(f"Input dim: {inputs.shape[1]}, Ability slots: {slots_ability.shape[1]}, Class slots: {slots_class.shape[1]}")

    # Train/val split
    perm = torch.randperm(n)
    val_n = max(1000, n // 10)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_ds = TensorDataset(inputs[train_idx], slots_ability[train_idx],
                             slots_class[train_idx], content_types[train_idx])
    val_ds = TensorDataset(inputs[val_idx], slots_ability[val_idx],
                           slots_class[val_idx], content_types[val_idx])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ContentVAE(
        input_dim=inputs.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        ability_slot_dim=slots_ability.shape[1],
        class_slot_dim=slots_class.shape[1],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Beta annealing: start at 0, linearly ramp to target over warmup epochs
    beta_target = args.beta
    beta_warmup = args.epochs // 5

    best_val_loss = float('inf')
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
            x, ab_target, cl_target, ct_target = [b.to(device) for b in batch]

            ab_pred, cl_pred, ct_logits, mu, logvar = model(x)
            loss, recon, kl, ct_loss = vae_loss(
                ab_pred, cl_pred, ct_logits, mu, logvar,
                ab_target, cl_target, ct_target, beta,
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
                x, ab_target, cl_target, ct_target = [b.to(device) for b in batch]
                ab_pred, cl_pred, ct_logits, mu, logvar = model(x)
                loss, recon, kl, ct_loss = vae_loss(
                    ab_pred, cl_pred, ct_logits, mu, logvar,
                    ab_target, cl_target, ct_target, beta,
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
                  f"val={val_total:.4f} recon={val_recon:.4f} ct_acc={val_ct_acc:.3f} "
                  f"β={beta:.3f} lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_total < best_val_loss:
            best_val_loss = val_total
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_dim": inputs.shape[1],
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "ability_slot_dim": slots_ability.shape[1],
                    "class_slot_dim": slots_class.shape[1],
                },
                "epoch": epoch,
                "val_loss": val_total,
            }, os.path.join(args.output_dir, "content_vae_best.pt"))

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
    parser.add_argument("--output-dir", default="generated/content_vae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (annealed from 0)")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
