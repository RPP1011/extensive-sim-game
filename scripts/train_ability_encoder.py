#!/usr/bin/env python3
"""Train ability embedding autoencoder via SupCon + reconstruction.

Encoder maps 80-dim ability properties to 32-dim L2-normalized embeddings.
Decoder reconstructs properties from embeddings. Joint loss:
  - Supervised contrastive (category clustering)
  - Reconstruction MSE (preserve quantitative detail)

Usage:
    # 1. Export training data (Rust):
    cargo run --release --bin xtask -- scenario oracle ability-encoder-export \
        --output generated/ability_encoder_data.json

    # 2. Train:
    uv run --with numpy --with torch scripts/train_ability_encoder.py \
        generated/ability_encoder_data.json -o generated/ability_encoder.json

    # 3. Use in self-play (Rust loads encoder weights, decoder optional):
    cargo run --release --bin xtask -- scenario oracle self-play generate scenarios/ \
        --ability-encoder generated/ability_encoder.json --episodes 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROP_DIM = 80
EMBED_DIM = 32
NUM_CATEGORIES = 9
NUM_TARGETING = 8

CATEGORY_NAMES = [
    "damage_unit", "damage_aoe", "cc_unit", "heal_unit", "heal_aoe",
    "defense", "utility", "summon", "obstacle",
]


class AbilityAutoencoder(nn.Module):
    """Encoder-decoder for ability properties.

    Encoder:  80 -> hidden -> 32 (L2-normalized)
    Decoder:  32 -> hidden -> 80
    """

    def __init__(self, input_dim=PROP_DIM, hidden_dim=64, embed_dim=EMBED_DIM):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, embed_dim)
        # Decoder
        self.dec1 = nn.Linear(embed_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        z = self.enc2(h)
        return F.normalize(z, dim=-1)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return self.dec2(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def export_weights(self) -> dict:
        """Export encoder + decoder weights for Rust."""
        return {
            "encoder": {
                "w1": self.enc1.weight.cpu().T.tolist(),
                "b1": self.enc1.bias.cpu().tolist(),
                "w2": self.enc2.weight.cpu().T.tolist(),
                "b2": self.enc2.bias.cpu().tolist(),
            },
            "decoder": {
                "w1": self.dec1.weight.cpu().T.tolist(),
                "b1": self.dec1.bias.cpu().tolist(),
                "w2": self.dec2.weight.cpu().T.tolist(),
                "b2": self.dec2.bias.cpu().tolist(),
            },
        }


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss (SupCon).

    Anchors with no positives (singleton classes) are excluded.
    """
    batch_size = embeddings.size(0)
    device = embeddings.device

    sim = torch.mm(embeddings, embeddings.T) / temperature

    labels_col = labels.unsqueeze(0)
    labels_row = labels.unsqueeze(1)
    mask_pos = (labels_row == labels_col).float()
    mask_pos.fill_diagonal_(0)

    valid = mask_pos.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    mask_self = torch.eye(batch_size, device=device).bool()
    sim_masked = sim.masked_fill(mask_self, float('-inf'))

    log_sum_exp = torch.logsumexp(sim_masked, dim=1)

    pos_count = mask_pos.sum(dim=1).clamp(min=1)
    pos_sim_sum = (sim * mask_pos).sum(dim=1)
    mean_pos = pos_sim_sum / pos_count

    loss = -(mean_pos - log_sum_exp)
    return loss[valid].mean()


def load_data(path: str):
    with open(path) as f:
        data = json.load(f)

    rows = data["abilities"]
    properties = torch.tensor([r["properties"] for r in rows], dtype=torch.float32)
    categories = torch.tensor([r["category_index"] for r in rows], dtype=torch.long)
    targeting = torch.tensor([r["targeting_index"] for r in rows], dtype=torch.long)
    names = [(r["hero_name"], r["ability_name"]) for r in rows]

    return properties, categories, targeting, names


def train(properties, categories, targeting, hidden_dim=64, epochs=200,
          lr=1e-3, temperature=0.1, weight_decay=1e-4, recon_weight=1.0):
    model = AbilityAutoencoder(PROP_DIM, hidden_dim, EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    properties = properties.to(DEVICE)
    categories = categories.to(DEVICE)
    targeting = targeting.to(DEVICE)

    n = len(properties)
    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("enc"))
    dec_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("dec"))
    print(f"Training on {n} abilities")
    print(f"Categories: {dict(zip(CATEGORY_NAMES, [int((categories == i).sum()) for i in range(NUM_CATEGORIES)]))}")
    print(f"Autoencoder: {PROP_DIM} -> {hidden_dim} -> {EMBED_DIM} -> {hidden_dim} -> {PROP_DIM}")
    print(f"  Encoder: {enc_params:,} params  |  Decoder: {dec_params:,} params  |  Total: {n_params:,}")
    print(f"  recon_weight={recon_weight}, temperature={temperature}")
    print(f"Device: {DEVICE}")
    print()

    combined_labels = categories * NUM_TARGETING + targeting

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()

        embeddings, reconstructed = model(properties)

        # SupCon losses
        loss_cat = supervised_contrastive_loss(embeddings, categories, temperature)
        loss_combined = supervised_contrastive_loss(embeddings, combined_labels, temperature)
        loss_supcon = 0.7 * loss_cat + 0.3 * loss_combined

        # Reconstruction loss
        loss_recon = F.mse_loss(reconstructed, properties)

        loss = loss_supcon + recon_weight * loss_recon

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(properties)
                sim = torch.mm(z, z.T)
                sim.fill_diagonal_(-1)
                _, topk = sim.topk(5, dim=1)
                topk_labels = categories[topk]
                pred = topk_labels.mode(dim=1).values
                acc = (pred == categories).float().mean().item()

            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"loss={loss.item():.4f} "
                f"(supcon={loss_supcon.item():.4f} recon={loss_recon.item():.4f})  "
                f"kNN@5={acc:.1%}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}"
            )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        z, x_hat = model(properties)
        sim = torch.mm(z, z.T)
        sim.fill_diagonal_(-1)

        for k in [1, 3, 5]:
            _, topk = sim.topk(k, dim=1)
            topk_labels = categories[topk]
            pred = topk_labels.mode(dim=1).values
            acc = (pred == categories).float().mean().item()
            print(f"  kNN@{k} accuracy: {acc:.1%}")

        # Per-category accuracy
        print("\n  Per-category kNN@5:")
        _, topk5 = sim.topk(5, dim=1)
        topk5_labels = categories[topk5]
        pred5 = topk5_labels.mode(dim=1).values
        for ci, name in enumerate(CATEGORY_NAMES):
            mask = categories == ci
            if mask.sum() > 0:
                cat_acc = (pred5[mask] == ci).float().mean().item()
                print(f"    {name:15s}: {cat_acc:.1%} ({int(mask.sum())} abilities)")

        # Reconstruction quality
        recon_mse = F.mse_loss(x_hat, properties).item()
        # Per-feature group reconstruction error
        feature_groups = [
            ("targeting",   0,  8),
            ("core",        8,  14),
            ("delivery",    14, 27),
            ("mechanics",   27, 32),
            ("ai_hint",     32, 38),
            ("dmg_type",    38, 41),
            ("damage",      41, 45),
            ("healing",     45, 48),
            ("hard_cc",     48, 55),
            ("soft_cc",     55, 59),
            ("other_cc",    59, 63),
            ("mobility",    63, 66),
            ("buffs",       66, 70),
            ("area",        70, 75),
            ("special",     75, 80),
        ]
        print(f"\n  Reconstruction MSE: {recon_mse:.6f}")
        print("  Per-group reconstruction MSE:")
        for name, lo, hi in feature_groups:
            group_mse = F.mse_loss(x_hat[:, lo:hi], properties[:, lo:hi]).item()
            print(f"    {name:15s}: {group_mse:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ability autoencoder (SupCon + reconstruction)")
    parser.add_argument("data", help="Path to ability encoder data JSON")
    parser.add_argument("--output", "-o", default="generated/ability_encoder.json")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--recon-weight", type=float, default=1.0,
                        help="Weight for reconstruction loss relative to SupCon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data}...")
    properties, categories, targeting, names = load_data(args.data)

    model = train(
        properties, categories, targeting,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        recon_weight=args.recon_weight,
    )

    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    weights = model.export_weights()
    with open(args.output, "w") as f:
        json.dump(weights, f)
    print(f"\nAutoencoder saved to {args.output}")

    # Save embeddings for visualization
    model.eval()
    with torch.no_grad():
        z = model.encode(properties.to(DEVICE)).cpu().numpy()

    embed_path = args.output.replace(".json", "_embeddings.json")
    embed_data = []
    for idx, (hero, abi) in enumerate(names):
        embed_data.append({
            "hero": hero,
            "ability": abi,
            "category": CATEGORY_NAMES[int(categories[idx])],
            "embedding": z[idx].tolist(),
        })
    with open(embed_path, "w") as f:
        json.dump(embed_data, f, indent=1)
    print(f"Embeddings saved to {embed_path}")


if __name__ == "__main__":
    main()
