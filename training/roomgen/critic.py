"""PCGRL Critic: learned tactical quality predictor on ELIT latent tokens.

Operates on the fixed-size latent interface (resolution-independent).
Provides a guidance signal during diffusion sampling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCGRLCritic(nn.Module):
    """Small transformer that predicts tactical quality from latent tokens.

    Input: latent tokens (K, d_model) from ELIT Read output
    Output: scalar quality score in [0, 1]
    """

    def __init__(
        self,
        d_model: int = 256,
        d_critic: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
    ):
        super().__init__()
        self.d_critic = d_critic

        # Project from ELIT latent dim to critic dim
        self.input_proj = nn.Linear(d_model, d_critic)

        # Timestep embedding
        self.time_proj = nn.Sequential(
            nn.Linear(d_critic, d_critic),
            nn.SiLU(),
            nn.Linear(d_critic, d_critic),
        )

        # Dimension conditioning
        self.dim_proj = nn.Linear(2, d_critic)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=d_critic,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ))

        # Output head
        self.output_norm = nn.LayerNorm(d_critic)
        self.output_head = nn.Sequential(
            nn.Linear(d_critic, d_critic),
            nn.GELU(),
            nn.Linear(d_critic, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        width: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        latent: (B, K, d_model) — ELIT latent tokens
        t: (B,) — diffusion timestep
        width: (B,) — room width
        depth: (B,) — room depth
        Returns: (B,) — quality score in [0, 1]
        """
        B, K, _ = latent.shape

        # Project to critic dimension
        x = self.input_proj(latent)  # (B, K, d_critic)

        # Add timestep conditioning
        half = self.d_critic // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        t_emb = torch.cat([
            (t.unsqueeze(-1) * freqs * 1000).sin(),
            (t.unsqueeze(-1) * freqs * 1000).cos(),
        ], dim=-1)
        t_cond = self.time_proj(t_emb)  # (B, d_critic)
        x = x + t_cond.unsqueeze(1)

        # Add dimension conditioning
        dims = torch.stack([width.float(), depth.float()], dim=-1)
        dim_cond = self.dim_proj(dims)  # (B, d_critic)
        x = x + dim_cond.unsqueeze(1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Mean pool → scalar
        x = self.output_norm(x.mean(dim=1))  # (B, d_critic)
        quality = self.output_head(x).squeeze(-1)  # (B,)
        return quality


def compute_quality_score(
    obs_type: torch.Tensor,
    height: torch.Tensor,
    elevation: torch.Tensor,
    width: int,
    depth: int,
    connected: bool,
    target_blocked: float = 0.15,
    target_cover: float = 0.3,
    target_chokepoints: int = 3,
) -> float:
    """Compute ground-truth tactical quality score for a room.

    Used to generate training labels for the critic.
    """
    # Blocked percentage
    interior = obs_type[1:depth-1, 1:width-1]
    total = interior.numel()
    blocked = (interior > 0).sum().item()
    blocked_pct = blocked / max(total, 1)

    # Cover density
    walkable_mask = (obs_type == 0)
    blocked_mask = (obs_type > 0).float()
    near_blocked = F.max_pool2d(
        blocked_mask.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=3, padding=1, stride=1,
    ).squeeze()
    near_cover = (walkable_mask.float() * near_blocked)
    cover_density = near_cover.sum().item() / walkable_mask.sum().item() if walkable_mask.sum() > 0 else 0

    # Chokepoint approximation
    chokepoints = 0
    for r in range(1, depth - 1):
        for c in range(1, width - 1):
            if obs_type[r, c] != 0:
                continue
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < depth and 0 <= nc < width and obs_type[nr, nc] == 0:
                    neighbors.append((dr, dc))
            if len(neighbors) == 2:
                d1, d2 = neighbors
                if d1[0] + d2[0] == 0 and d1[1] + d2[1] == 0:
                    chokepoints += 1

    # Spawn quality: check if both sides have cover
    left_cover = near_cover[1:depth-1, 1:width//4].sum().item()
    right_cover = near_cover[1:depth-1, 3*width//4:width-1].sum().item()
    spawn_diff = abs(left_cover - right_cover) / max(left_cover + right_cover, 1)

    # Composite quality score
    quality = (
        (1.0 if connected else 0.0)
        * max(0, min(1, 1 - abs(blocked_pct - target_blocked) / 0.1))
        * max(0, min(1, 1 - abs(cover_density - target_cover) / 0.15))
        * max(0, min(1, 1 - abs(chokepoints - target_chokepoints) / 2))
        * max(0, min(1, 1 - spawn_diff / 3.0))
    )

    return quality


class CriticTrainer:
    """Training loop for the PCGRL critic."""

    def __init__(self, critic: PCGRLCritic, lr: float = 3e-4, device: str = "cpu"):
        self.critic = critic.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=0.01)

    def train_step(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        width: torch.Tensor,
        depth: torch.Tensor,
        quality: torch.Tensor,
    ) -> float:
        """Single training step. Returns MSE loss."""
        self.critic.train()
        pred = self.critic(
            latent.to(self.device),
            t.to(self.device),
            width.to(self.device),
            depth.to(self.device),
        )
        loss = F.mse_loss(pred, quality.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
