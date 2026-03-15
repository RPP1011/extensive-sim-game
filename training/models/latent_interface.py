"""ELIT-style latent interface: Read -> latent processing -> Write.

Decouples spatial token count from core compute. K learned latent tokens
compress the spatial tokens, process them through transformer blocks,
then write updated information back to the spatial tokens.

Supports tail dropping during training for importance ordering.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentInterface(nn.Module):
    """ELIT-style latent interface: Read -> latent processing -> Write.

    Decouples spatial token count from core compute. K learned latent tokens
    compress the 20 spatial tokens, process them through transformer blocks,
    then write updated information back to the spatial tokens.

    Supports tail dropping during training for importance ordering.
    """

    def __init__(self, d_model: int = 128, n_latents: int = 12, n_heads: int = 8,
                 n_latent_blocks: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents

        # Learned latent tokens
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

        # Read: latents attend to spatial tokens (cross-attention)
        self.read_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.read_norm_q = nn.LayerNorm(d_model)
        self.read_norm_kv = nn.LayerNorm(d_model)
        self.read_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.read_ff_norm = nn.LayerNorm(d_model)

        # Latent transformer blocks (self-attention among latents)
        self.latent_blocks = nn.ModuleList()
        for _ in range(n_latent_blocks):
            self.latent_blocks.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ))

        # Write: spatial tokens attend to latents (cross-attention)
        self.write_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.write_norm_q = nn.LayerNorm(d_model)
        self.write_norm_kv = nn.LayerNorm(d_model)
        self.write_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.write_ff_norm = nn.LayerNorm(d_model)

        # Zero-init Write output projection -> identity pass-through at init
        nn.init.zeros_(self.write_attn.out_proj.weight)
        nn.init.zeros_(self.write_attn.out_proj.bias)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        n_latents_override: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_tokens: (B, N, d) -- entity/threat/position/aggregate tokens
            spatial_mask: (B, N) -- True where padded
            n_latents_override: if set, use only first J latents (tail dropping)
        Returns:
            updated_tokens: (B, N, d) -- spatial tokens enriched by latent processing
            pooled_latents: (B, d) -- mean of latent tokens (for temporal cell + heads)
        """
        B = spatial_tokens.shape[0]
        K = n_latents_override if n_latents_override is not None else self.n_latents

        # Initialize latents -- take first K (importance ordering via tail dropping)
        L = self.latents[:K].unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        # Read: latents attend to spatial tokens
        q = self.read_norm_q(L)
        kv = self.read_norm_kv(spatial_tokens)
        read_out, _ = self.read_attn(q, kv, kv, key_padding_mask=spatial_mask)
        L = L + read_out
        L = L + self.read_ff(self.read_ff_norm(L))

        # Latent self-attention blocks
        for block in self.latent_blocks:
            L = block(L)

        # Write: spatial tokens attend to latents
        q_w = self.write_norm_q(spatial_tokens)
        kv_w = self.write_norm_kv(L)
        write_out, _ = self.write_attn(q_w, kv_w, kv_w)
        updated_tokens = spatial_tokens + write_out
        updated_tokens = updated_tokens + self.write_ff(self.write_ff_norm(updated_tokens))

        # Pool latents for downstream heads
        pooled_latents = L.mean(dim=1)  # (B, d)

        return updated_tokens, pooled_latents
