"""V5 Actor-Critic: d=128, 8 heads, latent interface, CfC temporal cell.

Key differences from V4:
- d_model=128 (was 32) -- 16d per attention head, genuinely expressive
- LatentInterface(K=12) between encoder and heads
- Aggregate token for crowd summary of truncated entities
- No external_cls_proj -- 128d ability CLS feeds directly (d_model matches)
- Tail dropping during training for importance-ordered latents
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_v5 import EntityEncoderV5, V5_DEFAULT_D, V5_DEFAULT_HEADS
from .latent_interface import LatentInterface
from .cfc_cell import CfCCell, CFC_H_DIM
from .combat_head import CombatPointerHeadV5, MAX_ABILITIES, NUM_COMBAT_TYPES

# Movement directions: 8 cardinal + stay
NUM_MOVE_DIRS = 9
V5_DEFAULT_LATENTS = 12


class AbilityActorCriticV5(nn.Module):
    """V5 actor-critic: d=128, 8 heads, latent interface, aggregate token.

    Key differences from V4:
    - d_model=128 (was 32) -- 16d per attention head, genuinely expressive
    - LatentInterface(K=12) between encoder and heads
    - Aggregate token for crowd summary of truncated entities
    - No external_cls_proj -- 128d ability CLS feeds directly (d_model matches)
    - Tail dropping during training for importance-ordered latents
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = V5_DEFAULT_D,
        d_ff: int = 256,
        n_layers: int = 4,
        n_heads: int = V5_DEFAULT_HEADS,
        entity_encoder_layers: int = 4,
        external_cls_dim: int = 0,
        h_dim: int = CFC_H_DIM,
        n_latents: int = V5_DEFAULT_LATENTS,
        n_latent_blocks: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.h_dim = h_dim
        self.n_latents = n_latents

        # Ability transformer (frozen during RL, used for CLS embeddings)
        # Lazy import to avoid circular dependency with model.py
        from model import AbilityTransformer
        self.transformer = AbilityTransformer(
            vocab_size=vocab_size, d_model=d_model, d_ff=d_ff,
            n_layers=n_layers, n_heads=n_heads,
        )

        # CLS projection: only if external_cls_dim != d_model
        self.external_cls_proj: nn.Module | None = None
        if external_cls_dim > 0 and external_cls_dim != d_model:
            self.external_cls_proj = nn.Linear(external_cls_dim, d_model)
        # When external_cls_dim == d_model (128), no projection needed -- direct feed

        # Entity encoder
        self.entity_encoder = EntityEncoderV5(
            d_model=d_model, n_heads=n_heads, n_layers=entity_encoder_layers,
        )

        # Latent interface
        self.latent_interface = LatentInterface(
            d_model=d_model, n_latents=n_latents, n_heads=n_heads,
            n_latent_blocks=n_latent_blocks,
        )

        # Cross-attention for abilities
        from model import CrossAttentionBlock
        self.cross_attn = CrossAttentionBlock(d_model, n_heads=n_heads)

        # Temporal cell: CfC (Closed-form Continuous-time)
        self.temporal_cell = CfCCell(d_model, h_dim)

        # Position head: predict target (x, y) waypoint in world space
        # Output is normalized (x/20, y/20) — the sim handles pathfinding
        # One correct waypoint = correct movement until next decision tick
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )

        # Combat pointer head
        self.combat_head = CombatPointerHeadV5(d_model)

    def project_cls(self, cls: torch.Tensor) -> torch.Tensor:
        if self.external_cls_proj is not None:
            return self.external_cls_proj(cls)
        return cls

    def _build_full_type_ids(self, entity_type_ids, n_zones, has_aggregate, device):
        B = entity_type_ids.shape[0]
        parts = [entity_type_ids]
        parts.append(torch.full((B, n_zones), 3, device=device, dtype=torch.long))
        if has_aggregate:
            parts.append(torch.full((B, 1), 4, device=device, dtype=torch.long))
        return torch.cat(parts, dim=1)

    def encode_state(
        self,
        entity_features, entity_type_ids,
        zone_features, entity_mask, zone_mask,
        ability_cls: list[torch.Tensor | None],
        aggregate_features=None,
        n_latents_override: int | None = None,
    ) -> dict:
        """Encode game state: entity encoder -> latent interface. No CfC."""
        tokens, full_mask = self.entity_encoder(
            entity_features, entity_type_ids, zone_features,
            entity_mask, zone_mask,
            aggregate_features,
        )

        # Latent interface: Read -> process -> Write
        tokens, pooled = self.latent_interface(tokens, full_mask, n_latents_override)

        # Cross-attend abilities
        ability_cross_embs = []
        for i in range(MAX_ABILITIES):
            if ability_cls[i] is not None:
                cls_i = self.project_cls(ability_cls[i])
                cross_emb = self.cross_attn(cls_i, tokens, full_mask)
                ability_cross_embs.append(cross_emb)
            else:
                ability_cross_embs.append(None)

        n_zones = zone_features.shape[1]
        has_agg = aggregate_features is not None
        full_type_ids = self._build_full_type_ids(
            entity_type_ids, n_zones, has_agg,
            entity_features.device,
        )

        return {
            "pooled": pooled,
            "tokens": tokens,
            "full_mask": full_mask,
            "ability_cross_embs": ability_cross_embs,
            "full_type_ids": full_type_ids,
        }

    def decide(self, pooled, tokens, full_mask, ability_cross_embs, full_type_ids,
               aggregate_features=None):
        """Run decision heads on CfC-enriched pooled representation."""
        # Target position: (x/20, y/20) normalized waypoint
        target_pos = self.position_head(pooled)  # (B, 2)

        combat_out = self.combat_head(pooled, tokens, full_mask, ability_cross_embs, full_type_ids)
        return {
            "target_pos": target_pos,  # (B, 2) — normalized (x/20, y/20)
            **combat_out,
        }

    def forward(
        self,
        entity_features, entity_type_ids,
        zone_features, entity_mask, zone_mask,
        ability_cls: list[torch.Tensor | None],
        aggregate_features=None,
        h_prev=None,
        n_latents_override: int | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Single-step forward (for inference). Returns (output_dict, h_new)."""
        enc = self.encode_state(
            entity_features, entity_type_ids,
            zone_features, entity_mask, zone_mask, ability_cls,
            aggregate_features,
            n_latents_override,
        )

        pooled, h_new = self.temporal_cell(enc["pooled"], h_prev)

        output = self.decide(
            pooled, enc["tokens"], enc["full_mask"],
            enc["ability_cross_embs"], enc["full_type_ids"],
            aggregate_features=aggregate_features,
        )

        return output, h_new
