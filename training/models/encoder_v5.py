"""V5 Entity Encoder: d=128, 8 heads, 6 type embeddings (includes aggregate).

Same structure as V3 but wider, with aggregate token support.
Entity types: 0=self, 1=enemy, 2=ally, 3=threat, 4=position, 5=aggregate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

AGG_FEATURE_DIM = 16
NUM_V5_TYPES = 6
V5_DEFAULT_D = 128
V5_DEFAULT_HEADS = 8


class EntityEncoderV5(nn.Module):
    """Entity encoder for V5: d=128, 8 heads, 6 type embeddings (includes aggregate).

    Same structure as V3 but wider, with aggregate token support.
    """

    ENTITY_DIM = 34  # 30 base + 4 spatial summary
    THREAT_DIM = 10  # 8 base + kind + LOS
    POSITION_DIM = 8
    AGG_DIM = AGG_FEATURE_DIM
    NUM_TYPES = NUM_V5_TYPES  # 0=self, 1=enemy, 2=ally, 3=threat, 4=position, 5=aggregate

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        self.entity_proj = nn.Linear(self.ENTITY_DIM, d_model)
        self.threat_proj = nn.Linear(self.THREAT_DIM, d_model)
        self.position_proj = nn.Linear(self.POSITION_DIM, d_model)
        self.agg_proj = nn.Linear(self.AGG_DIM, d_model)
        self.type_emb = nn.Embedding(self.NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        entity_features: torch.Tensor,      # (B, E, 34)
        entity_type_ids: torch.Tensor,       # (B, E)
        threat_features: torch.Tensor,       # (B, T, 10)
        entity_mask: torch.Tensor,           # (B, E)
        threat_mask: torch.Tensor,           # (B, T)
        position_features: torch.Tensor | None = None,  # (B, P, 8)
        position_mask: torch.Tensor | None = None,
        aggregate_features: torch.Tensor | None = None,  # (B, 16)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tokens, full_mask) including all token types."""
        B = entity_features.shape[0]
        device = entity_features.device

        # Project each token type
        ent_tokens = self.entity_proj(entity_features) + self.type_emb(entity_type_ids)
        thr_tokens = self.threat_proj(threat_features) + self.type_emb(
            torch.full((B, threat_features.shape[1]), 3, device=device, dtype=torch.long)
        )

        parts = [ent_tokens, thr_tokens]
        masks = [entity_mask, threat_mask]

        if position_features is not None:
            pos_tokens = self.position_proj(position_features) + self.type_emb(
                torch.full((B, position_features.shape[1]), 4, device=device, dtype=torch.long)
            )
            parts.append(pos_tokens)
            masks.append(position_mask if position_mask is not None else
                        torch.zeros(B, position_features.shape[1], dtype=torch.bool, device=device))

        if aggregate_features is not None:
            agg_token = self.agg_proj(aggregate_features).unsqueeze(1) + self.type_emb(
                torch.full((B, 1), 5, device=device, dtype=torch.long)
            )
            parts.append(agg_token)
            masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))  # never masked

        tokens = torch.cat(parts, dim=1)
        tokens = self.input_norm(tokens)
        full_mask = torch.cat(masks, dim=1)

        tokens = self.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.out_norm(tokens)

        return tokens, full_mask
