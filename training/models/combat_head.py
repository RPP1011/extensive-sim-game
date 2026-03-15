"""V5 Combat Pointer Head: combat type classification + pointer targeting.

Combat types: attack(0), hold(1), ability_0..7(2..9) = 10 total.
Uses scaled dot-product attention for target selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_ABILITIES = 8
NUM_COMBAT_TYPES = 2 + MAX_ABILITIES  # 10


class CombatPointerHeadV5(nn.Module):
    """Combat head for V5: same structure as V4 but at d=128."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.combat_type_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, NUM_COMBAT_TYPES),
        )
        self.pointer_key = nn.Linear(d_model, d_model)
        self.attack_query = nn.Linear(d_model, d_model)
        self.ability_queries = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(MAX_ABILITIES)
        ])
        self.scale = d_model ** -0.5

    def forward(self, pooled, entity_tokens, entity_mask, ability_cross_embs, entity_type_ids_full):
        B, N, D = entity_tokens.shape
        combat_logits = self.combat_type_head(pooled)
        keys = self.pointer_key(entity_tokens)

        atk_q = self.attack_query(pooled).unsqueeze(1)
        atk_ptr = (atk_q @ keys.transpose(-1, -2)).squeeze(1) * self.scale
        atk_mask = (entity_type_ids_full != 1) | entity_mask
        atk_ptr = atk_ptr.masked_fill(atk_mask, -1e9)

        ability_ptrs = []
        for i in range(MAX_ABILITIES):
            if ability_cross_embs[i] is not None:
                ab_q = self.ability_queries[i](ability_cross_embs[i]).unsqueeze(1)
                ab_ptr = (ab_q @ keys.transpose(-1, -2)).squeeze(1) * self.scale
                ab_ptr = ab_ptr.masked_fill(entity_mask, -1e9)
                ability_ptrs.append(ab_ptr)
            else:
                ability_ptrs.append(None)

        return {"combat_logits": combat_logits, "attack_ptr": atk_ptr, "ability_ptrs": ability_ptrs}
