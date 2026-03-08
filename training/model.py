"""Ability Transformer model.

Small transformer encoder (2 layers, 4 heads, d_model=64) for learning
compositional structure from ability DSL token sequences.

Supports three modes:
  1. Pre-training: masked token prediction (MLM)
  2. Fine-tuning: urgency + target prediction from [CLS] embedding
  3. Generation: autoregressive token prediction (future)

Architecture follows the grokking paper's setup: no dropout, designed for
high weight decay (AdamW lambda=1.0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbilityTransformer(nn.Module):
    """Transformer encoder for ability DSL tokens.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in vocabulary.
    d_model : int
        Embedding / hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feed-forward inner dimension.
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    pad_id : int
        Token ID used for padding (masked out in attention).
    cls_id : int
        Token ID for [CLS] — initialized to zero per grokking plan.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 256,
        pad_id: int = 0,
        cls_id: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.cls_id = cls_id

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learned positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder layers (no dropout — per grokking plan §3.3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability with high weight decay
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm on output
        self.out_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reduced scale per Kumar et al. (2310.06110).

        3x smaller init starts closer to the Goldilocks weight norm zone,
        reducing grokking delay. Zero-init [CLS] embedding per grokking plan §3.2.
        """
        nn.init.normal_(self.token_emb.weight, std=0.007)
        nn.init.normal_(self.pos_emb.weight, std=0.007)

        # Zero-init [CLS] token embedding
        with torch.no_grad():
            self.token_emb.weight[self.cls_id].zero_()
            # Keep padding at zero
            self.token_emb.weight[self.pad_id].zero_()

        # Initialize linear layers with reduced scale
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode token sequence.

        Parameters
        ----------
        input_ids : (batch, seq_len) int tensor
        attention_mask : (batch, seq_len) float tensor, 1=attend 0=ignore

        Returns
        -------
        hidden : (batch, seq_len, d_model) float tensor
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Build key_padding_mask: True = ignore
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == self.pad_id

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.out_norm(x)

        return x

    def cls_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract [CLS] embedding (position 0).

        Returns
        -------
        cls : (batch, d_model) float tensor
        """
        hidden = self.forward(input_ids, attention_mask)
        return hidden[:, 0, :]


class MLMHead(nn.Module):
    """Masked language model head for pre-training."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict masked tokens.

        Parameters
        ----------
        hidden : (batch, seq_len, d_model)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class DecisionHead(nn.Module):
    """Fine-tuning head: predict urgency + target from [CLS] embedding.

    Used in Phase 2 to replace the per-category MLP ability evaluators.
    """

    def __init__(self, d_model: int, n_targets: int = 3):
        super().__init__()
        self.urgency = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.target = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, cls_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict urgency and target scores.

        Parameters
        ----------
        cls_emb : (batch, d_model) — [CLS] embedding

        Returns
        -------
        urgency : (batch, 1) — sigmoid output in [0, 1]
        target_logits : (batch, n_targets)
        """
        return self.urgency(cls_emb), self.target(cls_emb)


class HintClassificationHead(nn.Module):
    """Auxiliary head: predict ability hint category from [CLS] embedding.

    Provides a direct semantic signal to the [CLS] token during pretraining,
    encouraging it to capture ability category information.
    """

    def __init__(self, d_model: int, n_classes: int = 6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """Predict hint category logits from [CLS] embedding.

        Parameters
        ----------
        cls_emb : (batch, d_model)

        Returns
        -------
        logits : (batch, n_classes)
        """
        return self.head(cls_emb)


class AbilityTransformerMLM(nn.Module):
    """Pre-training model: transformer + MLM head."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        self.mlm_head = MLMHead(self.transformer.d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.transformer(input_ids, attention_mask)
        return self.mlm_head(hidden)


class EntityEncoder(nn.Module):
    """Encode game state entities into d_model tokens for cross-attention.

    Input: flat game_state (batch, 70) = 7 entities × 10 features.
    Output: entity tokens (batch, 7, d_model) with learned type embeddings.

    Entity order: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2]
    Entity types: 0=self, 1=enemy, 2=ally
    """

    ENTITY_DIM = 30
    NUM_ENTITIES = 7
    NUM_TYPES = 3  # self, enemy, ally
    # Type assignment for each slot
    TYPE_IDS = [0, 1, 1, 1, 2, 2, 2]  # self, 3×enemy, 3×ally

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.ENTITY_DIM, d_model)
        self.type_emb = nn.Embedding(self.NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Self-attention over entities (matches pretrain_entity.py architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, game_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state entities.

        Parameters
        ----------
        game_state : (batch, 210)

        Returns
        -------
        entity_tokens : (batch, 7, d_model)
        entity_mask : (batch, 7) — True where entity doesn't exist (for key_padding_mask)
        """
        B = game_state.shape[0]
        device = game_state.device

        entities = game_state.view(B, self.NUM_ENTITIES, self.ENTITY_DIM)

        tokens = self.proj(entities)
        type_ids = torch.tensor(self.TYPE_IDS, device=device, dtype=torch.long)
        tokens = tokens + self.type_emb(type_ids).unsqueeze(0)
        tokens = self.input_norm(tokens)

        # Entity mask: exists feature is index 29
        exists = entities[:, :, 29]
        entity_mask = exists < 0.5  # True = ignore (padding)

        tokens = self.encoder(tokens, src_key_padding_mask=entity_mask)
        tokens = self.out_norm(tokens)

        return tokens, entity_mask


class EntityEncoderV2(nn.Module):
    """Variable-length entity encoder with dual projections for units + threats.

    Handles arbitrary numbers of entities and threats via self-attention.
    Type embeddings: 0=self, 1=enemy, 2=ally, 3=threat.

    Input format (per-sample, variable length):
        entities: list of 30-dim feature vectors (units)
        entity_types: list of type IDs (0/1/2)
        threats: list of 8-dim feature vectors

    All tokens are projected to d_model and processed jointly through
    shared self-attention.
    """

    ENTITY_DIM = 30
    THREAT_DIM = 8
    NUM_TYPES = 4  # self=0, enemy=1, ally=2, threat=3

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        # Dual projections: different input dims → same d_model
        self.entity_proj = nn.Linear(self.ENTITY_DIM, d_model)
        self.threat_proj = nn.Linear(self.THREAT_DIM, d_model)
        self.type_emb = nn.Embedding(self.NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Self-attention over all tokens (entities + threats)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode variable-length entities + threats.

        Parameters
        ----------
        entity_features : (B, max_entities, 30) — padded entity features
        entity_type_ids : (B, max_entities) — long, type IDs (0/1/2)
        threat_features : (B, max_threats, 8) — padded threat features
        entity_mask : (B, max_entities) — True where entity doesn't exist
        threat_mask : (B, max_threats) — True where threat doesn't exist

        Returns
        -------
        tokens : (B, max_entities + max_threats, d_model)
        full_mask : (B, max_entities + max_threats) — True = padding
        """
        B = entity_features.shape[0]
        device = entity_features.device

        # Project entities and threats to d_model
        ent_tokens = self.entity_proj(entity_features)  # (B, E, d)
        threat_tokens = self.threat_proj(threat_features)  # (B, T, d)

        # Add type embeddings
        ent_tokens = ent_tokens + self.type_emb(entity_type_ids)
        n_threats = threat_features.shape[1]
        threat_type_ids = torch.full(
            (B, n_threats), 3, device=device, dtype=torch.long,
        )
        threat_tokens = threat_tokens + self.type_emb(threat_type_ids)

        # Concatenate into single sequence
        tokens = torch.cat([ent_tokens, threat_tokens], dim=1)  # (B, E+T, d)
        tokens = self.input_norm(tokens)

        full_mask = torch.cat([entity_mask, threat_mask], dim=1)  # (B, E+T)

        tokens = self.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.out_norm(tokens)

        return tokens, full_mask


class CrossAttentionBlock(nn.Module):
    """Cross-attention: ability [CLS] attends to game state entity tokens.

    Query: [CLS] embedding from ability transformer (1 token)
    Key/Value: entity tokens from EntityEncoder (7 tokens)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : (batch, d_model) — [CLS] embedding
        kv : (batch, n_entities, d_model) — entity tokens
        kv_mask : (batch, n_entities) — True where entity is padding

        Returns
        -------
        out : (batch, d_model)
        """
        # Expand query to (batch, 1, d_model) for attention
        q = self.norm_q(query).unsqueeze(1)
        k = v = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=kv_mask)
        attn_out = attn_out.squeeze(1)  # (batch, d_model)

        # Residual + FF
        x = query + attn_out
        x = x + self.ff(self.norm_ff(x))

        return x


class AbilityTransformerDecision(nn.Module):
    """Fine-tuning model: transformer + cross-attention + decision head.

    Ability tokens go through the transformer encoder to produce a [CLS] embedding.
    Game state entities are encoded into tokens. The [CLS] embedding cross-attends
    to entity tokens, producing a context-aware representation for the decision head.
    """

    def __init__(
        self,
        vocab_size: int,
        game_state_dim: int = 0,
        n_targets: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        d = self.transformer.d_model

        self.has_cross_attn = game_state_dim > 0
        if self.has_cross_attn:
            n_heads = kwargs.get("n_heads", 4)
            n_layers = kwargs.get("n_layers", 2)
            self.entity_encoder = EntityEncoder(d, n_heads=n_heads, n_layers=n_layers)
            self.cross_attn = CrossAttentionBlock(d, n_heads=n_heads)

        self.decision_head = DecisionHead(d, n_targets)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        game_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_emb = self.transformer.cls_embedding(input_ids, attention_mask)

        if self.has_cross_attn and game_state is not None:
            entity_tokens, entity_mask = self.entity_encoder(game_state)
            cls_emb = self.cross_attn(cls_emb, entity_tokens, entity_mask)

        return self.decision_head(cls_emb)


# ---------------------------------------------------------------------------
# Actor-Critic for RL (replaces full AI)
# ---------------------------------------------------------------------------

# Action indices (must match Rust NUM_ACTIONS=14):
#   0: attack nearest, 1: attack weakest, 2: attack focus
#   3-10: use ability 0-7
#   11: move toward, 12: move away, 13: hold
NUM_ACTIONS = 14
NUM_BASE_ACTIONS = 6  # 3 attack + 2 move + hold
MAX_ABILITIES = 8


class AbilityActorCritic(nn.Module):
    """Actor-critic policy over all 14 actions.

    Base actions (attack/move/hold) use pooled entity state.
    Ability actions use cross-attention of ability [CLS] with entity tokens.
    Value head is state-only (ability-independent).
    """

    def __init__(
        self,
        vocab_size: int,
        game_state_dim: int = 210,
        **kwargs,
    ):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        d = self.transformer.d_model
        self.d_model = d

        n_heads = kwargs.get("n_heads", 4)
        n_layers = kwargs.get("n_layers", 2)
        self.entity_encoder = EntityEncoder(d, n_heads=n_heads, n_layers=n_layers)
        self.cross_attn = CrossAttentionBlock(d, n_heads=n_heads)

        # Base action head: pooled state → 6 logits
        self.base_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, NUM_BASE_ACTIONS),
        )

        # Ability logit projection: cross-attended CLS → 1 scalar per ability
        self.ability_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Value head: pooled state → scalar
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def encode_entities(self, game_state: torch.Tensor):
        """Encode game state → entity tokens + mask + pooled state."""
        entity_tokens, entity_mask = self.entity_encoder(game_state)
        # Mean pool over existing entities
        exists = (~entity_mask).float().unsqueeze(-1)  # (B, 7, 1)
        pooled = (entity_tokens * exists).sum(1) / exists.sum(1).clamp(min=1)
        return entity_tokens, entity_mask, pooled

    def forward_policy(
        self,
        game_state: torch.Tensor,
        ability_cls: list[torch.Tensor | None],
    ) -> torch.Tensor:
        """Compute action logits for a batch.

        Parameters
        ----------
        game_state : (B, 210)
        ability_cls : list of MAX_ABILITIES tensors, each (B, d_model) or None
            Pre-computed [CLS] embeddings for each ability slot.
            None means ability slot is empty.

        Returns
        -------
        logits : (B, 14) — raw logits (apply mask before softmax)
        """
        B = game_state.shape[0]
        device = game_state.device
        entity_tokens, entity_mask, pooled = self.encode_entities(game_state)

        # Base action logits: [attack_near, attack_weak, attack_focus, move_toward, move_away, hold]
        base_logits = self.base_head(pooled)  # (B, 6)

        # Ability logits: cross-attend each ability CLS to entities
        ability_logits = torch.full((B, MAX_ABILITIES), -1e9, device=device)
        for i in range(MAX_ABILITIES):
            if ability_cls[i] is not None:
                cross_emb = self.cross_attn(ability_cls[i], entity_tokens, entity_mask)
                ability_logits[:, i] = self.ability_proj(cross_emb).squeeze(-1)

        # Combine: [attack×3, ability×8, move×2, hold]
        logits = torch.cat([
            base_logits[:, :3],      # attack_near, attack_weak, attack_focus
            ability_logits,           # ability 0-7
            base_logits[:, 3:],      # move_toward, move_away, hold
        ], dim=1)

        return logits

    def forward_value(self, game_state: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s).

        Parameters
        ----------
        game_state : (B, 210)

        Returns
        -------
        value : (B, 1)
        """
        _, _, pooled = self.encode_entities(game_state)
        return self.value_head(pooled)

    def forward(
        self,
        game_state: torch.Tensor,
        ability_cls: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        B = game_state.shape[0]
        entity_tokens, entity_mask, pooled = self.encode_entities(game_state)

        base_logits = self.base_head(pooled)
        ability_logits = torch.full((B, MAX_ABILITIES), -1e9, device=game_state.device)
        for i in range(MAX_ABILITIES):
            if ability_cls[i] is not None:
                cross_emb = self.cross_attn(ability_cls[i], entity_tokens, entity_mask)
                ability_logits[:, i] = self.ability_proj(cross_emb).squeeze(-1)

        logits = torch.cat([
            base_logits[:, :3],
            ability_logits,
            base_logits[:, 3:],
        ], dim=1)

        value = self.value_head(pooled)
        return logits, value


class AbilityActorCriticV2(nn.Module):
    """Actor-critic with variable-length entity encoder V2.

    Same action space as V1 (14 actions), but uses EntityEncoderV2 for
    variable-length entities + threats instead of fixed 7-slot encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        entity_encoder_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        d = self.transformer.d_model
        self.d_model = d

        n_heads = kwargs.get("n_heads", 4)
        self.entity_encoder = EntityEncoderV2(
            d_model=d, n_heads=n_heads, n_layers=entity_encoder_layers,
        )
        self.cross_attn = CrossAttentionBlock(d, n_heads=n_heads)

        self.base_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, NUM_BASE_ACTIONS),
        )

        self.ability_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def encode_entities(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
    ):
        """Encode v2 game state → tokens + mask + pooled."""
        tokens, full_mask = self.entity_encoder(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask,
        )
        exist = (~full_mask).float().unsqueeze(-1)
        pooled = (tokens * exist).sum(1) / exist.sum(1).clamp(min=1)
        return tokens, full_mask, pooled

    def forward_value(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state value V(s). Returns (B, 1)."""
        _, _, pooled = self.encode_entities(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask,
        )
        return self.value_head(pooled)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        ability_cls: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        B = entity_features.shape[0]
        device = entity_features.device
        tokens, full_mask, pooled = self.encode_entities(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask,
        )

        base_logits = self.base_head(pooled)
        ability_logits = torch.full((B, MAX_ABILITIES), -1e9, device=device)
        for i in range(MAX_ABILITIES):
            if ability_cls[i] is not None:
                cross_emb = self.cross_attn(ability_cls[i], tokens, full_mask)
                ability_logits[:, i] = self.ability_proj(cross_emb).squeeze(-1)

        logits = torch.cat([
            base_logits[:, :3],
            ability_logits,
            base_logits[:, 3:],
        ], dim=1)

        value = self.value_head(pooled)
        return logits, value


# ---------------------------------------------------------------------------
# V3: Pointer-based action space with position tokens
# ---------------------------------------------------------------------------

# Action types for pointer architecture:
#   0: attack (pointer selects enemy target)
#   1: move (pointer selects entity/position/threat target)
#   2: hold (no pointer needed)
#   3-10: ability 0-7 (pointer selects target based on ability targeting type)
NUM_ACTION_TYPES = 3 + MAX_ABILITIES  # 11
POSITION_DIM = 8


class EntityEncoderV3(nn.Module):
    """Entity encoder with position tokens (type=4).

    Extends V2 with an additional position projection for area-of-interest
    tokens that represent cover spots, elevated positions, chokepoints, etc.

    Type embeddings: 0=self, 1=enemy, 2=ally, 3=threat, 4=position.
    """

    ENTITY_DIM = 30
    THREAT_DIM = 8
    POSITION_DIM = 8
    NUM_TYPES = 5  # self=0, enemy=1, ally=2, threat=3, position=4

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        self.entity_proj = nn.Linear(self.ENTITY_DIM, d_model)
        self.threat_proj = nn.Linear(self.THREAT_DIM, d_model)
        self.position_proj = nn.Linear(self.POSITION_DIM, d_model)
        self.type_emb = nn.Embedding(self.NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode entities + threats + positions via shared self-attention.

        Parameters
        ----------
        entity_features : (B, E, 30)
        entity_type_ids : (B, E) — long
        threat_features : (B, T, 8)
        entity_mask : (B, E) — True = padding
        threat_mask : (B, T) — True = padding
        position_features : (B, P, 8) or None
        position_mask : (B, P) or None — True = padding

        Returns
        -------
        tokens : (B, E+T+P, d_model)
        full_mask : (B, E+T+P) — True = padding
        """
        B = entity_features.shape[0]
        device = entity_features.device

        ent_tokens = self.entity_proj(entity_features)
        threat_tokens = self.threat_proj(threat_features)

        ent_tokens = ent_tokens + self.type_emb(entity_type_ids)
        n_threats = threat_features.shape[1]
        threat_type_ids = torch.full((B, n_threats), 3, device=device, dtype=torch.long)
        threat_tokens = threat_tokens + self.type_emb(threat_type_ids)

        parts = [ent_tokens, threat_tokens]
        masks = [entity_mask, threat_mask]

        if position_features is not None and position_features.shape[1] > 0:
            pos_tokens = self.position_proj(position_features)
            n_pos = position_features.shape[1]
            pos_type_ids = torch.full((B, n_pos), 4, device=device, dtype=torch.long)
            pos_tokens = pos_tokens + self.type_emb(pos_type_ids)
            parts.append(pos_tokens)
            if position_mask is not None:
                masks.append(position_mask)
            else:
                masks.append(torch.zeros(B, n_pos, dtype=torch.bool, device=device))

        tokens = torch.cat(parts, dim=1)
        tokens = self.input_norm(tokens)
        full_mask = torch.cat(masks, dim=1)

        tokens = self.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.out_norm(tokens)

        return tokens, full_mask


class PointerHead(nn.Module):
    """Pointer head: produces target distributions over entity tokens.

    Uses scaled dot-product attention between a query (from pooled state
    or cross-attended ability CLS) and keys (entity token projections).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Action type head: pooled state → action type logits
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_ACTION_TYPES),
        )

        # Shared key projection for all pointer queries
        self.pointer_key = nn.Linear(d_model, d_model)

        # Per-action-type query projections
        self.attack_query = nn.Linear(d_model, d_model)
        self.move_query = nn.Linear(d_model, d_model)
        self.ability_queries = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(MAX_ABILITIES)
        ])

        self.scale = d_model ** -0.5

    def forward(
        self,
        pooled: torch.Tensor,
        entity_tokens: torch.Tensor,
        entity_mask: torch.Tensor,
        ability_cross_embs: list[torch.Tensor | None],
        entity_type_ids_full: torch.Tensor,
    ) -> dict:
        """Compute action type logits and per-type pointer distributions.

        Parameters
        ----------
        pooled : (B, d_model) — mean-pooled entity representation
        entity_tokens : (B, N, d_model) — all tokens (entity+threat+position)
        entity_mask : (B, N) — True = padding
        ability_cross_embs : list of (B, d_model) or None per ability slot
        entity_type_ids_full : (B, N) — type IDs for all tokens (0-4)

        Returns
        -------
        dict with keys:
            type_logits: (B, 11)
            attack_ptr: (B, N) — logits for attack target
            move_ptr: (B, N) — logits for move target
            ability_ptrs: list of (B, N) or None per ability
        """
        B, N, D = entity_tokens.shape
        device = entity_tokens.device

        type_logits = self.action_type_head(pooled)
        keys = self.pointer_key(entity_tokens)  # (B, N, D)

        # Attack pointer: only enemies (type=1) are valid targets
        atk_q = self.attack_query(pooled).unsqueeze(1)  # (B, 1, D)
        atk_ptr = (atk_q @ keys.transpose(-1, -2)).squeeze(1) * self.scale  # (B, N)
        atk_mask = (entity_type_ids_full != 1) | entity_mask  # True = invalid
        atk_ptr = atk_ptr.masked_fill(atk_mask, -1e9)

        # Move pointer: enemies(1), allies(2), threats(3), positions(4)
        mv_q = self.move_query(pooled).unsqueeze(1)
        mv_ptr = (mv_q @ keys.transpose(-1, -2)).squeeze(1) * self.scale
        # Only self (type=0) and padding are invalid move targets
        mv_mask = (entity_type_ids_full == 0) | entity_mask
        mv_ptr = mv_ptr.masked_fill(mv_mask, -1e9)

        # Ability pointers
        ability_ptrs = []
        for i in range(MAX_ABILITIES):
            if ability_cross_embs[i] is not None:
                ab_q = self.ability_queries[i](ability_cross_embs[i]).unsqueeze(1)
                ab_ptr = (ab_q @ keys.transpose(-1, -2)).squeeze(1) * self.scale
                # Default: all non-padding tokens are valid (masking per ability
                # targeting type is applied externally when building target masks)
                ab_ptr = ab_ptr.masked_fill(entity_mask, -1e9)
                ability_ptrs.append(ab_ptr)
            else:
                ability_ptrs.append(None)

        return {
            "type_logits": type_logits,
            "attack_ptr": atk_ptr,
            "move_ptr": mv_ptr,
            "ability_ptrs": ability_ptrs,
        }


class AbilityActorCriticV3(nn.Module):
    """Actor-critic with pointer-based action space and position tokens.

    Action = (action_type, target_pointer):
        action_type: attack(0), move(1), hold(2), ability_0..7(3..10)
        target_pointer: scaled dot-product attention over entity tokens

    Uses EntityEncoderV3 (entities + threats + positions) and PointerHead.
    """

    def __init__(
        self,
        vocab_size: int,
        entity_encoder_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        d = self.transformer.d_model
        self.d_model = d

        n_heads = kwargs.get("n_heads", 4)
        self.entity_encoder = EntityEncoderV3(
            d_model=d, n_heads=n_heads, n_layers=entity_encoder_layers,
        )
        self.cross_attn = CrossAttentionBlock(d, n_heads=n_heads)
        self.pointer_head = PointerHead(d)

        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def encode_entities(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ):
        """Encode v3 game state → tokens + mask + pooled."""
        tokens, full_mask = self.entity_encoder(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
        )
        exist = (~full_mask).float().unsqueeze(-1)
        pooled = (tokens * exist).sum(1) / exist.sum(1).clamp(min=1)
        return tokens, full_mask, pooled

    def _build_full_type_ids(
        self,
        entity_type_ids: torch.Tensor,
        n_threats: int,
        n_positions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build type IDs for the full token sequence (entities+threats+positions)."""
        B = entity_type_ids.shape[0]
        threat_types = torch.full((B, n_threats), 3, device=device, dtype=torch.long)
        parts = [entity_type_ids, threat_types]
        if n_positions > 0:
            pos_types = torch.full((B, n_positions), 4, device=device, dtype=torch.long)
            parts.append(pos_types)
        return torch.cat(parts, dim=1)

    def forward_value(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute state value V(s). Returns (B, 1)."""
        _, _, pooled = self.encode_entities(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
        )
        return self.value_head(pooled)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        ability_cls: list[torch.Tensor | None],
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Returns (pointer_output_dict, state_value).

        pointer_output_dict has keys: type_logits, attack_ptr, move_ptr, ability_ptrs
        """
        B = entity_features.shape[0]
        device = entity_features.device
        tokens, full_mask, pooled = self.encode_entities(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
        )

        # Cross-attend each ability CLS to entity tokens
        ability_cross_embs = []
        for i in range(MAX_ABILITIES):
            if ability_cls[i] is not None:
                cross_emb = self.cross_attn(ability_cls[i], tokens, full_mask)
                ability_cross_embs.append(cross_emb)
            else:
                ability_cross_embs.append(None)

        # Build full type IDs for pointer masking
        n_threats = threat_features.shape[1]
        n_positions = position_features.shape[1] if position_features is not None else 0
        full_type_ids = self._build_full_type_ids(
            entity_type_ids, n_threats, n_positions, device,
        )

        pointer_out = self.pointer_head(
            pooled, tokens, full_mask, ability_cross_embs, full_type_ids,
        )

        value = self.value_head(pooled)
        return pointer_out, value
