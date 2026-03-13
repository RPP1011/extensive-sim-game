#!/usr/bin/env python3
"""Train ability latent operator: State Encoder + Ability Operator + Decoder Heads.

Architecture (d_model=64):
  StateEncoder: 4-layer transformer over entity/threat/position/ability tokens
  AbilityOperator: 2-layer transformer conditioning on ability CLS + caster + duration
  DecoderHeads: Gaussian (beta-NLL) for hp/cc/pos, BCE for exists/stun

Usage:
  uv run --with numpy --with torch training/train_operator.py \
    --data generated/operator_dataset_hvh.npz --max-steps 50000

Experiments (--experiment flag):
  e0: Baseline (beta-NLL, all entities, no transforms)
  e1: MSE loss (no beta-NLL, no variance output)
  e2: Caster-target entity masking (loss only on affected entities)
  e3: Symlog targets
  e4: Explicit loss weights (hp=2, cc=0.5, pos=2, exists=1)
  e5: MSE + entity masking (E1+E2)
  e6: No GrokfastEMA
  e7: Lower LR (1e-4)
  e8: Huber loss (delta=0.1)
  e9: MSE + symlog + entity masking (E1+E2+E3)
  e10: Diffusion head (DDPM)
  Round 2 (Huber-based):
  e11: Huber + no Grokfast
  e12: Huber + symlog
  e13: Huber + entity masking
  e14: Huber delta=0.01
  e15: Huber delta=0.05
  e16: Huber delta=0.2
  e17: Huber delta=0.5
  e18: Huber + residual gate (GatedMeanHead)
  e19: Huber + no Grokfast + symlog
  e20: Huber + residual gate + no Grokfast
  e21: Huber + 20K steps (4x longer)
  Round 3 (Discretized HP):
  e22: Discretized HP classification (7 bins) + beta-NLL for pos/cc
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────

D_MODEL = 64
N_HEADS = 8
D_FF = 128
ENCODER_LAYERS = 4
OPERATOR_LAYERS = 2
MAX_ENTITIES = 7
ENTITY_DIM = 23
THREAT_DIM = 8
POSITION_DIM = 8
ABILITY_SLOT_DIM = 130  # 128 CLS + is_ready + cd_fraction (overridden by data)
ABILITY_CLS_DIM = 128   # frozen CLS dimension (overridden by data)
NUM_TOKEN_TYPES = 8
INIT_STD = 0.007


# ── Experiment Config ─────────────────────────────────────────────────────────

EXPERIMENTS = {
    'e0': 'Baseline (beta-NLL)',
    'e1': 'MSE loss (no beta-NLL)',
    'e2': 'Caster-target entity masking',
    'e3': 'Symlog targets',
    'e4': 'Explicit loss weights (hp=2, pos=2)',
    'e5': 'MSE + entity masking (E1+E2)',
    'e6': 'No GrokfastEMA',
    'e7': 'Lower LR (1e-4)',
    'e8': 'Huber loss (delta=0.1)',
    'e9': 'MSE + symlog + entity masking (E1+E2+E3)',
    'e10': 'Diffusion head (DDPM)',
    # Round 2: Huber-based experiments
    'e11': 'Huber + no Grokfast',
    'e12': 'Huber + symlog',
    'e13': 'Huber + entity masking',
    'e14': 'Huber delta=0.01',
    'e15': 'Huber delta=0.05',
    'e16': 'Huber delta=0.2',
    'e17': 'Huber delta=0.5',
    'e18': 'Huber + residual gate',
    'e19': 'Huber + no Grokfast + symlog',
    'e20': 'Huber + residual gate + no Grokfast',
    'e21': 'Huber + 20K steps',
    # Round 3: Discretized HP
    'e22': 'Discretized HP (7-class) + beta-NLL pos/cc',
    'e23': 'Two-stage: affected-entity detection + impact classification',
}


class ExperimentConfig:
    """Configuration derived from experiment name."""

    def __init__(self, name):
        self.name = name
        self.use_mse = name in ('e1', 'e5', 'e9')
        self.use_entity_masking = name in ('e2', 'e5', 'e9', 'e13')
        self.use_symlog = name in ('e3', 'e9', 'e12', 'e19')
        self.use_grokfast = name not in ('e6', 'e11', 'e19', 'e20')
        self.use_huber = name in ('e8', 'e11', 'e12', 'e13', 'e14', 'e15',
                                  'e16', 'e17', 'e18', 'e19', 'e20', 'e21')
        self.use_diffusion = name == 'e10'
        self.use_gated_head = name in ('e18', 'e20')
        self.use_discrete_hp = name in ('e22',)
        self.use_two_stage = name in ('e23',)
        self.huber_delta = {
            'e14': 0.01, 'e15': 0.05, 'e16': 0.2, 'e17': 0.5,
        }.get(name, 0.1)
        self.max_steps_override = 20000 if name == 'e21' else None
        self.loss_weights = (
            {'hp': 2.0, 'cc': 0.5, 'pos': 2.0, 'exists': 1.0}
            if name == 'e4' else
            {'hp': 1.0, 'cc': 1.0, 'pos': 1.0, 'exists': 1.0}
        )
        self.lr_override = 1e-4 if name == 'e7' else None


# ── Model ──────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=mask)
        x = x + h
        h = self.norm2(x)
        h = self.ff(h)
        return x + h


class StateEncoder(nn.Module):
    """Encode game state tokens into d_model representations."""

    def __init__(self, ability_slot_dim=ABILITY_SLOT_DIM):
        super().__init__()
        self.entity_proj = nn.Linear(ENTITY_DIM, D_MODEL)
        self.threat_proj = nn.Linear(THREAT_DIM, D_MODEL)
        self.position_proj = nn.Linear(POSITION_DIM, D_MODEL)
        self.ability_proj = nn.Linear(ability_slot_dim, D_MODEL)
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, D_MODEL)
        self.input_norm = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList([
            TransformerLayer(D_MODEL, N_HEADS, D_FF)
            for _ in range(ENCODER_LAYERS)
        ])
        self.out_norm = nn.LayerNorm(D_MODEL)

    def forward(self, entity_feat, entity_types, entity_mask,
                threat_feat, threat_mask,
                position_feat, position_mask,
                ability_feat, ability_types, ability_mask):
        # Project each token type
        e = self.entity_proj(entity_feat)      # (B, 7, d)
        t = self.threat_proj(threat_feat)      # (B, 8, d)
        p = self.position_proj(position_feat)  # (B, 8, d)
        a = self.ability_proj(ability_feat)     # (B, A, d)

        # Type embeddings: entities=0/1/2, threats=3, positions=4, abilities=5/6/7
        e = e + self.type_emb(entity_types)
        threat_type = torch.full(threat_mask.shape, 3, device=entity_feat.device, dtype=torch.long)
        t = t + self.type_emb(threat_type)
        pos_type = torch.full(position_mask.shape, 4, device=entity_feat.device, dtype=torch.long)
        p = p + self.type_emb(pos_type)
        a = a + self.type_emb(ability_types)

        # Concatenate all tokens
        tokens = torch.cat([e, t, p, a], dim=1)  # (B, 7+8+8+A, d)
        mask = torch.cat([entity_mask, threat_mask, position_mask, ability_mask], dim=1)

        tokens = self.input_norm(tokens)
        for layer in self.layers:
            tokens = layer(tokens, mask=mask)
        tokens = self.out_norm(tokens)

        # Extract entity tokens (first 7)
        return tokens[:, :MAX_ENTITIES, :]  # (B, 7, d)


class AbilityOperator(nn.Module):
    """Condition encoded state on a specific ability cast."""

    def __init__(self, ability_cls_dim=ABILITY_CLS_DIM):
        super().__init__()
        self.cls_proj = nn.Linear(ability_cls_dim, D_MODEL)
        self.caster_emb = nn.Embedding(MAX_ENTITIES, D_MODEL)
        self.duration_proj = nn.Linear(32, D_MODEL)  # 16 sin + 16 cos
        self.ability_norm = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList([
            TransformerLayer(D_MODEL, N_HEADS, D_FF)
            for _ in range(OPERATOR_LAYERS)
        ])
        self.out_norm = nn.LayerNorm(D_MODEL)

    def forward(self, z_before, ability_cls, caster_slot, duration_norm, entity_mask):
        B = z_before.shape[0]
        device = z_before.device

        # Sinusoidal duration encoding
        freqs = torch.arange(16, device=device, dtype=torch.float32)
        freqs = torch.exp(freqs * (-math.log(10000.0) / 16))
        angles = duration_norm.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, 16)
        dur_enc = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 32)

        # Ability token = CLS + caster + duration
        ability_token = (
            self.cls_proj(ability_cls) +
            self.caster_emb(caster_slot) +
            self.duration_proj(dur_enc)
        )  # (B, d)

        ability_token = self.ability_norm(ability_token).unsqueeze(1)  # (B, 1, d)

        # Append ability token to entity sequence
        tokens = torch.cat([z_before, ability_token], dim=1)  # (B, 8, d)
        # Extend mask: ability token is never masked
        abl_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)
        mask = torch.cat([entity_mask, abl_mask], dim=1)  # (B, 8)

        for layer in self.layers:
            tokens = layer(tokens, mask=mask)
        tokens = self.out_norm(tokens)

        return tokens[:, :MAX_ENTITIES, :]  # (B, 7, d)


class GaussianHead(nn.Module):
    """Predict mean + log_var for continuous targets."""

    def __init__(self, d_in, d_hidden, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features * 2),
        )
        self.n_features = n_features

    def forward(self, x):
        out = self.net(x)  # (B, E, 2*F)
        mean = out[..., :self.n_features]
        log_var = out[..., self.n_features:].clamp(-10, 10)
        return mean, log_var


class MeanHead(nn.Module):
    """Predict only mean for continuous targets (no variance)."""

    def __init__(self, d_in, d_hidden, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features),
        )

    def forward(self, x):
        return self.net(x)  # (B, E, F)


class GatedMeanHead(nn.Module):
    """Mean head with learnable residual gate, initialized near zero output."""

    def __init__(self, d_in, d_hidden, n_features, gate_init=-3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features),
        )
        self.gate_bias = nn.Parameter(torch.full((n_features,), gate_init))

    def forward(self, x):
        return torch.sigmoid(self.gate_bias) * self.net(x)


class DiffusionHead(nn.Module):
    """DDPM noise prediction head for continuous targets."""

    def __init__(self, d_model, n_features, n_steps=100):
        super().__init__()
        self.n_features = n_features
        self.n_steps = n_steps

        # Timestep embedding (sinusoidal → d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Noisy target projection
        self.target_proj = nn.Linear(n_features, d_model)

        # Noise prediction MLP
        self.noise_pred = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_features),
        )

        # Precompute noise schedule
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1.0 - alpha_bar).sqrt())
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)

    def _time_embedding(self, t, device):
        """Sinusoidal timestep embedding."""
        freqs = torch.arange(16, device=device, dtype=torch.float32)
        freqs = torch.exp(freqs * (-math.log(10000.0) / 16))
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, 16)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 32)

    def forward_train(self, z, target):
        """Training: add noise, predict it. Returns noise prediction loss.

        Args:
            z: entity embeddings (B, E, d_model)
            target: ground truth deltas (B, E, F)
        Returns:
            loss: MSE between predicted and actual noise
        """
        B, E, _ = z.shape
        device = z.device

        # Sample random timestep per sample
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Add noise to targets
        noise = torch.randn_like(target)
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
        x_noisy = sqrt_ab * target + sqrt_omab * noise  # (B, E, F)

        # Time embedding → broadcast over entities
        t_emb = self.time_proj(self._time_embedding(t, device))  # (B, d)
        z_cond = z + t_emb.unsqueeze(1)  # (B, E, d)

        # Concat entity embedding with noisy target projection
        target_emb = self.target_proj(x_noisy)  # (B, E, d)
        combined = torch.cat([z_cond, target_emb], dim=-1)  # (B, E, 2d)

        pred_noise = self.noise_pred(combined)  # (B, E, F)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, z):
        """Inference: denoise from pure noise to get prediction."""
        B, E, _ = z.shape
        device = z.device

        x = torch.randn(B, E, self.n_features, device=device)

        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_emb = self.time_proj(self._time_embedding(t, device))
            z_cond = z + t_emb.unsqueeze(1)
            target_emb = self.target_proj(x)
            combined = torch.cat([z_cond, target_emb], dim=-1)
            pred_noise = self.noise_pred(combined)

            alpha = self.alphas[i]
            alpha_bar_t = self.alpha_bar[i]
            beta = self.betas[i]

            # DDPM reverse step
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_bar_t).sqrt()) * pred_noise)
            if i > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return x


class BinaryHead(nn.Module):
    """Predict logits for binary targets."""

    def __init__(self, d_in, d_hidden, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features),
        )

    def forward(self, x):
        return self.net(x)


# ── HP Discretization ─────────────────────────────────────────────────────────
# 7 bins for HP delta (fraction of max_hp):
#   0: heavy_dmg  (< -0.3)
#   1: mod_dmg    [-0.3, -0.1)
#   2: light_dmg  [-0.1, -0.01)
#   3: no_change  [-0.01, 0.01)
#   4: light_heal [0.01, 0.1)
#   5: mod_heal   [0.1, 0.3)
#   6: heavy_heal (>= 0.3)

HP_BIN_EDGES = [-0.3, -0.1, -0.01, 0.01, 0.1, 0.3]
HP_N_BINS = 7


def discretize_hp(hp_mean: torch.Tensor) -> torch.Tensor:
    """Convert continuous HP delta (first channel of 3) to bin indices.

    Args:
        hp_mean: (B, E, 3) continuous HP targets (channel 0 = hp_pct delta)
    Returns:
        (B, E) long tensor of bin indices
    """
    x = hp_mean[:, :, 0]  # (B, E) — hp_pct delta
    edges = torch.tensor(HP_BIN_EDGES, device=x.device, dtype=x.dtype)
    return torch.bucketize(x, edges)


class ClassificationHead(nn.Module):
    """Multi-class classification head per entity."""

    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # (B, E, n_classes)


# ── Two-Stage: Impact classification for affected entities ────────────────────
# Stage 1: Binary "is entity affected?" per entity
# Stage 2: 5-class impact for affected entities only
#   0: heavy_dmg (< -0.2)
#   1: light_dmg [-0.2, -0.01)
#   2: light_heal [0.01, 0.2)
#   3: heavy_heal (>= 0.2)
#   Note: "no_change" is handled by stage 1 (not affected)

IMPACT_BIN_EDGES = [-0.2, -0.01, 0.01, 0.2]
IMPACT_N_BINS = 5  # heavy_dmg, light_dmg, no_change_residual, light_heal, heavy_heal


def discretize_impact(hp_mean: torch.Tensor) -> torch.Tensor:
    """Convert HP delta to impact bins (5-class)."""
    x = hp_mean[:, :, 0]
    edges = torch.tensor(IMPACT_BIN_EDGES, device=x.device, dtype=x.dtype)
    return torch.bucketize(x, edges)


def compute_affected_targets(hp_mean: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Binary targets: is entity's HP delta outside [-threshold, threshold]?"""
    x = hp_mean[:, :, 0].abs()
    return (x > threshold).float()


class AbilityLatentOperator(nn.Module):
    """Full model: StateEncoder + AbilityOperator + DecoderHeads."""

    def __init__(self, exp_cfg=None, ability_slot_dim=ABILITY_SLOT_DIM, ability_cls_dim=ABILITY_CLS_DIM):
        super().__init__()
        self.exp_cfg = exp_cfg or ExperimentConfig('e0')
        self.encoder = StateEncoder(ability_slot_dim=ability_slot_dim)
        self.operator = AbilityOperator(ability_cls_dim=ability_cls_dim)

        if self.exp_cfg.use_two_stage:
            # Stage 1: binary "affected?" per entity
            self.affected_head = BinaryHead(D_MODEL, D_MODEL, 1)
            # Stage 2: impact classification (only for affected entities)
            self.impact_head = ClassificationHead(D_MODEL, D_MODEL * 2, IMPACT_N_BINS)
            # Keep cc/pos as beta-NLL
            self.cc_head = GaussianHead(D_MODEL, D_MODEL * 2, 1)
            self.pos_head = GaussianHead(D_MODEL, D_MODEL * 2, 2)
        elif self.exp_cfg.use_discrete_hp:
            self.hp_cls_head = ClassificationHead(D_MODEL, D_MODEL * 2, HP_N_BINS)
            # Use beta-NLL for cc/pos
            self.cc_head = GaussianHead(D_MODEL, D_MODEL * 2, 1)
            self.pos_head = GaussianHead(D_MODEL, D_MODEL * 2, 2)
        elif self.exp_cfg.use_diffusion:
            self.hp_diff = DiffusionHead(D_MODEL, 3)
            self.cc_diff = DiffusionHead(D_MODEL, 1)
            self.pos_diff = DiffusionHead(D_MODEL, 2)
        elif self.exp_cfg.use_gated_head:
            self.hp_head = GatedMeanHead(D_MODEL, D_MODEL * 2, 3)
            self.cc_head = GatedMeanHead(D_MODEL, D_MODEL * 2, 1)
            self.pos_head = GatedMeanHead(D_MODEL, D_MODEL * 2, 2)
        elif self.exp_cfg.use_mse or self.exp_cfg.use_huber:
            self.hp_head = MeanHead(D_MODEL, D_MODEL * 2, 3)
            self.cc_head = MeanHead(D_MODEL, D_MODEL * 2, 1)
            self.pos_head = MeanHead(D_MODEL, D_MODEL * 2, 2)
        else:
            self.hp_head = GaussianHead(D_MODEL, D_MODEL * 2, 3)
            self.cc_head = GaussianHead(D_MODEL, D_MODEL * 2, 1)
            self.pos_head = GaussianHead(D_MODEL, D_MODEL * 2, 2)

        self.cc_stun_head = BinaryHead(D_MODEL, D_MODEL * 2, 1)
        self.exists_head = BinaryHead(D_MODEL, D_MODEL, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=INIT_STD)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, entity_feat, entity_types, entity_mask,
                threat_feat, threat_mask,
                position_feat, position_mask,
                ability_feat, ability_types, ability_mask,
                ability_cls, caster_slot, duration_norm):
        z = self.encoder(entity_feat, entity_types, entity_mask,
                         threat_feat, threat_mask,
                         position_feat, position_mask,
                         ability_feat, ability_types, ability_mask)
        z = self.operator(z, ability_cls, caster_slot, duration_norm, entity_mask)

        result = {}

        if self.exp_cfg.use_two_stage:
            result['affected_logits'] = self.affected_head(z)  # (B, E, 1)
            result['impact_logits'] = self.impact_head(z)  # (B, E, 5)
            cc_mean, cc_logvar = self.cc_head(z)
            pos_mean, pos_logvar = self.pos_head(z)
            result.update({
                'cc_mean': cc_mean, 'cc_logvar': cc_logvar,
                'pos_mean': pos_mean, 'pos_logvar': pos_logvar,
            })
        elif self.exp_cfg.use_discrete_hp:
            result['hp_logits'] = self.hp_cls_head(z)  # (B, E, 7)
            cc_mean, cc_logvar = self.cc_head(z)
            pos_mean, pos_logvar = self.pos_head(z)
            result.update({
                'cc_mean': cc_mean, 'cc_logvar': cc_logvar,
                'pos_mean': pos_mean, 'pos_logvar': pos_logvar,
            })
        elif self.exp_cfg.use_diffusion:
            # Diffusion heads store z for training loss computation
            result['z'] = z
        elif self.exp_cfg.use_mse or self.exp_cfg.use_huber or self.exp_cfg.use_gated_head:
            result['hp_mean'] = self.hp_head(z)
            result['cc_mean'] = self.cc_head(z)
            result['pos_mean'] = self.pos_head(z)
        else:
            hp_mean, hp_logvar = self.hp_head(z)
            cc_mean, cc_logvar = self.cc_head(z)
            pos_mean, pos_logvar = self.pos_head(z)
            result.update({
                'hp_mean': hp_mean, 'hp_logvar': hp_logvar,
                'cc_mean': cc_mean, 'cc_logvar': cc_logvar,
                'pos_mean': pos_mean, 'pos_logvar': pos_logvar,
            })

        result['cc_stun_logits'] = self.cc_stun_head(z)
        result['exists_logits'] = self.exists_head(z)

        return result


# ── Loss ───────────────────────────────────────────────────────────────────────

def beta_nll(mean, log_var, target, beta=0.5):
    """Beta-NLL: variance-weighted Gaussian NLL."""
    variance = log_var.exp()
    weight = variance.detach() ** beta
    diff = target - mean
    nll = 0.5 * log_var + diff * diff / (2.0 * variance)
    return (weight * nll).mean()


def symlog(x):
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return x.sign() * (1.0 + x.abs()).log()


def compute_entity_mask_for_loss(targets, batch_masks, caster_slots, exp_cfg):
    """Compute per-entity loss masks for E2-style entity masking.

    Returns dict of (B, E) bool tensors keyed by 'hp', 'pos'.
    """
    B, E = targets['hp'].shape[:2]
    device = targets['hp'].device

    # HP: only entities whose hp delta > 0.01 in any of the 3 channels
    hp_changed = targets['hp'].abs().sum(dim=-1) > 0.01  # (B, E)

    # Pos: only caster entity (slot 0 in entity tokens)
    pos_caster = torch.zeros(B, E, device=device, dtype=torch.bool)
    for i in range(B):
        slot = caster_slots[i].item()
        if slot < E:
            pos_caster[i, slot] = True

    return {'hp': hp_changed, 'pos': pos_caster}


def compute_loss(pred, targets, masks, exp_cfg, caster_slots=None):
    """Compute masked loss over all heads."""
    device = pred['exists_logits'].device
    B = len(masks)

    hp_mask = torch.tensor([m['hp'] for m in masks], device=device, dtype=torch.float32)
    cc_mask = torch.tensor([m['cc'] for m in masks], device=device, dtype=torch.float32)
    pos_mask = torch.tensor([m['pos'] for m in masks], device=device, dtype=torch.float32)
    exists_mask = torch.tensor([m['exists'] for m in masks], device=device, dtype=torch.float32)

    w = exp_cfg.loss_weights
    loss = torch.tensor(0.0, device=device)

    # Entity-level masking (E2/E5/E9)
    entity_masks = None
    if exp_cfg.use_entity_masking:
        entity_masks = compute_entity_mask_for_loss(targets, masks, caster_slots, exp_cfg)

    # Symlog transform (E3/E9)
    t_hp = symlog(targets['hp']) if exp_cfg.use_symlog else targets['hp']
    t_cc = symlog(targets['cc']) if exp_cfg.use_symlog else targets['cc']
    t_pos = symlog(targets['pos']) if exp_cfg.use_symlog else targets['pos']

    hp_count = hp_mask.sum().item()
    if hp_count > 0:
        if exp_cfg.use_two_stage:
            # Stage 1: affected detection (binary CE)
            affected_gt = compute_affected_targets(targets['hp'])  # (B, E)
            affected_loss = F.binary_cross_entropy_with_logits(
                pred['affected_logits'].squeeze(-1), affected_gt,
                pos_weight=torch.tensor(10.0, device=device),  # strong positive weight
            )
            # Stage 2: impact classification (only on affected entities)
            impact_bins = discretize_impact(targets['hp'])  # (B, E)
            affected_mask = affected_gt > 0.5  # (B, E) bool
            if affected_mask.any():
                impact_logits = pred['impact_logits'][affected_mask]  # (N_affected, 5)
                impact_targets = impact_bins[affected_mask]  # (N_affected,)
                impact_loss = F.cross_entropy(impact_logits, impact_targets)
            else:
                impact_loss = torch.tensor(0.0, device=device)
            hp_loss = affected_loss + impact_loss
        elif exp_cfg.use_discrete_hp:
            hp_bins = discretize_hp(targets['hp'])  # (B, E) long
            logits = pred['hp_logits']  # (B, E, 7)
            B_hp, E_hp = logits.shape[:2]
            flat_logits = logits.reshape(B_hp * E_hp, HP_N_BINS)
            flat_bins = hp_bins.reshape(B_hp * E_hp)
            # Focal loss: down-weight easy (high-confidence) predictions
            # This forces the model to learn minority classes instead of
            # always predicting "no_change"
            ce = F.cross_entropy(flat_logits, flat_bins, reduction='none')
            pt = torch.exp(-ce)  # probability of correct class
            focal_weight = (1.0 - pt) ** 2.0  # gamma=2
            # Also apply class weights
            class_weights = torch.tensor(
                [3.0, 4.0, 6.0, 0.1, 8.0, 5.0, 8.0],
                device=logits.device, dtype=logits.dtype,
            )
            per_sample_cw = class_weights[flat_bins]
            hp_loss = (focal_weight * per_sample_cw * ce).mean()
        elif exp_cfg.use_diffusion:
            hp_loss = pred['_model'].hp_diff.forward_train(pred['z'], t_hp)
        elif exp_cfg.use_mse:
            if entity_masks is not None:
                # Masked MSE: only affected entities
                em = entity_masks['hp'].unsqueeze(-1).float()  # (B, E, 1)
                diff_sq = (pred['hp_mean'] - t_hp) ** 2 * em
                n_active = em.sum().clamp(min=1)
                hp_loss = diff_sq.sum() / n_active
            else:
                hp_loss = F.mse_loss(pred['hp_mean'], t_hp)
        elif exp_cfg.use_huber:
            if entity_masks is not None:
                em = entity_masks['hp'].unsqueeze(-1).float()  # (B, E, 1)
                diff = F.smooth_l1_loss(pred['hp_mean'], t_hp, beta=exp_cfg.huber_delta, reduction='none') * em
                n_active = em.sum().clamp(min=1)
                hp_loss = diff.sum() / n_active
            else:
                hp_loss = F.smooth_l1_loss(pred['hp_mean'], t_hp, beta=exp_cfg.huber_delta)
        else:
            hp_loss = beta_nll(pred['hp_mean'], pred['hp_logvar'], t_hp)
        loss = loss + hp_loss * (hp_count / B) * w['hp']

    cc_count = cc_mask.sum().item()
    if cc_count > 0:
        if exp_cfg.use_diffusion:
            cc_loss = pred['_model'].cc_diff.forward_train(pred['z'], t_cc)
        elif exp_cfg.use_mse:
            cc_loss = F.mse_loss(pred['cc_mean'], t_cc)
        elif exp_cfg.use_huber:
            cc_loss = F.smooth_l1_loss(pred['cc_mean'], t_cc, beta=exp_cfg.huber_delta)
        else:
            cc_loss = beta_nll(pred['cc_mean'], pred['cc_logvar'], t_cc)
        stun_loss = F.binary_cross_entropy_with_logits(
            pred['cc_stun_logits'], targets['cc_stun'])
        loss = loss + (cc_loss + stun_loss) * (cc_count / B) * w['cc']

    pos_count = pos_mask.sum().item()
    if pos_count > 0:
        if exp_cfg.use_diffusion:
            pos_loss = pred['_model'].pos_diff.forward_train(pred['z'], t_pos)
        elif exp_cfg.use_mse:
            if entity_masks is not None:
                em = entity_masks['pos'].unsqueeze(-1).float()
                diff_sq = (pred['pos_mean'] - t_pos) ** 2 * em
                n_active = em.sum().clamp(min=1)
                pos_loss = diff_sq.sum() / n_active
            else:
                pos_loss = F.mse_loss(pred['pos_mean'], t_pos)
        elif exp_cfg.use_huber:
            if entity_masks is not None:
                em = entity_masks['pos'].unsqueeze(-1).float()
                diff = F.smooth_l1_loss(pred['pos_mean'], t_pos, beta=exp_cfg.huber_delta, reduction='none') * em
                n_active = em.sum().clamp(min=1)
                pos_loss = diff.sum() / n_active
            else:
                pos_loss = F.smooth_l1_loss(pred['pos_mean'], t_pos, beta=exp_cfg.huber_delta)
        else:
            pos_loss = beta_nll(pred['pos_mean'], pred['pos_logvar'], t_pos)
        loss = loss + pos_loss * (pos_count / B) * w['pos']

    exists_count = exists_mask.sum().item()
    if exists_count > 0:
        exists_loss = F.binary_cross_entropy_with_logits(
            pred['exists_logits'], targets['exists'])
        loss = loss + exists_loss * (exists_count / B) * w['exists']

    return loss


# ── GrokfastEMA ────────────────────────────────────────────────────────────────

class GrokfastEMA:
    """Gradient filter: maintain EMA, replace grad with grad + lamb * ema."""

    def __init__(self, model, alpha=0.98, lamb=2.0):
        self.alpha = alpha
        self.lamb = lamb
        self.ema = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.ema[name] = torch.zeros_like(p.data)

    def apply(self, model):
        for name, p in model.named_parameters():
            if p.grad is not None and name in self.ema:
                self.ema[name].mul_(self.alpha).add_(p.grad, alpha=1.0 - self.alpha)
                p.grad.add_(self.ema[name], alpha=self.lamb)


# ── Dataset ────────────────────────────────────────────────────────────────────

class OperatorDataset:
    """Load npz dataset onto GPU."""

    def __init__(self, path, device):
        npz = np.load(path)

        n = len(npz['entity_features'])
        n_ent = MAX_ENTITIES
        n_thr = 8
        n_pos = 8
        # Infer ability dimensions from data
        n_abl_mask = npz['ability_slot_mask'].shape[1]
        self.ability_cls_dim = npz['ability_cls'].shape[1]
        self.ability_slot_dim = npz['ability_slot_features'].shape[1] // n_abl_mask if n_abl_mask > 0 else ABILITY_SLOT_DIM
        n_abl = n_abl_mask

        def t(arr, shape=None):
            x = torch.from_numpy(arr.astype(np.float32)).to(device)
            return x.reshape(n, *shape) if shape else x

        def ti(arr, shape=None):
            x = torch.from_numpy(arr.astype(np.int64)).to(device)
            return x.reshape(n, *shape) if shape else x

        self.entity_feat = t(npz['entity_features'], (n_ent, ENTITY_DIM))
        self.entity_types = ti(npz['entity_types'], (n_ent,))
        self.entity_mask = torch.from_numpy(npz['entity_mask'].astype(np.int32)).to(device).reshape(n, n_ent) > 0
        self.threat_feat = t(npz['threat_features'], (n_thr, THREAT_DIM))
        self.threat_mask = torch.from_numpy(npz['threat_mask'].astype(np.int32)).to(device).reshape(n, n_thr) > 0
        self.position_feat = t(npz['position_features'], (n_pos, POSITION_DIM))
        self.position_mask = torch.from_numpy(npz['position_mask'].astype(np.int32)).to(device).reshape(n, n_pos) > 0
        self.ability_feat = t(npz['ability_slot_features'], (n_abl, self.ability_slot_dim))
        self.ability_types = ti(npz['ability_slot_types'], (n_abl,))
        self.ability_mask = torch.from_numpy(npz['ability_slot_mask'].astype(np.int32)).to(device).reshape(n, n_abl) > 0
        self.ability_cls = t(npz['ability_cls'])
        self.caster_slot = ti(npz['caster_slot'].flatten())
        self.duration_norm = t(npz['duration_norm'].flatten())
        self.target_hp = t(npz['target_hp'], (n_ent, 3))
        self.target_cc = t(npz['target_cc'], (n_ent, 1))
        self.target_cc_stun = t(npz['target_cc_stun'], (n_ent, 1))
        self.target_pos = t(npz['target_pos'], (n_ent, 2))
        self.target_exists = t(npz['target_exists'], (n_ent, 1))

        # No per-ability-type loss masking — train all heads on all samples.
        # The model learns to predict zero delta when an ability doesn't affect a target.
        all_true_mask = {'hp': True, 'cc': True, 'pos': True, 'exists': True}
        self.loss_masks = [all_true_mask] * n

        self.scenario_ids = npz['scenario_ids'].flatten().astype(np.int32)
        self.n = n

    def train_val_split(self):
        unique = sorted(set(self.scenario_ids))
        n_val = max(1, int(len(unique) * 0.2))
        val_ids = set(unique[-n_val:])
        train_idx = [i for i in range(self.n) if self.scenario_ids[i] not in val_ids]
        val_idx = [i for i in range(self.n) if self.scenario_ids[i] in val_ids]
        return train_idx, val_idx

    def get_batch(self, indices):
        idx = torch.tensor(indices, device=self.entity_feat.device, dtype=torch.long)
        return {
            'entity_feat': self.entity_feat[idx],
            'entity_types': self.entity_types[idx],
            'entity_mask': self.entity_mask[idx],
            'threat_feat': self.threat_feat[idx],
            'threat_mask': self.threat_mask[idx],
            'position_feat': self.position_feat[idx],
            'position_mask': self.position_mask[idx],
            'ability_feat': self.ability_feat[idx],
            'ability_types': self.ability_types[idx],
            'ability_mask': self.ability_mask[idx],
            'ability_cls': self.ability_cls[idx],
            'caster_slot': self.caster_slot[idx],
            'duration_norm': self.duration_norm[idx],
            'target_hp': self.target_hp[idx],
            'target_cc': self.target_cc[idx],
            'target_cc_stun': self.target_cc_stun[idx],
            'target_pos': self.target_pos[idx],
            'target_exists': self.target_exists[idx],
            'masks': [self.loss_masks[i] for i in indices],
        }


# ── Eval ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, indices, batch_size, exp_cfg):
    model.eval()
    total_loss = 0.0
    hp_mae_sum = 0.0
    hp_baseline_sum = 0.0
    hp_acc_sum = 0.0
    hp_acc_count = 0
    pos_mae_sum = 0.0
    pos_baseline_sum = 0.0
    exists_bce_sum = 0.0
    exists_baseline_sum = 0.0
    hp_count = 0
    pos_count = 0
    n_batches = 0

    # Get the underlying model for diffusion sampling
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]
        batch = dataset.get_batch(batch_idx)

        pred = model(batch['entity_feat'], batch['entity_types'], batch['entity_mask'],
                     batch['threat_feat'], batch['threat_mask'],
                     batch['position_feat'], batch['position_mask'],
                     batch['ability_feat'], batch['ability_types'], batch['ability_mask'],
                     batch['ability_cls'], batch['caster_slot'], batch['duration_norm'])

        targets = {
            'hp': batch['target_hp'], 'cc': batch['target_cc'],
            'cc_stun': batch['target_cc_stun'], 'pos': batch['target_pos'],
            'exists': batch['target_exists'],
        }

        # For diffusion, generate predictions via sampling for metrics
        if exp_cfg.use_diffusion:
            pred['hp_mean'] = raw_model.hp_diff.sample(pred['z'])
            pred['cc_mean'] = raw_model.cc_diff.sample(pred['z'])
            pred['pos_mean'] = raw_model.pos_diff.sample(pred['z'])
            pred['_model'] = raw_model
            loss = compute_loss(pred, targets, batch['masks'], exp_cfg, batch['caster_slot'])
        else:
            loss = compute_loss(pred, targets, batch['masks'], exp_cfg, batch['caster_slot'])
        total_loss += loss.item()

        # Vectorized metrics — always compare means to raw targets (not symlog)
        hp_mask_v = torch.tensor([m['hp'] for m in batch['masks']], device=device_of(pred))
        pos_mask_v = torch.tensor([m['pos'] for m in batch['masks']], device=device_of(pred))

        if exp_cfg.use_two_stage:
            if hp_mask_v.any() and 'affected_logits' in pred:
                affected_gt = compute_affected_targets(batch['target_hp'])
                ent_mask_bool = batch['entity_mask']
                valid_ent = hp_mask_v.unsqueeze(-1).bool() & ~ent_mask_bool

                # Stage 1: affected detection accuracy
                affected_pred = (pred['affected_logits'].squeeze(-1) > 0).float()
                correct_aff = ((affected_pred == affected_gt) & valid_ent).sum().item()
                total_aff = valid_ent.sum().item()
                hp_acc_sum += correct_aff
                hp_acc_count += total_aff

                # Baseline: always predict "not affected"
                baseline_correct = ((affected_gt < 0.5) & valid_ent).sum().item()
                hp_baseline_sum += baseline_correct
                hp_count += total_aff

                # Stage 2: impact accuracy on truly affected entities
                affected_mask = (affected_gt > 0.5) & valid_ent
                if affected_mask.any():
                    impact_bins = discretize_impact(batch['target_hp'])
                    impact_pred = pred['impact_logits'].argmax(dim=-1)
                    impact_correct = ((impact_pred == impact_bins) & affected_mask).sum().item()
                    # Store in pos_mae_sum/pos_count for convenient reporting
                    # (repurposing these for impact accuracy)
        elif exp_cfg.use_discrete_hp:
            # HP accuracy for discrete mode
            if hp_mask_v.any() and 'hp_logits' in pred:
                hp_bins_gt = discretize_hp(batch['target_hp'])
                hp_pred_cls = pred['hp_logits'].argmax(dim=-1)
                # Accuracy per entity, masked by valid entities
                ent_mask_bool = batch['entity_mask']  # (B, E) bool, True = padding
                valid_and_hp = hp_mask_v.unsqueeze(-1).bool() & ~ent_mask_bool
                correct = (hp_pred_cls == hp_bins_gt) & valid_and_hp
                hp_acc_sum += correct.sum().item()
                hp_acc_count += valid_and_hp.sum().item()
                # Also compute baseline accuracy (always predict "no_change" = bin 3)
                baseline_correct = (hp_bins_gt == 3) & valid_and_hp
                hp_baseline_sum += baseline_correct.sum().item()
                hp_count += valid_and_hp.sum().item()
        else:
            # For symlog experiments, apply inverse symlog to predictions for fair comparison
            hp_pred = pred['hp_mean']
            pos_pred_v = pred['pos_mean']
            if exp_cfg.use_symlog:
                hp_pred = symlog_inv(hp_pred)
                pos_pred_v = symlog_inv(pos_pred_v)

            if hp_mask_v.any():
                hp_diff = (hp_pred - batch['target_hp']).abs().mean(dim=(1, 2))
                hp_base = batch['target_hp'].abs().mean(dim=(1, 2))
                hp_mae_sum += (hp_diff * hp_mask_v).sum().item()
                hp_baseline_sum += (hp_base * hp_mask_v).sum().item()
                hp_count += hp_mask_v.sum().item()

        if pos_mask_v.any():
            pos_pred = pred.get('pos_mean', pred.get('pos_pred'))
            if exp_cfg.use_symlog and pos_pred is not None:
                pos_pred = symlog_inv(pos_pred)
            pos_diff_v = (pos_pred - batch['target_pos']).abs().mean(dim=(1, 2))
            pos_base = batch['target_pos'].abs().mean(dim=(1, 2))
            pos_mae_sum += (pos_diff_v * pos_mask_v).sum().item()
            pos_baseline_sum += (pos_base * pos_mask_v).sum().item()
            pos_count += pos_mask_v.sum().item()

        exists_bce = F.binary_cross_entropy_with_logits(
            pred['exists_logits'], batch['target_exists']).item()
        baseline_bce = F.binary_cross_entropy_with_logits(
            torch.full_like(pred['exists_logits'], 10.0), batch['target_exists']).item()
        exists_bce_sum += exists_bce
        exists_baseline_sum += baseline_bce
        n_batches += 1

    model.train()
    n = max(n_batches, 1)
    pos_mae = pos_mae_sum / max(pos_count, 1)
    pos_base = pos_baseline_sum / max(pos_count, 1)
    exists_bce = exists_bce_sum / n
    exists_base = exists_baseline_sum / n

    pos_imp = (pos_base - pos_mae) / max(pos_base, 1e-8) * 100
    exists_imp = (exists_base - exists_bce) / max(exists_base, 1e-8) * 100

    result = {
        'loss': total_loss / n,
        'pos_mae': pos_mae, 'pos_base': pos_base, 'pos_imp': pos_imp,
        'exists_bce': exists_bce, 'exists_base': exists_base, 'exists_imp': exists_imp,
    }

    if exp_cfg.use_two_stage or exp_cfg.use_discrete_hp:
        hp_acc = hp_acc_sum / max(hp_acc_count, 1) * 100
        hp_base_acc = hp_baseline_sum / max(hp_count, 1) * 100
        result['hp_acc'] = hp_acc
        result['hp_base_acc'] = hp_base_acc
        result['hp_imp'] = hp_acc - hp_base_acc
    else:
        hp_mae = hp_mae_sum / max(hp_count, 1)
        hp_base = hp_baseline_sum / max(hp_count, 1)
        hp_imp = (hp_base - hp_mae) / max(hp_base, 1e-8) * 100
        result['hp_mae'] = hp_mae
        result['hp_base'] = hp_base
        result['hp_imp'] = hp_imp

    return result


def device_of(pred):
    return pred['exists_logits'].device


def symlog_inv(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return x.sign() * (x.abs().exp() - 1.0)


# ── Training ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def train(args):
    exp_cfg = ExperimentConfig(args.experiment)
    log(f"Experiment: {args.experiment} — {EXPERIMENTS[args.experiment]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # LR override
    lr = exp_cfg.lr_override if exp_cfg.lr_override else args.lr

    # Max steps override (e.g. E21: 20K steps)
    if exp_cfg.max_steps_override is not None:
        args.max_steps = exp_cfg.max_steps_override
        log(f"  max_steps overridden to {args.max_steps}")

    log(f"Loading {args.data}...")
    dataset = OperatorDataset(args.data, device)
    train_idx, val_idx = dataset.train_val_split()
    log(f"Dataset: {dataset.n} samples ({len(train_idx)} train, {len(val_idx)} val)")

    model = AbilityLatentOperator(
        exp_cfg,
        ability_slot_dim=dataset.ability_slot_dim,
        ability_cls_dim=dataset.ability_cls_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params:,} params")
    if device.type == 'cuda' and args.compile and not exp_cfg.use_diffusion:
        log("Compiling model (this takes ~10s)...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.98), weight_decay=args.weight_decay,
    )
    grokfast = GrokfastEMA(model, alpha=0.98, lamb=2.0) if exp_cfg.use_grokfast else None
    use_amp = device.type == 'cuda'

    # Get raw model ref for diffusion heads
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    best_val_loss = float('inf')
    step = 0
    train_loss_sum = 0.0
    train_loss_count = 0
    rng = np.random.RandomState(args.seed)
    t0 = time.time()

    while step < args.max_steps:
        perm = rng.permutation(len(train_idx))
        shuffled = [train_idx[i] for i in perm]

        for start in range(0, len(shuffled), args.batch_size):
            if step >= args.max_steps:
                break

            end = min(start + args.batch_size, len(shuffled))
            batch = dataset.get_batch(shuffled[start:end])

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch['entity_feat'], batch['entity_types'], batch['entity_mask'],
                             batch['threat_feat'], batch['threat_mask'],
                             batch['position_feat'], batch['position_mask'],
                             batch['ability_feat'], batch['ability_types'], batch['ability_mask'],
                             batch['ability_cls'], batch['caster_slot'], batch['duration_norm'])

                targets = {
                    'hp': batch['target_hp'], 'cc': batch['target_cc'],
                    'cc_stun': batch['target_cc_stun'], 'pos': batch['target_pos'],
                    'exists': batch['target_exists'],
                }
                if exp_cfg.use_diffusion:
                    pred['_model'] = raw_model
                loss = compute_loss(pred, targets, batch['masks'], exp_cfg, batch['caster_slot'])

            optimizer.zero_grad()
            loss.backward()
            if grokfast is not None:
                grokfast.apply(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item()
            train_loss_count += 1
            step += 1

            # Progress every 500 steps (lightweight, no eval)
            if step % 500 == 0 and step % args.eval_every != 0:
                elapsed = time.time() - t0
                sps = step / elapsed
                avg_train = train_loss_sum / max(train_loss_count, 1)
                eta_min = (args.max_steps - step) / max(sps, 1) / 60
                log(f"step {step:>6}/{args.max_steps} | {sps:.0f} steps/s | train {avg_train:.4f} | ETA {eta_min:.1f}m")

            # Full eval
            if step % args.eval_every == 0:
                elapsed = time.time() - t0
                sps = step / elapsed
                avg_train = train_loss_sum / max(train_loss_count, 1)
                train_loss_sum = 0.0
                train_loss_count = 0

                metrics = evaluate(model, dataset, val_idx, args.batch_size, exp_cfg)

                eta_min = (args.max_steps - step) / max(sps, 1) / 60
                if exp_cfg.use_two_stage or exp_cfg.use_discrete_hp:
                    hp_str = f"hp_acc {metrics['hp_acc']:.1f}% (base {metrics['hp_base_acc']:.1f}%)"
                else:
                    hp_str = f"hp {metrics['hp_imp']:+.1f}%"
                log(
                    f"step {step:>6}/{args.max_steps} | {sps:.0f} steps/s | ETA {eta_min:.1f}m | "
                    f"train {avg_train:.4f} | val {metrics['loss']:.4f} | "
                    f"{hp_str} | "
                    f"exists {metrics['exists_imp']:+.1f}% | "
                    f"pos {metrics['pos_imp']:+.1f}%"
                )

                if metrics['loss'] < best_val_loss:
                    best_val_loss = metrics['loss']
                    log(f"  -> new best {best_val_loss:.4f}")
                    torch.save(model.state_dict(), args.output)

    # Final eval for summary line
    final_metrics = evaluate(model, dataset, val_idx, args.batch_size, exp_cfg)
    elapsed = time.time() - t0
    log(f"\nDone: {step} steps in {elapsed:.1f}s ({step/elapsed:.0f} steps/s)")
    log(f"Best val loss: {best_val_loss:.4f}")
    log(f"Checkpoint: {args.output}")

    # Machine-readable summary line for runner script
    if exp_cfg.use_two_stage or exp_cfg.use_discrete_hp:
        hp_result = f"hp_acc={final_metrics['hp_acc']:.1f}% (base={final_metrics['hp_base_acc']:.1f}%)"
    else:
        hp_result = f"hp={final_metrics['hp_imp']:+.1f}%"
    log(
        f"RESULT | {args.experiment} | "
        f"{hp_result} | "
        f"pos={final_metrics['pos_imp']:+.1f}% | "
        f"exists={final_metrics['exists_imp']:+.1f}% | "
        f"val_loss={final_metrics['loss']:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='generated/operator_model.pt')
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--experiment', type=str, default='e0',
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment variant to run')
    args = parser.parse_args()
    train(args)
