"""Closed-form Continuous-time cell (Hasani et al.).

Input-dependent time constants learn when to update vs retain memories.
Drop-in replacement for TemporalGRU with the same forward signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CFC_H_DIM = 256  # default CfC hidden dimension (wider than GRU for richer temporal state)


class CfCCell(nn.Module):
    """Closed-form Continuous-time cell (Hasani et al.).

    Input-dependent time constants learn when to update vs retain memories.
    Drop-in replacement for TemporalGRU with the same forward signature.
    """

    def __init__(self, d_model: int, h_dim: int = CFC_H_DIM):
        super().__init__()
        self.h_dim = h_dim
        total = d_model + h_dim
        self.f_gate = nn.Linear(total, h_dim)
        self.h_gate = nn.Linear(total, h_dim)
        self.t_a = nn.Linear(total, h_dim)
        self.t_b = nn.Linear(total, h_dim)
        self.proj = nn.Linear(h_dim, d_model)

        # Init: start near full memory retention
        nn.init.constant_(self.f_gate.bias, 1.0)
        nn.init.constant_(self.t_b.bias, 1.0)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor | None = None,
        delta_t: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model) -- pooled entity representation for this tick
            h_prev: (B, h_dim) -- hidden state from previous tick, or None
            delta_t: time scaling factor (1.0 = standard tick interval)
        Returns:
            projected: (B, d_model) -- temporally enriched representation
            h_new: (B, h_dim) -- hidden state to propagate
        """
        if h_prev is None:
            h_prev = torch.zeros(x.shape[0], self.h_dim, device=x.device)
        combined = torch.cat([x, h_prev], dim=-1)
        f = torch.sigmoid(self.f_gate(combined))
        candidate = torch.tanh(self.h_gate(combined))
        t = torch.sigmoid(self.t_a(combined)) * delta_t + self.t_b(combined)
        h_new = torch.tanh(f * h_prev + (1 - f) * candidate * t)
        return self.proj(h_new), h_new
