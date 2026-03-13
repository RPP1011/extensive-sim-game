"""Grokfast: gradient filtering for accelerated grokking.

Implements the EMA gradient filter from Lee et al. (2405.20233, 2024).
Amplifies slow-varying gradient components (generalization signal) over
fast-varying components (memorization signal).

Usage:
    from grokfast import GrokfastEMA
    gf = GrokfastEMA(model, alpha=0.98, lamb=2.0)

    # In training loop, after loss.backward():
    gf.step()          # modifies gradients in-place
    optimizer.step()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GrokfastEMA:
    """Exponential moving average gradient filter.

    After loss.backward(), call .step() to amplify slow gradient components.
    This is applied *before* optimizer.step().

    Parameters
    ----------
    model : nn.Module
    alpha : float
        EMA decay factor. Higher = more smoothing = stronger amplification
        of slow components. Default 0.98 per paper.
    lamb : float
        Amplification factor for the slow (EMA) gradient component.
        Default 2.0 per paper.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.98, lamb: float = 2.0):
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.grads: Optional[dict[str, torch.Tensor]] = None

    def step(self):
        """Filter gradients in-place. Call after backward(), before optimizer.step()."""
        if self.grads is None:
            self.grads = {
                n: p.grad.data.detach().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            return

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in self.grads:
                    self.grads[n] = p.grad.data.detach().clone()
                else:
                    self.grads[n] = self.grads[n] * self.alpha + p.grad.data.detach() * (1 - self.alpha)
                p.grad.data = p.grad.data + self.grads[n] * self.lamb
