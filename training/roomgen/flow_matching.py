"""Rectified flow (linear interpolation) for room grid diffusion.

Objective: learn velocity field v(x_t, t) where x_t = (1-t)*x_0 + t*x_1
and x_0 is data, x_1 is noise.

For obstacle types (discrete): use continuous relaxation — embed one-hot
to continuous vectors, diffuse in embedding space, argmax at output.
"""

import math

import torch
import torch.nn.functional as F


def logit_normal_sample(shape: tuple, device: torch.device, loc: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Sample timesteps from logit-normal distribution.

    Concentrates samples around t=0.5 where the flow is hardest to learn.
    """
    u = torch.randn(shape, device=device) * scale + loc
    t = torch.sigmoid(u)
    return t


def rectified_flow_forward(
    obs_type: torch.Tensor,
    height: torch.Tensor,
    elevation: torch.Tensor,
    n_obs_types: int = 9,
) -> dict:
    """Compute noisy inputs and velocity targets for rectified flow training.

    Args:
        obs_type: (B, D, W) long — ground truth obstacle types
        height: (B, D, W) float — ground truth heights
        elevation: (B, D, W) float — ground truth elevations

    Returns dict with:
        noisy_obs: (B, D, W) long — noisy obstacle type indices
        noisy_height: (B, D, W) float
        noisy_elevation: (B, D, W) float
        target_obs_logits: (B, D, W, 9) float — velocity target for obs types
        target_height: (B, D, W) float — velocity target for height
        target_elevation: (B, D, W) float — velocity target for elevation
        t: (B,) float — sampled timesteps
    """
    B = obs_type.shape[0]
    device = obs_type.device

    # Sample timesteps from logit-normal
    t = logit_normal_sample((B,), device)  # (B,)
    t_expand = t.view(B, 1, 1)  # (B, 1, 1) for broadcasting

    # --- Continuous channels: linear interpolation ---
    noise_h = torch.randn_like(height)
    noise_e = torch.randn_like(elevation)

    noisy_height = (1 - t_expand) * height + t_expand * noise_h
    noisy_elevation = (1 - t_expand) * elevation + t_expand * noise_e

    # Velocity targets for continuous: v = noise - data
    target_height = noise_h - height
    target_elevation = noise_e - elevation

    # --- Discrete channel: continuous relaxation ---
    # Convert to one-hot, interpolate with uniform noise
    obs_onehot = F.one_hot(obs_type, n_obs_types).float()  # (B, D, W, 9)
    noise_obs = torch.randn_like(obs_onehot)  # Gaussian noise in logit space

    t_obs = t.view(B, 1, 1, 1)
    noisy_obs_logits = (1 - t_obs) * obs_onehot + t_obs * noise_obs

    # Velocity target for discrete: noise - one_hot
    target_obs_logits = noise_obs - obs_onehot

    # Convert noisy logits to indices for embedding input
    noisy_obs = noisy_obs_logits.argmax(dim=-1)  # (B, D, W)

    return {
        "noisy_obs": noisy_obs,
        "noisy_height": noisy_height,
        "noisy_elevation": noisy_elevation,
        "target_obs_logits": target_obs_logits,
        "target_height": target_height,
        "target_elevation": target_elevation,
        "t": t,
    }


def compute_loss(
    pred: dict,
    target: dict,
    mask: torch.Tensor,
) -> dict:
    """Compute velocity prediction loss.

    pred: output from ELITDiT.forward()
    target: output from rectified_flow_forward()
    mask: (B, D, W) bool
    """
    # Obstacle type: MSE on logits (or cross-entropy on direction)
    pred_obs = pred["obs_logits"]  # (B, D, W, 9)
    target_obs = target["target_obs_logits"]  # (B, D, W, 9)
    mask_4d = mask.unsqueeze(-1).float()
    obs_loss = ((pred_obs - target_obs) ** 2 * mask_4d).sum() / mask_4d.sum().clamp(min=1) / pred_obs.shape[-1]

    # Height: MSE
    mask_f = mask.float()
    height_loss = ((pred["height"] - target["target_height"]) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1)

    # Elevation: MSE
    elev_loss = ((pred["elevation"] - target["target_elevation"]) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1)

    total = obs_loss + height_loss + elev_loss

    return {
        "total": total,
        "obs": obs_loss,
        "height": height_loss,
        "elevation": elev_loss,
    }


@torch.no_grad()
def euler_sample(
    model,
    text_emb: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    n_steps: int = 40,
    cfg_scale: float = 3.0,
    j_budget: int | None = None,
    j_budget_uncond: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Euler reverse sampling with optional classifier-free guidance (CCFG).

    Args:
        model: ELITDiT
        text_emb: (B, d_text) — conditioned text embedding
        width: (B,) long
        depth: (B,) long
        n_steps: number of Euler steps
        cfg_scale: guidance scale (1.0 = no guidance)
        j_budget: latent tokens per group for conditioned path
        j_budget_uncond: latent tokens per group for unconditioned path (CCFG)

    Returns dict with final predictions.
    """
    B = text_emb.shape[0]
    D_max = depth.max().item()
    W_max = width.max().item()

    # Initialize from noise
    obs_type = torch.randint(0, 9, (B, D_max, W_max), device=device)
    height = torch.randn(B, D_max, W_max, device=device)
    elevation = torch.randn(B, D_max, W_max, device=device)

    # Build mask
    mask = torch.zeros(B, D_max, W_max, dtype=torch.bool, device=device)
    for i in range(B):
        mask[i, :depth[i], :width[i]] = True

    # Null text embedding for CFG
    null_text = torch.zeros_like(text_emb)

    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = 1.0 - step * dt
        t = torch.full((B,), t_val, device=device)

        # Conditioned prediction
        pred_cond = model(
            obs_type, height, elevation, t, text_emb, width, depth,
            mask=mask, j_budget=j_budget,
        )

        if cfg_scale > 1.0:
            # Unconditioned prediction (CCFG: reduced budget)
            j_unc = j_budget_uncond or max(2, (j_budget or model.j_per_group) // 3)
            pred_uncond = model(
                obs_type, height, elevation, t, null_text, width, depth,
                mask=mask, j_budget=j_unc,
            )

            # CFG combination
            obs_v = pred_uncond["obs_logits"] + cfg_scale * (
                pred_cond["obs_logits"] - pred_uncond["obs_logits"]
            )
            h_v = pred_uncond["height"] + cfg_scale * (
                pred_cond["height"] - pred_uncond["height"]
            )
            e_v = pred_uncond["elevation"] + cfg_scale * (
                pred_cond["elevation"] - pred_uncond["elevation"]
            )
        else:
            obs_v = pred_cond["obs_logits"]
            h_v = pred_cond["height"]
            e_v = pred_cond["elevation"]

        # Euler step (reverse: subtract velocity)
        # In rectified flow, x_{t-dt} = x_t - dt * v(x_t, t)
        height = height - dt * h_v
        elevation = elevation - dt * e_v

        # For discrete: update logits accumulator and re-argmax
        # Maintain soft logits and step them
        if step == 0:
            obs_logits = torch.zeros(B, D_max, W_max, 9, device=device)
        obs_logits = obs_logits - dt * obs_v
        obs_type = obs_logits.argmax(dim=-1)

    return {
        "obs_type": obs_type,
        "obs_logits": obs_logits,
        "height": height,
        "elevation": elevation,
        "mask": mask,
    }
