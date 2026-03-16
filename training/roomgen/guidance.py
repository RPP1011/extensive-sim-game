"""PhyScene-style differentiable guidance functions for room generation.

These functions compute constraint losses on the spatial domain output (after Write)
and can be used to steer diffusion sampling toward valid, tactically interesting layouts.
"""

import torch
import torch.nn.functional as F


def collision_loss(obs_logits: torch.Tensor) -> torch.Tensor:
    """Penalize cells with ambiguous obstacle predictions (high entropy on non-floor types).

    obs_logits: (B, D, W, 9) — pre-softmax type logits
    """
    probs = F.softmax(obs_logits, dim=-1)
    non_floor_probs = probs[..., 1:]  # exclude floor channel
    non_floor_mass = non_floor_probs.sum(dim=-1)  # (B, D, W)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)  # (B, D, W)
    return (non_floor_mass * entropy).mean()


def boundary_loss(obs_logits: torch.Tensor, width: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Enforce perimeter cells are walls, interior cells near edges prefer floor.

    obs_logits: (B, D, W, 9)
    width: (B,) int
    depth: (B,) int
    """
    B, D_max, W_max, C = obs_logits.shape
    probs = F.softmax(obs_logits, dim=-1)
    device = obs_logits.device

    total_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        w, d = width[i].item(), depth[i].item()
        p = probs[i, :d, :w]  # (d, w, 9)

        # Perimeter should be wall (type=1)
        # Top row, bottom row, left col, right col
        perim_probs = torch.cat([
            p[0, :, 1],       # top row, wall prob
            p[d-1, :, 1],     # bottom row
            p[1:d-1, 0, 1],   # left col (excl corners)
            p[1:d-1, w-1, 1], # right col
        ])
        perim_loss = -perim_probs.clamp(min=1e-8).log().mean()

        # Inner ring (1 cell inside perimeter) should prefer floor
        if d > 2 and w > 2:
            inner = torch.cat([
                p[1, 1:w-1].reshape(-1, C),   # row 1
                p[d-2, 1:w-1].reshape(-1, C), # row d-2
                p[2:d-2, 1].reshape(-1, C),   # col 1
                p[2:d-2, w-2].reshape(-1, C), # col w-2
            ])
            inner_loss = inner[:, 1:].sum(dim=-1).mean() * 0.1
        else:
            inner_loss = torch.tensor(0.0, device=device)

        total_loss = total_loss + perim_loss + inner_loss

    return total_loss / B


def connectivity_loss(
    obs_logits: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    n_iterations: int = 30,
) -> torch.Tensor:
    """Soft pathfinding penalty using iterative Bellman relaxation.

    Ensures paths exist between player spawn side (left) and enemy spawn side (right).

    obs_logits: (B, D, W, 9)
    """
    B = obs_logits.shape[0]
    device = obs_logits.device
    probs = F.softmax(obs_logits, dim=-1)

    total_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        w, d = width[i].item(), depth[i].item()
        floor_prob = probs[i, :d, :w, 0]  # (d, w) — probability walkable

        # Occupancy: high = likely blocked
        occupancy = 1.0 - floor_prob

        # Initialize distance from left column (player side)
        dist = torch.full((d, w), 1e6, device=device)
        # Seed: column 1 (just inside perimeter)
        mid_row = d // 2
        for r in range(max(1, mid_row - 2), min(d - 1, mid_row + 3)):
            dist[r, 1] = 0.0

        # Iterative Bellman updates
        for _ in range(n_iterations):
            # 4-directional neighbors
            up = F.pad(dist[1:, :], (0, 0, 0, 1), value=1e6)
            down = F.pad(dist[:-1, :], (0, 0, 1, 0), value=1e6)
            left = F.pad(dist[:, 1:], (0, 1, 0, 0), value=1e6)
            right = F.pad(dist[:, :-1], (1, 0, 0, 0), value=1e6)

            min_neighbor = torch.stack([up, down, left, right]).min(dim=0).values + 1.0
            # Blocked cells have high cost
            cost = occupancy * 100.0
            dist = torch.minimum(dist, min_neighbor + cost)

        # Check distance to right side (enemy spawn)
        goal_col = w - 2  # just inside right perimeter
        goal_dists = dist[max(1, mid_row - 2):min(d - 1, mid_row + 3), goal_col]
        total_loss = total_loss + goal_dists.min() / (w + d)

    return total_loss / B


def cover_density_loss(obs_logits: torch.Tensor, target_density: float = 0.3) -> torch.Tensor:
    """Penalize deviation from target cover density.

    Cover density = fraction of walkable cells adjacent to a blocked cell.
    """
    probs = F.softmax(obs_logits, dim=-1)
    blocked = 1.0 - probs[..., 0]  # non-floor probability (B, D, W)
    walkable = probs[..., 0]

    # Soft "near blocked" via max-pooling
    near_blocked = F.max_pool2d(
        blocked.unsqueeze(1),  # (B, 1, D, W)
        kernel_size=3, padding=1, stride=1,
    ).squeeze(1)  # (B, D, W)

    near_cover = walkable * near_blocked
    density = near_cover.sum(dim=(1, 2)) / walkable.sum(dim=(1, 2)).clamp(min=1)
    return ((density - target_density) ** 2).mean()


def blocked_pct_loss(
    obs_logits: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    target_range: tuple[float, float] = (0.02, 0.35),
) -> torch.Tensor:
    """Soft blocked percentage constraint within [lo, hi]."""
    B = obs_logits.shape[0]
    device = obs_logits.device
    probs = F.softmax(obs_logits, dim=-1)

    total_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        w, d = width[i].item(), depth[i].item()
        # Interior cells only (exclude perimeter)
        if d > 2 and w > 2:
            interior = probs[i, 1:d-1, 1:w-1, :]
            blocked_pct = (1.0 - interior[..., 0]).mean()
            low_penalty = F.relu(target_range[0] - blocked_pct)
            high_penalty = F.relu(blocked_pct - target_range[1])
            total_loss = total_loss + low_penalty + high_penalty

    return total_loss / B


def combined_guidance(
    obs_logits: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    lambda_coll: float = 1.0,
    lambda_bound: float = 5.0,
    lambda_conn: float = 2.0,
    lambda_cover: float = 1.0,
    lambda_block: float = 2.0,
) -> torch.Tensor:
    """Compute combined guidance loss."""
    loss = torch.tensor(0.0, device=obs_logits.device)

    if lambda_coll > 0:
        loss = loss + lambda_coll * collision_loss(obs_logits)
    if lambda_bound > 0:
        loss = loss + lambda_bound * boundary_loss(obs_logits, width, depth)
    if lambda_conn > 0:
        loss = loss + lambda_conn * connectivity_loss(obs_logits, width, depth)
    if lambda_cover > 0:
        loss = loss + lambda_cover * cover_density_loss(obs_logits)
    if lambda_block > 0:
        loss = loss + lambda_block * blocked_pct_loss(obs_logits, width, depth)

    return loss


def guidance_scale_schedule(t: float, base_scale: float = 1.0) -> float:
    """Ramp schedule: stronger guidance at noisier (early) steps.

    t in [0, 1] where 1=fully noisy, 0=clean.
    """
    return base_scale * (1.0 - (1.0 - t) ** 2)
