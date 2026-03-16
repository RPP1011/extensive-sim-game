#!/usr/bin/env python3
"""Generate rooms using trained ELIT-DiT with guidance and optional critic.

Usage:
    # Generate a single room
    uv run --with torch python training/roomgen/sample.py \
        --weights generated/elit_dit_weights.pt \
        --prompt "Tight corridor with flanking barricades" \
        --room-type Pressure

    # Batch generate
    uv run --with torch python training/roomgen/sample.py \
        --weights generated/elit_dit_weights.pt \
        --prompts-file prompts.txt \
        --output generated/ml_rooms.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roomgen.elit_dit import ELITDiT
from roomgen.flow_matching import euler_sample
from roomgen.guidance import combined_guidance, guidance_scale_schedule
from roomgen.text_encoder import build_text_encoder
from roomgen.dataset import RoomGridDataset


def guided_sample(
    model: ELITDiT,
    text_emb: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    n_steps: int = 40,
    cfg_scale: float = 3.0,
    guidance_weight: float = 1.0,
    critic=None,
    critic_weight: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Euler sampling with PhyScene guidance and optional PCGRL critic."""
    B = text_emb.shape[0]
    D_max = depth.max().item()
    W_max = width.max().item()

    # Initialize from noise
    obs_type = torch.randint(0, 9, (B, D_max, W_max), device=device)
    height_val = torch.randn(B, D_max, W_max, device=device)
    elevation_val = torch.randn(B, D_max, W_max, device=device)
    obs_logits = torch.zeros(B, D_max, W_max, 9, device=device)

    mask = torch.zeros(B, D_max, W_max, dtype=torch.bool, device=device)
    for i in range(B):
        mask[i, :depth[i], :width[i]] = True

    null_text = torch.zeros_like(text_emb)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = 1.0 - step * dt
        t = torch.full((B,), t_val, device=device)

        # Conditioned prediction
        pred_cond = model(
            obs_type, height_val, elevation_val, t, text_emb, width, depth, mask=mask,
        )

        if cfg_scale > 1.0:
            j_unc = max(2, model.j_per_group // 3)
            pred_uncond = model(
                obs_type, height_val, elevation_val, t, null_text, width, depth,
                mask=mask, j_budget=j_unc,
            )
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

        # Apply guidance on spatial output
        if guidance_weight > 0:
            scale = guidance_scale_schedule(t_val, guidance_weight)
            # Compute current predicted clean output
            current_obs = obs_logits - dt * obs_v
            current_obs.requires_grad_(True)

            g_loss = combined_guidance(current_obs, width, depth)
            grad = torch.autograd.grad(g_loss, current_obs, retain_graph=False)[0]
            obs_v = obs_v + scale * grad

            current_obs = current_obs.detach()

        # Euler step
        height_val = height_val - dt * h_v
        elevation_val = elevation_val - dt * e_v
        obs_logits = obs_logits - dt * obs_v
        obs_type = obs_logits.argmax(dim=-1)

    return {
        "obs_type": obs_type,
        "obs_logits": obs_logits,
        "height": height_val,
        "elevation": elevation_val,
        "mask": mask,
    }


def postprocess(result: dict, width: int, depth: int) -> dict:
    """Post-process model output to discrete room grid.

    1. Argmax on obstacle-type channel
    2. Quantize height/elevation
    3. Enforce perimeter walls
    4. Extract grid arrays
    """
    obs = result["obs_type"][0, :depth, :width].cpu()
    height = result["height"][0, :depth, :width].cpu().clamp(0, 3.0)
    elevation = result["elevation"][0, :depth, :width].cpu().clamp(0, 2.0)

    # Enforce perimeter walls
    obs[0, :] = 1  # top
    obs[depth - 1, :] = 1  # bottom
    obs[:, 0] = 1  # left
    obs[:, width - 1] = 1  # right

    # Quantize height to 1 decimal
    height = (height * 10).round() / 10
    elevation = (elevation * 10).round() / 10

    # Set floor height/elevation to 0
    floor_mask = obs == 0
    height[floor_mask] = 0.0

    return {
        "width": width,
        "depth": depth,
        "obstacle_type": obs.tolist(),
        "height": height.tolist(),
        "elevation": elevation.tolist(),
    }


def validate_room(grid: dict) -> bool:
    """Check connectivity and blocked percentage."""
    obs = grid["obstacle_type"]
    w, d = grid["width"], grid["depth"]

    # Blocked percentage
    total = 0
    blocked = 0
    for r in range(1, d - 1):
        for c in range(1, w - 1):
            total += 1
            if obs[r][c] != 0:
                blocked += 1

    if total == 0:
        return False
    pct = blocked / total
    if pct < 0.02 or pct > 0.35:
        return False

    # BFS connectivity
    start = None
    goal = None
    for r in range(1, d - 1):
        if obs[r][w // 6] == 0:
            start = (r, w // 6)
            break
    for r in range(1, d - 1):
        if obs[r][w - w // 6] == 0:
            goal = (r, w - w // 6)
            break

    if start is None or goal is None:
        return False

    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < d and 0 <= nc < w and (nr, nc) not in visited and obs[nr][nc] == 0:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False


def main():
    parser = argparse.ArgumentParser(description="Generate rooms with ELIT-DiT")
    parser.add_argument("--weights", required=True, help="Path to trained weights")
    parser.add_argument("--prompt", default=None, help="Text prompt for single room")
    parser.add_argument("--prompts-file", default=None, help="File with one prompt per line")
    parser.add_argument("--room-type", default="Entry")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--text-encoder", default="minilm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading weights from {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})

    model = ELITDiT(
        d_model=model_args.get("d_model", 256),
        n_heads=model_args.get("n_heads", 8),
        d_ff=model_args.get("d_model", 256) * 4,
        d_text=384,
        n_head_blocks=model_args.get("n_head_blocks", 2),
        n_core_blocks=model_args.get("n_core_blocks", 8),
        n_tail_blocks=model_args.get("n_tail_blocks", 2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Text encoder
    text_enc = build_text_encoder(args.text_encoder, device=str(device))
    text_enc = text_enc.to(device)

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = ["A standard tactical room with scattered cover"]

    room_type_idx = RoomGridDataset.ROOM_TYPE_TO_IDX.get(args.room_type, 0)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] '{prompt}'")

        with torch.no_grad():
            text_emb = text_enc([prompt]).to(device)
            rt = torch.tensor([room_type_idx], device=device)

            # Predict dimensions
            pred_dims = model.predict_dims(text_emb, rt)
            w = int(pred_dims[0, 0].round().clamp(8, 64).item())
            d = int(pred_dims[0, 1].round().clamp(8, 64).item())
            print(f"  Predicted dimensions: {w}x{d}")

        width_t = torch.tensor([w], device=device)
        depth_t = torch.tensor([d], device=device)

        for attempt in range(args.max_retries):
            result = guided_sample(
                model, text_emb, width_t, depth_t,
                n_steps=args.n_steps,
                cfg_scale=args.cfg_scale,
                guidance_weight=args.guidance_weight,
                device=device,
            )

            grid = postprocess(result, w, d)
            if validate_room(grid):
                print(f"  Valid room generated (attempt {attempt + 1})")
                grid["prompt"] = prompt
                grid["room_type"] = args.room_type
                results.append(grid)
                break
            else:
                print(f"  Validation failed (attempt {attempt + 1}), retrying...")
        else:
            print(f"  Failed after {args.max_retries} attempts, skipping")

    # Output
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nWrote {len(results)} rooms to {args.output}")
    else:
        for r in results:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
