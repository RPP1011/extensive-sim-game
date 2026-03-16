#!/usr/bin/env python3
"""Inference script for Rust subprocess integration.

Called by Rust via std::process::Command. Reads request from stdin JSON,
writes room grid JSON to stdout.

Usage (standalone test):
    echo '{"prompt":"tight corridor","room_type":"Pressure","seed":42}' | \
        uv run --with torch python training/roomgen/infer.py --weights generated/elit_dit_weights.pt

Protocol:
    Input (stdin, one JSON object):
        {
            "prompt": "text description",
            "room_type": "Entry",
            "seed": 42,
            "width": null,   // optional override
            "depth": null    // optional override
        }

    Output (stdout, one JSON object):
        {
            "width": 20,
            "depth": 20,
            "obstacle_type": [[0,1,...], ...],
            "height": [[0.0, 2.0, ...], ...],
            "elevation": [[0.0, 0.0, ...], ...],
            "success": true
        }
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roomgen.elit_dit import ELITDiT
from roomgen.sample import guided_sample, postprocess, validate_room
from roomgen.text_encoder import build_text_encoder
from roomgen.dataset import RoomGridDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--text-encoder", default="minilm")
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
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

    text_enc = build_text_encoder(args.text_encoder, device=str(device))
    text_enc = text_enc.to(device)

    # Read request from stdin
    request = json.loads(sys.stdin.read())
    prompt = request.get("prompt", "")
    room_type = request.get("room_type", "Entry")
    seed = request.get("seed", 0)
    width_override = request.get("width")
    depth_override = request.get("depth")

    torch.manual_seed(seed)

    room_type_idx = RoomGridDataset.ROOM_TYPE_TO_IDX.get(room_type, 0)

    with torch.no_grad():
        text_emb = text_enc([prompt]).to(device)
        rt = torch.tensor([room_type_idx], device=device)

        if width_override and depth_override:
            w = int(min(64, max(8, width_override)))
            d = int(min(64, max(8, depth_override)))
        else:
            pred_dims = model.predict_dims(text_emb, rt)
            w = int(pred_dims[0, 0].round().clamp(8, 64).item())
            d = int(pred_dims[0, 1].round().clamp(8, 64).item())

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
            grid["success"] = True
            print(json.dumps(grid))
            return

    # All retries failed
    print(json.dumps({"success": False, "width": w, "depth": d}))


if __name__ == "__main__":
    main()
