"""Variable-size room grid dataset for ELIT-DiT training."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


# Obstacle type vocabulary size (0=floor..8=ramp)
NUM_OBS_TYPES = 9


class RoomGridDataset(Dataset):
    """Load room JSONL records into multi-channel grid tensors.

    Each sample returns:
        obs_type: (D, W) int tensor, values in [0, 8]
        height: (D, W) float tensor
        elevation: (D, W) float tensor
        room_type: int, index into ROOM_TYPES
        width: int
        depth: int
        caption: str (may be empty if not yet captioned)
    """

    ROOM_TYPES = ["Entry", "Pressure", "Pivot", "Setpiece", "Recovery", "Climax", "Open"]
    ROOM_TYPE_TO_IDX = {rt: i for i, rt in enumerate(ROOM_TYPES)}

    def __init__(self, jsonl_path: str | Path, max_dim: int = 64):
        self.records = []
        self.max_dim = max_dim

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                w, d = rec["width"], rec["depth"]
                if w > max_dim or d > max_dim or w < 8 or d < 8:
                    continue
                self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        w = rec["width"]
        d = rec["depth"]

        obs_type = torch.tensor(rec["grid"]["obstacle_type"], dtype=torch.long)  # (D, W)
        height = torch.tensor(rec["grid"]["height"], dtype=torch.float32)
        elevation = torch.tensor(rec["grid"]["elevation"], dtype=torch.float32)
        room_type_idx = self.ROOM_TYPE_TO_IDX.get(rec["room_type"], 0)
        caption = rec.get("caption", "")

        return {
            "obs_type": obs_type,
            "height": height,
            "elevation": elevation,
            "room_type": room_type_idx,
            "width": w,
            "depth": d,
            "caption": caption,
        }


def collate_room_grids(batch):
    """Pad variable-size grids to the max dimensions in the batch.

    Returns a dict with:
        obs_type: (B, D_max, W_max) long
        height: (B, D_max, W_max) float
        elevation: (B, D_max, W_max) float
        room_type: (B,) long
        width: (B,) long
        depth: (B,) long
        mask: (B, D_max, W_max) bool — True for valid cells
        captions: list[str]
    """
    max_w = max(s["width"] for s in batch)
    max_d = max(s["depth"] for s in batch)
    B = len(batch)

    obs_type = torch.zeros(B, max_d, max_w, dtype=torch.long)
    height = torch.zeros(B, max_d, max_w, dtype=torch.float32)
    elevation = torch.zeros(B, max_d, max_w, dtype=torch.float32)
    mask = torch.zeros(B, max_d, max_w, dtype=torch.bool)
    room_type = torch.zeros(B, dtype=torch.long)
    width = torch.zeros(B, dtype=torch.long)
    depth = torch.zeros(B, dtype=torch.long)
    captions = []

    for i, s in enumerate(batch):
        d, w = s["depth"], s["width"]
        obs_type[i, :d, :w] = s["obs_type"]
        height[i, :d, :w] = s["height"]
        elevation[i, :d, :w] = s["elevation"]
        mask[i, :d, :w] = True
        room_type[i] = s["room_type"]
        width[i] = w
        depth[i] = d
        captions.append(s["caption"])

    return {
        "obs_type": obs_type,
        "height": height,
        "elevation": elevation,
        "room_type": room_type,
        "width": width,
        "depth": depth,
        "mask": mask,
        "captions": captions,
    }
