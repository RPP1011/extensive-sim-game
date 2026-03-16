#!/usr/bin/env python3
"""GPU inference server for IMPALA actors via shared memory.

Single-batch design: a Rust batcher thread collects requests from rayon threads
into one contiguous batch, signals "request_ready", and this server processes
the whole batch in one GPU forward pass.

Layout (must match gpu_client.rs):
  Header (512 bytes):
    [0x00] magic: u32 = 0x47505549
    [0x04] version: u32 = 1
    [0x08] cls_dim: u32
    [0x0C] max_batch_size: u32
    [0x10] sample_size: u32
    [0x14] response_sample_size: u32 = 16
    [0x40] flag: u32 (0=idle, 1=request_ready, 2=response_ready)
    [0x44] batch_size: u32
    [0x80] reload_path: 64 bytes (null-terminated)
    [0xC0] reload_request: u32
    [0xC4] reload_ack: u32

  [512..] request_data: max_batch × sample_size (contiguous)
  [512 + max_batch × sample_size ..] response_data: max_batch × 16

Usage:
    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/gpu_inference_server.py \
        --weights generated/actor_critic_v4_full_unfrozen.pt \
        --shm-name impala_inf --max-batch-size 1024
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    AbilityActorCriticV4, AbilityActorCriticV5,
    MAX_ABILITIES, NUM_COMBAT_TYPES, POSITION_DIM, AGG_FEATURE_DIM,
)
from tokenizer import AbilityTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ENTITIES = 20
MAX_THREATS = 6
MAX_POSITIONS = 8
ENTITY_DIM = 34
THREAT_DIM = 10

SHM_MAGIC = 0x47505549
SHM_VERSION = 1
HEADER_SIZE = 512
OFF_MAGIC = 0x00
OFF_VERSION = 0x04
OFF_CLS_DIM = 0x08
OFF_MAX_BATCH = 0x0C
OFF_SAMPLE_SIZE = 0x10
OFF_RESP_SAMPLE_SIZE = 0x14
OFF_FLAG = 0x40
OFF_BATCH_SIZE = 0x44
OFF_RELOAD_PATH = 0x80
RELOAD_PATH_LEN = 256
OFF_RELOAD_REQ = 0x180
OFF_RELOAD_ACK = 0x184

OFF_H_DIM = 0x18  # GRU hidden dimension stored in header
OFF_AGG_DIM = 0x1C  # aggregate token feature dimension

RESPONSE_BASE_SIZE = 24  # move_dx(f32) + move_dy(f32) + combat_type(1) + pad(1) + target(2) + 3×f32 log-probs


def compute_sample_size(cls_dim: int, h_dim: int = 0, agg_dim: int = 0) -> int:
    ent_mask_padded = (MAX_ENTITIES + 3) & ~3
    thr_mask_padded = (MAX_THREATS + 3) & ~3
    pos_mask_padded = (MAX_POSITIONS + 3) & ~3
    return (8 + MAX_ENTITIES * ENTITY_DIM * 4 + MAX_ENTITIES * 4 + ent_mask_padded +
            MAX_THREATS * THREAT_DIM * 4 + thr_mask_padded +
            MAX_POSITIONS * POSITION_DIM * 4 + pos_mask_padded +
            12 + MAX_ABILITIES + MAX_ABILITIES * cls_dim * 4 +
            h_dim * 4 +      # hidden_state_in
            agg_dim * 4)      # aggregate_features


def compute_response_size(h_dim: int = 0) -> int:
    return RESPONSE_BASE_SIZE + h_dim * 4  # hidden_state_out


class InferenceServer:
    def __init__(
        self,
        model: AbilityActorCriticV4 | AbilityActorCriticV5,
        cls_dim: int,
        max_batch_size: int,
        temperature: float = 1.0,
        h_dim: int = 0,
        agg_dim: int = 0,
        n_latents: int = 0,
    ):
        self.model = model
        self.cls_dim = cls_dim
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.h_dim = h_dim
        self.agg_dim = agg_dim
        self.n_latents = n_latents  # 0 = use model default (no override)
        self.is_v5 = isinstance(model, AbilityActorCriticV5)
        self.sample_size = compute_sample_size(cls_dim, h_dim, agg_dim)
        self.response_sample_size = compute_response_size(h_dim)
        self.model.eval()
        self._field_offsets = self._compute_field_offsets()

        self.total_inferences = 0
        self.total_batches = 0
        self.t_start = time.time()

    def _compute_field_offsets(self) -> list[tuple[int, int]]:
        """Compute (byte_offset, byte_size) for each field within a sample."""
        ent_mask_padded = (MAX_ENTITIES + 3) & ~3
        thr_mask_padded = (MAX_THREATS + 3) & ~3
        pos_mask_padded = (MAX_POSITIONS + 3) & ~3

        off = 0
        fields = []
        # 0: counts (8 bytes: 4 x u16)
        fields.append((off, 8)); off += 8
        # 1: ent_feat (MAX_ENTITIES * ENTITY_DIM * 4)
        sz = MAX_ENTITIES * ENTITY_DIM * 4
        fields.append((off, sz)); off += sz
        # 2: ent_types (MAX_ENTITIES * 4)
        sz = MAX_ENTITIES * 4
        fields.append((off, sz)); off += sz
        # 3: ent_mask (MAX_ENTITIES bytes, padded to 4)
        fields.append((off, MAX_ENTITIES)); off += ent_mask_padded
        # 4: thr_feat (MAX_THREATS * THREAT_DIM * 4)
        sz = MAX_THREATS * THREAT_DIM * 4
        fields.append((off, sz)); off += sz
        # 5: thr_mask (MAX_THREATS bytes, padded to 4)
        fields.append((off, MAX_THREATS)); off += thr_mask_padded
        # 6: pos_feat (MAX_POSITIONS * POSITION_DIM * 4)
        sz = MAX_POSITIONS * POSITION_DIM * 4
        fields.append((off, sz)); off += sz
        # 7: pos_mask (MAX_POSITIONS bytes, padded to 4)
        fields.append((off, MAX_POSITIONS)); off += pos_mask_padded
        # 8: combat_mask (NUM_COMBAT_TYPES bytes + 2 padding = 12)
        fields.append((off, NUM_COMBAT_TYPES)); off += 12
        # 9: ability_has (MAX_ABILITIES bytes)
        fields.append((off, MAX_ABILITIES)); off += MAX_ABILITIES
        # 10: ability_cls (MAX_ABILITIES * cls_dim * 4)
        sz = MAX_ABILITIES * self.cls_dim * 4
        fields.append((off, sz)); off += sz
        # 11: hidden_state_in (h_dim * 4) — GRU hidden state from previous tick
        if self.h_dim > 0:
            sz = self.h_dim * 4
            fields.append((off, sz)); off += sz
        # 12: aggregate_features (agg_dim * 4) — crowd summary token
        if self.agg_dim > 0:
            sz = self.agg_dim * 4
            fields.append((off, sz)); off += sz
        return fields

    def create_shm(self, shm_path: str) -> mmap.mmap:
        req_region = self.max_batch_size * self.sample_size
        resp_region = self.max_batch_size * self.response_sample_size
        total_size = HEADER_SIZE + req_region + resp_region

        if os.path.exists(shm_path):
            os.unlink(shm_path)

        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, total_size)
        mm = mmap.mmap(fd, total_size)
        os.close(fd)

        # Write header
        struct.pack_into('<IIIIII', mm, 0,
                         SHM_MAGIC, SHM_VERSION, self.cls_dim,
                         self.max_batch_size, self.sample_size,
                         self.response_sample_size)
        # Write h_dim and agg_dim at dedicated offsets
        struct.pack_into('<I', mm, OFF_H_DIM, self.h_dim)
        struct.pack_into('<I', mm, OFF_AGG_DIM, self.agg_dim)
        # Clear flag and batch_size
        struct.pack_into('<II', mm, OFF_FLAG, 0, 0)
        # Clear reload fields
        struct.pack_into('<II', mm, OFF_RELOAD_REQ, 0, 0)

        mm.flush()
        return mm

    def parse_batch(self, mm: mmap.mmap, batch_size: int) -> dict:
        """Parse contiguous batch from SHM into tensors — vectorized, no Python loop."""
        B = batch_size
        ss = self.sample_size
        fo = self._field_offsets

        # Read entire batch region as (B, sample_size) uint8 array
        raw = np.frombuffer(
            mm, dtype=np.uint8, count=B * ss,
            offset=HEADER_SIZE,
        ).reshape(B, ss)

        # Extract each field by slicing columns, then reinterpret dtype
        # Entity features: f32[B, 20, 30]
        foff, fsz = fo[1]
        ent_feat = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, MAX_ENTITIES, ENTITY_DIM)

        # Entity types: i32[B, 20]
        foff, fsz = fo[2]
        ent_types = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.int32).reshape(B, MAX_ENTITIES)

        # Entity mask: u8[B, 20]
        foff, fsz = fo[3]
        ent_mask = raw[:, foff:foff + fsz].copy()

        # Threat features: f32[B, 4, 8]
        foff, fsz = fo[4]
        thr_feat = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, MAX_THREATS, THREAT_DIM)

        # Threat mask: u8[B, 4]
        foff, fsz = fo[5]
        thr_mask = raw[:, foff:foff + fsz].copy()

        # Position features: f32[B, 4, 8]
        foff, fsz = fo[6]
        pos_feat = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, MAX_POSITIONS, POSITION_DIM)

        # Position mask: u8[B, 4]
        foff, fsz = fo[7]
        pos_mask = raw[:, foff:foff + fsz].copy()

        # Combat mask: u8[B, 10]
        foff, fsz = fo[8]
        combat_mask = raw[:, foff:foff + fsz].copy()

        # Ability has: u8[B, 8]
        foff, fsz = fo[9]
        ability_has = raw[:, foff:foff + fsz].copy()

        # Ability CLS: f32[B, 8, cls_dim]
        foff, fsz = fo[10]
        ability_cls = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, MAX_ABILITIES, self.cls_dim)

        result = {
            "ent_feat": torch.from_numpy(ent_feat).to(DEVICE),
            "ent_types": torch.from_numpy(ent_types).long().to(DEVICE),
            "ent_mask": torch.from_numpy(ent_mask).bool().to(DEVICE),
            "thr_feat": torch.from_numpy(thr_feat).to(DEVICE),
            "thr_mask": torch.from_numpy(thr_mask).bool().to(DEVICE),
            "pos_feat": torch.from_numpy(pos_feat).to(DEVICE),
            "pos_mask": torch.from_numpy(pos_mask).bool().to(DEVICE),
            "combat_mask": torch.from_numpy(combat_mask).bool().to(DEVICE),
            "ability_has": ability_has,
            "ability_cls": torch.from_numpy(ability_cls).to(DEVICE),
        }

        # Hidden state input for GRU temporal context
        if self.h_dim > 0 and len(fo) > 11:
            foff, fsz = fo[11]
            h_in = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, self.h_dim)
            result["h_prev"] = torch.from_numpy(h_in).to(DEVICE)

        # Aggregate features for V5
        agg_field_idx = 12 if self.h_dim > 0 else 11
        if self.agg_dim > 0 and len(fo) > agg_field_idx:
            foff, fsz = fo[agg_field_idx]
            agg = np.ascontiguousarray(raw[:, foff:foff + fsz]).view(np.float32).reshape(B, self.agg_dim)
            result["aggregate_features"] = torch.from_numpy(agg).to(DEVICE)
            # Debug: print first batch's aggregate values once
            if not hasattr(self, '_agg_debug_done'):
                self._agg_debug_done = True
                print(f"  [gpu] AGG DEBUG: first sample agg[14:16] = "
                      f"{agg[0, 14]:.4f}, {agg[0, 15]:.4f}", flush=True)

        return result

    @torch.no_grad()
    def infer(self, batch: dict, batch_size: int) -> bytes:
        """Run inference and return binary response."""
        ability_cls: list[torch.Tensor | None] = [None] * MAX_ABILITIES
        for ab in range(MAX_ABILITIES):
            if batch["ability_has"][:, ab].any():
                cls_tensor = batch["ability_cls"][:, ab]
                mask = torch.from_numpy(batch["ability_has"][:, ab]).bool().to(DEVICE)
                cls_tensor = cls_tensor * mask.unsqueeze(-1).float()
                if mask.any():
                    ability_cls[ab] = cls_tensor

        h_prev = batch.get("h_prev", None)
        agg_feat = batch.get("aggregate_features", None)

        if self.is_v5:
            output, h_new = self.model(
                batch["ent_feat"], batch["ent_types"],
                batch["thr_feat"], batch["ent_mask"], batch["thr_mask"],
                ability_cls,
                batch["pos_feat"], batch["pos_mask"],
                aggregate_features=agg_feat,
                h_prev=h_prev,
                n_latents_override=self.n_latents if self.n_latents > 0 else None,
            )
            value = None
        else:
            output, value, h_new = self.model(
                batch["ent_feat"], batch["ent_types"],
                batch["thr_feat"], batch["ent_mask"], batch["thr_mask"],
                ability_cls,
                batch["pos_feat"], batch["pos_mask"],
                h_prev=h_prev,
                aggregate_features=agg_feat,
            )

        # Target position head: model outputs (x/20, y/20) normalized waypoint
        if "target_pos" in output:
            target_pos = output["target_pos"]  # (B, 2) normalized
            # Denormalize: multiply by 20 to get world coordinates
            target_pos_world = target_pos * 20.0  # (B, 2) world space
            # Log prob for position head = 0 (deterministic, no sampling)
            lp_move = torch.zeros(batch_size, device=DEVICE)
        else:
            # Legacy: discrete movement
            move_logits = output["move_logits"] / self.temperature
            move_probs = F.softmax(move_logits, dim=-1)
            move_dirs = torch.multinomial(move_probs, 1).squeeze(-1)
            move_logp_raw = F.log_softmax(output["move_logits"], dim=-1)
            lp_move = move_logp_raw.gather(1, move_dirs.unsqueeze(1)).squeeze(1)
            # Convert to position for SHM
            dir_vecs_t = torch.tensor([
                [0.0, 1.0], [0.707, 0.707], [1.0, 0.0], [0.707, -0.707],
                [0.0, -1.0], [-0.707, -0.707], [-1.0, 0.0], [-0.707, 0.707], [0.0, 0.0],
            ], device=DEVICE)
            target_pos_world = dir_vecs_t[move_dirs]  # Just direction, not real position

        combat_logits = output["combat_logits"].masked_fill(~batch["combat_mask"], -1e9) / self.temperature

        combat_probs = F.softmax(combat_logits, dim=-1)
        combat_types = torch.multinomial(combat_probs, 1).squeeze(-1)
        combat_logp_raw = F.log_softmax(
            output["combat_logits"].masked_fill(~batch["combat_mask"], -1e9), dim=-1)
        lp_combat = combat_logp_raw.gather(1, combat_types.unsqueeze(1)).squeeze(1)

        target_indices = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        lp_pointer = torch.zeros(batch_size, device=DEVICE)

        # Vectorized pointer sampling by combat type group
        attack_mask = combat_types == 0
        if attack_mask.any():
            ptr_logits = output["attack_ptr"][attack_mask] / self.temperature
            ptr_probs = F.softmax(ptr_logits, dim=-1)
            ti = torch.multinomial(ptr_probs, 1).squeeze(-1)
            target_indices[attack_mask] = ti
            lp_raw = F.log_softmax(output["attack_ptr"][attack_mask], dim=-1)
            lp_pointer[attack_mask] = lp_raw.gather(1, ti.unsqueeze(1)).squeeze(1).clamp(min=-10)

        ab_ptrs = output.get("ability_ptrs", [])
        for ab_idx, ab_ptr in enumerate(ab_ptrs):
            if ab_ptr is None:
                continue
            ct = ab_idx + 2
            ab_mask = combat_types == ct
            if not ab_mask.any():
                continue
            ptr_logits = ab_ptr[ab_mask] / self.temperature
            ptr_probs = F.softmax(ptr_logits, dim=-1)
            ti = torch.multinomial(ptr_probs, 1).squeeze(-1)
            target_indices[ab_mask] = ti
            lp_raw = F.log_softmax(ab_ptr[ab_mask], dim=-1)
            lp_pointer[ab_mask] = lp_raw.gather(1, ti.unsqueeze(1)).squeeze(1).clamp(min=-10)

        # Vectorized response packing via numpy
        # Layout: target_x(f32) target_y(f32) combat_type(u8) pad(u8) target_idx(u16) lp_move(f32) lp_combat(f32) lp_pointer(f32)
        move_pos_cpu = np.ascontiguousarray(target_pos_world.cpu().numpy().astype(np.float32))  # (B, 2)
        rss = self.response_sample_size
        resp_np = np.zeros((batch_size, rss), dtype=np.uint8)
        ct_cpu = combat_types.cpu().numpy().astype(np.uint8)
        ti_cpu = target_indices.cpu().numpy().astype(np.uint16)
        lpm_cpu = lp_move.cpu().numpy().astype(np.float32)
        lpc_cpu = lp_combat.cpu().numpy().astype(np.float32)
        lpp_cpu = lp_pointer.cpu().numpy().astype(np.float32)
        resp_np[:, 0:4] = np.ascontiguousarray(move_pos_cpu[:, 0]).view(np.uint8).reshape(-1, 4)
        resp_np[:, 4:8] = np.ascontiguousarray(move_pos_cpu[:, 1]).view(np.uint8).reshape(-1, 4)
        resp_np[:, 8] = ct_cpu                                                 # combat_type
        resp_np[:, 9] = 0                                                      # pad
        resp_np[:, 10] = ti_cpu.view(np.uint8).reshape(-1, 2)[:, 0]           # target_idx lo
        resp_np[:, 11] = ti_cpu.view(np.uint8).reshape(-1, 2)[:, 1]           # target_idx hi
        resp_np[:, 12:16] = lpm_cpu.view(np.uint8).reshape(-1, 4)
        resp_np[:, 16:20] = lpc_cpu.view(np.uint8).reshape(-1, 4)
        resp_np[:, 20:24] = lpp_cpu.view(np.uint8).reshape(-1, 4)
        # Pack GRU hidden state output
        if self.h_dim > 0:
            h_cpu = h_new.cpu().numpy().astype(np.float32)
            h_bytes = h_cpu.view(np.uint8).reshape(batch_size, self.h_dim * 4)
            resp_np[:, 16:16 + self.h_dim * 4] = h_bytes
        resp = bytes(resp_np.tobytes())

        self.total_inferences += batch_size
        self.total_batches += 1
        return bytes(resp)

    def reload_weights(self, path: str):
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd)
            self.model.eval()
            print(f"  Reloaded weights from {path}", flush=True)
        except Exception as e:
            print(f"  Failed to reload: {e}", flush=True)

    def poll_loop(self, mm: mmap.mmap):
        """Poll flag, parse batch, infer, write response."""
        print("Polling for requests...", flush=True)
        idle_spins = 0
        resp_base = HEADER_SIZE + self.max_batch_size * self.sample_size

        while True:
            # Check reload
            if struct.unpack_from('<I', mm, OFF_RELOAD_REQ)[0] != 0:
                path_raw = mm[OFF_RELOAD_PATH:OFF_RELOAD_PATH + RELOAD_PATH_LEN]
                path = path_raw.split(b'\x00')[0].decode('utf-8')
                self.reload_weights(path)
                struct.pack_into('<I', mm, OFF_RELOAD_ACK, 1)
                struct.pack_into('<I', mm, OFF_RELOAD_REQ, 0)

            # Check flag
            flag = struct.unpack_from('<I', mm, OFF_FLAG)[0]
            if flag != 1:
                idle_spins += 1
                if idle_spins > 100000:
                    time.sleep(0.0001)
                continue

            idle_spins = 0

            # Read batch_size
            batch_size = struct.unpack_from('<I', mm, OFF_BATCH_SIZE)[0]
            if batch_size == 0 or batch_size > self.max_batch_size:
                struct.pack_into('<I', mm, OFF_FLAG, 0)
                continue

            # Parse and infer
            t0 = time.perf_counter()
            batch = self.parse_batch(mm, batch_size)
            t1 = time.perf_counter()
            resp_data = self.infer(batch, batch_size)
            t2 = time.perf_counter()

            # Write response
            mm[resp_base:resp_base + len(resp_data)] = resp_data

            # Signal response_ready
            struct.pack_into('<I', mm, OFF_FLAG, 2)
            t3 = time.perf_counter()

            parse_ms = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            write_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            if self.total_batches <= 10 or self.total_batches % 500 == 0:
                elapsed = time.time() - self.t_start
                rate = self.total_inferences / max(elapsed, 1)
                print(f"  batch {self.total_batches}: bs={batch_size} "
                      f"parse={parse_ms:.2f}ms infer={infer_ms:.2f}ms "
                      f"write={write_ms:.2f}ms total={total_ms:.2f}ms "
                      f"({rate:.0f} inf/sec)", flush=True)


def main():
    p = argparse.ArgumentParser(description="GPU inference server (shared memory)")
    p.add_argument("--weights", required=True, help="V4 checkpoint (.pt)")
    p.add_argument("--shm-name", default="impala_inf")
    p.add_argument("--max-batch-size", type=int, default=1024,
                   help="Maximum batch size for GPU inference")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--external-cls-dim", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--h-dim", type=int, default=0,
                   help="GRU hidden dimension for temporal context (0 = disabled)")
    p.add_argument("--model-version", type=int, default=4, choices=[4, 5],
                   help="Model version: 4 (V4, d=32 default) or 5 (V5, d=128 default)")
    p.add_argument("--n-latents", type=int, default=0,
                   help="V5 latent count override for inference (0 = use model default)")
    args = p.parse_args()

    tok = AbilityTokenizer()
    h_dim = args.h_dim
    # Enable aggregate features for V5, or for V4 with target_proj
    agg_dim = AGG_FEATURE_DIM if args.model_version == 5 else 0
    if agg_dim == 0:
        # Check if V4 checkpoint has target_proj — if so, enable aggregate
        try:
            ckpt_check = torch.load(args.weights, map_location="cpu", weights_only=False)
            sd_check = ckpt_check.get("model_state_dict", ckpt_check)
            if any("target_proj" in k or "target_move_head" in k for k in sd_check):
                agg_dim = AGG_FEATURE_DIM
                print(f"V4 with target_proj detected — enabling aggregate features (agg_dim={agg_dim})")
        except Exception:
            pass

    if args.model_version == 5:
        model = AbilityActorCriticV5(
            vocab_size=tok.vocab_size,
            entity_encoder_layers=args.entity_encoder_layers,
            external_cls_dim=args.external_cls_dim,
            h_dim=h_dim if h_dim > 0 else 256,
            d_model=args.d_model, d_ff=args.d_ff if args.d_ff > 0 else args.d_model * 2,
            n_layers=args.n_layers, n_heads=args.n_heads,
        ).to(DEVICE)
    else:
        model = AbilityActorCriticV4(
            vocab_size=tok.vocab_size,
            entity_encoder_layers=args.entity_encoder_layers,
            external_cls_dim=args.external_cls_dim,
            h_dim=h_dim if h_dim > 0 else 64,
            d_model=args.d_model, d_ff=args.d_ff,
            n_layers=args.n_layers, n_heads=args.n_heads,
        ).to(DEVICE)

    ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    version_str = "V5" if args.model_version == 5 else "V4"
    total_params = sum(pp.numel() for pp in model.parameters())
    print(f"Model {version_str}: {total_params:,} params on {DEVICE} (h_dim={h_dim}, agg_dim={agg_dim})")

    cls_dim = args.external_cls_dim if args.external_cls_dim > 0 else args.d_model
    server = InferenceServer(model, cls_dim, args.max_batch_size,
                             temperature=args.temperature, h_dim=h_dim,
                             agg_dim=agg_dim, n_latents=args.n_latents)

    shm_path = f"/dev/shm/{args.shm_name}"
    mm = server.create_shm(shm_path)
    req_region = args.max_batch_size * server.sample_size
    resp_region = args.max_batch_size * server.response_sample_size
    total = HEADER_SIZE + req_region + resp_region
    print(f"Shared memory: {shm_path} "
          f"(max_batch={args.max_batch_size}, sample={server.sample_size}B, "
          f"total={total}B)")
    print(f"Ready", flush=True)

    try:
        server.poll_loop(mm)
    except KeyboardInterrupt:
        print("\nShutting down", flush=True)
    finally:
        mm.close()
        try:
            os.unlink(shm_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
