#!/usr/bin/env python3
"""Phase 1: Masked token pre-training for the ability transformer.

Trains the transformer to predict randomly masked tokens in ability DSL
sequences.  Uses grokking-informed settings:
  - AdamW with weight_decay=1.0, betas=(0.9, 0.98)
  - No dropout (regularization via weight decay only)
  - Extended training with accuracy-based stopping, not loss-based
  - Minibatch stochasticity as implicit regularization

Usage:
    uv run --with numpy --with torch training/pretrain.py \
        generated/ability_dataset/ \
        -o generated/ability_transformer_pretrained.pt \
        --max-steps 500000

    # With diagnostics:
    uv run --with numpy --with torch --with scikit-learn --with matplotlib \
        training/pretrain.py generated/ability_dataset/ \
        -o generated/ability_transformer_pretrained.pt \
        --diagnostics-dir diagnostics/pretrain/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add training/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerMLM, HintClassificationHead, ReconstructionDecoder, BehavioralHead
from tokenizer import AbilityTokenizer, MASK, PAD, KEYWORDS, PUNCTUATION, NUM_TOKENS, DUR_TOKENS, SPECIAL_TOKENS
from grokfast import GrokfastEMA

# ---------------------------------------------------------------------------
# Behavioral profiles dataset
# ---------------------------------------------------------------------------

class BehavioralProfiles:
    """Loads ability_profiles.npz as flat GPU tensors for fast batched sampling.

    Pre-builds a per-ability index so we can map MLM dataset indices → random
    behavioral (condition, outcome) pairs entirely on GPU.
    """

    def __init__(self, npz_path: str, device: torch.device):
        data = np.load(npz_path, allow_pickle=True)
        ability_ids = data["ability_id"]  # (N,)

        # All data on GPU
        self.conditions = torch.tensor(data["condition"], dtype=torch.float32, device=device)  # (N, 4)
        self.outcomes = torch.tensor(data["outcome"], dtype=torch.float32, device=device)      # (N, 119)

        # Decode ability names
        names_bytes = data["ability_names"].tobytes()
        self.ability_names = names_bytes.decode("utf-8").split("\n")
        self.name_to_id = {n: i for i, n in enumerate(self.ability_names)}

        # Per-ability sample ranges: for each ability_id, store (start, count) into sorted arrays
        # Sort by ability_id for contiguous ranges
        order = np.argsort(ability_ids)
        self.conditions = self.conditions[order]
        self.outcomes = self.outcomes[order]
        sorted_ids = ability_ids[order]

        n_abilities = len(self.ability_names)
        self.offsets = torch.zeros(n_abilities, dtype=torch.long, device=device)
        self.counts = torch.zeros(n_abilities, dtype=torch.long, device=device)
        for aid in range(n_abilities):
            mask = sorted_ids == aid
            indices = np.where(mask)[0]
            if len(indices) > 0:
                self.offsets[aid] = int(indices[0])
                self.counts[aid] = len(indices)

        # Z-score normalize outcomes per dimension
        self.outcome_mean = self.outcomes.mean(dim=0)  # (119,)
        self.outcome_std = self.outcomes.std(dim=0).clamp(min=1e-6)  # (119,)
        self.outcomes = (self.outcomes - self.outcome_mean) / self.outcome_std

        print(f"BehavioralProfiles: {n_abilities} abilities, "
              f"{len(self.conditions)} samples on {device} (z-normed)")

    def sample_batch(self, ability_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random (condition, outcome) for each ability_id in batch.

        Returns (conditions, outcomes, valid_mask) all on GPU.
        ability_ids: (B,) long tensor of behavioral profile ability IDs (-1 = no match).
        """
        B = ability_ids.shape[0]
        valid = (ability_ids >= 0) & (ability_ids < len(self.counts))
        safe_ids = ability_ids.clamp(0)  # clamp for indexing, masked out later

        offsets = self.offsets[safe_ids]   # (B,)
        counts = self.counts[safe_ids]     # (B,)

        # Random offset within each ability's range
        rand_off = (torch.rand(B, device=ability_ids.device) * counts.float()).long()
        rand_off = rand_off.clamp(max=counts.clamp(min=1) - 1)
        sample_idx = offsets + rand_off

        conditions = self.conditions[sample_idx]  # (B, 4)
        outcomes = self.outcomes[sample_idx]       # (B, 119)

        # Zero out invalid entries
        valid = valid & (counts > 0)
        conditions = conditions * valid.unsqueeze(1).float()
        outcomes = outcomes * valid.unsqueeze(1).float()

        return conditions, outcomes, valid


# Token type classification for per-type accuracy breakdown
_STRUCTURAL_TOKENS: set[str] = set(PUNCTUATION) | {"ability", "passive", "deliver", "in", "on_hit",
    "on_arrival", "on_complete", "on_hit_buff", "target", "range", "cooldown", "cast", "hint", "cost",
    "charges", "recharge", "recast", "recast_window"}
_NUMERIC_TOKENS: set[str] = set(NUM_TOKENS) | set(DUR_TOKENS)
_EFFECT_TOKENS: set[str] = {"damage", "heal", "shield", "stun", "slow", "knockback", "dash", "buff",
    "debuff", "root", "silence", "fear", "taunt", "pull", "swap", "reflect", "lifesteal",
    "damage_modify", "self_damage", "execute", "blind", "resurrect", "overheal_shield",
    "absorb_to_heal", "shield_steal", "immunity", "detonate", "polymorph", "banish",
    "confuse", "charm", "stealth", "blink", "summon", "suppress", "grounded"}
# Hint category labels for auxiliary [CLS] loss
HINT_CATEGORIES = ["damage", "heal", "buff", "defense", "crowd_control", "utility"]
HINT_TO_IDX = {h: i for i, h in enumerate(HINT_CATEGORIES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def augment_ability_text(text: str) -> str:
    """Data augmentation: randomly reorder property lines in ability DSL.

    Property lines (target, range, cooldown, cast, hint, cost, charges,
    recharge, recast, recast_window) are order-independent in the DSL.
    Shuffling them creates semantically equivalent training examples that
    help the model learn position-invariant property understanding.
    """
    lines = text.split("\n")
    header_end = -1
    prop_lines = []
    other_lines = []

    # Find the opening brace, then collect property lines vs effect lines
    in_props = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "{" in stripped and not in_props:
            in_props = True
            other_lines.append(line)
            continue

        if in_props and stripped and not stripped.startswith(("deliver", "damage", "heal",
                "shield", "stun", "slow", "knockback", "dash", "buff", "debuff",
                "root", "silence", "fear", "taunt", "pull", "swap", "reflect",
                "lifesteal", "stealth", "blink", "summon", "execute", "blind",
                "resurrect", "when", "}")):
            # Likely a property line (target:, cooldown:, hint:, etc.)
            if ":" in stripped or stripped.startswith(("charges", "recast", "unstoppable", "toggle")):
                prop_lines.append(line)
                if header_end < 0:
                    header_end = i
                continue

        other_lines.append(line)

    if len(prop_lines) > 1:
        random.shuffle(prop_lines)
        # Re-insert property lines after the header
        result = []
        inserted = False
        for line in other_lines:
            result.append(line)
            if "{" in line and not inserted:
                result.extend(prop_lines)
                inserted = True
        return "\n".join(result)

    return text


def _extract_ability_name(text: str) -> str | None:
    """Extract ability/passive name from DSL text."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ability ") or line.startswith("passive "):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].rstrip("{").strip()
    return None


def _extract_hint(text: str) -> int:
    """Extract hint category index from ability text. Returns -1 if not found."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("hint:"):
            hint = line.split(":", 1)[1].strip().split()[0].split(",")[0]
            return HINT_TO_IDX.get(hint, -1)
    return -1


class NpzMLMDataset:
    """Fast dataset from pre-tokenized npz file. All tensors on GPU."""

    def __init__(
        self,
        npz_path: Path,
        tokenizer: AbilityTokenizer,
        mask_prob: float = 0.15,
        augment: bool = True,
    ):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.augment = augment

        data = np.load(npz_path, allow_pickle=True)
        self.token_ids = torch.tensor(data["token_ids"], dtype=torch.long, device=DEVICE)
        self.lengths = torch.tensor(data["lengths"], dtype=torch.long, device=DEVICE)
        self.hints = torch.tensor(data["hints"], dtype=torch.long, device=DEVICE)
        self.texts = list(data["texts"]) if "texts" in data else []
        self.n = self.token_ids.shape[0]
        self.max_len = self.token_ids.shape[1]

        # Behavioral profile IDs: maps each dataset entry → ability_id in profiles (-1 = no match)
        self.behavioral_ids: torch.Tensor | None = None

        n_with_hint = (self.hints >= 0).sum().item()
        print(f"NpzMLMDataset: {self.n} abilities, max_len={self.max_len}, hints={n_with_hint}/{self.n}")

    def build_behavioral_ids(self, profiles: BehavioralProfiles):
        """Map each ability text to its behavioral profile ID."""
        ids = []
        for text in self.texts:
            name = _extract_ability_name(text)
            aid = profiles.name_to_id.get(name, -1) if name else -1
            ids.append(aid)
        # If no texts (val set with augment=False), use stored ability names from token patterns
        if not ids:
            ids = [-1] * self.n
        self.behavioral_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        matched = (self.behavioral_ids >= 0).sum().item()
        print(f"  Behavioral mapping: {matched}/{len(ids)} abilities matched")

    def __len__(self) -> int:
        return self.n

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(self.n, (batch_size,), device=DEVICE)

        if self.augment and self.texts:
            # Re-tokenize augmented text for ~50% of samples
            seqs = []
            idx_list = indices.tolist()
            for i in idx_list:
                if random.random() < 0.5:
                    augmented = augment_ability_text(self.texts[i])
                    seqs.append(self.tokenizer.encode(augmented, add_cls=True))
                else:
                    row = self.token_ids[i]
                    length = self.lengths[i].item()
                    seqs.append(row[:length].tolist())
            # Variable length — pad to max in batch
            max_len = max(len(s) for s in seqs)
            pad_id = self.tokenizer.pad_id
            padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=DEVICE)
            for j, s in enumerate(seqs):
                padded[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=DEVICE)
            recon_targets = padded.clone()
        else:
            recon_targets = self.token_ids[indices]
            padded = recon_targets.clone()
            max_len = self.max_len

        # Build attention mask
        attention_mask = (padded != self.tokenizer.pad_id).float()

        # Apply masking on GPU
        mask_id = self.tokenizer.mask_id
        vocab_size = self.tokenizer.vocab_size

        # Random mask: skip position 0 ([CLS]) and padding
        rand = torch.rand_like(padded, dtype=torch.float)
        maskable = (torch.arange(max_len, device=DEVICE).unsqueeze(0) > 0) & (padded != self.tokenizer.pad_id)
        to_mask = (rand < self.mask_prob) & maskable

        labels = torch.full_like(padded, -100)
        labels[to_mask] = padded[to_mask]

        # 80% [MASK], 10% random, 10% keep
        r = torch.rand_like(padded, dtype=torch.float)
        input_ids = padded.clone()
        input_ids[to_mask & (r < 0.8)] = mask_id
        random_tokens = torch.randint(vocab_size, padded.shape, device=DEVICE)
        input_ids[to_mask & (r >= 0.8) & (r < 0.9)] = random_tokens[to_mask & (r >= 0.8) & (r < 0.9)]

        hint_labels = self.hints[indices]

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "recon_targets": recon_targets,
            "hint_labels": hint_labels,
        }
        if self.behavioral_ids is not None:
            result["behavioral_ids"] = self.behavioral_ids[indices]
        return result

    def split(self, val_frac: float = 0.15) -> tuple["NpzMLMDataset", "NpzMLMDataset"]:
        n_val = max(1, int(self.n * val_frac))
        perm = torch.randperm(self.n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        val_ds = NpzMLMDataset.__new__(NpzMLMDataset)
        val_ds.tokenizer = self.tokenizer
        val_ds.mask_prob = self.mask_prob
        val_ds.augment = False
        val_ds.token_ids = self.token_ids[val_idx]
        val_ds.lengths = self.lengths[val_idx]
        val_ds.hints = self.hints[val_idx]
        val_ds.texts = []
        val_ds.behavioral_ids = self.behavioral_ids[val_idx] if self.behavioral_ids is not None else None
        val_ds.n = len(val_idx)
        val_ds.max_len = self.max_len

        train_ds = NpzMLMDataset.__new__(NpzMLMDataset)
        train_ds.tokenizer = self.tokenizer
        train_ds.mask_prob = self.mask_prob
        train_ds.augment = self.augment
        train_ds.token_ids = self.token_ids[train_idx]
        train_ds.lengths = self.lengths[train_idx]
        train_ds.hints = self.hints[train_idx]
        train_ds.texts = [self.texts[i] for i in train_idx.tolist()] if self.augment else []
        train_ds.behavioral_ids = self.behavioral_ids[train_idx] if self.behavioral_ids is not None else None
        train_ds.n = len(train_idx)
        train_ds.max_len = self.max_len

        print(f"Split: {train_ds.n} train, {val_ds.n} val")
        return train_ds, val_ds


class AbilityMLMDataset:
    """Loads .ability files, tokenizes, and applies random masking."""

    def __init__(
        self,
        ability_dir: Path,
        tokenizer: AbilityTokenizer,
        mask_prob: float = 0.15,
        holdout_hashes: set[str] | None = None,
        augment: bool = True,
        span_masking: bool = False,
        mean_span_len: float = 3.0,
        no_mask_numeric: bool = False,
    ):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.augment = augment
        self.span_masking = span_masking
        self.mean_span_len = mean_span_len
        self.no_mask_numeric = no_mask_numeric

        # Build set of numeric token IDs to skip during masking
        self._numeric_ids: set[int] = set()
        if no_mask_numeric:
            for tok_str in NUM_TOKENS + DUR_TOKENS:
                if tok_str in tokenizer.tok2id:
                    self._numeric_ids.add(tokenizer.tok2id[tok_str])

        # Load all .ability files
        files = sorted(ability_dir.glob("*.ability"))
        self.texts: list[str] = []
        for f in files:
            text = f.read_text().strip()
            if not text:
                continue
            # Skip holdout abilities if filter provided
            if holdout_hashes and _hash_structure(text) in holdout_hashes:
                continue
            self.texts.append(text)

        print(f"Loaded {len(self.texts)} abilities from {ability_dir}")

        # Pre-tokenize all abilities
        self.encoded: list[list[int]] = []
        self.hints: list[int] = []  # hint category index per ability
        for text in self.texts:
            ids = tokenizer.encode(text, add_cls=True)
            if len(ids) > 3:  # skip trivially short
                self.encoded.append(ids)
                self.hints.append(_extract_hint(text))

        n_with_hint = sum(1 for h in self.hints if h >= 0)
        print(f"Tokenized {len(self.encoded)} abilities (skipped {len(self.texts) - len(self.encoded)} too short)")
        print(f"Hint labels: {n_with_hint}/{len(self.encoded)} ({100*n_with_hint/max(len(self.encoded),1):.0f}%)")

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        tokenizer: AbilityTokenizer,
        mask_prob: float = 0.15,
        augment: bool = True,
    ) -> "AbilityMLMDataset":
        """Create dataset from a list of DSL text strings."""
        ds = cls.__new__(cls)
        ds.tokenizer = tokenizer
        ds.mask_prob = mask_prob
        ds.augment = augment
        ds.span_masking = False
        ds.mean_span_len = 3.0
        ds.no_mask_numeric = False
        ds._numeric_ids = set()
        ds.behavioral_ids = None

        ds.texts = texts
        ds.encoded = []
        ds.hints = []
        for text in texts:
            ids = tokenizer.encode(text, add_cls=True)
            if len(ids) > 3:
                ds.encoded.append(ids)
                ds.hints.append(_extract_hint(text))

        n_with_hint = sum(1 for h in ds.hints if h >= 0)
        print(f"Tokenized {len(ds.encoded)} abilities ({n_with_hint} with hints)")
        return ds

    def build_behavioral_ids(self, profiles: BehavioralProfiles):
        """Map each ability text to its behavioral profile ID."""
        ids = []
        for text in self.texts:
            name = _extract_ability_name(text)
            aid = profiles.name_to_id.get(name, -1) if name else -1
            ids.append(aid)
        self.behavioral_ids = ids
        matched = sum(1 for x in ids if x >= 0)
        print(f"  Behavioral mapping: {matched}/{len(ids)} abilities matched")

    def __len__(self) -> int:
        return len(self.encoded)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch with random masking applied."""
        indices = random.choices(range(len(self.encoded)), k=batch_size)
        if self.augment:
            # Re-tokenize with augmented text for ~50% of samples
            seqs = []
            for i in indices:
                if random.random() < 0.5:
                    augmented = augment_ability_text(self.texts[i])
                    seqs.append(self.tokenizer.encode(augmented, add_cls=True))
                else:
                    seqs.append(self.encoded[i])
            batch = self._make_batch(seqs)
        else:
            batch = self._make_batch([self.encoded[i] for i in indices])
        # Add hint labels for auxiliary [CLS] loss
        batch["hint_labels"] = torch.tensor(
            [self.hints[i] for i in indices], dtype=torch.long, device=DEVICE
        )
        if self.behavioral_ids is not None:
            batch["behavioral_ids"] = torch.tensor(
                [self.behavioral_ids[i] for i in indices], dtype=torch.long, device=DEVICE
            )
        return batch

    def _make_batch(self, sequences: list[list[int]]) -> dict[str, torch.Tensor]:
        """Create masked batch from pre-tokenized sequences."""
        max_len = min(max(len(s) for s in sequences), self.tokenizer.max_length)

        input_ids = []
        labels = []
        attention_masks = []

        pad_id = self.tokenizer.pad_id
        mask_id = self.tokenizer.mask_id
        vocab_size = self.tokenizer.vocab_size

        for seq in sequences:
            seq = seq[:max_len]
            orig = list(seq)
            masked = list(seq)
            label = [-100] * len(seq)  # -100 = ignore in CE loss

            if self.span_masking:
                # Span masking: geometric span lengths, targeting mask_prob total
                self._apply_span_mask(masked, label, orig, mask_id, vocab_size)
            else:
                for i in range(1, len(seq)):  # skip [CLS] at position 0
                    if self.no_mask_numeric and orig[i] in self._numeric_ids:
                        continue  # never mask numeric tokens
                    if random.random() < self.mask_prob:
                        label[i] = orig[i]
                        r = random.random()
                        if r < 0.8:
                            masked[i] = mask_id
                        elif r < 0.9:
                            masked[i] = random.randint(0, vocab_size - 1)
                        # else: keep original (10%)

            # Pad
            pad_len = max_len - len(seq)
            masked += [pad_id] * pad_len
            label += [-100] * pad_len
            attn = [1] * len(seq) + [0] * pad_len

            input_ids.append(masked)
            labels.append(label)
            attention_masks.append(attn)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
            "labels": torch.tensor(labels, dtype=torch.long, device=DEVICE),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.float, device=DEVICE),
        }

        # Reconstruction targets: original (unmasked) tokens, padded to max_len
        recon_targets = []
        for seq in sequences:
            seq = seq[:max_len]
            pad_len = max_len - len(seq)
            recon_targets.append(list(seq) + [self.tokenizer.pad_id] * pad_len)
        result["recon_targets"] = torch.tensor(recon_targets, dtype=torch.long, device=DEVICE)

        return result

    def _apply_span_mask(
        self,
        masked: list[int],
        label: list[int],
        orig: list[int],
        mask_id: int,
        vocab_size: int,
    ):
        """Apply span masking: mask contiguous spans with geometric length distribution.

        Targets self.mask_prob fraction of tokens overall. Each span starts at a
        random position and extends for geom(1/mean_span_len) tokens.
        """
        seq_len = len(masked)
        n_to_mask = max(1, int((seq_len - 1) * self.mask_prob))  # -1 for [CLS]
        masked_set: set[int] = set()

        while len(masked_set) < n_to_mask:
            # Sample span start uniformly from unmasked non-CLS positions
            start = random.randint(1, seq_len - 1)
            # Geometric span length: P(len=k) = (1-p)^(k-1) * p, mean = 1/p
            span_len = min(
                int(np.random.geometric(1.0 / self.mean_span_len)),
                seq_len - start,
                n_to_mask - len(masked_set),
            )
            for i in range(start, start + span_len):
                if i not in masked_set:
                    if self.no_mask_numeric and orig[i] in self._numeric_ids:
                        continue  # skip numeric tokens in spans
                    masked_set.add(i)
                    label[i] = orig[i]
                    r = random.random()
                    if r < 0.8:
                        masked[i] = mask_id
                    elif r < 0.9:
                        masked[i] = random.randint(0, vocab_size - 1)
                    # else: keep original (10%)

    def split(self, val_frac: float = 0.15) -> tuple["AbilityMLMDataset", "AbilityMLMDataset"]:
        """Split into train/val datasets. Returns (train, val)."""
        n_val = max(1, int(len(self.encoded) * val_frac))
        indices = list(range(len(self.encoded)))
        random.shuffle(indices)

        val_ds = AbilityMLMDataset.__new__(AbilityMLMDataset)
        val_ds.tokenizer = self.tokenizer
        val_ds.mask_prob = self.mask_prob
        val_ds.augment = False  # No augmentation on val set
        val_ds.span_masking = False  # No span masking on val set
        val_ds.mean_span_len = self.mean_span_len
        val_ds.no_mask_numeric = False  # Val always masks everything for fair comparison
        val_ds._numeric_ids = self._numeric_ids
        val_ds.texts = [self.texts[i] for i in indices[:n_val]]
        val_ds.encoded = [self.encoded[i] for i in indices[:n_val]]
        val_ds.hints = [self.hints[i] for i in indices[:n_val]]
        val_ds.behavioral_ids = [self.behavioral_ids[i] for i in indices[:n_val]] if self.behavioral_ids else None

        train_ds = AbilityMLMDataset.__new__(AbilityMLMDataset)
        train_ds.tokenizer = self.tokenizer
        train_ds.mask_prob = self.mask_prob
        train_ds.augment = self.augment
        train_ds.span_masking = self.span_masking
        train_ds.mean_span_len = self.mean_span_len
        train_ds.no_mask_numeric = self.no_mask_numeric
        train_ds._numeric_ids = self._numeric_ids
        train_ds.texts = [self.texts[i] for i in indices[n_val:]]
        train_ds.encoded = [self.encoded[i] for i in indices[n_val:]]
        train_ds.hints = [self.hints[i] for i in indices[n_val:]]
        train_ds.behavioral_ids = [self.behavioral_ids[i] for i in indices[n_val:]] if self.behavioral_ids else None

        print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds


def _hash_structure(text: str) -> str:
    """Simple hash for holdout exclusion."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _build_token_type_map(tokenizer: AbilityTokenizer) -> dict[int, str]:
    """Map token IDs to type categories for per-type accuracy."""
    type_map: dict[int, str] = {}
    for tok_str, tok_id in tokenizer.tok2id.items():
        if tok_str in _STRUCTURAL_TOKENS:
            type_map[tok_id] = "structural"
        elif tok_str in _NUMERIC_TOKENS:
            type_map[tok_id] = "numeric"
        elif tok_str in _EFFECT_TOKENS:
            type_map[tok_id] = "effect"
        elif tok_str in SPECIAL_TOKENS:
            type_map[tok_id] = "special"
        else:
            type_map[tok_id] = "keyword"
    return type_map


@torch.no_grad()
def evaluate(
    model: AbilityTransformerMLM,
    dataset: AbilityMLMDataset,
    batch_size: int = 256,
    n_batches: int = 10,
    hint_head: nn.Module | None = None,
    recon_decoder: nn.Module | None = None,
    cls_proj: nn.Module | None = None,
    pad_id: int = 0,
) -> dict[str, float]:
    """Evaluate masked token accuracy and loss on validation set."""
    model.eval()
    if hint_head is not None:
        hint_head.eval()
    if recon_decoder is not None:
        recon_decoder.eval()
    if cls_proj is not None:
        cls_proj.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    # Per-token-type tracking
    type_map = _build_token_type_map(dataset.tokenizer)
    type_correct: dict[str, int] = {}
    type_total: dict[str, int] = {}

    # Hint classification tracking
    hint_correct = 0
    hint_total = 0

    # Reconstruction tracking
    recon_correct = 0
    recon_total = 0

    for _ in range(n_batches):
        batch = dataset.sample_batch(min(batch_size, len(dataset)))
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()

        # Accuracy on masked positions only
        mask_positions = batch["labels"] != -100
        if mask_positions.any():
            preds = logits.argmax(dim=-1)
            correct = (preds == batch["labels"]) & mask_positions
            total_correct += correct.sum().item()
            total_masked += mask_positions.sum().item()

            # Per-token-type accuracy
            labels_np = batch["labels"].cpu().numpy()
            correct_np = correct.cpu().numpy()
            mask_np = mask_positions.cpu().numpy()
            for b in range(labels_np.shape[0]):
                for s in range(labels_np.shape[1]):
                    if mask_np[b, s]:
                        tid = labels_np[b, s]
                        ttype = type_map.get(tid, "other")
                        type_total[ttype] = type_total.get(ttype, 0) + 1
                        if correct_np[b, s]:
                            type_correct[ttype] = type_correct.get(ttype, 0) + 1

        # Hint classification accuracy
        if hint_head is not None and "hint_labels" in batch:
            hint_labels = batch["hint_labels"]
            valid = hint_labels >= 0
            if valid.any():
                cls_emb = model.transformer.cls_embedding(batch["input_ids"], batch["attention_mask"])
                hint_logits = hint_head(cls_emb)
                hint_preds = hint_logits.argmax(dim=-1)
                hint_correct += (hint_preds[valid] == hint_labels[valid]).sum().item()
                hint_total += valid.sum().item()

        # Reconstruction accuracy
        if recon_decoder is not None and "recon_targets" in batch:
            recon_targets = batch["recon_targets"]
            cls_emb = model.transformer.cls_embedding(batch["input_ids"], batch["attention_mask"])
            recon_cls = cls_proj(cls_emb) if cls_proj is not None else cls_emb
            recon_logits = recon_decoder(recon_cls, seq_len=recon_targets.shape[1])
            recon_preds = recon_logits.argmax(dim=-1)
            non_pad = recon_targets != pad_id
            recon_correct += (recon_preds[non_pad] == recon_targets[non_pad]).sum().item()
            recon_total += non_pad.sum().item()

    model.train()
    if hint_head is not None:
        hint_head.train()
    if recon_decoder is not None:
        recon_decoder.train()
    if cls_proj is not None:
        cls_proj.train()

    acc = total_correct / total_masked if total_masked > 0 else 0.0
    result = {
        "val_loss": total_loss / n_batches,
        "masked_token_acc": acc,
        "n_masked": total_masked,
    }

    # Per-type accuracies
    for ttype in ["structural", "numeric", "effect", "keyword"]:
        t = type_total.get(ttype, 0)
        c = type_correct.get(ttype, 0)
        result[f"acc_{ttype}"] = c / t if t > 0 else 0.0

    if hint_total > 0:
        result["hint_acc"] = hint_correct / hint_total

    if recon_total > 0:
        result["recon_acc"] = recon_correct / recon_total

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def maybe_visualize(
    model: AbilityTransformerMLM,
    dataset: AbilityMLMDataset,
    step: int,
    output_dir: Path,
):
    """Generate t-SNE visualization of [CLS] embeddings if dependencies available."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    embeddings = []
    hints = []

    tok = dataset.tokenizer
    # Use the hint token that follows "hint :" in each ability
    for text in dataset.texts[:200]:  # cap for speed
        ids = tok.encode(text, add_cls=True)
        ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        cls = model.transformer.cls_embedding(ids_t)
        embeddings.append(cls[0].detach().cpu().numpy())

        # Extract hint from text
        hint = "unknown"
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("hint:"):
                hint = line.split(":", 1)[1].strip().split()[0].split(",")[0]
                break
        hints.append(hint)

    model.train()

    if len(embeddings) < 10:
        return

    emb_arr = np.array(embeddings)
    perp = min(30, len(emb_arr) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb_arr)

    label_set = sorted(set(hints))
    colors = {l: i for i, l in enumerate(label_set)}
    c = [colors[h] for h in hints]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap="tab10", s=15, alpha=0.7)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.cm.tab10(colors[l] / max(len(label_set) - 1, 1)),
               markersize=8, label=l) for l in label_set]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title(f"[CLS] embeddings step {step} (color=hint)")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"cls_hint_{step:07d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load behavioral profiles early (needed for dataset mapping)
    _behavioral_profiles: BehavioralProfiles | None = None
    if args.behavioral_data:
        _behavioral_profiles = BehavioralProfiles(args.behavioral_data, DEVICE)

    # Load dataset — npz (pre-tokenized), profiles npz (has dsl_texts), or directory
    ability_path = Path(args.ability_dir)
    if ability_path.suffix == ".npz":
        # Check if this is a profiles npz (has dsl_texts) vs pre-tokenized MLM npz (has token_ids)
        _probe = np.load(ability_path, allow_pickle=True)
        has_token_ids = "token_ids" in _probe.files
        has_dsl_texts = "dsl_texts" in _probe.files
        del _probe

        if has_token_ids:
            dataset = NpzMLMDataset(
                ability_path, tokenizer,
                mask_prob=args.mask_prob,
                augment=not args.no_augment,
            )
        elif has_dsl_texts:
            # Profiles npz — extract DSL texts and build dataset
            _data = np.load(ability_path, allow_pickle=True)
            dsl_blob = _data["dsl_texts"].tobytes().decode("utf-8")
            texts = [t.strip() for t in dsl_blob.split("---SEPARATOR---") if t.strip()]
            del _data
            print(f"Loaded {len(texts)} abilities from profiles npz")
            dataset = AbilityMLMDataset.from_texts(
                texts, tokenizer,
                mask_prob=args.mask_prob,
                augment=not args.no_augment,
            )
        else:
            raise ValueError(f"Unrecognized npz format: {ability_path}")

        if _behavioral_profiles is not None:
            dataset.build_behavioral_ids(_behavioral_profiles)
        train_ds, val_ds = dataset.split(val_frac=args.val_frac)
    else:
        dataset = AbilityMLMDataset(
            ability_path, tokenizer,
            mask_prob=args.mask_prob,
            augment=not args.no_augment,
        )
        if _behavioral_profiles is not None:
            dataset.build_behavioral_ids(_behavioral_profiles)
        train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    # Model
    model = AbilityTransformerMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
        cls_id=tokenizer.cls_id,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Auxiliary hint classification head
    hint_head: HintClassificationHead | None = None
    if not args.no_hint_loss:
        hint_head = HintClassificationHead(args.d_model, n_classes=len(HINT_CATEGORIES)).to(DEVICE)
        hint_params = sum(p.numel() for p in hint_head.parameters())
        print(f"Hint head parameters: {hint_params:,}")
        print(f"Hint loss weight: {args.hint_loss_weight}")

    # CLS projection (optional): encoder d_model → wider CLS before recon decoder
    cls_proj: nn.Module | None = None
    recon_d = args.d_model
    if args.cls_proj_dim > 0:
        recon_d = args.cls_proj_dim
        if args.cls_proj_mlp:
            cls_proj = nn.Sequential(
                nn.Linear(args.d_model, recon_d),
                nn.GELU(),
                nn.Linear(recon_d, recon_d),
            ).to(DEVICE)
            proj_params = sum(p.numel() for p in cls_proj.parameters())
            print(f"CLS MLP projection: {args.d_model} → {recon_d} ({proj_params:,} params)")
        else:
            cls_proj = nn.Linear(args.d_model, recon_d).to(DEVICE)
            proj_params = sum(p.numel() for p in cls_proj.parameters())
            print(f"CLS linear projection: {args.d_model} → {recon_d} ({proj_params:,} params)")

    # Reconstruction decoder for CLS bottleneck
    recon_decoder: ReconstructionDecoder | None = None
    if not args.no_recon:
        recon_decoder = ReconstructionDecoder(
            d_model=recon_d,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_seq_len,
            n_layers=args.recon_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
        ).to(DEVICE)
        recon_params = sum(p.numel() for p in recon_decoder.parameters())
        print(f"Reconstruction decoder parameters: {recon_params:,} (d={recon_d})")
        print(f"Reconstruction loss weight: {args.recon_weight}")

    # Behavioral outcome prediction head
    behavioral_head: BehavioralHead | None = None
    behavioral_profiles = _behavioral_profiles
    if behavioral_profiles is not None:
        behavioral_head = BehavioralHead(
            d_model=args.d_model, hidden_dim=args.behavioral_hidden,
        ).to(DEVICE)
        beh_params = sum(p.numel() for p in behavioral_head.parameters())
        print(f"Behavioral head parameters: {beh_params:,}")
        print(f"Behavioral loss weight: {args.behavioral_weight}")

    # Freeze encoder if requested (for behavioral finetuning on frozen CLS)
    if args.freeze_encoder:
        for p in model.parameters():
            p.requires_grad = False
        if hint_head is not None:
            for p in hint_head.parameters():
                p.requires_grad = False
        if recon_decoder is not None:
            for p in recon_decoder.parameters():
                p.requires_grad = False
        if cls_proj is not None:
            for p in cls_proj.parameters():
                p.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters())
        print(f"Encoder frozen: {frozen:,} params")

    # Optimizer — grokking plan §2.1
    all_params = [p for p in model.parameters() if p.requires_grad]
    if hint_head is not None:
        all_params += [p for p in hint_head.parameters() if p.requires_grad]
    if cls_proj is not None:
        all_params += [p for p in cls_proj.parameters() if p.requires_grad]
    if recon_decoder is not None:
        all_params += [p for p in recon_decoder.parameters() if p.requires_grad]
    if behavioral_head is not None:
        all_params += [p for p in behavioral_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    # Linear warmup — grokking plan §2.1
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps,
    )

    # Batch size — grokking plan §2.4
    batch_size = min(args.batch_size, len(train_ds) // 2) if len(train_ds) > 4 else len(train_ds)
    print(f"Batch size: {batch_size}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            state = torch.load(resume_path, map_location=DEVICE, weights_only=True)
            # Load model (strict=False to allow missing recon_decoder keys from older checkpoints)
            model.load_state_dict(
                {k: v for k, v in state.items() if k.startswith(("transformer.", "mlm_head."))},
                strict=False,
            )
            # Load reconstruction decoder if keys present
            if recon_decoder is not None:
                recon_keys = {k.removeprefix("recon_decoder."): v for k, v in state.items() if k.startswith("recon_decoder.")}
                if recon_keys:
                    recon_decoder.load_state_dict(recon_keys)
                    print("  Loaded reconstruction decoder state")
            # Load hint head if keys present
            if hint_head is not None:
                hint_keys = {k.removeprefix("hint_head."): v for k, v in state.items() if k.startswith("hint_head.")}
                if hint_keys:
                    hint_head.load_state_dict(hint_keys)
                    print("  Loaded hint head state")
            # Load behavioral head if keys present
            if behavioral_head is not None:
                beh_keys = {k.removeprefix("behavioral_head."): v for k, v in state.items() if k.startswith("behavioral_head.")}
                if beh_keys:
                    behavioral_head.load_state_dict(beh_keys)
                    print("  Loaded behavioral head state")
            # Infer step from CSV log
            csv_path = Path(args.output).with_suffix(".csv")
            if csv_path.exists():
                with open(csv_path) as cf:
                    for line in cf:
                        pass
                    try:
                        start_step = int(line.split(",")[0])
                    except (ValueError, UnboundLocalError):
                        pass
            print(f"  Resuming from step {start_step}")

    # Grokfast EMA gradient filter (Lee et al., 2405.20233)
    # Wrap model + aux heads in a single module for Grokfast
    class _CombinedForGrokfast(nn.Module):
        def __init__(self, main, aux, recon, proj, beh=None):
            super().__init__()
            self.main = main
            if aux is not None:
                self.aux = aux
            if recon is not None:
                self.recon = recon
            if beh is not None:
                self.beh = beh
            if proj is not None:
                self.proj = proj
    combined = _CombinedForGrokfast(model, hint_head, recon_decoder, cls_proj, behavioral_head).to(DEVICE)
    gf = GrokfastEMA(combined, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")

    # Metrics logging
    log_path = Path(args.output).with_suffix(".csv")
    if start_step > 0:
        log_file = open(log_path, "a", newline="")
    else:
        log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow([
            "step", "train_loss", "val_loss", "masked_token_acc",
            "acc_structural", "acc_numeric", "acc_effect", "acc_keyword", "hint_acc",
            "recon_acc", "beh_mse",
            "weight_norm", "grad_norm", "lr", "max_eigenvalue", "elapsed_s",
        ])

    # Training — monitor for anti-grokking via spectral diagnostics
    best_acc = 0.0
    start_time = time.time()
    model.train()

    print(f"\nStarting pre-training: max_steps={args.max_steps}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast + spectral monitoring (anti-grokking detection)")
    print(f"Device: {DEVICE}\n")

    for step in range(start_step + 1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)

        # Single encoder pass — reuse hidden states for MLM, hint, and recon
        hidden = model.transformer(batch["input_ids"], batch["attention_mask"])
        logits = model.mlm_head(hidden)
        mlm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )

        cls_emb = hidden[:, 0, :]  # [CLS] at position 0

        # Auxiliary hint classification loss
        loss = mlm_loss
        if hint_head is not None:
            hint_labels = batch["hint_labels"]
            valid = hint_labels >= 0
            if valid.any():
                hint_logits = hint_head(cls_emb[valid])
                hint_loss = F.cross_entropy(hint_logits, hint_labels[valid])
                loss = mlm_loss + args.hint_loss_weight * hint_loss

        # Reconstruction loss from CLS
        if recon_decoder is not None:
            recon_targets = batch["recon_targets"]
            recon_cls = cls_proj(cls_emb) if cls_proj is not None else cls_emb
            recon_logits = recon_decoder(recon_cls, seq_len=recon_targets.shape[1])
            recon_loss = F.cross_entropy(
                recon_logits.view(-1, recon_logits.size(-1)),
                recon_targets.view(-1),
                ignore_index=tokenizer.pad_id,
            )
            loss = loss + args.recon_weight * recon_loss

        # Behavioral outcome prediction loss from CLS
        beh_loss_val = 0.0
        if behavioral_head is not None and behavioral_profiles is not None and "behavioral_ids" in batch:
            beh_ids = batch["behavioral_ids"]  # (B,)
            beh_conds, beh_outs, beh_valid = behavioral_profiles.sample_batch(beh_ids)
            if beh_valid.any():
                beh_pred = behavioral_head(cls_emb[beh_valid], beh_conds[beh_valid])
                beh_loss = F.smooth_l1_loss(beh_pred, beh_outs[beh_valid])
                loss = loss + args.behavioral_weight * beh_loss
                beh_loss_val = beh_loss.item()

        optimizer.zero_grad()
        loss.backward()

        # Grokfast: amplify slow gradient components before optimizer step
        gf.step()

        # Track gradient norm (diagnostic — grokking plan §4.4)
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        optimizer.step()
        if step <= args.warmup_steps:
            warmup_scheduler.step()

        # Evaluation
        if step % args.eval_every == 0:
            metrics = evaluate(model, val_ds, batch_size=batch_size, hint_head=hint_head,
                              recon_decoder=recon_decoder, cls_proj=cls_proj,
                              pad_id=tokenizer.pad_id)

            # Weight norm (diagnostic — grokking plan §4.4)
            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

            # Spectral monitoring: track max eigenvalue across weight matrices
            # (anti-grokking detection — Prakash & Martin, 2602.02859)
            max_eig = 0.0
            for p in model.parameters():
                if p.ndim == 2 and p.shape[0] >= 4 and p.shape[1] >= 4:
                    try:
                        s = torch.linalg.svdvals(p.data)
                        max_eig = max(max_eig, s[0].item())
                    except Exception:
                        pass

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            log_writer.writerow([
                step, f"{mlm_loss.item():.6f}", f"{metrics['val_loss']:.6f}",
                f"{metrics['masked_token_acc']:.4f}",
                f"{metrics.get('acc_structural', 0):.4f}",
                f"{metrics.get('acc_numeric', 0):.4f}",
                f"{metrics.get('acc_effect', 0):.4f}",
                f"{metrics.get('acc_keyword', 0):.4f}",
                f"{metrics.get('hint_acc', 0):.4f}",
                f"{metrics.get('recon_acc', 0):.4f}",
                f"{beh_loss_val:.6f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{lr:.6f}",
                f"{max_eig:.4f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

            acc = metrics["masked_token_acc"]
            marker = ""
            if acc > best_acc:
                best_acc = acc
                # Save model + all aux heads in a single state dict
                save_state = dict(model.state_dict())
                if hint_head is not None:
                    for k, v in hint_head.state_dict().items():
                        save_state[f"hint_head.{k}"] = v
                if cls_proj is not None:
                    for k, v in cls_proj.state_dict().items():
                        save_state[f"cls_proj.{k}"] = v
                if recon_decoder is not None:
                    for k, v in recon_decoder.state_dict().items():
                        save_state[f"recon_decoder.{k}"] = v
                if behavioral_head is not None:
                    for k, v in behavioral_head.state_dict().items():
                        save_state[f"behavioral_head.{k}"] = v
                torch.save(save_state, args.output)
                marker = " *"

            hint_str = f" | hint {metrics.get('hint_acc', 0):.3f}" if hint_head else ""
            recon_str = f" | recon {metrics.get('recon_acc', 0):.3f}" if recon_decoder else ""
            beh_str = f" | beh {beh_loss_val:.4f}" if behavioral_head else ""
            print(
                f"step {step:>7d} | "
                f"train {mlm_loss.item():.4f} | "
                f"val {metrics['val_loss']:.4f} | "
                f"acc {acc:.4f} "
                f"[S:{metrics.get('acc_structural',0):.2f} "
                f"N:{metrics.get('acc_numeric',0):.2f} "
                f"E:{metrics.get('acc_effect',0):.2f} "
                f"K:{metrics.get('acc_keyword',0):.2f}]"
                f"{hint_str}{recon_str}{beh_str} | "
                f"w {weight_norm:.1f} | "
                f"eig {max_eig:.2f}"
                f"{marker}"
            )

        # Diagnostics
        if args.diagnostics_dir and step % args.diag_every == 0:
            maybe_visualize(model, val_ds, step, Path(args.diagnostics_dir))

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best masked token accuracy: {best_acc:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase 1: Ability transformer pre-training (MLM)")
    p.add_argument("ability_dir", help="Directory of .ability files")
    p.add_argument("-o", "--output", default="generated/ability_transformer_pretrained.pt")

    # Grokking plan settings
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0, help="High weight decay per grokking plan")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.15)

    # Architecture — 4 layers per Murty et al. (structural grokking)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=256)

    # Grokfast (Lee et al., 2405.20233) — EMA gradient filter
    p.add_argument("--grokfast-alpha", type=float, default=0.98, help="EMA decay for gradient filter")
    p.add_argument("--grokfast-lamb", type=float, default=2.0, help="Amplification of slow gradient components")

    # Span masking (SpanBERT-style)
    p.add_argument("--span-masking", action="store_true", help="Use span masking instead of single-token")
    p.add_argument("--mean-span-len", type=float, default=3.0, help="Mean span length for span masking")

    # Masking options
    p.add_argument("--no-mask-numeric", action="store_true", help="Never mask numeric/duration tokens")

    # Auxiliary hint classification loss
    p.add_argument("--no-hint-loss", action="store_true", help="Disable auxiliary hint classification loss")
    p.add_argument("--hint-loss-weight", type=float, default=0.5, help="Weight for hint classification loss")

    # Reconstruction loss (CLS bottleneck)
    p.add_argument("--no-recon", action="store_true", help="Disable reconstruction loss")
    p.add_argument("--recon-weight", type=float, default=1.0, help="Weight for reconstruction loss")
    p.add_argument("--recon-layers", type=int, default=2, help="Number of decoder layers for reconstruction")

    # CLS projection (widen CLS before recon decoder)
    p.add_argument("--cls-proj-dim", type=int, default=0, help="Project CLS to this dim before recon (0=disabled)")
    p.add_argument("--cls-proj-mlp", action="store_true", help="Use 2-layer MLP instead of linear for CLS projection")

    # Behavioral outcome prediction from CLS
    p.add_argument("--behavioral-data", type=str, default=None,
                   help="Path to ability_profiles.npz for behavioral outcome prediction")
    p.add_argument("--behavioral-weight", type=float, default=1.0,
                   help="Weight for behavioral outcome MSE loss")
    p.add_argument("--behavioral-hidden", type=int, default=256,
                   help="Hidden dim for behavioral prediction head")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze transformer encoder (train only aux heads)")

    # Resume
    p.add_argument("--resume", help="Resume from checkpoint (.pt)")
    p.add_argument("--no-augment", action="store_true", help="Disable data augmentation")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-hashes", help="Path to holdout_hashes.txt for exclusion")
    p.add_argument("--diagnostics-dir", help="Directory for t-SNE plots")
    p.add_argument("--diag-every", type=int, default=10_000)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
