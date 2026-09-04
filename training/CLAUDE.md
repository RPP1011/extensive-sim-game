# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this holds

7 standalone PyTorch scripts (`analyze_vae_dataset.py`, `eval_vae_quality.py`,
`export_content_vae.py`, `train_ability_vae.py`, `train_ability_vqvae.py`,
`train_content_vae.py`) plus `data/holdout_hashes.txt` and `data/holdout_structures.jsonl`
(480 lines each — a held-out set of generated ability DSL blocks keyed by hash, e.g.
`{"combo": "tether_ring", "dsl": "ability HoldoutTR0 { ... }", "hash": "..."}`).

This is a *pure-Python* VAE/VQ-VAE research track for compressing ability/class DSL into
a latent space — independent of, and architecturally different from, `crates/ability-vae`
(which is a Rust `burn`-based token-sequence VAE / flow-matching model). The scripts here
operate on a fixed **142-dim ability slot vector** / **75-dim class slot vector** /
**124-dim context input vector** layout (documented in `eval_vae_quality.py`'s docstring
and `analyze_vae_dataset.py`'s feature-group comment), not on tokenized DSL text.

- `train_ability_vae.py` / `train_ability_vqvae.py`: train a conditional VAE / VQ-VAE
  on `generated/ability_dataset.npz` (142-dim slots + 19-way archetype one-hot), export
  weights to `generated/ability_{vae,vqvae}_weights.json` for Rust inference.
- `train_content_vae.py`: trains a larger conditional VAE (`ContentVAE`, factored
  decoder heads) on `generated/vae_training_data.npz` (+ optional class data), jointly
  modeling both ability and class content types with a content-type classifier head.
- `export_content_vae.py`: exports a `ContentVAE` checkpoint's weights to JSON.
- `eval_vae_quality.py`: loads exported VAE/VQ-VAE JSON weights and
  `generated/ability_dataset.npz`, reports reconstruction MSE, categorical accuracy,
  and prior-sample quality.
- `analyze_vae_dataset.py`: distribution/dead-dimension diagnostics over sweep
  contexts, the final dataset, or the LLM content store JSONL.

## How to invoke

```bash
uv run --with torch --with numpy python training/train_ability_vae.py
uv run --with torch --with numpy python training/train_ability_vqvae.py
uv run --with numpy --with torch --find-links https://download.pytorch.org/whl/cu124 \
    python3 training/train_content_vae.py --data generated/vae_training_data.npz \
    --class-data generated/vae_training_class.npz --epochs 200 --latent-dim 32
uv run --with torch --find-links https://download.pytorch.org/whl/cu124 \
    python3 training/export_content_vae.py --checkpoint generated/content_vae/content_vae_best.pt
uv run --with torch --with numpy python training/eval_vae_quality.py
uv run --with pandas python3 training/analyze_vae_dataset.py --all
```
All scripts read from a `generated/` directory that does not exist in this checkout —
see Live or stale below for how that data would have been produced.

## What in `crates/` depends on this

Nothing. No crate reads `training/data/holdout_hashes.txt` or
`training/data/holdout_structures.jsonl`, and no crate reads the exported
`*_weights.json` files these scripts produce (searched all of `crates/` — no matches
for `holdout_hashes`, `holdout_structures`, `ability_vae_weights`,
`ability_vqvae_weights`, or `content_vae_weights`). The intended Rust consumer
(inference code that would load `generated/ability_vae_weights.json`) does not exist
in the current crate set.

## Live or stale

**Stale.** Every training script's `generated/*.npz` input was meant to come from
`cargo run --bin xtask -- synth-abilities ...` / `xtask vae-dataset` (named explicitly
in `train_ability_vae.py`'s and `analyze_vae_dataset.py`'s own error messages), and
`xtask` was retired in the Phase 7 wolf-sim wipe (2026-05-02; see root `CLAUDE.md` and
`scripts/CLAUDE.md`). There is currently no supported way to regenerate the `.npz`
files this directory needs. The `data/holdout_*` files are the one committed artifact
here, but nothing reads them. Treat this as a pre-wipe research track with no live
Rust counterpart today — `crates/ability-vae` (the closest thing to a "live" successor)
is itself non-buildable per its own `CLAUDE.md` (dangling `tactical_sim` dependency).
