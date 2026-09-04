# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate does

`ability_operator` trains an **Ability Latent Operator**: a small transformer model that predicts
the post-cast game state (HP/shield/resource deltas, CC state, position deltas, death) from the
pre-cast state plus a frozen ability embedding. It is an offline ML crate (Burn framework), not
part of the deterministic sim engine — it's a research/training tool for approximating ability
effects in latent space, presumably for RL rollout speedups or a learned combat surrogate.

There is no runtime/inference-only path here — the crate is training-only (one binary,
`train_operator`, plus a library of model/data/loss code it calls into).

## Commands

This crate is in the workspace root `Cargo.toml`'s `[workspace] exclude` list, so it is **not**
built or tested by plain `cargo build` / `cargo test` from the repo root. Always pass `--manifest-path`
or run from inside the crate directory:

```bash
cargo build --manifest-path crates/ability_operator/Cargo.toml
cargo test --manifest-path crates/ability_operator/Cargo.toml

# Train (from repo root):
cargo run --manifest-path crates/ability_operator/Cargo.toml --bin train_operator -- \
    --data <path/to/dataset.npz> --backend tch   # or --backend wgpu
```

`--backend tch` (the CLI default) uses libtorch/CUDA via the `tch` crate; `--backend wgpu` uses
Burn's wgpu backend. `tch` requires a working libtorch install — expect build/link issues if it
isn't set up; `wgpu` is the easier path to just get something running.

## Architecture / key types (all in `src/`)

- **`model.rs`** — the model itself, three stages composed in `AbilityLatentOperator`:
  - `StateEncoder`: projects heterogeneous input tokens (entities, threats, positions, ability
    slots — each with its own linear projection + a shared `type_emb` embedding) into a single
    sequence, runs it through a `TransformerEncoder`, and returns the first `MAX_ENTITIES` (7)
    tokens as the latent entity state `z_before`.
  - `AbilityOperator`: builds one "ability token" from the frozen ability CLS embedding + caster
    slot embedding + sinusoidal duration encoding, appends it to `z_before`, and runs a second
    (smaller) transformer so entities can attend to the ability token — this is where the model
    learns AoE/targeting-style asymmetric attention. Outputs `z_after`.
  - `DecoderHeads`: five small heads reading `z_after` — `GaussianHead` (mean, log_var) for hp,
    cc-duration, position, and `BinaryHead` (logits) for is-stunned and exists/death.
  - Dimensions/constants (entity/threat/position/ability feature widths, `D_MODEL=64`,
    `N_HEADS=8`, encoder/operator layer counts, `MAX_ENTITIES=7`, etc.) all live at the top of
    `model.rs` — check there before touching shapes elsewhere.
- **`data.rs`** — `OperatorDataset<B>`: loads a pre-generated `.npz` file (via `ndarray-npy`)
  straight into GPU tensors, and does `train_val_split` by held-out `scenario_ids` (last 20% of
  unique scenario IDs go to val — not a random row split). Also has `to_inner()` to convert an
  autodiff-backend dataset to its inner backend for eval.
- **`loss.rs`** — `beta_nll` (variance-weighted NLL for the Gaussian heads) and
  `bce_with_logits` for the binary heads, combined per-sample by `compute_loss`. `LossMask`
  (derived from an 80-dim ability property vector via `LossMask::from_props`) gates which of the
  four loss groups (hp/cc/pos/exists) get gradient per sample — e.g. a pure-damage ability
  contributes no position loss. The `PROP_*` constants are indices into that 80-dim vector and
  must stay in sync with whatever generates the dataset.
- **`grokfast.rs`** — `GrokfastEma`: an EMA gradient filter (Lee et al. "Grokfast") meant to
  accelerate grokking. **Note:** `GrokfastEma::apply()` is a documented no-op stub — the real
  per-tensor filtering path (`apply_to_tensor`) exists and is tested, but `train.rs` currently
  constructs a `GrokfastEma` (`_grokfast`, underscore-prefixed) and never calls either method in
  the training loop. Grokfast is effectively wired up but not active.
- **`train.rs`** — the training loop (`train()`) and `evaluate()`. Notable choices: manual LCG
  shuffle (no `rand` dependency), AdamW with grad-norm clipping, loss/metric accumulation kept on
  GPU tensors with a single CPU sync per eval interval (perf-motivated), checkpointing via Burn's
  `NamedMpkFileRecorder` whenever val loss improves.
- **`src/bin/train.rs`** — the `train_operator` CLI (clap), wires args → `TrainConfig`, picks the
  `tch` or `wgpu` backend, loads the dataset, calls `train::train`.

## Non-obvious things

- **Dataset input is undocumented/stale in one place**: `data.rs`'s module doc says the dataset
  comes from `xtask scenario oracle operator-dataset`, but the workspace root CLAUDE.md notes
  `xtask` was retired (Phase 7, 2026-05-02). Whatever currently generates the `.npz` fixture this
  crate expects is not obviously present in the workspace — check `docs/` or git history before
  assuming that command still exists.
- **`ability_cls` dimension mismatch in doc comments**: `data.rs` documents `ability_cls` as
  `(N, 32)` ("frozen ability CLS embedding"), but the actual tensor is reshaped to
  `ABILITY_CLS_DIM` = **128** (`model.rs`), consistent with `ABILITY_SLOT_DIM = 130` (128 CLS + 2
  scalar features). Trust the `128` constant in `model.rs`, not the `32` in the doc comments.
- **No Cargo dependency on `ability-vae` or `combat-trainer`.** All three are excluded ML/RL
  crates in the workspace, but there is no path dependency between any of them — checked both
  directions in all three `Cargo.toml`s. The relationship to `ability-vae` is purely a data
  pipeline convention: the "frozen ability CLS embedding" this crate consumes as
  `ability_cls`/`ABILITY_CLS_DIM` almost certainly originates from `ability-vae`'s VAE encoder
  (which also pools a `[CLS]` token, see `crates/ability-vae/src/model.rs`), produced out-of-band
  and baked into the `.npz` dataset — there's no code-level link, only a shared embedding
  convention. `combat-trainer` (RL training against `rl4burn` + `tactical_sim`) appears unrelated
  to this crate entirely; no shared types, deps, or naming convention found.
- **`burn` is pulled in with both `wgpu` and `tch` feature backends simultaneously** (see
  `Cargo.toml`), so building this crate pulls in both wgpu and libtorch toolchains regardless of
  which backend you actually run with.
