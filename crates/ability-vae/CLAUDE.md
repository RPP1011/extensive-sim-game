# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

Not one VAE — it's a small research playground of several generative approaches for
producing `.ability` DSL text (the ability/passive syntax parsed by `dsl_compiler`'s
grammar), built on the `burn` ML framework:

- **Token-sequence VAE** (`main.rs` + `model.rs` + `data.rs`, binary `train-ability-vae`):
  the crate's namesake. Tokenizes ability DSL text, encodes with a transformer into a
  16-dim latent (μ, σ), decodes autoregressively with cross-attention on `z`. Standard
  VAE loss (reconstruction + KL warmup) plus word dropout to force reliance on `z`.
- **Grammar-space flow matching** (`grammar_space.rs` + `diffusion.rs` + `guided_sample.rs`,
  binary `train-diffusion`): a hand-built invertible mapping `[0,1]^48 ↔ valid DSL text`
  (every point in the unit hypercube decodes to a syntactically valid ability). A
  conditional-flow-matching model (`FlowModel`/`FiLMFlowModel`) learns to map Gaussian
  noise to points in this space, conditioned on either one-hot labels or a text
  embedding, with classifier-free guidance. `guided_sample.rs` adds a keyword-based
  natural-language → constraint parser so a description like `"fire damage AoE with
  stun, 8s cooldown"` fixes some of the 48 dims before sampling the rest.
- **Text encoder** (`text_encoder.rs`, binary `train-text-encoder`): a from-scratch
  bag-of-(word + char-ngram) transformer embedder trained with Matryoshka
  Representation Learning (truncatable at 64/128/256 dims) via in-batch InfoNCE,
  pretrained on STS-B then fine-tuned on (description, grammar-vector) pairs.
- **Tree decoder** (`tree_decoder.rs`, binary `train-tree`): an alternative to the flow
  model — autoregressively predicts the 48 grammar dims one at a time (categorical
  softmax over 32 bins for discrete dims, sigmoid regression for continuous ones),
  cross-attending on the text encoder's memory, with a KV-cache-free incremental
  generation loop and auxiliary classification heads (hint/element/target/domain).
- **Context-conditioned generation** (`context_encoder.rs` + `ability_generator.rs`):
  a non-text path — encodes structured game state (behavior ledger, class/archetype
  hashes, settlement context, existing-ability set) into a conditioning vector via a
  small transformer, and a parallel rule-based `AbilityGenerator` that deterministically
  maps (archetype, tier, level, behavior) to a grammar-space vector without any model,
  for synchronous in-game use.
- **Quality scoring** (`quality.rs`): hand-written heuristics (coherence, power
  balance, purpose-matching, tag consistency) that score a 48-dim vector 0.0–1.0;
  used to filter/rank generated candidates (`generate-dataset` binary samples 50K
  random vectors, keeps the top 5K plus feature-targeted batches).
- **Grokfast** (`grokfast.rs`): a generic `burn` gradient-EMA filter (Lee et al.)
  usable by any of the training loops to accelerate grokking.

`lib.rs` only exports modules used by the `bin/*` targets and downstream code
(`grammar_space`, `diffusion`, `guided_sample`, `text_encoder`, `quality`, `grokfast`,
`tree_decoder`, `ability_generator`, `context_encoder`) — `model.rs` and `data.rs`
(the token-sequence VAE) are private to `main.rs` and not part of the public API.

## Status: currently non-buildable

This crate depends on `tactical_sim = { path = "../tactical_sim" }` (`crates/tactical_sim`)
for `tactical_sim::effects::dsl::parse_abilities` (DSL parsing/validation, used
throughout for round-trip checks and in `#[cfg(test)]`) and
`tactical_sim::sim::ability_transformer::{tokenizer::AbilityTokenizer, tokenizer_vocab::VOCAB}`
(used only by `data.rs`/`model.rs`, the token-sequence VAE). `crates/tactical_sim` was
deleted in the Phase 7 wolf-sim wipe (2026-05-02, see root `CLAUDE.md`) and never
replaced — there is no `parse_abilities` or `AbilityTokenizer` equivalent anywhere else
in the current crate set (`dsl_ast`/`dsl_compiler` don't expose either). Confirmed by
attempting `cargo build --manifest-path crates/ability-vae/Cargo.toml`, which fails at
dependency resolution because `crates/tactical_sim/Cargo.toml` doesn't exist.
`crates/combat-trainer` (the other excluded ML crate) has the exact same dangling
dependency. Fixing this crate means either restoring/reimplementing `tactical_sim`'s
DSL parser + ability tokenizer, or repointing these calls at `dsl_ast`/`dsl_compiler`.

## Build & test

This crate is in the workspace root `Cargo.toml`'s `exclude` list, has no `[workspace]`
table of its own, and is **not** a member of the root workspace's package graph (verify
with `cargo metadata --no-deps` from the repo root — `ability-vae` is absent from the
package list). It is an implicit standalone single-package workspace rooted at its own
`Cargo.toml`. `cargo build -p ability-vae` from the repo root will not find it. Build it
with an explicit manifest path instead:

```bash
cargo build --manifest-path crates/ability-vae/Cargo.toml --release
cargo test --manifest-path crates/ability-vae/Cargo.toml
```

(Or `cd crates/ability-vae` and run bare `cargo build`/`cargo test` — it resolves its
own `Cargo.lock`/target dir independent of the workspace root.) Both currently fail per
**Status** above until the `tactical_sim` dependency is resolved.

Feature flags select the burn backend — `gpu` (default: `burn/tch` + CUDA via
`LibTorchDevice::Cuda(0)`, hardcoded in each `fn main`) or `cpu` (`burn/ndarray`).
Use `--no-default-features --features cpu` for a GPU-less machine. Seven `[[bin]]`
targets are declared (`train-ability-vae`, `train-diffusion`, `train-text-encoder`,
`generate-dataset`, `train-e2e`, `train-tree`, `generate-abilities`) — pass `--bin
<name>` to build/run one in isolation.

## Data flow

Training input for the VAE and flow-matching paths is `dataset/abilities/**/*.ability`
(recursively discovered, split into individual `ability`/`passive` blocks by brace
matching in `data.rs`/`train_diffusion.rs`'s local `split_ability_blocks`). The
flow-matching path additionally round-trips through `grammar_space::encode` (best-effort
DSL → 48-dim vector, used only to build training targets) and `grammar_space::decode`
(48-dim vector → DSL, the actual generative direction — this is the one guaranteed to
always produce parseable output). `train-e2e` and `train-tree` jointly train a
`text_encoder::StaticEmbedder` against the flow model / tree decoder respectively, so
inference can run purely from a natural-language ability description with no
`grammar_space::encode` step. Trained weights are meant to be saved as burn `NamedMpkFileRecorder`
checkpoints (see `generate_abilities.rs`'s loading code) — no checkpoint files are
committed to the repo.

## Non-obvious things

- **Two disjoint generative representations, not one pipeline.** The token-sequence VAE
  (`model.rs`) operates directly on tokenized DSL text with a learned vocab from
  `tactical_sim`'s tokenizer. The flow-matching/tree-decoder path operates on the
  hand-designed 48-dim `grammar_space` instead, which is not learned — it's a fixed,
  manually-authored bijection-ish encode/decode pair with lookup tables (`COMBAT_AMOUNT`,
  `CAMPAIGN_DURATION`, etc.) and `lerp`/`log_lerp` scaling. These two approaches don't
  share model weights or training code; picking one is a per-binary decision.
- **`grammar_space::encode` is lossy and partial by design** ("mainly for training data
  encoding" per its doc comment) — it only reads the *first* ability/passive/effect in a
  parsed block and defaults most categorical dims to a fixed bin rather than truly
  inverting `decode`. Don't rely on `decode(encode(x)) == x`.
- **Domain split (combat vs. campaign) changes the effect vocabulary entirely** —
  `D_DOMAIN` (dim 1) selects between disjoint `COMBAT_*`/`CAMPAIGN_*` const tables for
  targets, hints, and effects; campaign effects never get area/tag/condition modifiers
  or delivery blocks, combat effects always can.
- Random sampling throughout this crate uses a hand-rolled 64-bit LCG (`rng =
  rng.wrapping_mul(6364136223846793005).wrapping_add(1)`), not the `rand` crate despite
  it being a listed dependency (only used, if at all, incidentally) — this is unrelated
  to the main workspace's `per_agent_u32` determinism convention (this crate has no
  simulation determinism requirement, it's offline training tooling).
- `tree_decoder.rs`'s `generate()` claims to avoid an O(T²) cost via incremental
  embedding, but still re-runs the full `TransformerDecoder::forward` over the entire
  accumulated sequence each step (no real KV cache) — the comment overstates what it does.
- The autoregressive `AbilityVAE::generate` in `model.rs` also has no KV cache and
  rebuilds the whole sequence tensor every step; its doc comment says so explicitly and
  recommends batch size 1-4.
