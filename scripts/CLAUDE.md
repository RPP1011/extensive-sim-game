# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this holds

17 standalone Python and shell scripts (no package, no `pyproject.toml`) for an
LLM-driven content-generation pipeline: generate ability/class DSL text via an LLM
(vLLM, Ollama, or Gemini API), score/validate it, extract slot vectors via a Rust
binary, and assemble training data for VAE models. A second, unrelated cluster of
`run_*.sh` wrappers launches `ability-vae` training binaries. One script
(`validate_mcts_data.py`) is unrelated to content-gen — it validates MCTS bootstrap
export data.

Scripts all use `uv run --with <deps> python3 scripts/<name>.py` (no venv/lockfile
committed) and read/write a `generated/` directory that does not exist in this
checkout — every script is meant to be run from the repo root.

Rough grouping:
- **LLM content generation**: `content_gen_server.py` (vLLM stdin/stdout server),
  `generate_content.py` / `ollama_generate_from_sweep.py` (Ollama), `gemini_generate.py`
  (Gemini API), `generate_descriptions.py` / `generate_quality_dataset.py` (LFM via
  vLLM, natural-language descriptions for `.ability` files).
- **Dataset assembly**: `build_vae_dataset.py`, `extract_slots.py` (shells out to a
  Rust `xtask` binary — see Stale below), `map_content_to_combat_dsl.py` (rewrites the
  simplified content-gen ability DSL into the combat-sim DSL dialect).
- **Analysis**: `sweep_stats.py`, `validate_mcts_data.py`.
- **Misc**: `engine_roundtrip.py` (safetensors round-trip check).
- **ability-vae launchers**: `run_vae.sh`, `run_diffusion.sh`, `run_e2e.sh`,
  `run_text_encoder.sh`, `run_tree.sh` — thin wrappers exporting `LIBTORCH`/
  `LD_LIBRARY_PATH` (hardcoded to a specific machine's `~/.cache/uv/...` path) then
  `cargo run -p ability-vae --release --bin <name>`.

## How to invoke

```bash
uv run --with pandas python3 scripts/build_vae_dataset.py --contexts ... --store ... --output ...
uv run --with httpx python3 scripts/generate_content.py --type ability --context "level 10 stealth ranger"
uv run --with google-genai --with pandas python3 scripts/gemini_generate.py --contexts ... --output ...
uv run --with numpy scripts/validate_mcts_data.py <path_to_mcts_export.jsonl>
bash scripts/run_vae.sh          # cargo run -p ability-vae --release
```
Each script's own docstring/header comment has the exact invocation and required
`--with` extras; several also expect an external server (vLLM on `:8000`/`:8100`,
Ollama on `:11434`, or `GEMINI_API_KEY`/`GOOGLE_API_KEY` set) already running.

## What in `crates/` depends on this

Nothing currently builds against it. `extract_slots.py` and the doc comments in
several scripts (`build_vae_dataset.py`, `analyze_vae_dataset.py` in `training/`)
invoke `cargo run --bin xtask -- vae-dataset ...` / `synth-abilities` / `mcts-bootstrap`
— **`xtask` was retired in the Phase 7 wolf-sim wipe (2026-05-02)** and does not exist
anywhere in this workspace (confirmed: no `xtask` package, only historical comments
referencing its retirement in `Cargo.toml`, `assets/config/default.toml`,
`crates/engine_gpu_rules/Cargo.toml`). The `run_*.sh` wrappers call
`cargo run -p ability-vae --release --bin <name>`, but `crates/ability-vae` is
workspace-`exclude`d and, per its own `crates/ability-vae/CLAUDE.md`, currently fails
to build at all (dangling dependency on the deleted `crates/tactical_sim`). So `-p
ability-vae` from the repo root won't even resolve the package.

`generate_descriptions.py` / `generate_quality_dataset.py` read `dataset/abilities/**/*.ability`,
which does exist in the repo and is live (also used by `crates/dsl_compiler`'s parser
regression corpus and `crates/combat-trainer`'s hero templates).

## Live or stale

Mostly **stale**. The scripts are self-contained and would run if their external
services/inputs existed, but the Rust-side pipeline they're built to feed
(`xtask vae-dataset`/`synth-abilities`/`mcts-bootstrap`, `ability-vae`'s training
binaries) is gone or non-buildable post Phase-7 wipe. Treat this directory as
pre-wipe tooling nobody has updated to match the current crate layout. The
LLM-calling scripts (`generate_content.py`, `gemini_generate.py`,
`ollama_generate_from_sweep.py`, `generate_descriptions.py`) are the most
self-contained/independently runnable — they only need an LLM endpoint and
`dataset/abilities/`, not the Rust pipeline.
