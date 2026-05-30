# Declarative Subkind Seeding — Plan Index & Parallelization

> Spec: `docs/superpowers/specs/2026-05-25-declarative-subkind-seeding-design.md`.
> Three plans. **Wave 1 = Plan A** (the compiler feature — one serial track, shared parser/codegen). **Wave 2 = Plan B ∥ Plan C** (the two game migrations — disjoint `.sim` files, built against Plan A's frozen grammar).

## Simplification vs the spec
The spec's "wave drain sets `creature_type = Enemy`" step is **dropped**: seed the Enemy pool with `spawn Enemy count N { alive: 0 }` (stamps `creature_type = Enemy`, alive=0). The drain (`summon_alloc.rs`) only writes `alive`/`hp`/`pos`/`move_speed` — never `creature_type` — so flipping a pool slot to `alive: 1` leaves its seeded `Enemy` subkind intact. No drain change, no per-game Rust.

## The frozen grammar contract (defines Wave-2 parallelism)
Plan A lands and freezes:
- **Seeding:** `spawn <Subkind> count <N|config.x> { <field>: <int|f32|slot|origin|scatter(r)|ring(r)>, ... }`. Compiler assigns slots (skipping slot-0), stamps `creature_type = <Subkind>` ordinal + `alive: 1` default, applies fields, seeds `pos`.
- **Render selector:** `agent when creature_type is <Subkind> { color (r,g,b) }` (alongside the existing `mana in [lo,hi]` form).
- **Subkind gating in rules:** `self.creature_type == <Subkind>` / `candidate.creature_type == <Subkind>` (already supported — predator_prey's `HareControl` uses it).
Plans B and C author `.sim` against exactly this; they share no files with each other or with A's compiler code.

## Plans & ownership
| # | Plan file | Depends on | Owns (files) |
|---|-----------|-----------|--------------|
| A | `…-subkind-seeding-compiler-impl.md` | — | `dsl_ast/src/{ast,parser}.rs`; `dsl_compiler/src/build_helper.rs` + `cg/emit/render.rs`; a probe `.sim`; `dsl_compiler`/`sims` seeding tests |
| B | `…-subkind-seeding-vs-impl.md` | A | `assets/sim/vampire_survivors.sim`; `crates/sims/tests/vampire_survivors_exec.rs` |
| C | `…-subkind-seeding-predator-impl.md` | A | `assets/sim/predator_prey.sim`; `crates/sims/tests/predator_prey_playable.rs` |

## Waves
```
Wave 1:  Plan A  (compiler: f32 init + spawn blocks + slot assignment + seeded positions + render creature_type selector)
Wave 2:  Plan B ∥ Plan C   (VS migration ∥ predator_prey migration — disjoint .sim files, both build on A)
```
After Wave 1 lands + is pushed to `origin/main`, dispatch B and C as parallel isolated agents (they branch from the updated main that has Plan A's grammar). Per the established workflow (the isolation-from-main gotcha): **merge Wave 1 to main first** so the Wave-2 agents have the new grammar.

## Unblocks
Plan B making `play vampire_survivors` a real game is the parity gate for the **engine spec's Plan E** (delete `vs_viewer`/`VsBridge`/`vs_ui`) — run that after B's visual parity check. Plan C completes the zero-Rust generality proof (`play predator_prey`).

## Done = all of
- `cargo test -p dsl_compiler -p sims` green (seeding + both games' playable tests).
- `make_playable("vampire_survivors")` seeds a live player + Enemy pool the drain fills; `make_playable("predator_prey")` seeds a player Hare + autonomous Hares + Wolves at scattered positions.
- Both `.sim`s gate by `creature_type` (mana-band retired); `play <both>` are real games (visual, user-side).
