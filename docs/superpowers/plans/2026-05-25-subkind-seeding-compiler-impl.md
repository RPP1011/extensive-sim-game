# Subkind Seeding — Compiler Feature Implementation Plan (Plan A — Wave 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Extend the DSL so a `.sim` declares its initial population by entity subkind (`spawn <Subkind> count <N> { … }`), with `f32` values and seeded positions, and so `render {}` can select agents by subkind — the slot array is compiler-owned.

**Architecture:** Extend the `init {}` grammar (`dsl_ast`) with `spawn <Subkind> count <N> { field: value }` blocks; extend `build_helper`'s `init`→`create_buffer_init` codegen to assign contiguous slot ranges per subkind, stamp `creature_type` + `alive`, apply int/f32 fields, and seed `pos` (`origin`/`scatter(r)`/`ring(r)` via `per_agent_u32`). Add a `creature_type is <Subkind>` selector to the `render {}` block + its descriptor emit. Internally serial (shared parser/codegen) — one plan.

**Tech Stack:** custom DSL (`dsl_ast`, `dsl_compiler`), `sims` codegen, `per_agent_u32` (P5).

---

## Architectural Impact Statement
- **Existing primitives searched:** `init_decl()`/`InitDecl`/`InitExpr::{Slot,Const}` (`dsl_ast/src/parser.rs:1072`, `ast.rs:136`); the `init`→`create_buffer_init` codegen ("Plan E-A6", `build_helper.rs:233`); `self.creature_type == <Subkind>` resolution (works — `predator_prey.sim` `HareControl`); the `render {}` emit (`cg/emit/render.rs`, engine spec Plan A); `per_agent_u32` (P5). Method: `rg`/`Read`.
- **Decision:** extend `init {}` (don't add a new block) + the render selector — seeding + role discrimination as compiler-lowered data.
- **Rule-compiler touchpoints:** DSL inputs `dsl_ast/src/{ast,parser}.rs`; generated outputs `sims` runtimes (`try_new` seeding) + render descriptors.
- **Hand-written downstream code:** NONE (all emitter-generated).
- **Constitution check:** P5 PASS (`scatter`/`ring` via `per_agent_u32`); P2 N/A (writes existing SoA cols at `try_new`); P1 PASS (compiler-lowered); P3 PASS (`creature_type` guards parity-safe); P10 PASS (headless seed tests); P8 PASS.
- **Runtime gate:** `subkind_seeding_exec` (`crates/sims/tests/`) — a probe `.sim` with two `spawn` blocks seeds the right per-subkind counts + `creature_type`s + positions (deterministic across two `try_new(seed)` calls).
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: `f32` init values
**Files:** `dsl_ast/src/ast.rs` (add `InitExpr::Float(f64)`), `dsl_ast/src/parser.rs:1107-1115` (accept floats), `dsl_compiler/src/build_helper.rs` (emit f32 bits).

- [ ] **Step 1:** Add `Float(f64)` to `InitExpr`. In `init_decl`'s number branch (`parser.rs:1107`), when `is_float` is true, produce `InitExpr::Float(n)` instead of erroring.
- [ ] **Step 2:** In `build_helper`'s init application, map `InitExpr::Float(v)` to the f32 bit pattern (`(v as f32).to_bits()`) written into the agent column (mirror how `InitExpr::Const` writes u32; floats go into f32 columns like `hp`/`mana`).
- [ ] **Step 3 (test):** a probe `.sim` `init { hp: 100.0 }`; headless test reads back `agent_hp_buf[any alive]` == 100.0. Run `RUST_MIN_STACK=33554432 cargo test -p sims --test subkind_seeding_exec`. Commit.

### Task 2: `spawn <Subkind> count <N> { … }` grammar
**Files:** `dsl_ast/src/ast.rs`, `dsl_ast/src/parser.rs`.

- [ ] **Step 1:** AST: `pub struct SpawnBlock { pub subkind: String, pub count: CountExpr, pub fields: Vec<InitStmt>, pub span: Span }` where `CountExpr = Lit(u32) | Config(String)`. Add `pub spawns: Vec<SpawnBlock>` to `InitDecl` (the flat `stmts` form stays for back-compat).
- [ ] **Step 2:** In `init_decl`, when a body item begins with `spawn`, parse `spawn <Ident:subkind> count <number|config.x> { <InitStmt,>* }` (reuse the existing field-stmt parser inside the braces). Bare `field: value` items still push to `stmts` (uniform form).
- [ ] **Step 3 (test):** parse a `.sim` with two `spawn` blocks; assert `init.spawns.len() == 2`, names + counts correct. `cargo test -p dsl_ast`. Commit.

### Task 3: Slot assignment + `creature_type` + field application
**Files:** `dsl_compiler/src/build_helper.rs` (the `init`→`create_buffer_init` region).

- [ ] **Step 1:** In the init codegen, when `init.spawns` is non-empty, compute contiguous slot ranges: start at slot 1 (skip the slot-0 `AgentId` sentinel), assign `count` slots per `spawn` block in declaration order; assert `Σcount + 1 ≤ agent_count` (typed compile error otherwise). For each block's range, emit `create_buffer_init`/fill writes: `creature_type = <subkind ordinal>` (resolve the subkind name to its `creature_type` ordinal — the same mapping `self.creature_type == <Subkind>` uses), `alive = 1` (unless the block sets it), then each declared field (int/f32 via Task 1).
- [ ] **Step 2:** Resolve the subkind→ordinal mapping: find how `resolve` assigns `creature_type` ordinals to `entity X : Agent` subkinds (search `creature_type` in `dsl_ast/src/resolve.rs`); reuse it so the seeder's stamp matches the rule-guard comparison.
- [ ] **Step 3 (test):** probe `.sim` `init { spawn A count 2 { hp: 10.0 }  spawn B count 3 { alive: 0 } }`; headless: 2 agents with creature_type=A/alive=1/hp=10, 3 with creature_type=B/alive=0. Commit.

### Task 4: Seeded positions (`origin` / `scatter(r)` / `ring(r)`)
**Files:** `dsl_ast/src/{ast,parser}.rs` (position-builtin init values), `dsl_compiler/src/build_helper.rs` (seeded fill).

- [ ] **Step 1:** Add `InitExpr::Pos(PosBuiltin)` where `PosBuiltin = Origin | Scatter(f64) | Ring(f64)`. Parse `origin` / `scatter(<num>)` / `ring(<num>)` as the value of a `pos:` field.
- [ ] **Step 2:** Codegen: for a `pos` field, emit per-slot fills. `Origin` → `[0,0,0]`. `Scatter(r)`/`Ring(r)` → compute per-slot via `per_agent_u32(seed, slot, 0, PURPOSE)` (P5): scatter = uniform point in a radius-`r` disc; ring = on the radius-`r` circle. Use the host-side `per_agent_u32` (the same fn the engine uses; `crates/engine/src/...`) so positions are deterministic for a given `(seed, slot)`.
- [ ] **Step 3 (test):** probe `init { spawn A count 8 { pos: scatter(40.0) } }`; headless: all 8 positions within radius 40 of origin; two `try_new(0x1234)` runs produce identical positions (P5). Commit.

### Task 5: `render {}` `creature_type is <Subkind>` selector
**Files:** `dsl_ast/src/{ast,parser}.rs` (render agent-visual grammar), `dsl_compiler/src/cg/emit/render.rs` (descriptor emit).

- [ ] **Step 1:** Extend the render `agent` visual: alongside `agent when <field> in [lo,hi] { color … }`, accept `agent when creature_type is <Subkind> { color … }`. Lower it to the same `RenderDescriptor::AgentVisual` `FieldRange` shape using `field: "creature_type"`, `lo == hi == <subkind ordinal>` (so the player's existing field-range renderer matches it exactly — no new descriptor variant).
- [ ] **Step 2 (test):** compile-gate: a `.sim` render block with `creature_type is Player` emits an `AgentVisual` whose `when.field == "creature_type"` and `lo==hi==<ordinal>`, parseable by `engine_play_api::RenderDescriptor::from_json`. Run `cargo test -p dsl_compiler`. Commit.

## Self-review note
The subkind→`creature_type` ordinal mapping is the contract Plans B/C's `== <Subkind>` guards rely on — Task 3 Step 2 must reuse the resolver's existing mapping (the one `self.creature_type == Hare` already uses), not invent one. Render selector reuses the existing `FieldRange` descriptor (lo==hi==ordinal) so the Wave-1 engine_play bridge needs no change.
