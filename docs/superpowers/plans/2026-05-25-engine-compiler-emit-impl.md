# Engine Compiler Emit Implementation Plan (Plan A — Wave 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Emit the three descriptors (`render`/`controls`/`ui`) from new `.sim` blocks, and generate a `PlayableRuntime` impl + `make_playable(name)` registry per fixture — so any compiled `.sim` is consumable by the generic player.

**Architecture:** Add three top-level DSL blocks (parser/AST in `dsl_ast`), each lowered to a `&'static str` JSON descriptor on the generated runtime (the exact mechanism the playable-VS Plan 5 used for `ui {}`: hand-emit JSON in `build_helper`, expose via accessor). Generate a `PlayableRuntime` impl wrapping the existing `step`/`set_config_*`/buffers, plus a `sims::make_playable` registry. JSON must match `engine_play_api`'s serde shapes (Plan 0, frozen).

**Tech Stack:** custom DSL (`dsl_ast`, `dsl_compiler`), `sims` build.rs codegen. **Depends on Plan 0** (the schemas to match). Internally serial (all tasks touch the shared parser/emit) — this is one plan precisely to avoid merge-tax across the three blocks.

---

## Architectural Impact Statement
- **Existing primitives searched:** `ConfigDecl`/`config_decl()` (`dsl_ast/src/parser.rs:3398`) as the block-grammar template; the `ui_descriptor()` emit pattern (playable-VS Plan 5, `cg/emit/ui_model.rs`); `RuntimeCfgField`/`set_config_*` (`cg/emit/kernel.rs:2600`, `build_helper.rs:2270`); `engine_play_api` schemas (Plan 0). Method: `rg`/`Read`.
- **Decision:** extend the DSL with `render`/`controls`/`ui` blocks + a generated trait impl — UI/render/controls become compiler-lowered data (P1-positive). The `ui` block is the absorbed Plan 5.
- **Rule-compiler touchpoints:** DSL inputs `dsl_ast/src/{ast,parser}.rs`; generated outputs `sims` runtimes (descriptors + trait impl + registry).
- **Hand-written downstream code:** NONE (all emitter-generated).
- **Constitution check:** P1 PASS (compiler-lowered data); P2 N/A (inert strings); P3 PASS (descriptors are host-side, parity-irrelevant); P5/P6/P7/P11 N/A; P10 PASS (gated by Plan D's runtime test); P8 PASS.
- **Runtime gate:** `descriptors_emit` compile-gate (`dsl_compiler/tests/engine_descriptors_emit.rs`) — a `.sim` with all three blocks emits `render_descriptor()`/`controls_descriptor()`/`ui_descriptor()` whose JSON parses via the Plan-0 `from_json`.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: `PlayableRuntime` impl + `make_playable` registry (no new grammar)
**Files:** Modify `crates/dsl_compiler/src/build_helper.rs`; create `crates/dsl_compiler/src/cg/emit/playable.rs`; `sims` deps.

- [ ] **Step 1:** Add `engine_play_api = { path = "../engine_play_api" }` to `crates/sims/Cargo.toml`.
- [ ] **Step 2:** In `build_helper.rs`, where the `GeneratedRuntime` impl block is emitted, generate an `impl engine_play_api::PlayableRuntime for GeneratedRuntime`:
  - `tick` → `self.tick`; `step` → call existing `step()` (the summon drain, where a fixture uses it, is already inside the viewer path — for now `step` calls the generated step; the drain is wired in Plan D for fixtures that need it).
  - `set_input(field, value)` → a generated `match field { "ctl.move_x" => self.set_config_ctl_move_x(value), … , "ctl.bolt_rate_level" => self.set_config_ctl_bolt_rate_level(value as u32), _ => {} }` built from the `@runtime` field list the emitter already has (`RuntimeCfgField`s). Field key = `"<block>.<field>"`.
  - `agent_snapshot` → readback of the common columns (pos/alive/hp/mana/move_speed/creature_type) — emit only for buffers that exist; default missing columns to 0. (Mirror the readback in `viewer_runtime/src/vs.rs`.)
  - `view_value(view, slot)` → `match view { "<name>" => f32::from_bits(readback(self.view_storage_<name>_primary_buf)[slot]), _ => 0.0 }` for each materialized view.
  - `render_descriptor`/`controls_descriptor`/`ui_descriptor` → return the emitted `&'static str` (Tasks 2-4; empty `"{}"`-shaped default until then).
- [ ] **Step 3:** Generate `sims::make_playable` — in `crates/sims/src/lib.rs` (or a generated module), `pub fn make_playable(name: &str, seed: u64, agents: u32) -> Option<Box<dyn engine_play_api::PlayableRuntime>>` matching each fixture name → `Box::new(<fixture>::GeneratedRuntime::try_new(seed, agents)?)`. The fixture-name list is the same `build.rs` list.
- [ ] **Step 4 (test):** `crates/sims/tests/playable_registry.rs` — `make_playable("vampire_survivors", SEED, 512)` is `Some`, `.tick()==0`, `.step()` advances tick, `.set_input("ctl.move_x", 1.0)` then snapshot is non-empty. Run `RUST_MIN_STACK=33554432 cargo test -p sims --test playable_registry`. Commit.

### Task 2: `controls {}` block (simplest grammar first)
**Files:** Modify `dsl_ast/src/{ast,parser}.rs`; create `dsl_compiler/src/cg/emit/controls.rs`; wire into `build_helper.rs`.

- [ ] **Step 1:** AST: `pub struct ControlsDecl { pub bindings: Vec<ControlBinding>, pub span: Span }`, `pub struct ControlBinding { pub key: String, pub field: String, pub value: f64, pub press: bool }`. Add `controls: Option<ControlsDecl>` to the program AST.
- [ ] **Step 2:** Parser `fn controls_decl()` for: `controls { key "w" -> ctl.move_y: 1.0  press? ... }` (bare-ident key or quoted; `-> <block>.<field>: <num>`; optional `press` keyword → `BindMode::Press` else `Hold`). Wire into top-level item dispatch beside `config_decl()`.
- [ ] **Step 3:** `controls.rs`: `pub fn controls_decl_to_json(d: &ControlsDecl) -> String` emitting `{"bindings":[{"key":"w","field":"ctl.move_y","value":1.0,"mode":"Hold"},…]}` (match Plan-0 serde). Emit `controls_descriptor()` accessor in `build_helper.rs` (default `{"bindings":[]}`).
- [ ] **Step 4 (test):** extend `engine_descriptors_emit.rs` — a probe `.sim` with a `controls{}` block emits a `controls_descriptor()` parseable by `engine_play_api::ControlsDescriptor::from_json`. Commit.

### Task 3: `render {}` block
**Files:** `dsl_ast/src/{ast,parser}.rs`; create `dsl_compiler/src/cg/emit/render.rs`; `build_helper.rs`.

- [ ] **Step 1:** AST mirroring the Plan-0 schema: `RenderDecl { arena_radius: f64, camera: CameraDecl, agents: Vec<AgentVisualDecl>, vfx: Vec<VfxDecl> }` (+ sub-structs). `camera: follow when <field> in [lo,hi] | observer`; `agent when <field> in [lo,hi] { color: (r,g,b) }`; `vfx on <RuleName> period <n> { ring radius <r> color (r,g,b) | beam_to_nearest when <field> in [lo,hi] color (r,g,b) }`.
- [ ] **Step 2:** Parser `fn render_decl()`. Wire into top-level dispatch.
- [ ] **Step 3:** `render.rs`: `render_decl_to_json` matching Plan-0 serde (externally-tagged: `{"Follow":{...}}`/`"Observer"`, `"Ring"`/`{"BeamToNearest":{...}}`). Emit `render_descriptor()` accessor.
- [ ] **Step 4 (test):** `engine_descriptors_emit.rs` — render block emits JSON parseable by `RenderDescriptor::from_json` with the expected agent/vfx counts. Commit.

### Task 4: `ui {}` block (absorbs playable-VS Plan 5)
**Files:** `dsl_ast/src/{ast,parser}.rs`; create `dsl_compiler/src/cg/emit/ui_model.rs`; `build_helper.rs`.

- [ ] **Step 1-4:** Implement exactly the playable-VS Plan 5 design (`docs/superpowers/plans/2026-05-25-dsl-ui-block-impl.md` Tasks 1-2): `ui {}` grammar (hud bars/text + menu/end screens), `ui_decl_to_json` matching `engine_ui::UiModel` serde (Plan C adds `from_json`), `ui_descriptor()` accessor. Compile-gate: the `ui` block emits a `ui_descriptor()` whose JSON has the declared widgets. Commit per the Plan 5 messages.

## Self-review note
The three blocks share `dsl_ast`/`dsl_compiler` files — done serially within this one plan (no inter-plan race). JSON shapes are the Plan-0 contract — validate each against `from_json` in the compile-gate before moving on. `make_playable` + `set_input` field names must match the generated `set_config_<block>_<field>` setters (playable-VS Plan 1 finding).
