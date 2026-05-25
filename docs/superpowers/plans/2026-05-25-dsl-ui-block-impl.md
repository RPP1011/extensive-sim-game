# DSL `ui {}` Block Implementation Plan (Plan 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `.sim` declare its game UI in a `ui {}` block that the compiler lowers to an `engine_ui::UiModel`, so the HUD/menu/death screens are rules-as-data rather than hand-written Rust in the viewer.

**Architecture:** Parse a `ui {}` block (AST in `dsl_ast`, mirroring the `config` block parser). The emitter serializes it to a **JSON descriptor string** exposed on the generated runtime as `pub fn ui_descriptor() -> &'static str`. `engine_ui` gains serde derives + `UiModel::from_json(&str)`. The viewer calls `UiModel::from_json(state.ui_descriptor())` instead of the hand-built `hud_model()`. The JSON-string seam keeps the headless `sims` crate free of any `engine_ui`/`egui` dependency. Bindings reference HUD value names (resolved by the viewer's `UiData`); card actions name `config.ctl` fields.

**Tech Stack:** custom DSL (`dsl_ast`, `dsl_compiler`), `engine_ui` (+ `serde`/`serde_json`), the viewer. **Depends on Plan 1 (parser/emit patterns), Plan 3 (the `.sim`), Plan 4 (the viewer consumer + the hand-built `UiModel` it replaces).** Must land LAST — shares `build_helper.rs`, `vampire_survivors.sim`, and `vs_viewer.rs` with earlier plans.

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `config_decl()`/`parse_config_field()` at `crates/dsl_ast/src/parser.rs:3398-3478` (grammar template)
  - `ConfigDecl`/`ConfigField` AST at `crates/dsl_ast/src/ast.rs:1127-1149`
  - runtime accessor emit at `crates/dsl_compiler/src/cg/emit/cross_cutting.rs:335` (`fold_view_xp_handles` shape)
  - `engine_ui::UiModel` (Plan 2), viewer `hud_model()` (Plan 4)
  - Search method: Explore + `Read`.
- **Decision:** extend the DSL with a `ui {}` block — UI as compiler-lowered data, *strengthening* P1 (UI leaves hand-written Rust). New emit target is a JSON string, not engine code, so `sims` gains no UI deps.
- **Rule-compiler touchpoints:**
  - DSL inputs edited: `crates/dsl_ast/src/{ast,parser}.rs`, `assets/sim/vampire_survivors.sim`
  - Generated outputs re-emitted: `sims::vampire_survivors` runtime (`ui_descriptor()`), via a new `cg/emit/ui_model.rs` + `build_helper.rs` hook
- **Hand-written downstream code:** NONE in the engine path; `engine_ui::from_json` is presentation-crate code.
- **Constitution check:**
  - P1: PASS — UI is now compiler-lowered data; no hand-written UI in the viewer for the declared model.
  - P2: N/A — `ui {}` touches no SoA/event layout; `ui_descriptor()` is an inert string.
  - P3/P4/P5/P6/P7/P11: N/A.
  - P10: PASS — descriptor parse is fallible (`from_json` returns `Result`); the viewer falls back to the hand-built model on parse error rather than panicking.
  - P8: PASS — this section.
- **Runtime gate:**
  - `ui_block_lowers_to_descriptor` at `crates/dsl_compiler/tests/ui_block_emit.rs` — compiling a `.sim` with a `ui {}` block emits a `ui_descriptor()` whose JSON contains the declared widgets.
  - `from_json_roundtrip` at `crates/engine_ui/src/data.rs` (test mod) — a serialized `UiModel` parses back equal.
- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] AIS reviewed post-design.

---

### Task 1: `ui {}` grammar (AST + parser)

**Files:**
- Modify: `crates/dsl_ast/src/ast.rs`, `crates/dsl_ast/src/parser.rs`
- Test: `crates/dsl_ast/tests/` (or inline parser test)

- [ ] **Step 1: AST types** (mirror `ConfigDecl`):
```rust
pub struct UiDecl { pub hud: Vec<UiWidget>, pub screens: Vec<UiScreen>, pub span: Span }
pub enum UiWidget {
    Bar { label: String, value: String, max: String, color: [u8;3] },
    Text { template: String },
}
pub struct UiCard { pub label: String, pub action_field: String } // action increments config.ctl.<field>
pub enum UiScreen {
    Menu { name: String, title: String, trigger: String, cards: Vec<UiCard> }, // trigger: "level_up"
    End  { name: String, title: String, trigger: String, summary: Vec<(String,String)>, restart_label: String },
}
```
Add `ui: Option<UiDecl>` to the top-level program/component AST (alongside `configs`).

- [ ] **Step 2: Parser.** Add `fn ui_decl()` recognizing:
```
ui {
  hud {
    bar "HP" value hp max hp_max color (220,40,40)
    bar "XP" value xp_into max xp_per_level color (40,160,240)
    text "Lv {level}  Kills {kills}  {time}s"
  }
  menu level_up "Level Up!" {
    card "Bolt Damage +" -> bolt_level
    card "Nova +" -> nova_level
  }
  screen dead "You Died" { summary time level kills  restart "Restart (R)" }
}
```
Wire `ui_decl()` into the top-level item dispatch next to `config_decl()`. Keep the grammar minimal — the bind-names are bare identifiers, colors are `(u8,u8,u8)`, actions are `-> <ident>`.

- [ ] **Step 3: Parse test.** Assert a fixture string parses into a `UiDecl` with the expected widget/screen counts. Run: `cargo test -p dsl_ast ui_`. Expected PASS.

- [ ] **Step 4: Commit.**
```bash
git add crates/dsl_ast/src/ast.rs crates/dsl_ast/src/parser.rs crates/dsl_ast/tests/
git commit -m "feat(dsl): parse ui {} block (hud widgets + menu/end screens)"
```

### Task 2: Emit the JSON descriptor on the runtime

**Files:**
- Create: `crates/dsl_compiler/src/cg/emit/ui_model.rs`
- Modify: `crates/dsl_compiler/src/build_helper.rs` (emit `ui_descriptor()` into the impl block), `crates/dsl_compiler/src/cg/emit/mod.rs` (register the module)
- Test: `crates/dsl_compiler/tests/ui_block_emit.rs`

- [ ] **Step 1: UiDecl → JSON string.** In `ui_model.rs`, `pub fn ui_decl_to_json(ui: &UiDecl) -> String` building JSON matching `engine_ui`'s serde shape (Task 3), e.g.:
```json
{"hud":[{"Bar":{"label":"HP","value":"hp","max":"hp_max","color":[220,40,40]}},
        {"Text":{"template":"Lv {level} Kills {kills} {time}s"}}],
 "screens":[{"name":"level_up","screen":{"Menu":{"title":"Level Up!",
        "cards":[{"label":"Bolt Damage +","action":{"Increment":"bolt_level"}}]}}}]}
```
(Hand-format JSON from the AST — no serde needed compiler-side. Escape strings.)

- [ ] **Step 2: Emit the accessor.** In `build_helper.rs`, where the impl block is generated, emit:
```rust
pub fn ui_descriptor(&self) -> &'static str { r####"<json>"#### }
```
using the string from Step 1 (empty `{"hud":[],"screens":[]}` if the `.sim` has no `ui {}`). Use `r####"..."####` to survive quotes in the JSON.

- [ ] **Step 3: Compile-gate test.** In `ui_block_emit.rs`, compile a `.sim` with a `ui {}` block (use `vampire_survivors.sim` after Task 4, or a small probe) and assert the emitted `runtime_core.rs` source contains `fn ui_descriptor` and the JSON has `"Bar"` + `"Menu"`. Run: `cargo test -p dsl_compiler --test ui_block_emit`. Expected PASS.

- [ ] **Step 4: Commit.**
```bash
git add crates/dsl_compiler/src/cg/emit/ui_model.rs crates/dsl_compiler/src/cg/emit/mod.rs crates/dsl_compiler/src/build_helper.rs crates/dsl_compiler/tests/ui_block_emit.rs
git commit -m "feat(dsl): emit ui {} block as ui_descriptor() JSON on generated runtime"
```

### Task 3: `engine_ui` descriptor parsing

**Files:**
- Modify: `crates/engine_ui/Cargo.toml`, `crates/engine_ui/src/model.rs`, `crates/engine_ui/src/data.rs`

- [ ] **Step 1: Add serde derives.** Add `serde = { version = "1", features = ["derive"] }` + `serde_json = "1"` to `engine_ui/Cargo.toml`. Derive `Serialize, Deserialize` on `Widget`, `UiAction`, `Card`, `Screen`, `NamedScreen`, `UiModel` (the enum reprs must match Task 1's JSON: externally-tagged, which is serde's default).

- [ ] **Step 2: `from_json`.**
```rust
impl UiModel {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}
```

- [ ] **Step 3: Round-trip test.**
```rust
#[test] fn from_json_roundtrip() {
    let m = UiModel { hud: vec![Widget::Text{template:"Lv {level}".into()}], screens: vec![] };
    let j = serde_json::to_string(&m).unwrap();
    let back = UiModel::from_json(&j).unwrap();
    assert_eq!(back.hud.len(), 1);
}
```
Run: `cargo test -p engine_ui`. Expected PASS. Commit.
```bash
git add crates/engine_ui/Cargo.toml crates/engine_ui/src/model.rs crates/engine_ui/src/data.rs
git commit -m "feat(engine_ui): serde-derive UiModel + from_json descriptor parsing"
```

### Task 4: Declare VS UI in the `.sim` + consume it in the viewer

**Files:**
- Modify: `assets/sim/vampire_survivors.sim` (add `ui {}`), `crates/viewer_runtime/src/bin/vs_viewer.rs` (consume `ui_descriptor()`)

- [ ] **Step 1: Add the `ui {}` block** to `vampire_survivors.sim` mirroring Plan 4's hand-built model (HP/XP bars, the `Lv/Kills/time/Enemies` text, the `level_up` menu over the full upgrade pool, the `dead` end screen). Build: `cargo build -p sims 2>&1 | tail -5`.

- [ ] **Step 2: Consume the descriptor in the viewer.** Replace `vs_ui::hud_model()` with:
```rust
let ui_model = engine_ui::UiModel::from_json(self.app.state.ui_descriptor())
    .unwrap_or_else(|e| { eprintln!("ui_descriptor parse failed ({e}); using fallback"); vs_ui::hud_model() });
```
Keep `vs_ui::hud_model()` as the fallback (P10 — no panic on parse failure). The menu's seeded card-draw (Plan 4 Task 3) still runs host-side; the DSL-declared menu defines the *pool/labels*, the host picks 3 — reconcile: either the DSL lists all options and the host samples 3, or the descriptor's menu cards ARE the 3 shown. Choose the former (DSL = pool, host samples) so the run stays varied; the descriptor's `Menu.cards` is the full pool.

- [ ] **Step 3: Verify identical rendering.** Run: `cargo run -p viewer_runtime --bin vs_viewer --release`. Expected: HUD/menu/death look identical to Plan 4 (now sourced from the `.sim`). `cargo test -p viewer_runtime -p engine_ui -p dsl_compiler` green.

- [ ] **Step 4: Commit.**
```bash
git add assets/sim/vampire_survivors.sim crates/viewer_runtime/src/bin/vs_viewer.rs
git commit -m "feat(vs): declare game UI in .sim ui {} block; viewer consumes descriptor"
```

## Self-review note
The JSON shape emitted compiler-side (Task 2) must exactly match `engine_ui`'s serde representation (Task 3) — externally-tagged enums (`{"Bar":{...}}`, `{"Increment":"bolt_level"}`). Validate with the round-trip + compile-gate tests before Task 4. If serde's default tagging differs from the hand-emitted JSON, fix the emitter to match serde (not vice-versa).
