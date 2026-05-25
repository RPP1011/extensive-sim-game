# engine_ui Descriptor Parse Implementation Plan (Plan C — Wave 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Give `engine_ui::UiModel` serde derives + `UiModel::from_json`, so the generic player (Plan B/D) can build a `UiModel` from the `ui_descriptor()` string the compiler emits (Plan A Task 4).

**Architecture:** Pure addition to the existing `engine_ui` crate (landed in the playable-VS work). Identical to the playable-VS Plan 5 Task 3. The serde enum representation (externally-tagged) is the contract Plan A's `ui_decl_to_json` must match.

**Tech Stack:** Rust, `serde`, `serde_json`. **Depends on Plan 0** only nominally (same serde-shape discipline); touches only the `engine_ui` crate → fully parallel with Plans A and B.

---

## Architectural Impact Statement
- **Existing primitives searched:** `engine_ui::{UiModel,Widget,Screen,NamedScreen,Card,UiAction}` (playable-VS Plan 2, `crates/engine_ui/src/model.rs`). Method: `Read`.
- **Decision:** extend `engine_ui` with serde — UI descriptor parsing for the DSL-declared UI path.
- **Rule-compiler touchpoints:** none.
- **Hand-written downstream code:** `engine_ui` edits — presentation crate, not rules.
- **Constitution check:** P1 PASS; P2–P7/P11 N/A; P10 PASS (`from_json` is `Result`); P8 PASS.
- **Runtime gate:** `ui_model_from_json_roundtrip` (test mod) — serialize→from_json equals original.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: serde derives + from_json
**Files:** Modify `crates/engine_ui/Cargo.toml`, `crates/engine_ui/src/model.rs`.

- [ ] **Step 1:** Add `serde = { version = "1", features = ["derive"] }` + `serde_json = "1"` to `crates/engine_ui/Cargo.toml`.
- [ ] **Step 2:** Derive `Serialize, Deserialize` on `Widget`, `UiAction`, `Card`, `Screen`, `NamedScreen`, `UiModel` in `model.rs` (default externally-tagged enums: `{"Bar":{...}}`, `{"Increment":"bolt_level"}`, `{"Menu":{...}}`).
- [ ] **Step 3:** Add to `impl UiModel`:
```rust
pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
```
- [ ] **Step 4 (test):** in `model.rs` `#[cfg(test)]`:
```rust
#[test] fn ui_model_from_json_roundtrip() {
    let m = UiModel {
        hud: vec![Widget::Bar{label:"HP".into(),value:"hp".into(),max:"hp_max".into(),color:[220,40,40]},
                  Widget::Text{template:"Lv {level}".into()}],
        screens: vec![NamedScreen{ name:"level_up".into(), screen: Screen::Menu{
            title:"Level Up".into(),
            cards: vec![Card{label:"Bolt +".into(), action: UiAction::Increment("bolt_level".into())}] }}],
    };
    let j = serde_json::to_string(&m).unwrap();
    let back = UiModel::from_json(&j).unwrap();
    assert_eq!(back.hud.len(), 2);
    assert_eq!(back.screens.len(), 1);
}
```
Run `cargo test -p engine_ui`. Expect PASS (the existing 3 tests still pass too). Commit:
```bash
git add crates/engine_ui/Cargo.toml crates/engine_ui/src/model.rs
git commit -m "feat(engine_ui): serde-derive UiModel + from_json for DSL-declared UI"
```

## Self-review note
The externally-tagged JSON shape here is the contract Plan A's `ui_decl_to_json` emits and Plan D consumes. If serde's default tagging and Plan A's hand-emitted JSON disagree, fix the emitter to match serde (validated by Plan A's compile-gate + Plan D's end-to-end).
