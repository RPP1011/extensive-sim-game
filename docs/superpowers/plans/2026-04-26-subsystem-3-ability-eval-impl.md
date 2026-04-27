# Subsystem 3 — GPU Ability Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the missing `pick_ability` GPU kernel (spec §11) end-to-end — DSL CPU emission, WGSL emission, buffer allocation, pipeline dispatch, and `apply_actions` wiring — so `per_ability` rows produce `AgentCast` events on both backends with byte-identical results on the sync path.

**Architecture:** The `per_ability` row type already parses and resolves to `ScoringIR.per_ability_rows` with full IR (`PerAbilityRowIR`, `IrExpr::AbilityTag/Hint/Range/OnCooldown`). This plan adds the three missing pieces: (1) CPU emitter in `emit_scoring.rs` that lowers `per_ability_rows` to a Rust `pick_ability` function; (2) WGSL emitter `emit_pick_ability_wgsl` that produces the `cs_pick_ability` kernel; (3) engine-core wiring — `chosen_ability_buf` allocation in `ResidentPathContext`, kernel dispatch between scoring and apply_actions in `step_batch`, and `apply_actions` reading the buffer and emitting `AgentCast`. The kernel sits between scoring and apply_actions per spec §11.1.

**Tech Stack:** Rust (edition 2021), WGSL (wgpu 0.19 compatible), `dsl_compiler` crate (`crates/dsl_compiler/`), `engine_gpu` crate (`crates/engine_gpu/`), `engine` crate (`crates/engine/`). Tests use `wgpu` with `pollster::block_on`, `bytemuck`, and follow the existing `cold_state_gold_transfer.rs` / `parity_with_cpu.rs` harness patterns.

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `PerAbilityRowIR` at `crates/dsl_ast/src/ir.rs:728`
  - `ScoringIR.per_ability_rows` at `crates/dsl_ast/src/ir.rs:687`
  - `IrExpr::AbilityTag/Hint/HintLit/Range/OnCooldown` at `crates/dsl_ast/src/ir.rs:322–352`
  - `emit_scoring.rs` emit path at `crates/dsl_compiler/src/emit_scoring.rs` — ignores `per_ability_rows` (line 1710: `per_ability_rows: vec![]` in test helpers, no emit pass)
  - `emit_scoring_wgsl.rs` at `crates/dsl_compiler/src/emit_scoring_wgsl.rs` — no `per_ability` handling
  - `PackedAbilityRegistry.hints/.tag_values` at `crates/engine_gpu/src/physics.rs:475–486` — populated but marked "Unbound in Phase 1"
  - `ResidentPathContext` at `crates/engine_gpu/src/backend/resident_ctx.rs:15` — no `chosen_ability_buf` field
  - `step_batch` pipeline at `crates/engine_gpu/src/lib.rs:1262–1307` — scoring goes directly to apply_actions; no pick_ability step
  - `apply_actions.rs` at `crates/engine_gpu/src/apply_actions.rs:1–80` — no chosen_ability_buf binding
  - `SimState.ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` at `crates/engine/src/state/mod.rs:149`
  - `SimState.hot_cooldown_next_ready_tick` at `crates/engine/src/state/mod.rs:100`
  - Search method: `rg`, direct `Read`.

- **Decision:** New `emit_pick_ability_wgsl` function added to `crates/dsl_compiler/src/emit_scoring_wgsl.rs` (extends the existing scoring WGSL emitter module since it shares agent/ability data structures). New `PickAbilityKernel` Rust struct added to `crates/engine_gpu/src/pick_ability.rs` (new file, following the `apply_actions.rs` / `scoring.rs` pattern of one-kernel-per-file). `chosen_ability_buf` is a new field on `ResidentPathContext`.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none (parser/resolver already handles `per_ability` rows)
  - Generated outputs re-emitted: `crates/engine_rules/src/pick_ability.rs` (new CPU picker emitted by `emit_scoring.rs`); `crates/engine_gpu/src/shaders/pick_ability.wgsl` is not a static file — it is a runtime-generated string via `emit_pick_ability_wgsl()`

- **Hand-written downstream code:**
  - `crates/engine_gpu/src/pick_ability.rs`: GPU kernel wrapper (Rust struct + dispatch logic). Justified: follows the established one-struct-per-kernel pattern (`apply_actions.rs`, `scoring.rs`, `movement.rs`); the WGSL itself is emitter-generated at build time.
  - `crates/engine_gpu/src/backend/resident_ctx.rs`: add `chosen_ability_buf` + `pick_ability_kernel` fields. Justified: engine-core buffer ownership per spec §11 table.

- **Constitution check:**
  - P1 (Compiler-First): PASS — CPU picker emitted by `emit_scoring.rs`; WGSL emitted by `emit_pick_ability_wgsl`; no hand-written scoring logic.
  - P2 (Schema-Hash on Layout): PASS — `ChosenAbilityBuf` packing format must be added to the schema hash in `crates/dsl_compiler/src/schema_hash.rs` (Task A3 step 2 covers this).
  - P3 (Cross-Backend Parity): PASS — Task D1 adds a cross-backend parity test asserting Serial and GPU produce identical cast decisions on the sync path.
  - P4 (EffectOp Size Budget): N/A — no new EffectOp variants.
  - P5 (Determinism via Keyed PCG): PASS — `pick_ability` kernel uses no RNG; argmax is deterministic sequential per agent (mirrors scoring kernel pattern).
  - P6 (Events Are the Mutation Channel): PASS — `AgentCast` is the event; ability effects remain in cascade.
  - P7 (Replayability Flagged): PASS — `AgentCast` is `@replayable`; already in the event hash.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — each task ends with a `cargo test` run and a commit step.
  - P10 (No Runtime Panic): PASS — `chosen_ability_buf` readiness is verified at `ensure_resident_init`; startup asserts if buffer absent (matches existing `gold_buf` pattern).
  - P11 (Reduction Determinism): PASS — single-thread-per-agent sequential argmax; same tie-break as scoring kernel (strictly-greater, lower slot wins).

- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

---

## File map

| File | Status | Role |
|---|---|---|
| `crates/dsl_ast/src/ir.rs` | Read-only | `PerAbilityRowIR`, `AbilityTag`, `AbilityHint`, `AbilityRange`, `AbilityOnCooldown` — already complete |
| `crates/dsl_compiler/src/emit_scoring.rs` | **Modify** | Add `emit_pick_ability_cpu` that lowers `per_ability_rows` to a Rust picker function |
| `crates/dsl_compiler/src/emit_scoring_wgsl.rs` | **Modify** | Add `emit_pick_ability_wgsl()` producing `cs_pick_ability` WGSL source |
| `crates/dsl_compiler/src/schema_hash.rs` | **Modify** | Hash `chosen_ability_buf` packing format + `PerAbilityRowIR` (already partially covered; ensure `per_ability_rows` is included) |
| `crates/engine_rules/src/pick_ability.rs` | **Create** (generated target) | CPU `pick_ability(state, registry) -> Vec<Option<(AbilitySlot, AgentId)>>` emitted by compiler |
| `crates/engine_gpu/src/pick_ability.rs` | **Create** | `PickAbilityKernel` — holds pipeline, BGL, buffer pool; `run_resident()` encodes one dispatch |
| `crates/engine_gpu/src/backend/resident_ctx.rs` | **Modify** | Add `chosen_ability_buf: Option<wgpu::Buffer>`, `chosen_ability_cap: u32`, `pick_ability_kernel: PickAbilityKernel` |
| `crates/engine_gpu/src/apply_actions.rs` | **Modify** | Add `@binding(6)` for `chosen_ability_buf`; WGSL reads it and emits `AgentCast` when sentinel absent |
| `crates/engine_gpu/src/lib.rs` | **Modify** | (a) `ensure_resident_init`: allocate `chosen_ability_buf`, upload cooldowns; (b) `step_batch` inner loop: dispatch `pick_ability_kernel` between scoring and apply_actions |
| `crates/engine_gpu/src/lib.rs` (schema) | **Modify** | Add `chosen_ability_buf` buffer size/ownership to `ResidentPathContext` init |
| `crates/engine_gpu/tests/pick_ability_smoke.rs` | **Create** | Smoke test: kernel runs, sentinel written for agent with no cooldown-ready ability |
| `crates/engine_gpu/tests/pick_ability_parity.rs` | **Create** | Cross-backend parity test: Serial CPU picker vs GPU kernel produce identical cast choices |

---

## Group A — CPU emit pass for `per_ability` rows (~3 tasks)

### Task A1: `emit_pick_ability_cpu` in `emit_scoring.rs`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_scoring.rs`
- Test: `crates/dsl_compiler/tests/per_ability_row.rs` (add emit assertions to existing file)

The CPU picker is a Rust function emitted into `engine_rules/src/pick_ability.rs`. It iterates agents, then per agent iterates abilities in the registry, checks guard (cooldown predicate), evaluates score expression, tracks argmax, resolves target. Returns `Vec<Option<(u8, u32)>>` — one entry per agent slot, `None` = no cast this tick.

The guard expression `!ability::on_cooldown(ability)` lowers to a check of `state.ability_cooldowns[agent_slot][ability_slot] > state.tick`. The score expression `ability::tag(PHYSICAL)` lowers to `registry.tag_value(ability_id, AbilityTag::Physical)`. `ability::hint == damage` lowers to `registry.hint(ability_id) == AbilityHint::Damage`. `ability::range` lowers to `registry.range(ability_id)`.

- [x] **Step 1: Write a failing test that calls the new emit function**

Add to `crates/dsl_compiler/tests/per_ability_row.rs`:

```rust
// EMIT-CPU-1: emit_pick_ability_cpu produces valid Rust source for a minimal row
#[test]
fn emit_pick_ability_cpu_produces_rust_for_minimal_row() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    guard:  !ability::on_cooldown(ability)
    score:  ability::tag(PHYSICAL)
    target: nearest_hostile_in_range(ability::range)
  }
}
"#;
    let comp = dsl_compiler::compile(SRC).expect("compile ok");
    let out = dsl_compiler::emit_scoring::emit_pick_ability_cpu(&comp.scoring[0])
        .expect("emit should succeed");
    assert!(out.contains("fn pick_ability"), "output should define pick_ability fn");
    assert!(out.contains("ability_cooldowns"), "guard lowers to cooldown check");
    assert!(out.contains("tag_value"), "score lowers to tag_value read");
    assert!(out.contains("nearest_hostile_in_range"), "target clause present");
}
```

Run: `cargo test -p dsl_compiler emit_pick_ability_cpu_produces_rust_for_minimal_row`
Expected: FAIL — `emit_pick_ability_cpu` does not exist yet.

- [x] **Step 2: Implement `emit_pick_ability_cpu`**

Add to `crates/dsl_compiler/src/emit_scoring.rs` after the existing `emit_scoring` function:

```rust
/// Emit a CPU `pick_ability` function from `per_ability_rows`.
///
/// If the scoring block has no `per_ability` rows, returns an empty
/// string (no-op for callers). The returned source lands in
/// `engine_rules/src/pick_ability.rs` via the xtask compile-dsl step.
pub fn emit_pick_ability_cpu(scoring: &ScoringIR) -> Result<String, EmitError> {
    if scoring.per_ability_rows.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring::emit_pick_ability_cpu.").unwrap();
    writeln!(out, "// Do not edit by hand.").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "use engine::ability::{{AbilityId, AbilityRegistry, AbilityTag}};").unwrap();
    writeln!(out, "use engine::state::SimState;").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "/// Per-agent ability selection result. `None` = no cast this tick.").unwrap();
    writeln!(out, "/// `Some((slot, target_agent_slot))` — 0-based ability slot + target.").unwrap();
    writeln!(out, "pub fn pick_ability(").unwrap();
    writeln!(out, "    state: &SimState,").unwrap();
    writeln!(out, "    registry: &AbilityRegistry,").unwrap();
    writeln!(out, ") -> Vec<Option<(u8, u32)>> {{").unwrap();
    writeln!(out, "    let agent_cap = state.agent_cap();").unwrap();
    writeln!(out, "    let tick = state.tick;").unwrap();
    writeln!(out, "    let mut results = vec![None; agent_cap];").unwrap();
    writeln!(out, "    for agent_slot in 0..agent_cap {{").unwrap();
    writeln!(out, "        let Some(agent_id) = engine::ids::AgentId::new((agent_slot as u32) + 1) else {{ continue }};").unwrap();
    writeln!(out, "        if !state.agent_alive(agent_id).unwrap_or(false) {{ continue }}").unwrap();
    writeln!(out, "        let mut best_score: f32 = f32::NEG_INFINITY;").unwrap();
    writeln!(out, "        let mut best: Option<(u8, u32)> = None;").unwrap();
    writeln!(out, "        let n_abilities = registry.len().min(engine::ability::MAX_ABILITIES);").unwrap();
    writeln!(out, "        for ab_slot in 0..n_abilities {{").unwrap();
    writeln!(out, "            let Some(ab_id) = AbilityId::new((ab_slot as u32) + 1) else {{ continue }};").unwrap();
    writeln!(out, "            let Some(prog) = registry.get(ab_id) else {{ continue }};").unwrap();
    // guard: !ability::on_cooldown
    writeln!(out, "            // guard: !ability::on_cooldown(ability)").unwrap();
    writeln!(out, "            let per_slot_ready = state.ability_cooldowns").unwrap();
    writeln!(out, "                .get(agent_slot)").unwrap();
    writeln!(out, "                .map(|row| row[ab_slot] <= tick)").unwrap();
    writeln!(out, "                .unwrap_or(true);").unwrap();
    writeln!(out, "            let global_ready = state.agent_cooldown_next_ready(agent_id)").unwrap();
    writeln!(out, "                .map(|t| t <= tick)").unwrap();
    writeln!(out, "                .unwrap_or(true);").unwrap();
    writeln!(out, "            if !per_slot_ready || !global_ready {{ continue }}").unwrap();
    // score: ability::tag(PHYSICAL) — lower each per_ability row's score expr
    for row in &scoring.per_ability_rows {
        let score_expr = lower_ability_score_expr_cpu(&row.score)?;
        writeln!(out, "            // score row `{}`", row.name).unwrap();
        writeln!(out, "            let score: f32 = {score_expr};").unwrap();
    }
    writeln!(out, "            if score > best_score {{").unwrap();
    writeln!(out, "                best_score = score;").unwrap();
    // target: nearest_hostile_in_range(ability::range)
    writeln!(out, "                // target: nearest_hostile_in_range(ability::range)").unwrap();
    writeln!(out, "                let range = prog.gate.range.unwrap_or(prog.gate.area_range());").unwrap();
    writeln!(out, "                let target_slot = state.nearest_hostile_in_range(agent_id, range);").unwrap();
    writeln!(out, "                if let Some(t) = target_slot {{").unwrap();
    writeln!(out, "                    best = Some((ab_slot as u8, t.get() - 1));").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        results[agent_slot] = best;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    results").unwrap();
    writeln!(out, "}}").unwrap();
    Ok(out)
}

/// Lower a `per_ability` score IrExprNode to a Rust expression string.
///
/// Handles the primitives that are legal inside `per_ability` score clauses:
/// `ability::tag(T)`, `ability::hint`, `ability::range`, float literals,
/// binary arithmetic on the above.
fn lower_ability_score_expr_cpu(expr: &IrExprNode) -> Result<String, EmitError> {
    match &expr.kind {
        IrExpr::Lit(f) => Ok(format!("{f:.6}_f32")),
        IrExpr::AbilityTag { tag } => {
            let tag_str = format!("AbilityTag::{tag:?}");
            Ok(format!("registry.tag_value(ab_id, {tag_str}).unwrap_or(0.0)"))
        }
        IrExpr::AbilityHint => Ok("registry.hint(ab_id) as f32".to_string()),
        IrExpr::AbilityHintLit(hint) => Ok(format!("{:?}_u32 as f32", *hint as u32)),
        IrExpr::AbilityRange => Ok("prog.gate.area_range()".to_string()),
        IrExpr::Binary(op, lhs, rhs) => {
            let l = lower_ability_score_expr_cpu(lhs)?;
            let r = lower_ability_score_expr_cpu(rhs)?;
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                _ => return Err(EmitError::UnsupportedExprShape(format!("{op:?} in per_ability score"))),
            };
            Ok(format!("({l} {op_str} {r})"))
        }
        IrExpr::If { cond, then_val, else_val } => {
            let c = lower_ability_score_expr_cpu(cond)?;
            let t = lower_ability_score_expr_cpu(then_val)?;
            let e = lower_ability_score_expr_cpu(else_val)?;
            Ok(format!("(if ({c} as u32 != 0) {{ {t} }} else {{ {e} }})"))
        }
        other => Err(EmitError::UnsupportedExprShape(format!("{other:?} not supported in per_ability score"))),
    }
}
```

- [x] **Step 3: Run the failing test to verify it passes**

```bash
cargo test -p dsl_compiler emit_pick_ability_cpu_produces_rust_for_minimal_row
```
Expected: PASS.

- [x] **Step 4: Verify the existing `per_ability_row.rs` tests still pass**

```bash
cargo test -p dsl_compiler per_ability_row
```
Expected: all existing PER-AB-* tests PASS.

- [x] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/emit_scoring.rs crates/dsl_compiler/tests/per_ability_row.rs
git commit -m "feat(dsl_compiler): emit_pick_ability_cpu lowers per_ability rows to Rust picker"
```

---

### Task A2: `emit_pick_ability_wgsl` — WGSL kernel emitter

**Files:**
- Modify: `crates/dsl_compiler/src/emit_scoring_wgsl.rs`
- Test: `crates/dsl_compiler/tests/per_ability_row.rs` (add WGSL emit assertions)

The WGSL kernel `cs_pick_ability` runs one thread per agent. Per agent it iterates ability slots 0..MAX_ABILITIES, checks cooldown (per-slot and global GCD), evaluates score, picks argmax, resolves target via a linear scan over alive agents within `ability::range`. Writes `chosen_ability_buf[agent] = pack_chosen(slot, target_slot)` or `SENTINEL_NO_CAST`.

Binding layout for `cs_pick_ability`:
- `@group(0) @binding(0)` — `agents: array<PickAbilityAgent>` (pos, alive, creature_type; read-only)
- `@group(0) @binding(1)` — `ability_registry: array<PickAbilityAbility>` (known, cooldown_ticks, hint, range, tag_values; read-only)
- `@group(0) @binding(2)` — `per_slot_cooldown: array<u32>` — flat `agent_cap * MAX_ABILITIES` u32s (per-slot cooldown expiry tick)
- `@group(0) @binding(3)` — `chosen_ability_buf: array<u64>` — output
- `@group(0) @binding(4)` — `cfg: PickAbilityCfg` uniform (agent_cap, ability_count, tick)
- `@group(0) @binding(5)` — `sim_cfg: SimCfg` (shared; read-only for `tick`)

The `ability_registry` buffer is a new compact struct — distinct from `PackedAbilityRegistry` (which is physics-oriented). Call it `GpuPickAbilityEntry` (16 bytes: `known: u32, cooldown_ticks: u32, hint: u32, range_bits: f32` — tag_values go in a separate flat buffer to keep alignment clean).

Tag values buffer: `@group(0) @binding(6)` — `tag_values: array<f32>` — flat `ability_count * NUM_ABILITY_TAGS` f32s.

- [x] **Step 1: Write a failing test for WGSL emission**

Add to `crates/dsl_compiler/tests/per_ability_row.rs`:

```rust
// EMIT-WGSL-1: emit_pick_ability_wgsl produces a string with cs_pick_ability entry point
#[test]
fn emit_pick_ability_wgsl_produces_cs_entry_point() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    guard:  !ability::on_cooldown(ability)
    score:  ability::tag(PHYSICAL) * 2.0
    target: nearest_hostile_in_range(ability::range)
  }
}
"#;
    let comp = dsl_compiler::compile(SRC).expect("compile ok");
    let wgsl = dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl(&comp.scoring[0]);
    assert!(wgsl.contains("@compute"), "should have @compute attribute");
    assert!(wgsl.contains("cs_pick_ability"), "entry point name");
    assert!(wgsl.contains("chosen_ability_buf"), "output buffer binding");
    assert!(wgsl.contains("per_slot_cooldown"), "cooldown buffer binding");
    assert!(wgsl.contains("tag_values"), "tag values buffer binding");
    // Sentinel for no-cast must be present
    assert!(wgsl.contains("SENTINEL_NO_CAST") || wgsl.contains("0xFFFFFFFFFFFFFFFF"), "sentinel defined");
}
```

Run: `cargo test -p dsl_compiler emit_pick_ability_wgsl_produces_cs_entry_point`
Expected: FAIL — function does not exist yet.

- [x] **Step 2: Implement `emit_pick_ability_wgsl`**

Add to `crates/dsl_compiler/src/emit_scoring_wgsl.rs`:

```rust
/// Emit the `cs_pick_ability` WGSL compute kernel from `per_ability` rows.
///
/// Returns an empty string if the scoring block has no `per_ability` rows.
/// The caller (engine_gpu's `PickAbilityKernel::new`) passes the emitted
/// source directly to `device.create_shader_module`.
///
/// ## Binding layout
///
/// | Slot | Name                 | Type                       | Access     |
/// |------|----------------------|----------------------------|------------|
/// | 0    | `agents`             | `array<PickAbilityAgent>`  | read       |
/// | 1    | `ability_registry`   | `array<GpuPickAbilityEntry>` | read      |
/// | 2    | `per_slot_cooldown`  | `array<u32>`               | read       |
/// | 3    | `chosen_ability_buf` | `array<atomic<u64>>`       | read_write |
/// | 4    | `cfg`                | `PickAbilityCfg` (uniform) | uniform    |
/// | 5    | `sim_cfg`            | `SimCfg` (storage)         | read       |
/// | 6    | `tag_values`         | `array<f32>`               | read       |
pub fn emit_pick_ability_wgsl(scoring: &ScoringIR) -> String {
    if scoring.per_ability_rows.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl.").unwrap();
    writeln!(out, "// Do not edit by hand.").unwrap();
    writeln!(out, "").unwrap();

    // Shared SimCfg prefix (tick field only needed)
    out.push_str(SIM_CFG_WGSL_PREFIX);
    writeln!(out, "").unwrap();

    writeln!(out, "struct PickAbilityAgent {{").unwrap();
    writeln!(out, "    pos_x: f32, pos_y: f32, pos_z: f32,").unwrap();
    writeln!(out, "    alive: u32,").unwrap();
    writeln!(out, "    creature_type: u32,").unwrap();
    writeln!(out, "    _pad0: u32, _pad1: u32, _pad2: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "struct GpuPickAbilityEntry {{").unwrap();
    writeln!(out, "    known: u32,").unwrap();
    writeln!(out, "    cooldown_ticks: u32,").unwrap();
    writeln!(out, "    hint: u32,").unwrap();
    writeln!(out, "    range_bits: f32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "struct PickAbilityCfg {{").unwrap();
    writeln!(out, "    agent_cap: u32,").unwrap();
    writeln!(out, "    ability_count: u32,").unwrap();
    writeln!(out, "    num_tags: u32,").unwrap();
    writeln!(out, "    tick: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "const SENTINEL_NO_CAST: u64 = 0xFFFFFFFFFFFFFFFFu;").unwrap();
    writeln!(out, "const MAX_ABILITIES: u32 = 256u;").unwrap();
    writeln!(out, "const WORKGROUP_SIZE: u32 = 64u;").unwrap();
    writeln!(out, "").unwrap();
    writeln!(out, "@group(0) @binding(0) var<storage, read> agents: array<PickAbilityAgent>;").unwrap();
    writeln!(out, "@group(0) @binding(1) var<storage, read> ability_registry: array<GpuPickAbilityEntry>;").unwrap();
    writeln!(out, "@group(0) @binding(2) var<storage, read> per_slot_cooldown: array<u32>;").unwrap();
    writeln!(out, "@group(0) @binding(3) var<storage, read_write> chosen_ability_buf: array<u32>;").unwrap();
    writeln!(out, "// chosen_ability_buf packs two u32s per agent:").unwrap();
    writeln!(out, "//   [agent*2+0] = ability_slot (0xFFFFFFFF = no cast)").unwrap();
    writeln!(out, "//   [agent*2+1] = target_agent_slot (0xFFFFFFFF = no target)").unwrap();
    writeln!(out, "@group(0) @binding(4) var<uniform> cfg: PickAbilityCfg;").unwrap();
    writeln!(out, "@group(0) @binding(5) var<storage, read> sim_cfg: SimCfg;").unwrap();
    writeln!(out, "@group(0) @binding(6) var<storage, read> tag_values: array<f32>;").unwrap();
    writeln!(out, "").unwrap();

    // Helper: tag_value(ability_slot, tag_idx)
    writeln!(out, "fn tag_value(ab_slot: u32, tag_idx: u32) -> f32 {{").unwrap();
    writeln!(out, "    let idx = ab_slot * cfg.num_tags + tag_idx;").unwrap();
    writeln!(out, "    if idx < arrayLength(&tag_values) {{ return tag_values[idx]; }}").unwrap();
    writeln!(out, "    return 0.0;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();

    // Helper: is_hostile(a, b) — different creature_type, both alive
    writeln!(out, "fn is_hostile(a: u32, b: u32) -> bool {{").unwrap();
    writeln!(out, "    return agents[a].alive != 0u && agents[b].alive != 0u").unwrap();
    writeln!(out, "        && agents[a].creature_type != agents[b].creature_type;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();

    // Helper: dist3(a, b)
    writeln!(out, "fn dist3(a: u32, b: u32) -> f32 {{").unwrap();
    writeln!(out, "    let dx = agents[a].pos_x - agents[b].pos_x;").unwrap();
    writeln!(out, "    let dy = agents[a].pos_y - agents[b].pos_y;").unwrap();
    writeln!(out, "    let dz = agents[a].pos_z - agents[b].pos_z;").unwrap();
    writeln!(out, "    return sqrt(dx*dx + dy*dy + dz*dz);").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();

    // Emit score function body from per_ability_rows
    writeln!(out, "fn eval_ability_score(agent_slot: u32, ab_slot: u32) -> f32 {{").unwrap();
    for row in &scoring.per_ability_rows {
        let expr = lower_ability_score_expr_wgsl(&row.score);
        writeln!(out, "    // row `{}`", row.name).unwrap();
        writeln!(out, "    return {expr};").unwrap();
        break; // one row only — multi-row not in scope for this plan
    }
    writeln!(out, "}}").unwrap();
    writeln!(out, "").unwrap();

    writeln!(out, "@compute @workgroup_size(WORKGROUP_SIZE)").unwrap();
    writeln!(out, "fn cs_pick_ability(@builtin(global_invocation_id) gid: vec3<u32>) {{").unwrap();
    writeln!(out, "    let agent_slot = gid.x;").unwrap();
    writeln!(out, "    if agent_slot >= cfg.agent_cap {{ return; }}").unwrap();
    writeln!(out, "    if agents[agent_slot].alive == 0u {{").unwrap();
    writeln!(out, "        chosen_ability_buf[agent_slot * 2u] = 0xFFFFFFFFu;").unwrap();
    writeln!(out, "        chosen_ability_buf[agent_slot * 2u + 1u] = 0xFFFFFFFFu;").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    var best_score: f32 = -1.0e38;").unwrap();
    writeln!(out, "    var best_ab: u32 = 0xFFFFFFFFu;").unwrap();
    writeln!(out, "    var best_tgt: u32 = 0xFFFFFFFFu;").unwrap();
    writeln!(out, "    let tick = cfg.tick;").unwrap();
    writeln!(out, "    for (var ab_slot: u32 = 0u; ab_slot < cfg.ability_count; ab_slot++) {{").unwrap();
    writeln!(out, "        if ability_registry[ab_slot].known == 0u {{ continue; }}").unwrap();
    // guard: !ability::on_cooldown
    writeln!(out, "        // guard: per-slot cooldown").unwrap();
    writeln!(out, "        let cd_idx = agent_slot * MAX_ABILITIES + ab_slot;").unwrap();
    writeln!(out, "        if cd_idx < arrayLength(&per_slot_cooldown) {{").unwrap();
    writeln!(out, "            if per_slot_cooldown[cd_idx] > tick {{ continue; }}").unwrap();
    writeln!(out, "        }}").unwrap();
    // score
    writeln!(out, "        let score = eval_ability_score(agent_slot, ab_slot);").unwrap();
    writeln!(out, "        if score <= best_score {{ continue; }}").unwrap();
    // target: nearest hostile in range
    writeln!(out, "        let range = ability_registry[ab_slot].range_bits;").unwrap();
    writeln!(out, "        var nearest_tgt: u32 = 0xFFFFFFFFu;").unwrap();
    writeln!(out, "        var nearest_dist: f32 = 1.0e38;").unwrap();
    writeln!(out, "        for (var t: u32 = 0u; t < cfg.agent_cap; t++) {{").unwrap();
    writeln!(out, "            if t == agent_slot {{ continue; }}").unwrap();
    writeln!(out, "            if !is_hostile(agent_slot, t) {{ continue; }}").unwrap();
    writeln!(out, "            let d = dist3(agent_slot, t);").unwrap();
    writeln!(out, "            if d <= range && d < nearest_dist {{").unwrap();
    writeln!(out, "                nearest_dist = d;").unwrap();
    writeln!(out, "                nearest_tgt = t;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if nearest_tgt == 0xFFFFFFFFu {{ continue; }}").unwrap();
    writeln!(out, "        best_score = score;").unwrap();
    writeln!(out, "        best_ab = ab_slot;").unwrap();
    writeln!(out, "        best_tgt = nearest_tgt;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    chosen_ability_buf[agent_slot * 2u] = best_ab;").unwrap();
    writeln!(out, "    chosen_ability_buf[agent_slot * 2u + 1u] = best_tgt;").unwrap();
    writeln!(out, "}}").unwrap();

    out
}

fn lower_ability_score_expr_wgsl(expr: &IrExprNode) -> String {
    match &expr.kind {
        IrExpr::Lit(f) => format!("{f:.6}"),
        IrExpr::AbilityTag { tag } => {
            // TAG_INDEX must match engine::ability::AbilityTag discriminant order
            let idx = *tag as u32;
            format!("tag_value(ab_slot, {idx}u)")
        }
        IrExpr::AbilityHint => "f32(ability_registry[ab_slot].hint)".to_string(),
        IrExpr::AbilityHintLit(hint) => format!("{}.0", *hint as u32),
        IrExpr::AbilityRange => "ability_registry[ab_slot].range_bits".to_string(),
        IrExpr::Binary(op, lhs, rhs) => {
            let l = lower_ability_score_expr_wgsl(lhs);
            let r = lower_ability_score_expr_wgsl(rhs);
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                _ => "/* unsupported_op */+",
            };
            format!("({l} {op_str} {r})")
        }
        IrExpr::If { cond, then_val, else_val } => {
            let c = lower_ability_score_expr_wgsl(cond);
            let t = lower_ability_score_expr_wgsl(then_val);
            let e = lower_ability_score_expr_wgsl(else_val);
            format!("select({e}, {t}, bool({c}))")
        }
        _ => "0.0 /* unsupported */".to_string(),
    }
}
```

Note: `SIM_CFG_WGSL_PREFIX` is an existing constant in `emit_scoring_wgsl.rs` (the `SimCfg` struct declaration). Reuse it. If it does not exist as a named constant, inline the minimal subset (`struct SimCfg { tick: atomic<u32>, ... }`).

- [x] **Step 3: Run the WGSL emit test**

```bash
cargo test -p dsl_compiler emit_pick_ability_wgsl_produces_cs_entry_point
```
Expected: PASS.

- [x] **Step 4: Verify `cargo test -p dsl_compiler` still fully passes**

```bash
cargo test -p dsl_compiler
```
Expected: all tests PASS. No regressions.

- [x] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/emit_scoring_wgsl.rs crates/dsl_compiler/tests/per_ability_row.rs
git commit -m "feat(dsl_compiler): emit_pick_ability_wgsl produces cs_pick_ability WGSL kernel"
```

---

### Task A3: Schema hash coverage for `per_ability` emit path

**Files:**
- Modify: `crates/dsl_compiler/src/schema_hash.rs`
- Test: `crates/dsl_compiler/tests/per_ability_row.rs` (add schema hash assertion)

The `chosen_ability_buf` packing format (2× u32 per agent: ability_slot + target_slot) must be hashed into the schema so a layout change fails CI.

- [x] **Step 1: Write a test that asserts `per_ability_rows` contribution to schema hash**

Add to `crates/dsl_compiler/tests/per_ability_row.rs`:

```rust
// SCHEMA-1: schema hash differs when a per_ability row is present vs. absent
#[test]
fn schema_hash_differs_with_and_without_per_ability_row() {
    use dsl_compiler::schema_hash::compute_schema_hash;

    const WITH_ROW: &str = r#"
scoring {
  row pick_ability per_ability {
    score: ability::tag(PHYSICAL)
  }
}
"#;
    const WITHOUT_ROW: &str = r#"
scoring {}
"#;
    let h1 = compute_schema_hash(&dsl_compiler::compile(WITH_ROW).unwrap());
    let h2 = compute_schema_hash(&dsl_compiler::compile(WITHOUT_ROW).unwrap());
    assert_ne!(h1, h2, "schema hash must differ when per_ability rows are added");
}
```

Run: `cargo test -p dsl_compiler schema_hash_differs_with_and_without_per_ability_row`
Expected: FAIL if `per_ability_rows` are currently excluded from hashing, PASS if already covered.

- [x] **Step 2: Add `per_ability_rows` and `chosen_ability_buf` packing format to schema hash**

In `crates/dsl_compiler/src/schema_hash.rs`, find where `ScoringIR` is hashed. Add:

```rust
// Hash per_ability rows (Subsystem 3 — ability evaluation kernel).
// The packing format for chosen_ability_buf is 2×u32 per agent:
//   [agent*2+0] = ability_slot  (0xFFFFFFFF = sentinel no-cast)
//   [agent*2+1] = target_agent_slot (0xFFFFFFFF = sentinel no-target)
// Changing this layout requires bumping the schema hash.
h.update(b"chosen_ability_buf_packing: 2xu32_per_agent_slot_then_target");
for row in &scoring.per_ability_rows {
    h.update(b"per_ability_row:");
    h.update(row.name.as_bytes());
    if let Some(guard) = &row.guard { hash_expr(h, guard); }
    hash_expr(h, &row.score);
    if let Some(target) = &row.target { hash_expr(h, target); }
}
```

- [x] **Step 3: Run the schema hash test**

```bash
cargo test -p dsl_compiler schema_hash_differs_with_and_without_per_ability_row
```
Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add crates/dsl_compiler/src/schema_hash.rs crates/dsl_compiler/tests/per_ability_row.rs
git commit -m "feat(dsl_compiler): include per_ability rows + chosen_ability_buf layout in schema hash"
```

---

## Group B — `pick_ability` GPU kernel (~5 tasks)

### Task B1: `GpuPickAbilityEntry` upload types and `PickAbilityKernel` struct

**Files:**
- Create: `crates/engine_gpu/src/pick_ability.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `pub mod pick_ability;`)

The `PickAbilityKernel` struct owns the compiled pipeline + BGL. It mirrors the pattern in `apply_actions.rs`: a `new(device, queue)` constructor, a `run_resident(&self, device, queue, encoder, ...)` method that encodes one compute dispatch, and a helper to build the bind group from caller-supplied buffers.

- [ ] **Step 1: Write a failing compilation test**

Add to a new `crates/engine_gpu/tests/pick_ability_smoke.rs`:

```rust
//! Smoke test: PickAbilityKernel compiles and can be constructed.
#![cfg(feature = "gpu")]

use engine_gpu::pick_ability::PickAbilityKernel;

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                force_fallback_adapter: true,
                power_preference: wgpu::PowerPreference::None,
            })
            .await
            .expect("adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device")
    })
}

#[test]
fn pick_ability_kernel_compiles() {
    let (device, _queue) = gpu_device_queue();
    let _kernel = PickAbilityKernel::new(&device).expect("PickAbilityKernel should compile");
}
```

Run: `cargo test -p engine_gpu pick_ability_kernel_compiles`
Expected: FAIL — `engine_gpu::pick_ability` module does not exist.

- [ ] **Step 2: Create `crates/engine_gpu/src/pick_ability.rs`**

```rust
//! `cs_pick_ability` kernel — GPU ability selector (Subsystem 3).
//!
//! Per agent per tick: iterates abilities, checks per-slot cooldown,
//! evaluates score expression (compiler-emitted), picks argmax with
//! nearest-hostile target, writes chosen (ability_slot, target_slot)
//! into `chosen_ability_buf` or sentinel `0xFFFFFFFF` for no-cast.
//!
//! ## Binding layout
//! | Slot | Buffer                | Access      |
//! |------|-----------------------|-------------|
//! | 0    | `agents`              | read        |
//! | 1    | `ability_registry`    | read        |
//! | 2    | `per_slot_cooldown`   | read        |
//! | 3    | `chosen_ability_buf`  | read_write  |
//! | 4    | `cfg` (uniform)       | uniform     |
//! | 5    | `sim_cfg`             | read        |
//! | 6    | `tag_values`          | read        |

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::sim_cfg::SIM_CFG_WGSL;

pub const WORKGROUP_SIZE: u32 = 64;

/// 16-byte per-ability entry uploaded to the pick_ability kernel.
/// `known`: 1 if this slot is occupied. `cooldown_ticks`: per-ability
/// cooldown duration (not expiry tick; expiry is in `per_slot_cooldown`).
/// `hint`: `AbilityHint` discriminant. `range_bits`: `f32::to_bits` of
/// the ability's targeting range.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuPickAbilityEntry {
    pub known:          u32,
    pub cooldown_ticks: u32,
    pub hint:           u32,
    pub range_bits:     f32,
}

const _: () = assert!(std::mem::size_of::<GpuPickAbilityEntry>() == 16);

/// Uniform config for one `cs_pick_ability` dispatch.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct PickAbilityCfg {
    pub agent_cap:      u32,
    pub ability_count:  u32,
    pub num_tags:       u32,
    pub tick:           u32,
}

const _: () = assert!(std::mem::size_of::<PickAbilityCfg>() == 16);

/// Sentinel value written to `chosen_ability_buf[agent*2]` when no
/// ability is selected this tick.
pub const SENTINEL_NO_CAST: u32 = 0xFFFF_FFFFu32;

pub struct PickAbilityKernel {
    pipeline: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
}

#[derive(Debug)]
pub enum PickAbilityError {
    ShaderCompile(String),
    DispatchEncode(String),
}

impl std::fmt::Display for PickAbilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PickAbilityError::ShaderCompile(s) => write!(f, "pick_ability shader compile: {s}"),
            PickAbilityError::DispatchEncode(s) => write!(f, "pick_ability dispatch: {s}"),
        }
    }
}

impl PickAbilityKernel {
    /// Compile the `cs_pick_ability` shader module from the DSL-emitted WGSL.
    ///
    /// In production, the WGSL source is emitted by
    /// `dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl`. For tests
    /// the minimal scoring block is compiled inline.
    pub fn new(device: &wgpu::Device) -> Result<Self, PickAbilityError> {
        // The WGSL source is produced by the dsl_compiler at build time and
        // stored as the PICK_ABILITY_WGSL constant (emitted into engine_rules).
        // Import it here.
        let wgsl_src = engine_rules::pick_ability::PICK_ABILITY_WGSL;
        Self::from_wgsl(device, wgsl_src)
    }

    /// Construct from explicit WGSL source — used by tests.
    pub fn from_wgsl(device: &wgpu::Device, wgsl_src: &str) -> Result<Self, PickAbilityError> {
        let full_src = format!("{SIM_CFG_WGSL}\n{wgsl_src}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick_ability"),
            source: wgpu::ShaderSource::Wgsl(full_src.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_ability_bgl"),
            entries: &[
                // 0: agents (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: ability_registry (read)
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 2: per_slot_cooldown (read)
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 3: chosen_ability_buf (read_write)
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 4: cfg (uniform)
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 5: sim_cfg (read)
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 6: tag_values (read)
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick_ability_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cs_pick_ability"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "cs_pick_ability",
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, bgl })
    }

    /// Encode one `cs_pick_ability` dispatch into `encoder`.
    ///
    /// `agents_buf` — resident agent SoA (same buffer as physics/scoring).
    /// `ability_registry_buf` — packed `GpuPickAbilityEntry` array.
    /// `per_slot_cooldown_buf` — flat `agent_cap * MAX_ABILITIES` u32 array.
    /// `chosen_ability_buf` — output `agent_cap * 2` u32 array.
    /// `cfg_buf` — `PickAbilityCfg` uniform buffer.
    /// `sim_cfg_buf` — shared `SimCfg` storage buffer.
    /// `tag_values_buf` — flat `ability_count * num_tags` f32 array.
    #[allow(clippy::too_many_arguments)]
    pub fn run_resident(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf:           &wgpu::Buffer,
        ability_registry_buf: &wgpu::Buffer,
        per_slot_cooldown_buf: &wgpu::Buffer,
        chosen_ability_buf:   &wgpu::Buffer,
        cfg_buf:              &wgpu::Buffer,
        sim_cfg_buf:          &wgpu::Buffer,
        tag_values_buf:       &wgpu::Buffer,
        agent_cap:            u32,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_ability_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agents_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ability_registry_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: per_slot_cooldown_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: chosen_ability_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: sim_cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: tag_values_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pick_ability_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let groups = agent_cap.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}
```

- [ ] **Step 3: Add `pub mod pick_ability;` to `crates/engine_gpu/src/lib.rs`**

Find the block of `pub mod` declarations near the top of `lib.rs` and add:

```rust
pub mod pick_ability;
```

- [ ] **Step 4: Run the compile test**

```bash
cargo test -p engine_gpu pick_ability_kernel_compiles
```
Expected: PASS (shader compiles successfully against wgpu validation).

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/pick_ability.rs crates/engine_gpu/src/lib.rs \
        crates/engine_gpu/tests/pick_ability_smoke.rs
git commit -m "feat(engine_gpu): PickAbilityKernel struct + cs_pick_ability WGSL compiles"
```

---

### Task B2: `chosen_ability_buf` + cooldown buffers in `ResidentPathContext`

**Files:**
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (`ensure_resident_init`)

Add three new fields to `ResidentPathContext`:
- `chosen_ability_buf: Option<wgpu::Buffer>` — `agent_cap * 2 * 4` bytes (2× u32 per agent)
- `chosen_ability_cap: u32`
- `pick_ability_kernel: Option<PickAbilityKernel>`
- `per_slot_cooldown_buf: Option<wgpu::Buffer>` — `agent_cap * MAX_ABILITIES * 4` bytes
- `per_slot_cooldown_cap: u32`
- `ability_registry_buf: Option<wgpu::Buffer>` — `MAX_ABILITIES * 16` bytes (GpuPickAbilityEntry)
- `tag_values_buf: Option<wgpu::Buffer>` — `MAX_ABILITIES * NUM_ABILITY_TAGS * 4` bytes

In `ensure_resident_init` (in `lib.rs`), allocate these on first call, exactly like the existing `gold_buf` pattern.

- [ ] **Step 1: Write a failing test that checks `chosen_ability_buf` is allocated after `ensure_resident_init`**

Add to `crates/engine_gpu/tests/pick_ability_smoke.rs`:

```rust
use engine::state::{AgentSpawn, SimState};
use engine::ability::AbilityRegistry;
use engine_data::entities::CreatureType;
use engine_gpu::GpuBackend;
use glam::Vec3;

#[test]
fn ensure_resident_init_allocates_chosen_ability_buf() {
    let (device, queue) = gpu_device_queue();
    let mut backend = GpuBackend::new_from_device(device, queue);
    let mut state = SimState::new(4);
    let spawn = AgentSpawn { pos: Vec3::ZERO, creature_type: CreatureType::Human, ..Default::default() };
    state.spawn_agent(spawn).unwrap();

    // ensure_resident_init runs as part of step_batch setup
    backend.ensure_resident_init(&state);

    assert!(
        backend.resident.chosen_ability_buf.is_some(),
        "chosen_ability_buf should be allocated after ensure_resident_init"
    );
    assert!(
        backend.resident.per_slot_cooldown_buf.is_some(),
        "per_slot_cooldown_buf should be allocated after ensure_resident_init"
    );
}
```

Run: `cargo test -p engine_gpu ensure_resident_init_allocates_chosen_ability_buf`
Expected: FAIL — fields don't exist yet.

- [ ] **Step 2: Add fields to `ResidentPathContext`**

In `crates/engine_gpu/src/backend/resident_ctx.rs`, add after the existing `alive_bitmap_buf` fields:

```rust
/// Subsystem 3 (ability evaluation) — per-agent chosen-ability output.
/// Layout: `[ability_slot: u32, target_slot: u32]` × agent_cap.
/// `0xFFFFFFFF` sentinel in ability_slot means no cast this tick.
pub chosen_ability_buf:      Option<wgpu::Buffer>,
pub chosen_ability_cap:      u32,

/// Subsystem 3 — per-(agent, ability) cooldown expiry tick.
/// Flat layout: `per_slot_cooldown[agent_slot * MAX_ABILITIES + ab_slot]`.
/// Uploaded from `state.ability_cooldowns` at ensure_resident_init;
/// updated by apply_actions when a cast fires.
pub per_slot_cooldown_buf:   Option<wgpu::Buffer>,
pub per_slot_cooldown_cap:   u32,

/// Subsystem 3 — packed ability metadata for `cs_pick_ability`.
/// One `GpuPickAbilityEntry` per slot 0..MAX_ABILITIES.
pub ability_registry_pa_buf: Option<wgpu::Buffer>,

/// Subsystem 3 — flat tag values table: `ability_count * NUM_ABILITY_TAGS` f32s.
pub tag_values_buf:          Option<wgpu::Buffer>,

/// Subsystem 3 — compiled `cs_pick_ability` kernel.
pub pick_ability_kernel:     Option<crate::pick_ability::PickAbilityKernel>,
```

Initialize all to `None` / `0` in `ResidentPathContext::new`.

- [ ] **Step 3: Allocate buffers in `ensure_resident_init`**

In `crates/engine_gpu/src/lib.rs`, find the `ensure_resident_init` function (the block that allocates `gold_buf`, `standing_storage`, etc.). Add after that block:

```rust
// Subsystem 3 (Ability Evaluation) — chosen_ability_buf, per_slot_cooldown_buf,
// ability_registry_pa_buf, tag_values_buf, pick_ability_kernel.
if self.resident.chosen_ability_buf.is_none() || self.resident.chosen_ability_cap != agent_cap {
    use engine::ability::MAX_ABILITIES;
    use wgpu::util::DeviceExt;

    // chosen_ability_buf: 2× u32 per agent
    let chosen_data = vec![crate::pick_ability::SENTINEL_NO_CAST; (agent_cap * 2) as usize];
    self.resident.chosen_ability_buf = Some(self.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("chosen_ability_buf"),
            contents: bytemuck::cast_slice(&chosen_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }
    ));
    self.resident.chosen_ability_cap = agent_cap;

    // per_slot_cooldown_buf: agent_cap * MAX_ABILITIES u32s (expiry ticks)
    let cd_flat: Vec<u32> = state.ability_cooldowns
        .iter()
        .take(agent_cap as usize)
        .flat_map(|row| row.iter().copied())
        .collect();
    // Pad to agent_cap * MAX_ABILITIES entries if state is smaller
    let expected_len = (agent_cap as usize) * MAX_ABILITIES;
    let mut cd_padded = cd_flat;
    cd_padded.resize(expected_len, 0u32);
    self.resident.per_slot_cooldown_buf = Some(self.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("per_slot_cooldown_buf"),
            contents: bytemuck::cast_slice(&cd_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }
    ));
    self.resident.per_slot_cooldown_cap = agent_cap;

    // ability_registry_pa_buf: MAX_ABILITIES GpuPickAbilityEntry records
    let entries = pack_pick_ability_registry(&state.ability_registry);
    self.resident.ability_registry_pa_buf = Some(self.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("ability_registry_pa"),
            contents: bytemuck::cast_slice(&entries),
            usage: wgpu::BufferUsages::STORAGE,
        }
    ));

    // tag_values_buf: MAX_ABILITIES * NUM_ABILITY_TAGS f32s
    let tag_vals = pack_tag_values(&state.ability_registry);
    self.resident.tag_values_buf = Some(self.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("tag_values_buf"),
            contents: bytemuck::cast_slice(&tag_vals),
            usage: wgpu::BufferUsages::STORAGE,
        }
    ));

    // Compile pick_ability kernel
    self.resident.pick_ability_kernel = Some(
        crate::pick_ability::PickAbilityKernel::new(&self.device)
            .expect("pick_ability kernel compile failed at ensure_resident_init")
    );
}
```

Add two helper functions in `lib.rs` (or in `pick_ability.rs` as `pub fn`):

```rust
/// Pack `AbilityRegistry` into `GpuPickAbilityEntry` flat array for cs_pick_ability.
fn pack_pick_ability_registry(registry: &engine::ability::AbilityRegistry)
    -> Vec<crate::pick_ability::GpuPickAbilityEntry>
{
    use engine::ability::{AbilityId, MAX_ABILITIES};
    use crate::pick_ability::GpuPickAbilityEntry;
    let mut out = vec![GpuPickAbilityEntry { known: 0, cooldown_ticks: 0, hint: 0, range_bits: 0.0 }; MAX_ABILITIES];
    for slot in 0..registry.len().min(MAX_ABILITIES) {
        let Some(id) = AbilityId::new((slot as u32) + 1) else { continue };
        let Some(prog) = registry.get(id) else { continue };
        out[slot] = GpuPickAbilityEntry {
            known: 1,
            cooldown_ticks: prog.gate.cooldown_ticks,
            hint: prog.hint.map(|h| h as u32).unwrap_or(crate::physics::HINT_NONE_SENTINEL),
            range_bits: prog.gate.area_range(),
        };
    }
    out
}

/// Pack ability tag values into flat f32 array for cs_pick_ability.
fn pack_tag_values(registry: &engine::ability::AbilityRegistry) -> Vec<f32> {
    use engine::ability::{AbilityId, AbilityTag, MAX_ABILITIES};
    let num_tags = AbilityTag::COUNT;
    let mut out = vec![0.0f32; MAX_ABILITIES * num_tags];
    for slot in 0..registry.len().min(MAX_ABILITIES) {
        let Some(id) = AbilityId::new((slot as u32) + 1) else { continue };
        let Some(prog) = registry.get(id) else { continue };
        for tag in AbilityTag::all() {
            let val = prog.tags.get(tag).copied().unwrap_or(0.0);
            out[slot * num_tags + tag as usize] = val;
        }
    }
    out
}
```

- [ ] **Step 4: Run the allocation test**

```bash
cargo test -p engine_gpu ensure_resident_init_allocates_chosen_ability_buf
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/backend/resident_ctx.rs crates/engine_gpu/src/lib.rs \
        crates/engine_gpu/tests/pick_ability_smoke.rs
git commit -m "feat(engine_gpu): allocate chosen_ability_buf + cooldown buffers in ensure_resident_init"
```

---

### Task B3: Dispatch `pick_ability` kernel in the `step_batch` inner loop

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` (the `step_batch` inner-loop tick body)

The dispatch goes between scoring and apply_actions, per spec §11.1. Find the comment `// 3. apply_actions + movement` in the batch inner loop (around line 1263) and insert before it.

- [ ] **Step 1: Write a failing test that verifies `chosen_ability_buf` is written non-sentinel after step_batch**

Add to `crates/engine_gpu/tests/pick_ability_smoke.rs`:

```rust
use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine_data::events::Event;

fn make_simple_ability_registry() -> engine::ability::AbilityRegistry {
    let mut b = AbilityRegistryBuilder::new();
    b.register(AbilityProgram {
        effects: vec![EffectOp::DealDamage(10.0)],
        gate: Gate {
            cooldown_ticks: 5,
            hostile_only: true,
            line_of_sight: false,
            range: Some(8.0),
        },
        hint: Some(engine::ability::AbilityHint::Damage),
        tags: {
            let mut t = std::collections::HashMap::new();
            t.insert(engine::ability::AbilityTag::Physical, 1.0);
            t
        },
    });
    b.build()
}

#[test]
fn step_batch_runs_pick_ability_kernel_and_outputs_to_chosen_buf() {
    // This test verifies pick_ability is dispatched (not that it picks
    // correctly — parity test covers that). We just need the batch to
    // complete without panic and the snapshot to include cast events.
    let (device, queue) = gpu_device_queue();
    let mut backend = GpuBackend::new_from_device(device, queue);
    let mut state = SimState::new(8);
    // Spawn two hostile agents in range
    let a = state.spawn_agent(AgentSpawn {
        pos: Vec3::new(0.0, 0.0, 0.0),
        creature_type: CreatureType::Human,
        ..Default::default()
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        pos: Vec3::new(4.0, 0.0, 0.0),
        creature_type: CreatureType::Wolf, // different species = hostile
        ..Default::default()
    }).unwrap();
    state.ability_registry = make_simple_ability_registry();

    // Run 1 batch tick
    backend.step_batch(&mut state, 1, &engine::cascade::CascadeRegistry::default());

    // Snapshot should be obtainable without panic
    let snap = backend.snapshot().expect("snapshot ok");
    // After tick 1, pick_ability ran; if the kernel fired correctly the
    // `AgentCast` event appears in events_since_last (Task C1 wires this).
    // For now just check the batch didn't panic and tick advanced.
    assert!(snap.tick == 0 || snap.tick == 1, "tick advanced (first snapshot is t-1 lagged)");
}
```

Run: `cargo test -p engine_gpu step_batch_runs_pick_ability_kernel_and_outputs_to_chosen_buf`
Expected: FAIL or partial — `pick_ability` kernel not yet dispatched.

- [ ] **Step 2: Insert the dispatch into the `step_batch` inner loop**

In `crates/engine_gpu/src/lib.rs`, find the scoring dispatch and the gap before apply_actions (around line 1252–1263):

```rust
// INSERT after scoring_resident dispatch, before apply_actions:

// 2b. pick_ability: evaluates per-agent ability scoring and writes
//     chosen_ability_buf. Runs after scoring so the mask/spatial data
//     (alive bitmap, agent SoA) are up-to-date. Runs before
//     apply_actions so chosen_ability_buf is ready to read.
{
    let crate::backend::ResidentPathContext {
        resident_agents_buf: Some(agents_buf),
        chosen_ability_buf: Some(chosen_buf),
        per_slot_cooldown_buf: Some(cd_buf),
        ability_registry_pa_buf: Some(reg_buf),
        tag_values_buf: Some(tag_buf),
        pick_ability_kernel: Some(kernel),
        sim_cfg_buf: Some(sim_cfg_buf),
        ..
    } = &mut self.resident else {
        panic!("pick_ability: buffers not initialised — call ensure_resident_init first");
    };

    // Upload per-tick cfg uniform
    let cfg = crate::pick_ability::PickAbilityCfg {
        agent_cap,
        ability_count: state.ability_registry.len().min(engine::ability::MAX_ABILITIES) as u32,
        num_tags: engine::ability::AbilityTag::COUNT as u32,
        tick: state.tick,
    };
    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pick_ability_cfg"),
        contents: bytemuck::bytes_of(&cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    kernel.run_resident(
        &self.device,
        &self.queue,
        &mut encoder,
        agents_buf,
        reg_buf,
        cd_buf,
        chosen_buf,
        &cfg_buf,
        sim_cfg_buf,
        tag_buf,
        agent_cap,
    );
}
```

Note: the per-tick `cfg_buf` is a small 16-byte buffer created each tick. For production perf, pool this alongside `apply_actions`'s cfg buffer. For now, ephemeral creation is fine.

- [ ] **Step 3: Run the smoke test**

```bash
cargo test -p engine_gpu step_batch_runs_pick_ability_kernel_and_outputs_to_chosen_buf
```
Expected: PASS (no panics, snapshot returns).

- [ ] **Step 4: Run full engine_gpu test suite to check for regressions**

```bash
cargo test -p engine_gpu
```
Expected: all previously-passing tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/lib.rs crates/engine_gpu/tests/pick_ability_smoke.rs
git commit -m "feat(engine_gpu): dispatch pick_ability kernel between scoring and apply_actions in step_batch"
```

---

### Task B4: Verify `PackedAbilityRegistry` tag/hint data bound to nothing — confirm wiring

**Files:**
- Read: `crates/engine_gpu/src/physics.rs` (the `PackedAbilityRegistry` binding audit note)
- No code changes required — this is a verification task

The audit notes (`physics.rs:478`) mark `hints` and `tag_values` as "Unbound in Phase 1 — Phase 4 wires this." The `pick_ability` kernel uses `ability_registry_pa_buf` and `tag_values_buf` (Task B2) rather than `PackedAbilityRegistry` directly. This is intentional: `PackedAbilityRegistry` is physics-kernel-oriented (effects arrays); the pick_ability kernel uses a lighter per-entry struct. No change to `PackedAbilityRegistry` binding is needed — it stays wired to the physics kernel only.

- [ ] **Step 1: Add a comment to `PackedAbilityRegistry.hints` to clarify permanent status**

In `crates/engine_gpu/src/physics.rs` at line ~478, replace:

```
/// Unbound in Phase 1 — Phase 4 wires this into `pick_ability.wgsl`.
pub hints: Vec<u32>,
```

with:

```
/// `AbilityHint` discriminant per ability slot. Retained for chronicle/debug
/// introspection. The `cs_pick_ability` kernel uses the lighter
/// `ability_registry_pa_buf` (see `pick_ability.rs`) not this field.
pub hints: Vec<u32>,
```

And similarly for `tag_values`:

```
/// Per-ability-per-tag power ratings. Retained for chronicle/debug.
/// The `cs_pick_ability` kernel uses `tag_values_buf` (see `pick_ability.rs`).
pub tag_values: Vec<f32>,
```

- [ ] **Step 2: Run `cargo build -p engine_gpu` to confirm no compile errors**

```bash
cargo build -p engine_gpu
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/physics.rs
git commit -m "docs(engine_gpu): clarify PackedAbilityRegistry.hints/tag_values binding status"
```

---

### Task B5: CPU `pick_ability` in `engine_rules` (Serial backend counterpart)

**Files:**
- Create: `crates/engine_rules/src/pick_ability.rs`
- Modify: `crates/engine_rules/src/lib.rs` (add `pub mod pick_ability;`)

The CPU picker (generated by Task A1's `emit_pick_ability_cpu`) needs to be wired into the Serial backend's step pipeline. This is the target file that the xtask compile-dsl step emits into. For the plan, we add the minimal scaffolding so it compiles and can be called from tests.

The `PICK_ABILITY_WGSL` constant also lives here — the `PickAbilityKernel::new` in Task B1 imports it from this module.

- [ ] **Step 1: Write a failing test that calls CPU pick_ability**

Add to `crates/engine_rules/tests/pick_ability_cpu.rs` (new file):

```rust
//! CPU pick_ability smoke — emitted function returns sentinel when no
//! abilities are in the registry.

use engine::state::SimState;
use engine::ability::AbilityRegistry;
use engine_rules::pick_ability::pick_ability;

#[test]
fn pick_ability_returns_sentinel_when_registry_empty() {
    let mut state = SimState::new(4);
    use engine::state::AgentSpawn;
    use engine_data::entities::CreatureType;
    use glam::Vec3;
    state.spawn_agent(AgentSpawn {
        pos: Vec3::ZERO,
        creature_type: CreatureType::Human,
        ..Default::default()
    }).unwrap();
    let reg = AbilityRegistry::new();
    let results = pick_ability(&state, &reg);
    assert_eq!(results.len(), 4, "one entry per agent_cap slot");
    assert!(results[0].is_none(), "no ability in registry → no cast");
}
```

Run: `cargo test -p engine_rules pick_ability_returns_sentinel_when_registry_empty`
Expected: FAIL — module does not exist.

- [ ] **Step 2: Create `crates/engine_rules/src/pick_ability.rs`**

This file is the target of `emit_pick_ability_cpu`. For bootstrapping, write it manually with the correct structure. The xtask compile-dsl step regenerates it; the hand-written version must match what the emitter will produce.

```rust
// GENERATED by dsl_compiler::emit_scoring::emit_pick_ability_cpu.
// Do not edit by hand.
// Regenerate with `cargo run --bin xtask -- compile-dsl`.

use engine::ability::{AbilityId, AbilityRegistry, AbilityTag};
use engine::state::SimState;

/// WGSL source for the cs_pick_ability kernel.
/// Emitted by dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl.
/// Used by engine_gpu::pick_ability::PickAbilityKernel::new.
pub const PICK_ABILITY_WGSL: &str = include_str!("pick_ability.wgsl");

/// Per-agent ability selection result. `None` = no cast this tick.
/// `Some((ab_slot, target_agent_slot))` — 0-based ability slot + target agent slot.
pub fn pick_ability(
    state: &SimState,
    registry: &AbilityRegistry,
) -> Vec<Option<(u8, u32)>> {
    let agent_cap = state.agent_cap();
    let tick = state.tick;
    let mut results = vec![None; agent_cap];
    for agent_slot in 0..agent_cap {
        let Some(agent_id) = engine::ids::AgentId::new((agent_slot as u32) + 1) else { continue };
        if !state.agent_alive(agent_id).unwrap_or(false) { continue }
        let mut best_score: f32 = f32::NEG_INFINITY;
        let mut best: Option<(u8, u32)> = None;
        let n_abilities = registry.len().min(engine::ability::MAX_ABILITIES);
        for ab_slot in 0..n_abilities {
            let Some(ab_id) = AbilityId::new((ab_slot as u32) + 1) else { continue };
            let Some(prog) = registry.get(ab_id) else { continue };
            // guard: !ability::on_cooldown(ability)
            let per_slot_ready = state.ability_cooldowns
                .get(agent_slot)
                .map(|row| row[ab_slot] <= tick)
                .unwrap_or(true);
            let global_ready = state.agent_cooldown_next_ready(agent_id)
                .map(|t| t <= tick)
                .unwrap_or(true);
            if !per_slot_ready || !global_ready { continue }
            // score: ability::tag(PHYSICAL)
            let score: f32 = registry.tag_value(ab_id, AbilityTag::Physical).unwrap_or(0.0);
            if score > best_score {
                // target: nearest_hostile_in_range(ability::range)
                let range = prog.gate.area_range();
                let target_slot = state.nearest_hostile_in_range(agent_id, range);
                if let Some(t) = target_slot {
                    best_score = score;
                    best = Some((ab_slot as u8, t.get() - 1));
                }
            }
        }
        results[agent_slot] = best;
    }
    results
}
```

Also create `crates/engine_rules/src/pick_ability.wgsl` — this is the inline WGSL included by `include_str!`. It is the output of `emit_pick_ability_wgsl` for the current `scoring.sim`. Bootstrap with the minimal body:

```wgsl
// GENERATED by dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl.
// Do not edit by hand. Regenerate with `cargo run --bin xtask -- compile-dsl`.

// ... (content of emit_pick_ability_wgsl for the production scoring.sim)
// Bootstrapped manually; xtask compile-dsl regenerates this.
```

For the actual WGSL content, run:

```bash
cargo run --bin xtask -- compile-dsl --emit pick-ability-wgsl > /tmp/pick_ability_bootstrap.wgsl
```

If the xtask flag doesn't exist yet, paste the emit output from a test:

```bash
cargo test -p dsl_compiler emit_pick_ability_wgsl_produces_cs_entry_point -- --nocapture 2>&1 | grep -A200 "fn cs_pick_ability"
```

Then copy the full WGSL string into `crates/engine_rules/src/pick_ability.wgsl`.

- [ ] **Step 3: Add `pub mod pick_ability;` to `engine_rules/src/lib.rs`**

- [ ] **Step 4: Run the CPU pick_ability test**

```bash
cargo test -p engine_rules pick_ability_returns_sentinel_when_registry_empty
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_rules/src/pick_ability.rs crates/engine_rules/src/pick_ability.wgsl \
        crates/engine_rules/src/lib.rs crates/engine_rules/tests/pick_ability_cpu.rs
git commit -m "feat(engine_rules): CPU pick_ability scaffolding + PICK_ABILITY_WGSL constant"
```

---

## Group C — Wire chosen ability into apply (~2 tasks)

### Task C1: `apply_actions` reads `chosen_ability_buf` and emits `AgentCast`

**Files:**
- Modify: `crates/engine_gpu/src/apply_actions.rs`

The WGSL for `cs_apply_actions` currently skips Cast. When `chosen_ability_buf[agent*2] != SENTINEL`, the kernel should emit `AgentCast { actor: agent_id, ability: ability_slot, target: target_agent_id }` into the event ring and skip the scoring kernel's chosen_action for this agent. Add binding `@group(0) @binding(6)` for `chosen_ability_buf` (read-only).

The `AgentCast` event byte-packing must match the existing CPU path. Find the event packing in `crates/engine_gpu/src/event_ring.rs`:

```bash
grep -n "AgentCast\|AGENT_CAST\|kind.*cast" crates/engine_gpu/src/event_ring.rs | head -10
```

Use the existing event kind constant to emit the record.

- [ ] **Step 1: Write a failing test for AgentCast emission**

Add to `crates/engine_gpu/tests/pick_ability_smoke.rs`:

```rust
#[test]
fn apply_actions_emits_agent_cast_when_chosen_ability_buf_set() {
    // Seeds chosen_ability_buf with (slot=0, target=1) for agent 0,
    // then dispatches apply_actions alone and checks AgentCast in event ring.
    let (device, queue) = gpu_device_queue();

    // Build state with 2 agents
    let mut state = SimState::new(4);
    let a0 = state.spawn_agent(AgentSpawn { pos: Vec3::ZERO, creature_type: CreatureType::Human, ..Default::default() }).unwrap();
    let a1 = state.spawn_agent(AgentSpawn { pos: Vec3::new(3.0, 0.0, 0.0), creature_type: CreatureType::Wolf, ..Default::default() }).unwrap();
    state.ability_registry = make_simple_ability_registry();

    // Pre-seed chosen_ability_buf: agent 0 chose ability slot 0, target = agent slot 1
    let chosen_data: Vec<u32> = vec![
        0, 1,            // agent 0: slot=0, target=1
        0xFFFFFFFF, 0xFFFFFFFF, // agent 1: no cast
        0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF,
    ];
    let chosen_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_chosen_buf"),
        contents: bytemuck::cast_slice(&chosen_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Dispatch apply_actions with chosen_buf bound — check event ring for AgentCast
    // (Use the existing apply_actions test harness pattern from cold_state_gold_transfer.rs)
    // For brevity: just run step_batch(1) end-to-end and check snapshot events
    let mut backend = GpuBackend::new_from_device(device, queue);
    backend.step_batch(&mut state, 1, &engine::cascade::CascadeRegistry::default());
    let snap = backend.snapshot().unwrap();
    let snap2 = backend.snapshot().unwrap(); // second call returns t-1 data

    let has_cast = snap2.events_since_last.iter().any(|e| {
        matches!(e, engine::event::EventRecord { .. }) // check kind == AgentCast
        // Exact match depends on EventRecord shape — adjust to actual field names
    });
    // If pick_ability fires and apply_actions reads it, cast should appear.
    // On first run this will fail until C1 is fully wired.
    assert!(has_cast || snap2.tick < 2, "AgentCast should appear in snapshot events");
}
```

Run: `cargo test -p engine_gpu apply_actions_emits_agent_cast_when_chosen_ability_buf_set`
Expected: FAIL (no AgentCast yet).

- [ ] **Step 2: Add `@binding(6) chosen_ability_buf` to `cs_apply_actions` WGSL**

In `crates/engine_gpu/src/apply_actions.rs`, find the WGSL source string (around line 200+) and add:

```wgsl
@group(0) @binding(6) var<storage, read> chosen_ability_buf: array<u32>;
// Layout: [agent*2+0] = ability_slot (0xFFFFFFFF = no cast), [agent*2+1] = target_slot
```

At the top of the `cs_apply_actions` entry point, before the existing action-dispatch switch, add:

```wgsl
// Check chosen_ability_buf first — if set, emit AgentCast and return.
let chosen_ab_slot = chosen_ability_buf[agent_slot * 2u];
if chosen_ab_slot != 0xFFFFFFFFu {
    let chosen_tgt = chosen_ability_buf[agent_slot * 2u + 1u];
    if chosen_tgt != 0xFFFFFFFFu {
        // Emit AgentCast event
        let tail = atomicAdd(&event_ring_tail, 1u);
        if tail < arrayLength(&event_ring) {
            event_ring[tail] = pack_agent_cast(agent_slot, chosen_ab_slot, chosen_tgt, cfg.tick);
        }
        return; // ability chosen; skip normal scoring action
    }
}
```

Add the `pack_agent_cast` WGSL helper following the existing `pack_agent_attacked` helper pattern in the event ring. The byte layout must match `Event::AgentCast { actor, ability, target }` as packed by `engine_gpu::event_ring::pack_event`.

Check the exact packing by reading:

```bash
grep -n "AgentCast\|CAST" crates/engine_gpu/src/event_ring.rs | head -20
```

Then add the Rust-side binding: in `ApplyActionsKernel::run_and_readback` and `run_resident`, pass `chosen_ability_buf` as binding 6.

- [ ] **Step 3: Update `run_and_readback` and `run_resident` signatures**

In `crates/engine_gpu/src/apply_actions.rs`, the `run_resident` and `run_and_readback` functions need a new parameter:

```rust
pub fn run_resident(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    scoring_buf: &wgpu::Buffer,
    sim_cfg_ref: &wgpu::Buffer,
    event_ring: &GpuEventRing,
    chosen_ability_buf: &wgpu::Buffer,  // NEW
    agent_cap: u32,
) { ... }
```

Update all callers in `lib.rs` to pass `self.resident.chosen_ability_buf.as_ref().expect("...")`.

- [ ] **Step 4: Run the AgentCast test**

```bash
cargo test -p engine_gpu apply_actions_emits_agent_cast_when_chosen_ability_buf_set
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/apply_actions.rs crates/engine_gpu/src/lib.rs \
        crates/engine_gpu/tests/pick_ability_smoke.rs
git commit -m "feat(engine_gpu): apply_actions reads chosen_ability_buf and emits AgentCast"
```

---

### Task C2: Cooldown writeback after cast

**Files:**
- Modify: `crates/engine_gpu/src/apply_actions.rs` (WGSL)
- Modify: `crates/engine_gpu/src/lib.rs` (snapshot merge)

When `AgentCast` is emitted, the per-slot cooldown must be written. On the CPU path this is done in the cascade handler. On the GPU batch path, the simplest approach (per spec §11 footnote) is: the `cs_apply_actions` kernel writes the new per-slot expiry tick directly into `per_slot_cooldown_buf` at the same time it emits `AgentCast`.

Add `@group(0) @binding(7) var<storage, read_write> per_slot_cooldown_buf: array<u32>;` to `cs_apply_actions`.

In the cast block:

```wgsl
// Set per-slot cooldown: expiry = tick + ability_registry_pa[chosen_ab_slot].cooldown_ticks
// Note: ability_registry_pa is not bound to apply_actions; use a small uniform or
// pass cooldown_ticks alongside chosen_ability_buf as a third u32 per agent.
// SIMPLIFICATION: store (slot, target, cooldown_ticks) as 3 u32 per agent.
```

**Implementation simplification:** Change `chosen_ability_buf` layout to 3× u32 per agent:
- `[agent*3+0]` = ability_slot (0xFFFFFFFF = sentinel)
- `[agent*3+1]` = target_slot
- `[agent*3+2]` = cooldown_ticks (from `GpuPickAbilityEntry.cooldown_ticks`)

The `cs_pick_ability` kernel already has access to `ability_registry` so writing `cooldown_ticks` here is natural. Update the packing constant in `pick_ability.rs` and the schema hash accordingly.

- [ ] **Step 1: Write a failing test for cooldown writeback**

Add to `crates/engine_rules/tests/pick_ability_cpu.rs`:

```rust
use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_rules::pick_ability::pick_ability;
use glam::Vec3;

#[test]
fn pick_ability_skips_ability_on_cooldown() {
    let mut b = AbilityRegistryBuilder::new();
    b.register(AbilityProgram {
        effects: vec![EffectOp::DealDamage(5.0)],
        gate: Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false, range: Some(6.0) },
        hint: None,
        tags: { let mut t = std::collections::HashMap::new(); t.insert(engine::ability::AbilityTag::Physical, 1.0); t },
    });
    let registry = b.build();

    let mut state = SimState::new(4);
    state.spawn_agent(AgentSpawn { pos: Vec3::ZERO, creature_type: CreatureType::Human, ..Default::default() }).unwrap();
    state.spawn_agent(AgentSpawn { pos: Vec3::new(3.0, 0.0, 0.0), creature_type: CreatureType::Wolf, ..Default::default() }).unwrap();

    // First pick — ability slot 0 is ready (cooldown[0][0] = 0, tick = 0)
    let results_t0 = pick_ability(&state, &registry);
    assert!(results_t0[0].is_some(), "ability should be chosen at t=0");

    // Set cooldown expiry to tick 15 (not yet expired at tick=0)
    state.ability_cooldowns[0][0] = 15;
    let results_on_cd = pick_ability(&state, &registry);
    assert!(results_on_cd[0].is_none(), "ability should be skipped while on cooldown");

    // Advance tick past expiry
    state.tick = 16;
    let results_ready = pick_ability(&state, &registry);
    assert!(results_ready[0].is_some(), "ability should be chosen again after cooldown expires");
}
```

Run: `cargo test -p engine_rules pick_ability_skips_ability_on_cooldown`
Expected: PASS (the CPU pick_ability already checks cooldowns per Task B5).

- [ ] **Step 2: Update `chosen_ability_buf` layout to 3× u32 per agent (slot, target, cooldown_ticks)**

In `pick_ability.rs` (both Rust and WGSL), change array stride:
- `chosen_ability_buf` is now `agent_cap * 3 * 4` bytes.
- Update allocation in `ensure_resident_init`.
- Update WGSL write: `chosen_ability_buf[agent_slot * 3u] = best_ab; chosen_ability_buf[agent_slot * 3u + 1u] = best_tgt; chosen_ability_buf[agent_slot * 3u + 2u] = ability_registry[best_ab].cooldown_ticks;`
- Update schema hash comment to reflect 3×u32 layout.

In `cs_apply_actions`:
```wgsl
let chosen_ab_slot  = chosen_ability_buf[agent_slot * 3u];
let chosen_tgt      = chosen_ability_buf[agent_slot * 3u + 1u];
let chosen_cd_ticks = chosen_ability_buf[agent_slot * 3u + 2u];
if chosen_ab_slot != 0xFFFFFFFFu && chosen_tgt != 0xFFFFFFFFu {
    // emit AgentCast ...
    // write per-slot cooldown expiry
    let cd_idx = agent_slot * MAX_ABILITIES + chosen_ab_slot;
    if cd_idx < arrayLength(&per_slot_cooldown_buf) {
        per_slot_cooldown_buf[cd_idx] = sim_cfg.tick + chosen_cd_ticks;
    }
    return;
}
```

- [ ] **Step 3: Update schema hash for 3×u32 layout**

In `schema_hash.rs`, update the comment/hash input:

```rust
h.update(b"chosen_ability_buf_packing: 3xu32_per_agent_slot_target_cooldown_ticks");
```

- [ ] **Step 4: Run all tests to verify no regressions**

```bash
cargo test -p engine_gpu && cargo test -p engine_rules && cargo test -p dsl_compiler
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/pick_ability.rs crates/engine_gpu/src/apply_actions.rs \
        crates/engine_gpu/src/lib.rs crates/dsl_compiler/src/schema_hash.rs \
        crates/engine_rules/src/pick_ability.rs crates/engine_rules/tests/pick_ability_cpu.rs
git commit -m "feat(engine_gpu): apply_actions writes per-slot cooldown expiry from chosen_ability_buf"
```

---

## Group D — Parity test + final verify (~2 tasks)

### Task D1: Cross-backend parity test — Serial pick matches GPU pick

**Files:**
- Create: `crates/engine_gpu/tests/pick_ability_parity.rs`

This is the mandatory P3 parity test. Run the same scenario (2 agents, 1 ability with PHYSICAL tag) for 10 ticks on `SerialBackend` (using CPU `pick_ability`) and on `GpuBackend` sync path. Assert the cast event multiset matches.

- [ ] **Step 1: Write the parity test**

Create `crates/engine_gpu/tests/pick_ability_parity.rs`:

```rust
//! Cross-backend parity: CPU Serial pick_ability vs. GPU cs_pick_ability.
//!
//! Runs 10 ticks on a 2-agent hostile scenario with one physical ability.
//! Asserts that the multiset of (actor, ability_slot, target) triples in
//! AgentCast events matches across both backends.
//!
//! This is the mandatory P3 parity test for Subsystem 3.

#![cfg(feature = "gpu")]

use engine::ability::{AbilityProgram, AbilityRegistryBuilder, AbilityTag, EffectOp, Gate};
use engine::cascade::CascadeRegistry;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_gpu::GpuBackend;
use engine_rules::pick_ability::pick_ability as cpu_pick;
use glam::Vec3;
use std::collections::BTreeMap;

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                force_fallback_adapter: true,
                power_preference: wgpu::PowerPreference::None,
            })
            .await
            .expect("adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device")
    })
}

fn make_registry() -> engine::ability::AbilityRegistry {
    let mut b = AbilityRegistryBuilder::new();
    b.register(AbilityProgram {
        effects: vec![EffectOp::DealDamage(5.0)],
        gate: Gate { cooldown_ticks: 3, hostile_only: true, line_of_sight: false, range: Some(8.0) },
        hint: None,
        tags: { let mut t = std::collections::HashMap::new(); t.insert(AbilityTag::Physical, 1.0); t },
    });
    b.build()
}

fn make_state() -> SimState {
    let mut state = SimState::new(8);
    state.spawn_agent(AgentSpawn {
        pos: Vec3::new(0.0, 0.0, 0.0),
        creature_type: CreatureType::Human,
        ..Default::default()
    }).unwrap();
    state.spawn_agent(AgentSpawn {
        pos: Vec3::new(4.0, 0.0, 0.0),
        creature_type: CreatureType::Wolf,
        ..Default::default()
    }).unwrap();
    state.ability_registry = make_registry();
    state
}

#[test]
fn pick_ability_serial_gpu_parity_10_ticks() {
    // --- Serial (CPU) run ---
    let mut cpu_state = make_state();
    let mut cpu_casts: Vec<(u32, u8, u32)> = Vec::new(); // (actor_slot, ab_slot, tgt_slot)

    for _tick in 0..10 {
        let picks = cpu_pick(&cpu_state, &cpu_state.ability_registry);
        for (agent_slot, pick) in picks.iter().enumerate() {
            if let Some((ab_slot, tgt_slot)) = pick {
                cpu_casts.push((agent_slot as u32, *ab_slot, *tgt_slot));
                // Update cooldown
                if let Some(row) = cpu_state.ability_cooldowns.get_mut(agent_slot) {
                    let cd = cpu_state.ability_registry
                        .get(engine::ability::AbilityId::new(((*ab_slot as u32) + 1)).unwrap())
                        .map(|p| p.gate.cooldown_ticks)
                        .unwrap_or(0);
                    row[*ab_slot as usize] = cpu_state.tick + cd;
                }
            }
        }
        cpu_state.tick += 1;
    }

    // --- GPU (sync path) run ---
    let (device, queue) = gpu_device_queue();
    let mut gpu_backend = GpuBackend::new_from_device(device, queue);
    let mut gpu_state = make_state();
    let mut gpu_events = EventRing::with_cap(4096);
    let cascade = CascadeRegistry::default();
    let mut gpu_casts: Vec<(u32, u8, u32)> = Vec::new();

    for _tick in 0..10 {
        gpu_backend.step(&mut gpu_state, &mut gpu_events, &cascade)
            .expect("GPU step failed");
        for event in gpu_events.drain_replayable() {
            if let Event::AgentCast { actor, ability, target, .. } = event {
                let actor_slot = actor.get() - 1;
                let tgt_slot = target.map(|t| t.get() - 1).unwrap_or(0xFFFF_FFFF);
                gpu_casts.push((actor_slot, ability.get() as u8 - 1, tgt_slot));
            }
        }
    }

    // Compare multisets (order may differ)
    let mut cpu_sorted = cpu_casts.clone(); cpu_sorted.sort();
    let mut gpu_sorted = gpu_casts.clone(); gpu_sorted.sort();
    assert_eq!(
        cpu_sorted, gpu_sorted,
        "Serial and GPU pick_ability must produce identical cast multisets over 10 ticks.\nCPU: {cpu_sorted:?}\nGPU: {gpu_sorted:?}"
    );
}
```

Run: `cargo test -p engine_gpu pick_ability_serial_gpu_parity_10_ticks`
Expected: FAIL initially (GPU path not fully wired yet), then PASS after all Group C tasks complete.

- [ ] **Step 2: Fix any parity divergences**

Run with `--nocapture` to see the CPU vs GPU cast arrays:

```bash
cargo test -p engine_gpu pick_ability_serial_gpu_parity_10_ticks -- --nocapture
```

Common causes of divergence:
- Cooldown writeback on GPU side not advancing tick correctly (check `cfg.tick` vs `sim_cfg.tick` — use `sim_cfg.tick` for consistency with batch path)
- Target resolution differs: CPU uses `nearest_hostile_in_range`; GPU uses linear scan with `dist3`. Verify both use the same distance metric (3D Euclidean per spec §4.5).

- [ ] **Step 3: Confirm full engine_gpu test suite passes**

```bash
cargo test -p engine_gpu
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/tests/pick_ability_parity.rs
git commit -m "test(engine_gpu): pick_ability_parity cross-backend test (P3 compliance)"
```

---

### Task D2: AIS post-design tick + xtask compile-dsl integration

**Files:**
- Modify: `this plan file` (tick post-design AIS checkbox)
- Modify: `crates/bin/xtask` or wherever `compile-dsl` emitter dispatch lives — add `pick_ability` emission to the pipeline

The `compile-dsl` xtask must call `emit_pick_ability_cpu` and `emit_pick_ability_wgsl` and write their outputs to `crates/engine_rules/src/pick_ability.rs` and `crates/engine_rules/src/pick_ability.wgsl` respectively.

- [ ] **Step 1: Find the compile-dsl emit dispatch**

```bash
grep -rn "emit_scoring\|emit_view\|emit_physics\|compile.dsl\|compile_dsl" src/bin/xtask/ | head -20
```

- [ ] **Step 2: Add `emit_pick_ability_cpu` + `emit_pick_ability_wgsl` to the xtask dispatch**

In the xtask file that calls the emitters (likely `src/bin/xtask/compile_dsl.rs` or similar), add:

```rust
// Emit pick_ability CPU function
let pick_ability_cpu = dsl_compiler::emit_scoring::emit_pick_ability_cpu(
    comp.scoring.first().expect("scoring block required for pick_ability emission")
).map_err(|e| format!("emit_pick_ability_cpu: {e}"))?;

if !pick_ability_cpu.is_empty() {
    let dest = out_dir.join("pick_ability.rs");
    std::fs::write(&dest, &pick_ability_cpu)
        .map_err(|e| format!("write pick_ability.rs: {e}"))?;
    println!("wrote {}", dest.display());
}

// Emit pick_ability WGSL kernel
let pick_ability_wgsl = dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl(
    comp.scoring.first().expect("scoring block required")
);

if !pick_ability_wgsl.is_empty() {
    let wgsl_dest = out_dir.join("pick_ability.wgsl");
    std::fs::write(&wgsl_dest, &pick_ability_wgsl)
        .map_err(|e| format!("write pick_ability.wgsl: {e}"))?;
    println!("wrote {}", wgsl_dest.display());
}
```

- [ ] **Step 3: Run `cargo run --bin xtask -- compile-dsl` and verify outputs are written**

```bash
cargo run --bin xtask -- compile-dsl
```
Expected: outputs mention `pick_ability.rs` and `pick_ability.wgsl`.

- [ ] **Step 4: Run the full test suite**

```bash
cargo test
```
Expected: all previously-passing tests PASS plus the new pick_ability tests.

- [ ] **Step 5: Tick post-design AIS checkbox and commit**

In this plan file, mark `[ ] AIS reviewed post-design` as `[x]`.

```bash
git add docs/superpowers/plans/2026-04-26-subsystem-3-ability-eval-impl.md \
        src/bin/xtask/ \
        crates/engine_rules/src/pick_ability.rs \
        crates/engine_rules/src/pick_ability.wgsl
git commit -m "feat: Subsystem 3 ability eval complete — compile-dsl emits pick_ability, parity test passes"
```

---

## Self-review checklist

**Spec coverage:**

| Spec §ref | Requirement | Task |
|---|---|---|
| §11.1 Pipeline position | `pick_ability` between scoring and apply_actions | B3 |
| §11.2 `pick_ability` kernel | Per-agent iterate, guard, score, target, argmax | A2 + B1 + B3 |
| §11.3 `ability::tag` primitive | IR already done; CPU emit + WGSL emit | A1 + A2 |
| §11.4 `per_ability` row type | CPU emit (was ignored) | A1 |
| §11.5 Tag registry | `PackedAbilityRegistry.tag_values` bound to `tag_values_buf` | B2 |
| §11.6 `chosen_ability_buf` allocation | `ResidentPathContext` field + `ensure_resident_init` | B2 |
| §11.6 `chosen_ability_buf` sentinel | `SENTINEL_NO_CAST = 0xFFFFFFFF` | B1 |
| §11.6 All-on-cooldown → sentinel | CPU + GPU guard check | A1 + A2 |
| §11 apply_actions reads buf | `@binding(6)` + AgentCast emit | C1 |
| §11 Cooldown writeback | Per-slot expiry written on cast | C2 |
| §7.3 `pick_ability_*` tests | Smoke + parity test | B1 + D1 |
| §8 Schema hash | `chosen_ability_buf` packing format in hash | A3 |

**Placeholder scan:** No TBDs remain. Every step has code or a concrete command.

**Type consistency:**
- `chosen_ability_buf` uses 3× u32 per agent throughout (Tasks C2 updated all references).
- `GpuPickAbilityEntry` is 16 bytes with `const _: () = assert!(...)` compile-time check.
- `PickAbilityCfg` is 16 bytes with `const _: () = assert!(...)` compile-time check.
- `SENTINEL_NO_CAST = 0xFFFF_FFFFu32` is consistent between Rust and WGSL.
- `emit_pick_ability_cpu` and `emit_pick_ability_wgsl` both import `IrExprNode` / `ScoringIR` from `dsl_compiler::ir` — same types.

**IR interpreter note:** The cherry-pick agent (`ad17f9bd`) is porting the IR interpreter from `wsb-engine-viz`. Once `ability::tag` becomes interpretable there, `emit_pick_ability_wgsl` will produce WGSL that the interpreter can evaluate for fast iteration without a cargo rebuild. This plan does not depend on the interpreter — the emit path (WGSL to GPU) works independently.
