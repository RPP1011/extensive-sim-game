# Runtime Input Channel Implementation Plan (Plan 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `@runtime`-annotated DSL config fields readable inside rules as per-tick host-writable values, and expose a host setter on the generated runtime — the input channel the playable game needs.

**Architecture:** The compiler already has the "Plan G tunable cfg" feature (`RuntimeCfgField` in `crates/dsl_compiler/src/cg/emit/kernel.rs:2600`): a `@runtime` config field is excluded from inline-`const` emission (`program.rs:958`), added to the per-kernel cfg uniform struct as `cfg.<name>`, and initialized in `build_cfg` with a default ("Host overwrites per tick"). This plan **characterizes the actual end-to-end state with a probe fixture, fills whatever gap the failing test reveals** (most likely: the generated `step()` writes only `[agent_count, tick, seed]` and never copies the public host-mirror fields into the cfg buffer), and locks it with a headless test.

**Tech Stack:** Rust, custom DSL (`crates/dsl_ast`, `crates/dsl_compiler`), wgpu compute, generated `*_runtime` crates via `crates/sims/build.rs`.

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `RuntimeCfgField` at `crates/dsl_compiler/src/cg/emit/kernel.rs:2600` (cfg-field synthesis from `@runtime`)
  - `prog.runtime_config_consts` skip at `crates/dsl_compiler/src/cg/emit/program.rs:958`
  - `ConfigField.runtime` flag at `crates/dsl_ast/src/ast.rs:1135` / parsed at `crates/dsl_ast/src/parser.rs:3449`
  - `mark_config_const_runtime` at `crates/dsl_compiler/src/cg/lower/driver.rs:2208`
  - generated host mirrors `pub <field>: <ty>` at `build_helper.rs:2270`; `step()` cfg write at `build_helper.rs:3041-3110`
  - Search method: `rg` + direct `Read`.
- **Decision:** extend the existing `@runtime` config path (complete it end-to-end) rather than add a new `input {}` block — the primitive already exists and generalizes config runtime-override.
- **Rule-compiler touchpoints:**
  - DSL inputs edited: `assets/sim/input_probe.sim` (new), `assets/sim/vampire_survivors.sim` (append `config ctl {}`)
  - Generated outputs re-emitted: `crates/sims` runtimes (`input_probe`, `vampire_survivors`) via build.rs
- **Hand-written downstream code:** NONE (host setter is emitter-generated).
- **Constitution check:**
  - P1 (Compiler-First): PASS — runtime cfg field flows through the emitter; the host setter is generated, not hand-written. Evidence: `kernel.rs:2600`, generated `set_*` in this plan.
  - P2 (Schema-Hash): N/A — cfg-uniform field, not Agent SoA or event variant.
  - P3 (Cross-Backend Parity): PASS — `cfg.<name>` read lowers identically; host write is the determinism boundary.
  - P5 (Keyed PCG): N/A.
  - P6 (Events Are the Mutation Channel): PASS — runtime cfg is read-only into rules.
  - P10 (No Runtime Panic): PASS — gated by the headless probe test.
  - P8 (AIS Required): PASS — this section.
- **Runtime gate:**
  - `input_probe_moves_agent` at `crates/sims/tests/input_probe_exec.rs` — after `set_ctl_drive(2.0)` + one `step()`, agent 0's `pos.x` increased by ~2.0.
- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] AIS reviewed post-design.

---

### Task 1: Probe fixture + characterization test (discover the real gap)

**Files:**
- Create: `assets/sim/input_probe.sim`
- Modify: `crates/sims/build.rs` (add `"input_probe"` to the fixture list — find the `matches!(name, ...)` list)
- Test: `crates/sims/tests/input_probe_exec.rs`

- [ ] **Step 1: Write the probe fixture.** A single rule that moves every alive agent by a `@runtime` config field each tick.

```
// assets/sim/input_probe.sim — minimal @runtime config probe.
event Tick { }

config probe {
  drive: f32 = 0.0 @runtime,
}

physics DriveX @phase(per_agent) {
  on Tick {} where (self.alive) {
    agents.set_pos(self, self.pos + vec3(config.probe.drive, 0.0, 0.0));
  }
}
```

- [ ] **Step 2: Register the fixture.** In `crates/sims/build.rs`, add `"input_probe"` to the fixture-name list (the same list `"vampire_survivors"` lives in). Run `cargo build -p sims 2>&1 | tail -20`. Expected: a `GeneratedRuntime` for `input_probe` is emitted (build succeeds, or surfaces a DSL error to fix).

- [ ] **Step 3: Write the failing end-to-end test.** This is the characterization — it tells us exactly what's wired.

```rust
// crates/sims/tests/input_probe_exec.rs
use sims::input_probe::GeneratedRuntime;

fn read_pos_x(rt: &mut GeneratedRuntime, slot: usize) -> f32 {
    // staging readback of agent_pos_buf (vec4 stride 16B); mirror crates/viewer_runtime/src/vs.rs read_vec4
    let buf = rt.agent_pos_buf.clone();
    let bytes = (slot as u64 + 1) * 16;
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb"), size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let sl = staging.slice(..bytes);
    sl.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    rt.gpu.device.poll(wgpu::PollType::Wait).unwrap();
    let words = bytemuck::cast_slice::<u8, u32>(&sl.get_mapped_range()).to_vec();
    f32::from_bits(words[slot * 4])
}

#[test]
fn input_probe_moves_agent() {
    let Some(mut rt) = GeneratedRuntime::try_new(0xABCD, 8) else {
        eprintln!("no GPU adapter; skipping"); return;
    };
    // seed: mark agent 0 alive at origin (use whatever seed API the runtime exposes;
    // if none, write agent_alive_buf[0]=1 + agent_pos_buf[0]=0 directly via queue.write_buffer).
    // ... (seeding helper — see Step 4 if a seed fn is needed) ...
    rt.set_probe_drive(2.0);   // <-- the host setter under test (exact name TBD by Step 4)
    let before = read_pos_x(&mut rt, 0);
    rt.step();
    let after = read_pos_x(&mut rt, 0);
    assert!((after - before - 2.0).abs() < 1e-3, "expected +2.0 from drive, got {} -> {}", before, after);
}
```

- [ ] **Step 4: Run it and record what's missing.** Run: `RUST_MIN_STACK=33554432 cargo test -p sims --test input_probe_exec -- --nocapture`. Three possible outcomes, each determines the next tasks:
  - (a) **Compile error: `set_probe_drive` not found** → the generator emits the `pub <field>` mirror but no setter. Add a generated setter (Task 2).
  - (b) **Test runs, agent does NOT move** → `step()` writes the default into the cfg buffer but not the host mirror. Fix the per-tick pack (Task 3).
  - (c) **Test passes** → the feature already works end-to-end; skip Tasks 2-3, go to Task 4 (lock + the `vampire_survivors` contract block).
  Record the outcome in the commit message.

- [ ] **Step 5: Commit the probe + characterization.**
```bash
git add assets/sim/input_probe.sim crates/sims/build.rs crates/sims/tests/input_probe_exec.rs
git commit -m "test(dsl): characterize @runtime config end-to-end via input_probe fixture"
```

### Task 2: Generated host setter (only if Task 1 outcome (a))

**Files:**
- Modify: `crates/dsl_compiler/src/build_helper.rs` (near the `pub <field>: <ty>` mirror emission, ~line 2270, and impl block)

- [ ] **Step 1: Emit a setter per `@runtime` field.** Where the generator emits `pub {name}: {scalar_ty},` mirrors, also emit into the `impl GeneratedRuntime` block:
```rust
// generated:
pub fn set_{name}(&mut self, v: {scalar_ty}) { self.{name} = v; }
```
Use the same field iteration that produces the mirrors (the `RuntimeCfgField` list / runtime_config_defaults). The setter name is `set_<block>_<field>` to match `cfg.config_<block>_<field>` (confirm the mirror's actual field name in the generated `runtime_core.rs` and match it).

- [ ] **Step 2: Rebuild + rerun the probe test.** Run: `cargo build -p sims 2>&1 | tail -5 && RUST_MIN_STACK=33554432 cargo test -p sims --test input_probe_exec`. Expected: compiles; test now either passes (done) or fails on movement → Task 3.

- [ ] **Step 3: Commit.**
```bash
git add crates/dsl_compiler/src/build_helper.rs
git commit -m "feat(dsl): generate set_<field> host setters for @runtime config fields"
```

### Task 3: Per-tick pack of host mirrors into the cfg buffer (only if outcome (b))

**Files:**
- Modify: `crates/dsl_compiler/src/build_helper.rs` `step()` cfg-write region (~3041-3110)

- [ ] **Step 1: Pack the runtime mirrors into the cfg words.** The cfg struct layout is `{ event_count, tick, seed, agent_cap, <runtime fields in ConfigConstId order> }` (per `RuntimeCfgField::render_*_suffix`, `kernel.rs:2632`). Today `step()` writes `[agent_count, tick, seed, 0]`. Extend it to append each runtime mirror field's bits in the same `ConfigConstId`-ascending order the emitter uses, e.g.:
```rust
let mut cfg_words: Vec<u32> = vec![0 /*event_count*/, self.tick as u32, self.seed as u32, self.agent_cap()];
// append @runtime mirrors in ConfigConstId order (generator emits this list):
cfg_words.push(self.{runtime_field_0}.to_bits_as_u32());  // f32 -> bits, u32/i32 -> as u32
// ... one push per runtime field ...
let cfg_bytes: &[u8] = bytemuck::cast_slice(&cfg_words);
for name in &cfg_buffer_names { self.gpu.queue.write_buffer(/* cfg_{name}_buf */, 0, cfg_bytes); }
```
The generator must emit one push per `@runtime` field (reuse the `RuntimeCfgField` list it already builds for the WGSL struct so order matches exactly). For f32 fields use `.to_bits()`, for i32 use `as u32`.

- [ ] **Step 2: Confirm cfg buffer is large enough.** The per-kernel `cfg_{name}_buf` must be `4 + N_runtime` u32 wide. Find its allocation in `try_new()` and size it from the same field count (search `cfg_` buffer creation). If it was hardcoded to 16 bytes, widen to `(4 + N) * 4`.

- [ ] **Step 3: Rebuild + rerun.** Run: `cargo build -p sims 2>&1 | tail -5 && RUST_MIN_STACK=33554432 cargo test -p sims --test input_probe_exec`. Expected: PASS (agent moves by 2.0).

- [ ] **Step 4: Commit.**
```bash
git add crates/dsl_compiler/src/build_helper.rs
git commit -m "feat(dsl): pack @runtime config mirrors into per-tick cfg uniform"
```

### Task 4: Lock lowering + add the vampire_survivors `config ctl {}` contract block

**Files:**
- Test: `crates/dsl_compiler/tests/runtime_config_emit.rs` (create)
- Modify: `assets/sim/vampire_survivors.sim` (append `config ctl {}` — additive only; do NOT touch rules, that's Plan 3)

- [ ] **Step 1: Compile-gate — runtime field reads cfg, not const.** Mirror `crates/dsl_compiler/tests/vampire_survivors_compile.rs:17-62` (`compile_sim` + `kernel_body_containing`).
```rust
#[test]
fn runtime_config_reads_cfg_not_const() {
    let art = compile_sim(&workspace_path("assets/sim/input_probe.sim")).expect("compiles");
    let body = kernel_body_containing(&art, "DriveX").expect("DriveX kernel");
    assert!(body.contains("cfg.config_probe_drive") || body.contains("cfg.") , "runtime field must read cfg uniform:\n{body}");
    assert!(!body.contains("const config_"), "runtime field must NOT bake a const:\n{body}");
}
```
Run: `cargo test -p dsl_compiler --test runtime_config_emit`. Expected PASS (compiler path already does this).

- [ ] **Step 2: Append the frozen contract block to vampire_survivors.sim.** Add exactly the `config ctl {}` block from the plan index (move_x/move_y/move_level/bolt_level/bolt_rate_level/nova_level/garlic_level/whip_level, all `f32 = 0.0 @runtime`). Do not reference these fields in any rule yet (Plan 3 does that). Run: `cargo build -p sims 2>&1 | tail -5`. Expected: compiles (an unreferenced `@runtime` field is harmless; if the emitter prunes unreferenced consts, that's fine — Plan 3 adds the references).

- [ ] **Step 3: Commit.**
```bash
git add crates/dsl_compiler/tests/runtime_config_emit.rs assets/sim/vampire_survivors.sim
git commit -m "feat(dsl): lock @runtime cfg lowering + add vampire_survivors ctl contract block"
```

## Self-review note
If Task 1 outcome is (c) — already works — Plan 1 collapses to Tasks 1 + 4. Either way the deliverable is identical: `set_*` setters that drive `cfg.<field>` reads, verified by `input_probe_exec`, with the `ctl` contract block ready for Plans 3 and 4.
