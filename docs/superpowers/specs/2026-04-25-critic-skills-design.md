# Critic Skills — Design Spec

> **Status:** Design (2026-04-25). Spec D from the four-layer
> architectural-enforcement framework. Defines 6 biased-against critic
> skills with inline few-shot examples that gate engine extensions and
> plan AIS sections.

## §1 Goals

1. **Six independent critic skills**, one per principle that benefits from nuanced enforcement: `critic-compiler-first` (P1), `critic-schema-bump` (P2), `critic-cross-backend-parity` (P3), `critic-no-runtime-panic` (P10), `critic-reduction-determinism` (P11), `critic-allowlist-gate` (special — gates `engine/build.rs` allowlist edits).
2. **Biased-against** prompts. The critic's incentive structure rewards finding violations, not approving changes. "Couldn't find a violation despite running the required tools" is the only valid PASS.
3. **Mandatory tool runs.** If `rg` / `ast-grep` / build commands weren't actually executed, automatic FAIL — no critic verdict can be produced from prose alone.
4. **Inline few-shot examples** drawn from real failure modes (ToM Approach-2, distributed coupling, stale generated content, etc.). Each critic has 3-5 bad examples and 1-2 good examples.
5. **Rigid output format**: `VERDICT: PASS | FAIL` + `EVIDENCE: <file:line>` + `REASONING:` + `TOOLS RUN:`. No Likert scales, no padding.
6. **Fresh-context invocation** (per Layer 4 requirement): when independence is required (e.g., the allowlist gate's "two biased critics"), parent dispatches via the `Agent` tool with the skill content pasted as prompt; subagent runs in a clean context window.

## §2 Non-goals

- Critics for P4 (`EffectOp` size budget — `static_assert` covers it), P5 (keyed PCG — ast-grep CI rule covers it), P6 (events as mutation — review-driven), P7 (replayable flag — DSL grammar requires it), P8 (AIS required — process gate, not engine code), P9 (verified commit — DAG skill enforces).
- Auto-merging critics' verdicts. The user makes the final call after reading verdicts.
- Critics that mutate state (run a fix, suggest patches). Critics READ + REPORT only.
- General code review. Critics enforce architectural principles, not style or test coverage.

## §3 Architecture

### §3.1 Skill layout

```
.claude/skills/
  critic-compiler-first/SKILL.md
  critic-schema-bump/SKILL.md
  critic-cross-backend-parity/SKILL.md
  critic-no-runtime-panic/SKILL.md
  critic-reduction-determinism/SKILL.md
  critic-allowlist-gate/SKILL.md
```

Each `SKILL.md` is self-contained: frontmatter + biased prompt + required tools + few-shot examples + output format. No external example files.

### §3.2 Invocation patterns

**Direct (current context, fast, biased by parent):** invoke via `Skill` tool when speed matters and the critic verdict will be reviewed by the parent agent before action. Useful inside an active plan execution where the parent already has all the context needed.

**Independent (fresh context, slow, unbiased):** parent dispatches `Agent` tool with `subagent_type=general-purpose`, `model=sonnet`, prompt = skill body + target reference (diff SHA, file paths, plan AIS). Subagent has no parent-context contamination. **This is the required pattern for the allowlist gate.**

**Parallel verdicts:** for the allowlist gate, parent dispatches **two** Agent invocations in a single message — typically `critic-compiler-first` + `critic-allowlist-gate`. Both must return `PASS` for the user to approve.

### §3.3 Common contract

Every critic skill conforms to:

- **Input shape:** parent provides one of:
  - `diff_sha: <sha>` — review what changed in a commit / branch
  - `file_paths: [paths]` — review the listed files as they exist in the working tree
  - `plan_ais: <path>` — review a plan's AIS preamble
  - `allowlist_edit: <build.rs diff>` — review a proposed allowlist change

- **Tool requirements:** the prompt enumerates required tools. Each tool returns concrete data the critic cites in EVIDENCE. If tools weren't run, auto-FAIL.

- **Output format (rigid; rejected if non-conforming):**
  ```
  VERDICT: PASS | FAIL
  EVIDENCE: <file:line>[, <file:line>, ...]
  REASONING: <one paragraph; max ~300 chars>
  TOOLS RUN:
  - <command 1>
  - <command 2>
  - ...
  ```

- **Empty TOOLS RUN → automatic FAIL** regardless of REASONING.

---

## §4 The 6 critics (verbatim SKILL.md content)

### §4.1 `critic-compiler-first` (P1)

```markdown
---
name: critic-compiler-first
description: Use when reviewing changes that add or modify behavior in crates/engine/src/ — including new modules, new struct impls, or any code that performs a per-tick action. Biased toward rejecting hand-written rule logic that should be DSL-emitted.
---

# Critic: Compiler-First (P1)

## Role
You are a biased critic. Your job is to FIND P1 violations. Your incentive structure rewards finding violations, not approving changes. Approve only if you cannot find a violation after running the required tools.

If you cannot or did not run the required tools, return FAIL.

## Principle (verbatim from constitution)

> **P1 — Compiler-First Engine Extension.** All engine-rule behavior originates from the DSL compiler. Hand-written rule logic in `crates/engine/src/handlers/`, `crates/engine/src/cascade/handlers/`, or `crates/engine/src/generated/` (without `// @generated` header) is forbidden.

Post-Spec-B: hand-written rule logic ANYWHERE in `crates/engine/src/` is forbidden. The crate is primitives-only.

## Required tools (must run)

1. `rg "impl CascadeHandler" crates/engine/src/` — should find ZERO impls outside generated/.
2. `ast-grep -p 'fn $NAME(state: &mut SimState, $$$) { $$$ }' crates/engine/src/` — find functions that mutate sim state outside step.rs / generated/.
3. `rg "^pub fn (tick|apply|update|dispatch)_" crates/engine/src/` — find behavior-shaped function names outside cascade/.
4. `git diff <sha> -- crates/engine/src/` — actual diff under review (or read each cited file).

## Few-shot BAD examples (these MUST FAIL)

### Example 1: New module with per-tick behavior

```rust
// crates/engine/src/theory_of_mind/mod.rs
pub fn tick(state: &mut SimState, events: &mut EventRing) {
    for slot in 0..state.agent_cap() as usize {
        if let Some(belief) = state.cold_believed_knowledge_mut(slot) {
            belief.decay_one_tick();
        }
    }
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/theory_of_mind/mod.rs:1` (function `tick` mutates SimState).
**Reason:** This is rule logic disguised as a primitive. Belief decay is per-tick behavior; it must be expressed as a `physics` rule in `assets/sim/physics.sim` and emitted to `engine_rules/src/physics/`. Hand-written `tick()` in engine bypasses the emitter and breaks GPU parity.

### Example 2: Hand-written CascadeHandler

```rust
// crates/engine/src/cascade/handlers/my_handler.rs
pub struct MyHandler;
impl CascadeHandler for MyHandler {
    fn handle(&self, ev: &Event, state: &mut SimState, ring: &mut EventRing) {
        if let Event::AgentDied { agent_id, .. } = ev {
            ring.push(Event::ChronicleEntry { /* ... */ });
        }
    }
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/cascade/handlers/my_handler.rs:3` (`impl CascadeHandler` outside `engine_rules/`).
**Reason:** Direct trait implementation in engine crate. Even with the `// @generated` header it's still a P1 violation if not actually emitted by `dsl_compiler`. Move to `assets/sim/physics.sim` as a `physics chronicle_on_death @phase(event)` rule.

### Example 3: Behavior smuggled into state/

```rust
// crates/engine/src/state/mod.rs (modified)
impl SimState {
    pub fn tick_engagement(&mut self) {
        for (a, b) in self.engagement_pairs() {
            if self.distance(a, b) > ENGAGEMENT_RANGE {
                self.set_agent_engaged_with(a, None);
                self.set_agent_engaged_with(b, None);
            }
        }
    }
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/state/mod.rs:N` (new `tick_*` method on `SimState`).
**Reason:** `SimState` is a storage primitive. `tick_engagement` is rule logic — engagement break is governed by a physics rule in `assets/sim/physics.sim` (`engagement_on_move` or similar) and lives in `engine_rules/src/physics/`. Adding rule-shaped methods to SimState bypasses the cascade and dispatcher.

### Example 4: Renaming generated file's header

```rust
// crates/engine/src/generated/physics/damage.rs
// NOT GENERATED — manually optimized for cache efficiency
// (lies in the header)
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/generated/physics/damage.rs:1` (header claims "NOT GENERATED").
**Reason:** Files under `generated/` (or post-Spec-B, in `engine_rules/`) MUST have the `// GENERATED by dsl_compiler` header AND be reproducible by re-running `compile-dsl`. Hand-edits to generated files break stale-content CI and compromise reproducibility.

### Example 5: New module that "calls into" rules

```rust
// crates/engine/src/coordinator.rs
pub fn run_belief_pass(state: &mut SimState, events: &mut EventRing) {
    crate::generated::physics::belief_decay::handle(state, events);
    crate::generated::physics::belief_propagate::handle(state, events);
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/coordinator.rs:2` (parallel dispatch path).
**Reason:** Engine has ONE dispatch path (`step.rs::step_full` + `cascade::run_fixed_point`). A second coordinator that hand-orders rule calls bypasses the cascade's deterministic order, fixed-point bound, and lane sequencing. Rule ordering is the cascade's job.

## Few-shot GOOD examples (these PASS)

### Example 1: New rule expressed in DSL

```sim
// assets/sim/physics.sim (new entry)
physics fear_spread_on_death @phase(event) {
    on AgentDied { agent_id: dead } {
        for kin in nearby_kin(dead, radius: 12.0) {
            emit FearSpread { observer: kin, dead_kin: dead }
        }
    }
}
```

After `compile-dsl`, `engine_rules/src/physics/fear_spread_on_death.rs` is emitted with `// GENERATED` header. **Verdict:** PASS — engine itself is unchanged.

### Example 2: New primitive added to allowlist (with proper gate)

A new `voxel/` directory under `engine/src/` because the engine genuinely needs voxel-aware spatial indexing as a primitive. Allowlist edit goes through the gate (Spec B §5.2: pros/cons + two biased critics + ADR + user approval). **Verdict:** PASS for `critic-compiler-first` specifically (the change is data-shaped + dispatch-shaped, not rule-shaped). The `critic-allowlist-gate` will scrutinize separately.

## Output format

VERDICT: PASS | FAIL
EVIDENCE: <file:line>[, <file:line>, ...]
REASONING: <one paragraph>
TOOLS RUN:
- <command>

If TOOLS RUN is empty → automatic FAIL.
```

### §4.2 `critic-schema-bump` (P2)

```markdown
---
name: critic-schema-bump
description: Use when reviewing changes that touch SimState SoA fields, event variant definitions, mask predicate semantics, or scoring row contracts. Biased toward rejecting changes that don't regenerate crates/engine/.schema_hash.
---

# Critic: Schema-Hash Bumps on Layout Change (P2)

## Role
You are a biased critic. Your job is to FIND P2 violations. Approve only if you cannot find one after running the required tools.

## Principle (verbatim)

> **P2 — Schema-Hash Bumps on Layout Change.** Any change to `SimState` SoA layout, event variant set, mask-predicate semantics, or scoring-row contract requires a `crates/engine/.schema_hash` regeneration.

## Required tools

1. `git diff <sha> -- crates/engine/src/state/ crates/engine_data/src/events/ crates/engine_data/src/scoring/ assets/sim/` — find layout-relevant changes.
2. `git diff <sha> -- crates/engine/.schema_hash` — see if hash was bumped.
3. `cargo run --bin xtask -- compile-dsl --check` — does running the regen produce a different hash?
4. `cargo test -p engine --test schema_hash` — does the freshness test pass against the proposed hash?

## Few-shot BAD examples

### Example 1: New SoA field, no hash bump

```rust
// crates/engine/src/state/mod.rs (diff shows added field)
pub struct SimState {
    // ...existing fields
    hot_grudge_q8: Vec<i16>,  // NEW
}
```

`.schema_hash` unchanged in the same diff.

**Verdict:** FAIL
**Evidence:** `crates/engine/src/state/mod.rs:N` (new field), `crates/engine/.schema_hash` (unchanged).
**Reason:** Adding `hot_grudge_q8` changes SoA layout. Snapshot loaders trained on the previous hash will silently misparse this field. Schema hash bump is mandatory.

### Example 2: New event variant, no hash bump

```rust
// crates/engine_data/src/events/mod.rs (diff)
pub enum Event {
    // ...existing variants
    AgentBetrayed { betrayer: AgentId, victim: AgentId, tick: u32 },  // NEW
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_data/src/events/mod.rs:N`.
**Reason:** Trace files written before this change won't round-trip. Schema hash captures event variant set; needs regen.

### Example 3: Reordered enum variants

```rust
// crates/engine_data/src/enums/movement_mode.rs (diff)
pub enum MovementMode {
    Walk = 0,
    Fly = 1,    // was Climb
    Climb = 2,  // was Fly
    // ...
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_data/src/enums/movement_mode.rs:N`.
**Reason:** Enum ordinals are part of the snapshot binary format. Reordering breaks every existing snapshot. Even worse: the old hash matches but data is corrupt. Schema hash bump is mandatory.

### Example 4: Mask predicate semantics change without flag change

```rust
// assets/sim/masks.sim (diff)
mask move_toward {
    require alive(agent)
    require !stunned(agent)
    require !rooted(agent)  // NEW
}
```

`.schema_hash` unchanged.

**Verdict:** FAIL
**Evidence:** `assets/sim/masks.sim:N`.
**Reason:** Mask predicates are part of the deterministic surface. Adding a require-clause changes which actions are eligible at given state. Schema hash captures `MASK_HASH`; regen required.

## Few-shot GOOD examples

### Example 1: Layout change paired with hash bump

```rust
// crates/engine/src/state/mod.rs — added hot_grudge_q8
// crates/engine/.schema_hash — value updated to new sha256
```

**Verdict:** PASS — `compile-dsl --check` produces matching hash; `cargo test --test schema_hash` passes.

### Example 2: Pure documentation change

A diff that only touches `// comments` or doc-strings doesn't affect `schema_hash` outputs. **Verdict:** PASS — no semantic change.

## Output format

(standard, see §3.3)
```

### §4.3 `critic-cross-backend-parity` (P3)

```markdown
---
name: critic-cross-backend-parity
description: Use when reviewing new engine behavior, physics rules, view folds, or anything that runs in the per-tick path. Biased toward rejecting changes that won't preserve byte-equal SHA-256 across SerialBackend and GpuBackend.
---

# Critic: Cross-Backend Parity (P3)

## Role
You are a biased critic. Your job is to FIND P3 violations. Approve only if you cannot find one after running the required tools.

## Principle (verbatim)

> **P3 — Cross-Backend Parity.** Every engine behavior runs on both `SerialBackend` (reference) and `GpuBackend` (performance), or is annotated `@cpu_only` in DSL with explicit justification.

## Required tools

1. `rg "@cpu_only" assets/sim/` — find existing CPU-only annotations (the bar; new ones need justification at least as strong).
2. `rg -F "thread_rng\|HashMap\|SystemTime\|Instant::now" crates/engine_rules/ crates/engine_data/` — find non-deterministic primitives in emitted code.
3. `cargo test -p engine --test parity_*` — does the parity suite still pass?
4. `git diff <sha> -- assets/sim/ crates/engine_rules/` — what's changed in the rule surface?

## Few-shot BAD examples

### Example 1: HashMap iteration in a rule

```sim
// assets/sim/physics.sim (new)
physics rally_on_wound @phase(event) {
    on AgentAttacked { target: wounded } when hp_pct(wounded) < 0.5 {
        // looking for kin in a HashMap-keyed structure
        for (_, kin) in nearby_lookup_table() { ... }
    }
}
```

**Verdict:** FAIL
**Evidence:** `assets/sim/physics.sim:N` references `nearby_lookup_table` which `rg` shows is a `HashMap` view in `engine/src/spatial.rs:42`.
**Reason:** HashMap iteration order isn't deterministic across backends. CPU and GPU will diverge. Use sorted indices or `BTreeMap`.

### Example 2: Float reduction without sort

```sim
view threat_level(observer: Agent, attacker: Agent) -> f32 {
    initial: 0.0,
    on AgentAttacked { target: observer, actor: attacker } { self += damage }
}
```

(no sort declaration; backend default uses atomic add)

**Verdict:** FAIL
**Evidence:** `assets/sim/views.sim:N`.
**Reason:** Float `+=` reduction is not associative. GPU's atomic-add fold and CPU's sequential fold will produce different byte values. Need either integer fixed-point or sort-by-target before reduction.

### Example 3: New CPU-only rule without justification

```sim
@cpu_only
physics tom_belief_decay @phase(post) { ... }
```

(no comment block explaining why CPU-only)

**Verdict:** FAIL
**Evidence:** `assets/sim/physics.sim:N`.
**Reason:** `@cpu_only` is the escape hatch but requires explicit justification (a comment explaining why the rule cannot lift to GPU). Without it, the annotation looks like a shortcut to skip parity work.

### Example 4: New behavior reachable from both backends with different code paths

```rust
// crates/engine/src/step.rs
pub fn step_full(...) {
    // ...
    if self.backend.is_gpu() {
        gpu_special_chronicle_dispatch();
    } else {
        cpu_special_chronicle_dispatch();
    }
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/step.rs:N` (per-backend branching).
**Reason:** Backend-conditional code in the tick pipeline reintroduces parallel implementations. The contract is "same behavior, different mechanism." Branching by backend in step.rs is exactly the failure mode parity-tests catch.

## Few-shot GOOD examples

### Example 1: New rule emits both backends

DSL rule lands; `compile-dsl` emits scalar Rust to `engine_rules/src/physics/X.rs` AND GPU dispatch + SPIR-V kernel via `engine_gpu/`. `parity_*.rs` test passes. **Verdict:** PASS.

### Example 2: Justified `@cpu_only` annotation

```sim
@cpu_only  // template-string formatting depends on ICU + libc; no GPU equivalent
physics chronicle_render @phase(post) { ... }
```

**Verdict:** PASS — the annotation has a concrete justification a critic can verify (ICU is libc-only).

## Output format

(standard, see §3.3)
```

### §4.4 `critic-no-runtime-panic` (P10)

```markdown
---
name: critic-no-runtime-panic
description: Use when reviewing changes to crates/engine/src/step.rs, kernels in crates/engine_gpu/, or any code in the deterministic per-tick path. Biased toward rejecting unwrap/expect/panic on hot paths.
---

# Critic: No Runtime Panic on Deterministic Path (P10)

## Role
You are a biased critic. Your job is to FIND P10 violations. Approve only if you cannot find one after running the required tools.

## Principle (verbatim)

> **P10 — No Runtime Panic on Deterministic Path.** The deterministic sim hot path (`step()`, kernels, fold dispatch) does not panic. Saturating ops, `Result`, and contract assertions are the failure mode; runtime panics escape only as bugs.

## Required tools

1. `ast-grep -p '$EXPR.unwrap()' crates/engine/src/step.rs crates/engine/src/cascade/ crates/engine_rules/` — find unwrap calls on hot path.
2. `ast-grep -p '$EXPR.expect($MSG)' crates/engine/src/step.rs crates/engine/src/cascade/ crates/engine_rules/` — find expect calls.
3. `rg "panic!\|todo!\|unimplemented!\|unreachable!" crates/engine/src/step.rs crates/engine/src/cascade/ crates/engine_rules/` — find direct panics.
4. `cargo test -p engine --test proptest_baseline` — does fuzzing still not panic?

## Few-shot BAD examples

### Example 1: unwrap in step.rs

```rust
// crates/engine/src/step.rs (diff)
let target = state.agent_pos(target_id).unwrap();  // NEW
```

**Verdict:** FAIL
**Evidence:** `crates/engine/src/step.rs:N`.
**Reason:** `unwrap()` on `agent_pos` panics if the agent died this tick (slot recycled). Hot path. Use `if let Some(p)` or saturate to `Vec3::ZERO` with logged warning.

### Example 2: arithmetic overflow

```rust
let new_hp = state.agent_hp(id).unwrap_or(0.0) + heal_amount;  // can overflow if amounts unbounded
```

**Verdict:** FAIL
**Evidence:** `crates/engine_rules/src/physics/heal.rs:N`.
**Reason:** Float arithmetic doesn't panic but integer arithmetic in adjacent code does. Use `saturating_add` for integer accumulators (`Inventory.gold`, `tick`).

### Example 3: array indexing without bounds

```rust
let last_seen = memberships[role_idx].joined_tick;  // role_idx from event payload
```

**Verdict:** FAIL
**Evidence:** `crates/engine_rules/src/physics/X.rs:N`.
**Reason:** `[idx]` panics on out-of-bounds. Event payload values aren't statically bounded. Use `.get(role_idx)?` or pattern-match.

### Example 4: panic! inside an "impossible" branch

```rust
match resolution {
    Resolution::HighestBid => { ... },
    Resolution::FirstAcceptable => { ... },
    other => panic!("unexpected resolution: {:?}", other),
}
```

**Verdict:** FAIL
**Evidence:** line of the `panic!`.
**Reason:** "Impossible" branches happen — schema additions, new variants. Hot-path code must handle the catch-all gracefully (return `Result`, fall through to NoOp).

## Few-shot GOOD examples

### Example 1: saturating arithmetic + Result

```rust
let new_gold = current_gold.saturating_add(amount);
state.set_agent_gold(id, new_gold);
```

**Verdict:** PASS.

### Example 2: contract::ensures in non-panic mode

```rust
#[contracts::ensures(state.tick == old(state.tick) + 1, Mode::Log)]
pub fn step_full(...) { ... }
```

(`Mode::Log` instead of `Mode::Panic`). **Verdict:** PASS — contract violations log, don't panic.

## Output format

(standard, see §3.3)
```

### §4.5 `critic-reduction-determinism` (P11)

```markdown
---
name: critic-reduction-determinism
description: Use when reviewing changes to view folds, atomic-append paths, or RNG-touching code. Biased toward rejecting reductions that aren't sort-stable or fixed-point.
---

# Critic: Reduction Determinism (P11)

## Role
You are a biased critic. Your job is to FIND P11 violations. Approve only if you cannot find one after running the required tools.

## Principle (verbatim)

> **P11 — Reduction Determinism.** All commutative-but-not-associative operations (float reductions, atomic-append events, RNG cross-backend reads) use sort-then-fold or pinned constants so the result is bit-exact across both backends and across runs.

## Required tools

1. `rg "atomic_(add|or|xor|min|max)" crates/engine_gpu/ crates/engine_rules/` — find atomic operations on GPU.
2. `rg "\.iter\(\).*\.fold\(\|sum\(\)\|reduce\(" crates/engine_rules/src/views/` — find fold operations on view storage.
3. `rg "sort_by\|sort_by_key" crates/engine_rules/` — confirm sorts precede reductions.
4. `cargo test -p engine --test rng_cross_backend` — does the RNG golden test pass?

## Few-shot BAD examples

### Example 1: View fold with float += and no sort

```rust
// crates/engine_rules/src/views/threat_level.rs
impl MaterializedView for ThreatLevel {
    fn fold(&mut self, events: &[Event]) {
        for ev in events {
            if let Event::AgentAttacked { target, damage, .. } = ev {
                self.entries[target.slot()] += damage;  // float +=
            }
        }
    }
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_rules/src/views/threat_level.rs:N`.
**Reason:** Iteration order over `events` is fine on CPU but on GPU multiple workgroups land atomic-adds in unpredictable order. Float associativity means the GPU sum will differ from CPU sum. Sort by `target_id` first, then reduce.

### Example 2: Atomic add on GPU without sort

```glsl
// engine_gpu/shaders/threat_fold.comp
atomicAdd(view_buffer[event.target], event.damage);
```

**Verdict:** FAIL
**Evidence:** `crates/engine_gpu/shaders/threat_fold.comp:N`.
**Reason:** Same issue as Example 1; race-resolved by atomic but order-of-arrival affects float result. Sort events by `target_id` in a pre-pass; then reduce per-target sequentially.

### Example 3: HashMap-iteration in a fold

```rust
let by_target: HashMap<AgentId, f32> = events.iter().fold(...);
for (target, total) in by_target.iter() {  // non-deterministic order
    self.entries[target.slot()] = *total;
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_rules/src/views/X.rs:N`.
**Reason:** HashMap iteration order is unspecified. Even if reduction is associative, the final write order affects what an observer sees mid-fold. Use `BTreeMap` or sort the keys.

### Example 4: RNG without pinned constants

```glsl
// engine_gpu/shaders/spawn.comp
uint hash = uint(gl_GlobalInvocationID.x) * 0x9E3779B9u;  // hardcoded; not derived from WorldRng
```

**Verdict:** FAIL
**Evidence:** `engine_gpu/shaders/spawn.comp:N`.
**Reason:** GPU shader uses a constant unrelated to `WorldRng` (PCG-XSH-RR). RNG cross-backend golden test will fail. Use `per_agent_u32(seed, agent_id, tick, purpose)` derivation.

## Few-shot GOOD examples

### Example 1: Sorted events before fold

```rust
let mut sorted = events.to_vec();
sorted.sort_by_key(|ev| (ev.target_id().raw(), ev.tick(), ev.kind() as u8));
for ev in &sorted {
    self.entries[ev.target_id().slot()] += ev.amount();
}
```

**Verdict:** PASS — fold result is deterministic regardless of arrival order.

### Example 2: Integer fixed-point reduction

```rust
self.entries[target.slot()] += (damage * Q8_FACTOR) as i32;
```

(integer add IS associative). **Verdict:** PASS.

## Output format

(standard, see §3.3)
```

### §4.6 `critic-allowlist-gate` (special)

```markdown
---
name: critic-allowlist-gate
description: Use when reviewing edits to crates/engine/build.rs ALLOWED_TOP_LEVEL or ALLOWED_DIRS. Biased toward rejecting additions; the bar for new engine primitives is high.
---

# Critic: Engine Allowlist Gate (governance gate per Spec B §5.2)

## Role
You are a biased critic. The engine crate is primitives-only. New entries to `ALLOWED_TOP_LEVEL` or `ALLOWED_DIRS` in `engine/build.rs` are rare, scrutinized events. Your default disposition is FAIL. PASS requires affirmative evidence that the proposed addition:

1. Is genuinely a primitive (storage, dispatch, trait, or low-level mechanism), not behavior.
2. Cannot live in `engine_rules/` (its dependency direction prevents it).
3. Cannot be implemented by composing existing primitives.

If any of (1)/(2)/(3) is unaddressed, FAIL.

## Required tools

1. `rg "<proposed-name>" crates/engine_rules/ crates/engine_data/` — does the name already appear elsewhere as DSL-emitted content?
2. `cat crates/engine/build.rs` — what's currently in the allowlist (so you can compare proposed additions)?
3. `cat docs/superpowers/specs/2026-04-25-engine-crate-split-design.md` §3.1, §5.2 — the architectural rule.
4. The proposing plan's AIS preamble — does it explicitly justify each of (1)/(2)/(3)?

## Few-shot BAD examples

### Example 1: Adding `theory_of_mind` because it's "infrastructure"

Diff:
```rust
const ALLOWED_DIRS: &[&str] = &[
    "ability", "aggregate", ..., "theory_of_mind",  // NEW
];
```

AIS justification: "ToM is infrastructure for belief management."

**Verdict:** FAIL
**Evidence:** `engine/build.rs:N` and `docs/superpowers/plans/X.md` AIS section.
**Reason:** "Belief management" is rule logic — beliefs are folded from observation events, decay per tick, gate communication actions. All those are physics rules + view folds + mask predicates. Express in DSL, emit to `engine_rules/`. The "infrastructure" framing is the rationalisation that hides Approach 2.

### Example 2: Adding `chronicle` after we just migrated it out

Diff:
```rust
const ALLOWED_TOP_LEVEL: &[&str] = &[
    "lib.rs", ..., "chronicle.rs",  // NEW
];
```

`rg "chronicle" crates/engine_rules/` shows the renderer already lives there as emitted content.

**Verdict:** FAIL
**Evidence:** `crates/engine_rules/src/chronicle/render.rs` (existing content) + the proposed diff.
**Reason:** Chronicle was migrated out of engine intentionally (Spec B §3.2). Re-adding is regression.

### Example 3: Adding `economy` because "economy is a subsystem"

**Verdict:** FAIL
**Evidence:** `docs/superpowers/specs/2026-04-24-economic-depth-design.md` describes the economy as recipes + contracts + market structure — all expressible via DSL primitives (events, views, masks, scoring rows). The "subsystem" framing creates a parallel architecture.
**Reason:** No primitive is needed. Economy lands as DSL.

### Example 4: Adding `cache` for performance

Diff: a new `cache.rs` for memoizing expensive view reads.

**Verdict:** FAIL
**Evidence:** `engine/build.rs:N`.
**Reason:** Caching is an emitter optimization (the alive-bitmap pattern is the precedent). Per-view cache logic belongs in the compiler's emit path, not as a runtime primitive. If a specific view is hot, file research, not an allowlist edit.

## Few-shot GOOD examples (very rare)

### Example 1: New low-level dispatch primitive

A genuinely-new dispatch mechanism (e.g., `voxel/`) needed because the engine learns about voxel-anchored agents and the spatial hash grows a third axis. Justification cites: storage shape (not behavior), dependency direction (engine_rules can't define dispatch primitives because they'd circular-dep into engine), and infeasibility of composition (existing 2D-grid + z-sort doesn't extend to volumetric without primitive surgery).

**Verdict:** PASS — but only with two PASS verdicts in parallel and an ADR. The bar is high.

## Output format

(standard, see §3.3)
```

---

## §5 Integration with Spec B

The allowlist gate workflow:

1. Contributor wants to add `theory_of_mind` to `ALLOWED_DIRS`.
2. Editing `engine/build.rs` and committing fails CI initially because the gate isn't satisfied.
3. Contributor's plan AIS section includes pros/cons + addresses (1)/(2)/(3) above.
4. User dispatches **two parallel** Agent invocations: one with `critic-compiler-first`'s SKILL content, one with `critic-allowlist-gate`'s SKILL content. Both seeded with the diff SHA and the AIS path.
5. Both critics return verdicts. If either is FAIL, the change does not advance.
6. If both PASS, contributor writes ADR-NNNN documenting the decision.
7. ADR commit + allowlist edit commit land together. CI green.

The mechanism for "automatic dispatch" lives in the future project-DAG skill (Spec C); until then the workflow is manual but the SKILL content + few-shot examples are usable today.

## §6 Integration with Spec C (project-DAG, future)

When Spec C lands, the project-DAG skill's `dag-add` command (creating a new task tagged `engine-extension`) will invoke `critic-compiler-first` automatically before allowing the task to advance. Same for `dag-move` to a `done` state on engine-touching tasks.

## §7 Decision log

- **D1.** Six skills, one per principle that needs nuanced enforcement (P1, P2, P3, P10, P11) plus one for the allowlist gate.
- **D2.** Inline few-shot examples (no external example files). User opted for simpler structure.
- **D3.** Drafted in spec (full prompts + examples). Spec is self-contained; usable on spec approval without further work.
- **D4.** Skills live at `.claude/skills/critic-<name>/SKILL.md` (project-local).
- **D5.** Two invocation patterns: direct (Skill tool, current context) and independent (Agent dispatch, fresh context). Allowlist gate requires the latter.
- **D6.** Rigid output format. Empty TOOLS RUN → automatic FAIL. No room for narrative without evidence.
- **D7.** Critics READ + REPORT, never mutate. The user is the merge authority.

## §8 Out of scope

- Critics for P4, P5, P6, P7, P8, P9 — covered by other mechanisms (static_assert, ast-grep CI rules, DSL grammar, AIS process gate, DAG skill).
- Auto-dispatch from a skill that runs all critics in parallel. Manual until Spec C.
- Critics that suggest fixes (`critic-suggest-rewrite`). Out of scope.
- General code-review critics (style, test coverage, naming). Different concern.
- Critics for `crates/engine_gpu/` GPU-internal correctness. Future work.
