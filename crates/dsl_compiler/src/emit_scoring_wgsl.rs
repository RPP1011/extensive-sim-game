//! WGSL emission for scoring — Phase 3 + 6c of the GPU megakernel plan.
//!
//! Companion to [`emit_scoring`]: same table shape
//! (`ScoringEntry { action_head, base, personality_weights,
//! modifier_count, modifiers[8] }`), emitted once as a WGSL storage
//! buffer layout + a `score_entry` WGSL function that mirrors the CPU
//! scorer in [`crate::emit_scoring`].
//!
//! ## What this file produces
//!
//! A single WGSL source string containing:
//!
//!   * The shared SoA storage bindings (agent data, mask bitmaps,
//!     scoring table, scoring output, config uniform) — see the
//!     "Binding layout" section below.
//!   * Per-view storage bindings (5-10) and read snippets produced by
//!     [`crate::emit_view_wgsl::emit_view_read_wgsl`], one per
//!     materialized view (`engaged_with`, `my_enemies`, `threat_level`,
//!     `kin_fear`, `pack_focus`, `rally_boost`). These replace the
//!     Phase 3 `eval_view_call` 0.0-stub; the emitter dispatches on
//!     `pred.field_id` (the runtime VIEW_ID) + the two arg-slot codes
//!     (`ARG_SELF=0`, `ARG_TARGET=1`, `ARG_WILDCARD=0xFE`).
//!   * Helper functions: `read_field`, `compare_scalar`, `eval_view_call`,
//!     `eval_predicate`, `score_entry`.
//!   * The single `cs_scoring` compute entry point: one thread per
//!     agent. Each thread walks `SCORING_TABLE` in order, computing
//!     `(best_action, best_target)` via a deterministic sequential
//!     argmax (lowest `action_head`, then lowest `target_id` wins on
//!     ties).
//!
//! The GPU-side scorer mirrors `crates/engine/src/policy/utility.rs`
//! exactly for every predicate kind today:
//!
//!   * `KIND_ALWAYS` — always true.
//!   * `KIND_SCALAR_COMPARE` — self- or target-side scalar compare
//!     (field_id low range = self; `0x4000 | fid` = target).
//!   * `KIND_VIEW_SCALAR_COMPARE` / `KIND_VIEW_GRADIENT` — now read
//!     real view storage via per-view `view_<name>_get` functions.
//!     Wildcards (`ARG_WILDCARD=0xFE` on slot1) loop over every
//!     attacker slot 0..view_agent_cap and sum the decayed-and-clamped
//!     cell value, mirroring `sum_for_first(a, tick)` on the CPU side.
//!
//! ## Determinism
//!
//! The GPU argmax is a single-thread-per-agent sequential reduce:
//! iterate entries by table order (0..N_ENTRIES), for each entry either
//! evaluate it once (self-only) or walk every alive candidate
//! (target-bound) in ascending slot order. Every time a new score
//! **strictly exceeds** the current best, update; ties preserve the
//! earlier (lower action_head / lower target_id) winner. This gives
//! byte-identical results to the CPU's loop in
//! `crates/engine/src/policy/utility.rs` as long as:
//!
//!   * Both sides walk entries in the same order (`SCORING_TABLE` is
//!     authored deterministically and uploaded verbatim).
//!   * Both sides walk candidates in the same order. The CPU pulls
//!     from `TargetMask::candidates_for(agent, kind)` which is
//!     populated from spatial-hash iteration order; the GPU walks
//!     slot 0..agent_cap and filters by mask bit, giving numerical
//!     slot-ascending order. When the CPU's spatial iteration order
//!     *also* yields slot-ascending (the current
//!     `SimState::spatial().within_radius` implementation iterates
//!     cells in lex-sorted order and within a cell in slot order for
//!     the small-world fixture), the two match byte-exact.
//!   * Stubbed view modifiers don't flip the argmax on the chosen
//!     test fixture. See the backend's parity-test doc for which
//!     fixtures are safe and which are "best effort until Phase 4".
//!
//! ## Binding layout
//!
//! 11 bindings total — under the 16-per-bind-group ceiling on every
//! adapter we care about (Vulkan / Metal / DX12 / GL / LLVMpipe).
//!
//!   * `@binding(0)` `agent_data: array<AgentData>` — packed per-slot
//!     struct carrying every scalar `read_field` needs. Matches
//!     `engine_gpu::scoring::GpuAgentData` 1:1.
//!   * `@binding(1)` `mask_bitmaps: array<u32>` — flat concatenation of
//!     all mask bitmaps, `mask_words_per_bitmap * N_MASKS` u32s in
//!     emitter-fixed order (`MASK_ORDER` below).
//!   * `@binding(2)` `scoring_table: array<ScoringEntryGpu>` — the
//!     `SCORING_TABLE` uploaded verbatim.
//!   * `@binding(3)` `scoring_out: array<ScoreOutput>` — per-agent
//!     `(chosen_action, chosen_target)`. Agent slot `i` writes to
//!     `scoring_out[i]`.
//!   * `@binding(4)` `cfg: ConfigUniform` — radii + table row count +
//!     mask-word count + `tick` + `view_agent_cap` (the latter two are
//!     consumed by the view read snippets emitted alongside).
//!   * `@binding(5)` `view_engaged_with_slots: array<u32>` — slot_map
//!     storage for `engaged_with`; cell value is `AgentId+1` or 0 when
//!     unset.
//!   * `@binding(6)` `view_my_enemies_cells: array<f32>` — pair_map
//!     scalar storage for `my_enemies`, flat row-major
//!     `[observer * N + attacker]`.
//!   * `@binding(7..11)` `view_<name>_cells: array<DecayCell>` — one
//!     binding per pair_map @decay view (`threat_level`, `kin_fear`,
//!     `pack_focus`, `rally_boost`). Each cell is a two-field struct:
//!     `{ value: f32, anchor_tick: u32 }`. The scoring kernel decays
//!     values on read (`value * pow(rate, tick - anchor_tick)`) and
//!     clamps per-view.

use std::fmt::Write;

use crate::emit_view_wgsl::{emit_view_read_wgsl, ViewShape, ViewStorageSpec};

/// Fixed workgroup size. Matches `emit_mask_wgsl::WORKGROUP_SIZE`.
pub const WORKGROUP_SIZE: u32 = 64;

/// Maximum modifier rows per `ScoringEntry`. Must match
/// `engine_rules::scoring::MAX_MODIFIERS` and the emitter's
/// `MAX_MODIFIERS` in `emit_scoring.rs`.
pub const MAX_MODIFIERS: u32 = 8;

/// Number of personality dimensions. Matches
/// `engine_rules::scoring::PERSONALITY_DIMS`.
pub const PERSONALITY_DIMS: u32 = 5;

/// Fixed mask order used by the scoring kernel. The backend uploads
/// bitmaps in this order; the shader reads `mask_bitmaps[mask_idx *
/// mask_words + word]` using an action-head → mask-index table. Kept
/// in sync with `engine_gpu::mask::FUSED_MASK_NAMES`.
///
/// Index = action_head discriminant's mask slot, or `MASK_SLOT_NONE` if
/// the action head has no corresponding mask (e.g. Cast, domain hooks).
pub const MASK_NAMES: &[&str] = &[
    "Attack",     // action_head 3 → mask_idx 0
    "MoveToward", // action_head 1 → mask_idx 1
    "Hold",       // action_head 0 → mask_idx 2
    "Flee",       // action_head 2 → mask_idx 3
    "Eat",        // action_head 7 → mask_idx 4
    "Drink",      // action_head 8 → mask_idx 5
    "Rest",       // action_head 9 → mask_idx 6
];

/// Index returned by `action_head_to_mask_idx` for heads that have no
/// corresponding fused mask bitmap (e.g. `Cast`, `UseItem`, …). Scoring
/// rows tagged with one of these heads are unconditionally skipped at
/// Phase 3 — matching the CPU's `mask.micro_kind[...]` gate, which those
/// heads would only pass under the `mark_domain_hook_micros_allowed`
/// permissive branch, and those rows carry no scoring modifiers in the
/// Phase 3 scope.
pub const MASK_SLOT_NONE: u32 = 0xFFFFu32;

/// Action-head discriminant → `MASK_NAMES` index. Aligned with
/// `engine::mask::MicroKind`:
///
///   0 Hold → 2, 1 MoveToward → 1, 2 Flee → 3, 3 Attack → 0,
///   4 Cast → NONE (Phase 4+), 5 UseItem → NONE, 6 Harvest → NONE,
///   7 Eat → 4, 8 Drink → 5, 9 Rest → 6, 10.. → NONE.
pub fn action_head_to_mask_idx(action_head: u16) -> u32 {
    match action_head {
        0 => 2,             // Hold
        1 => 1,             // MoveToward
        2 => 3,             // Flee
        3 => 0,             // Attack
        7 => 4,             // Eat
        8 => 5,             // Drink
        9 => 6,             // Rest
        _ => MASK_SLOT_NONE, // Cast / UseItem / Harvest / … Phase 4+
    }
}

/// Action-head discriminant → "has target binding" flag. Matches
/// `engine::mask::MicroKind::target_slot().is_some()`. Only Attack and
/// MoveToward carry targets at v1. Kept in sync with that table.
pub fn action_head_is_target_bound(action_head: u16) -> bool {
    matches!(action_head, 1 | 3)
}

/// Emit the scoring WGSL module with the Phase 3 view-call stub
/// (returns 0.0 for every view read). Kept for callers that don't have
/// view storage wired up yet; [`emit_scoring_wgsl_with_views`] is the
/// real Phase 6c entry point.
pub fn emit_scoring_wgsl() -> String {
    emit_scoring_wgsl_with_views(&[])
}

/// Emit the scoring WGSL module, splicing in real per-view read
/// functions produced by [`emit_view_read_wgsl`]. Pass the same
/// `ViewStorageSpec` list the engine_gpu side uses to provision buffers
/// — the emitter reads `spec.view_name` (to map VIEW_ID → fn name) and
/// `spec.shape` (to generate the right call signature).
///
/// Views that appear in [`SCORING_VIEW_IDS`] but not in the given
/// `specs` slice fall back to the 0.0 stub with a loud comment, so a
/// partial wiring doesn't silently break parity.
pub fn emit_scoring_wgsl_with_views(specs: &[ViewStorageSpec]) -> String {
    let mut out = String::new();
    emit_header(&mut out);
    emit_types(&mut out);
    emit_bindings(&mut out);
    emit_view_bindings(&mut out, specs);
    emit_view_read_snippets(&mut out, specs);
    emit_helpers(&mut out);
    emit_read_field(&mut out);
    emit_eval_view_call(&mut out, specs);
    emit_eval_predicate(&mut out);
    emit_score_entry(&mut out);
    emit_kernel(&mut out);
    out
}

/// Scoring-table VIEW_ID constants. Must match
/// `engine_rules::scoring::PredicateDescriptor::VIEW_ID_*`. Kept here
/// so the emitter can dispatch on VIEW_ID without depending on
/// engine_rules (the compiler crate has no such dep, intentionally).
pub const VIEW_ID_THREAT_LEVEL: u16 = 0;
pub const VIEW_ID_MY_ENEMIES: u16 = 1;
pub const VIEW_ID_KIN_FEAR: u16 = 2;
pub const VIEW_ID_PACK_FOCUS: u16 = 3;
pub const VIEW_ID_RALLY_BOOST: u16 = 4;

/// Arg-slot codes — mirror of
/// `engine_rules::scoring::PredicateDescriptor::ARG_*`.
pub const ARG_SELF: u8 = 0;
pub const ARG_TARGET: u8 = 1;
pub const ARG_WILDCARD: u8 = 0xFE;
#[allow(dead_code)]
pub const ARG_NONE: u8 = 0xFF;

/// Map a runtime VIEW_ID to the DSL view name the scoring table
/// references. Keeps the dispatch at [`emit_eval_view_call`] in lockstep
/// with `engine_rules::scoring::PredicateDescriptor::VIEW_ID_*`.
pub fn view_id_to_name(view_id: u16) -> Option<&'static str> {
    match view_id {
        VIEW_ID_THREAT_LEVEL => Some("threat_level"),
        VIEW_ID_MY_ENEMIES => Some("my_enemies"),
        VIEW_ID_KIN_FEAR => Some("kin_fear"),
        VIEW_ID_PACK_FOCUS => Some("pack_focus"),
        VIEW_ID_RALLY_BOOST => Some("rally_boost"),
        _ => None,
    }
}

/// VIEW_IDs the scoring table references. Integration layer builds
/// `ViewStorageSpec`s for these views + `engaged_with` (not scored
/// today but owned by the view storage bind group).
pub const SCORING_VIEW_IDS: &[u16] = &[
    VIEW_ID_THREAT_LEVEL,
    VIEW_ID_MY_ENEMIES,
    VIEW_ID_KIN_FEAR,
    VIEW_ID_PACK_FOCUS,
    VIEW_ID_RALLY_BOOST,
];

fn emit_header(out: &mut String) {
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_wgsl (Phase 3).").unwrap();
    writeln!(out, "// Single compute entry `cs_scoring`; one thread per agent.").unwrap();
    writeln!(out, "// Mirrors engine/src/policy/utility.rs score_entry + argmax.").unwrap();
    writeln!(
        out,
        "// View calls are stubbed (return 0) — wired by Phase 4 (task 185)."
    )
    .unwrap();
    writeln!(out).unwrap();
}

fn emit_types(out: &mut String) {
    // Packed per-agent struct — 16 f32s (64 bytes). Matches
    // `engine_gpu::scoring::GpuAgentData`.
    writeln!(out, "struct Vec3f32 {{ x: f32, y: f32, z: f32 }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "struct AgentData {{").unwrap();
    writeln!(out, "    pos: Vec3f32,").unwrap();
    writeln!(out, "    hp: f32,").unwrap();
    writeln!(out, "    max_hp: f32,").unwrap();
    writeln!(out, "    shield_hp: f32,").unwrap();
    writeln!(out, "    attack_range: f32,").unwrap();
    writeln!(out, "    hunger: f32,").unwrap();
    writeln!(out, "    thirst: f32,").unwrap();
    writeln!(out, "    fatigue: f32,").unwrap();
    writeln!(out, "    alive: u32,").unwrap();
    writeln!(out, "    creature_type: u32,").unwrap();
    // **Precomputed** `hp_pct = hp / max_hp`, populated on the CPU
    // side before upload. Done CPU-side to dodge a precision
    // divergence between strict-IEEE Rust (`80.0_f32 / 100.0_f32 ≈
    // 0x3F4CCCCD`) and relaxed GPU f32 division (some adapters return
    // `0x3F4CCCCC`, one ULP smaller). The 1-ULP gap flips
    // `hp_pct >= 0.8` from true on CPU to false on GPU and breaks
    // scoring parity. By doing the division once on CPU we hand the
    // GPU a pre-rounded value and `read_field(_, _, 2)` becomes a
    // straight memory read.
    writeln!(out, "    hp_pct: f32,").unwrap();
    writeln!(out, "    target_hp_pct_unused: f32,").unwrap();
    // 8 bytes of trailing padding so the struct is 64 bytes — the
    // matching Rust `GpuAgentData` carries the same trailing pad. The
    // backend uploads 64-byte slots; without these the WGSL struct
    // would be 56 bytes and the second slot would read stale data
    // from the first.
    writeln!(out, "    _pad2: u32,").unwrap();
    writeln!(out, "    _pad3: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    // Scoring table entry. Layout-matched to `ScoringEntry` from
    // engine_rules::scoring but laid out in a WGSL-friendly form (the
    // #[repr(C)] CPU struct has non-uniform field sizes which we smear
    // into u32 arrays here).
    //
    // On the CPU, `ScoringEntry` = { action_head:u16, base:f32,
    //   personality_weights:[f32;5], modifier_count:u8,
    //   modifiers:[ModifierRow;8] }. Sizing:
    //     action_head(2) + pad(2) + base(4) + weights(20) + count(1)
    //       + pad(3) + modifiers(8 * 20) = 192 bytes.
    //
    // For WGSL we flatten into a fixed-size layout where every field is
    // at a 4-byte-aligned offset. See `GpuScoringEntry` in the backend.
    let modifiers = MAX_MODIFIERS;
    let personality = PERSONALITY_DIMS;
    writeln!(out, "struct PredicateDescriptor {{").unwrap();
    writeln!(out, "    kind: u32,").unwrap();
    writeln!(out, "    op: u32,").unwrap();
    writeln!(out, "    field_id: u32,").unwrap();
    // Payload is 12 bytes → 3 u32s for alignment convenience.
    writeln!(out, "    payload0: u32,").unwrap();
    writeln!(out, "    payload1: u32,").unwrap();
    writeln!(out, "    payload2: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "struct ModifierRow {{").unwrap();
    writeln!(out, "    predicate: PredicateDescriptor,").unwrap();
    writeln!(out, "    delta: f32,").unwrap();
    writeln!(out, "    _pad: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "struct ScoringEntryGpu {{").unwrap();
    writeln!(out, "    action_head: u32,").unwrap();
    writeln!(out, "    modifier_count: u32,").unwrap();
    writeln!(out, "    base: f32,").unwrap();
    writeln!(out, "    _pad_hdr: u32,").unwrap();
    writeln!(
        out,
        "    personality_weights: array<f32, {personality}>,"
    )
    .unwrap();
    // Trailing pad so the struct ends on 16-byte alignment prior to the
    // modifiers array (each ModifierRow is 32 bytes).
    writeln!(out, "    _pad_after_weights0: f32,").unwrap();
    writeln!(out, "    _pad_after_weights1: f32,").unwrap();
    writeln!(out, "    _pad_after_weights2: f32,").unwrap();
    writeln!(
        out,
        "    modifiers: array<ModifierRow, {modifiers}>,"
    )
    .unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "struct ScoreOutput {{").unwrap();
    writeln!(out, "    chosen_action: u32,").unwrap();
    writeln!(out, "    chosen_target: u32,").unwrap();
    // Pack the winning score for debuggability and determinism
    // crosschecks. The backend doesn't consume it yet; Phase 6 may.
    writeln!(out, "    best_score_bits: u32,").unwrap();
    // Debug slot — stash a probe value during diagnostic runs. Always
    // 0 in production; the backend's parity test asserts equality on
    // it too (so it must match between GPU and CPU). The parity-test
    // CPU path leaves it at 0; if a future diagnostic puts a real
    // value here, the CPU mirror in `cpu_score_outputs` must match.
    writeln!(out, "    debug: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "struct ConfigUniform {{").unwrap();
    writeln!(out, "    combat_attack_range: f32,").unwrap();
    writeln!(out, "    movement_max_move_radius: f32,").unwrap();
    writeln!(out, "    num_entries: u32,").unwrap();
    writeln!(out, "    num_mask_words: u32,").unwrap();
    // Current tick — consumed by view_<name>_get on @decay views for
    // the `tick - anchor_tick` decay math. Uploaded as part of the
    // cfg uniform so no extra binding is needed.
    writeln!(out, "    tick: u32,").unwrap();
    // Agent capacity — the view read snippets reference a WGSL-level
    // `view_agent_cap` symbol (see emit_view_wgsl::emit_pair_map_*),
    // which the emitter generates as a module-scope `const` from this
    // cfg field. Uploaded as u32 so a future non-N² storage layout
    // doesn't require reshaping the uniform.
    writeln!(out, "    view_agent_cap: u32,").unwrap();
    writeln!(out, "    _pad0: u32,").unwrap();
    writeln!(out, "    _pad1: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    // Pair-map @decay cell layout. Matches the scoring kernel's
    // per-view DecayCell buffers — one `f32` base value + one `u32`
    // anchor tick. Callers read `cell.value`, `cell.anchor_tick`; the
    // on-read decay formula is emitted by emit_view_wgsl.
    writeln!(out, "struct DecayCell {{").unwrap();
    writeln!(out, "    value: f32,").unwrap();
    writeln!(out, "    anchor_tick: u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
}

fn emit_bindings(out: &mut String) {
    writeln!(
        out,
        "@group(0) @binding(0) var<storage, read> agent_data: array<AgentData>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(1) var<storage, read> mask_bitmaps: array<u32>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(2) var<storage, read> scoring_table: array<ScoringEntryGpu>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(3) var<storage, read_write> scoring_out: array<ScoreOutput>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(4) var<uniform> cfg: ConfigUniform;"
    )
    .unwrap();
    writeln!(out).unwrap();
}

/// Emit the per-view storage bindings, one binding per view in
/// `specs` (order determined by [`scoring_view_binding_order`]). Each
/// pair_map @decay view gets ONE binding carrying an
/// `array<DecayCell>`; non-decay pair_maps get an `array<f32>`;
/// slot_maps get an `array<u32>`.
///
/// Bindings start at index 5 (right after the 5 core scoring
/// bindings); the integration layer on the engine_gpu side wires
/// buffers in the same order.
fn emit_view_bindings(out: &mut String, specs: &[ViewStorageSpec]) {
    for (i, spec) in scoring_view_binding_order(specs).into_iter().enumerate() {
        let binding = SCORING_CORE_BINDINGS + i as u32;
        let snake = &spec.snake;
        let (wgsl_ty, comment) = match &spec.shape {
            ViewShape::SlotMap { .. } => (
                format!("array<u32>"),
                format!("slot_map storage for `{}`", spec.view_name),
            ),
            ViewShape::PairMapScalar => (
                format!("array<f32>"),
                format!("pair_map<f32> storage for `{}`", spec.view_name),
            ),
            ViewShape::PairMapDecay { rate } => (
                format!("array<DecayCell>"),
                format!(
                    "pair_map<DecayCell> @decay(rate={rate}) storage for `{}`",
                    spec.view_name
                ),
            ),
            ViewShape::Lazy => continue,
        };
        let storage_name = match &spec.shape {
            ViewShape::SlotMap { .. } => format!("view_{snake}_slots"),
            _ => format!("view_{snake}_cells"),
        };
        writeln!(out, "// {comment}").unwrap();
        writeln!(
            out,
            "@group(0) @binding({binding}) var<storage, read> {storage_name}: {wgsl_ty};"
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Emit the per-view `fn view_<snake>_get(...)` functions using
/// [`emit_view_read_wgsl`]. Each snippet assumes:
///   * `view_<snake>_cells` / `view_<snake>_slots` buffer is bound
///     (emit_view_bindings handles that);
///   * a module-scope `view_agent_cap: u32` const is in scope — we
///     emit a `let view_agent_cap = cfg.view_agent_cap;` at every call
///     site via a small shim (emit_view_read_wgsl references
///     `view_agent_cap` as a free symbol).
///
/// Since emit_view_read_wgsl expects `view_agent_cap` as a free symbol
/// and we can't introduce it as a WGSL `const` (the value is only
/// known at dispatch time), we wrap each emitted read function with a
/// one-line helper that shadows the symbol from `cfg`. The simplest
/// wiring is a WGSL-level `let view_agent_cap = cfg.view_agent_cap;`
/// at the top of every read function — but emit_view_read_wgsl emits
/// the function body including that let itself? No, it uses
/// `view_agent_cap` as a direct reference. We emit a wrapper.
fn emit_view_read_snippets(out: &mut String, specs: &[ViewStorageSpec]) {
    // Emit a module-scope accessor to bridge `cfg.view_agent_cap` into
    // the `view_agent_cap` symbol the per-view snippets reference.
    // WGSL doesn't allow reading a uniform from a module-scope `const`
    // initializer, so we emit a small helper function and have our
    // wrapper `view_<name>_get_adapter` evaluate it at call time.
    //
    // Strategy: we rename the emit_view_wgsl function by text-editing
    // (simpler than shadowing): the snippet uses `view_agent_cap` as
    // an identifier, so we do a one-line string substitution to point
    // it at `cfg.view_agent_cap` before splicing the snippet in.
    for spec in scoring_view_binding_order(specs) {
        let snippet = match emit_view_read_wgsl(spec) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Two text rewrites on the upstream emit_view_wgsl snippet:
        //
        // 1. Replace the free `view_agent_cap` symbol with
        //    `cfg.view_agent_cap` (the snippet was authored against a
        //    module-scope const that we don't have — our cap lives in
        //    the cfg uniform).
        //
        // 2. For pair_map @decay views, short-circuit `pow(rate, 0)`
        //    to `1.0` exactly. The integration layer uploads cells
        //    with `anchor_tick = state.tick`, so `dt = 0` on every
        //    read. WGSL's `pow(x, 0)` isn't required to return
        //    bit-exact `1.0` (Vulkan/SPIR-V `OpExtInst Pow` allows ~3
        //    ULPs of error and the implementations we've tested do
        //    drift), so multiplying by it injects 1-ULP noise into
        //    the `value * pow(rate, 0)` result and breaks byte-exact
        //    parity with the CPU-uploaded value. We wrap the pow with
        //    a `select` so dt=0 returns 1.0 exactly.
        //
        // The substring we patch is stable across `render_float_wgsl`
        // outputs because the rate literal contains a `.` (every
        // shipped decay rate is a fraction, and emit_view_wgsl
        // formats them via `f64::to_string` which emits at least one
        // decimal digit). Defensive: if a future rate emitter shape
        // breaks the pattern, the snippet stays correct, just slower
        // by one extra branch — and the parity test catches the
        // regression loudly.
        let rewritten = snippet.replace("view_agent_cap", "cfg.view_agent_cap");
        let rewritten = rewrite_pow_short_circuit(&rewritten);
        out.push_str(&rewritten);
        writeln!(out).unwrap();
    }
}

/// Rewrite every `cell.value * pow(rate, f32(dt))` expression in the
/// snippet to `cell.value * select(pow(rate, f32(dt)), 1.0, dt == 0u)`.
/// This skips the pow op when the integration layer uploads with
/// `anchor_tick == read_tick` (the byte-exact path) — IEEE `pow(x, 0)`
/// is mathematically 1.0 but WGSL doesn't bit-guarantee it.
fn rewrite_pow_short_circuit(src: &str) -> String {
    // The exact pattern from emit_pair_map_decay_read is:
    //   let decayed = cell.value * pow({rate_lit}, f32(dt));
    // We rewrite it to:
    //   let decayed = cell.value * select(pow({rate_lit}, f32(dt)), 1.0, dt == 0u);
    //
    // Only ever appears once per snippet (one decay view per snippet),
    // so a single substring substitution is enough.
    let needle = "let decayed = cell.value * pow(";
    let pos = match src.find(needle) {
        Some(p) => p,
        None => return src.to_string(),
    };
    let after_needle = pos + needle.len();
    // Find the closing `);` of `pow(...);`. The pow argument list is
    // `{rate_lit}, f32(dt)` with no nested parens beyond `f32(dt)`.
    let rest = &src[after_needle..];
    let close_paren = match find_matching_close_paren(rest) {
        Some(p) => p,
        None => return src.to_string(),
    };
    let pow_args = &rest[..close_paren]; // e.g. "0.98, f32(dt)"
    let post = &rest[close_paren + 1..]; // ");\n    ..."
    let mut out = String::with_capacity(src.len() + 64);
    out.push_str(&src[..pos]);
    out.push_str("let decayed = cell.value * select(pow(");
    out.push_str(pow_args);
    out.push_str("), 1.0, dt == 0u)");
    out.push_str(post);
    out
}

/// Walk a WGSL substring tracking paren depth, return the index of the
/// closing `)` that matches an implicit opening at position 0. The
/// substring is assumed to start AFTER an open paren (i.e. depth=1
/// initially); the helper returns the byte index of the matching
/// close. Returns None if the string is unbalanced.
fn find_matching_close_paren(s: &str) -> Option<usize> {
    let mut depth = 1i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Deterministic ordering of views for binding-index assignment. Sort
/// by view name so a re-run produces the same WGSL and the engine_gpu
/// side can bind buffers in matching order without coordinating a
/// separate list.
pub fn scoring_view_binding_order(specs: &[ViewStorageSpec]) -> Vec<&ViewStorageSpec> {
    let mut sorted: Vec<&ViewStorageSpec> = specs
        .iter()
        .filter(|s| !matches!(s.shape, ViewShape::Lazy))
        .collect();
    sorted.sort_by(|a, b| a.view_name.cmp(&b.view_name));
    sorted
}

/// Core scoring bindings (agent_data, mask_bitmaps, scoring_table,
/// scoring_out, cfg) — view bindings start at this index.
pub const SCORING_CORE_BINDINGS: u32 = 5;

fn emit_helpers(out: &mut String) {
    // Predicate-kind + op discriminants. Must match
    // engine_rules::scoring::PredicateDescriptor::KIND_* / OP_* literals.
    writeln!(out, "const KIND_ALWAYS: u32 = 0u;").unwrap();
    writeln!(out, "const KIND_SCALAR_COMPARE: u32 = 1u;").unwrap();
    writeln!(out, "const KIND_GRADIENT: u32 = 6u;").unwrap();
    writeln!(out, "const KIND_VIEW_SCALAR_COMPARE: u32 = 7u;").unwrap();
    writeln!(out, "const KIND_VIEW_GRADIENT: u32 = 8u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const OP_LT: u32 = 0u;").unwrap();
    writeln!(out, "const OP_LE: u32 = 1u;").unwrap();
    writeln!(out, "const OP_EQ: u32 = 2u;").unwrap();
    writeln!(out, "const OP_GE: u32 = 3u;").unwrap();
    writeln!(out, "const OP_GT: u32 = 4u;").unwrap();
    writeln!(out, "const OP_NE: u32 = 5u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const TARGET_FIELD_BASE: u32 = 0x4000u;").unwrap();
    writeln!(out, "const TARGET_FIELD_END: u32 = 0x8000u;").unwrap();
    writeln!(out, "const NO_TARGET: u32 = 0xFFFFFFFFu;").unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn vec3_distance(a: Vec3f32, b: Vec3f32) -> f32 {{\n\
         \x20   let dx = a.x - b.x;\n\
         \x20   let dy = a.y - b.y;\n\
         \x20   let dz = a.z - b.z;\n\
         \x20   return sqrt(dx*dx + dy*dy + dz*dz);\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    // `mask_bit` — read bit `slot` of `mask_idx`-th bitmap.
    writeln!(
        out,
        "fn mask_bit(mask_idx: u32, slot: u32) -> bool {{\n\
         \x20   let base = mask_idx * cfg.num_mask_words;\n\
         \x20   let word = base + (slot / 32u);\n\
         \x20   let bit = slot % 32u;\n\
         \x20   return (mask_bitmaps[word] & (1u << bit)) != 0u;\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    // `action_head_to_mask_idx` — inlined as a big switch. No cross-link
    // with the CPU table because WGSL has no fn-pointer equivalent;
    // keep these in sync by hand (there's a test in the backend that
    // asserts every action head's mask slot matches).
    writeln!(out, "fn action_head_to_mask_idx(ah: u32) -> u32 {{").unwrap();
    writeln!(out, "    switch (ah) {{").unwrap();
    writeln!(out, "        case 0u: {{ return 2u; }}").unwrap(); // Hold
    writeln!(out, "        case 1u: {{ return 1u; }}").unwrap(); // MoveToward
    writeln!(out, "        case 2u: {{ return 3u; }}").unwrap(); // Flee
    writeln!(out, "        case 3u: {{ return 0u; }}").unwrap(); // Attack
    writeln!(out, "        case 7u: {{ return 4u; }}").unwrap(); // Eat
    writeln!(out, "        case 8u: {{ return 5u; }}").unwrap(); // Drink
    writeln!(out, "        case 9u: {{ return 6u; }}").unwrap(); // Rest
    writeln!(out, "        default: {{ return 0xFFFFu; }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn action_head_is_target_bound(ah: u32) -> bool {{\n\
         \x20   return ah == 1u || ah == 3u;\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn compare_scalar(op: u32, lhs: f32, rhs: f32) -> bool {{\n\
         \x20   // NaN short-circuits to false, matching IEEE + CPU semantics.\n\
         \x20   if (lhs != lhs || rhs != rhs) {{ return false; }}\n\
         \x20   switch (op) {{\n\
         \x20       case 0u: {{ return lhs < rhs; }}\n\
         \x20       case 1u: {{ return lhs <= rhs; }}\n\
         \x20       case 2u: {{ return lhs == rhs; }}\n\
         \x20       case 3u: {{ return lhs >= rhs; }}\n\
         \x20       case 4u: {{ return lhs > rhs; }}\n\
         \x20       case 5u: {{ return lhs != rhs; }}\n\
         \x20       default: {{ return false; }}\n\
         \x20   }}\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn f32_from_u32_bits(bits: u32) -> f32 {{\n\
         \x20   return bitcast<f32>(bits);\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();
}

fn emit_read_field(out: &mut String) {
    // Exact match of engine::policy::utility::read_field. Agent-side
    // fields live in the low range (0..8); target-side are offset by
    // TARGET_FIELD_BASE (0x4000) and cover the subset CPU dispatches on.
    // Personality dims (8..=12) return 0 — the CPU side is a constant
    // placeholder there.
    writeln!(
        out,
        "fn read_field(agent_slot: u32, target_slot: u32, field_id: u32) -> f32 {{"
    )
    .unwrap();
    writeln!(out, "    // Target-side range 0x4000..0x8000.").unwrap();
    writeln!(
        out,
        "    if (field_id >= TARGET_FIELD_BASE && field_id < TARGET_FIELD_END) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        if (target_slot == NO_TARGET) {{ return bitcast<f32>(0x7FC00000u); }}"
    )
    .unwrap();
    writeln!(
        out,
        "        let tf = field_id - TARGET_FIELD_BASE;"
    )
    .unwrap();
    writeln!(out, "        let t = agent_data[target_slot];").unwrap();
    writeln!(out, "        switch (tf) {{").unwrap();
    writeln!(out, "            case 0u: {{ return t.hp; }}").unwrap();
    writeln!(out, "            case 1u: {{ return t.max_hp; }}").unwrap();
    // Use the CPU-precomputed hp_pct rather than re-dividing — see
    // the AgentData struct comment for the precision rationale.
    writeln!(out, "            case 2u: {{ return t.hp_pct; }}").unwrap();
    writeln!(out, "            case 3u: {{ return t.shield_hp; }}").unwrap();
    writeln!(
        out,
        "            default: {{ return bitcast<f32>(0x7FC00000u); }}"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    let a = agent_data[agent_slot];").unwrap();
    writeln!(out, "    switch (field_id) {{").unwrap();
    writeln!(out, "        case 0u: {{ return a.hp; }}").unwrap();
    writeln!(out, "        case 1u: {{ return a.max_hp; }}").unwrap();
    // Use the CPU-precomputed hp_pct rather than re-dividing.
    writeln!(out, "        case 2u: {{ return a.hp_pct; }}").unwrap();
    writeln!(out, "        case 3u: {{ return a.shield_hp; }}").unwrap();
    writeln!(out, "        case 4u: {{ return a.attack_range; }}").unwrap();
    writeln!(out, "        case 5u: {{ return a.hunger; }}").unwrap();
    writeln!(out, "        case 6u: {{ return a.thirst; }}").unwrap();
    writeln!(out, "        case 7u: {{ return a.fatigue; }}").unwrap();
    writeln!(
        out,
        "        case 8u: {{ return 0.0; }}"
    )
    .unwrap(); // personality aggression
    writeln!(out, "        case 9u: {{ return 0.0; }}").unwrap(); // social_drive
    writeln!(out, "        case 10u: {{ return 0.0; }}").unwrap(); // ambition
    writeln!(out, "        case 11u: {{ return 0.0; }}").unwrap(); // altruism
    writeln!(out, "        case 12u: {{ return 0.0; }}").unwrap(); // curiosity
    writeln!(
        out,
        "        default: {{ return bitcast<f32>(0x7FC00000u); }}"
    )
    .unwrap(); // NaN
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_eval_view_call(out: &mut String, specs: &[ViewStorageSpec]) {
    // Per-VIEW_ID dispatch into the emitted `view_<name>_get` snippets.
    // Each arm resolves the two arg-slot codes from the predicate
    // payload, fans out to the view's read function, and (for
    // wildcards) loops over every attacker slot to emulate
    // sum_for_first(a, tick) on the CPU side.
    //
    // The `specs` slice determines which VIEW_IDs have real bindings.
    // Any VIEW_ID without a matching spec falls through to the 0.0
    // stub with a comment so the divergence is visible in the emitted
    // source. In practice the engine_gpu integration always passes
    // specs for every VIEW_ID in SCORING_VIEW_IDS; the stub is
    // defensive against partial wiring (e.g. a future task that adds a
    // new VIEW_ID before its storage).
    writeln!(out, "const ARG_SELF: u32 = 0u;").unwrap();
    writeln!(out, "const ARG_TARGET: u32 = 1u;").unwrap();
    writeln!(out, "const ARG_WILDCARD: u32 = 0xFEu;").unwrap();
    writeln!(out, "const ARG_NONE: u32 = 0xFFu;").unwrap();
    writeln!(out).unwrap();

    // Helper: resolve an arg-slot code to an agent slot index.
    // Returns NO_TARGET for "unbound" (ARG_TARGET with no target
    // slot) / "unknown" (any other code). Matches `resolve_slot` in
    // engine/src/policy/utility.rs.
    writeln!(
        out,
        "fn resolve_view_arg(code: u32, agent_slot: u32, target_slot: u32) -> u32 {{\n\
         \x20   if (code == ARG_SELF) {{ return agent_slot; }}\n\
         \x20   if (code == ARG_TARGET) {{ return target_slot; }}\n\
         \x20   return NO_TARGET;\n\
         }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Build a lookup from VIEW_ID → (view name, shape). Only views
    // whose spec is present in `specs` get a real dispatch arm; the
    // rest use the stub.
    let view_by_id: Vec<(u16, &ViewStorageSpec)> = SCORING_VIEW_IDS
        .iter()
        .filter_map(|vid| {
            let name = view_id_to_name(*vid)?;
            specs
                .iter()
                .find(|s| s.view_name == name)
                .map(|s| (*vid, s))
        })
        .collect();

    writeln!(
        out,
        "fn eval_view_call(agent_slot: u32, target_slot: u32, pred: PredicateDescriptor) -> f32 {{"
    )
    .unwrap();
    writeln!(out, "    let slot0 = (pred.payload1) & 0xFFu;").unwrap();
    writeln!(out, "    let slot1 = (pred.payload1 >> 8u) & 0xFFu;").unwrap();
    writeln!(out, "    let view_id = pred.field_id;").unwrap();
    writeln!(out, "    let a = resolve_view_arg(slot0, agent_slot, target_slot);").unwrap();
    writeln!(out, "    if (a == NO_TARGET) {{ return bitcast<f32>(0x7FC00000u); }}").unwrap();
    writeln!(out, "    switch (view_id) {{").unwrap();

    for (vid, spec) in &view_by_id {
        let name = &spec.view_name;
        let snake = &spec.snake;
        writeln!(out, "        case {vid}u: {{").unwrap();
        writeln!(out, "            // VIEW `{name}`").unwrap();
        match &spec.shape {
            ViewShape::PairMapScalar => {
                writeln!(out, "            if (slot1 == ARG_WILDCARD) {{").unwrap();
                writeln!(out, "                var total: f32 = 0.0;").unwrap();
                writeln!(
                    out,
                    "                for (var t: u32 = 0u; t < cfg.view_agent_cap; t = t + 1u) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "                    total = total + view_{snake}_get(a, t);"
                )
                .unwrap();
                writeln!(out, "                }}").unwrap();
                writeln!(out, "                return total;").unwrap();
                writeln!(out, "            }}").unwrap();
                writeln!(
                    out,
                    "            let b = resolve_view_arg(slot1, agent_slot, target_slot);"
                )
                .unwrap();
                writeln!(
                    out,
                    "            if (b == NO_TARGET) {{ return bitcast<f32>(0x7FC00000u); }}"
                )
                .unwrap();
                writeln!(out, "            return view_{snake}_get(a, b);").unwrap();
            }
            ViewShape::PairMapDecay { .. } => {
                writeln!(out, "            if (slot1 == ARG_WILDCARD) {{").unwrap();
                writeln!(out, "                var total: f32 = 0.0;").unwrap();
                writeln!(
                    out,
                    "                for (var t: u32 = 0u; t < cfg.view_agent_cap; t = t + 1u) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "                    total = total + view_{snake}_get(a, t, cfg.tick);"
                )
                .unwrap();
                writeln!(out, "                }}").unwrap();
                writeln!(out, "                return total;").unwrap();
                writeln!(out, "            }}").unwrap();
                writeln!(
                    out,
                    "            let b = resolve_view_arg(slot1, agent_slot, target_slot);"
                )
                .unwrap();
                writeln!(
                    out,
                    "            if (b == NO_TARGET) {{ return bitcast<f32>(0x7FC00000u); }}"
                )
                .unwrap();
                writeln!(
                    out,
                    "            return view_{snake}_get(a, b, cfg.tick);"
                )
                .unwrap();
            }
            ViewShape::SlotMap { .. } => {
                // slot_map views aren't referenced from scoring today
                // but include for completeness — returns the partner
                // slot + 1 as f32 (0 for "no partner"). Scoring tables
                // that care would compare against the encoded value.
                writeln!(
                    out,
                    "            return f32(view_{snake}_get(a));"
                )
                .unwrap();
            }
            ViewShape::Lazy => unreachable!(),
        }
        writeln!(out, "        }}").unwrap();
    }

    // Default arm: unknown or not-wired VIEW_ID.
    writeln!(out, "        default: {{").unwrap();
    writeln!(
        out,
        "            // VIEW_ID not wired — fall back to the 0.0 stub. The"
    )
    .unwrap();
    writeln!(
        out,
        "            // integration layer should register every VIEW_ID in"
    )
    .unwrap();
    writeln!(out, "            // SCORING_VIEW_IDS via emit_scoring_wgsl_with_views.").unwrap();
    writeln!(out, "            return 0.0;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_eval_predicate(out: &mut String) {
    writeln!(
        out,
        "fn eval_predicate(pred: PredicateDescriptor, agent_slot: u32, target_slot: u32) -> bool {{"
    )
    .unwrap();
    writeln!(out, "    switch (pred.kind) {{").unwrap();
    writeln!(out, "        case 0u: {{ return true; }}").unwrap(); // KIND_ALWAYS
    writeln!(out, "        case 1u: {{").unwrap();
    writeln!(
        out,
        "            let lhs = read_field(agent_slot, target_slot, pred.field_id);"
    )
    .unwrap();
    writeln!(
        out,
        "            let rhs = f32_from_u32_bits(pred.payload0);"
    )
    .unwrap();
    writeln!(out, "            return compare_scalar(pred.op, lhs, rhs);").unwrap();
    writeln!(out, "        }}").unwrap(); // KIND_SCALAR_COMPARE
    writeln!(out, "        case 7u: {{").unwrap();
    writeln!(
        out,
        "            let lhs = eval_view_call(agent_slot, target_slot, pred);"
    )
    .unwrap();
    writeln!(
        out,
        "            let rhs = f32_from_u32_bits(pred.payload0);"
    )
    .unwrap();
    writeln!(out, "            return compare_scalar(pred.op, lhs, rhs);").unwrap();
    writeln!(out, "        }}").unwrap(); // KIND_VIEW_SCALAR_COMPARE
    writeln!(out, "        default: {{ return false; }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_score_entry(out: &mut String) {
    // Mirror of CPU's score_entry:
    //
    //   score = base + Σ (personality_weights[i] * personality[i])
    //         + Σ (predicate ? delta : 0) for boolean modifiers
    //         + Σ (view_call * delta)     for gradient modifiers
    //
    // Personality is all-zero at Phase 3 (the CPU `read_personality`
    // placeholder), so the dot-product is a no-op; keeping the line
    // means the shader mirrors the CPU shape 1:1 (a diff shows up if
    // either side starts populating personality without the other).
    writeln!(
        out,
        "fn score_entry(entry_idx: u32, agent_slot: u32, target_slot: u32) -> f32 {{"
    )
    .unwrap();
    writeln!(out, "    let e = scoring_table[entry_idx];").unwrap();
    writeln!(out, "    var score: f32 = e.base;").unwrap();
    // Personality dot-product: agent-side personality is zero for every
    // agent at the Phase 3 scope; the dot-product is skipped here, but
    // the weights are still laid out in GpuScoringEntry so Phase 4+ can
    // wire a real personality vector without reshaping the struct.
    writeln!(
        out,
        "    let n = min(e.modifier_count, {MAX_MODIFIERS}u);"
    )
    .unwrap();
    writeln!(out, "    for (var i: u32 = 0u; i < n; i = i + 1u) {{").unwrap();
    writeln!(out, "        let row = e.modifiers[i];").unwrap();
    writeln!(
        out,
        "        if (row.predicate.kind == KIND_VIEW_GRADIENT) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            let v = eval_view_call(agent_slot, target_slot, row.predicate);"
    )
    .unwrap();
    // CPU uses is_finite; finite(f32) is !NaN && !Inf. Our stub returns
    // 0.0 which is finite, so the add is a no-op on Phase 3.
    writeln!(
        out,
        "            let is_finite_v = (v == v) && (v != bitcast<f32>(0x7F800000u)) && (v != bitcast<f32>(0xFF800000u));"
    )
    .unwrap();
    writeln!(out, "            if (is_finite_v) {{").unwrap();
    writeln!(out, "                score = score + v * row.delta;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            if (eval_predicate(row.predicate, agent_slot, target_slot)) {{"
    )
    .unwrap();
    writeln!(out, "                score = score + row.delta;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    return score;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_kernel(out: &mut String) {
    // Per-agent scoring. Single-thread-per-agent design:
    //   * Walk SCORING_TABLE in order (index = table row).
    //   * For each row: check the mask bit for the row's action_head.
    //     - Skipped rows contribute nothing.
    //     - target-bound rows: walk alive candidate slots 0..n, filter
    //       by the action's mask bit (same bit — the fused kernel sets
    //       the bit iff at least one candidate exists; the target loop
    //       re-filters using the per-mask "target is alive + in-range"
    //       logic via the mask_attack/mask_move_toward predicates,
    //       BUT at Phase 3 we approximate that with alive + hostility
    //       for Attack and alive-and-not-self for MoveToward — same
    //       per-mask DSL predicate shape).
    //     - self-only rows: call score_entry once with target = NO_TARGET.
    //   * Argmax: strictly-greater wins; ties preserve the earlier
    //     (lower entry_idx, lower target_slot) winner. Matches CPU.
    writeln!(
        out,
        "@compute @workgroup_size({WORKGROUP_SIZE})"
    )
    .unwrap();
    writeln!(
        out,
        "fn cs_scoring(@builtin(global_invocation_id) gid: vec3<u32>) {{"
    )
    .unwrap();
    writeln!(out, "    let agent_slot = gid.x;").unwrap();
    writeln!(out, "    let n = arrayLength(&agent_data);").unwrap();
    writeln!(out, "    if (agent_slot >= n) {{ return; }}").unwrap();
    // Dead slot — write a sentinel Hold action (same failsafe as CPU's
    // fallthrough when no mask is set; the backend's readback doesn't
    // care about dead slots).
    writeln!(out, "    if (agent_data[agent_slot].alive == 0u) {{").unwrap();
    writeln!(out, "        scoring_out[agent_slot].chosen_action = 0u;").unwrap();
    writeln!(
        out,
        "        scoring_out[agent_slot].chosen_target = NO_TARGET;"
    )
    .unwrap();
    writeln!(out, "        scoring_out[agent_slot].best_score_bits = 0u;").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Sentinel: no valid best yet.").unwrap();
    writeln!(out, "    var best_score: f32 = 0.0;").unwrap();
    writeln!(out, "    var best_action: u32 = 0u;").unwrap();
    writeln!(out, "    var best_target: u32 = NO_TARGET;").unwrap();
    writeln!(out, "    var found_any: bool = false;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (var e_idx: u32 = 0u; e_idx < cfg.num_entries; e_idx = e_idx + 1u) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        let action_head = scoring_table[e_idx].action_head;"
    )
    .unwrap();
    writeln!(
        out,
        "        let mask_idx = action_head_to_mask_idx(action_head);"
    )
    .unwrap();
    // Skip rows whose action isn't in the fused mask set. Phase 4+
    // domain hooks join the mask kernel and this branch narrows.
    writeln!(out, "        if (mask_idx == 0xFFFFu) {{ continue; }}").unwrap();
    writeln!(
        out,
        "        if (!mask_bit(mask_idx, agent_slot)) {{ continue; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        if (action_head_is_target_bound(action_head)) {{"
    )
    .unwrap();
    writeln!(out, "            // Walk candidate slots in ascending order.").unwrap();
    writeln!(
        out,
        "            for (var t: u32 = 0u; t < n; t = t + 1u) {{"
    )
    .unwrap();
    writeln!(out, "                if (t == agent_slot) {{ continue; }}").unwrap();
    writeln!(
        out,
        "                if (agent_data[t].alive == 0u) {{ continue; }}"
    )
    .unwrap();
    // Radius gate — both Attack and MoveToward have a `from` radius
    // enforced by the fused mask kernel. We re-check here because the
    // per-agent mask bit tells us "at least one candidate exists", not
    // "this specific target is a candidate". For Attack the radius is
    // `combat.attack_range`; for MoveToward it's `movement.max_move_radius`.
    writeln!(out, "                let radius = select(").unwrap();
    writeln!(out, "                    cfg.movement_max_move_radius,").unwrap();
    writeln!(out, "                    cfg.combat_attack_range,").unwrap();
    writeln!(out, "                    action_head == 3u").unwrap();
    writeln!(out, "                );").unwrap();
    writeln!(
        out,
        "                let d = vec3_distance(agent_data[agent_slot].pos, agent_data[t].pos);"
    )
    .unwrap();
    writeln!(
        out,
        "                if (d > radius) {{ continue; }}"
    )
    .unwrap();
    // For Attack we also need hostility, matching the DSL mask's
    // `is_hostile(self, target)` clause. For MoveToward the DSL
    // predicate only requires `target != self && target alive`, which
    // is satisfied by the filters above.
    writeln!(out, "                if (action_head == 3u) {{").unwrap();
    writeln!(
        out,
        "                    let self_ct = agent_data[agent_slot].creature_type;"
    )
    .unwrap();
    writeln!(
        out,
        "                    let tgt_ct = agent_data[t].creature_type;"
    )
    .unwrap();
    writeln!(
        out,
        "                    if (!is_hostile_ct(self_ct, tgt_ct)) {{ continue; }}"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                let s = score_entry(e_idx, agent_slot, t);"
    )
    .unwrap();
    writeln!(out, "                if (!found_any || s > best_score) {{").unwrap();
    writeln!(out, "                    best_score = s;").unwrap();
    writeln!(out, "                    best_action = action_head;").unwrap();
    writeln!(out, "                    best_target = t;").unwrap();
    writeln!(out, "                    found_any = true;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            let s = score_entry(e_idx, agent_slot, NO_TARGET);"
    )
    .unwrap();
    writeln!(out, "            if (!found_any || s > best_score) {{").unwrap();
    writeln!(out, "                best_score = s;").unwrap();
    writeln!(out, "                best_action = action_head;").unwrap();
    writeln!(out, "                best_target = NO_TARGET;").unwrap();
    writeln!(out, "                found_any = true;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    // No mask-allowed row? Fall through to Hold (action_head = 0).
    writeln!(out, "    if (!found_any) {{").unwrap();
    writeln!(out, "        best_action = 0u;").unwrap();
    writeln!(out, "        best_target = NO_TARGET;").unwrap();
    writeln!(out, "        best_score = 0.0;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    scoring_out[agent_slot].chosen_action = best_action;").unwrap();
    writeln!(out, "    scoring_out[agent_slot].chosen_target = best_target;").unwrap();
    writeln!(
        out,
        "    scoring_out[agent_slot].best_score_bits = bitcast<u32>(best_score);"
    )
    .unwrap();
    // No debug probe in production — the scoring kernel writes the
    // canonical (action, target, score, debug=0) per slot. Tests can
    // re-instrument by patching the emitter.
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Inline hostility table. Mirrors is_hostile_to in engine_rules
    // entities. Same symmetric closure as emit_mask_wgsl's is_hostile,
    // but here we take creature-type ordinals directly (no slot lookup).
    writeln!(
        out,
        "fn is_hostile_ct(a: u32, b: u32) -> bool {{\n\
         \x20   // Human<->Wolf, Human<->Dragon, Wolf<->Deer, Wolf<->Dragon, Deer<->Dragon.\n\
         \x20   if (a == 0u && b == 1u) {{ return true; }}\n\
         \x20   if (a == 1u && b == 0u) {{ return true; }}\n\
         \x20   if (a == 0u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 0u) {{ return true; }}\n\
         \x20   if (a == 1u && b == 2u) {{ return true; }}\n\
         \x20   if (a == 2u && b == 1u) {{ return true; }}\n\
         \x20   if (a == 1u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 1u) {{ return true; }}\n\
         \x20   if (a == 2u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 2u) {{ return true; }}\n\
         \x20   return false;\n\
         }}"
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The emitted module is a single WGSL source with one compute entry
    /// named `cs_scoring` and the five storage / uniform bindings the
    /// backend wires. Structural-only; the kernel's runtime behaviour
    /// is exercised by the Phase 3 parity test in `engine_gpu`.
    #[test]
    fn scoring_module_has_expected_shape() {
        let src = emit_scoring_wgsl();

        // One compute entry.
        assert!(
            src.contains("@compute @workgroup_size(64)"),
            "missing workgroup attr:\n{src}"
        );
        assert_eq!(
            src.matches("@compute").count(),
            1,
            "expected exactly one @compute entry:\n{src}"
        );
        assert!(
            src.contains("fn cs_scoring(@builtin(global_invocation_id)"),
            "missing cs_scoring entry point:\n{src}"
        );

        // Five core bindings — agent_data, mask_bitmaps, scoring_table,
        // scoring_out, cfg — all on group 0. View bindings (5..) only
        // appear when emit_scoring_wgsl_with_views receives specs.
        assert!(src.contains("@group(0) @binding(0) var<storage, read> agent_data"));
        assert!(src.contains("@group(0) @binding(1) var<storage, read> mask_bitmaps"));
        assert!(src.contains("@group(0) @binding(2) var<storage, read> scoring_table"));
        assert!(src.contains("@group(0) @binding(3) var<storage, read_write> scoring_out"));
        assert!(src.contains("@group(0) @binding(4) var<uniform> cfg: ConfigUniform"));

        // Argmax hooks + fall-through to Hold when no row fires.
        assert!(
            src.contains("scoring_out[agent_slot].chosen_action = best_action"),
            "missing chosen_action write:\n{src}"
        );
        assert!(
            src.contains("best_action = 0u"),
            "missing Hold fallthrough:\n{src}"
        );

        // With no view specs, the eval_view_call dispatch falls
        // through to the 0.0 default arm.
        assert!(src.contains("return 0.0"));
    }

    /// Action-head mask slot mapping matches the engine's MicroKind
    /// discriminants 1:1 in both directions. Regression guard: if
    /// anyone reshuffles `MASK_NAMES` or the `action_head_to_mask_idx`
    /// switch, the two here should still round-trip for every slot.
    #[test]
    fn action_head_mask_slot_round_trip() {
        // Forward: every mask in MASK_NAMES should have an action head
        // that maps back to its slot.
        let heads = [
            (0u16, "Hold"),
            (1, "MoveToward"),
            (2, "Flee"),
            (3, "Attack"),
            (7, "Eat"),
            (8, "Drink"),
            (9, "Rest"),
        ];
        for (ah, name) in heads {
            let slot = action_head_to_mask_idx(ah);
            assert_ne!(slot, MASK_SLOT_NONE, "head {ah} ({name}) has no mask slot");
            assert_eq!(
                MASK_NAMES[slot as usize], name,
                "head {ah} mapped to slot {slot} whose name is {}",
                MASK_NAMES[slot as usize]
            );
        }

        // Heads that shouldn't have a mask slot (domain hooks + Cast).
        for &ah in &[4u16, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17] {
            assert_eq!(
                action_head_to_mask_idx(ah),
                MASK_SLOT_NONE,
                "head {ah} unexpectedly has a mask slot"
            );
        }
    }

    /// Only Attack / MoveToward are target-bound at v1.
    #[test]
    fn target_bound_flags_match_micro_kind() {
        for ah in 0u16..=17 {
            let expect = ah == 1 || ah == 3;
            assert_eq!(
                action_head_is_target_bound(ah),
                expect,
                "head {ah}"
            );
        }
    }

    /// With view specs passed in, each materialized view spawns a
    /// binding in the emitter-fixed order and the
    /// `view_<name>_get` function body appears verbatim in the module
    /// source. Defensive regression — if the emit_view_wgsl snippets
    /// ever stop producing the expected function name, this fails loud.
    #[test]
    fn wired_views_emit_bindings_and_read_fns() {
        use crate::emit_view_wgsl::{FoldSpec, ViewShape, ViewStorageSpec};

        let specs = vec![
            ViewStorageSpec {
                view_name: "my_enemies".into(),
                snake: "my_enemies".into(),
                shape: ViewShape::PairMapScalar,
                clamp: Some((0.0, 1.0)),
                initial: 0.0,
                folds: vec![FoldSpec {
                    event_name: "AgentAttacked".into(),
                    first_key_field: "target".into(),
                    second_key_field: Some("actor".into()),
                }],
            },
            ViewStorageSpec {
                view_name: "threat_level".into(),
                snake: "threat_level".into(),
                shape: ViewShape::PairMapDecay { rate: 0.98 },
                clamp: Some((0.0, 1000.0)),
                initial: 0.0,
                folds: vec![],
            },
        ];

        let src = emit_scoring_wgsl_with_views(&specs);
        // Binding 5 = first view in sorted-by-name order = my_enemies.
        assert!(
            src.contains("@group(0) @binding(5) var<storage, read> view_my_enemies_cells: array<f32>"),
            "missing my_enemies binding:\n{src}"
        );
        // Binding 6 = threat_level.
        assert!(
            src.contains("@group(0) @binding(6) var<storage, read> view_threat_level_cells: array<DecayCell>"),
            "missing threat_level binding:\n{src}"
        );
        assert!(src.contains("fn view_my_enemies_get("), "missing view_my_enemies_get:\n{src}");
        assert!(src.contains("fn view_threat_level_get("), "missing view_threat_level_get:\n{src}");
        // eval_view_call dispatch has a case for VIEW_ID_MY_ENEMIES (1).
        assert!(src.contains("case 1u:"), "missing VIEW_ID_MY_ENEMIES case:\n{src}");
        // Wildcard loop form.
        assert!(
            src.contains("for (var t: u32 = 0u; t < cfg.view_agent_cap"),
            "missing wildcard loop:\n{src}"
        );
        // cfg.view_agent_cap substitution inside the view read snippet.
        assert!(
            src.contains("let n = cfg.view_agent_cap;"),
            "missing cfg.view_agent_cap rewrite in view snippet:\n{src}"
        );
    }

    /// Round-trip: `view_id_to_name` ↔ `SCORING_VIEW_IDS` stay in sync.
    #[test]
    fn scoring_view_ids_resolve_to_names() {
        for &vid in SCORING_VIEW_IDS {
            assert!(
                view_id_to_name(vid).is_some(),
                "VIEW_ID {vid} in SCORING_VIEW_IDS has no name mapping"
            );
        }
    }
}
