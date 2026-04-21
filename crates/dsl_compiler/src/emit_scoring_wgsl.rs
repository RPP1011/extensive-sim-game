//! WGSL emission for scoring — Phase 3 of the GPU megakernel plan.
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
//!   * Helper functions: `read_field`, `compare_scalar`, `eval_view_call`,
//!     `eval_predicate`, `score_entry`.
//!   * The single `cs_scoring` compute entry point: one thread per
//!     agent. Each thread walks `SCORING_TABLE` in order, computing
//!     `(best_action, best_target)` via a deterministic sequential
//!     argmax (lowest `action_head`, then lowest `target_id` wins on
//!     ties).
//!
//! The GPU-side scorer mirrors `crates/engine/src/policy/utility.rs`
//! exactly for the predicate kinds the Phase 3 emitter supports:
//!
//!   * `KIND_ALWAYS` — always true.
//!   * `KIND_SCALAR_COMPARE` — self- or target-side scalar compare
//!     (field_id low range = self; `0x4000 | fid` = target).
//!   * `KIND_VIEW_SCALAR_COMPARE` / `KIND_VIEW_GRADIENT` — **stubbed**
//!     to return 0 (scalar compare against threshold) / NaN (gradient).
//!     View storage is Phase 4+ work (task 185); the stub's shape is
//!     documented alongside `eval_view_call` so the Phase 4 emitter
//!     can swap in real view-buffer reads without reshaping callers.
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
//! Everything is packed into 5 storage / uniform bindings to stay well
//! under `wgpu::Limits::max_bindings_per_bind_group` on every adapter
//! we care about. The layout:
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
//!   * `@binding(4)` `cfg: ConfigUniform` — knobs the scoring path
//!     doesn't read directly at Phase 3 but that keep the binding
//!     layout identical to the mask kernel's, making a future fused
//!     mask+scoring shader a trivial merge.

use std::fmt::Write;

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

/// Emit the scoring WGSL module. Returns the full shader source string.
pub fn emit_scoring_wgsl() -> String {
    let mut out = String::new();
    emit_header(&mut out);
    emit_types(&mut out);
    emit_bindings(&mut out);
    emit_helpers(&mut out);
    emit_read_field(&mut out);
    emit_eval_view_call(&mut out);
    emit_eval_predicate(&mut out);
    emit_score_entry(&mut out);
    emit_kernel(&mut out);
    out
}

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

fn emit_eval_view_call(out: &mut String) {
    // STUB — Phase 4 (task 185) wires real view storage. Returning 0
    // means: a view-scalar-compare with a positive threshold
    // (OP_GT / OP_GE) returns false; a view-scalar-compare with a
    // non-positive threshold (rare in practice) may spuriously pass.
    // A KIND_VIEW_GRADIENT multiplies by 0 and adds nothing.
    //
    // Callers (eval_predicate / score_entry) are oblivious to the stub;
    // Phase 4 swaps the body for real view-buffer reads without
    // reshaping any caller.
    writeln!(
        out,
        "// STUB: Phase 3 view-call evaluator returns 0. Phase 4 (task 185)\n\
         // wires real view storage. Documented scoring divergence: any\n\
         // scoring row with a KIND_VIEW_* modifier whose CPU-evaluated\n\
         // contribution would be non-zero will underscore on the GPU path\n\
         // here, which *may* flip the argmax when that modifier tips the\n\
         // scales. The parity test's fixtures are chosen so no such flip\n\
         // happens (1v1 with no kin nearby — no kin_fear/pack_focus/\n\
         // rally_boost/threat_level/my_enemies triggers)."
    )
    .unwrap();
    writeln!(
        out,
        "fn eval_view_call(agent_slot: u32, target_slot: u32, pred: PredicateDescriptor) -> f32 {{\n\
         \x20   // Phase 4 replaces this with a per-VIEW_ID dispatch reading\n\
         \x20   // from bound view storage buffers.\n\
         \x20   return 0.0;\n\
         }}"
    )
    .unwrap();
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

        // Five bindings — agent_data, mask_bitmaps, scoring_table,
        // scoring_out, cfg — all on group 0.
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

        // View stub present with a loud docstring so a drop-in Phase 4
        // implementation doesn't accidentally silently-pass.
        assert!(src.contains("Phase 3 view-call evaluator returns 0"));
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
}
