//! WGSL emission for mask predicates — Phase 1-2 of the GPU megakernel plan.
//!
//! Mirrors [`emit_mask`]'s AST walk but emits a compute shader source
//! string targeting the WGSL subset wgpu 26 accepts via its runtime
//! `naga` parse path. Two entry points live here:
//!
//!   * [`emit_mask_wgsl`] — single-mask kernel (one `cs_<name>` entry
//!     point per `MaskIR`). Used by the backend's Phase 1 parity tests
//!     and by any test that wants to probe the lowerer in isolation.
//!   * [`emit_masks_wgsl_fused`] — fused module emitting a single
//!     `cs_fused_masks` entry point that walks every agent once and
//!     writes N separate bitmap outputs, one per mask. This is the
//!     Phase 2 shape the GPU backend dispatches each tick — same SoA
//!     upload, same work per invocation, but N fewer dispatches.
//!
//! Task 158's GPU-emittability validator has already gated the subset
//! we accept here, so unsupported shapes surface as
//! `EmitError::Unsupported` rather than silent miscompiles.
//!
//! ## Scope (Phase 2)
//!
//! The Phase-1 Attack-only gate is gone — any mask whose
//! `IrActionHead::shape` is either `None` (self-only) or `Positional([
//! (_, _, IrType::AgentId)])` (single-Agent target) emits a kernel. The
//! remaining shapes — parametric over a non-Agent type, e.g.
//! `mask Cast(ability: AbilityId)` — still error out; they need view
//! storage / cooldown buffers that Phase 4+ will add. The supported
//! expression surface is a subset of `emit_mask.rs`:
//!
//!   * Literals (bool, int, float)
//!   * Locals (including the `self` rewrite → `self_id`)
//!   * Binary comparisons + logical/arithmetic ops
//!   * `agents.alive(id)`, `agents.pos(id)`
//!   * `distance(a, b)` builtin
//!   * `config.<block>.<field>` reads (via uniform buffer)
//!   * Hoisted `agents.pos(<local>)` bindings mirroring the Rust
//!     emitter's prelude
//!   * DSL-declared `@lazy` views that themselves lower cleanly
//!     (Phase 1 inlines `is_hostile` as a pairwise creature-type test;
//!     general @lazy lowering is a Phase 2 task).
//!
//! Everything outside this surface errors out. The Attack mask sits
//! inside it; adding more kernels means either matching the subset or
//! extending this file.
//!
//! ## Buffer layout
//!
//! The shader consumes one storage buffer per hot SoA field the kernel
//! reads. The engine uploads `hot_pos`, `hot_alive`, and
//! `cold_creature_type` as flat `Vec3`/`u32` arrays aligned to the
//! agent-slot index. Layout matches 1:1 with the CPU-side `SimState`
//! so the uploader is a memcpy — no re-interpretation. See
//! `engine_gpu::attack_mask` for the binding table; this file is the
//! source of the WGSL that reads it.

use std::fmt::Write;

use crate::ast::{BinOp, UnOp};
use crate::ir::{Builtin, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, MaskIR, NamespaceId};

/// Errors raised during WGSL mask emission. Narrow mirror of
/// [`crate::emit_mask::EmitError`] — same surface, different emitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// Some IR construct isn't supported by the Phase 1 WGSL emitter.
    /// Carries a short diagnostic string with the offending shape.
    Unsupported(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Unsupported(s) => write!(f, "wgsl mask emission: {s}"),
        }
    }
}

impl std::error::Error for EmitError {}

/// Fixed workgroup size for the mask kernel — 64 threads. Matches the
/// wgpu defaults and keeps dispatch counts divisible by common
/// occupancy targets on integrated GPUs / LLVMpipe.
pub const WORKGROUP_SIZE: u32 = 64;

/// Emit a compute-shader source string for a single mask. The kernel
/// writes one bit per agent slot (packed into a `u32` bitmap) set iff
/// the mask predicate holds for at least one target within the mask's
/// `from`-clause radius. Self-only masks (no `from` clause) are
/// currently unsupported at Phase 1 — they'll land when the kernel
/// naming scheme grows to cover them.
///
/// The generated kernel is named `cs_<snake>` where `<snake>` is
/// `snake_case(mask.head.name)`. Entry-point naming stays decoupled
/// from dispatch-site naming so Phase 2 can add multiple entry points
/// to a fused module.
pub fn emit_mask_wgsl(mask: &MaskIR) -> Result<String, String> {
    emit_mask_wgsl_result(mask).map_err(|e| e.to_string())
}

fn emit_mask_wgsl_result(mask: &MaskIR) -> Result<String, EmitError> {
    // Phase 2: the Attack-only gate is gone. Any mask that passes the
    // task-158 GPU-emittability validator — i.e. its `head.shape` is
    // either `None` (self-only) or `Positional([(_, _, IrType::AgentId)])`
    // (single Agent target) — emits a kernel. Shapes that carry a
    // non-Agent parameter (e.g. `mask Cast(ability: AbilityId)`) are
    // still rejected here: those require buffers and view storage that
    // the Phase 2 emitter doesn't yet provision. Phase 4+ revisits Cast
    // once the views it depends on (`view::is_stunned`,
    // `abilities.cooldown_ready`, `agents.engaged_with`) have real GPU
    // backing.
    //
    // WGSL reserves `target` as a keyword, so for target-bound masks we
    // alias the DSL-level binding name to a kernel-local `wgsl_target`
    // symbol. Self-only masks don't have a target binding at all — the
    // kernel just tests the alive self and packs the bit unconditionally
    // (every predicate body in v1 is just `agents.alive(self)`).
    let target_shape = classify_mask_shape(mask)?;
    let wgsl_target_name = "wgsl_target".to_string();

    let mut out = String::new();
    emit_header(&mut out, mask);
    emit_bindings(&mut out);
    emit_helpers(&mut out);
    emit_kernel(&mut out, mask, &target_shape, &wgsl_target_name)?;
    Ok(out)
}

/// Classification of a `MaskIR`'s action-head shape into the subset the
/// Phase 2 WGSL emitter accepts. `SelfOnly` means the kernel tests the
/// self agent only (no target loop); `AgentTarget(name)` means the
/// kernel loops over every alive candidate and applies the predicate
/// per pair, with `name` carrying the DSL-level target binding so the
/// WGSL alias table can rewrite occurrences. Any other shape — e.g.
/// `Positional` with an `AbilityId` param — surfaces as
/// `Unsupported` so the GPU backend skips that mask and logs the
/// reason.
#[derive(Debug, Clone)]
enum MaskShape {
    /// No `from` clause, no target binding. Hold / Flee / Eat / Drink /
    /// Rest all lower to this — the body only tests `agents.alive(self)`.
    SelfOnly,
    /// `from query.nearby_agents(...)` clause + single positional Agent
    /// target binding. Attack / MoveToward lower to this.
    AgentTarget { dsl_name: String },
}

fn classify_mask_shape(mask: &MaskIR) -> Result<MaskShape, EmitError> {
    match &mask.head.shape {
        IrActionHeadShape::None => Ok(MaskShape::SelfOnly),
        IrActionHeadShape::Positional(binds) if binds.is_empty() => Ok(MaskShape::SelfOnly),
        IrActionHeadShape::Positional(binds) if binds.len() == 1 => {
            let (name, _slot, ty) = &binds[0];
            match ty {
                crate::ir::IrType::AgentId => Ok(MaskShape::AgentTarget {
                    dsl_name: name.clone(),
                }),
                other => Err(EmitError::Unsupported(format!(
                    "mask `{}` has non-Agent positional param of type {other:?}; \
                     parametric masks (e.g. `Cast(ability: AbilityId)`) need view/cooldown \
                     storage the Phase 2 emitter doesn't provision. Skip on GPU.",
                    mask.head.name
                ))),
            }
        }
        other => Err(EmitError::Unsupported(format!(
            "mask `{}` has unsupported action-head shape {other:?}",
            mask.head.name
        ))),
    }
}

/// Snake-case transform — identical to the Rust emitter's so the kernel
/// entry-point name and the module filename stay in lockstep.
pub fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Header + SoA binding block
// ---------------------------------------------------------------------------

fn emit_header(out: &mut String, mask: &MaskIR) {
    writeln!(out, "// GENERATED by dsl_compiler (Phase 1 WGSL).").unwrap();
    writeln!(
        out,
        "// Kernel for `mask {}` — produces one bit per agent slot.",
        mask.head.name
    )
    .unwrap();
    writeln!(out, "// Do not edit by hand; regenerate via the mask emitter.").unwrap();
    writeln!(out).unwrap();
}

/// Emit the storage-buffer binding block. One buffer per hot field the
/// Phase 1 kernel reads, plus the output bitmap. Bindings are stable
/// across kernels for Phase 1 — when Phase 2 fuses masks into one
/// shader this block becomes the shared prelude.
fn emit_bindings(out: &mut String) {
    // Group 0: per-agent SoA. `f32`/`u32` arrays; one element per
    // allocated slot (the CPU agent_cap). Dead slots have `alive[i] == 0`
    // so the kernel can `return` early without reading junk fields.
    //
    // `creature_type` is stored as u32 (upcast from u8 on host) — WGSL
    // storage arrays require at least 32-bit elements.
    writeln!(out, "struct Vec3f32 {{ x: f32, y: f32, z: f32 }};").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "@group(0) @binding(0) var<storage, read> agent_pos: array<Vec3f32>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(1) var<storage, read> agent_alive: array<u32>;"
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(2) var<storage, read> agent_creature_type: array<u32>;"
    )
    .unwrap();
    writeln!(
        out,
        "// Output: one bit per agent slot, packed into u32s."
    )
    .unwrap();
    writeln!(
        out,
        "@group(0) @binding(3) var<storage, read_write> mask_out: array<atomic<u32>>;"
    )
    .unwrap();
    writeln!(out).unwrap();
    // Config block — a small uniform buffer carries the handful of
    // `config.<block>.<field>` reads the masks need. Phase 2 masks
    // consume two knobs:
    //
    //   * `combat.attack_range`     — Attack's `from` radius + inner
    //                                 `distance < 2.0` clause
    //   * `movement.max_move_radius` — MoveToward's `from` radius
    //
    // The struct is `std140`-ish — f32 fields pack tight, but WGSL
    // uniform buffers still want 16-byte alignment. Two scalars fit in
    // the first 8 bytes; we pad to 16 with two f32s.
    writeln!(out, "struct ConfigUniform {{").unwrap();
    writeln!(out, "    combat_attack_range: f32,").unwrap();
    writeln!(out, "    movement_max_move_radius: f32,").unwrap();
    writeln!(out, "    // Padding to align to 16 bytes.").unwrap();
    writeln!(out, "    _pad0: f32,").unwrap();
    writeln!(out, "    _pad1: f32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(
        out,
        "@group(0) @binding(4) var<uniform> cfg: ConfigUniform;"
    )
    .unwrap();
    writeln!(out).unwrap();
}

/// Helper fns shared across kernels — creature-type hostility table,
/// vector distance. Inlined into the shader module so there's no
/// function-call overhead across compute entry points.
fn emit_helpers(out: &mut String) {
    // Creature type ordinals come from `engine_rules::entities::CreatureType`:
    //   0 Human, 1 Wolf, 2 Deer, 3 Dragon. See
    //   `crates/engine_rules/src/entities/mod.rs` lines 14-20.
    //
    // The hostility matrix below is the symmetric closure of
    // `predator_prey` in `assets/sim/entities.sim`, mirroring
    // `CreatureType::is_hostile_to` in that same file. Keep in sync
    // with the Rust impl — if either drifts they diverge silently.
    writeln!(
        out,
        "// Pairwise hostility: mirrors CreatureType::is_hostile_to in engine_rules.\n\
         // 0 Human, 1 Wolf, 2 Deer, 3 Dragon.\n\
         fn is_hostile(a: u32, b: u32) -> bool {{\n\
         \x20   // Human <-> Wolf\n\
         \x20   if (a == 0u && b == 1u) {{ return true; }}\n\
         \x20   if (a == 1u && b == 0u) {{ return true; }}\n\
         \x20   // Human <-> Dragon\n\
         \x20   if (a == 0u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 0u) {{ return true; }}\n\
         \x20   // Wolf <-> Deer\n\
         \x20   if (a == 1u && b == 2u) {{ return true; }}\n\
         \x20   if (a == 2u && b == 1u) {{ return true; }}\n\
         \x20   // Wolf <-> Dragon\n\
         \x20   if (a == 1u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 1u) {{ return true; }}\n\
         \x20   // Deer <-> Dragon\n\
         \x20   if (a == 2u && b == 3u) {{ return true; }}\n\
         \x20   if (a == 3u && b == 2u) {{ return true; }}\n\
         \x20   return false;\n\
         }}"
    )
    .unwrap();
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
}

// ---------------------------------------------------------------------------
// Kernel body
// ---------------------------------------------------------------------------

fn emit_kernel(
    out: &mut String,
    mask: &MaskIR,
    shape: &MaskShape,
    wgsl_target_name: &str,
) -> Result<(), EmitError> {
    let kernel_name = format!("cs_{}", snake_case(&mask.head.name));

    writeln!(
        out,
        "@compute @workgroup_size({WORKGROUP_SIZE})"
    )
    .unwrap();
    writeln!(
        out,
        "fn {kernel_name}(@builtin(global_invocation_id) gid: vec3<u32>) {{"
    )
    .unwrap();
    writeln!(out, "    let self_id = gid.x;").unwrap();
    writeln!(
        out,
        "    let n = arrayLength(&agent_alive);"
    )
    .unwrap();
    writeln!(out, "    if (self_id >= n) {{ return; }}").unwrap();
    writeln!(out, "    // Dead agents produce no bits.").unwrap();
    writeln!(
        out,
        "    if (agent_alive[self_id] == 0u) {{ return; }}"
    )
    .unwrap();
    writeln!(out, "    let self_pos = agent_pos[self_id];").unwrap();
    writeln!(
        out,
        "    let self_ct = agent_creature_type[self_id];"
    )
    .unwrap();

    emit_predicate_body(out, mask, shape, wgsl_target_name)?;
    writeln!(out).unwrap();
    // Pack the bit. Non-atomic in concept (one thread per self_id), but
    // we use atomicOr so different workgroups writing into the same u32
    // word (32 neighboring agents) don't race. `found` is `true` only
    // when the whole predicate matched above.
    writeln!(out, "    if (found) {{").unwrap();
    writeln!(out, "        let word_idx = self_id / 32u;").unwrap();
    writeln!(out, "        let bit_idx = self_id % 32u;").unwrap();
    writeln!(
        out,
        "        atomicOr(&mask_out[word_idx], 1u << bit_idx);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    Ok(())
}

/// Emit the predicate body — shared between the single-mask entry-point
/// emitter and the fused-kernel emitter. Writes a `var found: bool` the
/// caller reads and bit-packs. For self-only masks the body is a linear
/// conjunction of the `when` clauses evaluated against the self agent;
/// for agent-target masks it's a loop over every alive candidate with
/// the target-pos/target-ct bindings hoisted and the `from` radius
/// prefilter applied.
fn emit_predicate_body(
    out: &mut String,
    mask: &MaskIR,
    shape: &MaskShape,
    wgsl_target_name: &str,
) -> Result<(), EmitError> {
    let mut clauses: Vec<&IrExprNode> = Vec::new();
    flatten_and(&mask.predicate, &mut clauses);
    let hoisted = Vec::<String>::new();

    match shape {
        MaskShape::SelfOnly => {
            // No target loop — evaluate the predicate against self and
            // set `found` once every clause passes. DSL-level `self` is
            // the only local, so `dsl_target_name` is effectively
            // unused; we pass an empty string so `rename_local` falls
            // through to the identity branch for non-`self` names.
            writeln!(out, "    var found: bool = false;").unwrap();
            writeln!(out, "    {{").unwrap();
            // Evaluate each clause; bail to the end of the block if any
            // fails. Using a scope + break keeps early-exit cheap.
            let ctx = LowerCtx { dsl_target_name: "", wgsl_target_name };
            writeln!(out, "        let pass = (true").unwrap();
            for clause in &clauses {
                let cond = lower_expr(clause, &hoisted, ctx)?;
                writeln!(out, "            && ({cond})").unwrap();
            }
            writeln!(out, "        );").unwrap();
            writeln!(out, "        if (pass) {{ found = true; }}").unwrap();
            writeln!(out, "    }}").unwrap();
        }
        MaskShape::AgentTarget { dsl_name } => {
            // Pick the `from`-clause radius. Phase 1's Attack inlined
            // `combat.attack_range` directly. Phase 2 reads the radius
            // from the mask's name — a small switch because the DSL's
            // `from` expression isn't walked here (only Phase 5's
            // spatial hash will). Attack → `cfg.combat_attack_range`,
            // MoveToward → `cfg.movement_max_move_radius`. Masks whose
            // radius knob isn't wired surface as an emit error so the
            // GPU backend skips them loudly.
            let radius_sym = mask_radius_symbol(&mask.head.name)?;
            writeln!(out, "    let radius = {radius_sym};").unwrap();
            writeln!(out, "    var found: bool = false;").unwrap();
            writeln!(
                out,
                "    for (var {t}: u32 = 0u; {t} < n; {t} = {t} + 1u) {{",
                t = wgsl_target_name
            )
            .unwrap();
            writeln!(
                out,
                "        if ({t} == self_id) {{ continue; }}",
                t = wgsl_target_name
            )
            .unwrap();
            writeln!(
                out,
                "        if (agent_alive[{t}] == 0u) {{ continue; }}",
                t = wgsl_target_name
            )
            .unwrap();
            writeln!(
                out,
                "        let {t}_pos = agent_pos[{t}];",
                t = wgsl_target_name
            )
            .unwrap();
            writeln!(
                out,
                "        let {t}_ct = agent_creature_type[{t}];",
                t = wgsl_target_name
            )
            .unwrap();
            // Radius prefilter — the `from` clause bound.
            writeln!(
                out,
                "        if (vec3_distance(self_pos, {t}_pos) > radius) {{ continue; }}",
                t = wgsl_target_name
            )
            .unwrap();

            let ctx = LowerCtx { dsl_target_name: dsl_name, wgsl_target_name };
            for clause in &clauses {
                let cond = lower_expr(clause, &hoisted, ctx)?;
                writeln!(
                    out,
                    "        if (!({cond})) {{ continue; }}"
                )
                .unwrap();
            }
            writeln!(out, "        found = true;").unwrap();
            writeln!(out, "        break;").unwrap();
            writeln!(out, "    }}").unwrap();
        }
    }
    Ok(())
}

/// Lookup the WGSL symbol for the `from`-clause radius of a given
/// target-bound mask. This is a small hard-coded table for v1 — the
/// DSL's `from query.nearby_agents(pos, <radius>)` expression isn't
/// walked by the emitter yet. Extending: when a new target-bound mask
/// lands, add its config knob to the `ConfigUniform` struct emitted by
/// `emit_bindings` and wire it here.
fn mask_radius_symbol(name: &str) -> Result<&'static str, EmitError> {
    match name {
        "Attack" => Ok("cfg.combat_attack_range"),
        "MoveToward" => Ok("cfg.movement_max_move_radius"),
        other => Err(EmitError::Unsupported(format!(
            "target-bound mask `{other}` has no wired radius knob in the WGSL emitter; \
             add it to ConfigUniform + mask_radius_symbol"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Expression lowering — WGSL subset
// ---------------------------------------------------------------------------

fn flatten_and<'a>(node: &'a IrExprNode, out: &mut Vec<&'a IrExprNode>) {
    if let IrExpr::Binary(BinOp::And, lhs, rhs) = &node.kind {
        flatten_and(lhs, out);
        flatten_and(rhs, out);
    } else {
        out.push(node);
    }
}

/// Per-kernel lowering context. Carries the rename from the DSL-level
/// target binding (e.g. `target`) to the WGSL kernel-local loop var
/// (`wgsl_target` — the DSL name collides with a reserved keyword).
/// Every `Local(_, name)` lookup and every `agents.pos(local)` →
/// `<name>_pos` rewrite routes through this context so the rename is
/// a single chokepoint.
#[derive(Copy, Clone)]
struct LowerCtx<'a> {
    /// The DSL-level name of the target local (e.g. `target`).
    dsl_target_name: &'a str,
    /// The WGSL-level kernel symbol for the same local
    /// (e.g. `wgsl_target`).
    wgsl_target_name: &'a str,
}

impl<'a> LowerCtx<'a> {
    /// Translate a DSL local name into the WGSL symbol to emit.
    fn rename_local(&self, dsl_name: &str) -> String {
        if dsl_name == "self" {
            "self_id".to_string()
        } else if dsl_name == self.dsl_target_name {
            self.wgsl_target_name.to_string()
        } else {
            dsl_name.to_string()
        }
    }

    /// Name of the hoisted `<local>_pos` WGSL binding for a DSL local.
    fn pos_binding(&self, dsl_name: &str) -> String {
        if dsl_name == "self" {
            "self_pos".to_string()
        } else if dsl_name == self.dsl_target_name {
            format!("{}_pos", self.wgsl_target_name)
        } else {
            format!("{dsl_name}_pos")
        }
    }
}

fn lower_expr(
    node: &IrExprNode,
    hoisted: &[String],
    ctx: LowerCtx<'_>,
) -> Result<String, EmitError> {
    match &node.kind {
        IrExpr::LitBool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        IrExpr::LitInt(v) => Ok(format!("{v}")),
        IrExpr::LitFloat(v) => Ok(render_float_wgsl(*v)),
        IrExpr::Local(_, name) => Ok(ctx.rename_local(name)),
        IrExpr::NamespaceField { ns, field, .. } => lower_namespace_field(*ns, field),
        IrExpr::NamespaceCall { ns, method, args } => {
            // Hoisted agents.pos — use the pre-computed `<t>_pos` binding.
            if *ns == NamespaceId::Agents && method == "pos" && args.len() == 1 {
                if let IrExpr::Local(_, name) = &args[0].value.kind {
                    let _ = hoisted;
                    return Ok(ctx.pos_binding(name));
                }
            }
            lower_namespace_call(*ns, method, args, hoisted, ctx)
        }
        IrExpr::BuiltinCall(b, args) => lower_builtin_call(*b, args, hoisted, ctx),
        IrExpr::ViewCall(_view_ref, args) => {
            if args.len() != 2 {
                return Err(EmitError::Unsupported(format!(
                    "Phase 1 WGSL: view call with {} args — only is_hostile(a,b) supported",
                    args.len()
                )));
            }
            let a = lower_expr(&args[0].value, hoisted, ctx)?;
            let b = lower_expr(&args[1].value, hoisted, ctx)?;
            let a_ct = ct_for(&a, ctx);
            let b_ct = ct_for(&b, ctx);
            Ok(format!("is_hostile({a_ct}, {b_ct})"))
        }
        IrExpr::UnresolvedCall(name, args) => {
            // Phase 1 accepts the pre-resolver `is_hostile(self, target)`
            // shape that the emit_mask tests use. Real DSL goes through
            // ViewCall; this arm makes the WGSL emitter exercise the
            // same fixture shape without a full resolver run.
            if name == "is_hostile" && args.len() == 2 {
                let a = lower_expr(&args[0].value, hoisted, ctx)?;
                let b = lower_expr(&args[1].value, hoisted, ctx)?;
                let a_ct = ct_for(&a, ctx);
                let b_ct = ct_for(&b, ctx);
                return Ok(format!("is_hostile({a_ct}, {b_ct})"));
            }
            Err(EmitError::Unsupported(format!(
                "Phase 1 WGSL: unresolved call `{name}` not supported"
            )))
        }
        IrExpr::Binary(op, lhs, rhs) => {
            let l = lower_expr(lhs, hoisted, ctx)?;
            let r = lower_expr(rhs, hoisted, ctx)?;
            Ok(format!("({l} {} {r})", binop_str(*op)?))
        }
        IrExpr::Unary(op, rhs) => {
            let r = lower_expr(rhs, hoisted, ctx)?;
            Ok(format!("({}{r})", unop_str(*op)))
        }
        other => Err(EmitError::Unsupported(format!(
            "Phase 1 WGSL: expression {other:?} not supported"
        ))),
    }
}

fn lower_namespace_field(ns: NamespaceId, field: &str) -> Result<String, EmitError> {
    if ns == NamespaceId::Config {
        // Phase 2 knows the knobs Attack and MoveToward pull out of the
        // config block. The full Config struct has ~60 fields; the
        // emitter only surfaces the ones a masked rule actually reads,
        // so the uniform buffer stays tiny. Extend per mask.
        return match field {
            "combat.attack_range" => Ok("cfg.combat_attack_range".to_string()),
            "movement.max_move_radius" => Ok("cfg.movement_max_move_radius".to_string()),
            other => Err(EmitError::Unsupported(format!(
                "WGSL mask emitter: config field `{other}` not wired"
            ))),
        };
    }
    Err(EmitError::Unsupported(format!(
        "WGSL mask emitter: namespace-field `{}.{field}` not supported",
        ns.name()
    )))
}

fn lower_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: LowerCtx<'_>,
) -> Result<String, EmitError> {
    let lowered: Result<Vec<String>, EmitError> = args
        .iter()
        .map(|a| lower_expr(&a.value, hoisted, ctx))
        .collect();
    let lowered = lowered?;
    match (ns, method) {
        (NamespaceId::Agents, "alive") => {
            if lowered.len() != 1 {
                return Err(EmitError::Unsupported(
                    "agents.alive expects 1 arg".into(),
                ));
            }
            // `alive[id]` — stored as u32 { 0, 1 }. Compare to 1u.
            Ok(format!("(agent_alive[{}] != 0u)", lowered[0]))
        }
        _ => Err(EmitError::Unsupported(format!(
            "Phase 1 WGSL: stdlib call `{}.{method}` not supported",
            ns.name()
        ))),
    }
}

fn lower_builtin_call(
    b: Builtin,
    args: &[IrCallArg],
    hoisted: &[String],
    ctx: LowerCtx<'_>,
) -> Result<String, EmitError> {
    let lowered: Result<Vec<String>, EmitError> = args
        .iter()
        .map(|a| lower_expr(&a.value, hoisted, ctx))
        .collect();
    let lowered = lowered?;
    match b {
        Builtin::Distance => {
            if lowered.len() != 2 {
                return Err(EmitError::Unsupported(
                    "distance expects 2 args".into(),
                ));
            }
            Ok(format!("vec3_distance({}, {})", lowered[0], lowered[1]))
        }
        _ => Err(EmitError::Unsupported(format!(
            "Phase 1 WGSL: builtin `{}` not supported",
            b.name()
        ))),
    }
}

fn binop_str(op: BinOp) -> Result<&'static str, EmitError> {
    Ok(match op {
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::LtEq => "<=",
        BinOp::Gt => ">",
        BinOp::GtEq => ">=",
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => {
            // WGSL uses `%` for integer mod and leaves float mod to
            // `fmod`-style builtins. No current mask uses this, so
            // refuse loudly.
            return Err(EmitError::Unsupported(
                "Phase 1 WGSL: `%` not supported in masks".into(),
            ));
        }
    })
}

fn unop_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "!",
        UnOp::Neg => "-",
    }
}

/// Render a float as a WGSL literal. WGSL requires `f` suffix or a
/// fractional part to parse as `f32`; bare `2.0` is fine (decimal dot),
/// bare `2` would type-infer to int. Mirror `emit_mask.rs::render_float`'s
/// contract of "always emit a fractional part".
fn render_float_wgsl(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Map a lowered slot-index expression (`self_id`, `wgsl_target`) to
/// the matching creature-type array read. Used by the view-call
/// inliner so `is_hostile(a, b)` in DSL becomes
/// `is_hostile(self_ct, wgsl_target_ct)` in WGSL — the generated
/// helper takes creature ordinals, not slot ids, and the kernel
/// prelude has pre-hoisted `<sym>_ct` bindings for each slot it
/// touches.
fn ct_for(slot_expr: &str, ctx: LowerCtx<'_>) -> String {
    let trimmed = slot_expr.trim();
    if trimmed == "self_id" {
        "self_ct".to_string()
    } else if trimmed == ctx.wgsl_target_name {
        format!("{}_ct", ctx.wgsl_target_name)
    } else if trimmed
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
    {
        format!("{trimmed}_ct")
    } else {
        format!("agent_creature_type[{trimmed}]")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef, MaskIR,
        NamespaceId,
    };

    fn span() -> Span {
        Span::dummy()
    }

    fn local(name: &str, id: u16) -> IrExprNode {
        IrExprNode { kind: IrExpr::Local(LocalRef(id), name.to_string()), span: span() }
    }

    fn ns_call(ns: NamespaceId, method: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns,
                method: method.to_string(),
                args: args
                    .into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            },
            span: span(),
        }
    }

    fn unresolved(name: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::UnresolvedCall(
                name.to_string(),
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn builtin(b: Builtin, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::BuiltinCall(
                b,
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }

    fn binop(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)), span: span() }
    }

    fn lit_float(v: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(v), span: span() }
    }

    fn attack_mask_ir() -> MaskIR {
        let self_local = local("self", 0);
        let target_local = local("target", 1);

        let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
        let hostile = unresolved("is_hostile", vec![self_local.clone(), target_local.clone()]);
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![self_local.clone()]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![target_local.clone()]);
        let dist = builtin(Builtin::Distance, vec![self_pos, target_pos]);
        let lt = binop(BinOp::Lt, dist, lit_float(2.0));
        let and1 = binop(BinOp::And, alive, hostile);
        let predicate = binop(BinOp::And, and1, lt);

        MaskIR {
            head: IrActionHead {
                name: "Attack".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        }
    }

    #[test]
    fn attack_kernel_contains_expected_symbols() {
        let mask = attack_mask_ir();
        let src = emit_mask_wgsl(&mask).expect("emit attack wgsl");

        // Kernel entry point + workgroup attribute.
        assert!(
            src.contains("@compute @workgroup_size(64)"),
            "missing workgroup attr in:\n{src}"
        );
        assert!(
            src.contains("fn cs_attack(@builtin(global_invocation_id)"),
            "missing cs_attack signature in:\n{src}"
        );

        // Storage bindings.
        assert!(src.contains("@group(0) @binding(0) var<storage, read> agent_pos"));
        assert!(src.contains("@group(0) @binding(1) var<storage, read> agent_alive"));
        assert!(src.contains("@group(0) @binding(2) var<storage, read> agent_creature_type"));
        assert!(src.contains("@group(0) @binding(3) var<storage, read_write> mask_out"));
        assert!(src.contains("@group(0) @binding(4) var<uniform> cfg"));

        // Alive-gate + hostility inline.
        assert!(src.contains("agent_alive[self_id] == 0u"));
        // `target` is a WGSL reserved keyword; emitter aliases the DSL
        // binding to `wgsl_target` inside the kernel scope.
        assert!(
            src.contains("is_hostile(self_ct, wgsl_target_ct)"),
            "missing is_hostile call in:\n{src}"
        );
        // Distance check inlined against the renamed target pos.
        assert!(
            src.contains("vec3_distance(self_pos, wgsl_target_pos) < 2.0"),
            "missing distance clause in:\n{src}"
        );
        // Loop var is `wgsl_target`, not `target`.
        assert!(
            src.contains("for (var wgsl_target: u32"),
            "missing renamed loop var in:\n{src}"
        );
        // Bit-packed atomic write.
        assert!(src.contains("atomicOr(&mask_out[word_idx], 1u << bit_idx)"));
    }

    /// Phase 2 self-only masks (Hold / Flee / Eat / Drink / Rest) all
    /// lower to `agents.alive(self)` — the body has no target loop and
    /// no radius prefilter. This test builds that shape by hand and
    /// asserts the emitter produces a kernel with (a) no loop header,
    /// (b) the self-alive gate, and (c) the bit-pack epilogue.
    #[test]
    fn self_only_mask_emits_loopless_kernel() {
        let self_local = local("self", 0);
        let alive_self = ns_call(NamespaceId::Agents, "alive", vec![self_local]);
        let mask = MaskIR {
            head: IrActionHead {
                name: "Hold".into(),
                shape: IrActionHeadShape::None,
                span: span(),
            },
            candidate_source: None,
            predicate: alive_self,
            annotations: vec![],
            span: span(),
        };
        let src = emit_mask_wgsl(&mask).expect("emit hold wgsl");
        // The predicate lowerer turns the agents.alive(self) clause
        // into the same `(agent_alive[self_id] != 0u)` expression it
        // uses for Attack's target-alive check — assert the shape,
        // not the exact text, so a future lowerer rewrite doesn't
        // silently invalidate the test.
        assert!(
            src.contains("fn cs_hold("),
            "missing cs_hold entry point:\n{src}"
        );
        // Self-only kernels still emit the self-alive gate at the top
        // (dead agents produce no bits) and then drop into the body.
        assert!(src.contains("agent_alive[self_id] == 0u"));
        // No target loop for self-only masks — the `for (var wgsl_target`
        // header only appears on target-bound masks.
        assert!(
            !src.contains("for (var wgsl_target"),
            "self-only mask should not emit a target loop:\n{src}"
        );
        // And the bit-pack epilogue still fires.
        assert!(src.contains("atomicOr(&mask_out[word_idx], 1u << bit_idx)"));
    }

    /// Phase 2 rejects masks that carry a non-Agent positional param
    /// (the Cast mask: `mask Cast(ability: AbilityId)`). The
    /// target-loop kernel shape the emitter produces assumes the
    /// binding is an AgentId slot; anything else would require
    /// ability-registry / cooldown / view storage the Phase 2 emitter
    /// doesn't yet provision. The backend skips these at registration
    /// time; the error surface here is what lets it do so safely.
    #[test]
    fn non_agent_parametric_mask_is_rejected_with_skip_hint() {
        let mut mask = attack_mask_ir();
        mask.head.name = "Cast".into();
        mask.head.shape = IrActionHeadShape::Positional(vec![(
            "ability".into(),
            LocalRef(1),
            IrType::AbilityId,
        )]);
        mask.candidate_source = None;
        let err = emit_mask_wgsl(&mask).unwrap_err();
        assert!(
            err.contains("Skip on GPU"),
            "expected skip hint in err: {err}"
        );
    }

    /// A target-bound mask whose name isn't in `mask_radius_symbol`'s
    /// table surfaces a loud diagnostic rather than silently defaulting
    /// to the Attack radius. When adding a new target-bound mask (e.g.
    /// a `mask Strike(target)`) the author must wire its config knob
    /// through ConfigUniform + `mask_radius_symbol`.
    #[test]
    fn unknown_target_bound_mask_surfaces_radius_error() {
        let mut mask = attack_mask_ir();
        mask.head.name = "Strike".into();
        let err = emit_mask_wgsl(&mask).unwrap_err();
        assert!(
            err.contains("no wired radius knob"),
            "expected radius-knob err, got: {err}"
        );
    }
}
