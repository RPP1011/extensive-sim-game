//! WGSL emission for mask predicates — Phase 1 of the GPU megakernel plan.
//!
//! Mirrors [`emit_mask`]'s AST walk but emits a compute shader source
//! string targeting the WGSL subset wgpu 26 accepts via its runtime
//! `naga` parse path. The Phase 1 deliverable is one kernel per mask
//! declaration, each producing a `u32` bitmap indexed by agent slot —
//! one bit per agent, `1` iff the predicate holds for that agent against
//! every enumerated target within the `from`-clause radius. Task 158's
//! GPU-emittability validator has already gated the subset we accept
//! here, so unsupported shapes surface as `EmitError::Unsupported`
//! rather than silent miscompiles.
//!
//! ## Scope (Phase 1)
//!
//! Only the `Attack` mask is wired end-to-end (`cs_attack` kernel + the
//! matching dispatch in `engine_gpu`). The emitter itself is deliberately
//! parameterised — `emit_mask_wgsl` walks any `MaskIR` — so Phase 2 can
//! turn on the remaining 7 masks by registering their kernels on the
//! GPU backend side without touching the emitter core. The supported
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
    // Phase 1 gate: Attack only. The rest of the emitter is generic —
    // we just refuse to instantiate a kernel for masks we haven't yet
    // vetted. Phase 2 removes this check.
    if mask.head.name != "Attack" {
        return Err(EmitError::Unsupported(format!(
            "Phase 1 WGSL emitter only supports `Attack`; got `{}`",
            mask.head.name
        )));
    }

    // WGSL reserves `target` as a keyword, so we alias the DSL-level
    // target binding name to a kernel-local `wgsl_target` symbol. The
    // Rust emitter's `target` binding has no such conflict — this
    // aliasing is WGSL-only.
    let dsl_target_name = match &mask.head.shape {
        IrActionHeadShape::Positional(binds) if binds.len() == 1 => binds[0].0.clone(),
        _ => {
            return Err(EmitError::Unsupported(format!(
                "mask `{}` must have a single positional target for Phase 1",
                mask.head.name
            )));
        }
    };
    let wgsl_target_name = "wgsl_target".to_string();

    let mut out = String::new();
    emit_header(&mut out, mask);
    emit_bindings(&mut out);
    emit_helpers(&mut out);
    emit_kernel(&mut out, mask, &dsl_target_name, &wgsl_target_name)?;
    Ok(out)
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
    // `config.<block>.<field>` reads Phase 1 masks need. Attack pulls
    // `combat.attack_range` out of it.
    writeln!(out, "struct ConfigUniform {{").unwrap();
    writeln!(out, "    combat_attack_range: f32,").unwrap();
    writeln!(out, "    // Padding to align to 16 bytes.").unwrap();
    writeln!(out, "    _pad0: f32,").unwrap();
    writeln!(out, "    _pad1: f32,").unwrap();
    writeln!(out, "    _pad2: f32,").unwrap();
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
    dsl_target_name: &str,
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
    // The attack `from` clause is fixed in `masks.sim` to
    // `query.nearby_agents(agents.pos(self), config.combat.attack_range)`.
    // We inline the radius read — Phase 1 hardcodes this knob; Phase 2
    // will walk the `from` expression generically.
    writeln!(
        out,
        "    let radius = cfg.combat_attack_range;"
    )
    .unwrap();
    // The Phase 1 dispatch has no spatial hash on GPU — we iterate
    // every alive slot and let the `distance < radius` clause filter.
    // Brute force is fine at N <= small-world (8-200 agents); Phase 5
    // ports the spatial hash.
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
    // Radius prefilter — the `from` clause bound. Note: for the attack
    // mask this is identical to the final `distance < 2.0` clause
    // because `attack_range == 2.0` in DSL defaults. If they drift,
    // the inner predicate still enforces the tighter bound.
    writeln!(
        out,
        "        if (vec3_distance(self_pos, {t}_pos) > radius) {{ continue; }}",
        t = wgsl_target_name
    )
    .unwrap();

    // Now lower the predicate body clauses. The lowerer rewrites every
    // occurrence of the DSL-level `target` local to `wgsl_target` so
    // the emitted expression references the loop variable the kernel
    // just established.
    let mut clauses: Vec<&IrExprNode> = Vec::new();
    flatten_and(&mask.predicate, &mut clauses);

    let hoisted = Vec::<String>::new();
    let ctx = LowerCtx { dsl_target_name, wgsl_target_name };
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
        // Phase 1 only knows `combat.attack_range`. Extend per mask.
        return match field {
            "combat.attack_range" => Ok("cfg.combat_attack_range".to_string()),
            other => Err(EmitError::Unsupported(format!(
                "Phase 1 WGSL: config field `{other}` not wired"
            ))),
        };
    }
    Err(EmitError::Unsupported(format!(
        "Phase 1 WGSL: namespace-field `{}.{field}` not supported",
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

    #[test]
    fn non_attack_mask_is_rejected_at_phase_1() {
        let mut mask = attack_mask_ir();
        mask.head.name = "Flee".into();
        let err = emit_mask_wgsl(&mask).unwrap_err();
        assert!(err.contains("Phase 1"), "unexpected err: {err}");
    }
}
