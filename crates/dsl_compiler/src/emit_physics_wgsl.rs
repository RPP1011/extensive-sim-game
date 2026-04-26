//! WGSL emission for physics rules — Phase 6a of the GPU megakernel plan.
//!
//! Companion to [`emit_physics`] (which emits Rust) — walks the same
//! `PhysicsIR` AST and produces a WGSL function per rule plus a single
//! `physics_dispatch(event_idx)` switch that branches on `event.kind`
//! and invokes every applicable rule.
//!
//! ## Shape of the output
//!
//! One WGSL source string. Contents:
//!
//!   * `physics_<rule_snake>(event_idx: u32)` — one fn per physics rule.
//!     Reads `events_in[event_idx]`, guards on matching `kind`,
//!     destructures the payload, then lowers the rule body.
//!   * `physics_dispatch(event_idx: u32)` — switches on the event's
//!     `kind` and calls every rule applicable to that kind. Tag-matched
//!     rules (currently none in `physics.sim`) would also be dispatched
//!     from here — the implementation is ready for them but no test
//!     exercises the path.
//!
//! This emitter does **not** own:
//!
//!   * The WGSL `struct EventSlot` layout — Task 188's event ring defines
//!     that, along with the `gpu_emit_event(kind, p0..pN)` push fn.
//!   * The agent SoA getter / setter fns (`state_agent_hp`,
//!     `state_set_agent_hp`, `state_kill_agent`, …) — the integration
//!     phase provisions those against whatever buffer shape the GPU
//!     backend settles on.
//!   * Spatial query primitives (`spatial_nearest_hostile_to`,
//!     `spatial_nearby_kin_count`, `spatial_nearby_kin_at`) — Task 186's
//!     `spatial_gpu::SPATIAL_WGSL` already provides these.
//!   * View read functions (`view_<name>_get`) — Task 185's
//!     `emit_view_wgsl.rs` owns the shape.
//!   * Config uniform bindings — shared with the mask / scoring WGSL.
//!
//! When the emitted source is concatenated with the stubs above, the
//! whole module becomes a valid standalone WGSL translation unit.
//!
//! ## Stub-function conventions
//!
//! All stub call shapes are documented alongside the lowerings that
//! produce them. The integration phase has to supply each of:
//!
//!   * **Agent SoA** (getters return `f32` / `u32` / sentinel-encoded
//!     optionals; setters return nothing):
//!     - `state_agent_alive(id: u32) -> bool`
//!     - `state_agent_hp(id: u32) -> f32`
//!     - `state_agent_max_hp(id: u32) -> f32`
//!     - `state_agent_shield_hp(id: u32) -> f32`
//!     - `state_agent_attack_damage(id: u32) -> f32`
//!     - `state_agent_stun_expires_at(id: u32) -> u32`
//!     - `state_agent_slow_expires_at(id: u32) -> u32`
//!     - `state_agent_slow_factor_q8(id: u32) -> u32`
//!     - `state_agent_gold(id: u32) -> i32`
//!     - `state_agent_cooldown_next_ready(id: u32) -> u32`
//!     - `state_agent_engaged_with(id: u32) -> u32` — sentinel
//!       `0xFFFFFFFFu` means "no partner".
//!     - `state_set_agent_hp(id: u32, v: f32)`
//!     - `state_set_agent_shield_hp(id: u32, v: f32)`
//!     - `state_set_agent_stun_expires_at(id: u32, v: u32)`
//!     - `state_set_agent_slow_expires_at(id: u32, v: u32)`
//!     - `state_set_agent_slow_factor_q8(id: u32, v: u32)`
//!     - `state_add_agent_gold(id: u32, delta: i32)` —
//!       `agents.add_gold` / `sub_gold`.
//!     - `state_adjust_standing(a: u32, b: u32, delta: i32)`
//!     - `state_set_agent_cooldown_next_ready(id: u32, v: u32)`
//!     - `state_set_agent_engaged_with(id: u32, partner: u32)`
//!     - `state_clear_agent_engaged_with(id: u32)`
//!     - `state_kill_agent(id: u32)`
//!     - `state_push_agent_memory(observer: u32, source: u32,
//!       payload: u32, confidence: f32, tick: u32)`
//!
//!   * **Spatial queries** (bounded iteration style):
//!     - `spatial_nearest_hostile_to(agent: u32, radius: f32) -> u32`
//!       — returns agent id or `0xFFFFFFFFu`.
//!     - `spatial_nearby_kin_count(agent: u32, radius: f32) -> u32` —
//!       count of kin candidates (bounded by spatial-hash cell cap).
//!     - `spatial_nearby_kin_at(agent: u32, radius: f32, idx: u32) -> u32`
//!       — i'th kin. Must be deterministic (spatial hash already sorts
//!       by slot).
//!
//!   * **Ability registry** (bounded iteration):
//!     - `abilities_is_known(ab: u32) -> bool`
//!     - `abilities_cooldown_ticks(ab: u32) -> u32`
//!     - `abilities_effects_count(ab: u32) -> u32` — up to
//!       `MAX_EFFECTS_PER_PROGRAM`.
//!     - `abilities_effect_op_at(ab: u32, idx: u32) -> EffectOp` —
//!       where `EffectOp { kind: u32, p0: u32, p1: u32 }` packs the
//!       discriminant + up to two u32-bitcast payload fields. The
//!       integration phase picks the concrete payload layout.
//!
//!   * **Event ring** (Task 188):
//!     - `gpu_emit_event(kind: u32, p0: u32, p1: u32, p2: u32, p3: u32,
//!       p4: u32, p5: u32)` — six u32 payload slots cover every event
//!       in `events.sim` (the widest `AgentCast` needs five: actor,
//!       ability, target, depth, tick). Unused slots pass `0u`.
//!     - `EVENT_KIND_<SCREAMING_SNAKE>` — scalar const per event
//!       variant.
//!     - `events_in: array<EventSlot>` — read binding. `EventSlot`
//!       carries `kind: u32` + six `u32` payload words.
//!
//!   * **Config** (world-scalars live in the shared `SimCfg` storage
//!     buffer emitted via `emit_sim_cfg_struct_wgsl`; see Task 2.8 of
//!     the GPU sim-state refactor):
//!     - `sim_cfg.engagement_range: f32`   (← `config.combat.engagement_range`)
//!     - `sim_cfg.cascade_max_iterations: u32` (← `cascade.max_iterations`)
//!     - `sim_cfg.tick: u32`               (← `state.tick` via the
//!       `wgsl_world_tick` alias injected by `physics.rs`)
//!
//!   * **View reads** (Task 185):
//!     - `view_<snake>_get(args..., [tick])` where applicable.
//!
//! Any rule that would need a stub outside this roster surfaces as
//! `EmitError::Unsupported` with a diagnostic that names the missing
//! primitive. That keeps integration failures loud.
//!
//! ## Reserved-word avoidance
//!
//! WGSL reserves a long keyword list (including `break`, `continue`,
//! `loop`, `let`, `var`, `fn`, `if`, `else`, `return`, `true`, `false`,
//! `match`, `struct`, `type`, `for`, `while`, `switch`, `pass`). When a
//! DSL binding or field name collides, we prefix with `wgsl_`. Common
//! physics.sim DSL names (`actor`, `target`, `mover`, `caster`, `ab`,
//! `amount`, etc.) don't collide — those pass through unchanged.
//!
//! ## GPU-emittability validator contract
//!
//! Task 158's validator guarantees:
//!   * POD types only (no `String`, no heap-allocated containers).
//!   * Bounded iteration (every `for` has a static cap).
//!   * No recursion (rules call stub fns, never themselves or each
//!     other).
//!
//! The emitter trusts those invariants: anything the validator would
//! have rejected shouldn't reach us. On the rare failure mode (IR shape
//! the emitter doesn't recognise), we emit a WGSL comment describing
//! the offending node and raise `EmitError::Unsupported(reason)` so the
//! backend skips the rule loudly.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::ast::{BinOp, UnOp};
use crate::ir::{
    Builtin, EventField, EventIR, EventTagIR, IrEmit, IrExpr, IrExprNode, IrPattern,
    IrPatternBinding, IrPhysicsPattern, IrStmt, IrStmtMatchArm, IrType, NamespaceId,
    PhysicsHandlerIR, PhysicsIR,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised during WGSL physics emission. Narrow mirror of
/// [`crate::emit_physics::EmitError`] — same diagnostic surface, different
/// emitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The pattern lacked an event/tag ref — the referenced name didn't
    /// resolve during lowering.
    UnresolvedEventInPattern(String),
    /// The `emit` statement names an event the compiler doesn't know.
    UnresolvedEventInEmit(String),
    /// Some IR construct isn't supported by the WGSL emitter. Carries a
    /// short diagnostic with the offending shape.
    Unsupported(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::UnresolvedEventInPattern(n) => {
                write!(f, "physics WGSL handler matches `{n}` which is not a declared event or tag")
            }
            EmitError::UnresolvedEventInEmit(n) => {
                write!(f, "physics WGSL handler emits event `{n}` which is not declared")
            }
            EmitError::Unsupported(s) => write!(f, "physics WGSL emission: {s}"),
        }
    }
}

impl std::error::Error for EmitError {}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Per-compilation context the physics WGSL emitter needs beyond the
/// IR rule itself: the full event catalog (for destructuring + dispatch
/// fan-out) and the tag catalog (for tag-matched handler argument
/// lists, even though `physics.sim` has no tag-matched rules today).
pub struct EmitContext<'a> {
    pub events: &'a [EventIR],
    pub event_tags: &'a [EventTagIR],
}

// ---------------------------------------------------------------------------
// Public emission entry points
// ---------------------------------------------------------------------------

/// Emit a single physics rule's WGSL fn. Output is a top-level
/// `fn physics_<snake>(event_idx: u32) { ... }` body. Callers
/// concatenate multiple rule outputs plus a dispatcher (via
/// [`emit_physics_dispatcher_wgsl`]) into the final shader.
pub fn emit_physics_wgsl(
    physics: &PhysicsIR,
    ctx: &EmitContext<'_>,
) -> Result<String, EmitError> {
    // `@cpu_only` rules have no WGSL representation. The CPU handler
    // (emitted separately in `emit_physics.rs`) runs the body; the GPU
    // event-kind dispatcher (below, via `applicable_rules`) also skips
    // this rule so its handler fn name is never referenced from WGSL.
    // Callers that concatenate per-rule output append an empty string,
    // keeping the GPU shader free of any mention of the rule.
    if physics.cpu_only {
        return Ok(String::new());
    }
    if physics.handlers.len() != 1 {
        return Err(EmitError::Unsupported(format!(
            "expected exactly one `on` handler per physics rule (got {} in `{}`)",
            physics.handlers.len(),
            physics.name
        )));
    }
    let handler = &physics.handlers[0];
    scan_emits(&handler.body)?;
    let fn_name = handler_fn_name(&physics.name);
    let mut out = String::new();

    writeln!(
        out,
        "// Generated from physics rule `{}`.",
        physics.name
    )
    .unwrap();
    writeln!(out, "fn {fn_name}(event_idx: u32) {{").unwrap();

    // Body layout:
    //   1. Read event slot.
    //   2. Kind guard (return early on mismatch).
    //   3. Destructure payload into locals.
    //   4. Optional `where` clause → early-return guard.
    //   5. Rule body, statement by statement.
    //
    // The event-slot local is named `ev_rec` (not `e`) so DSL-author
    // bindings that choose short single-letter names like `e` (as in
    // `expires_at_tick: e` on EffectSlowApplied) don't collide with
    // the fn-scope event record. Internal reads reference `ev_rec.kind`
    // / `ev_rec.payload[N]` / `ev_rec.tick`.
    let (kind_const, destructure) = emit_event_destructure(handler, ctx)?;
    writeln!(out, "    let ev_rec = events_in[event_idx];").unwrap();
    writeln!(out, "    if (ev_rec.kind != {kind_const}) {{ return; }}").unwrap();
    for line in &destructure {
        writeln!(out, "    {line}").unwrap();
    }

    if let Some(where_clause) = &handler.where_clause {
        let cond = lower_expr(where_clause)?;
        writeln!(out, "    if (!({cond})) {{ return; }}").unwrap();
    }

    for stmt in &handler.body {
        emit_stmt(&mut out, stmt, 4)?;
    }

    writeln!(out, "}}").unwrap();
    Ok(out)
}

/// Emit the `physics_dispatch(event_idx)` function. Switches on the
/// event's `kind` and calls every rule applicable to that kind. Applies
/// both kind-matched rules (`on EventName { ... }`) and tag-matched
/// rules (`on @TagName { ... }` — none in physics.sim today but the
/// dispatcher supports them).
pub fn emit_physics_dispatcher_wgsl(
    physics: &[PhysicsIR],
    ctx: &EmitContext<'_>,
) -> String {
    let applicable = applicable_rules(physics, ctx);
    let mut kinds: Vec<&str> = applicable.keys().map(|s| s.as_str()).collect();
    kinds.sort();

    let mut out = String::new();
    writeln!(
        out,
        "// Dispatcher — switches on events_in[event_idx].kind and calls every"
    )
    .unwrap();
    writeln!(
        out,
        "// physics rule applicable to that kind. Unknown kinds are a no-op."
    )
    .unwrap();
    writeln!(out, "fn physics_dispatch(event_idx: u32) {{").unwrap();
    if kinds.is_empty() {
        // Still emit a well-formed fn so downstream concatenation can
        // rely on the symbol being present.
        writeln!(out, "    // No physics rules in scope.").unwrap();
        writeln!(out, "    return;").unwrap();
    } else {
        // Read the kind from the event record. Uses the same binding
        // name convention as the per-rule fns (`ev_rec.*`) for visual
        // consistency.
        writeln!(out, "    let kind = events_in[event_idx].kind;").unwrap();
        for kind in &kinds {
            let kind_const = event_kind_const(kind);
            let rule_names = applicable.get(*kind).unwrap();
            writeln!(out, "    if (kind == {kind_const}) {{").unwrap();
            for rule in rule_names {
                let fn_name = handler_fn_name(rule);
                writeln!(out, "        {fn_name}(event_idx);").unwrap();
            }
            writeln!(out, "    }}").unwrap();
        }
    }
    writeln!(out, "}}").unwrap();
    out
}

// ---------------------------------------------------------------------------
// Rule applicability — which kinds does which rule fire on?
// ---------------------------------------------------------------------------

/// Map event-kind name → ordered list of physics-rule names that fire
/// on that kind. Kind-matched rules come first (by rule name for
/// reorder stability), then tag-matched rules (also by rule name).
fn applicable_rules<'a>(
    physics: &'a [PhysicsIR],
    ctx: &'a EmitContext<'a>,
) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();

    // Skip `@cpu_only` rules — their handler fns don't exist in WGSL,
    // so the dispatcher must not reference them. The CPU dispatcher
    // (emit_physics.rs) still routes these through the Rust side.
    let mut sorted: Vec<&PhysicsIR> = physics.iter().filter(|p| !p.cpu_only).collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    // Kind-matched.
    for p in &sorted {
        for h in &p.handlers {
            if let IrPhysicsPattern::Kind(pat) = &h.pattern {
                out.entry(pat.name.clone()).or_default().push(p.name.clone());
            }
        }
    }
    // Tag-matched — one entry per event that claims the tag.
    for p in &sorted {
        for h in &p.handlers {
            if let IrPhysicsPattern::Tag { tag, .. } = &h.pattern {
                let Some(tref) = tag else { continue };
                for event in ctx.events {
                    if event.tags.iter().any(|r| *r == *tref) {
                        out.entry(event.name.clone()).or_default().push(p.name.clone());
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Event destructuring
// ---------------------------------------------------------------------------

/// Emit the destructure prelude for a physics handler. Returns the
/// `EVENT_KIND_*` constant name used in the kind guard and one WGSL
/// `let` per bound field. The field-to-payload-slot mapping follows the
/// event's field declaration order: slot 0 = first field, slot 1 =
/// second, etc. `f32` fields use `bitcast<f32>(e.payload_N)`; `u32` /
/// id fields use `e.payload_N` directly; `i32` fields cast via
/// `bitcast<i32>(e.payload_N)`.
fn emit_event_destructure(
    handler: &PhysicsHandlerIR,
    ctx: &EmitContext<'_>,
) -> Result<(String, Vec<String>), EmitError> {
    match &handler.pattern {
        IrPhysicsPattern::Kind(pat) => {
            let Some(event_ref) = pat.event else {
                return Err(EmitError::UnresolvedEventInPattern(pat.name.clone()));
            };
            let event = &ctx.events[event_ref.0 as usize];
            let fields = fields_with_implicit_tick(event);
            let kind_const = event_kind_const(&event.name);
            let mut lines = Vec::new();
            for b in &pat.bindings {
                if let IrPattern::Bind { name, .. } = &b.value {
                    let field = fields.iter().find(|f| f.name == b.field).ok_or_else(|| {
                        EmitError::Unsupported(format!(
                            "handler binds unknown field `{}` on event `{}`",
                            b.field, event.name
                        ))
                    })?;
                    let slot = payload_slot(&fields, &b.field).ok_or_else(|| {
                        EmitError::Unsupported(format!(
                            "can't map field `{}` of event `{}` to payload slot",
                            b.field, event.name
                        ))
                    })?;
                    let wgsl_name = wgsl_ident(name);
                    let (ty_str, rhs) = if slot == u32::MAX {
                        // `tick` — read from the dedicated EventRecord.tick
                        // field, not the payload array.
                        ("u32".to_string(), "ev_rec.tick".to_string())
                    } else {
                        payload_read(&field.ty, slot)?
                    };
                    lines.push(format!("let {wgsl_name}: {ty_str} = {rhs};"));
                }
            }
            Ok((kind_const, lines))
        }
        IrPhysicsPattern::Tag { tag, bindings, name, .. } => {
            // Tag-matched handlers get a stub destructure that fetches
            // the tag's fields by ordinal. The per-event concrete field
            // order is resolved per-event at dispatch time; the WGSL
            // dispatcher already knows which kind it's handling (it
            // branches on `kind`), so the emitter can read fields by
            // tag-declared index assuming the uploader preserves the
            // `@tag` field order at the front of each event's payload.
            //
            // No shipped physics rule uses tag-matching, so we raise
            // Unsupported until one lands and exercises the path. When
            // that happens, swap this branch out for the same
            // kind-style destructure wrapped in a per-concrete-kind
            // switch.
            let _ = (tag, bindings);
            Err(EmitError::Unsupported(format!(
                "tag-matched handler `@{name}` — no physics.sim rule uses tag matching; \
                 implement per-concrete-kind destructure when a tag-matched rule lands"
            )))
        }
    }
}

/// Map a field name to its payload-slot index. Slots are assigned in
/// the event's declaration order; the implicit `tick` is read from the
/// event record's dedicated `tick: u32` field (see `payload_tick_read`)
/// rather than a payload slot.
fn payload_slot(fields: &[EventField], name: &str) -> Option<u32> {
    // The `tick` field lives outside the payload array on EventRecord.
    // Bindings that reference it read `e.tick` directly; we return
    // `u32::MAX` as a sentinel to flag that the caller should use the
    // tick-specific read path instead.
    if name == "tick" {
        return Some(u32::MAX);
    }
    fields
        .iter()
        .filter(|f| f.name != "tick")
        .position(|f| f.name == name)
        .map(|p| p as u32)
}

/// Emit the `(type, rhs)` pair for reading a single payload slot.
///
/// Payload slots are indexed into the `payload[N]` array carried by
/// the event_ring primitive's `EventRecord` struct (8 u32 slots). The
/// emitter previously used `e.payload_N` bare field syntax; switching
/// to the array access form keeps us in lockstep with task 188's WGSL
/// `EventRecord { kind: u32, tick: u32, payload: array<u32, 8> }`
/// layout so both emitters can share the same record struct definition.
fn payload_read(ty: &IrType, slot: u32) -> Result<(String, String), EmitError> {
    match ty {
        IrType::F32 | IrType::F64 => Ok((
            "f32".into(),
            format!("bitcast<f32>(ev_rec.payload[{slot}])"),
        )),
        IrType::Bool => Ok((
            "bool".into(),
            format!("(ev_rec.payload[{slot}] != 0u)"),
        )),
        IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 => Ok((
            "i32".into(),
            format!("bitcast<i32>(ev_rec.payload[{slot}])"),
        )),
        IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64 => Ok((
            "u32".into(),
            format!("ev_rec.payload[{slot}]"),
        )),
        IrType::AgentId
        | IrType::ItemId
        | IrType::GroupId
        | IrType::QuestId
        | IrType::AuctionId
        | IrType::EventId
        | IrType::AbilityId => Ok(("u32".into(), format!("ev_rec.payload[{slot}]"))),
        other => Err(EmitError::Unsupported(format!(
            "payload field of type {other:?} — no WGSL payload read mapping"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Statement lowering
// ---------------------------------------------------------------------------

fn emit_stmt(out: &mut String, stmt: &IrStmt, indent: usize) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    match stmt {
        IrStmt::Let { name, value, .. } => {
            let v = lower_expr(value)?;
            let ident = wgsl_ident(name);
            // WGSL requires `var` (not `let`) for mutable bindings, but
            // every physics `let` in physics.sim is effectively
            // immutable post-initialisation. Use `let` — simpler, also
            // allows the compiler to avoid aliasing analysis.
            writeln!(out, "{pad}let {ident} = {v};").unwrap();
        }
        IrStmt::Emit(emit) => emit_emit(out, emit, indent)?,
        IrStmt::If { cond, then_body, else_body, .. } => {
            let c = lower_expr(cond)?;
            writeln!(out, "{pad}if ({c}) {{").unwrap();
            for s in then_body {
                emit_stmt(out, s, indent + 4)?;
            }
            if let Some(eb) = else_body {
                writeln!(out, "{pad}}} else {{").unwrap();
                for s in eb {
                    emit_stmt(out, s, indent + 4)?;
                }
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        IrStmt::For { binder_name, iter, filter, body, .. } => {
            emit_for_stmt(out, binder_name, iter, filter.as_ref(), body, indent)?;
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            emit_match_stmt(out, scrutinee, arms, indent)?;
        }
        IrStmt::SelfUpdate { .. } => {
            return Err(EmitError::Unsupported(
                "`self += ...` self-update statements only valid in fold bodies".into(),
            ));
        }
        IrStmt::Expr(e) => {
            let v = lower_expr(e)?;
            writeln!(out, "{pad}{v};").unwrap();
        }
        IrStmt::BeliefObserve { .. } => {
            // Belief mutations write to cold_beliefs (CPU-only SoA storage).
            // WGSL code generation is deferred to Plan ToM Task 5.
            return Err(EmitError::Unsupported(
                "`beliefs().observe()` WGSL code generation not yet implemented \
                 (deferred to Plan ToM Task 5)"
                    .into(),
            ));
        }
    }
    Ok(())
}

/// Lower a `for <binder> in <iter> [where <filter>] { body }`. Only
/// spatial-query iteration sources are supported — the validator
/// guarantees bounded iteration and those are the only bounded sources
/// the physics.sim rules use (`query.nearby_kin`, occasionally
/// `abilities.effects`). The loop expands to a `for (var i = 0u; i <
/// count; i = i + 1u)` using count + at helpers the integration phase
/// provides.
fn emit_for_stmt(
    out: &mut String,
    binder: &str,
    iter: &IrExprNode,
    filter: Option<&IrExprNode>,
    body: &[IrStmt],
    indent: usize,
) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    let binder = wgsl_ident(binder);
    let iter_shape = classify_for_iter(iter)?;
    match iter_shape {
        ForIter::NearbyKin { agent, radius } => {
            // Bounded loop: count + indexed-at helper. Gives us a
            // deterministic walk that matches the CPU's
            // `SpatialHash::within_radius` iteration order.
            let idx_var = "wgsl_kin_idx";
            let count_var = "wgsl_kin_count";
            writeln!(
                out,
                "{pad}let {count_var} = spatial_nearby_kin_count({agent}, {radius});"
            )
            .unwrap();
            writeln!(
                out,
                "{pad}for (var {idx_var}: u32 = 0u; {idx_var} < {count_var}; {idx_var} = {idx_var} + 1u) {{"
            )
            .unwrap();
            writeln!(
                out,
                "{pad}    let {binder} = spatial_nearby_kin_at({agent}, {radius}, {idx_var});"
            )
            .unwrap();
            if let Some(f) = filter {
                let cond = lower_expr(f)?;
                writeln!(out, "{pad}    if (!({cond})) {{ continue; }}").unwrap();
            }
            for s in body {
                emit_stmt(out, s, indent + 4)?;
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        ForIter::AbilityEffects { ability } => {
            let idx_var = "wgsl_effect_idx";
            let count_var = "wgsl_effect_count";
            writeln!(
                out,
                "{pad}let {count_var} = abilities_effects_count({ability});"
            )
            .unwrap();
            writeln!(
                out,
                "{pad}for (var {idx_var}: u32 = 0u; {idx_var} < {count_var}; {idx_var} = {idx_var} + 1u) {{"
            )
            .unwrap();
            writeln!(
                out,
                "{pad}    let {binder} = abilities_effect_op_at({ability}, {idx_var});"
            )
            .unwrap();
            if let Some(f) = filter {
                let cond = lower_expr(f)?;
                writeln!(out, "{pad}    if (!({cond})) {{ continue; }}").unwrap();
            }
            for s in body {
                emit_stmt(out, s, indent + 4)?;
            }
            writeln!(out, "{pad}}}").unwrap();
        }
    }
    Ok(())
}

/// Classification of a `for ... in <iter>` iteration source. Only the
/// two shapes the physics.sim rules exercise are supported.
enum ForIter {
    /// `for kin in query.nearby_kin(<agent>, <radius>)`. Yields an
    /// `AgentId` (u32) per iteration.
    NearbyKin { agent: String, radius: String },
    /// `for op in abilities.effects(<ability>)`. Yields an `EffectOp`
    /// struct per iteration.
    AbilityEffects { ability: String },
}

fn classify_for_iter(iter: &IrExprNode) -> Result<ForIter, EmitError> {
    if let IrExpr::NamespaceCall { ns, method, args } = &iter.kind {
        match (*ns, method.as_str()) {
            (NamespaceId::Query, "nearby_kin") => {
                if args.len() != 2 {
                    return Err(EmitError::Unsupported(format!(
                        "query.nearby_kin expected 2 args, got {}",
                        args.len()
                    )));
                }
                let agent = lower_expr(&args[0].value)?;
                let radius = lower_expr(&args[1].value)?;
                return Ok(ForIter::NearbyKin { agent, radius });
            }
            (NamespaceId::Abilities, "effects") => {
                if args.len() != 1 {
                    return Err(EmitError::Unsupported(format!(
                        "abilities.effects expected 1 arg, got {}",
                        args.len()
                    )));
                }
                let ability = lower_expr(&args[0].value)?;
                return Ok(ForIter::AbilityEffects { ability });
            }
            _ => {}
        }
    }
    Err(EmitError::Unsupported(format!(
        "`for` loop iter source {iter:?} — only query.nearby_kin and abilities.effects \
         are WGSL-emittable"
    )))
}

/// Lower a match statement. The only shape physics.sim uses today is
/// `match op { Damage { amount } => ..., ... }` over `EffectOp`
/// variants. We emit an `if`-chain branching on
/// `<scrut>.kind == EFFECT_OP_KIND_<VARIANT>`, binding the struct
/// fields from fixed payload slots. Wildcards and bare binds produce
/// an unconditional fall-through.
fn emit_match_stmt(
    out: &mut String,
    scrutinee: &IrExprNode,
    arms: &[IrStmtMatchArm],
    indent: usize,
) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    let scrut = lower_expr(scrutinee)?;
    writeln!(
        out,
        "{pad}// match on EffectOp variant (kind discriminant + up to two payload u32s)"
    )
    .unwrap();
    // Copy the scrutinee into a local so repeated reads don't re-run
    // whatever call produced it.
    writeln!(out, "{pad}let wgsl_match_scrut = {scrut};").unwrap();
    // Emit an if/else-if chain. Each arm contributes its `if (...)` or
    // `else if (...)` header + body; a single closing `}` lands after
    // the last arm. This keeps the chain syntactically valid — a per-arm
    // close would produce the `}} else if` double-brace form WGSL
    // rejects.
    if arms.is_empty() {
        return Ok(());
    }
    for (i, arm) in arms.iter().enumerate() {
        emit_match_arm(out, arm, indent, i == 0)?;
    }
    writeln!(out, "{pad}}}").unwrap();
    Ok(())
}

fn emit_match_arm(
    out: &mut String,
    arm: &IrStmtMatchArm,
    indent: usize,
    first: bool,
) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    // Header: `if` on the first arm, `} else if` on subsequent arms.
    // We use a plain `String` + `push_str` for the prefix so the runtime
    // value can contain a literal `}` without the `format!`-style `}}`
    // escape applying. The `writeln!` below uses `{header}` unchanged.
    let header = if first { "if".to_string() } else { "} else if".to_string() };
    match &arm.pattern {
        IrPattern::Wildcard | IrPattern::Bind { .. } => {
            // Unconditional fall-through arm. WGSL doesn't have an
            // `else` without an `if` so we chain a trivially-true
            // condition.
            writeln!(out, "{pad}{header} (true) {{").unwrap();
            for s in &arm.body {
                emit_stmt(out, s, indent + 4)?;
            }
        }
        IrPattern::Struct { name, bindings, .. } => {
            let kind_const = effect_op_kind_const(name);
            writeln!(
                out,
                "{pad}{header} (wgsl_match_scrut.kind == {kind_const}) {{"
            )
            .unwrap();
            // Bind the fields from payload slots. `EffectOp` carries
            // up to two u32 payload words (see module doc); the mapping
            // to named fields is encoded in `effect_op_field_slot`.
            for b in bindings {
                emit_match_field_binding(out, name, b, indent + 4)?;
            }
            for s in &arm.body {
                emit_stmt(out, s, indent + 4)?;
            }
        }
        IrPattern::Ctor { name, inner, .. } => {
            let kind_const = effect_op_kind_const(name);
            writeln!(
                out,
                "{pad}{header} (wgsl_match_scrut.kind == {kind_const}) {{"
            )
            .unwrap();
            // Positional ctors are unusual in physics.sim; we destructure
            // by slot position. Nested patterns that aren't bare binds
            // surface as Unsupported.
            for (i, inner_pat) in inner.iter().enumerate() {
                if let IrPattern::Bind { name: bname, .. } = inner_pat {
                    let sub_pad = " ".repeat(indent + 4);
                    let (ty, field) = effect_op_payload_slot(name, i)?;
                    writeln!(
                        out,
                        "{sub_pad}let {b}: {ty} = {field};",
                        b = wgsl_ident(bname)
                    )
                    .unwrap();
                } else {
                    return Err(EmitError::Unsupported(format!(
                        "ctor match sub-pattern {inner_pat:?} not supported — only bare binds"
                    )));
                }
            }
            for s in &arm.body {
                emit_stmt(out, s, indent + 4)?;
            }
        }
        other => {
            return Err(EmitError::Unsupported(format!(
                "match arm pattern {other:?} not supported in physics WGSL"
            )));
        }
    }
    // Note: we deliberately *don't* close the arm's brace here. The
    // caller closes the last arm's brace once all arms have contributed;
    // intermediate arms are closed via the next arm's `} else if`
    // prefix.
    Ok(())
}

fn emit_match_field_binding(
    out: &mut String,
    variant: &str,
    b: &IrPatternBinding,
    indent: usize,
) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    match &b.value {
        IrPattern::Bind { name, .. } => {
            let (ty, rhs) = effect_op_field_slot(variant, &b.field)?;
            writeln!(
                out,
                "{pad}let {bind}: {ty} = {rhs};",
                bind = wgsl_ident(name)
            )
            .unwrap();
            Ok(())
        }
        IrPattern::Wildcard => Ok(()),
        other => Err(EmitError::Unsupported(format!(
            "match struct sub-binding for `{}.{}` pattern {other:?} not supported",
            variant, b.field
        ))),
    }
}

/// Map a named `EffectOp` variant field to its payload slot + type.
/// Encodes the `assets/sim/enums.sim`-declared field order per variant
/// so the emitter can destructure without tagging each event shape
/// ahead of time. The integration phase is free to change the payload
/// packing; when it does, keep this table in lockstep.
fn effect_op_field_slot(variant: &str, field: &str) -> Result<(&'static str, String), EmitError> {
    // (variant, field, payload_slot, type)
    let table: &[(&str, &str, u32, &str)] = &[
        ("Damage", "amount", 0, "f32"),
        ("Heal", "amount", 0, "f32"),
        ("Shield", "amount", 0, "f32"),
        ("Stun", "duration_ticks", 0, "u32"),
        ("Slow", "duration_ticks", 0, "u32"),
        ("Slow", "factor_q8", 1, "u32"),
        ("TransferGold", "amount", 0, "i32"),
        ("ModifyStanding", "delta", 0, "i32"),
        ("CastAbility", "ability", 0, "u32"),
        ("CastAbility", "selector", 1, "u32"),
    ];
    for (v, f, slot, ty) in table {
        if *v == variant && *f == field {
            let rhs = match *ty {
                "f32" => format!("bitcast<f32>(wgsl_match_scrut.p{slot})"),
                "i32" => format!("bitcast<i32>(wgsl_match_scrut.p{slot})"),
                _ => format!("wgsl_match_scrut.p{slot}"),
            };
            return Ok((ty, rhs));
        }
    }
    Err(EmitError::Unsupported(format!(
        "EffectOp::{variant} has no field `{field}` in the physics WGSL mapping"
    )))
}

fn effect_op_payload_slot(variant: &str, idx: usize) -> Result<(&'static str, String), EmitError> {
    let (_, ty) = match (variant, idx) {
        ("Damage" | "Heal" | "Shield", 0) => ("amount", "f32"),
        ("Stun", 0) => ("duration_ticks", "u32"),
        ("Slow", 0) => ("duration_ticks", "u32"),
        ("Slow", 1) => ("factor_q8", "u32"),
        ("TransferGold", 0) => ("amount", "i32"),
        ("ModifyStanding", 0) => ("delta", "i32"),
        ("CastAbility", 0) => ("ability", "u32"),
        ("CastAbility", 1) => ("selector", "u32"),
        (_, _) => {
            return Err(EmitError::Unsupported(format!(
                "EffectOp::{variant} has no positional slot {idx}"
            )))
        }
    };
    let rhs = match ty {
        "f32" => format!("bitcast<f32>(wgsl_match_scrut.p{idx})"),
        "i32" => format!("bitcast<i32>(wgsl_match_scrut.p{idx})"),
        _ => format!("wgsl_match_scrut.p{idx}"),
    };
    Ok((ty, rhs))
}

fn effect_op_kind_const(variant: &str) -> String {
    format!("EFFECT_OP_KIND_{}", screaming_snake(variant))
}

// ---------------------------------------------------------------------------
// Emit (event push) lowering
// ---------------------------------------------------------------------------

fn emit_emit(out: &mut String, emit: &IrEmit, indent: usize) -> Result<(), EmitError> {
    let pad = " ".repeat(indent);
    if emit.event.is_none() {
        return Err(EmitError::UnresolvedEventInEmit(emit.event_name.clone()));
    }

    // Task 203 — chronicle events route to a dedicated GPU ring to keep
    // the hot path out of the observability tail. Detect by event name:
    // only `ChronicleEntry` has this special dispatch. Payload shape is
    // fixed (`template_id, agent, target, tick`), matching the DSL
    // emit sites in `assets/sim/physics.sim` (8 chronicle rules).
    if emit.event_name == "ChronicleEntry" {
        return emit_chronicle_emit(out, emit, indent, &pad);
    }

    let kind_const = event_kind_const(&emit.event_name);

    // Build payload args in declared (source) field order. `tick` is a
    // dedicated header word on the event record (not a payload slot), so
    // a spelled-out `tick:` field becomes the call's `tick` arg instead
    // of a payload slot. If no `tick:` is provided, the emitter appends
    // `wgsl_world_tick` (the CPU-driven tick uniform).
    //
    // Event-ring contract (task 188): `gpu_emit_event(kind, tick, p0..p7)`
    // takes 8 payload slots — matches `EventRecord.payload: array<u32, 8>`.
    let mut tick_arg: Option<String> = None;
    let mut slots: Vec<String> = Vec::new();
    for f in &emit.fields {
        let lowered = lower_expr(&f.value)?;
        if f.name == "tick" {
            // `tick` goes into the header word, not the payload array.
            // The DSL allows either an explicit `tick: N` field (e.g.
            // `cast` forwards the caster's event tick through nested
            // emits) or relies on the implicit append below. The lowered
            // value is already a u32 — no bitcast needed.
            tick_arg = Some(lowered);
            continue;
        }
        let slot = lower_emit_field(&f.value, &lowered)?;
        slots.push(slot);
    }
    let tick = tick_arg.unwrap_or_else(|| "wgsl_world_tick".to_string());
    while slots.len() < 8 {
        slots.push("0u".into());
    }
    if slots.len() > 8 {
        return Err(EmitError::Unsupported(format!(
            "event `{}` has {} payload fields; WGSL gpu_emit_event caps at 8",
            emit.event_name,
            slots.len()
        )));
    }
    writeln!(
        out,
        "{pad}gpu_emit_event({kind_const}, {tick}, {s0}, {s1}, {s2}, {s3}, {s4}, {s5}, {s6}, {s7});",
        s0 = slots[0],
        s1 = slots[1],
        s2 = slots[2],
        s3 = slots[3],
        s4 = slots[4],
        s5 = slots[5],
        s6 = slots[6],
        s7 = slots[7],
    )
    .unwrap();
    Ok(())
}

/// Task 203 — lower `emit ChronicleEntry { template_id, agent, target, tick? }`
/// to a call on the dedicated chronicle ring instead of the main
/// `gpu_emit_event`. Keeps observability volume off the hot path.
///
/// The DSL surface for `ChronicleEntry` is fixed at four fields in the
/// event catalog (`template_id: u32, agent: AgentId, target: AgentId`)
/// plus the implicit `tick` every event carries; the emitter enforces
/// this here by name so a new field added to the event would surface
/// as `Unsupported` rather than silently getting dropped.
fn emit_chronicle_emit(
    out: &mut String,
    emit: &IrEmit,
    _indent: usize,
    pad: &str,
) -> Result<(), EmitError> {
    let mut template_id: Option<String> = None;
    let mut agent: Option<String> = None;
    let mut target: Option<String> = None;
    let mut tick_arg: Option<String> = None;

    for f in &emit.fields {
        let lowered = lower_expr(&f.value)?;
        match f.name.as_str() {
            "template_id" => {
                // template_id is a u32 literal in every chronicle site;
                // lower_emit_field keeps the type discipline for it.
                template_id = Some(lower_emit_field(&f.value, &lowered)?);
            }
            "agent" => agent = Some(lowered),
            "target" => target = Some(lowered),
            "tick" => tick_arg = Some(lowered),
            other => {
                return Err(EmitError::Unsupported(format!(
                    "ChronicleEntry emit has unexpected field `{other}` — \
                     only template_id/agent/target/tick are allowed",
                )));
            }
        }
    }

    let template_id = template_id.ok_or_else(|| {
        EmitError::Unsupported(
            "ChronicleEntry emit missing `template_id` field".to_string(),
        )
    })?;
    let agent = agent.ok_or_else(|| {
        EmitError::Unsupported(
            "ChronicleEntry emit missing `agent` field".to_string(),
        )
    })?;
    let target = target.ok_or_else(|| {
        EmitError::Unsupported(
            "ChronicleEntry emit missing `target` field".to_string(),
        )
    })?;
    let tick = tick_arg.unwrap_or_else(|| "wgsl_world_tick".to_string());

    writeln!(
        out,
        "{pad}gpu_emit_chronicle_event({template_id}, {agent}, {target}, {tick});",
    )
    .unwrap();
    Ok(())
}

/// Convert a lowered payload expression into a u32 bitcast fit for
/// passing to `gpu_emit_event`. We keep the full lowered text so the
/// caller can pass e.g. `bitcast<u32>(amount)` directly when the source
/// field was typed as f32.
fn lower_emit_field(expr: &IrExprNode, lowered: &str) -> Result<String, EmitError> {
    // Best-effort payload cast. If the expression's resolved type is
    // f32, bitcast to u32; if i32, bitcast to u32; otherwise pass the
    // raw value (u32 / AgentId already u32-shaped). The emitter can't
    // consult a type registry here, so we infer from shape: integer
    // literals → `Xu`; float literals → `bitcast<u32>(Xf)`; everything
    // else passes through with a trailing `as u32`-equivalent when the
    // lowered text looks like a bitcast<f32> read (mirror-cast to u32).
    match &expr.kind {
        IrExpr::LitInt(v) => {
            // Prefer an unsigned literal so WGSL doesn't complain about
            // signed-to-unsigned narrowing.
            if *v < 0 {
                Ok(format!("bitcast<u32>({v})"))
            } else {
                Ok(format!("{v}u"))
            }
        }
        IrExpr::LitBool(b) => Ok(if *b { "1u".into() } else { "0u".into() }),
        IrExpr::LitFloat(_) => Ok(format!("bitcast<u32>({lowered})")),
        // Binary / unary / call nodes — we assume f32 if the lowered
        // text contains a `.`, otherwise u32. That's a heuristic, but
        // every emit site in physics.sim is either an id (u32) or a
        // plain scalar whose shape matches its DSL declaration.
        _ => {
            if lowered_looks_float(lowered) || lowered_looks_i32(lowered) {
                Ok(format!("bitcast<u32>({lowered})"))
            } else {
                Ok(lowered.to_string())
            }
        }
    }
}

/// Heuristic matching for i32-typed DSL bindings emitted as-is. The
/// emit_field wrapper bit-casts these to u32 so the emitted
/// `gpu_emit_event(...)` call always passes u32 args. Known i32-typed
/// identifiers in physics.sim include `delta` (ModifyStanding payload),
/// `reason` (EngagementBroken field), and `amount` when paired with
/// `bitcast<i32>` in the match destructure. The exhaustive list below
/// is pinned by the physics.sim rule set.
fn lowered_looks_i32(lowered: &str) -> bool {
    if lowered.contains("bitcast<i32>") {
        return true;
    }
    matches!(lowered, "delta" | "reason")
}

fn lowered_looks_float(lowered: &str) -> bool {
    // Heuristic: anything that decays to an `f32` in the Rust emitter
    // will contain either a float literal, a `bitcast<f32>` destructure
    // read, or one of our float-producing stub calls. Known bare
    // bindings that are f32-typed (e.g. `amount`, `damage`, `confidence`,
    // local lets named with float hints) also get treated as float.
    // A more robust approach would be a type-lattice pass; the
    // heuristic is pinned by the physics.sim rule set and guarded by
    // `payload_of` tests in unit suite.
    if lowered.contains("bitcast<f32>") {
        return true;
    }
    if lowered.contains(".0") || lowered.contains("e-") || lowered.contains("e+") {
        // float literals (`12.0`, `1e-3`, `1e+6`)
        return true;
    }
    // Known float-returning stubs.
    for hint in [
        "state_agent_hp",
        "state_agent_max_hp",
        "state_agent_shield_hp",
        "state_agent_attack_damage",
    ] {
        if lowered.contains(hint) {
            return true;
        }
    }
    // Known f32-typed bare identifiers in physics.sim rules. Because
    // lowering already strips surrounding expression context, a bare
    // name comes through as just the identifier string. Exact equality
    // ensures we don't false-positive substrings (`amount_x` wouldn't
    // match `amount`).
    matches!(
        lowered,
        "amount"
            | "damage"
            | "confidence"
            | "a"
            | "shield"
            | "absorbed"
            | "residual"
            | "cur_hp"
            | "new_hp"
            | "max_hp"
            | "hp_pct"
    )
}

// ---------------------------------------------------------------------------
// Expression lowering
// ---------------------------------------------------------------------------

fn lower_expr(node: &IrExprNode) -> Result<String, EmitError> {
    lower_expr_kind(&node.kind)
}

fn lower_expr_kind(kind: &IrExpr) -> Result<String, EmitError> {
    match kind {
        IrExpr::LitBool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        IrExpr::LitInt(v) => {
            if *v < 0 {
                Ok(format!("({v})"))
            } else {
                Ok(format!("{v}"))
            }
        }
        IrExpr::LitFloat(v) => Ok(render_float_wgsl(*v)),
        IrExpr::Local(_, name) => Ok(wgsl_ident(name)),
        IrExpr::NamespaceField { ns, field, .. } => lower_namespace_field(*ns, field),
        IrExpr::NamespaceCall { ns, method, args } => lower_namespace_call(*ns, method, args),
        IrExpr::Binary(op, lhs, rhs) => {
            // `<expr> == None` / `<expr> != None` — GPU-emittable rules
            // use the sentinel-encoding pattern (`engaged_with_or(x, x)
            // != x`) instead of `Option`, so we don't need to translate
            // `== None` here. If a rule still uses it, surface an error
            // so the author migrates to the sentinel idiom.
            if matches!(op, BinOp::Eq | BinOp::NotEq) {
                if let IrExpr::EnumVariant { ty, variant } = &rhs.kind {
                    if ty.is_empty() && variant == "None" {
                        return Err(EmitError::Unsupported(
                            "comparison against `None` in physics WGSL — use sentinel \
                             encoding (`x_or(foo, foo) != foo`) instead"
                                .into(),
                        ));
                    }
                }
            }
            let l = lower_expr(lhs)?;
            let r = lower_expr(rhs)?;
            Ok(format!("({l} {} {r})", binop_str(*op)?))
        }
        IrExpr::Unary(op, rhs) => {
            let r = lower_expr(rhs)?;
            Ok(format!("({}{r})", unop_str(*op)))
        }
        IrExpr::BuiltinCall(b, args) => lower_builtin_call(*b, args),
        IrExpr::If { cond, then_expr, else_expr } => {
            let c = lower_expr(cond)?;
            let t = lower_expr(then_expr)?;
            let e = match else_expr {
                Some(e) => lower_expr(e)?,
                None => {
                    return Err(EmitError::Unsupported("`if` expression without else".into()))
                }
            };
            // WGSL's ternary is `select(false_branch, true_branch, cond)`.
            Ok(format!("select({e}, {t}, {c})"))
        }
        IrExpr::EnumVariant { ty, variant } => {
            // Best-effort lowering: emit a `<TY>_<VARIANT>` constant.
            // The integration phase defines these for every enum the
            // physics rules actually reference (`TargetSelector`,
            // `EffectOp`, etc.).
            if ty.is_empty() {
                Ok(format!("ENUM_{}", screaming_snake(variant)))
            } else {
                Ok(format!("{}_{}", screaming_snake(ty), screaming_snake(variant)))
            }
        }
        IrExpr::Field { base, field_name, .. } => {
            let b = lower_expr(base)?;
            Ok(format!("{b}.{field_name}"))
        }
        other => Err(EmitError::Unsupported(format!(
            "expression shape {other:?} not supported in physics WGSL emission"
        ))),
    }
}

fn lower_namespace_field(ns: NamespaceId, field: &str) -> Result<String, EmitError> {
    match ns {
        NamespaceId::Config => {
            // Task 2.8 — world-scalar `config.combat.engagement_range`
            // migrated off the per-kernel `cfg` uniform onto the shared
            // `SimCfg` storage buffer (`sim_cfg.engagement_range`). The
            // mask/scoring emitters perform the analogous swap for their
            // own world-scalar reads. Any future `config.<ns>.<field>`
            // that lands in the shared SimCfg extends this match.
            match field {
                "combat.engagement_range" => Ok("sim_cfg.engagement_range".into()),
                _ => Err(EmitError::Unsupported(format!(
                    "config field `.{field}` not wired in physics WGSL \
                     (world-scalars live in SimCfg; kernel-locals in cfg)",
                ))),
            }
        }
        NamespaceId::World => {
            if field == "tick" {
                // Emitter-side alias: the emitter keeps spelling the
                // tick reference as the bare identifier `wgsl_world_tick`
                // so integration code can choose where the value comes
                // from. Physics resolves it to `sim_cfg.tick` via a
                // function-scope `let` injected by `wrap_rule_with_tick_alias`.
                return Ok("wgsl_world_tick".into());
            }
            Err(EmitError::Unsupported(format!(
                "world field `.{field}` not wired in physics WGSL"
            )))
        }
        NamespaceId::Cascade => {
            if field == "max_iterations" {
                // Task 2.8 — migrated from `cfg.cascade_max_iterations`
                // to the shared SimCfg storage buffer. The kernel reads
                // this in the `cast_depth` guard, which is one lookup per
                // physics-dispatch thread — cheap enough that running it
                // through the shared buffer instead of a dedicated uniform
                // has negligible overhead.
                return Ok("sim_cfg.cascade_max_iterations".into());
            }
            Err(EmitError::Unsupported(format!(
                "cascade field `.{field}` not wired in physics WGSL"
            )))
        }
        _ => Err(EmitError::Unsupported(format!(
            "namespace-field `{}.{field}` not supported",
            ns.name()
        ))),
    }
}

fn lower_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[crate::ir::IrCallArg],
) -> Result<String, EmitError> {
    let lowered: Vec<String> = args
        .iter()
        .map(|a| {
            if a.name.is_some() {
                return Err(EmitError::Unsupported(format!(
                    "named argument on stdlib call `{}.{method}` not supported",
                    ns.name()
                )));
            }
            lower_expr(&a.value)
        })
        .collect::<Result<_, _>>()?;
    match (ns, method) {
        // Agent scalar getters.
        (NamespaceId::Agents, "alive") => {
            expect_arity(args, 1, "agents.alive")?;
            // Alive-bitmap lowering: `agents.alive(x)` reads the
            // per-tick packed bitmap at binding slot 22 instead of
            // the full 64-byte `AgentSlot` cacheline. The bitmap is
            // written once at the top of each tick (sync: host-packed
            // in `run_batch`; resident: GPU `alive_pack_kernel`).
            // `alive_bit(slot)` + `slot_of(id)` are emitted in the
            // shader prefix; `state_agent_alive` is retired.
            Ok(format!(
                "(slot_of({id}) != 0xFFFFFFFFu && alive_bit(slot_of({id})))",
                id = lowered[0]
            ))
        }
        (NamespaceId::Agents, "hp") => {
            expect_arity(args, 1, "agents.hp")?;
            Ok(format!("state_agent_hp({})", lowered[0]))
        }
        (NamespaceId::Agents, "max_hp") => {
            expect_arity(args, 1, "agents.max_hp")?;
            Ok(format!("state_agent_max_hp({})", lowered[0]))
        }
        (NamespaceId::Agents, "shield_hp") => {
            expect_arity(args, 1, "agents.shield_hp")?;
            Ok(format!("state_agent_shield_hp({})", lowered[0]))
        }
        (NamespaceId::Agents, "attack_damage") => {
            expect_arity(args, 1, "agents.attack_damage")?;
            Ok(format!("state_agent_attack_damage({})", lowered[0]))
        }
        (NamespaceId::Agents, "stun_expires_at_tick") => {
            expect_arity(args, 1, "agents.stun_expires_at_tick")?;
            Ok(format!("state_agent_stun_expires_at({})", lowered[0]))
        }
        (NamespaceId::Agents, "slow_expires_at_tick") => {
            expect_arity(args, 1, "agents.slow_expires_at_tick")?;
            Ok(format!("state_agent_slow_expires_at({})", lowered[0]))
        }
        (NamespaceId::Agents, "slow_factor_q8") => {
            expect_arity(args, 1, "agents.slow_factor_q8")?;
            Ok(format!("state_agent_slow_factor_q8({})", lowered[0]))
        }
        (NamespaceId::Agents, "gold") => {
            expect_arity(args, 1, "agents.gold")?;
            Ok(format!("state_agent_gold({})", lowered[0]))
        }
        (NamespaceId::Agents, "cooldown_next_ready") => {
            expect_arity(args, 1, "agents.cooldown_next_ready")?;
            Ok(format!("state_agent_cooldown_next_ready({})", lowered[0]))
        }
        (NamespaceId::Agents, "engaged_with") => {
            // Returns the AgentId sentinel (0xFFFFFFFFu) when unset.
            // Consumers sentinel-check with the `_or` sibling below.
            expect_arity(args, 1, "agents.engaged_with")?;
            Ok(format!("state_agent_engaged_with({})", lowered[0]))
        }
        (NamespaceId::Agents, "engaged_with_or") => {
            expect_arity(args, 2, "agents.engaged_with_or")?;
            // Emit an inline sentinel select so the rule body can stay
            // in the scalar subset. `ENGAGED_SENTINEL` is a module-level
            // const (`0xFFFFFFFFu`) the integration layer exposes.
            Ok(format!(
                "select({fallback}, state_agent_engaged_with({id}), state_agent_engaged_with({id}) != 0xFFFFFFFFu)",
                id = lowered[0],
                fallback = lowered[1]
            ))
        }
        // Agent setters / mutators.
        (NamespaceId::Agents, "set_hp") => {
            expect_arity(args, 2, "agents.set_hp")?;
            Ok(format!("state_set_agent_hp({}, {})", lowered[0], lowered[1]))
        }
        (NamespaceId::Agents, "set_shield_hp") => {
            expect_arity(args, 2, "agents.set_shield_hp")?;
            Ok(format!(
                "state_set_agent_shield_hp({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Agents, "set_stun_expires_at_tick") => {
            expect_arity(args, 2, "agents.set_stun_expires_at_tick")?;
            Ok(format!(
                "state_set_agent_stun_expires_at({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Agents, "set_slow_expires_at_tick") => {
            expect_arity(args, 2, "agents.set_slow_expires_at_tick")?;
            Ok(format!(
                "state_set_agent_slow_expires_at({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Agents, "set_slow_factor_q8") => {
            expect_arity(args, 2, "agents.set_slow_factor_q8")?;
            Ok(format!(
                "state_set_agent_slow_factor_q8({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Agents, "set_gold") => {
            expect_arity(args, 2, "agents.set_gold")?;
            Ok(format!("state_set_agent_gold({}, {})", lowered[0], lowered[1]))
        }
        (NamespaceId::Agents, "add_gold") => {
            expect_arity(args, 2, "agents.add_gold")?;
            Ok(format!("state_add_agent_gold({}, {})", lowered[0], lowered[1]))
        }
        (NamespaceId::Agents, "sub_gold") => {
            expect_arity(args, 2, "agents.sub_gold")?;
            // Lower to add_gold with a negated delta to keep the stub
            // surface tight (one delta-add fn rather than paired add/sub).
            Ok(format!("state_add_agent_gold({}, -({}))", lowered[0], lowered[1]))
        }
        (NamespaceId::Agents, "adjust_standing") => {
            expect_arity(args, 3, "agents.adjust_standing")?;
            Ok(format!(
                "state_adjust_standing({}, {}, {})",
                lowered[0], lowered[1], lowered[2]
            ))
        }
        (NamespaceId::Agents, "kill") => {
            expect_arity(args, 1, "agents.kill")?;
            Ok(format!("state_kill_agent({})", lowered[0]))
        }
        (NamespaceId::Agents, "set_cooldown_next_ready") => {
            expect_arity(args, 2, "agents.set_cooldown_next_ready")?;
            Ok(format!(
                "state_set_agent_cooldown_next_ready({}, {})",
                lowered[0], lowered[1]
            ))
        }
        // agents.record_cast_cooldowns(caster, ability, t) on GPU writes
        // ONLY the global cooldown cursor (`t + ability.cooldown_ticks`),
        // preserving pre-Task-4 single-cursor semantics. CPU-side lowering
        // writes both global (GCD) + local cursors. Subsystem (3) — GPU
        // ability evaluation — will expose `ability_cooldowns` to GPU and
        // align the two paths.
        (NamespaceId::Agents, "record_cast_cooldowns") => {
            expect_arity(args, 3, "agents.record_cast_cooldowns")?;
            Ok(format!(
                "state_set_agent_cooldown_next_ready({caster}, ({t} + abilities_cooldown_ticks({ab})))",
                caster = lowered[0],
                ab = lowered[1],
                t = lowered[2],
            ))
        }
        (NamespaceId::Agents, "set_engaged_with") => {
            expect_arity(args, 2, "agents.set_engaged_with")?;
            Ok(format!(
                "state_set_agent_engaged_with({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Agents, "clear_engaged_with") => {
            expect_arity(args, 1, "agents.clear_engaged_with")?;
            Ok(format!("state_clear_agent_engaged_with({})", lowered[0]))
        }
        (NamespaceId::Agents, "record_memory") => {
            expect_arity(args, 5, "agents.record_memory")?;
            Ok(format!(
                "state_push_agent_memory({o}, {s}, {p}, {c}, {t})",
                o = lowered[0],
                s = lowered[1],
                p = lowered[2],
                c = lowered[3],
                t = lowered[4],
            ))
        }
        // Spatial queries.
        (NamespaceId::Query, "nearest_hostile_to") => {
            expect_arity(args, 2, "query.nearest_hostile_to")?;
            Ok(format!(
                "spatial_nearest_hostile_to({}, {})",
                lowered[0], lowered[1]
            ))
        }
        (NamespaceId::Query, "nearest_hostile_to_or") => {
            expect_arity(args, 3, "query.nearest_hostile_to_or")?;
            Ok(format!(
                "select({fallback}, spatial_nearest_hostile_to({a}, {r}), spatial_nearest_hostile_to({a}, {r}) != 0xFFFFFFFFu)",
                a = lowered[0],
                r = lowered[1],
                fallback = lowered[2]
            ))
        }
        // nearby_kin is only usable as a `for`-iter source; a raw call
        // expression to it has no scalar meaning.
        (NamespaceId::Query, "nearby_kin") => Err(EmitError::Unsupported(
            "`query.nearby_kin` must appear as the iter source of a `for` loop, \
             not as a scalar expression"
                .into(),
        )),
        // Ability registry.
        (NamespaceId::Abilities, "is_known") => {
            expect_arity(args, 1, "abilities.is_known")?;
            Ok(format!("abilities_is_known({})", lowered[0]))
        }
        (NamespaceId::Abilities, "cooldown_ticks") => {
            expect_arity(args, 1, "abilities.cooldown_ticks")?;
            Ok(format!("abilities_cooldown_ticks({})", lowered[0]))
        }
        (NamespaceId::Abilities, "effects") => Err(EmitError::Unsupported(
            "`abilities.effects` must appear as the iter source of a `for` loop, \
             not as a scalar expression"
                .into(),
        )),
        // ----------------------------------------------------------------
        // Roadmap §1 — Memberships (grammar stub). Grammar-only: the GPU
        // physics emitter refuses these until Subsystem §1 lands a GPU-
        // visible mirror of the per-agent memberships slab.
        // See `docs/superpowers/roadmap.md:161-211`.
        // ----------------------------------------------------------------
        (NamespaceId::Membership, "is_group_member")
        | (NamespaceId::Membership, "is_group_leader")
        | (NamespaceId::Membership, "can_join_group")
        | (NamespaceId::Membership, "is_outcast") => Err(EmitError::Unsupported(format!(
            "memberships primitive `membership::{method}` pending runtime impl"
        ))),
        // ----------------------------------------------------------------
        // Roadmap §3 — Relationships (grammar stub). GPU mirror pending
        // until the runtime lands. See `docs/superpowers/roadmap.md:279-311`.
        // ----------------------------------------------------------------
        (NamespaceId::Relationship, "is_hostile")
        | (NamespaceId::Relationship, "is_friendly")
        | (NamespaceId::Relationship, "knows_well") => Err(EmitError::Unsupported(format!(
            "relationships primitive `relationship::{method}` pending runtime impl"
        ))),
        // ----------------------------------------------------------------
        // Roadmap §6 — Theory-of-mind (grammar stub). GPU mirror pending
        // until the runtime lands. See `docs/superpowers/roadmap.md:447-506`.
        // ----------------------------------------------------------------
        (NamespaceId::TheoryOfMind, "believes_knows")
        | (NamespaceId::TheoryOfMind, "can_deceive")
        | (NamespaceId::TheoryOfMind, "is_surprised_by") => Err(EmitError::Unsupported(format!(
            "theory_of_mind primitive `theory_of_mind::{method}` pending runtime impl"
        ))),
        // ----------------------------------------------------------------
        // Roadmap §7 — Groups (grammar stub). GPU mirror pending until
        // the runtime lands. See `docs/superpowers/roadmap.md:510-574`.
        // ----------------------------------------------------------------
        (NamespaceId::Group, "exists")
        | (NamespaceId::Group, "is_active")
        | (NamespaceId::Group, "has_leader")
        | (NamespaceId::Group, "can_afford_from_treasury") => Err(EmitError::Unsupported(format!(
            "groups primitive `group::{method}` pending runtime impl"
        ))),
        // ----------------------------------------------------------------
        // Roadmap §12 — Quests (grammar stub). GPU mirror pending until
        // the runtime lands. See `docs/superpowers/roadmap.md:811-872`.
        // ----------------------------------------------------------------
        (NamespaceId::Quest, "can_accept")
        | (NamespaceId::Quest, "is_target")
        | (NamespaceId::Quest, "party_near_destination") => Err(EmitError::Unsupported(format!(
            "quests primitive `quest::{method}` pending runtime impl"
        ))),
        _ => Err(EmitError::Unsupported(format!(
            "stdlib call `{}.{method}` not supported in physics WGSL emission",
            ns.name()
        ))),
    }
}

fn lower_builtin_call(b: Builtin, args: &[crate::ir::IrCallArg]) -> Result<String, EmitError> {
    let lowered: Vec<String> = args
        .iter()
        .map(|a| {
            if a.name.is_some() {
                return Err(EmitError::Unsupported(format!(
                    "named argument on builtin `{}` not supported",
                    b.name()
                )));
            }
            lower_expr(&a.value)
        })
        .collect::<Result<_, _>>()?;
    match b {
        Builtin::Min => {
            expect_arity(args, 2, "min")?;
            Ok(format!("min({}, {})", lowered[0], lowered[1]))
        }
        Builtin::Max => {
            expect_arity(args, 2, "max")?;
            Ok(format!("max({}, {})", lowered[0], lowered[1]))
        }
        Builtin::Clamp => {
            expect_arity(args, 3, "clamp")?;
            Ok(format!(
                "clamp({}, {}, {})",
                lowered[0], lowered[1], lowered[2]
            ))
        }
        Builtin::Abs => {
            expect_arity(args, 1, "abs")?;
            Ok(format!("abs({})", lowered[0]))
        }
        Builtin::Floor => {
            expect_arity(args, 1, "floor")?;
            Ok(format!("floor({})", lowered[0]))
        }
        Builtin::Ceil => {
            expect_arity(args, 1, "ceil")?;
            Ok(format!("ceil({})", lowered[0]))
        }
        Builtin::Round => {
            expect_arity(args, 1, "round")?;
            Ok(format!("round({})", lowered[0]))
        }
        Builtin::Sqrt => {
            expect_arity(args, 1, "sqrt")?;
            Ok(format!("sqrt({})", lowered[0]))
        }
        Builtin::SaturatingAdd => {
            expect_arity(args, 2, "saturating_add")?;
            // WGSL has no native saturating_add. We emit a min-based
            // clamp to u32::MAX. Good enough for the `cast` rule's
            // `tick + cooldown_ticks` site, which is the only caller.
            Ok(format!(
                "select(0xFFFFFFFFu, ({a} + {b}), ({a} + {b}) >= {a})",
                a = lowered[0],
                b = lowered[1]
            ))
        }
        _ => Err(EmitError::Unsupported(format!(
            "builtin `{}` not supported in physics WGSL emission",
            b.name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn expect_arity(
    args: &[crate::ir::IrCallArg],
    expected: usize,
    name: &str,
) -> Result<(), EmitError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(EmitError::Unsupported(format!(
            "`{name}` expects {expected} arg(s), got {}",
            args.len()
        )))
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
        BinOp::Mod => "%",
    })
}

fn unop_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "!",
        UnOp::Neg => "-",
    }
}

fn render_float_wgsl(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Scan emit statements and verify every target event is resolvable.
fn scan_emits(stmts: &[IrStmt]) -> Result<(), EmitError> {
    for s in stmts {
        match s {
            IrStmt::Emit(e) => {
                if e.event.is_none() {
                    return Err(EmitError::UnresolvedEventInEmit(e.event_name.clone()));
                }
            }
            IrStmt::If { then_body, else_body, .. } => {
                scan_emits(then_body)?;
                if let Some(b) = else_body {
                    scan_emits(b)?;
                }
            }
            IrStmt::For { body, .. } => scan_emits(body)?,
            IrStmt::Match { arms, .. } => {
                for arm in arms {
                    scan_emits(&arm.body)?;
                }
            }
            IrStmt::Let { .. }
            | IrStmt::Expr(_)
            | IrStmt::SelfUpdate { .. }
            | IrStmt::BeliefObserve { .. } => {}
        }
    }
    Ok(())
}

/// Synthesize the implicit `tick: u32` on event fields (matches the Rust
/// emitter's convention).
fn fields_with_implicit_tick(event: &EventIR) -> Vec<EventField> {
    let mut out = event.fields.clone();
    if !out.iter().any(|f| f.name == "tick") {
        out.push(EventField {
            name: "tick".into(),
            ty: IrType::U32,
            span: crate::ast::Span::dummy(),
        });
    }
    out
}

/// Snake-case transform matching the other WGSL emitters.
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

fn screaming_snake(name: &str) -> String {
    snake_case(name).to_uppercase()
}

fn event_kind_const(name: &str) -> String {
    format!("EVENT_KIND_{}", screaming_snake(name))
}

fn handler_fn_name(rule: &str) -> String {
    format!("physics_{}", snake_case(rule))
}

/// Translate a DSL identifier into a WGSL-safe one. WGSL reserves a
/// long keyword list; if we hit one we prefix with `wgsl_`. The task
/// 187 brief lists `pass break continue loop let var fn if else return
/// true false match struct type for while switch`; a comprehensive
/// reserved-word list is below.
fn wgsl_ident(name: &str) -> String {
    if is_wgsl_reserved(name) {
        format!("wgsl_{name}")
    } else {
        name.to_string()
    }
}

fn is_wgsl_reserved(name: &str) -> bool {
    matches!(
        name,
        // Keywords (WGSL 2023-ish; superset of what task 187 lists)
        "alias"
            | "array"
            | "atomic"
            | "bitcast"
            | "bool"
            | "break"
            | "case"
            | "const"
            | "const_assert"
            | "continue"
            | "continuing"
            | "default"
            | "diagnostic"
            | "discard"
            | "else"
            | "enable"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "let"
            | "loop"
            | "mat2x2"
            | "mat2x3"
            | "mat2x4"
            | "mat3x2"
            | "mat3x3"
            | "mat3x4"
            | "mat4x2"
            | "mat4x3"
            | "mat4x4"
            | "override"
            | "requires"
            | "return"
            | "struct"
            | "switch"
            | "true"
            | "type"
            | "var"
            | "vec2"
            | "vec3"
            | "vec4"
            | "while"
            // Reserved words that are legal-but-risky (task 187 flags these)
            | "pass"
            | "match"
            | "macro"
            | "async"
            | "await"
            | "do"
            | "goto"
            | "yield"
            // WGSL-reserved words that show up as natural DSL binding
            // names (e.g. `EffectGoldTransfer { from: from, to: to }`).
            // The emitter prefixes these with `wgsl_` before use.
            | "from"
            | "to"
            | "in"
            | "out"
            | "target"
            | "private"
            | "new"
            | "old"
            | "ref"
            | "workgroup"
            | "storage"
            | "function"
            | "uniform"
            | "read"
            | "write"
            | "read_write"
            | "sampler"
            | "texture_1d"
            | "texture_2d"
            | "texture_2d_array"
            | "texture_3d"
            | "texture_cube"
            | "texture_cube_array"
            | "texture_multisampled_2d"
            | "texture_storage_1d"
            | "texture_storage_2d"
            | "texture_storage_2d_array"
            | "texture_storage_3d"
            | "texture_depth_2d"
            | "texture_depth_2d_array"
            | "texture_depth_cube"
            | "texture_depth_cube_array"
            | "texture_depth_multisampled_2d"
            | "i32"
            | "u32"
            | "f32"
            | "f16"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{
        EventField, EventIR, EventRef, IrCallArg, IrEmit, IrEventPattern, IrExpr, IrExprNode,
        IrFieldInit, IrPattern, IrPatternBinding, IrStmt, IrStmtMatchArm, LocalRef, NamespaceId,
        PhysicsHandlerIR, PhysicsIR,
    };

    fn span() -> Span {
        Span::dummy()
    }

    fn local(name: &str, id: u16) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Local(LocalRef(id), name.to_string()),
            span: span(),
        }
    }

    fn pattern_bind(field: &str, name: &str, id: u16) -> IrPatternBinding {
        IrPatternBinding {
            field: field.to_string(),
            value: IrPattern::Bind {
                name: name.to_string(),
                local: LocalRef(id),
            },
            span: span(),
        }
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

    fn lit_f(v: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::LitFloat(v),
            span: span(),
        }
    }

    // A basic "Damage" event: `{ actor: AgentId, target: AgentId,
    // amount: f32 }`. Mirrors `EffectDamageApplied` on-shape.
    fn damage_event() -> EventIR {
        EventIR {
            name: "EffectDamageApplied".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "amount".into(), ty: IrType::F32, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        }
    }

    fn agent_died_event() -> EventIR {
        EventIR {
            name: "AgentDied".into(),
            fields: vec![EventField {
                name: "agent_id".into(),
                ty: IrType::AgentId,
                span: span(),
            }],
            tags: vec![],
            annotations: vec![],
            span: span(),
        }
    }

    fn agent_attacked_event() -> EventIR {
        EventIR {
            name: "AgentAttacked".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "damage".into(), ty: IrType::F32, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        }
    }

    fn fear_spread_event() -> EventIR {
        EventIR {
            name: "FearSpread".into(),
            fields: vec![
                EventField { name: "observer".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "dead_kin".into(), ty: IrType::AgentId, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        }
    }

    // --- Test 1: simple `damage` rule (shield-first, emit, lethal path) ---

    /// Models the core of `physics damage`:
    ///   on EffectDamageApplied { actor: c, target: t, amount: a } {
    ///     if agents.alive(t) { if a > 0.0 {
    ///       let new_hp = max(agents.hp(t) - a, 0.0)
    ///       agents.set_hp(t, new_hp)
    ///     } }
    ///   }
    fn damage_rule() -> PhysicsIR {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "EffectDamageApplied".into(),
            event: Some(EventRef(0)),
            bindings: vec![
                pattern_bind("actor", "c", 0),
                pattern_bind("target", "t", 1),
                pattern_bind("amount", "a", 2),
            ],
            span: span(),
        });
        let cur_hp_minus_a = IrExprNode {
            kind: IrExpr::Binary(
                BinOp::Sub,
                Box::new(ns_call(NamespaceId::Agents, "hp", vec![local("t", 1)])),
                Box::new(local("a", 2)),
            ),
            span: span(),
        };
        let new_hp = IrExprNode {
            kind: IrExpr::BuiltinCall(
                Builtin::Max,
                vec![
                    IrCallArg { name: None, value: cur_hp_minus_a, span: span() },
                    IrCallArg { name: None, value: lit_f(0.0), span: span() },
                ],
            ),
            span: span(),
        };
        let inner = vec![
            IrStmt::Let {
                name: "new_hp".into(),
                local: LocalRef(3),
                value: new_hp,
                span: span(),
            },
            IrStmt::Expr(ns_call(
                NamespaceId::Agents,
                "set_hp",
                vec![local("t", 1), local("new_hp", 3)],
            )),
        ];
        let a_positive = IrExprNode {
            kind: IrExpr::Binary(
                BinOp::Gt,
                Box::new(local("a", 2)),
                Box::new(lit_f(0.0)),
            ),
            span: span(),
        };
        let alive_check = ns_call(NamespaceId::Agents, "alive", vec![local("t", 1)]);
        let body = vec![IrStmt::If {
            cond: alive_check,
            then_body: vec![IrStmt::If {
                cond: a_positive,
                then_body: inner,
                else_body: None,
                span: span(),
            }],
            else_body: None,
            span: span(),
        }];
        PhysicsIR {
            name: "damage".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        }
    }

    #[test]
    fn damage_rule_emits_wgsl_fn_with_kind_guard_and_stubs() {
        let p = damage_rule();
        let ev = damage_event();
        let ctx = EmitContext {
            events: std::slice::from_ref(&ev),
            event_tags: &[],
        };
        let out = emit_physics_wgsl(&p, &ctx).unwrap();

        // Function name + kind guard.
        assert!(
            out.contains("fn physics_damage(event_idx: u32)"),
            "missing fn signature:\n{out}"
        );
        assert!(
            out.contains("if (ev_rec.kind != EVENT_KIND_EFFECT_DAMAGE_APPLIED)"),
            "missing kind guard:\n{out}"
        );

        // Payload destructuring.
        assert!(out.contains("let c: u32 = ev_rec.payload[0];"), "missing actor bind:\n{out}");
        assert!(
            out.contains("let t: u32 = ev_rec.payload[1];"),
            "missing target bind:\n{out}"
        );
        assert!(
            out.contains("let a: f32 = bitcast<f32>(ev_rec.payload[2]);"),
            "missing amount bitcast bind:\n{out}"
        );

        // Stub calls — alive check + hp read + set_hp.
        // `agents.alive(t)` lowers to `alive_bit(slot_of(t))` against
        // the per-tick alive bitmap at binding 22.
        assert!(
            out.contains("alive_bit(slot_of(t))"),
            "missing alive bitmap call:\n{out}"
        );
        assert!(
            out.contains("state_agent_hp(t)"),
            "missing hp stub call:\n{out}"
        );
        assert!(
            out.contains("state_set_agent_hp(t, new_hp)"),
            "missing set_hp stub call:\n{out}"
        );

        // Builtin lowering.
        assert!(out.contains("max("), "missing max() builtin lowering:\n{out}");
    }

    // --- Test 2: `heal` rule (simpler — single clamp branch, no emit) ---

    fn heal_rule() -> PhysicsIR {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "EffectHealApplied".into(),
            event: Some(EventRef(0)),
            bindings: vec![
                pattern_bind("actor", "c", 0),
                pattern_bind("target", "t", 1),
                pattern_bind("amount", "a", 2),
            ],
            span: span(),
        });
        let cur_hp_plus_a = IrExprNode {
            kind: IrExpr::Binary(
                BinOp::Add,
                Box::new(ns_call(NamespaceId::Agents, "hp", vec![local("t", 1)])),
                Box::new(local("a", 2)),
            ),
            span: span(),
        };
        let new_hp = IrExprNode {
            kind: IrExpr::BuiltinCall(
                Builtin::Min,
                vec![
                    IrCallArg { name: None, value: cur_hp_plus_a, span: span() },
                    IrCallArg {
                        name: None,
                        value: ns_call(NamespaceId::Agents, "max_hp", vec![local("t", 1)]),
                        span: span(),
                    },
                ],
            ),
            span: span(),
        };
        let inner = vec![
            IrStmt::Let {
                name: "new_hp".into(),
                local: LocalRef(3),
                value: new_hp,
                span: span(),
            },
            IrStmt::Expr(ns_call(
                NamespaceId::Agents,
                "set_hp",
                vec![local("t", 1), local("new_hp", 3)],
            )),
        ];
        let a_positive = IrExprNode {
            kind: IrExpr::Binary(
                BinOp::Gt,
                Box::new(local("a", 2)),
                Box::new(lit_f(0.0)),
            ),
            span: span(),
        };
        let alive_check = ns_call(NamespaceId::Agents, "alive", vec![local("t", 1)]);
        let body = vec![IrStmt::If {
            cond: alive_check,
            then_body: vec![IrStmt::If {
                cond: a_positive,
                then_body: inner,
                else_body: None,
                span: span(),
            }],
            else_body: None,
            span: span(),
        }];
        PhysicsIR {
            name: "heal".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        }
    }

    #[test]
    fn heal_rule_emits_min_clamp_and_max_hp_stub() {
        let p = heal_rule();
        // Create a dummy EffectHealApplied event matching the rule's
        // signature (same shape as damage, different name).
        let ev = EventIR {
            name: "EffectHealApplied".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "amount".into(), ty: IrType::F32, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        };
        let ctx = EmitContext {
            events: std::slice::from_ref(&ev),
            event_tags: &[],
        };
        let out = emit_physics_wgsl(&p, &ctx).unwrap();

        assert!(
            out.contains("fn physics_heal(event_idx: u32)"),
            "missing heal fn signature:\n{out}"
        );
        assert!(
            out.contains("state_agent_max_hp(t)"),
            "missing max_hp stub:\n{out}"
        );
        assert!(out.contains("min("), "missing min() builtin lowering:\n{out}");
        assert!(
            out.contains("state_set_agent_hp(t, new_hp)"),
            "missing set_hp:\n{out}"
        );
    }

    // --- Test 3: `fear_spread_on_death` (for-loop + emit) ---

    fn fear_spread_rule() -> PhysicsIR {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentDied".into(),
            event: Some(EventRef(0)),
            bindings: vec![pattern_bind("agent_id", "dead", 0)],
            span: span(),
        });
        let body = vec![IrStmt::For {
            binder: LocalRef(1),
            binder_name: "kin".into(),
            iter: ns_call(
                NamespaceId::Query,
                "nearby_kin",
                vec![local("dead", 0), lit_f(12.0)],
            ),
            filter: None,
            body: vec![IrStmt::Emit(IrEmit {
                event_name: "FearSpread".into(),
                event: Some(EventRef(1)),
                fields: vec![
                    IrFieldInit {
                        name: "observer".into(),
                        value: local("kin", 1),
                        span: span(),
                    },
                    IrFieldInit {
                        name: "dead_kin".into(),
                        value: local("dead", 0),
                        span: span(),
                    },
                ],
                span: span(),
            })],
            span: span(),
        }];
        PhysicsIR {
            name: "fear_spread_on_death".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        }
    }

    #[test]
    fn fear_spread_rule_lowers_for_loop_and_gpu_emit_event() {
        let p = fear_spread_rule();
        let died = agent_died_event();
        let fear = fear_spread_event();
        let ctx = EmitContext {
            events: &[died, fear],
            event_tags: &[],
        };
        let out = emit_physics_wgsl(&p, &ctx).unwrap();

        assert!(
            out.contains("fn physics_fear_spread_on_death(event_idx: u32)"),
            "missing fn signature:\n{out}"
        );
        assert!(
            out.contains("if (ev_rec.kind != EVENT_KIND_AGENT_DIED)"),
            "missing kind guard:\n{out}"
        );
        assert!(
            out.contains("let dead: u32 = ev_rec.payload[0];"),
            "missing dead bind:\n{out}"
        );

        // Bounded iteration via count + indexed-at helpers.
        assert!(
            out.contains("spatial_nearby_kin_count(dead, 12.0)"),
            "missing nearby_kin_count stub:\n{out}"
        );
        assert!(
            out.contains("spatial_nearby_kin_at(dead, 12.0,"),
            "missing nearby_kin_at stub:\n{out}"
        );

        // Emit call — `tick` is the header word (slot before payload),
        // not a payload slot. Payload has 8 slots (matches event_ring);
        // fields beyond the event's declared arity are padded with `0u`.
        assert!(
            out.contains("gpu_emit_event(EVENT_KIND_FEAR_SPREAD, wgsl_world_tick, kin, dead, 0u, 0u, 0u, 0u, 0u, 0u);"),
            "missing gpu_emit_event for FearSpread:\n{out}"
        );
    }

    // --- Test 4: dispatcher covers every rule that fires on the kind ---

    #[test]
    fn dispatcher_routes_each_kind_to_its_rules() {
        let damage = damage_rule();
        let heal = heal_rule();
        let ev_damage = damage_event();
        let ev_heal = EventIR {
            name: "EffectHealApplied".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "amount".into(), ty: IrType::F32, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        };
        let ctx = EmitContext {
            events: &[ev_damage, ev_heal],
            event_tags: &[],
        };
        let rules = [damage, heal];
        let dispatcher = emit_physics_dispatcher_wgsl(&rules, &ctx);

        assert!(
            dispatcher.contains("fn physics_dispatch(event_idx: u32)"),
            "missing dispatcher fn:\n{dispatcher}"
        );
        assert!(
            dispatcher.contains("if (kind == EVENT_KIND_EFFECT_DAMAGE_APPLIED)"),
            "missing damage kind branch:\n{dispatcher}"
        );
        assert!(
            dispatcher.contains("physics_damage(event_idx);"),
            "missing damage call:\n{dispatcher}"
        );
        assert!(
            dispatcher.contains("if (kind == EVENT_KIND_EFFECT_HEAL_APPLIED)"),
            "missing heal kind branch:\n{dispatcher}"
        );
        assert!(
            dispatcher.contains("physics_heal(event_idx);"),
            "missing heal call:\n{dispatcher}"
        );
    }

    // --- Test 5: reserved-word dodging ---

    #[test]
    fn wgsl_ident_prefixes_reserved_words() {
        assert_eq!(wgsl_ident("actor"), "actor");
        // `target` is WGSL-reserved as of wgpu 26 / naga 26 — an updated
        // reserved-word list folds it in now, so the prefix fires.
        assert_eq!(wgsl_ident("target"), "wgsl_target");
        assert_eq!(wgsl_ident("pass"), "wgsl_pass");
        assert_eq!(wgsl_ident("break"), "wgsl_break");
        assert_eq!(wgsl_ident("loop"), "wgsl_loop");
        assert_eq!(wgsl_ident("let"), "wgsl_let");
        assert_eq!(wgsl_ident("match"), "wgsl_match");
        // New reservations the physics shader hit in the wild.
        assert_eq!(wgsl_ident("from"), "wgsl_from");
        assert_eq!(wgsl_ident("to"), "wgsl_to");
        assert_eq!(wgsl_ident("new"), "wgsl_new");
        assert_eq!(wgsl_ident("old"), "wgsl_old");
    }

    #[test]
    fn snake_case_matches_other_emitters() {
        assert_eq!(snake_case("FearSpread"), "fear_spread");
        assert_eq!(snake_case("AgentAttacked"), "agent_attacked");
        assert_eq!(snake_case("agent_attacked"), "agent_attacked");
    }

    // --- Test 6: multi-handler rules are rejected with a clear error ---

    #[test]
    fn multi_handler_rule_rejected() {
        let pattern_a = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentDied".into(),
            event: Some(EventRef(0)),
            bindings: vec![pattern_bind("agent_id", "a", 0)],
            span: span(),
        });
        let pattern_b = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentAttacked".into(),
            event: Some(EventRef(1)),
            bindings: vec![pattern_bind("actor", "a", 0)],
            span: span(),
        });
        let p = PhysicsIR {
            name: "multi".into(),
            handlers: vec![
                PhysicsHandlerIR {
                    pattern: pattern_a,
                    where_clause: None,
                    body: vec![],
                    span: span(),
                },
                PhysicsHandlerIR {
                    pattern: pattern_b,
                    where_clause: None,
                    body: vec![],
                    span: span(),
                },
            ],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        };
        let died = agent_died_event();
        let atkd = agent_attacked_event();
        let ctx = EmitContext {
            events: &[died, atkd],
            event_tags: &[],
        };
        let err = emit_physics_wgsl(&p, &ctx).unwrap_err();
        assert!(matches!(err, EmitError::Unsupported(_)), "unexpected err: {err:?}");
    }

    // --- Test 7: unresolved emit target surfaces as UnresolvedEventInEmit ---

    #[test]
    fn unresolved_emit_target_reports_error() {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentDied".into(),
            event: Some(EventRef(0)),
            bindings: vec![pattern_bind("agent_id", "a", 0)],
            span: span(),
        });
        let p = PhysicsIR {
            name: "broken_emit".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body: vec![IrStmt::Emit(IrEmit {
                    event_name: "NoSuchEvent".into(),
                    event: None,
                    fields: vec![],
                    span: span(),
                })],
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        };
        let died = agent_died_event();
        let ctx = EmitContext {
            events: std::slice::from_ref(&died),
            event_tags: &[],
        };
        let err = emit_physics_wgsl(&p, &ctx).unwrap_err();
        assert!(
            matches!(err, EmitError::UnresolvedEventInEmit(ref s) if s == "NoSuchEvent"),
            "unexpected err: {err:?}"
        );
    }

    // --- Test 8: match on EffectOp variants ---

    #[test]
    fn match_on_effect_op_emits_kind_branches() {
        // Build a minimal `cast` subset: match op { Damage { amount } =>
        // emit EffectDamageApplied { ... } }
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentCast".into(),
            event: Some(EventRef(0)),
            bindings: vec![
                pattern_bind("actor", "caster", 0),
                pattern_bind("target", "target", 1),
                pattern_bind("ability", "ab", 2),
            ],
            span: span(),
        });
        let match_stmt = IrStmt::Match {
            scrutinee: local("op", 3),
            arms: vec![IrStmtMatchArm {
                pattern: IrPattern::Struct {
                    name: "Damage".into(),
                    ctor: None,
                    bindings: vec![IrPatternBinding {
                        field: "amount".into(),
                        value: IrPattern::Bind {
                            name: "amount".into(),
                            local: LocalRef(4),
                        },
                        span: span(),
                    }],
                },
                body: vec![IrStmt::Emit(IrEmit {
                    event_name: "EffectDamageApplied".into(),
                    event: Some(EventRef(1)),
                    fields: vec![
                        IrFieldInit {
                            name: "actor".into(),
                            value: local("caster", 0),
                            span: span(),
                        },
                        IrFieldInit {
                            name: "target".into(),
                            value: local("target", 1),
                            span: span(),
                        },
                        IrFieldInit {
                            name: "amount".into(),
                            value: local("amount", 4),
                            span: span(),
                        },
                    ],
                    span: span(),
                })],
                span: span(),
            }],
            span: span(),
        };
        let cast_ev = EventIR {
            name: "AgentCast".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "ability".into(), ty: IrType::AbilityId, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        };
        let dmg_ev = damage_event();
        let p = PhysicsIR {
            name: "cast".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body: vec![match_stmt],
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        };
        let ctx = EmitContext {
            events: &[cast_ev, dmg_ev],
            event_tags: &[],
        };
        let out = emit_physics_wgsl(&p, &ctx).unwrap();
        assert!(
            out.contains("wgsl_match_scrut.kind == EFFECT_OP_KIND_DAMAGE"),
            "missing EffectOp::Damage branch:\n{out}"
        );
        assert!(
            out.contains("let amount: f32 = bitcast<f32>(wgsl_match_scrut.p0);"),
            "missing amount field binding:\n{out}"
        );
        assert!(
            out.contains("gpu_emit_event(EVENT_KIND_EFFECT_DAMAGE_APPLIED"),
            "missing gpu_emit_event for damage:\n{out}"
        );
    }

    // --- Task 203: chronicle emits route to the dedicated chronicle ring ---

    fn chronicle_entry_event() -> EventIR {
        EventIR {
            name: "ChronicleEntry".into(),
            fields: vec![
                EventField { name: "template_id".into(), ty: IrType::U32, span: span() },
                EventField { name: "agent".into(), ty: IrType::AgentId, span: span() },
                EventField { name: "target".into(), ty: IrType::AgentId, span: span() },
            ],
            tags: vec![],
            annotations: vec![],
            span: span(),
        }
    }

    fn lit_u32(v: u64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::LitInt(v as i64),
            span: span(),
        }
    }

    /// `chronicle_death` on AgentDied { agent_id: a } emits
    ///   `ChronicleEntry { template_id: 1, agent: a, target: a }`.
    /// The WGSL emitter must route this through `gpu_emit_chronicle_event`
    /// rather than `gpu_emit_event(EVENT_KIND_CHRONICLE_ENTRY, ...)`.
    #[test]
    fn chronicle_emit_routes_to_chronicle_ring_helper() {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentDied".into(),
            event: Some(EventRef(0)),
            bindings: vec![pattern_bind("agent_id", "a", 0)],
            span: span(),
        });
        let body = vec![IrStmt::Emit(IrEmit {
            event_name: "ChronicleEntry".into(),
            event: Some(EventRef(1)),
            fields: vec![
                IrFieldInit { name: "template_id".into(), value: lit_u32(1), span: span() },
                IrFieldInit { name: "agent".into(), value: local("a", 0), span: span() },
                IrFieldInit { name: "target".into(), value: local("a", 0), span: span() },
            ],
            span: span(),
        })];
        let p = PhysicsIR {
            name: "chronicle_death".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        };
        let died = agent_died_event();
        let chronicle = chronicle_entry_event();
        let ctx = EmitContext {
            events: &[died, chronicle],
            event_tags: &[],
        };
        let out = emit_physics_wgsl(&p, &ctx).unwrap();

        // The new emission site uses the chronicle helper.
        assert!(
            out.contains("gpu_emit_chronicle_event(1u, a, a, wgsl_world_tick);"),
            "chronicle emit should route to gpu_emit_chronicle_event:\n{out}"
        );
        // It must NOT fall through to the main-ring helper.
        assert!(
            !out.contains("gpu_emit_event(EVENT_KIND_CHRONICLE_ENTRY"),
            "chronicle emit must not hit the main event ring:\n{out}"
        );
    }

    #[test]
    fn chronicle_emit_rejects_unknown_fields() {
        let pattern = IrPhysicsPattern::Kind(IrEventPattern {
            name: "AgentDied".into(),
            event: Some(EventRef(0)),
            bindings: vec![pattern_bind("agent_id", "a", 0)],
            span: span(),
        });
        let body = vec![IrStmt::Emit(IrEmit {
            event_name: "ChronicleEntry".into(),
            event: Some(EventRef(1)),
            fields: vec![
                IrFieldInit { name: "template_id".into(), value: lit_u32(1), span: span() },
                IrFieldInit { name: "agent".into(), value: local("a", 0), span: span() },
                IrFieldInit { name: "target".into(), value: local("a", 0), span: span() },
                // Bogus field — a future schema change would surface
                // here rather than silently getting dropped.
                IrFieldInit { name: "extra".into(), value: lit_u32(42), span: span() },
            ],
            span: span(),
        })];
        let p = PhysicsIR {
            name: "chronicle_with_extra".into(),
            handlers: vec![PhysicsHandlerIR {
                pattern,
                where_clause: None,
                body,
                span: span(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(),
        };
        let died = agent_died_event();
        let chronicle = chronicle_entry_event();
        let ctx = EmitContext {
            events: &[died, chronicle],
            event_tags: &[],
        };
        let err = emit_physics_wgsl(&p, &ctx).unwrap_err();
        match err {
            EmitError::Unsupported(s) => {
                assert!(
                    s.contains("unexpected field `extra`"),
                    "error message should name the bogus field: {s}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
