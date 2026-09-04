//! Two-pass name resolution: AST → IR.
//!
//! Pass 1: collect all top-level decl names into a `SymbolTable`, assign IR
//! indices. Duplicate names (same kind) are errors.
//! Pass 2: walk each decl's bodies, resolving identifiers against a local
//! scope stack, the stdlib symbol table, and the top-level decls.
//!
//! Unresolvable call callees become `UnresolvedCall` (flagged for a later
//! milestone, 1b). Bare unresolved identifiers are errors.

use std::collections::HashMap;

use crate::ast::{self, ActionHeadShape, AssertExpr, BinOp, Decl, Expr, ExprKind, Program, Span, Stmt};
use crate::ir::*;
use crate::resolve_error::ResolveError;

// ---------------------------------------------------------------------------
// Stdlib symbol table
// ---------------------------------------------------------------------------

mod stdlib {
    use super::*;

    pub fn seed(symbols: &mut SymbolTable) {
        let prims = [
            ("bool", IrType::Bool),
            ("i8",  IrType::I8),
            ("u8",  IrType::U8),
            ("i16", IrType::I16),
            ("u16", IrType::U16),
            ("i32", IrType::I32),
            ("u32", IrType::U32),
            ("i64", IrType::I64),
            ("u64", IrType::U64),
            ("f32", IrType::F32),
            ("f64", IrType::F64),
            ("vec3", IrType::Vec3),
            ("string", IrType::String),
            ("String", IrType::String),
            ("AgentId", IrType::AgentId),
            // `Agent` as a type alias for AgentId. The DSL surface uses
            // `Agent` for agent-typed parameters (`view threat_level(a:
            // Agent, b: Agent) -> f32` etc.); without this entry,
            // `Agent` falls through to IrType::Named("Agent") and the
            // CG type-mapper widens it to u32, causing view-call type
            // checks to fail when the call passes typed AgentId values.
            // Phase 6 Task 2.5 surfaced this when wildcard short-circuit
            // unblocked real view-call type-checks.
            ("Agent", IrType::AgentId),
            ("ItemId", IrType::ItemId),
            ("GroupId", IrType::GroupId),
            ("QuestId", IrType::QuestId),
            ("AuctionId", IrType::AuctionId),
            ("EventId", IrType::EventId),
            ("AbilityId", IrType::AbilityId),
            // Tick / other pseudo-primitives commonly seen in fixtures.
            ("Tick", IrType::U64),
        ];
        for (n, t) in prims {
            symbols.stdlib_types.insert(n.to_string(), t);
        }
        // Aggregations + quantifiers (parsed as dedicated AST nodes, but we
        // still reserve the names so they don't shadow).
        symbols.builtins.insert("count".into(), Builtin::Count);
        symbols.builtins.insert("sum".into(), Builtin::Sum);
        symbols.builtins.insert("forall".into(), Builtin::Forall);
        symbols.builtins.insert("exists".into(), Builtin::Exists);
        // Spatial.
        symbols.builtins.insert("distance".into(), Builtin::Distance);
        symbols.builtins.insert("planar_distance".into(), Builtin::PlanarDistance);
        symbols.builtins.insert("z_separation".into(), Builtin::ZSeparation);
        // ID dereference.
        symbols.builtins.insert("entity".into(), Builtin::Entity);
        // Numeric.
        symbols.builtins.insert("min".into(), Builtin::Min);
        symbols.builtins.insert("max".into(), Builtin::Max);
        symbols.builtins.insert("clamp".into(), Builtin::Clamp);
        symbols.builtins.insert("abs".into(), Builtin::Abs);
        symbols.builtins.insert("floor".into(), Builtin::Floor);
        symbols.builtins.insert("ceil".into(), Builtin::Ceil);
        symbols.builtins.insert("round".into(), Builtin::Round);
        symbols.builtins.insert("ln".into(), Builtin::Ln);
        symbols.builtins.insert("log2".into(), Builtin::Log2);
        symbols.builtins.insert("log10".into(), Builtin::Log10);
        symbols.builtins.insert("sqrt".into(), Builtin::Sqrt);
        // Vec3 math primitives. WGSL natives — emit as `length(...)` /
        // `dot(...)` / `normalize(...)`. Used by movement / flocking
        // physics rules + speed-cap invariants.
        symbols.builtins.insert("normalize".into(), Builtin::Normalize);
        symbols.builtins.insert("length".into(), Builtin::Length);
        symbols.builtins.insert("dot".into(), Builtin::Dot);
        symbols.builtins.insert("saturating_add".into(), Builtin::SaturatingAdd);
        // Vec3 constructor. Three-arg call returning Vec3F32; operands are F32.
        symbols.builtins.insert("vec3".into(), Builtin::Vec3);
        // Explicit numeric casts: `f32(x)` / `u32(x)` / `i32(x)`. The
        // names overlap with the type-name entries in `stdlib_types`;
        // resolution disambiguates structurally — `f32` as a *call*
        // target resolves to the builtin (here), `f32` as a bare type
        // position resolves through `stdlib_types`.
        symbols.builtins.insert("f32".into(), Builtin::F32Cast);
        symbols.builtins.insert("u32".into(), Builtin::U32Cast);
        symbols.builtins.insert("i32".into(), Builtin::I32Cast);
        // `next_waypoint(group)` — design-target placeholder for the
        // crowd_navigation fixture's group-leader goal-handoff. Returns
        // a sentinel `vec3(0,0,0)` until a real quest/landmark runtime
        // lands. See `Builtin::NextWaypoint` for the per-variant doc.
        symbols.builtins.insert("next_waypoint".into(), Builtin::NextWaypoint);

        // Typed namespaces. Each has its own field / method schema below.
        for (name, id) in [
            ("world", NamespaceId::World),
            ("cascade", NamespaceId::Cascade),
            ("event", NamespaceId::Event),
            ("mask", NamespaceId::Mask),
            ("action", NamespaceId::Action),
            ("rng", NamespaceId::Rng),
            ("query", NamespaceId::Query),
            // `spatial.<name>(...)` — references to declared
            // `spatial_query <name>` filters (Phase 7 Task 4).
            // Method dispatch checks `symbols.spatial_queries`; an
            // unknown name surfaces `UnknownSpatialQuery` rather
            // than carrying through as an opaque `NamespaceCall`.
            ("spatial", NamespaceId::Spatial),
            ("voxel", NamespaceId::Voxel),
            ("config", NamespaceId::Config),
            // `view::<name>(...)` disambiguation namespace. The resolver
            // rewrites calls of this shape into `IrExpr::ViewCall(ref,
            // args)` once it resolves `<name>` against `symbols.views`.
            // No declared fields — only method-call syntax is valid.
            ("view", NamespaceId::View),
            // Legacy collection / accessor namespaces — kept for iteration
            // source use (`count(a in agents ...)`). No declared fields.
            ("agents", NamespaceId::Agents),
            ("items", NamespaceId::Items),
            ("groups", NamespaceId::Groups),
            ("quests", NamespaceId::Quests),
            ("auctions", NamespaceId::Auctions),
            ("tick", NamespaceId::Tick),
            // Ability-registry accessor: `is_known(id)`, `cooldown_ticks(id)`,
            // `effects(id)`. Used by the `cast` physics rule.
            ("abilities", NamespaceId::Abilities),
            // Singular alias for `abilities` — lets designers write the
            // natural singular form `ability::on_cooldown(<slot>)` in mask /
            // physics predicates. Shares the same method schema as `abilities::`.
            ("ability", NamespaceId::Abilities),
            // Terrain backend accessor — MVP Task 81. Sole method:
            // `terrain.line_of_sight(from, to) -> bool`. Routed through
            // `SimState.terrain: Arc<dyn TerrainQuery>` at emit time.
            // The flat-plane default keeps every legacy scoring /
            // physics path unchanged; examples and future mask rows
            // opt in by reading through this namespace.
            ("terrain", NamespaceId::Terrain),
            // Roadmap §1 — Memberships. Grammar stub (no runtime state
            // yet); predicates return bool and emitters return
            // `Unsupported`. See `docs/superpowers/roadmap.md:161-211`.
            ("membership", NamespaceId::Membership),
            // Roadmap §3 — Relationships. Grammar stub (no runtime state
            // yet). See `docs/superpowers/roadmap.md:279-311`.
            ("relationship", NamespaceId::Relationship),
            // Roadmap §6 — Theory-of-mind. Grammar stub (no runtime state
            // yet). See `docs/superpowers/roadmap.md:447-506`.
            ("theory_of_mind", NamespaceId::TheoryOfMind),
            // Roadmap §7 — Groups. Grammar stub (Pod shape exists from
            // Plan 1 T16; instance data pending). Singular `group`
            // — distinct from the legacy collection accessor `groups`.
            // See `docs/superpowers/roadmap.md:510-574`.
            ("group", NamespaceId::Group),
            // Roadmap §12 — Quests. Grammar stub (Pod shape exists from
            // Plan 1 T16; instance data pending). Singular `quest` —
            // distinct from the legacy collection accessor `quests`.
            // See `docs/superpowers/roadmap.md:811-872`.
            ("quest", NamespaceId::Quest),
            // Plan G G3f (2026-05-09) — Threats scoring primitives.
            // Methods (`in_zone`, `intensity_at`, `nearest`,
            // `dir_away_from_nearest`) dispatch directly to
            // `Builtin::Threats*` variants in `resolve_call` — bypasses
            // the generic `NamespaceCall` route so the lowering pass
            // sees a single closed-set Builtin enum to dispatch on.
            // The threats materialised view (G3g, future) wires the
            // per-cell walk; today's lowering emits sentinel values.
            // See `docs/plans/g3_threats_view_design.md`.
            ("threats", NamespaceId::Threats),
            // Sim-wide event-trace accessors used by metrics /
            // invariants / probes (NOT physics / view bodies):
            //   * `events.this_tick` (NamespaceField; iter source)
            //   * `events.at_tick(t)` / `events.range(lo, hi)`
            //     (NamespaceCall returning an event-list iter source)
            //   * `events.kind_count(KIND)` (NamespaceCall returning u32)
            // Per-field / per-method schemas live in `field_type` /
            // `method_sig` below. The events namespace stays META-LEVEL
            // today: shape classifiers in `cg::emit::{metrics,probes,
            // invariants}` recognise the IR node + emit per-name SKIP
            // setters the runtime fills in. Physics / view bodies that
            // reach for `events.*` fail at lowering with
            // `UnsupportedNamespaceCall` (not yet GPU-side).
            ("events", NamespaceId::Events),
            // Static lookup tables — `tables.<name>(<idx>)` reads a
            // u32 cell from a `table <name>: u32[N] = […]` decl.
            // Per-name method routing happens in `resolve_call` via
            // a fresh `Builtin::TableLookup` variant; the namespace
            // entry exists so `tables.X(idx)` doesn't fall through
            // to `UnsupportedNamespaceCall` at lower time.
            ("tables", NamespaceId::Tables),
            // Voxel-region-indices spec Phase 4b — navgrid index reads.
            // `navgrid.walkable(cx, cz) -> bool` resolves the per-region
            // navgrid the engine_voxel::build_navgrid built at host time.
            // Used in physics rules to gate movement by terrain walkability.
            ("navgrid", NamespaceId::Navgrid),
        ] {
            symbols.stdlib_namespaces.insert(name.to_string(), id);
        }

        // Engine stdlib sum types visible to the DSL. These aren't declared
        // by the `enum <Name> { ... }` surface (which only supports unit
        // variants) — they're struct-shape enums owned by the engine
        // (`EffectOp`, `TargetSelector` in `crates/engine/src/ability/
        // program.rs`). We seed the symbol table with their names + variants
        // so `match` patterns can reference them and `TargetSelector::Target`
        // resolves to an `EnumVariant` expression. The emitter rewrites the
        // path to `crate::ability::<Ty>::<Variant>` at emission time.
        seed_stdlib_enum(
            symbols,
            "TargetSelector",
            &["Target", "Caster"],
        );
        seed_stdlib_enum(
            symbols,
            "EffectOp",
            &[
                "Damage",
                "Heal",
                "Shield",
                "Stun",
                "Slow",
                "TransferGold",
                "ModifyStanding",
                "CastAbility",
            ],
        );
    }

    /// Register a stdlib-owned enum (struct-shape or otherwise) under a
    /// synthetic `EnumRef` so `resolve_ident` recognises `<Ty>::<Variant>`
    /// and bare variant names starting uppercase. Emitter decides the
    /// concrete Rust path; see `qualified_variant_name` in
    /// `emit_physics.rs`.
    fn seed_stdlib_enum(symbols: &mut SymbolTable, name: &str, variants: &[&str]) {
        // Synthetic ref — not stored in any `Compilation::enums` slot, so
        // the index is arbitrary. Using `u16::MAX - N` keeps stdlib refs
        // out of the user-declared range.
        let idx = (u16::MAX as usize)
            .saturating_sub(symbols.enums.len() + 1) as u16;
        let variants_vec: Vec<String> = variants.iter().map(|s| s.to_string()).collect();
        symbols
            .enums
            .entry(name.to_string())
            .or_insert((EnumRef(idx), variants_vec.clone()));
        for v in variants {
            // `or_insert_with` so a later user-declared enum that also owns
            // a variant of this name wins (matches the variant-owner contract
            // in `Decl::Enum` handling).
            symbols
                .enum_variant_owner
                .entry(v.to_string())
                .or_insert_with(|| name.to_string());
        }
    }

    /// Field schema for typed stdlib namespaces.
    ///
    /// Returns `None` if the namespace doesn't declare this field — which
    /// either means the field is unknown (a later pass may error) or the
    /// namespace is a legacy collection without a declared field schema.
    pub fn field_type(ns: NamespaceId, field: &str) -> Option<IrType> {
        match (ns, field) {
            (NamespaceId::World, "tick") => Some(IrType::U64),
            (NamespaceId::World, "seed") => Some(IrType::U64),
            (NamespaceId::World, "n_agents_alive") => Some(IrType::U32),
            (NamespaceId::Cascade, "iterations") => Some(IrType::U32),
            (NamespaceId::Cascade, "phase") => Some(IrType::Enum {
                name: "CascadePhase".into(),
                variants: vec!["Pre".into(), "Event".into(), "Post".into()],
            }),
            // Compile-time constant — the cascade framework's per-tick
            // iteration ceiling (`crate::cascade::MAX_CASCADE_ITERATIONS`,
            // currently 8). Used by the `cast` physics rule to bound the
            // recursion depth of nested `CastAbility` effects. Typed as
            // `u8` so the emitter can compare it directly to
            // `Event::AgentCast.depth: u8` without a widening cast.
            (NamespaceId::Cascade, "max_iterations") => Some(IrType::U8),
            (NamespaceId::Event, "kind") => Some(IrType::Named("EventKindId".into())),
            (NamespaceId::Event, "tick") => Some(IrType::U64),
            // `events.this_tick` — list of events recorded on the
            // current world.tick. Iter source for `count(e in
            // events.this_tick where e.kind == X)` folds in metric /
            // invariant / probe bodies. The element type is left
            // un-narrowed (Unknown) because the trace stream carries a
            // tagged-union of every declared event kind in the .sim
            // file; the `e.kind == X` filter inside the fold body is
            // the discriminant test. Per-element field reads (`e.kind`,
            // `e.by`, ...) flow through `IrExpr::Field` on the binder
            // local without a per-field schema today.
            (NamespaceId::Events, "this_tick") => {
                Some(IrType::List(Box::new(IrType::Unknown)))
            }
            (NamespaceId::Mask, "rejections") => Some(IrType::U64),
            (NamespaceId::Action, "head") => Some(IrType::Named("ActionHeadKind".into())),
            (NamespaceId::Action, "target") => {
                Some(IrType::Optional(Box::new(IrType::Named("AnyId".into()))))
            }
            _ => None,
        }
    }

    /// Method schema for typed stdlib namespaces: returns `(arity, return_ty)`
    /// when the method is declared. Arg types are documented in `stdlib.md`
    /// and enforced by a later type-checking pass — 1a only checks arity.
    pub fn method_sig(ns: NamespaceId, method: &str) -> Option<(usize, IrType)> {
        match (ns, method) {
            (NamespaceId::Rng, "uniform") => Some((2, IrType::F32)),
            (NamespaceId::Rng, "gauss") => Some((2, IrType::F32)),
            (NamespaceId::Rng, "coin") => Some((0, IrType::Bool)),
            // Gap #C close (stdlib_math_probe, 2026-05-04): the
            // `rng.uniform_int(lo, hi)` surface advertised
            // `(i32, i32) -> i32`, but the DSL has no i32 source
            // (no literal suffix, no cast surface, no fixture with
            // an i32 field), making the call unreachable. Switched
            // to `(u32, u32) -> u32` so a bare-positive-literal pair
            // (e.g. `rng.uniform_int(0, 4)`) — which lowers to
            // `LitValue::U32` per `IrExpr::LitInt` — typechecks
            // straight through. See gap report
            // `docs/superpowers/notes/2026-05-04-stdlib_math_probe.md`.
            (NamespaceId::Rng, "uniform_int") => Some((2, IrType::U32)),
            (NamespaceId::Query, "nearby_agents") => {
                Some((2, IrType::List(Box::new(IrType::AgentId))))
            }
            (NamespaceId::Query, "within_planar") => {
                Some((2, IrType::List(Box::new(IrType::AgentId))))
            }
            (NamespaceId::Query, "nearby_items") => {
                Some((2, IrType::List(Box::new(IrType::ItemId))))
            }
            (NamespaceId::Voxel, "neighbors_above") => {
                Some((1, IrType::List(Box::new(IrType::Vec3))))
            }
            (NamespaceId::Voxel, "neighbors_below") => {
                Some((1, IrType::List(Box::new(IrType::Vec3))))
            }
            (NamespaceId::Voxel, "surface_height") => Some((2, IrType::I32)),
            // `agents` accessors used by physics rules. `hp`/`shield_hp` are
            // getters; `set_hp`/`set_shield_hp` are mutators returning unit;
            // `alive` predicates the slot; `kill` flips the alive bit and
            // tears the agent out of the spatial index. See
            // `docs/dsl/stdlib.md` for the canonical signatures.
            (NamespaceId::Agents, "alive") => Some((1, IrType::Bool)),
            (NamespaceId::Agents, "pos") => Some((1, IrType::Vec3)),
            // Phase-7-post-nuke unlock #3: vec3 read + write surface so
            // physics bodies can mutate per-agent position / velocity.
            // Companion to `pos`; written by `set_pos` / `set_vel`.
            (NamespaceId::Agents, "vel") => Some((1, IrType::Vec3)),
            (NamespaceId::Agents, "set_pos") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "set_vel") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "hp") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "max_hp") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "shield_hp") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "attack_damage") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "set_hp") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "set_shield_hp") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "kill") => Some((1, IrType::Unknown)),
            // Status-effect accessors. Task 143 retired the per-tick
            // decrement pass; stun/slow are now stored as absolute expiry
            // ticks (`world.tick < expires_at_tick` means active). The
            // `slow_factor_q8` accessor still reads the raw q8 slot; the
            // `slow_factor` lazy view wraps that with the expiry check.
            (NamespaceId::Agents, "stun_expires_at_tick") => Some((1, IrType::U32)),
            (NamespaceId::Agents, "set_stun_expires_at_tick") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "slow_expires_at_tick") => Some((1, IrType::U32)),
            (NamespaceId::Agents, "set_slow_expires_at_tick") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "slow_factor_q8") => Some((1, IrType::I16)),
            (NamespaceId::Agents, "set_slow_factor_q8") => Some((2, IrType::Unknown)),
            // Inventory / economy.
            (NamespaceId::Agents, "gold") => Some((1, IrType::I64)),
            (NamespaceId::Agents, "set_gold") => Some((2, IrType::Unknown)),
            // Adds `delta` to the agent's gold using `i64::wrapping_add` —
            // the legacy `TransferGoldHandler` uses wrapping arithmetic so
            // i64 overflow doesn't panic in debug builds. No-op if the slot
            // is absent.
            (NamespaceId::Agents, "add_gold") => Some((2, IrType::Unknown)),
            // Subtracts `delta` from the agent's gold using `i64::wrapping_sub`.
            // Paired with `add_gold` for the gold-transfer handler so the
            // two sides of a transfer each use the legacy wrapping op.
            (NamespaceId::Agents, "sub_gold") => Some((2, IrType::Unknown)),
            // Standing (symmetric pair storage, clamped [-1000, 1000] by
            // the `@materialized` `standing` view — lowering targets
            // `state.views.standing.adjust(...)`).
            (NamespaceId::Agents, "adjust_standing") => Some((3, IrType::Unknown)),
            (NamespaceId::Agents, "hunger") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "thirst") => Some((1, IrType::F32)),
            (NamespaceId::Agents, "rest_timer") => Some((1, IrType::F32)),
            // Species-level hostility predicate. Returns `false` when either
            // agent lacks a creature type (dead / uninitialised slot). The
            // DSL-declared `view is_hostile(a, b)` body forwards here so the
            // hostility matrix stays on `CreatureType::is_hostile_to` without
            // a hand-written `crate::rules::*` shim.
            (NamespaceId::Agents, "is_hostile_to") => Some((2, IrType::Bool)),
            // Audit fix HIGH #4 — primitive for the `record_memory` physics
            // rule. Args: `(observer, source, payload, confidence, tick)`.
            // Quantises `confidence` to q8, constructs a `MemoryEvent`, and
            // pushes it onto the observer's cold memory ring.
            (NamespaceId::Agents, "record_memory") => Some((5, IrType::Unknown)),
            // Cooldown accessor — used by the cast handler to set the
            // caster's next-ready tick after all effects dispatch.
            (NamespaceId::Agents, "cooldown_next_ready") => Some((1, IrType::U32)),
            (NamespaceId::Agents, "set_cooldown_next_ready") => Some((2, IrType::Unknown)),
            // Post-cast dual-cursor bookkeeping (2026-04-22 ability-cooldowns
            // subsystem). Args: `(caster, ability, now)`. Writes BOTH the
            // per-agent global cursor (with `config.combat.global_cooldown_ticks`)
            // and the per-(agent, slot) local cursor (with the ability's
            // own `gate.cooldown_ticks`). Replaces `set_cooldown_next_ready`
            // in the `physics cast` rule; the split-primitive form fixes
            // the shared-cursor bug where all abilities on one agent were
            // gated by a single cursor.
            (NamespaceId::Agents, "record_cast_cooldowns") => Some((3, IrType::Unknown)),
            // Ability registry accessors. `is_known` tells the cast handler
            // whether to bail out silently on an unregistered ability id;
            // `cooldown_ticks` returns the program's `gate.cooldown_ticks`;
            // `effects` yields the program's ordered `EffectOp` list for the
            // dispatch for-loop to iterate.
            (NamespaceId::Abilities, "is_known") => Some((1, IrType::Bool)),
            (NamespaceId::Abilities, "cooldown_ticks") => Some((1, IrType::U32)),
            (NamespaceId::Abilities, "effects") => {
                Some((1, IrType::List(Box::new(IrType::Named("EffectOp".into())))))
            }
            // Mask-gate accessors for the `Cast` DSL mask (task 157).
            // `known(agent, ability)` is the 2-arg mask-side sibling of
            // the physics-side `is_known(ability)` — the emitter lowers
            // it into a registry `get(...).is_some()`, ignoring the
            // agent argument (mask-gate does not yet key on per-agent
            // spellbooks). `cooldown_ready(agent, ability)` folds the
            // "state.tick >= agent_cooldown_next_ready" read into a
            // single boolean the mask predicate can `&&`-chain. The
            // `hostile_only(ability)` / `range(ability)` pair exposes
            // the program's `Gate.hostile_only` / `Area::SingleTarget
            // .range` fields so the target-side filter can stay in the
            // engine's `inferred_cast_target` helper (the mask DSL's
            // `from`-clause only accepts an `AgentId` source).
            (NamespaceId::Abilities, "known") => Some((2, IrType::Bool)),
            (NamespaceId::Abilities, "cooldown_ready") => Some((2, IrType::Bool)),
            // Designer-facing inverted form of `cooldown_ready`. Takes a
            // literal slot index and lets the mask / physics predicate
            // phrase gates as `ability::on_cooldown(s)` (returns `true`
            // when the slot is still on cooldown — the natural "gate
            // blocks" reading). The implicit subject is the rule's
            // `self`; the slot arg coerces to `u8` via argument lowering
            // in the emitter.
            (NamespaceId::Abilities, "on_cooldown") => Some((1, IrType::Bool)),
            (NamespaceId::Abilities, "hostile_only") => Some((1, IrType::Bool)),
            (NamespaceId::Abilities, "range") => Some((1, IrType::F32)),
            // Terrain seam — MVP Task 81 + voxel-engine integration
            // Phase D (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
            // Three methods:
            // - `terrain.line_of_sight(from: vec3, to: vec3) -> bool`
            // - `terrain.height_at(x: f32, y: f32) -> f32`
            // - `terrain.walkable(pos: vec3, mode: u32) -> bool`
            // All three lower to WGSL helpers that read the
            // `voxel_grid` storage binding (Phase C's GPU mirror).
            (NamespaceId::Terrain, "line_of_sight") => Some((2, IrType::Bool)),
            (NamespaceId::Terrain, "height_at") => Some((2, IrType::F32)),
            (NamespaceId::Terrain, "walkable") => Some((2, IrType::Bool)),
            // Voxel-region-indices Phase 4b — `navgrid.walkable(cx,
            // cz) -> bool`. Reads the per-region navgrid built host-
            // side via `engine_voxel::build_navgrid`. Cells outside
            // the navgrid extent return `false` (consistent with the
            // CPU-side `cell_at` returning None / non-walkable).
            (NamespaceId::Navgrid, "walkable") => Some((2, IrType::Bool)),
            // Engagement accessor — wraps `state.agent_engaged_with(id)`,
            // returning `Option<AgentId>` so the mask predicate can
            // compare against `None` (the engagement-lock clause in
            // `mask Cast`). Task 157.
            (NamespaceId::Agents, "engaged_with") => {
                Some((1, IrType::Optional(Box::new(IrType::AgentId))))
            }
            // Engagement accessors used by the `engagement_on_move` /
            // `engagement_on_death` DSL physics rules (task 163).
            //
            // `set_engaged_with(agent, partner)` eagerly writes the SoA
            // `hot_engaged_with` slot to `Some(partner)` so same-tick
            // cascade handlers observe the new partner before the view-
            // fold phase rebuilds `state.views.engaged_with`. Split from
            // `clear_engaged_with(agent)` so the DSL surface doesn't
            // need an `Option` ctor for the two-arg setter (the
            // generated Rust still calls the single bounds-tolerant
            // `state.set_agent_engaged_with` for both).
            //
            // `engaged_with_or` is the unwrap-or-default sibling of
            // `engaged_with` — returns the partner if any, else
            // `default`. Lets the rule body sentinel on the agent
            // itself when no partner is set, avoiding an `if let Some`
            // narrowing inside the physics body (which the GPU-
            // emittable subset doesn't yet support).
            (NamespaceId::Agents, "set_engaged_with") => Some((2, IrType::Unknown)),
            (NamespaceId::Agents, "clear_engaged_with") => Some((1, IrType::Unknown)),
            (NamespaceId::Agents, "engaged_with_or") => Some((2, IrType::AgentId)),
            // Spatial lookup wrapping `SpatialHash::within_radius` with
            // the species hostility predicate. Returns the nearest hostile
            // (argmin on distance; ties broken on raw AgentId) within
            // `radius`, or `None`. The `_or` sibling returns a caller-
            // supplied sentinel when nothing matches so the physics rule
            // can stay in the GPU-emittable subset (no `if let Some`
            // narrowing required). Task 163.
            (NamespaceId::Query, "nearest_hostile_to") => {
                Some((2, IrType::Optional(Box::new(IrType::AgentId))))
            }
            (NamespaceId::Query, "nearest_hostile_to_or") => Some((3, IrType::AgentId)),
            // Same-species spatial scan — sibling of `nearest_hostile_to`.
            // Task 167 — the `fear_spread_on_death` physics rule iterates
            // every alive same-species neighbour within `radius` of a
            // newly-dead agent and emits a `FearSpread` event per kin.
            // Returns a `List<Agent>` (lowered as `Vec<AgentId>`) so the
            // physics body can `for kin in query.nearby_kin(...)`.
            // Bounded by the cell-reach cap in `SpatialHash::within_radius`.
            (NamespaceId::Query, "nearby_kin") => {
                Some((2, IrType::List(Box::new(IrType::AgentId))))
            }
            // Top-K topological neighbour query — sibling of `nearby_kin`.
            // Returns up to `k` same-species neighbours of `center`
            // sorted ascending by distance (ties broken on raw
            // `AgentId`), drawn from candidates inside `max_radius`.
            // Signature:
            //   `nearest_k(center: AgentId, k: u32 literal, max_radius: f32)
            //    -> List<AgentId>`
            // The `k` argument MUST be a non-negative integer literal so
            // the GPU emitter (planned, see PP Stage 3 / task #62) can
            // bake the heap size into a compile-time `array<u32, K>`.
            // CPU lowering: `crate::spatial::nearest_k(state, ...)`.
            // Used for topological neighbour patterns (Ballerini-style
            // flocking, K-closest threat assessment, etc.).
            (NamespaceId::Query, "nearest_k") => {
                Some((3, IrType::List(Box::new(IrType::AgentId))))
            }
            // -------------------------------------------------------------
            // Roadmap §1 — Memberships. Predicates on `cold_memberships`.
            // All return bool. The `kind` arg of `is_group_member` would
            // ideally type to a `GroupKind` enum, but that enum doesn't
            // exist in the IR yet — fall back to `Unknown` and let
            // whoever implements Subsystem §1 pick the concrete ID type.
            // See `docs/superpowers/roadmap.md:180-182`.
            // -------------------------------------------------------------
            // `is_group_member(agent, kind)` — `kind` is a `GroupKind`
            // discriminator (Family/Religion/Faction/...); TODO: resolve
            // to `IrType::Named("GroupKind")` once the kind enum lands.
            (NamespaceId::Membership, "is_group_member") => Some((2, IrType::Bool)),
            // `is_group_leader(agent)` — true iff agent holds any
            // leader role across its memberships.
            (NamespaceId::Membership, "is_group_leader") => Some((1, IrType::Bool)),
            // `can_join_group(agent, group)` — evaluates
            // `group.eligibility_predicate` against `agent`.
            (NamespaceId::Membership, "can_join_group") => Some((2, IrType::Bool)),
            // `is_outcast(agent, group)` — state.md:69 "outcasts cannot
            // vote"; semantically `standing_q8 < OUTCAST_THRESHOLD`.
            (NamespaceId::Membership, "is_outcast") => Some((2, IrType::Bool)),
            // -------------------------------------------------------------
            // Roadmap §3 — Relationships. Predicates on `cold_relationships`.
            // All return bool. Per the roadmap these replace Combat
            // Foundation's stub `is_hostile_to` once the relationship
            // runtime lands — the grammar stub keeps the two surface
            // forms coexisting until the cutover.
            // See `docs/superpowers/roadmap.md:306-309`.
            // -------------------------------------------------------------
            // `is_hostile(a, b)` — valence_q8 < HOSTILE_THRESHOLD.
            (NamespaceId::Relationship, "is_hostile") => Some((2, IrType::Bool)),
            // `is_friendly(a, b)` — valence_q8 > FRIENDLY_THRESHOLD.
            (NamespaceId::Relationship, "is_friendly") => Some((2, IrType::Bool)),
            // `knows_well(a, b)` — familiarity > 0.5 (roadmap.md:309).
            (NamespaceId::Relationship, "knows_well") => Some((2, IrType::Bool)),
            // -------------------------------------------------------------
            // Roadmap §6 — Theory-of-mind. Predicates on the 32-bit
            // `Relationship.believed_knowledge` domain bitset. All return
            // bool. The `domain` / `fact` args would ideally type to
            // `DomainId` / `FactId` enums, but those don't exist in the
            // IR yet — fall back to `Unknown`. TODO: once Subsystem §6
            // introduces the DomainId / FactId enums, tighten the
            // argument types by name (similar to how `EventKindId` is
            // referenced via `IrType::Named`).
            // See `docs/superpowers/roadmap.md:471-475`.
            // -------------------------------------------------------------
            // `believes_knows(observer, subject, domain)` — primary
            // bit-read against `Relationship{self→subject}.believed_knowledge`.
            (NamespaceId::TheoryOfMind, "believes_knows") => Some((3, IrType::Bool)),
            // `can_deceive(observer, subject, fact)` — sugar for
            // `!believes_knows(subject, fact)` (roadmap.md:473).
            (NamespaceId::TheoryOfMind, "can_deceive") => Some((3, IrType::Bool)),
            // `is_surprised_by(observer, subject, domain)` — fires when
            // an observed action contradicts the domain bit.
            (NamespaceId::TheoryOfMind, "is_surprised_by") => Some((3, IrType::Bool)),
            // -------------------------------------------------------------
            // Roadmap §7 — Groups. Predicates on the `AggregatePool<Group>`.
            // All return bool. The `cost` arg of `can_afford_from_treasury`
            // is scalar gold (state.md:1134 treasury is an `i64`); argument
            // type enforcement lands when Subsystem §7 wires the real
            // method_sig (1a is arity-only).
            // See `docs/superpowers/roadmap.md:545-546`.
            // -------------------------------------------------------------
            // `exists(id)` — is this GroupId slot populated?
            (NamespaceId::Group, "exists") => Some((1, IrType::Bool)),
            // `is_active(id)` — populated AND `dissolved_tick.is_none()`
            // (state.md:1107).
            (NamespaceId::Group, "is_active") => Some((1, IrType::Bool)),
            // `has_leader(id)` — `leader_id.is_some()` (state.md:1116).
            (NamespaceId::Group, "has_leader") => Some((1, IrType::Bool)),
            // `can_afford_from_treasury(g, cost)` — `treasury >= cost`.
            (NamespaceId::Group, "can_afford_from_treasury") => Some((2, IrType::Bool)),
            // -------------------------------------------------------------
            // Roadmap §12 — Quests. Predicates on the `AggregatePool<Quest>`.
            // All return bool. `is_target(entity, q)` takes an `AnyId` —
            // the entity can be an `AgentId` (Hunt kill-target) or a
            // settlement / location (Escort / Deliver). 1a falls back to
            // `IrType::Unknown` for the entity arg; TODO: tighten once
            // Subsystem §12 fixes the entity-kind taxonomy.
            // See `docs/superpowers/roadmap.md:843-845`.
            // -------------------------------------------------------------
            // `can_accept(agent, q)` — checks party capacity + eligibility.
            (NamespaceId::Quest, "can_accept") => Some((2, IrType::Bool)),
            // `is_target(entity, q)` — entity is AnyId per above.
            (NamespaceId::Quest, "is_target") => Some((2, IrType::Bool)),
            // `party_near_destination(party, q)` — spatial gate on the
            // party's centroid vs `quest.destination`.
            (NamespaceId::Quest, "party_near_destination") => Some((2, IrType::Bool)),
            // -------------------------------------------------------------
            // Plan G G3f (2026-05-09) — Threats scoring primitives.
            // Each method dispatches to a `Builtin::Threats*` variant
            // in `resolve_call`; the entries here document the typed
            // signature for 1b consumers (and future tooling that
            // wants to advertise the `threats.*` surface). Arity 1 in
            // every case — single agent / position arg.
            // See `docs/plans/g3_threats_view_design.md`.
            // -------------------------------------------------------------
            (NamespaceId::Threats, "in_zone") => Some((1, IrType::Bool)),
            (NamespaceId::Threats, "intensity_at") => Some((1, IrType::F32)),
            (NamespaceId::Threats, "nearest") => Some((1, IrType::AgentId)),
            (NamespaceId::Threats, "dir_away_from_nearest") => Some((1, IrType::Vec3)),
            // -------------------------------------------------------------
            // Events namespace — sim-wide event-trace accessors used by
            // metrics / invariants / probes. Returned types are
            // informational at 1a (the metric/invariant/probe shape
            // classifiers don't enforce the inner element type today —
            // they pattern-match on the surrounding `count(e in
            // events.<method>(...) where ...)` shape and emit a
            // per-name SKIP setter the runtime fills in). The arities
            // are checked by 1b's NamespaceCall arity validator.
            //
            // `events.at_tick(tick: u32) -> [Event]` — events recorded
            // at a specific past tick.
            (NamespaceId::Events, "at_tick") => {
                Some((1, IrType::List(Box::new(IrType::Unknown))))
            }
            // `events.range(from: u32, to: u32) -> [Event]` — half-open
            // range query over the trace history.
            (NamespaceId::Events, "range") => {
                Some((2, IrType::List(Box::new(IrType::Unknown))))
            }
            // `events.kind_count(kind: EventKindId) -> u32` — count of
            // events of a given kind in the current tick. Provided so
            // metric authors don't have to spell out the full
            // `count(e in events.this_tick where e.kind == X)` fold
            // for the common case.
            (NamespaceId::Events, "kind_count") => Some((1, IrType::U32)),
            _ => None,
        }
    }
}

/// Plan G G3f — map a `threats.<method>` name to the corresponding
/// `Builtin::Threats*` variant. Returns `None` for unknown methods so
/// `resolve_call` can fall through to the generic `NamespaceCall`
/// route (which 1b surfaces as an unknown-method diagnostic).
fn threats_method_builtin(method: &str) -> Option<Builtin> {
    match method {
        "in_zone" => Some(Builtin::ThreatsInZone),
        "intensity_at" => Some(Builtin::ThreatsIntensityAt),
        "nearest" => Some(Builtin::ThreatsNearest),
        "dir_away_from_nearest" => Some(Builtin::ThreatsDirAwayFromNearest),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Symbol table
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SymbolTable {
    pub events: HashMap<String, EventRef>,
    /// `event_tag` declarations keyed by their *lowercased* name (matches the
    /// `@tag_name` annotation form). Value carries the IR ref and the tag's
    /// PascalCase source name.
    pub event_tags: HashMap<String, (EventTagRef, String)>,
    /// User-declared `enum` types keyed by PascalCase name. Value carries the
    /// IR ref plus the full variant list for lookup during expression resolution.
    pub enums: HashMap<String, (EnumRef, Vec<String>)>,
    /// Reverse index: variant name → owning enum name. Populated only for
    /// variants whose enum owns the variant exclusively (same variant in two
    /// enums stays ambiguous and resolves by left context).
    pub enum_variant_owner: HashMap<String, String>,
    pub entities: HashMap<String, EntityRef>,
    pub physics: HashMap<String, PhysicsRef>,
    pub masks: HashMap<String, MaskRef>,
    pub scoring: HashMap<String, ScoringRef>,
    pub views: HashMap<String, ViewRef>,
    pub verbs: HashMap<String, VerbRef>,
    /// `spatial_query <name>(...)` declarations registered in pass 1.
    /// `from spatial.<name>(...)` references resolve against this map
    /// (Phase 7 Task 4). Pre-populated so pass 2 can route the
    /// dotted/`::`-flat call shapes through `NamespaceCall { ns:
    /// Spatial, method: <name>, … }` and surface unknown names as
    /// `ResolveError::UnknownSpatialQuery`. Name → index into
    /// `Compilation::spatial_queries`, matching the `views` /
    /// `verbs` convention.
    pub spatial_queries: HashMap<String, SpatialQueryRef>,
    pub invariants: HashMap<String, InvariantRef>,
    pub probes: HashMap<String, ProbeRef>,
    pub metrics: HashMap<String, MetricRef>,
    /// `config` block name → `(ConfigRef, field-name → field-type)`. Populated
    /// in pass 1 so pass-2 body lowering can resolve `config.<block>.<field>`
    /// into a typed `NamespaceField { ns: Config, field: "<block>.<field>" }`.
    pub configs: HashMap<String, (ConfigRef, HashMap<String, IrType>)>,
    /// Static lookup tables (`table <name>: <ty>[N] = […]`) declared
    /// at top level. Name → typed `TableId` handle into
    /// `Compilation::tables`. Looked up by namespace resolution
    /// when the surface text reads `tables.<name>(<idx>)`.
    pub tables: HashMap<String, crate::ir::TableId>,
    /// Voxel-region kinds — per spec
    /// `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
    /// §6.1.2. Name (PascalCase, e.g. `Settlement`) → typed
    /// [`crate::ir::VoxelRegionKindId`] into
    /// [`crate::ir::Compilation::region_kinds`]. The `region_kind`
    /// + `region_indices` decl pair lives in one IR slot, keyed by
    /// name.
    pub region_kinds: HashMap<String, crate::ir::VoxelRegionKindId>,
    /// Region-attached indices — per spec §7.2. Name (PascalCase,
    /// e.g. `Navgrid`) → typed [`crate::ir::IndexId`] into
    /// [`crate::ir::Compilation::indices`]. Looked up by the
    /// `region_indices Settlement { Navgrid }` resolver to bind
    /// each index-kind reference to its declared `index navgrid(…)`.
    pub indices: HashMap<String, crate::ir::IndexId>,
    pub builtins: HashMap<String, Builtin>,
    pub stdlib_types: HashMap<String, IrType>,
    /// Sim-wide accessor namespaces: `world`, `cascade`, `event`, `mask`,
    /// `action`, `rng`, `query`, `voxel`, plus the legacy collection
    /// accessors (`agents`, `items`, `groups`, `quests`, `auctions`,
    /// `tick`). Each maps to a typed `NamespaceId` the IR uses; per-field
    /// and per-method schemas are declared in `stdlib::field_type` /
    /// `stdlib::method_sig`.
    pub stdlib_namespaces: HashMap<String, NamespaceId>,
    // Span of first declaration — for duplicate-decl diagnostics.
    pub first_span: HashMap<(&'static str, String), Span>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut s = Self::default();
        stdlib::seed(&mut s);
        s
    }

    fn record_first(&mut self, kind: &'static str, name: &str, span: Span) {
        self.first_span.insert((kind, name.to_string()), span);
    }

    fn first_of(&self, kind: &'static str, name: &str) -> Option<Span> {
        self.first_span.get(&(kind, name.to_string())).copied()
    }
}

// ---------------------------------------------------------------------------
// Local scope (stacked)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LocalBinding {
    name: String,
    local: LocalRef,
    #[allow(dead_code)]
    ty: IrType,
}

#[derive(Debug, Default)]
struct LocalScope {
    stack: Vec<Vec<LocalBinding>>,
    next_id: u16,
    // Tracks whether `self` has been bound in the current decl.
    self_bound: bool,
}

impl LocalScope {
    fn new() -> Self {
        LocalScope { stack: vec![Vec::new()], next_id: 0, self_bound: false }
    }

    fn push(&mut self) {
        self.stack.push(Vec::new());
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    fn fresh(&mut self) -> LocalRef {
        let r = LocalRef(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        r
    }

    fn bind(&mut self, name: &str, ty: IrType) -> LocalRef {
        let local = self.fresh();
        if name == "self" {
            self.self_bound = true;
        }
        self.stack
            .last_mut()
            .unwrap()
            .push(LocalBinding { name: name.to_string(), local, ty });
        local
    }

    fn lookup(&self, name: &str) -> Option<&LocalBinding> {
        for frame in self.stack.iter().rev() {
            for b in frame.iter().rev() {
                if b.name == name {
                    return Some(b);
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn resolve(program: Program) -> Result<Compilation, ResolveError> {
    let mut symbols = SymbolTable::new();
    let mut comp = Compilation::default();

    // Pass 1: collect top-level names + reserve IR slots (still empty).
    collect(&program, &mut symbols, &mut comp)?;

    // Pass 2: resolve bodies into the reserved slots.
    resolve_bodies(&program, &symbols, &mut comp)?;

    // Pass 3: cross-rule validation that needs the whole `Compilation`
    // in hand. Physics bodies must stay SPIR-V-emittable (compiler/spec.md
    // §1.2); the validator checks cross-rule recursion + per-handler
    // GPU-emittability.
    validate_physics_bodies(&comp)?;

    Ok(comp)
}

// ---------------------------------------------------------------------------
// Pass 1: collect
// ---------------------------------------------------------------------------

fn collect(
    program: &Program,
    symbols: &mut SymbolTable,
    comp: &mut Compilation,
) -> Result<(), ResolveError> {
    // We pre-allocate empty IR shells with the right names/spans so indices
    // are stable. Pass 2 will overwrite the bodies.
    // Pass 1a: collect event_tags + enums first so that event decls can
    // resolve their tag annotations in-place during pass 1b.
    for decl in &program.decls {
        match decl {
            Decl::EventTag(d) => {
                let key = lowercase_tag_name(&d.name);
                check_dup(symbols, "event_tag", &d.name, d.span, |s| {
                    s.event_tags.contains_key(&key)
                })?;
                let idx = push_idx(comp.event_tags.len(), "event_tag")?;
                symbols.event_tags.insert(key, (EventTagRef(idx), d.name.clone()));
                symbols.record_first("event_tag", &d.name, d.span);
                let fields = d
                    .fields
                    .iter()
                    .map(|f| EventField {
                        name: f.name.clone(),
                        ty: resolve_type(&f.ty, symbols),
                        span: f.span,
                    })
                    .collect();
                comp.event_tags.push(EventTagIR {
                    name: d.name.clone(),
                    fields,
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Enum(d) => {
                check_dup(symbols, "enum", &d.name, d.span, |s| s.enums.contains_key(&d.name))?;
                let idx = push_idx(comp.enums.len(), "enum")?;
                let variants: Vec<String> =
                    d.variants.iter().map(|v| v.name.clone()).collect();
                for v in &variants {
                    symbols
                        .enum_variant_owner
                        .entry(v.clone())
                        .or_insert_with(|| d.name.clone());
                }
                symbols.enums.insert(d.name.clone(), (EnumRef(idx), variants.clone()));
                symbols.record_first("enum", &d.name, d.span);
                comp.enums.push(EnumIR {
                    name: d.name.clone(),
                    variants,
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            _ => {}
        }
    }

    for decl in &program.decls {
        match decl {
            Decl::Event(d) => {
                check_dup(symbols, "event", &d.name, d.span, |s| s.events.contains_key(&d.name))?;
                let idx = push_idx(comp.events.len(), "event")?;
                symbols.events.insert(d.name.clone(), EventRef(idx));
                symbols.record_first("event", &d.name, d.span);
                // Partition annotations: `@tag_name` annotations whose name
                // matches a declared event_tag become tag refs. Non-tag
                // annotations (replayable, non_replayable, high_volume, ...)
                // stay on the event.
                let mut tag_refs: Vec<EventTagRef> = Vec::new();
                let mut non_tag_anns: Vec<ast::Annotation> =
                    Vec::with_capacity(d.annotations.len());
                for ann in &d.annotations {
                    if ann.args.is_empty()
                        && symbols.event_tags.contains_key(&ann.name)
                    {
                        let (tref, _) = symbols.event_tags[&ann.name];
                        tag_refs.push(tref);
                    } else {
                        non_tag_anns.push(ann.clone());
                    }
                }
                let engine_kind_id = crate::engine_events::engine_event_kind_id_for_name(&d.name);
                comp.events.push(EventIR {
                    name: d.name.clone(),
                    fields: Vec::new(),
                    tags: tag_refs,
                    annotations: non_tag_anns,
                    span: d.span,
                    engine_kind_id,
                });
            }
            Decl::EventTag(_) | Decl::Enum(_) => {
                // Already collected in the pre-pass above.
            }
            Decl::Entity(d) => {
                check_dup(symbols, "entity", &d.name, d.span, |s| s.entities.contains_key(&d.name))?;
                let idx = push_idx(comp.entities.len(), "entity")?;
                symbols.entities.insert(d.name.clone(), EntityRef(idx));
                symbols.record_first("entity", &d.name, d.span);
                comp.entities.push(EntityIR {
                    name: d.name.clone(),
                    root: d.root,
                    fields: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Physics(d) => {
                check_dup(symbols, "physics", &d.name, d.span, |s| s.physics.contains_key(&d.name))?;
                let idx = push_idx(comp.physics.len(), "physics")?;
                symbols.physics.insert(d.name.clone(), PhysicsRef(idx));
                symbols.record_first("physics", &d.name, d.span);
                comp.physics.push(PhysicsIR {
                    name: d.name.clone(),
                    handlers: Vec::new(),
                    annotations: d.annotations.clone(),
                    cpu_only: d.cpu_only,
                    span: d.span,
                });
            }
            Decl::Mask(d) => {
                let key = d.head.name.clone();
                check_dup(symbols, "mask", &key, d.span, |s| s.masks.contains_key(&key))?;
                let idx = push_idx(comp.masks.len(), "mask")?;
                symbols.masks.insert(key.clone(), MaskRef(idx));
                symbols.record_first("mask", &key, d.span);
                comp.masks.push(MaskIR {
                    head: IrActionHead {
                        name: d.head.name.clone(),
                        shape: IrActionHeadShape::None,
                        span: d.head.span,
                    },
                    candidate_source: None,
                    predicate: IrExprNode { kind: IrExpr::LitBool(true), span: d.span },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Scoring(d) => {
                // Scoring blocks are unnamed; use synthetic name keyed by
                // index + span. Duplicates are tolerated (multiple blocks are
                // allowed per spec).
                let synthetic = format!("__scoring_{}", comp.scoring.len());
                let idx = push_idx(comp.scoring.len(), "scoring")?;
                symbols.scoring.insert(synthetic, ScoringRef(idx));
                comp.scoring.push(ScoringIR {
                    entries: Vec::new(),
                    per_ability_rows: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::View(d) => {
                check_dup(symbols, "view", &d.name, d.span, |s| s.views.contains_key(&d.name))?;
                let idx = push_idx(comp.views.len(), "view")?;
                symbols.views.insert(d.name.clone(), ViewRef(idx));
                symbols.record_first("view", &d.name, d.span);
                let storage_packing = lower_storage_annotation(&d.annotations)?;
                comp.views.push(ViewIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    return_ty: IrType::Unknown,
                    body: ViewBodyIR::Expr(IrExprNode { kind: IrExpr::LitBool(true), span: d.span }),
                    annotations: d.annotations.clone(),
                    kind: ViewKind::Lazy,
                    decay: None,
                    belief_gated: d.annotations.iter().any(|a| a.name == "belief_gated"),
                    storage_packing,
                    social_merges: Vec::new(),
                    span: d.span,
                });
            }
            Decl::Verb(d) => {
                check_dup(symbols, "verb", &d.name, d.span, |s| s.verbs.contains_key(&d.name))?;
                let idx = push_idx(comp.verbs.len(), "verb")?;
                symbols.verbs.insert(d.name.clone(), VerbRef(idx));
                symbols.record_first("verb", &d.name, d.span);
                comp.verbs.push(VerbIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    action: VerbActionIR {
                        name: d.action.name.clone(),
                        args: Vec::new(),
                        span: d.action.span,
                    },
                    when: None,
                    body: Vec::new(),
                    scoring: None,
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Invariant(d) => {
                check_dup(symbols, "invariant", &d.name, d.span, |s| {
                    s.invariants.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.invariants.len(), "invariant")?;
                symbols.invariants.insert(d.name.clone(), InvariantRef(idx));
                symbols.record_first("invariant", &d.name, d.span);
                comp.invariants.push(InvariantIR {
                    name: d.name.clone(),
                    scope: Vec::new(),
                    mode: d.mode,
                    predicate: IrExprNode { kind: IrExpr::LitBool(true), span: d.span },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Probe(d) => {
                check_dup(symbols, "probe", &d.name, d.span, |s| s.probes.contains_key(&d.name))?;
                let idx = push_idx(comp.probes.len(), "probe")?;
                symbols.probes.insert(d.name.clone(), ProbeRef(idx));
                symbols.record_first("probe", &d.name, d.span);
                comp.probes.push(ProbeIR {
                    name: d.name.clone(),
                    scenario: d.scenario.clone(),
                    seed: d.seed,
                    seeds: d.seeds.clone(),
                    ticks: d.ticks,
                    tolerance: d.tolerance,
                    asserts: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Metric(block) => {
                for m in &block.metrics {
                    check_dup(symbols, "metric", &m.name, m.span, |s| {
                        s.metrics.contains_key(&m.name)
                    })?;
                    let idx = push_idx(comp.metrics.len(), "metric")?;
                    symbols.metrics.insert(m.name.clone(), MetricRef(idx));
                    symbols.record_first("metric", &m.name, m.span);
                    comp.metrics.push(MetricIR {
                        name: m.name.clone(),
                        value: IrExprNode { kind: IrExpr::LitBool(true), span: m.span },
                        window: m.window,
                        emit_every: m.emit_every,
                        conditioned_on: None,
                        alert_when: None,
                        annotations: block.annotations.clone(),
                        span: m.span,
                    });
                }
            }
            Decl::Config(d) => {
                check_dup(symbols, "config", &d.name, d.span, |s| {
                    s.configs.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.configs.len(), "config")?;
                let mut field_types: HashMap<String, IrType> = HashMap::new();
                let mut fields_ir: Vec<ConfigFieldIR> = Vec::with_capacity(d.fields.len());
                for f in &d.fields {
                    let ty = resolve_type(&f.ty, symbols);
                    if field_types.contains_key(&f.name) {
                        return Err(ResolveError::DuplicateDecl {
                            kind: "config_field",
                            name: format!("{}.{}", d.name, f.name),
                            first: d.span,
                            second: f.span,
                        });
                    }
                    field_types.insert(f.name.clone(), ty.clone());
                    fields_ir.push(ConfigFieldIR {
                        name: f.name.clone(),
                        ty,
                        default: f.default.clone(),
                        runtime: f.runtime,
                        span: f.span,
                    });
                }
                symbols.configs.insert(d.name.clone(), (ConfigRef(idx), field_types));
                symbols.record_first("config", &d.name, d.span);
                comp.configs.push(ConfigIR {
                    name: d.name.clone(),
                    fields: fields_ir,
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Query(d) => {
                // The modern `query <name>(...) -> ... sort_by ... limit
                // ... { <body> }` surface registers as a SpatialQueryIR
                // alongside the legacy `spatial_query <name>(...) =
                // <expr>` form. Pass-1 reserves the symbol slot + carries
                // the annotations (so `@top_k(K)` and `@spatial(...)` are
                // visible to consumers). The richer fields (sort_by,
                // limit, return_ty, body filter) lower in pass-2 once a
                // physics rule consumes the query — until then the
                // placeholder filter `LitBool(true)` mirrors the
                // SpatialQueryIR pass-1 convention. Stage 3 lock-in:
                // declared queries appear in `comp.spatial_queries` even
                // before any consumer wires them.
                check_dup(symbols, "spatial_query", &d.name, d.span, |s| {
                    s.spatial_queries.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.spatial_queries.len(), "spatial_query")?;
                symbols
                    .spatial_queries
                    .insert(d.name.clone(), SpatialQueryRef(idx));
                symbols.record_first("spatial_query", &d.name, d.span);
                comp.spatial_queries.push(SpatialQueryIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    filter: IrExprNode {
                        kind: IrExpr::LitBool(true),
                        span: d.span,
                    },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::SpatialQuery(d) => {
                // Phase 7 Task 4. Pass-1 reserves the slot + records
                // the name; pass-2 fills the resolved filter expr.
                check_dup(symbols, "spatial_query", &d.name, d.span, |s| {
                    s.spatial_queries.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.spatial_queries.len(), "spatial_query")?;
                symbols
                    .spatial_queries
                    .insert(d.name.clone(), SpatialQueryRef(idx));
                symbols.record_first("spatial_query", &d.name, d.span);
                comp.spatial_queries.push(SpatialQueryIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    filter: IrExprNode {
                        kind: IrExpr::LitBool(true),
                        span: d.span,
                    },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Init(_) => {
                // Plan E-A6 — `init` blocks are consumed directly by
                // `dsl_compiler::build_helper` from the parsed Program.
                // No Compilation IR slot today; resolver passes through.
            }
            Decl::Debug(_) => {
                // `debug { depth: ..., wgsl_*: ... }` blocks are consumed
                // directly by `dsl_compiler::build_helper` from the
                // parsed Program (mirrors `init`). No Compilation IR
                // slot — the values feed straight into LowerOpts.
            }
            Decl::AgentField(_) => {
                // Gap plague_city#P-A — `field <name>: <type>` decls
                // are extracted by `dsl_compiler::build_helper` /
                // `dsl_compiler::custom_agent_fields::populate` from
                // the parsed Program BEFORE any lowering pass touches
                // `self.<name>` or `agents.set_<name>(...)`. Mirrors
                // the `init` + `debug` pass-through precedent: no
                // Compilation IR slot here — the registry lives in
                // process-static memory keyed by the leaked field
                // name, and reset to "no custom fields" on each new
                // compile via `clear_for_compile` at the build helper
                // entry point.
            }
            Decl::Table(d) => {
                check_dup(symbols, "table", &d.name, d.span, |s| {
                    s.tables.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.tables.len(), "table")?;
                symbols
                    .tables
                    .insert(d.name.clone(), crate::ir::TableId(idx as u32));
                symbols.record_first("table", &d.name, d.span);
                // Reserve the IR slot with placeholder values; pass-2
                // populates `element_ty` + bounds-checked `values`.
                comp.tables.push(crate::ir::TableIR {
                    name: d.name.clone(),
                    element_ty: IrType::U32,
                    length: d.length,
                    values: Vec::new(),
                    span: d.span,
                });
            }
            Decl::RegionKind(d) => {
                // Reserve the IR slot keyed by the kind name. The
                // sibling `Decl::RegionIndices` decl is matched in
                // pass-2 and merges its `index_kinds` into the same
                // slot.
                check_dup(symbols, "region_kind", &d.name, d.span, |s| {
                    s.region_kinds.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.region_kinds.len(), "region_kind")?;
                symbols.region_kinds.insert(
                    d.name.clone(),
                    crate::ir::VoxelRegionKindId(idx as u32),
                );
                symbols.record_first("region_kind", &d.name, d.span);
                comp.region_kinds.push(crate::ir::RegionKindIR {
                    name: d.name.clone(),
                    max_active: d.max_active,
                    index_kind_names: Vec::new(),
                    span: d.span,
                });
            }
            Decl::RegionIndices(_) => {
                // Handled in pass-2 — needs the `region_kind` IR slot
                // to exist first. Pass-1 leaves it for pass-2 to merge.
            }
            Decl::Index(d) => {
                check_dup(symbols, "index", &d.name, d.span, |s| {
                    s.indices.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.indices.len(), "index")?;
                symbols
                    .indices
                    .insert(d.name.clone(), crate::ir::IndexId(idx as u32));
                symbols.record_first("index", &d.name, d.span);
                comp.indices.push(crate::ir::IndexIR {
                    name: d.name.clone(),
                    region_param_name: d.region_param_name.clone(),
                    output_type_name: d.output_type_name.clone(),
                    storage: d.storage.clone(),
                    cost_class: d.cost_class,
                    rebuild_on: d.rebuild_on.clone(),
                    build_body: d.build_body.clone(),
                    build_body_ast: d.build_body_ast.clone(),
                    span: d.span,
                });
            }
            Decl::PhysicsApply(_) => {
                // apply-form: ignored in pass 1 — handled by param_rules lowering (T5+).
            }
            Decl::Belief(d) => {
                // Plan I — beliefs share the ViewIR slot table with
                // views; lookups by name flow through the same
                // `symbols.views` map so `belief.<name>(…)` and
                // `<name>(…)` reads in expressions resolve uniformly.
                // Storage hint is filled in by the lowering pass
                // (`crates/dsl_compiler/src/cg/lower/view.rs`) from
                // the signature shape, not from an annotation — so
                // the slot starts with `ViewKind::Belief` and an empty
                // social_merges list (populated in Pass-2 below).
                check_dup(symbols, "view", &d.name, d.span, |s| s.views.contains_key(&d.name))?;
                let idx = push_idx(comp.views.len(), "view")?;
                symbols.views.insert(d.name.clone(), ViewRef(idx));
                symbols.record_first("view", &d.name, d.span);
                comp.views.push(ViewIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    return_ty: IrType::Unknown,
                    body: ViewBodyIR::Expr(IrExprNode {
                        kind: IrExpr::LitBool(true),
                        span: d.span,
                    }),
                    annotations: d.annotations.clone(),
                    kind: ViewKind::Belief,
                    decay: None,
                    belief_gated: d.annotations.iter().any(|a| a.name == "belief_gated"),
                    storage_packing: Packing::None,
                    social_merges: Vec::new(),
                    span: d.span,
                });
            }
        }
    }
    Ok(())
}

fn push_idx(len: usize, kind: &'static str) -> Result<u16, ResolveError> {
    u16::try_from(len).map_err(|_| ResolveError::TooManyDecls { kind })
}

/// Known engine helpers callable from an `index` build body.
/// Closed set per spec §7.2 worked examples. Adding a new helper
/// is a 2-line edit here + the engine-side implementation in
/// Phase 4. Per spec design: the engine intrinsics are a fixed
/// catalog rather than a plug-in surface, so the validation point
/// can stay here (vs needing a per-runtime registry hook).
const KNOWN_INDEX_ENGINE_HELPERS: &[&str] = &[
    // §7.2 navgrid example
    "column_reduce_xz",
    "per_cell_classify",
    "connect_neighbors",
    // §7.2 vismap example
    "raycast_batch",
    "subdivide_view_cells",
    "pairs_within_radius",
    "sample_rays_between",
];

/// Known top-level constants exposed to index build bodies.
/// Spec §7.2 references `AGENT_STEP_HEIGHT`, `MAX_VIS_RANGE`,
/// `VIEW_CELL_SIZE`, `RAYS_PER_PAIR` — host-side constants the
/// engine surfaces into the build environment.
const KNOWN_INDEX_BUILD_CONSTS: &[&str] = &[
    "AGENT_STEP_HEIGHT",
    "MAX_VIS_RANGE",
    "VIEW_CELL_SIZE",
    "RAYS_PER_PAIR",
];

fn validate_index_build_body(d: &ast::IndexDecl) -> Result<(), ResolveError> {
    use ast::IndexBuildStmt;

    // Build the local-binding scope incrementally as we walk the
    // stmts. The region param is always in scope; lets add their
    // names. Top-level constants are a flat set checked separately.
    let mut locals: Vec<String> = vec![d.region_param_name.clone()];
    let mut saw_return = false;

    for stmt in &d.build_body_ast.stmts {
        if saw_return {
            return Err(ResolveError::InvalidViewKind {
                view_name: d.name.clone(),
                detail: format!(
                    "index `{}` build body has stmts after the return expression",
                    d.name
                ),
                span: d.span,
            });
        }
        match stmt {
            IndexBuildStmt::Let { name, value, span } => {
                validate_index_build_expr(d, value, &locals, *span)?;
                locals.push(name.clone());
            }
            IndexBuildStmt::Return { value, span } => {
                validate_index_build_expr(d, value, &locals, *span)?;
                saw_return = true;
            }
        }
    }

    // Empty body is allowed (Phase 2a fixtures use it as a stub);
    // Phase 4 will require a non-empty body when wiring the build
    // kernel. Track here for future tightening.
    Ok(())
}

fn validate_index_build_expr(
    d: &ast::IndexDecl,
    expr: &ast::IndexBuildExpr,
    locals: &[String],
    parent_span: Span,
) -> Result<(), ResolveError> {
    use ast::IndexBuildExpr;
    match expr {
        IndexBuildExpr::EngineCall { name, args, span } => {
            if !KNOWN_INDEX_ENGINE_HELPERS.contains(&name.as_str()) {
                return Err(ResolveError::InvalidViewKind {
                    view_name: d.name.clone(),
                    detail: format!(
                        "index `{}` build body calls unknown engine helper `engine::{}` — \
                         known helpers: {}",
                        d.name,
                        name,
                        KNOWN_INDEX_ENGINE_HELPERS.join(", ")
                    ),
                    span: *span,
                });
            }
            for arg in args {
                validate_index_build_expr(d, arg, locals, *span)?;
            }
            Ok(())
        }
        IndexBuildExpr::Var { name, span } => {
            if locals.contains(name) || KNOWN_INDEX_BUILD_CONSTS.contains(&name.as_str()) {
                Ok(())
            } else {
                Err(ResolveError::InvalidViewKind {
                    view_name: d.name.clone(),
                    detail: format!(
                        "index `{}` build body references unknown identifier `{}` — \
                         expected a let-bound local, the region param `{}`, or one of: {}",
                        d.name,
                        name,
                        d.region_param_name,
                        KNOWN_INDEX_BUILD_CONSTS.join(", ")
                    ),
                    span: *span,
                })
            }
        }
        IndexBuildExpr::Member { base, field, span } => {
            // Member access is restricted to `region.<field>` form.
            // Field is opaque — Phase 3 (region runtime) will
            // catalog the valid VoxelRegion fields.
            if base != &d.region_param_name {
                return Err(ResolveError::InvalidViewKind {
                    view_name: d.name.clone(),
                    detail: format!(
                        "index `{}` build body has member access on `{}` — only `{}.<field>` is supported",
                        d.name, base, d.region_param_name
                    ),
                    span: *span,
                });
            }
            let _ = field;
            Ok(())
        }
        IndexBuildExpr::Int { .. } => {
            // Plain integer literals are always fine.
            let _ = parent_span;
            Ok(())
        }
    }
}

fn check_dup(
    symbols: &SymbolTable,
    kind: &'static str,
    name: &str,
    second: Span,
    contains: impl FnOnce(&SymbolTable) -> bool,
) -> Result<(), ResolveError> {
    if contains(symbols) {
        let first = symbols.first_of(kind, name).unwrap_or(Span::dummy());
        return Err(ResolveError::DuplicateDecl {
            kind,
            name: name.to_string(),
            first,
            second,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pass 2: resolve bodies
// ---------------------------------------------------------------------------

fn resolve_bodies(
    program: &Program,
    symbols: &SymbolTable,
    comp: &mut Compilation,
) -> Result<(), ResolveError> {
    let mut event_idx = 0;
    let mut entity_idx = 0;
    let mut physics_idx = 0;
    let mut mask_idx = 0;
    let mut scoring_idx = 0;
    let mut view_idx = 0;
    let mut verb_idx = 0;
    let mut invariant_idx = 0;
    let mut probe_idx = 0;
    let mut metric_start_idx = 0usize;
    let mut spatial_query_idx = 0;

    for decl in &program.decls {
        match decl {
            Decl::Event(d) => {
                let fields: Vec<EventField> = d
                    .fields
                    .iter()
                    .map(|f| EventField {
                        name: f.name.clone(),
                        ty: resolve_type(&f.ty, symbols),
                        span: f.span,
                    })
                    .collect();
                // Validate the tag contract: for each tag this event claims,
                // every required tag field must appear on the event with a
                // matching type.
                let tag_refs = comp.events[event_idx].tags.clone();
                for tref in &tag_refs {
                    let tag_ir = &comp.event_tags[tref.0 as usize];
                    let mut details: Vec<String> = Vec::new();
                    for tf in &tag_ir.fields {
                        match fields.iter().find(|f| f.name == tf.name) {
                            None => details.push(format!("missing field `{}`", tf.name)),
                            Some(ef) if ef.ty != tf.ty => details.push(format!(
                                "field `{}` has type mismatch",
                                tf.name
                            )),
                            _ => {}
                        }
                    }
                    if !details.is_empty() {
                        // Locate the annotation span for diagnostics.
                        let tag_lower = lowercase_tag_name(&tag_ir.name);
                        let ann_span = d
                            .annotations
                            .iter()
                            .find(|a| a.name == tag_lower)
                            .map(|a| a.span)
                            .unwrap_or(d.span);
                        return Err(ResolveError::EventTagContractViolated {
                            event: d.name.clone(),
                            tag: tag_ir.name.clone(),
                            details,
                            span: ann_span,
                        });
                    }
                }
                comp.events[event_idx].fields = fields;
                event_idx += 1;
            }
            Decl::EventTag(_) | Decl::Enum(_) => {
                // No body lowering beyond what pass 1 already did.
            }
            Decl::Entity(d) => {
                let fields = d
                    .fields
                    .iter()
                    .map(|f| resolve_entity_field(f, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                comp.entities[entity_idx].fields = fields;
                entity_idx += 1;
            }
            Decl::Physics(d) => {
                let handlers = d
                    .handlers
                    .iter()
                    .map(|h| {
                        let mut scope = LocalScope::new();
                        // self is implicit in physics handlers (the entity
                        // whose action/event is being handled).
                        scope.bind("self", IrType::Unknown);
                        let pattern = resolve_physics_pattern(&h.pattern, &mut scope, symbols, comp)?;
                        let where_clause = h
                            .where_clause
                            .as_ref()
                            .map(|w| resolve_expr(w, &mut scope, symbols))
                            .transpose()?;
                        let body = resolve_stmts(&h.body, &mut scope, symbols)?;
                        Ok::<_, ResolveError>(PhysicsHandlerIR {
                            pattern,
                            where_clause,
                            body,
                            span: h.span,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.physics[physics_idx].handlers = handlers;
                physics_idx += 1;
            }
            Decl::Mask(d) => {
                let mut scope = LocalScope::new();
                scope.bind("self", IrType::Unknown);
                // Task 138: resolve the `from` expression before binding
                // the head's target parameter so the enumeration source
                // can only reference `self` — the target binding is what
                // this expression *produces*, not a free variable.
                let candidate_source = match &d.candidate_source {
                    Some(expr) => Some(resolve_expr(expr, &mut scope, symbols)?),
                    None => None,
                };
                let head = resolve_action_head(&d.head, &mut scope, symbols);
                let predicate = resolve_expr(&d.predicate, &mut scope, symbols)?;
                // Closed-operator-set validation (spec §2.5). Mask
                // predicates compile to GPU boolean kernels and stay
                // restricted by design, even as physics bodies gained
                // `for`/`match`.
                validate_mask_body(&d.head.name, &predicate)?;
                if let Some(cs) = &candidate_source {
                    validate_mask_body(&d.head.name, cs)?;
                }
                comp.masks[mask_idx].head = head;
                comp.masks[mask_idx].candidate_source = candidate_source;
                comp.masks[mask_idx].predicate = predicate;
                mask_idx += 1;
            }
            Decl::Scoring(d) => {
                let entries = d
                    .entries
                    .iter()
                    .map(|e| {
                        let mut scope = LocalScope::new();
                        scope.bind("self", IrType::Unknown);
                        let head = resolve_action_head(&e.head, &mut scope, symbols);
                        let expr = resolve_expr(&e.expr, &mut scope, symbols)?;
                        // Closed-operator-set validation (spec §2.5).
                        // Scoring rows share the mask kernel surface;
                        // reject `match` at resolve time. Physics retains
                        // the richer `for`/`match` surface per task 155.
                        validate_scoring_body(&expr)?;
                        Ok::<_, ResolveError>(ScoringEntryIR { head, expr, span: e.span })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // `per_ability` rows — GPU ability evaluation Phase 2.
                // The row's three clauses share a single local scope
                // seeded with `self` (the scoring agent) and `ability`
                // (the implicit binder for the currently-iterated
                // ability slot; `ability::tag(...)`-style primitives
                // read it out of context without a user-visible bind).
                let per_ability_rows = d
                    .per_ability_rows
                    .iter()
                    .map(|r| {
                        let mut scope = LocalScope::new();
                        scope.bind("self", IrType::Unknown);
                        // Seed the implicit `ability` local so expressions
                        // like `ability::on_cooldown(ability)` resolve the
                        // inner argument without an UnknownIdent error.
                        // Phase 3 lowers this into the per-iteration slot
                        // index; at Phase 2 it's just a stand-in.
                        scope.bind("ability", IrType::AbilityId);
                        // Seed `target` as the ability's cast target
                        // (see spec's `pick_ability` fixture:
                        // `target.hp_frac`). Phase 3 resolves it via
                        // the row's `target:` clause; at Phase 2 it
                        // stays a typed stand-in so per_ability bodies
                        // referencing the cast target resolve cleanly.
                        scope.bind("target", IrType::AgentId);
                        let guard = match &r.guard {
                            Some(g) => Some(resolve_expr(g, &mut scope, symbols)?),
                            None => None,
                        };
                        let score = resolve_expr(&r.score, &mut scope, symbols)?;
                        let target = match &r.target {
                            Some(t) => Some(resolve_expr(t, &mut scope, symbols)?),
                            None => None,
                        };
                        // `weights:` — utility-table addend resolved
                        // against the same scope as `score:`. The
                        // lowerer composes `score + weights`. Closes
                        // Gap C from `gaps_observed.md` (2026-05-11).
                        let weights = match &r.weights {
                            Some(w) => Some(resolve_expr(w, &mut scope, symbols)?),
                            None => None,
                        };
                        // Apply the same closed-operator-set validation
                        // the standard rows take — per_ability rows
                        // lower onto the same kernel surface (scoring +
                        // apply_actions side buffer), so `match`-in-body
                        // stays rejected until Phase 4 proves otherwise.
                        if let Some(g) = &guard {
                            validate_scoring_body(g)?;
                        }
                        validate_scoring_body(&score)?;
                        if let Some(t) = &target {
                            validate_scoring_body(t)?;
                        }
                        if let Some(w) = &weights {
                            validate_scoring_body(w)?;
                        }
                        Ok::<_, ResolveError>(PerAbilityRowIR {
                            name: r.name.clone(),
                            guard,
                            score,
                            target,
                            weights,
                            span: r.span,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.scoring[scoring_idx].entries = entries;
                comp.scoring[scoring_idx].per_ability_rows = per_ability_rows;
                scoring_idx += 1;
            }
            Decl::View(d) => {
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let return_ty = resolve_type(&d.return_ty, symbols);
                // Plan G G3f follow-up — gap (b) from threats_struct_probe.sim.
                // When @dispatch(per_agent_event_scan) is set, the WGSL emit
                // already binds `source_candidate` as a per-(observer, source)
                // pair-iteration kernel local (cg/emit/wgsl_body.rs). Surface
                // it at the AST resolver layer so authors can write
                // `agents.<field>(source_candidate)` in the fold body —
                // unblocks real per-cell content (cell.source = caster id,
                // cell.expires_at_tick from busy_until_tick, etc.) instead
                // of the placeholder constants threats_struct_probe writes
                // today.
                let is_per_agent_event_scan = d.annotations.iter().any(|a| {
                    a.name == "dispatch"
                        && a.args.iter().any(|arg| {
                            arg.key.is_none()
                                && matches!(
                                    &arg.value,
                                    ast::AnnotationValue::Ident(name)
                                        if name == "per_agent_event_scan"
                                )
                        })
                });
                let body = match &d.body {
                    ast::ViewBody::Expr(e) => ViewBodyIR::Expr(resolve_expr(e, &mut scope, symbols)?),
                    ast::ViewBody::Fold { initial, handlers, clamp } => {
                        let initial = resolve_expr(initial, &mut scope, symbols)?;
                        let handlers_ir = handlers
                            .iter()
                            .map(|h| {
                                let mut inner = LocalScope::new();
                                // Copy outer scope bindings into the inner
                                // scope so fold handlers see the view
                                // parameters.
                                for binding in scope.stack.iter().flatten() {
                                    inner.stack[0].push(binding.clone());
                                }
                                inner.next_id = scope.next_id;
                                if is_per_agent_event_scan {
                                    // Same emit identifier name the WGSL
                                    // kernel uses (`let source_candidate =
                                    // gid.y;`). Authors writing
                                    // `agents.<field>(source_candidate)`
                                    // therefore lower to the same SoA read
                                    // the per-pair iteration's source slot
                                    // already addresses.
                                    inner.bind("source_candidate", IrType::AgentId);
                                }
                                let pattern =
                                    resolve_event_pattern(&h.pattern, &mut inner, symbols);
                                let body = resolve_stmts(&h.body, &mut inner, symbols)?;
                                Ok::<_, ResolveError>(FoldHandlerIR {
                                    pattern,
                                    body,
                                    span: h.span,
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let clamp = match clamp {
                            Some((lo, hi)) => Some((
                                resolve_expr(lo, &mut scope, symbols)?,
                                resolve_expr(hi, &mut scope, symbols)?,
                            )),
                            None => None,
                        };
                        ViewBodyIR::Fold { initial, handlers: handlers_ir, clamp }
                    }
                };
                // Fold-body operator-set validation (spec §2.3). Only
                // `@materialized` fold views are checked — lazy views are
                // plain expressions and already restricted by the
                // stdlib-call surface.
                if let ViewBodyIR::Fold { handlers, .. } = &body {
                    for h in handlers {
                        validate_fold_body(&d.name, &h.body)?;
                    }
                }
                // Parse and validate `@decay(rate=R, per=tick)` if present.
                let decay = lower_decay_hint(&d.annotations, &d.body, symbols)?;
                // Parse `@lazy` / `@materialized(on_event=[...],
                // storage=<hint>)` to set the view kind. Spec §2.3 + §9 D31.
                let kind = lower_view_kind(&d.name, &d.annotations, &d.body, d.span)?;
                comp.views[view_idx].params = params;
                comp.views[view_idx].return_ty = return_ty;
                comp.views[view_idx].body = body;
                comp.views[view_idx].decay = decay;
                comp.views[view_idx].kind = kind;
                view_idx += 1;
            }
            Decl::Verb(d) => {
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let action_args = d
                    .action
                    .args
                    .iter()
                    .map(|a| resolve_call_arg(a, &mut scope, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                let action = VerbActionIR {
                    name: d.action.name.clone(),
                    args: action_args,
                    span: d.action.span,
                };
                let when = d
                    .when
                    .as_ref()
                    .map(|e| resolve_expr(e, &mut scope, symbols))
                    .transpose()?;
                // Verb body — `emit <Event>{...}` and / or `apply_ability
                // <a> [by <c>] [target <t>]` statements, in source order.
                // Both lower into the synthesised cascade physics body via
                // verb_expand. Each statement carries a fresh `IrStmt`
                // variant (Emit / ApplyAbility) so the lift is identity —
                // see `crates/dsl_compiler/src/cg/lower/verb_expand.rs`.
                let body = d
                    .body
                    .iter()
                    .map(|s| match s {
                        ast::VerbBodyStmt::Emit(e) => {
                            resolve_emit(e, &mut scope, symbols).map(IrStmt::Emit)
                        }
                        ast::VerbBodyStmt::ApplyAbility(a) => {
                            // Symbolic-name surface (2026-05-12): when the
                            // parser captured a bare-identifier ability
                            // operand on `ability_name`, the resolver
                            // synthesizes a placeholder `LitInt(0)` for
                            // the `ability` IR expression so resolve_expr
                            // doesn't try to look up the name in the
                            // identifier scope (where it would either
                            // mis-resolve to an EnumVariant with no owner
                            // type or surface UnknownIdent). The lowerer
                            // reads `ability_name` and substitutes the
                            // resolved AbilityId from the registry.
                            let ability = if a.ability_name.is_some() {
                                IrExprNode {
                                    kind: IrExpr::LitInt(0),
                                    span: a.ability.span,
                                }
                            } else {
                                resolve_expr(&a.ability, &mut scope, symbols)?
                            };
                            let caster = match &a.caster {
                                Some(c) => Some(resolve_expr(c, &mut scope, symbols)?),
                                None => None,
                            };
                            let target = match &a.target {
                                Some(t) => Some(resolve_expr(t, &mut scope, symbols)?),
                                None => None,
                            };
                            Ok(IrStmt::ApplyAbility {
                                ability,
                                ability_name: a.ability_name.clone(),
                                caster,
                                target,
                                span: a.span,
                            })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scoring = d
                    .scoring
                    .as_ref()
                    .map(|e| resolve_expr(e, &mut scope, symbols))
                    .transpose()?;
                comp.verbs[verb_idx].params = params;
                comp.verbs[verb_idx].action = action;
                comp.verbs[verb_idx].when = when;
                comp.verbs[verb_idx].body = body;
                comp.verbs[verb_idx].scoring = scoring;
                verb_idx += 1;
            }
            Decl::Invariant(d) => {
                let mut scope = LocalScope::new();
                // Invariants don't have an implicit self — only their scope
                // params. A metric / probe / invariant that mentions `self`
                // without a param is an error (SelfInTopLevel).
                let scope_params = resolve_params(&d.scope, &mut scope, symbols);
                let predicate = resolve_expr(&d.predicate, &mut scope, symbols)?;
                comp.invariants[invariant_idx].scope = scope_params;
                comp.invariants[invariant_idx].predicate = predicate;
                invariant_idx += 1;
            }
            Decl::Probe(d) => {
                let asserts = d
                    .asserts
                    .iter()
                    // Drop the parse-and-discard Raw form — it lives
                    // outside the IR's closed Count/Pr/Mean shape.
                    // The probe still appears in the IR (just with
                    // fewer asserts) so the lowering pipeline keeps
                    // its structural invariants. Re-introduce when
                    // the probe runner grows a generic-expr evaluator.
                    .filter(|a| !matches!(a, AssertExpr::Raw { .. }))
                    .map(|a| {
                        let mut scope = LocalScope::new();
                        scope.bind("self", IrType::Unknown);
                        scope.bind("action", IrType::Unknown);
                        resolve_assert(a, &mut scope, symbols)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.probes[probe_idx].asserts = asserts;
                probe_idx += 1;
            }
            Decl::Metric(block) => {
                for m in &block.metrics {
                    let mut scope = LocalScope::new();
                    let value = resolve_expr(&m.value, &mut scope, symbols)?;
                    let cond = m
                        .conditioned_on
                        .as_ref()
                        .map(|e| resolve_expr(e, &mut scope, symbols))
                        .transpose()?;
                    // `alert when` clauses see implicit bindings: `value`
                    // (scalar metrics), `max_bin` (histograms), AND the
                    // metric's own name (designers commonly read the
                    // most-recent value by referring back to the metric
                    // name itself — `alert_when: pack_aggression > 50.0`
                    // — instead of the generic `value` binding). Bind
                    // all three as Unknown so 1b can specialize.
                    let mut alert_scope = LocalScope::new();
                    alert_scope.bind("value", IrType::Unknown);
                    alert_scope.bind("max_bin", IrType::Unknown);
                    alert_scope.bind(&m.name, IrType::Unknown);
                    let alert = m
                        .alert_when
                        .as_ref()
                        .map(|e| resolve_expr(e, &mut alert_scope, symbols))
                        .transpose()?;
                    let slot = metric_start_idx;
                    comp.metrics[slot].value = value;
                    comp.metrics[slot].conditioned_on = cond;
                    comp.metrics[slot].alert_when = alert;
                    metric_start_idx += 1;
                }
            }
            Decl::Config(_) => {
                // Pass 1 already materialised the full IR (fields + defaults).
                // No body expressions to lower.
            }
            Decl::Query(_) => {}
            Decl::SpatialQuery(d) => {
                // Phase 7 Task 4. Mirrors the `Decl::View` arm: fresh
                // scope, bind params (the first two of which MUST be
                // `self` and `candidate`; surface a typed error
                // otherwise), resolve the filter, fill the reserved
                // IR slot.
                if d.params.len() < 2
                    || d.params[0].name != "self"
                    || d.params[1].name != "candidate"
                {
                    return Err(ResolveError::SpatialQueryRequiresSelfCandidateBinders {
                        decl_name: d.name.clone(),
                        span: d.span,
                    });
                }
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let filter = resolve_expr(&d.filter, &mut scope, symbols)?;
                comp.spatial_queries[spatial_query_idx].params = params;
                comp.spatial_queries[spatial_query_idx].filter = filter;
                spatial_query_idx += 1;
            }
            Decl::Init(_) => {
                // Plan E-A6 — handled in build_helper, no IR pass-2 work.
            }
            Decl::Debug(_) => {
                // `debug { ... }` — handled in build_helper, no IR pass-2 work.
            }
            Decl::AgentField(_) => {
                // Gap plague_city#P-A — `field <name>: <type>` decls
                // are interned process-locally by
                // `dsl_compiler::custom_agent_fields::populate`
                // BEFORE lowering. No IR pass-2 work here.
            }
            Decl::RegionKind(_) => {
                // Pass-1 already populated the IR slot. Pass-2 is a
                // no-op for the kind decl itself; `RegionIndices`
                // below merges into the same slot.
            }
            Decl::Index(d) => {
                // Phase 2b — validate build body: every engine
                // helper call must name a known helper, every bare
                // identifier must be either the region param,
                // a let-bound local, or a known top-level constant.
                validate_index_build_body(d)?;
            }
            Decl::RegionIndices(d) => {
                // Pair `region_indices <Name> { … }` with its
                // matching `region_kind <Name>` slot from pass-1.
                let Some(VoxelRegionKindId(idx)) = symbols.region_kinds.get(&d.name).copied()
                else {
                    return Err(ResolveError::InvalidViewKind {
                        view_name: d.name.clone(),
                        detail: format!(
                            "`region_indices {}` references kind `{}` with no `region_kind` decl — \
                             every `region_indices` must pair with a `region_kind` of the same name",
                            d.name, d.name
                        ),
                        span: d.span,
                    });
                };
                let slot = &mut comp.region_kinds[idx as usize];
                if !slot.index_kind_names.is_empty() {
                    // Duplicate `region_indices` for the same kind —
                    // spec §6.1.2 implies a 1:1 mapping (one
                    // `region_kind` + one `region_indices` per name).
                    return Err(ResolveError::InvalidViewKind {
                        view_name: d.name.clone(),
                        detail: format!(
                            "duplicate `region_indices {}` — each region kind may have at most one `region_indices` decl",
                            d.name
                        ),
                        span: d.span,
                    });
                }
                slot.index_kind_names = d.index_kinds.clone();
                // Phase 2a cross-decl validation: every name in the
                // `region_indices` body must resolve to a declared
                // `index <name>(...)` decl. Case-insensitive match
                // against the symbol table — spec example writes
                // both `region_indices Settlement { Navgrid }`
                // (PascalCase) and `index navgrid(...)` (lowercase),
                // suggesting either style is conventionally
                // acceptable; we accept both and TODO-mark the
                // canonicalisation choice for tightening once a
                // real fixture forces it.
                let known_indices_lc: Vec<String> = symbols
                    .indices
                    .keys()
                    .map(|k| k.to_lowercase())
                    .collect();
                for kind in &d.index_kinds {
                    let kind_lc = kind.to_lowercase();
                    if !known_indices_lc.contains(&kind_lc) {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: d.name.clone(),
                            detail: format!(
                                "`region_indices {} {{ {} }}` references unknown index `{}` — \
                                 declare an `index {}(region: VoxelRegion) -> <Output> {{ … }}` decl, \
                                 or remove from the region_indices body",
                                d.name, kind, kind, kind_lc
                            ),
                            span: d.span,
                        });
                    }
                }
            }
            Decl::Table(d) => {
                let TableId(idx) = symbols.tables[&d.name];
                let element_ty = match d.element_ty_name.as_str() {
                    "u32" => IrType::U32,
                    other => {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: d.name.clone(),
                            detail: format!(
                                "table `{}`: element type `{other}` not \
                                 supported (first cut accepts only `u32`)",
                                d.name
                            ),
                            span: d.span,
                        });
                    }
                };
                if d.values.len() as u32 != d.length {
                    return Err(ResolveError::InvalidViewKind {
                        view_name: d.name.clone(),
                        detail: format!(
                            "table `{}`: declared length {} ≠ initializer length {}",
                            d.name,
                            d.length,
                            d.values.len()
                        ),
                        span: d.span,
                    });
                }
                for (i, v) in d.values.iter().enumerate() {
                    if *v < 0 || *v > u32::MAX as i64 {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: d.name.clone(),
                            detail: format!(
                                "table `{}`: value at index {i} ({v}) out of range for u32",
                                d.name
                            ),
                            span: d.span,
                        });
                    }
                }
                comp.tables[idx as usize].element_ty = element_ty;
                comp.tables[idx as usize].values = d.values.clone();
            }
            Decl::Belief(d) => {
                // Plan I — resolve the belief body into the reserved
                // ViewIR slot. Shape mirrors the `Decl::View` fold
                // path: observer + key params in scope; each
                // propagation handler gets the event-pattern binders
                // in its inner scope. Social-merge handlers reuse the
                // same event-pattern binder shape, then look up the
                // named source-agent identifier among those binders.
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let return_ty = resolve_type(&d.return_ty, symbols);
                validate_belief_signature(&d.name, &params, &return_ty, d.span)?;
                // Mirror the view-decl resolver: `@dispatch(per_agent_event_scan)`
                // surfaces `source_candidate` as an inner-scope binder
                // so fold bodies can call `agents.<field>(source_candidate)`
                // and have it lower to the per-pair iteration's source
                // slot. Lets belief decls share the same fold-body
                // surface as view decls (threats_struct_probe.sim relies
                // on this).
                let is_per_agent_event_scan = d.annotations.iter().any(|a| {
                    a.name == "dispatch"
                        && a.args.iter().any(|arg| {
                            arg.key.is_none()
                                && matches!(
                                    &arg.value,
                                    ast::AnnotationValue::Ident(name)
                                        if name == "per_agent_event_scan"
                                )
                        })
                });
                let body = match &d.body {
                    ast::ViewBody::Expr(_) => {
                        // Parser only emits the Fold body for belief
                        // decls today; this arm is a defensive guard
                        // in case the parser surface grows. Map to a
                        // typed error so callers don't see a panic.
                        return Err(ResolveError::UnsupportedBeliefSignature {
                            belief_name: d.name.clone(),
                            detail: "belief bodies must use `initial: … on … { … }` fold shape, \
                                     not a single expression"
                                .to_string(),
                            span: d.span,
                        });
                    }
                    ast::ViewBody::Fold { initial, handlers, clamp } => {
                        let initial = resolve_expr(initial, &mut scope, symbols)?;
                        let handlers_ir = handlers
                            .iter()
                            .map(|h| {
                                let mut inner = LocalScope::new();
                                for binding in scope.stack.iter().flatten() {
                                    inner.stack[0].push(binding.clone());
                                }
                                inner.next_id = scope.next_id;
                                if is_per_agent_event_scan {
                                    inner.bind("source_candidate", IrType::AgentId);
                                }
                                let pattern =
                                    resolve_event_pattern(&h.pattern, &mut inner, symbols);
                                let body = resolve_stmts(&h.body, &mut inner, symbols)?;
                                Ok::<_, ResolveError>(FoldHandlerIR {
                                    pattern,
                                    body,
                                    span: h.span,
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let clamp = match clamp {
                            Some((lo, hi)) => Some((
                                resolve_expr(lo, &mut scope, symbols)?,
                                resolve_expr(hi, &mut scope, symbols)?,
                            )),
                            None => None,
                        };
                        ViewBodyIR::Fold { initial, handlers: handlers_ir, clamp }
                    }
                };
                if let ViewBodyIR::Fold { handlers, .. } = &body {
                    for h in handlers {
                        validate_fold_body(&d.name, &h.body)?;
                    }
                }
                let social_merges = d
                    .social_merges
                    .iter()
                    .map(|m| {
                        let mut inner = LocalScope::new();
                        for binding in scope.stack.iter().flatten() {
                            inner.stack[0].push(binding.clone());
                        }
                        inner.next_id = scope.next_id;
                        if is_per_agent_event_scan {
                            inner.bind("source_candidate", IrType::AgentId);
                        }
                        let pattern = resolve_event_pattern(&m.pattern, &mut inner, symbols);
                        let where_clause = match &m.where_clause {
                            Some(e) => Some(resolve_expr(e, &mut inner, symbols)?),
                            None => None,
                        };
                        let source_agent = match inner.lookup(&m.source_agent_name) {
                            Some(b) => b.local,
                            None => {
                                let bound: Vec<String> = inner
                                    .stack
                                    .iter()
                                    .flatten()
                                    .map(|b| b.name.clone())
                                    .collect();
                                return Err(ResolveError::UnknownSocialMergeSource {
                                    belief_name: d.name.clone(),
                                    source_name: m.source_agent_name.clone(),
                                    bound,
                                    span: m.span,
                                });
                            }
                        };
                        let op = match m.op {
                            ast::SocialMergeOpName::BitOr => MergeOp::BitOr,
                            ast::SocialMergeOpName::Max => MergeOp::Max,
                            ast::SocialMergeOpName::Min => MergeOp::Min,
                            ast::SocialMergeOpName::Replace => MergeOp::Replace,
                        };
                        Ok::<_, ResolveError>(SocialMergeHandler {
                            pattern,
                            where_clause,
                            source_agent,
                            op,
                            span: m.span,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let decay = lower_decay_hint_for(
                    &d.annotations,
                    &d.body,
                    symbols,
                    /*is_belief=*/ true,
                )?;
                comp.views[view_idx].params = params;
                comp.views[view_idx].return_ty = return_ty;
                comp.views[view_idx].body = body;
                comp.views[view_idx].social_merges = social_merges;
                comp.views[view_idx].decay = decay;
                view_idx += 1;
            }
            Decl::PhysicsApply(_) => {
                // apply-form: ignored in pass 2 — handled by param_rules lowering (T5+).
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn resolve_type(ty: &ast::TypeRef, symbols: &SymbolTable) -> IrType {
    match &ty.kind {
        ast::TypeKind::Named(n) => {
            if let Some(t) = symbols.stdlib_types.get(n) {
                return t.clone();
            }
            if let Some(r) = symbols.entities.get(n) {
                return IrType::EntityRef(*r);
            }
            if let Some(r) = symbols.events.get(n) {
                return IrType::EventRef(*r);
            }
            if let Some((_, variants)) = symbols.enums.get(n) {
                return IrType::Enum { name: n.clone(), variants: variants.clone() };
            }
            // Probably a user-defined enum / struct we don't have a decl kind
            // for in 1a — keep as Named.
            IrType::Named(n.clone())
        }
        ast::TypeKind::Generic { name, args } => match name.as_str() {
            "SortedVec" | "RingBuffer" | "SmallVec" | "Array" => {
                let (elem, cap) = extract_elem_cap(args, symbols);
                match name.as_str() {
                    "SortedVec" => IrType::SortedVec(Box::new(elem), cap),
                    "RingBuffer" => IrType::RingBuffer(Box::new(elem), cap),
                    "SmallVec" => IrType::SmallVec(Box::new(elem), cap),
                    "Array" => IrType::Array(Box::new(elem), cap),
                    _ => unreachable!(),
                }
            }
            "Option" => {
                if let Some(ast::TypeArg::Type(inner)) = args.first() {
                    IrType::Optional(Box::new(resolve_type(inner, symbols)))
                } else {
                    IrType::Named(name.clone())
                }
            }
            _ => IrType::Named(name.clone()),
        },
        ast::TypeKind::List(inner) => IrType::List(Box::new(resolve_type(inner, symbols))),
        ast::TypeKind::Tuple(inners) => {
            IrType::Tuple(inners.iter().map(|t| resolve_type(t, symbols)).collect())
        }
        ast::TypeKind::Option(inner) => IrType::Optional(Box::new(resolve_type(inner, symbols))),
    }
}

fn extract_elem_cap(args: &[ast::TypeArg], symbols: &SymbolTable) -> (IrType, u16) {
    let mut elem = IrType::Unknown;
    let mut cap: u16 = 0;
    for a in args {
        match a {
            ast::TypeArg::Type(t) => elem = resolve_type(t, symbols),
            ast::TypeArg::Const(n) => cap = u16::try_from(*n).unwrap_or(0),
        }
    }
    (elem, cap)
}

// ---------------------------------------------------------------------------
// Entity fields
// ---------------------------------------------------------------------------

fn resolve_entity_field(
    f: &ast::EntityField,
    symbols: &SymbolTable,
) -> Result<EntityFieldIR, ResolveError> {
    let value = match &f.value {
        ast::EntityFieldValue::Type(t) => EntityFieldValueIR::Type(resolve_type(t, symbols)),
        ast::EntityFieldValue::StructLiteral { ty, fields } => {
            let ty = resolve_type(ty, symbols);
            let fields = fields
                .iter()
                .map(|g| resolve_entity_field(g, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            EntityFieldValueIR::StructLiteral { ty, fields }
        }
        ast::EntityFieldValue::AnonStruct(fields) => {
            let fields = fields
                .iter()
                .map(|g| resolve_entity_field(g, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            EntityFieldValueIR::AnonStruct { fields }
        }
        ast::EntityFieldValue::List(exprs) => {
            let mut scope = LocalScope::new();
            let exprs = exprs
                .iter()
                .map(|e| resolve_expr(e, &mut scope, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            EntityFieldValueIR::List(exprs)
        }
        ast::EntityFieldValue::Expr(e) => {
            let mut scope = LocalScope::new();
            EntityFieldValueIR::Expr(resolve_expr(e, &mut scope, symbols)?)
        }
    };
    Ok(EntityFieldIR {
        name: f.name.clone(),
        value,
        annotations: f.annotations.clone(),
        span: f.span,
    })
}

// ---------------------------------------------------------------------------
// Params / action heads / event patterns
// ---------------------------------------------------------------------------

fn resolve_params(
    params: &[ast::Param],
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Vec<IrParam> {
    params
        .iter()
        .map(|p| {
            let ty = resolve_type(&p.ty, symbols);
            let local = scope.bind(&p.name, ty.clone());
            IrParam { name: p.name.clone(), local, ty, span: p.span }
        })
        .collect()
}

fn resolve_action_head(
    head: &ast::ActionHead,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrActionHead {
    let shape = match &head.shape {
        ActionHeadShape::None => IrActionHeadShape::None,
        ActionHeadShape::Positional(params) => {
            // Task 157 — typed positional heads. Unannotated params
            // default to `AgentId` to preserve the implicit-agent
            // contract every existing target-bound mask relies on
            // (`Attack(target)`, `MoveToward(target)`). Annotated
            // params resolve their type via the shared `resolve_type`
            // pass so `Cast(ability: AbilityId)` surfaces the
            // non-agent head without touching other call sites.
            let bound = params
                .iter()
                .map(|(n, ty)| {
                    let resolved = match ty {
                        Some(t) => resolve_type(t, symbols),
                        None => IrType::AgentId,
                    };
                    let local = scope.bind(n, resolved.clone());
                    (n.clone(), local, resolved)
                })
                .collect();
            IrActionHeadShape::Positional(bound)
        }
        ActionHeadShape::Named(bindings) => {
            let bs = bindings
                .iter()
                .map(|b| resolve_pattern_binding(b, scope, symbols))
                .collect();
            IrActionHeadShape::Named(bs)
        }
    };
    IrActionHead { name: head.name.clone(), shape, span: head.span }
}

fn resolve_event_pattern(
    p: &ast::EventPattern,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrEventPattern {
    let event = symbols.events.get(&p.name).copied();
    let bindings = p
        .bindings
        .iter()
        .map(|b| resolve_pattern_binding(b, scope, symbols))
        .collect();
    IrEventPattern { name: p.name.clone(), event, bindings, span: p.span }
}

/// Resolve a physics `on` pattern. A `PhysicsPattern::Tag` validates that
/// the referenced `event_tag` exists and that every binding names a field
/// declared on the tag. The kind variant wraps the standard event pattern.
fn resolve_physics_pattern(
    p: &ast::PhysicsPattern,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
    comp: &Compilation,
) -> Result<IrPhysicsPattern, ResolveError> {
    match p {
        ast::PhysicsPattern::Kind(pat) => {
            Ok(IrPhysicsPattern::Kind(resolve_event_pattern(pat, scope, symbols)))
        }
        ast::PhysicsPattern::Tag { name, bindings, span } => {
            let Some((tref, _)) = symbols.event_tags.get(name) else {
                let suggestions: Vec<String> = symbols
                    .event_tags
                    .keys()
                    .take(3)
                    .cloned()
                    .collect();
                return Err(ResolveError::UnknownEventTag {
                    name: name.clone(),
                    span: *span,
                    suggestions,
                });
            };
            let tag_ir = &comp.event_tags[tref.0 as usize];
            let allowed: std::collections::HashSet<&str> =
                tag_ir.fields.iter().map(|f| f.name.as_str()).collect();
            for b in bindings {
                // `tick` is always available on every event, not listed on
                // the tag — permit it as a synthetic reference.
                if b.field == "tick" {
                    continue;
                }
                if !allowed.contains(b.field.as_str()) {
                    return Err(ResolveError::TagBindingUnknown {
                        tag: tag_ir.name.clone(),
                        field: b.field.clone(),
                        span: b.span,
                    });
                }
            }
            let resolved_bindings = bindings
                .iter()
                .map(|b| resolve_pattern_binding(b, scope, symbols))
                .collect();
            Ok(IrPhysicsPattern::Tag {
                name: name.clone(),
                tag: Some(*tref),
                bindings: resolved_bindings,
                span: *span,
            })
        }
    }
}

fn resolve_pattern_binding(
    b: &ast::PatternBinding,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrPatternBinding {
    let value = resolve_pattern_value(&b.value, scope, symbols);
    IrPatternBinding { field: b.field.clone(), value, span: b.span }
}

fn resolve_pattern_value(
    v: &ast::PatternValue,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrPattern {
    match v {
        ast::PatternValue::Bind(n) => {
            let local = scope.bind(n, IrType::Unknown);
            IrPattern::Bind { name: n.clone(), local }
        }
        ast::PatternValue::Ctor { name, inner } => {
            let ctor = ctor_ref(name, symbols);
            let inner = inner
                .iter()
                .map(|p| resolve_pattern_value(p, scope, symbols))
                .collect();
            IrPattern::Ctor { name: name.clone(), ctor, inner }
        }
        ast::PatternValue::Struct { name, bindings } => {
            let ctor = ctor_ref(name, symbols);
            let bindings = bindings
                .iter()
                .map(|b| resolve_pattern_binding(b, scope, symbols))
                .collect();
            IrPattern::Struct { name: name.clone(), ctor, bindings }
        }
        ast::PatternValue::Expr(e) => {
            // Best-effort: try to resolve the expression against the current
            // scope. If it fails (unknown ident), we keep Raw.
            let mut throwaway = scope_clone(scope);
            match resolve_expr(e, &mut throwaway, symbols) {
                Ok(ir) => IrPattern::Expr(ir),
                Err(_) => IrPattern::Expr(IrExprNode {
                    kind: IrExpr::Raw(Box::new(e.clone())),
                    span: e.span,
                }),
            }
        }
        ast::PatternValue::Wildcard => IrPattern::Wildcard,
    }
}

fn scope_clone(s: &LocalScope) -> LocalScope {
    LocalScope {
        stack: s.stack.clone(),
        next_id: s.next_id,
        self_bound: s.self_bound,
    }
}

fn ctor_ref(name: &str, symbols: &SymbolTable) -> Option<CtorRef> {
    if let Some(r) = symbols.events.get(name) {
        return Some(CtorRef::Event(*r));
    }
    if let Some(r) = symbols.entities.get(name) {
        return Some(CtorRef::Entity(*r));
    }
    None
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn resolve_stmts(
    stmts: &[Stmt],
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<Vec<IrStmt>, ResolveError> {
    stmts.iter().map(|s| resolve_stmt(s, scope, symbols)).collect()
}

fn resolve_stmt(
    stmt: &Stmt,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrStmt, ResolveError> {
    match stmt {
        Stmt::Let { name, value, span } => {
            let v = resolve_expr(value, scope, symbols)?;
            let local = scope.bind(name, IrType::Unknown);
            Ok(IrStmt::Let { name: name.clone(), local, value: v, span: *span })
        }
        Stmt::Emit(e) => Ok(IrStmt::Emit(resolve_emit(e, scope, symbols)?)),
        Stmt::ApplyAbility(a) => {
            // #132A: opaque ability_expr resolution. The expression
            // typically resolves to AbilityId (e.g. `self.action_ability`);
            // codegen reads it as u32 at the dispatch boundary.
            //
            // Slice δ part 3 (#161): optional `by <caster>` operand —
            // resolves to an AgentId expression in the rule's scope.
            // Typically `e.actor` for PerEvent rules destructuring the
            // event payload, or `self` for explicit PerAgent self-cast.
            //
            // Symbolic-name surface (2026-05-12): see the verb-body arm
            // above for the rationale — when `ability_name` is set we
            // skip resolving the placeholder ability expression and let
            // the lowerer substitute the registry-resolved AbilityId.
            let ability = if a.ability_name.is_some() {
                IrExprNode {
                    kind: IrExpr::LitInt(0),
                    span: a.ability.span,
                }
            } else {
                resolve_expr(&a.ability, scope, symbols)?
            };
            let caster = match &a.caster {
                Some(c) => Some(resolve_expr(c, scope, symbols)?),
                None => None,
            };
            let target = match &a.target {
                Some(t) => Some(resolve_expr(t, scope, symbols)?),
                None => None,
            };
            Ok(IrStmt::ApplyAbility {
                ability,
                ability_name: a.ability_name.clone(),
                caster,
                target,
                span: a.span,
            })
        }
        Stmt::For { binder, iter, filter, body, span } => {
            let iter_ir = resolve_expr(iter, scope, symbols)?;
            scope.push();
            let local = scope.bind(binder, IrType::Unknown);
            let filter_ir = filter
                .as_ref()
                .map(|f| resolve_expr(f, scope, symbols))
                .transpose()?;
            let body_ir = resolve_stmts(body, scope, symbols)?;
            scope.pop();
            Ok(IrStmt::For {
                binder: local,
                binder_name: binder.clone(),
                iter: iter_ir,
                filter: filter_ir,
                body: body_ir,
                span: *span,
            })
        }
        Stmt::ForEachAgent { binder, body, span } => {
            // Bind the per-iteration variable in a fresh scope; reads of
            // `<binder>` and `<binder>.<field>` inside the body resolve
            // through the standard local-binding path. Lowering wires the
            // binder name into `ctx.fold_binder_name` so the candidate-
            // side `AgentRef::PerPairCandidate` channel handles those
            // reads. Iteration order is the deterministic linear scan
            // `0..agent_cap` (P3 cross-backend parity, P11 reduction
            // determinism — same on CPU and GPU).
            scope.push();
            let local = scope.bind(binder, IrType::Unknown);
            let body_ir = resolve_stmts(body, scope, symbols)?;
            scope.pop();
            Ok(IrStmt::ForEachAgent {
                binder: local,
                binder_name: binder.clone(),
                body: body_ir,
                span: *span,
            })
        }
        Stmt::If { cond, then_body, else_body, span } => {
            let cond = resolve_expr(cond, scope, symbols)?;
            scope.push();
            let then_body = resolve_stmts(then_body, scope, symbols)?;
            scope.pop();
            let else_body = match else_body {
                Some(b) => {
                    scope.push();
                    let r = resolve_stmts(b, scope, symbols)?;
                    scope.pop();
                    Some(r)
                }
                None => None,
            };
            Ok(IrStmt::If { cond, then_body, else_body, span: *span })
        }
        Stmt::Match { scrutinee, arms, span } => {
            let scrutinee = resolve_expr(scrutinee, scope, symbols)?;
            let arms = arms
                .iter()
                .map(|a| {
                    scope.push();
                    let pattern = resolve_pattern_value(&a.pattern, scope, symbols);
                    let body = resolve_stmts(&a.body, scope, symbols)?;
                    scope.pop();
                    Ok::<_, ResolveError>(IrStmtMatchArm { pattern, body, span: a.span })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrStmt::Match { scrutinee, arms, span: *span })
        }
        Stmt::SelfUpdate { op, value, span } => {
            let value = resolve_expr(value, scope, symbols)?;
            Ok(IrStmt::SelfUpdate { op: op.clone(), value, span: *span })
        }
        Stmt::SelfAppend { fields, span } => {
            // Plan G G3b/G3c — resolve each `field: <expr>` binding.
            // The field name list is preserved verbatim; the per-cell
            // struct layout is inferred from these names + the
            // resolved expression types at lowering time.
            let resolved: Vec<IrFieldInit> = fields
                .iter()
                .map(|f| {
                    let value = resolve_expr(&f.value, scope, symbols)?;
                    Ok::<_, ResolveError>(IrFieldInit {
                        name: f.name.clone(),
                        value,
                        span: f.span,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrStmt::SelfAppend { fields: resolved, span: *span })
        }
        Stmt::Expr(e) => Ok(IrStmt::Expr(resolve_expr(e, scope, symbols)?)),
        Stmt::BeliefObserve(b) => {
            // Validate that each assigned field is a known BeliefState field.
            const BELIEF_FIELDS: &[&str] = &[
                "last_known_pos",
                "last_known_hp",
                "last_known_max_hp",
                "last_known_creature_type",
                "last_updated_tick",
                "confidence",
            ];
            for f in &b.fields {
                if !BELIEF_FIELDS.contains(&f.name.as_str()) {
                    return Err(ResolveError::UnknownBeliefField {
                        field: f.name.clone(),
                        valid: BELIEF_FIELDS.to_vec(),
                        span: f.span,
                    });
                }
            }
            // Resolve the observer / target as plain identifier expressions
            // so they bind to the existing local scope (e.g. event bindings).
            let observer_span = b.span;
            let target_span = b.span;
            let observer_expr = crate::ast::Expr {
                kind: crate::ast::ExprKind::Ident(b.observer.clone()),
                span: observer_span,
            };
            let target_expr = crate::ast::Expr {
                kind: crate::ast::ExprKind::Ident(b.target.clone()),
                span: target_span,
            };
            let observer = resolve_expr(&observer_expr, scope, symbols)?;
            let target = resolve_expr(&target_expr, scope, symbols)?;
            let fields = b
                .fields
                .iter()
                .map(|f| {
                    Ok::<_, ResolveError>(crate::ir::IrFieldInit {
                        name: f.name.clone(),
                        value: resolve_expr(&f.value, scope, symbols)?,
                        span: f.span,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrStmt::BeliefObserve { observer, target, fields, span: b.span })
        }
    }
}

fn resolve_emit(
    e: &ast::EmitStmt,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrEmit, ResolveError> {
    let event = symbols.events.get(&e.event_name).copied();
    let fields = e
        .fields
        .iter()
        .map(|f| {
            Ok::<_, ResolveError>(IrFieldInit {
                name: f.name.clone(),
                value: resolve_expr(&f.value, scope, symbols)?,
                span: f.span,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IrEmit { event_name: e.event_name.clone(), event, fields, span: e.span })
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn resolve_expr(
    e: &Expr,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExprNode, ResolveError> {
    let span = e.span;
    let kind = match &e.kind {
        ExprKind::Int(v) => IrExpr::LitInt(*v),
        ExprKind::Float(v) => IrExpr::LitFloat(*v),
        ExprKind::Bool(v) => IrExpr::LitBool(*v),
        ExprKind::String(v) => IrExpr::LitString(v.clone()),
        ExprKind::Ident(name) => resolve_ident(name, span, scope, symbols)?,
        ExprKind::Field(base, name) => {
            // Fast path: `<namespace>.<field>` where `<namespace>` is a bare
            // identifier naming a typed stdlib namespace.
            if let ExprKind::Ident(ns_name) = &base.kind {
                if scope.lookup(ns_name).is_none() {
                    if let Some(ns) = symbols.stdlib_namespaces.get(ns_name) {
                        // `config.<block>` — tag the block with Unknown type;
                        // the outer `Field(_, "<field>")` wrap below promotes
                        // it into a typed `config.<block>.<field>` lookup.
                        let ty = if *ns == NamespaceId::Config {
                            if symbols.configs.contains_key(name) {
                                IrType::Unknown
                            } else {
                                return Err(ResolveError::UnknownIdent {
                                    name: format!("config.{name}"),
                                    span,
                                    suggestions: symbols
                                        .configs
                                        .keys()
                                        .take(3)
                                        .cloned()
                                        .collect(),
                                });
                            }
                        } else {
                            stdlib::field_type(*ns, name).unwrap_or(IrType::Unknown)
                        };
                        return Ok(IrExprNode {
                            kind: IrExpr::NamespaceField {
                                ns: *ns,
                                field: name.clone(),
                                ty,
                            },
                            span,
                        });
                    }
                }
            }
            // Two-hop `config.<block>.<field>` — the inner `config.<block>`
            // resolved above as `NamespaceField{ns:Config, field:<block>}`;
            // fold this access into a single lookup carrying the full path
            // and the declared field type.
            if let ExprKind::Field(inner_base, inner_field) = &base.kind {
                if let ExprKind::Ident(ns_name) = &inner_base.kind {
                    if scope.lookup(ns_name).is_none() {
                        if let Some(ns) = symbols.stdlib_namespaces.get(ns_name) {
                            if *ns == NamespaceId::Config {
                                let Some((_, field_types)) = symbols.configs.get(inner_field)
                                else {
                                    return Err(ResolveError::UnknownIdent {
                                        name: format!("config.{inner_field}"),
                                        span,
                                        suggestions: symbols
                                            .configs
                                            .keys()
                                            .take(3)
                                            .cloned()
                                            .collect(),
                                    });
                                };
                                let Some(ty) = field_types.get(name) else {
                                    let suggestions: Vec<String> =
                                        field_types.keys().take(3).cloned().collect();
                                    return Err(ResolveError::UnknownIdent {
                                        name: format!("config.{inner_field}.{name}"),
                                        span,
                                        suggestions,
                                    });
                                };
                                return Ok(IrExprNode {
                                    kind: IrExpr::NamespaceField {
                                        ns: NamespaceId::Config,
                                        field: format!("{inner_field}.{name}"),
                                        ty: ty.clone(),
                                    },
                                    span,
                                });
                            }
                        }
                    }
                }
            }
            let base_ir = resolve_expr(base, scope, symbols)?;
            IrExpr::Field {
                base: Box::new(base_ir),
                field_name: name.clone(),
                field: None,
            }
        }
        ExprKind::Index(base, idx) => IrExpr::Index(
            Box::new(resolve_expr(base, scope, symbols)?),
            Box::new(resolve_expr(idx, scope, symbols)?),
        ),
        ExprKind::Call(callee, args) => resolve_call(callee, args, span, scope, symbols)?,
        ExprKind::Binary { op, lhs, rhs } => {
            // GPU ability evaluation Phase 2: the `ability::hint ==
            // <ident>` shape compares the currently-scored ability's
            // hint against a lowercase hint literal. The RHS spelling
            // (`damage`, `defense`, `crowd_control`, `utility`) would
            // otherwise resolve as an UnknownIdent — so when one side
            // of an equality is `ability::hint`, we promote a bare
            // hint ident on the other side to `AbilityHintLit` before
            // falling through to generic Binary resolution.
            let (lhs_r, rhs_r) = resolve_hint_compare_or_default(*op, lhs, rhs, scope, symbols)?;
            IrExpr::Binary(*op, Box::new(lhs_r), Box::new(rhs_r))
        }
        ExprKind::Unary { op, rhs } => {
            IrExpr::Unary(*op, Box::new(resolve_expr(rhs, scope, symbols)?))
        }
        ExprKind::In { item, set } => IrExpr::In(
            Box::new(resolve_expr(item, scope, symbols)?),
            Box::new(resolve_expr(set, scope, symbols)?),
        ),
        ExprKind::Contains { set, item } => IrExpr::Contains(
            Box::new(resolve_expr(set, scope, symbols)?),
            Box::new(resolve_expr(item, scope, symbols)?),
        ),
        ExprKind::Quantifier { kind, binder, iter, body } => {
            let iter_ir = resolve_expr(iter, scope, symbols)?;
            scope.push();
            let local = scope.bind(binder, IrType::Unknown);
            let body_ir = resolve_expr(body, scope, symbols)?;
            scope.pop();
            IrExpr::Quantifier {
                kind: *kind,
                binder: local,
                binder_name: binder.clone(),
                iter: Box::new(iter_ir),
                body: Box::new(body_ir),
            }
        }
        ExprKind::Fold { kind, binder, iter, body } => {
            let iter_ir = iter
                .as_ref()
                .map(|i| resolve_expr(i, scope, symbols))
                .transpose()?;
            scope.push();
            let local = binder.as_ref().map(|b| scope.bind(b, IrType::Unknown));
            let body_ir = resolve_expr(body, scope, symbols)?;
            scope.pop();
            IrExpr::Fold {
                kind: *kind,
                binder: local,
                binder_name: binder.clone(),
                iter: iter_ir.map(Box::new),
                body: Box::new(body_ir),
            }
        }
        ExprKind::List(items) => IrExpr::List(
            items
                .iter()
                .map(|i| resolve_expr(i, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ExprKind::Tuple(items) => IrExpr::Tuple(
            items
                .iter()
                .map(|i| resolve_expr(i, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ExprKind::Struct { name, fields } => {
            let ctor = ctor_ref(name, symbols);
            let fields = fields
                .iter()
                .map(|f| {
                    Ok::<_, ResolveError>(IrFieldInit {
                        name: f.name.clone(),
                        value: resolve_expr(&f.value, scope, symbols)?,
                        span: f.span,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::StructLit { name: name.clone(), ctor, fields }
        }
        ExprKind::Ctor { name, args } => {
            let ctor = ctor_ref(name, symbols);
            let args = args
                .iter()
                .map(|a| resolve_expr(a, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::Ctor { name: name.clone(), ctor, args }
        }
        ExprKind::Match { scrutinee, arms } => {
            let scrutinee = resolve_expr(scrutinee, scope, symbols)?;
            let arms = arms
                .iter()
                .map(|a| {
                    scope.push();
                    let pattern = resolve_pattern_value(&a.pattern, scope, symbols);
                    let body = resolve_expr(&a.body, scope, symbols)?;
                    scope.pop();
                    Ok::<_, ResolveError>(IrMatchArm { pattern, body, span: a.span })
                })
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::Match { scrutinee: Box::new(scrutinee), arms }
        }
        ExprKind::If { cond, then_expr, else_expr } => IrExpr::If {
            cond: Box::new(resolve_expr(cond, scope, symbols)?),
            then_expr: Box::new(resolve_expr(then_expr, scope, symbols)?),
            else_expr: match else_expr {
                Some(x) => Some(Box::new(resolve_expr(x, scope, symbols)?)),
                None => None,
            },
        },
        ExprKind::PerUnit { expr, delta } => IrExpr::PerUnit {
            expr: Box::new(resolve_expr(expr, scope, symbols)?),
            delta: Box::new(resolve_expr(delta, scope, symbols)?),
        },
        // ----------------------------------------------------------------
        // Plan ToM Task 8 — belief read expressions
        // ----------------------------------------------------------------
        ExprKind::BeliefsAccessor { observer, target, field } => {
            // Validate the field name against the known BeliefState fields.
            const BELIEF_FIELDS: &[&str] = &[
                "last_known_pos",
                "last_known_hp",
                "last_known_max_hp",
                "last_known_creature_type",
                "last_updated_tick",
                "confidence",
            ];
            if !BELIEF_FIELDS.contains(&field.as_str()) {
                return Err(ResolveError::UnknownBeliefField {
                    field: field.clone(),
                    valid: BELIEF_FIELDS.to_vec(),
                    span,
                });
            }
            IrExpr::BeliefsAccessor {
                observer: Box::new(resolve_expr(observer, scope, symbols)?),
                target: Box::new(resolve_expr(target, scope, symbols)?),
                field: field.clone(),
            }
        }
        ExprKind::BeliefsConfidence { observer, target } => IrExpr::BeliefsConfidence {
            observer: Box::new(resolve_expr(observer, scope, symbols)?),
            target: Box::new(resolve_expr(target, scope, symbols)?),
        },
        ExprKind::BeliefsView { observer, view_name } => IrExpr::BeliefsView {
            observer: Box::new(resolve_expr(observer, scope, symbols)?),
            view_name: view_name.clone(),
        },
        ExprKind::Block { bindings, expr } => {
            // Resolve each binding into the local scope so the final
            // expression sees the bound names. Bind values are
            // resolved-and-discarded today — the LocalRefs in the
            // final expression have no actual storage. Used by the
            // parse-time `let` prelude in @lazy view bodies; those
            // views aren't yet wired into the lowering pipeline,
            // so the dangling Locals aren't reached.
            for (name, value) in bindings {
                let _ = resolve_expr(value, scope, symbols)?;
                scope.bind(name, IrType::Unknown);
            }
            return resolve_expr(expr, scope, symbols);
        }
    };
    Ok(IrExprNode { kind, span })
}

fn resolve_ident(
    name: &str,
    span: Span,
    scope: &LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExpr, ResolveError> {
    // Bare `true` / `false` are literals. The parser emits them as Ident.
    if name == "true" {
        return Ok(IrExpr::LitBool(true));
    }
    if name == "false" {
        return Ok(IrExpr::LitBool(false));
    }
    if let Some(b) = scope.lookup(name) {
        return Ok(IrExpr::Local(b.local, b.name.clone()));
    }
    if name == "self" {
        // `self` is only valid inside a decl that binds it — if the scope
        // didn't see it, this is SelfInTopLevel.
        return Err(ResolveError::SelfInTopLevel { span });
    }
    if name == "_" {
        // Wildcard placeholder for view-call argument slots. Used in
        // scoring predicates on self-only rows (no target binding) to
        // mean "sum over all values for this slot": e.g.
        // `view::threat_level(self, _)` = Σ threat(self, x). The
        // scoring emitter recognises this sentinel (Local with name
        // `_`) as arg_slot = 0xFE (sum-wildcard). Outside a scoring
        // view-call it is an error — but 1a leaves that diagnostic to
        // the scoring lowering to keep the match local.
        return Ok(IrExpr::Local(crate::ir::LocalRef(u16::MAX - 1), "_".to_string()));
    }
    if let Some(r) = symbols.entities.get(name) {
        return Ok(IrExpr::Entity(*r));
    }
    if let Some(r) = symbols.events.get(name) {
        return Ok(IrExpr::Event(*r));
    }
    if let Some(r) = symbols.views.get(name) {
        return Ok(IrExpr::View(*r));
    }
    if let Some(r) = symbols.verbs.get(name) {
        return Ok(IrExpr::Verb(*r));
    }
    if let Some(ns) = symbols.stdlib_namespaces.get(name) {
        return Ok(IrExpr::Namespace(*ns));
    }
    if let Some(t) = symbols.stdlib_types.get(name) {
        // The identifier referred to a type name used as a value. In 1a we
        // don't have a dedicated "type-as-value" node; fall through and keep
        // it as an unresolved enum variant marker (closest analogue: "ALL_CAPS
        // CONSTANT", "Stone", etc — also handled below).
        let _ = t;
    }
    // GPU ability evaluation Phase 2 primitives that parse as a
    // flattened namespaced identifier WITHOUT a `(...)` suffix.
    // `ability::tag(...)` and `ability::on_cooldown(...)` take args
    // and route through `resolve_ability_eval_call`; the naked forms
    // (`ability::hint`, `ability::range`) are handled here.
    // See `docs/spec/engine.md §11`
    // §Architecture.
    if name == "ability::hint" || name == "abilities::hint" {
        return Ok(IrExpr::AbilityHint);
    }
    if name == "ability::range" || name == "abilities::range" {
        return Ok(IrExpr::AbilityRange);
    }
    // `EnumName::Variant` — recognise the two-segment form and validate
    // against the declared enum.
    if let Some((lhs, rhs)) = name.split_once("::") {
        if let Some((_, variants)) = symbols.enums.get(lhs) {
            if variants.iter().any(|v| v == rhs) {
                return Ok(IrExpr::EnumVariant {
                    ty: lhs.to_string(),
                    variant: rhs.to_string(),
                });
            }
            return Err(ResolveError::UnknownIdent {
                name: name.to_string(),
                span,
                suggestions: variants.iter().take(3).cloned().collect(),
            });
        }
    }
    // Identifiers that start uppercase are likely enum variants or constants
    // (Conquest, Family, Religion, Stone, FleeSet, AGGRO_RANGE, ...). Check
    // user-declared enums first so `CulturalTransgression` resolves to its
    // owning enum's variant; everything else stays typeless and waits for
    // 1b type inference.
    if starts_upper(name) {
        let ty = symbols
            .enum_variant_owner
            .get(name)
            .cloned()
            .unwrap_or_default();
        return Ok(IrExpr::EnumVariant { ty, variant: name.to_string() });
    }
    // Otherwise: bare lowercase ident with no match. This is an unknown
    // identifier — error out with suggestions.
    let suggestions = suggest_idents(name, scope, symbols);
    Err(ResolveError::UnknownIdent { name: name.to_string(), span, suggestions })
}

fn resolve_call(
    callee: &Expr,
    args: &[ast::CallArg],
    span: Span,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExpr, ResolveError> {
    // GPU ability evaluation Phase 2 primitives. Intercepted here
    // before the generic Ident-call fallthrough so the arguments
    // don't get pre-resolved (e.g. `PHYSICAL` would otherwise land
    // as a stray `EnumVariant` and the tag name would be buried
    // inside an opaque arg list).
    //
    // The parser flattens `ability::tag(PHYSICAL)` into
    // `Call(Ident("ability::tag"), [Ident("PHYSICAL")])`; the name
    // is a single string with `::` preserved.
    if let ExprKind::Ident(name) = &callee.kind {
        if let Some(expr) = resolve_ability_eval_call(name, args, span, scope, symbols)? {
            return Ok(expr);
        }
    }
    // `<namespace>.<method>(...)` — resolved against the stdlib method
    // schema. An unknown method on a known namespace stays structured
    // (ns+method kept), with `Unknown` return type; 1b flags it.
    if let ExprKind::Field(base, method) = &callee.kind {
        if let ExprKind::Ident(ns_name) = &base.kind {
            if scope.lookup(ns_name).is_none() {
                if let Some(ns) = symbols.stdlib_namespaces.get(ns_name) {
                    let ir_args = args
                        .iter()
                        .map(|a| resolve_call_arg(a, scope, symbols))
                        .collect::<Result<Vec<_>, _>>()?;
                    // `view::<name>(...)` — rewrite to ViewCall when the
                    // method resolves against the declared views. Unknown
                    // method names stay NamespaceCall so 1b diagnostics can
                    // surface them.
                    if *ns == NamespaceId::View {
                        if let Some(view_ref) = symbols.views.get(method) {
                            return Ok(IrExpr::ViewCall(*view_ref, ir_args));
                        }
                    }
                    // Plan G G3f — `threats.<method>(...)` dispatches
                    // to `Builtin::Threats*` so the lowering sees a
                    // closed-set enum to dispatch on. Unknown methods
                    // on the threats namespace fall through to the
                    // generic NamespaceCall route below — 1b surfaces
                    // the typo.
                    if *ns == NamespaceId::Threats {
                        if let Some(b) = threats_method_builtin(method) {
                            return Ok(IrExpr::BuiltinCall(b, ir_args));
                        }
                    }
                    // `spatial.<name>(...)` — Phase 7 Task 4. Reject the
                    // call eagerly when no `spatial_query <name>`
                    // declaration backs it; lowering (Task 5) needs a
                    // resolved declaration, so silent carry-through
                    // would just defer a defect.
                    if *ns == NamespaceId::Spatial {
                        if !symbols.spatial_queries.contains_key(method) {
                            return Err(ResolveError::UnknownSpatialQuery {
                                name: method.clone(),
                                span,
                            });
                        }
                        return Ok(IrExpr::NamespaceCall {
                            ns: NamespaceId::Spatial,
                            method: method.clone(),
                            args: ir_args,
                        });
                    }
                    // Arity is informational here; 1a doesn't surface it as
                    // an error. 1b will compare `ir_args.len()` against
                    // `stdlib::method_sig(ns, method).0`.
                    let _ = stdlib::method_sig(*ns, method);
                    return Ok(IrExpr::NamespaceCall {
                        ns: *ns,
                        method: method.clone(),
                        args: ir_args,
                    });
                }
            }
        }
    }
    // Only resolve callees that are a bare Ident. Anything else (method
    // chain, field-call) falls through as UnresolvedCall or Raw.
    if let ExprKind::Ident(name) = &callee.kind {
        let ir_args = args
            .iter()
            .map(|a| resolve_call_arg(a, scope, symbols))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(b) = symbols.builtins.get(name) {
            return Ok(IrExpr::BuiltinCall(*b, ir_args));
        }
        // `view::<name>(...)` — the parser flattens `ns::method` into a
        // single ident with `::` preserved. Recognise the `view::` prefix
        // and route through ViewCall the same way the dotted-field path
        // does. Keeps the two syntactic forms interchangeable.
        if let Some(tail) = name.strip_prefix("view::") {
            if let Some(view_ref) = symbols.views.get(tail) {
                return Ok(IrExpr::ViewCall(*view_ref, ir_args));
            }
        }
        // Generic `<ns>::<method>(...)` routing — the parser-flattened
        // sibling of the `<ns>.<method>(...)` dotted path handled above.
        // If `<ns>` is a registered stdlib namespace, lift the call into
        // a structured `NamespaceCall` so the resolver's type inference
        // (and the emitter's per-namespace dispatch) treats the two
        // surface forms interchangeably. Exact mirror of the dotted
        // branch: arity is informational at 1a; 1b enforces it.
        if let Some((ns_name, method)) = name.split_once("::") {
            if scope.lookup(ns_name).is_none() {
                if let Some(ns) = symbols.stdlib_namespaces.get(ns_name) {
                    // `spatial::<name>(...)` — Phase 7 Task 4. Same
                    // eager-reject behaviour as the dotted form.
                    if *ns == NamespaceId::Spatial {
                        if !symbols.spatial_queries.contains_key(method) {
                            return Err(ResolveError::UnknownSpatialQuery {
                                name: method.to_string(),
                                span,
                            });
                        }
                        return Ok(IrExpr::NamespaceCall {
                            ns: NamespaceId::Spatial,
                            method: method.to_string(),
                            args: ir_args,
                        });
                    }
                    // Plan G G3f — `threats::<method>(...)` mirrors
                    // the dotted form: dispatch to `Builtin::Threats*`.
                    if *ns == NamespaceId::Threats {
                        if let Some(b) = threats_method_builtin(method) {
                            return Ok(IrExpr::BuiltinCall(b, ir_args));
                        }
                    }
                    let _ = stdlib::method_sig(*ns, method);
                    return Ok(IrExpr::NamespaceCall {
                        ns: *ns,
                        method: method.to_string(),
                        args: ir_args,
                    });
                }
            }
        }
        if let Some(r) = symbols.views.get(name) {
            return Ok(IrExpr::ViewCall(*r, ir_args));
        }
        if let Some(r) = symbols.verbs.get(name) {
            return Ok(IrExpr::VerbCall(*r, ir_args));
        }
        // Local or unresolved.
        if scope.lookup(name).is_some() {
            // Calling a local (unusual; treat as unresolved for 1b).
            return Ok(IrExpr::UnresolvedCall(name.clone(), ir_args));
        }
        return Ok(IrExpr::UnresolvedCall(name.clone(), ir_args));
    }
    // Complex callee: keep raw for 1b.
    let _ = span;
    Ok(IrExpr::Raw(Box::new(Expr {
        kind: ExprKind::Call(
            Box::new(callee.clone()),
            args.to_vec(),
        ),
        span,
    })))
}

fn resolve_call_arg(
    a: &ast::CallArg,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrCallArg, ResolveError> {
    Ok(IrCallArg {
        name: a.name.clone(),
        value: resolve_expr(&a.value, scope, symbols)?,
        span: a.span,
    })
}

/// Detect `ability::hint == <hint_ident>` (or the mirror `<hint_ident>
/// == ability::hint`) at the AST layer. When matched, the hint ident
/// side lowers to `IrExpr::AbilityHintLit(<hint>)` instead of
/// producing an UnknownIdent error (bare lowercase hint idents aren't
/// otherwise in scope).
///
/// Symmetric across `Eq` / `NotEq`; all other operators fall through.
/// Symmetric across sides so `damage == ability::hint` parses too.
///
/// Returns the two resolved operands. If the shape doesn't match, both
/// sides go through `resolve_expr` unchanged.
fn resolve_hint_compare_or_default(
    op: BinOp,
    lhs: &Expr,
    rhs: &Expr,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<(IrExprNode, IrExprNode), ResolveError> {
    // Only `==` / `!=` — the only ops where a hint literal is
    // meaningful. Every other op falls through.
    let is_equality = matches!(op, BinOp::Eq | BinOp::NotEq);
    if !is_equality {
        return Ok((
            resolve_expr(lhs, scope, symbols)?,
            resolve_expr(rhs, scope, symbols)?,
        ));
    }
    if let Some((hint_side, other_side, hint_on_left)) =
        match_hint_compare(lhs, rhs)
    {
        // Resolve the hint accessor the normal way (it lowers to
        // `IrExpr::AbilityHint` via the `resolve_ident` hook).
        let hint_node = resolve_expr(hint_side, scope, symbols)?;
        // Hint-literal span: whichever side wasn't the accessor.
        let lit_span = if hint_on_left { rhs.span } else { lhs.span };
        let lit_node = IrExprNode {
            kind: IrExpr::AbilityHintLit(other_side),
            span: lit_span,
        };
        // Restore original order so source-level lhs/rhs are preserved
        // on the resulting Binary.
        return Ok(if hint_on_left {
            (hint_node, lit_node)
        } else {
            (lit_node, hint_node)
        });
    }
    Ok((
        resolve_expr(lhs, scope, symbols)?,
        resolve_expr(rhs, scope, symbols)?,
    ))
}

/// If one of `(lhs, rhs)` is the AST form `Ident("ability::hint")` and
/// the other is a bare lowercase hint ident, return the hint side, the
/// parsed `AbilityHint`, and a flag telling which side the hint ident
/// was on (`true` == hint on lhs, `false` == hint on rhs — useful for
/// preserving source order).
fn match_hint_compare<'a>(
    lhs: &'a Expr,
    rhs: &'a Expr,
) -> Option<(&'a Expr, AbilityHint, bool)> {
    let lhs_is_hint = is_ability_hint_accessor(lhs);
    let rhs_is_hint = is_ability_hint_accessor(rhs);
    if lhs_is_hint && !rhs_is_hint {
        if let Some(hint) = as_hint_literal(rhs) {
            return Some((lhs, hint, true));
        }
    }
    if rhs_is_hint && !lhs_is_hint {
        if let Some(hint) = as_hint_literal(lhs) {
            return Some((rhs, hint, false));
        }
    }
    None
}

fn is_ability_hint_accessor(e: &Expr) -> bool {
    matches!(
        &e.kind,
        ExprKind::Ident(n) if n == "ability::hint" || n == "abilities::hint"
    )
}

fn as_hint_literal(e: &Expr) -> Option<AbilityHint> {
    if let ExprKind::Ident(n) = &e.kind {
        return AbilityHint::parse_ident(n);
    }
    None
}

/// GPU ability evaluation Phase 2 primitives that shape-match a
/// flattened-ident callee into a dedicated `IrExpr` variant.
///
/// Returns `Ok(Some(expr))` if the callee matches a known primitive
/// and lowers successfully; `Ok(None)` if the callee does not match
/// (caller falls through to the generic Ident-call path); `Err` if
/// the callee matches but its arguments are malformed (e.g. unknown
/// tag name).
///
/// Only argument-taking primitives route through here —
/// `ability::hint`, `ability::range` have no `()` suffix and are
/// handled in `resolve_ident` instead. See
/// `docs/spec/engine.md §11`
/// §Architecture.
fn resolve_ability_eval_call(
    name: &str,
    args: &[ast::CallArg],
    span: Span,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<Option<IrExpr>, ResolveError> {
    // `ability::tag(TAG_NAME)` — reads the current ability's tag
    // value. The argument must be a bare identifier from the
    // `AbilityTag` vocabulary (identifier-case; e.g. `PHYSICAL`).
    if name == "ability::tag" || name == "abilities::tag" {
        if args.len() != 1 {
            return Err(ResolveError::UnknownIdent {
                name: format!(
                    "{name} takes exactly one argument (a tag name), got {}",
                    args.len()
                ),
                span,
                suggestions: vec![],
            });
        }
        let arg = &args[0];
        if arg.name.is_some() {
            return Err(ResolveError::UnknownIdent {
                name: format!("{name}: tag argument must be positional (no `key:` form)"),
                span,
                suggestions: vec![],
            });
        }
        let tag_ident = match &arg.value.kind {
            ExprKind::Ident(s) => s.clone(),
            _ => {
                return Err(ResolveError::UnknownIdent {
                    name: format!(
                        "{name}: tag argument must be a bare identifier \
                         (e.g. `PHYSICAL`); got a non-identifier expression"
                    ),
                    span: arg.span,
                    suggestions: vec![],
                });
            }
        };
        return match AbilityTag::parse_ident(&tag_ident) {
            Some(tag) => Ok(Some(IrExpr::AbilityTag { tag })),
            None => Err(ResolveError::UnknownIdent {
                name: format!(
                    "unknown ability tag `{tag_ident}`; valid tags are \
                     PHYSICAL, MAGICAL, CROWD_CONTROL, HEAL, DEFENSE, UTILITY"
                ),
                span: arg.span,
                suggestions: vec![
                    "PHYSICAL".into(),
                    "CROWD_CONTROL".into(),
                    "DEFENSE".into(),
                ],
            }),
        };
    }
    // `ability::on_cooldown(<slot_expr>)` — returns a boolean telling
    // whether the given ability slot is still on cooldown for the
    // scoring agent. Takes one positional argument (typically the
    // implicit `ability` local inside a per_ability row; a literal
    // slot index also works).
    if name == "ability::on_cooldown" || name == "abilities::on_cooldown" {
        if args.len() != 1 {
            return Err(ResolveError::UnknownIdent {
                name: format!(
                    "{name} takes exactly one argument (a slot expression), got {}",
                    args.len()
                ),
                span,
                suggestions: vec![],
            });
        }
        let arg = &args[0];
        if arg.name.is_some() {
            return Err(ResolveError::UnknownIdent {
                name: format!(
                    "{name}: slot argument must be positional (no `key:` form)"
                ),
                span,
                suggestions: vec![],
            });
        }
        let slot_expr = resolve_expr(&arg.value, scope, symbols)?;
        return Ok(Some(IrExpr::AbilityOnCooldown(Box::new(slot_expr))));
    }
    Ok(None)
}

fn resolve_assert(
    a: &AssertExpr,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrAssertExpr, ResolveError> {
    match a {
        AssertExpr::Count { filter, op, value, span } => Ok(IrAssertExpr::Count {
            filter: resolve_expr(filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
        AssertExpr::Pr { action_filter, obs_filter, op, value, span } => Ok(IrAssertExpr::Pr {
            action_filter: resolve_expr(action_filter, scope, symbols)?,
            obs_filter: resolve_expr(obs_filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
        AssertExpr::Mean { scalar, filter, op, value, span } => Ok(IrAssertExpr::Mean {
            scalar: resolve_expr(scalar, scope, symbols)?,
            filter: resolve_expr(filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
        // Filtered out by the caller (probe-decl resolve loop) before
        // reaching here — defensive arm so the match stays exhaustive.
        AssertExpr::Raw { span, .. } => Ok(IrAssertExpr::Count {
            filter: IrExprNode { kind: IrExpr::LitBool(true), span: *span },
            op: ">=".to_string(),
            value: IrExprNode { kind: IrExpr::LitInt(0), span: *span },
            span: *span,
        }),
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn starts_upper(s: &str) -> bool {
    s.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
}

/// Convert a tag declaration's PascalCase name (`Harmful`) to its
/// annotation-form lowercase (`harmful`). Matches a lossless 1:1 map —
/// variants like `XyzName` become `xyzname` which won't collide in
/// practice since tags are author-chosen single-word PascalCase.
pub(crate) fn lowercase_tag_name(name: &str) -> String {
    name.to_ascii_lowercase()
}

fn suggest_idents(name: &str, scope: &LocalScope, symbols: &SymbolTable) -> Vec<String> {
    let mut pool: Vec<&str> = Vec::new();
    for frame in &scope.stack {
        for b in frame {
            pool.push(&b.name);
        }
    }
    for k in symbols.events.keys() {
        pool.push(k);
    }
    for k in symbols.entities.keys() {
        pool.push(k);
    }
    for k in symbols.views.keys() {
        pool.push(k);
    }
    for k in symbols.verbs.keys() {
        pool.push(k);
    }
    for k in symbols.builtins.keys() {
        pool.push(k);
    }
    let mut ranked: Vec<(usize, String)> = pool
        .iter()
        .map(|s| (edit_distance(name, s), s.to_string()))
        .collect();
    ranked.sort_by_key(|p| p.0);
    ranked.retain(|p| p.0 <= 3);
    ranked.into_iter().map(|p| p.1).take(3).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// @decay annotation lowering (spec §2.3, §9 D31)
// ---------------------------------------------------------------------------

/// Walk the view's annotations; if a `@decay(...)` annotation exists,
/// validate it and return a typed `DecayHint`. Validates:
///
/// - Paired with `@materialized` (errors otherwise — v1 only supports
///   anchor-pattern decay on event-folded views).
/// - Host body is a `Fold` (lazy views have no persistent state to decay).
///
/// **Two equivalent shapes are accepted:**
///
/// 1. **Legacy multiplicative** — `@decay(rate = R, per = tick)`. `R`
///    is a float literal in `[0.0, 1.0)`. Implies `mode = mul`. The
///    `0.0` value is the "full reset every tick" idiom — the per-tick
///    decay multiplier zeroes the previous storage before the fold's
///    event handlers add the current tick's contributions. `1.0` is
///    rejected (no-op).
///
/// 2. **Explicit mode + magnitude** — `@decay(per = tick, mode = <mul|sub>, by = N)`:
///    - `mode = mul, by = R` is identical to the legacy `rate = R` form.
///    - `mode = sub, by = N` (positive integer) emits a saturating
///      per-cell subtract: `cell = saturating_sub(old, N)`. Targets
///      integer-valued storage (u8/u16/u32). N must be > 0.
///
/// **Optional `gate = MaskName`** (sub mode only today): names a mask
/// declared elsewhere in the .sim. Cells where the mask predicate
/// evaluates FALSE are skipped by the per-cell decay step. The name is
/// resolved against the global symbol table here; downstream emit may
/// surface a "gate mask predicate references view-storage cells —
/// cross-binding plumbing not yet implemented in decay kernels"
/// diagnostic when it cannot inline the predicate's WGSL body. In that
/// case the kernel emits a TODO marker rather than silently skipping
/// the gate (caller must address the architectural gap before tom_probe-
/// style decay subsumption can land).
///
/// The two forms are mutually exclusive: callers either spell `rate`
/// (legacy) or `mode + by` (explicit). Mixing is rejected.
fn lower_decay_hint(
    annotations: &[ast::Annotation],
    body: &ast::ViewBody,
    symbols: &SymbolTable,
) -> Result<Option<DecayHint>, ResolveError> {
    lower_decay_hint_for(annotations, body, symbols, /*is_belief=*/ false)
}

fn lower_decay_hint_for(
    annotations: &[ast::Annotation],
    body: &ast::ViewBody,
    symbols: &SymbolTable,
    is_belief: bool,
) -> Result<Option<DecayHint>, ResolveError> {
    let ann = match annotations.iter().find(|a| a.name == "decay") {
        Some(a) => a,
        None => return Ok(None),
    };

    // Must coexist with `@materialized` — except on `belief` decls,
    // where the materialized-ness is implicit in the keyword.
    let has_materialized = annotations.iter().any(|a| a.name == "materialized");
    if !has_materialized && !is_belief {
        return Err(ResolveError::InvalidDecayHint {
            detail:
                "`@decay` requires a sibling `@materialized(...)` annotation on the same view"
                    .into(),
            span: ann.span,
        });
    }

    // Must be a fold body.
    if !matches!(body, ast::ViewBody::Fold { .. }) {
        return Err(ResolveError::InvalidDecayHint {
            detail: "`@decay` only applies to `@materialized` fold views (the anchor pattern needs a base value + event handlers)".into(),
            span: ann.span,
        });
    }

    let mut rate: Option<f64> = None;
    let mut per: Option<String> = None;
    let mut mode: Option<String> = None;
    let mut by_int: Option<i64> = None;
    let mut by_float: Option<f64> = None;
    let mut gate_name: Option<(String, ast::Span)> = None;
    for arg in &ann.args {
        let key = match &arg.key {
            Some(k) => k.as_str(),
            None => {
                return Err(ResolveError::InvalidDecayHint {
                    detail:
                        "`@decay` arguments must be `key = value` (got a positional arg)".into(),
                    span: arg.span,
                });
            }
        };
        match key {
            "rate" => {
                let r = match &arg.value {
                    ast::AnnotationValue::Float(f) => *f,
                    ast::AnnotationValue::Int(i) => *i as f64,
                    other => {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!("`rate` must be a float literal; got {other:?}"),
                            span: arg.span,
                        });
                    }
                };
                rate = Some(r);
            }
            "per" => {
                let p = match &arg.value {
                    ast::AnnotationValue::Ident(s) => s.clone(),
                    other => {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!(
                                "`per` must be an identifier (e.g. `tick`); got {other:?}"
                            ),
                            span: arg.span,
                        });
                    }
                };
                per = Some(p);
            }
            "mode" => {
                let m = match &arg.value {
                    ast::AnnotationValue::Ident(s) => s.clone(),
                    other => {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!(
                                "`mode` must be an identifier (`mul` or `sub`); got {other:?}"
                            ),
                            span: arg.span,
                        });
                    }
                };
                mode = Some(m);
            }
            "by" => match &arg.value {
                ast::AnnotationValue::Int(i) => by_int = Some(*i),
                ast::AnnotationValue::Float(f) => by_float = Some(*f),
                other => {
                    return Err(ResolveError::InvalidDecayHint {
                        detail: format!(
                            "`by` must be a numeric literal; got {other:?}"
                        ),
                        span: arg.span,
                    });
                }
            },
            "gate" => {
                let n = match &arg.value {
                    ast::AnnotationValue::Ident(s) => s.clone(),
                    other => {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!(
                                "`gate` must be a mask name identifier; got {other:?}"
                            ),
                            span: arg.span,
                        });
                    }
                };
                gate_name = Some((n, arg.span));
            }
            other => {
                return Err(ResolveError::InvalidDecayHint {
                    detail: format!(
                        "unknown `@decay` argument `{other}`; expected `rate`, `per`, `mode`, `by`, or `gate`"
                    ),
                    span: arg.span,
                });
            }
        }
    }

    let per = per.ok_or_else(|| ResolveError::InvalidDecayHint {
        detail: "missing required argument `per`".into(),
        span: ann.span,
    })?;
    let per_unit = match per.as_str() {
        "tick" => DecayUnit::Tick,
        other => {
            return Err(ResolveError::InvalidDecayHint {
                detail: format!(
                    "unsupported `per` unit `{other}`; only `tick` is supported in v1"
                ),
                span: ann.span,
            });
        }
    };

    // Decide which form was used:
    //   * `rate = R` → legacy `mul` mode, magnitude carried by `rate`.
    //   * `mode = X, by = Y` → explicit, `rate` rejected as a
    //     coexisting key.
    // Mixing both forms is a hard error to keep the surface predictable.
    let (resolved_mode, resolved_rate, resolved_by): (DecayMode, f32, u32) = match (rate, mode) {
        (Some(_), Some(_)) => {
            return Err(ResolveError::InvalidDecayHint {
                detail: "`@decay` accepts either `rate = R` (legacy) OR \
                    `mode = ..., by = ...` (explicit), not both".into(),
                span: ann.span,
            });
        }
        (Some(r), None) => {
            // Legacy multiplicative form: `rate = R`.
            if by_int.is_some() || by_float.is_some() {
                return Err(ResolveError::InvalidDecayHint {
                    detail: "`@decay(rate = ...)` does not accept `by`; \
                        the rate IS the multiplier. Use `mode = mul, by = R` if you \
                        want the explicit spelling.".into(),
                    span: ann.span,
                });
            }
            // `rate = 0.0` is the "full reset every tick" idiom; `rate
            // = 1.0` stays rejected because it makes the decay kernel a
            // no-op; callers should drop the annotation instead.
            if !(r >= 0.0 && r < 1.0) || !r.is_finite() {
                return Err(ResolveError::InvalidDecayHint {
                    detail: format!(
                        "`rate` must be a finite float in the half-open interval [0.0, 1.0); got {r}"
                    ),
                    span: ann.span,
                });
            }
            (DecayMode::Mul, r as f32, 0)
        }
        (None, Some(m)) => {
            // Explicit `mode + by` form.
            match m.as_str() {
                "mul" => {
                    let mag = match (by_int, by_float) {
                        (None, Some(f)) => f,
                        (Some(i), None) => i as f64,
                        (None, None) => {
                            return Err(ResolveError::InvalidDecayHint {
                                detail: "`mode = mul` requires `by = <float in [0.0, 1.0)>`".into(),
                                span: ann.span,
                            });
                        }
                        (Some(_), Some(_)) => unreachable!("by is parsed once"),
                    };
                    if !(mag >= 0.0 && mag < 1.0) || !mag.is_finite() {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!(
                                "`mode = mul` requires `by` in the half-open interval [0.0, 1.0); got {mag}"
                            ),
                            span: ann.span,
                        });
                    }
                    (DecayMode::Mul, mag as f32, 0)
                }
                "sub" => {
                    let n = match (by_int, by_float) {
                        (Some(i), None) => i,
                        (None, Some(_)) => {
                            return Err(ResolveError::InvalidDecayHint {
                                detail: "`mode = sub` requires `by = <positive int>`; \
                                    floats are not allowed (saturating-sub targets integer storage)".into(),
                                span: ann.span,
                            });
                        }
                        (None, None) => {
                            return Err(ResolveError::InvalidDecayHint {
                                detail: "`mode = sub` requires `by = <positive int>`".into(),
                                span: ann.span,
                            });
                        }
                        (Some(_), Some(_)) => unreachable!("by is parsed once"),
                    };
                    if n <= 0 || n > u32::MAX as i64 {
                        return Err(ResolveError::InvalidDecayHint {
                            detail: format!(
                                "`mode = sub` requires `by > 0` (and ≤ u32::MAX); got {n}"
                            ),
                            span: ann.span,
                        });
                    }
                    (DecayMode::Sub, 0.0, n as u32)
                }
                other => {
                    return Err(ResolveError::InvalidDecayHint {
                        detail: format!(
                            "unknown `mode` value `{other}`; expected `mul` or `sub`"
                        ),
                        span: ann.span,
                    });
                }
            }
        }
        (None, None) => {
            return Err(ResolveError::InvalidDecayHint {
                detail: "`@decay` requires either `rate = R` (legacy) OR \
                    `mode = ..., by = ...` (explicit)".into(),
                span: ann.span,
            });
        }
    };

    // Resolve the optional `gate = MaskName` against the symbol table.
    // Surfaces an unknown-mask error if the name doesn't match a
    // declared mask (ResolveError::UnknownSymbol-style; we use the
    // InvalidDecayHint variant to keep the locus tied to the
    // annotation's own diagnostics path).
    let gate = match gate_name {
        Some((name, span)) => match symbols.masks.get(&name) {
            Some(mref) => Some(*mref),
            None => {
                return Err(ResolveError::InvalidDecayHint {
                    detail: format!(
                        "`gate = {name}` does not match any declared mask; \
                         declare `mask {name}(...) when <pred>` first"
                    ),
                    span,
                });
            }
        },
        None => None,
    };

    Ok(Some(DecayHint {
        rate: resolved_rate,
        per: per_unit,
        mode: resolved_mode,
        sub_by: resolved_by,
        gate,
        span: ann.span,
    }))
}

// ---------------------------------------------------------------------------
// @storage(<packing>) annotation lowering
// ---------------------------------------------------------------------------

/// Walk the view's annotations; if a `@storage(<name>)` annotation
/// exists, validate it and return the typed [`Packing`] discriminator.
/// Returns `Packing::None` when the annotation is absent.
///
/// **Surface today:** single positional ident arg — `@storage(packed_q8)`.
/// Future packings (`packed_q4`, `packed_q16`, …) extend this match list.
/// The annotation does not require sibling `@materialized` / `@decay`;
/// it independently controls per-cell storage layout for the view's
/// primary buffer. The decay + fold WGSL emit consult the IR field to
/// branch between the legacy "one logical cell per u32" path and the
/// q8-packed "4 cells per u32 word" path.
fn lower_storage_annotation(
    annotations: &[ast::Annotation],
) -> Result<Packing, ResolveError> {
    let ann = match annotations.iter().find(|a| a.name == "storage") {
        Some(a) => a,
        None => return Ok(Packing::None),
    };
    if ann.args.len() != 1 {
        return Err(ResolveError::InvalidStorageAnnotation {
            detail: format!(
                "`@storage(<name>)` takes exactly one positional argument; got {}",
                ann.args.len()
            ),
            span: ann.span,
        });
    }
    let arg = &ann.args[0];
    if arg.key.is_some() {
        return Err(ResolveError::InvalidStorageAnnotation {
            detail: "`@storage(<name>)` argument must be positional (no `key = value`)".into(),
            span: arg.span,
        });
    }
    let name = match &arg.value {
        ast::AnnotationValue::Ident(s) => s.as_str(),
        other => {
            return Err(ResolveError::InvalidStorageAnnotation {
                detail: format!(
                    "`@storage(<name>)` argument must be an identifier; got {other:?}"
                ),
                span: arg.span,
            });
        }
    };
    match name {
        "packed_q8" => Ok(Packing::Q8),
        other => Err(ResolveError::InvalidStorageAnnotation {
            detail: format!(
                "unknown packing `{other}`; supported packings: `packed_q8`"
            ),
            span: arg.span,
        }),
    }
}

// ---------------------------------------------------------------------------
// @lazy / @materialized annotation lowering (spec §2.3, §9 D31)
// ---------------------------------------------------------------------------

/// Resolve `@lazy` and `@materialized(...)` annotations on a view declaration
/// into a typed `ViewKind`. Spec §2.3 + §9 D31.
///
/// - `@lazy` (or no annotation) → `ViewKind::Lazy`.
/// - `@materialized(storage = <hint>)` → `ViewKind::Materialized(<storage>)`.
/// - `@materialized` with no `storage` defaults to `pair_map` (spec §9 D31).
/// - `@lazy` and `@materialized` are mutually exclusive; both on the same view
///   is a hard error.
/// - `@materialized` requires a `Fold` body (the event-fold path needs event
///   handlers).
///
/// Supported storage hints (spec §9 D31):
/// - `pair_map` — dense `HashMap<(K1, K2), V>`.
/// - `per_entity_topk(K, keyed_on = <param>)` — bounded per-entity slots.
/// - `lazy_cached` — compute-on-demand + per-tick cache.
fn lower_view_kind(
    view_name: &str,
    annotations: &[ast::Annotation],
    body: &ast::ViewBody,
    view_span: Span,
) -> Result<ViewKind, ResolveError> {
    let lazy_ann = annotations.iter().find(|a| a.name == "lazy");
    let mat_ann = annotations.iter().find(|a| a.name == "materialized");

    // Mutual-exclusion check.
    if let (Some(la), Some(ma)) = (lazy_ann, mat_ann) {
        let span = if la.span.start < ma.span.start { ma.span } else { la.span };
        return Err(ResolveError::InvalidViewKind {
            view_name: view_name.to_string(),
            detail: "`@lazy` and `@materialized` are mutually exclusive on the same view".into(),
            span,
        });
    }

    // Default: no annotation → lazy.
    if lazy_ann.is_some() || mat_ann.is_none() {
        // `@lazy` on a fold body is nonsensical — fold handlers only fire
        // for materialized views. Flag the mismatch.
        if matches!(body, ast::ViewBody::Fold { .. }) {
            let span = lazy_ann.map(|a| a.span).unwrap_or(view_span);
            return Err(ResolveError::InvalidViewKind {
                view_name: view_name.to_string(),
                detail:
                    "`@lazy` views must have an expression body; got a fold body (only `@materialized` views fold events)"
                        .into(),
                span,
            });
        }
        return Ok(ViewKind::Lazy);
    }

    // `@materialized(...)` — requires a fold body.
    let ma = mat_ann.unwrap();
    if !matches!(body, ast::ViewBody::Fold { .. }) {
        return Err(ResolveError::InvalidViewKind {
            view_name: view_name.to_string(),
            detail:
                "`@materialized` views must have a fold body (`initial:` / `on <Event> { ... }` / `clamp:`)"
                    .into(),
            span: ma.span,
        });
    }

    // Parse the annotation arguments. Known keys: `on_event`, `storage`.
    // Unknown keys error out so typos are caught at resolve time.
    let mut storage: Option<StorageHint> = None;
    let mut storage_span = ma.span;
    let mut saw_on_event = false;
    for arg in &ma.args {
        let key = match &arg.key {
            Some(k) => k.as_str(),
            None => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: view_name.to_string(),
                    detail:
                        "`@materialized(...)` arguments must be `key = value` (got a positional arg)"
                            .into(),
                    span: arg.span,
                });
            }
        };
        match key {
            "on_event" => {
                // Validate shape: a list of Idents. Contents are cross-
                // checked against declared events elsewhere; here we just
                // require the list form so typos surface early.
                match &arg.value {
                    ast::AnnotationValue::List(items) => {
                        for it in items {
                            if !matches!(it, ast::AnnotationValue::Ident(_)) {
                                return Err(ResolveError::InvalidViewKind {
                                    view_name: view_name.to_string(),
                                    detail: format!(
                                        "`on_event` list entries must be event identifiers; got {it:?}"
                                    ),
                                    span: arg.span,
                                });
                            }
                        }
                    }
                    other => {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: view_name.to_string(),
                            detail: format!(
                                "`on_event` must be a list of event identifiers (e.g. `[AgentAttacked, EffectDamageApplied]`); got {other:?}"
                            ),
                            span: arg.span,
                        });
                    }
                }
                saw_on_event = true;
            }
            "storage" => {
                storage = Some(parse_storage_hint(view_name, arg)?);
                storage_span = arg.span;
            }
            other => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: view_name.to_string(),
                    detail: format!(
                        "unknown `@materialized` argument `{other}`; expected `on_event` or `storage`"
                    ),
                    span: arg.span,
                });
            }
        }
    }
    let _ = saw_on_event; // presence is advisory — handlers are in the body

    // Sibling view-shape annotations — `@symmetric_pair_topk(K = N)`
    // and `@per_entity_ring(K = N)`. Each supplies the storage hint
    // directly; conflicting with an explicit `storage = ...` inside
    // `@materialized(...)` is a hard error. GPU cold-state replay
    // plan (2026-04-22) tasks 1.3 + 1.4.
    let sym_ann = annotations.iter().find(|a| a.name == "symmetric_pair_topk");
    let ring_ann = annotations.iter().find(|a| a.name == "per_entity_ring");
    if let (Some(a), Some(b)) = (sym_ann, ring_ann) {
        let span = if a.span.start < b.span.start { b.span } else { a.span };
        return Err(ResolveError::InvalidViewKind {
            view_name: view_name.to_string(),
            detail:
                "`@symmetric_pair_topk` and `@per_entity_ring` are mutually exclusive view-shape annotations"
                    .into(),
            span,
        });
    }
    if let Some(ann) = sym_ann.or(ring_ann) {
        if storage.is_some() {
            return Err(ResolveError::InvalidViewKind {
                view_name: view_name.to_string(),
                detail: format!(
                    "`@{}` conflicts with an explicit `@materialized(storage = ...)` hint; drop one",
                    ann.name
                ),
                span: ann.span,
            });
        }
        let k = annotation_k_arg(ann)?;
        let hint = if ann.name == "symmetric_pair_topk" {
            StorageHint::SymmetricPairTopK { k }
        } else {
            StorageHint::PerEntityRing { k }
        };
        return Ok(ViewKind::Materialized(hint));
    }

    // Default storage hint is `pair_map` per spec §9 D31.
    let storage = storage.unwrap_or(StorageHint::PairMap);
    let _ = storage_span;
    Ok(ViewKind::Materialized(storage))
}

/// Extract the `K = <positive int>` argument from a view-shape annotation
/// like `@symmetric_pair_topk(K = 8)` or `@per_entity_ring(K = 64)`.
/// Returns the K value clamped into `u16` (storage layer uses small K —
/// typical values are 8..=64). Errors if `K` is missing, non-int, out of
/// range, or if unknown sibling keys appear.
fn annotation_k_arg(ann: &ast::Annotation) -> Result<u16, ResolveError> {
    let mut k: Option<u16> = None;
    for arg in &ann.args {
        let key = match &arg.key {
            Some(k) => k.as_str(),
            None => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: ann.name.clone(),
                    detail: format!(
                        "`@{}(...)` requires `key = value` args (e.g. `K = 8`); got a positional arg",
                        ann.name
                    ),
                    span: arg.span,
                });
            }
        };
        match key {
            "K" => {
                let n = match &arg.value {
                    ast::AnnotationValue::Int(n) => *n,
                    other => {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: ann.name.clone(),
                            detail: format!(
                                "`K` must be a positive integer literal; got {other:?}"
                            ),
                            span: arg.span,
                        });
                    }
                };
                if n <= 0 || n > u16::MAX as i64 {
                    return Err(ResolveError::InvalidViewKind {
                        view_name: ann.name.clone(),
                        detail: format!(
                            "`K = {n}` out of range; must satisfy 1 <= K <= {}",
                            u16::MAX
                        ),
                        span: arg.span,
                    });
                }
                k = Some(n as u16);
            }
            other => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: ann.name.clone(),
                    detail: format!(
                        "unknown `@{}` argument `{other}`; expected `K`",
                        ann.name
                    ),
                    span: arg.span,
                });
            }
        }
    }
    k.ok_or_else(|| ResolveError::InvalidViewKind {
        view_name: ann.name.clone(),
        detail: format!("`@{}` requires a `K = <n>` argument", ann.name),
        span: ann.span,
    })
}

/// Parse a `storage = <hint>` annotation argument into a `StorageHint`.
/// Accepts:
/// - `pair_map`
/// - `lazy_cached`
/// - `per_entity_topk` (bare) — defaults `K=1, keyed_on=0` (task 139).
/// - `per_entity_topk(K = N)` — task 196. The call form carries the K
///   slot count as a `key = value` argument. `keyed_on` defaults to 0
///   (the view's first parameter) — authors drop in the named form once
///   we support views keyed on the second parameter.
fn parse_storage_hint(
    view_name: &str,
    arg: &ast::AnnotationArg,
) -> Result<StorageHint, ResolveError> {
    match &arg.value {
        ast::AnnotationValue::Ident(name) => match name.as_str() {
            "pair_map" => Ok(StorageHint::PairMap),
            "lazy_cached" => Ok(StorageHint::LazyCached),
            "per_entity_topk" => Ok(StorageHint::PerEntityTopK { k: 1, keyed_on: 0 }),
            other => Err(ResolveError::InvalidViewKind {
                view_name: view_name.to_string(),
                detail: format!(
                    "unsupported `storage` hint `{other}`; expected `pair_map`, `per_entity_topk`, or `lazy_cached`"
                ),
                span: arg.span,
            }),
        },
        ast::AnnotationValue::Call { name, args } => match name.as_str() {
            "per_entity_topk" => parse_per_entity_topk_call(view_name, args, arg.span),
            other => Err(ResolveError::InvalidViewKind {
                view_name: view_name.to_string(),
                detail: format!(
                    "`storage = {other}(...)` is not a known parameterised hint; \
                     only `per_entity_topk(K = N)` accepts call-form arguments"
                ),
                span: arg.span,
            }),
        },
        other => Err(ResolveError::InvalidViewKind {
            view_name: view_name.to_string(),
            detail: format!(
                "`storage` must be an identifier (e.g. `pair_map`); got {other:?}"
            ),
            span: arg.span,
        }),
    }
}

/// Resolve `per_entity_topk(K = N, ...)` call-form args. Only `K` is
/// recognised today; any other key errors so typos don't silently slip
/// through. `K` must be a positive i64 that fits in u16 (we store it
/// as `u16` in `StorageHint::PerEntityTopK` because the runtime uses
/// small K — typical values are 1..=16).
fn parse_per_entity_topk_call(
    view_name: &str,
    args: &[ast::AnnotationArg],
    _call_span: Span,
) -> Result<StorageHint, ResolveError> {
    let mut k: Option<u16> = None;
    for inner in args {
        let key = match &inner.key {
            Some(k) => k.as_str(),
            None => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: view_name.to_string(),
                    detail:
                        "`per_entity_topk(...)` requires `key = value` args (e.g. `K = 8`); got a positional arg"
                            .into(),
                    span: inner.span,
                });
            }
        };
        match key {
            "K" => {
                let n = match &inner.value {
                    ast::AnnotationValue::Int(n) => *n,
                    other => {
                        return Err(ResolveError::InvalidViewKind {
                            view_name: view_name.to_string(),
                            detail: format!(
                                "`K` must be a positive integer literal; got {other:?}"
                            ),
                            span: inner.span,
                        });
                    }
                };
                if n <= 0 || n > u16::MAX as i64 {
                    return Err(ResolveError::InvalidViewKind {
                        view_name: view_name.to_string(),
                        detail: format!(
                            "`K = {n}` out of range; must satisfy 1 <= K <= {}",
                            u16::MAX
                        ),
                        span: inner.span,
                    });
                }
                k = Some(n as u16);
            }
            other => {
                return Err(ResolveError::InvalidViewKind {
                    view_name: view_name.to_string(),
                    detail: format!(
                        "unknown `per_entity_topk` argument `{other}`; expected `K`"
                    ),
                    span: inner.span,
                });
            }
        }
    }
    Ok(StorageHint::PerEntityTopK {
        k: k.unwrap_or(1),
        keyed_on: 0,
    })
}

// ---------------------------------------------------------------------------
// View fold-body operator-set validator (spec §2.3)
// ---------------------------------------------------------------------------
//
// Fold bodies are restricted to the closed operator set documented in
// spec §2.3 so the event-fold path compiles to commutative, GPU-friendly
// updates. User-defined helper calls, recursion, unbounded loops, and
// cross-view composition are rejected here. Stdlib 1-hop accessors and
// built-in math are allowed.

fn validate_fold_body(view_name: &str, body: &[IrStmt]) -> Result<(), ResolveError> {
    for s in body {
        validate_fold_stmt(view_name, s)?;
    }
    Ok(())
}

/// Plan I — gate `belief` signatures to the shapes the lowering's
/// storage-hint inference table supports. Surfaces an actionable
/// `UnsupportedBeliefSignature` for anything outside the matrix.
fn validate_belief_signature(
    name: &str,
    params: &[IrParam],
    return_ty: &IrType,
    span: Span,
) -> Result<(), ResolveError> {
    if params.is_empty() {
        return Err(ResolveError::UnsupportedBeliefSignature {
            belief_name: name.to_string(),
            detail: "belief must declare at least an observer param: `(observer: Agent, …)`"
                .to_string(),
            span,
        });
    }
    if params.len() > 2 {
        return Err(ResolveError::UnsupportedBeliefSignature {
            belief_name: name.to_string(),
            detail: format!(
                "belief takes {} params; max is 2 (observer plus one optional key)",
                params.len()
            ),
            span,
        });
    }
    if !matches!(params[0].ty, IrType::AgentId) {
        return Err(ResolveError::UnsupportedBeliefSignature {
            belief_name: name.to_string(),
            detail: format!(
                "first param `{}` must have type `Agent`; got `{:?}`",
                params[0].name, params[0].ty
            ),
            span: params[0].span,
        });
    }
    if let Some(second) = params.get(1) {
        let ok = matches!(
            second.ty,
            IrType::AgentId | IrType::U8 | IrType::U32 | IrType::I32
        );
        if !ok {
            return Err(ResolveError::UnsupportedBeliefSignature {
                belief_name: name.to_string(),
                detail: format!(
                    "second param `{}` must be `Agent` or a scalar key (u8 / u32 / i32); got `{:?}`",
                    second.name, second.ty
                ),
                span: second.span,
            });
        }
    }
    let return_ok = matches!(
        return_ty,
        IrType::Bool
            | IrType::U8
            | IrType::U32
            | IrType::I32
            | IrType::F32
            | IrType::EntityRef(_)
    );
    if !return_ok {
        return Err(ResolveError::UnsupportedBeliefSignature {
            belief_name: name.to_string(),
            detail: format!(
                "return type `{return_ty:?}` not supported; \
                 belief cells must be `bool`, a scalar (u8/u32/i32/f32), \
                 or a registered struct entity"
            ),
            span,
        });
    }
    Ok(())
}

fn validate_fold_stmt(view_name: &str, s: &IrStmt) -> Result<(), ResolveError> {
    match s {
        IrStmt::Let { value, .. } => validate_fold_expr(view_name, value),
        IrStmt::SelfUpdate { op, value, span } => {
            if !matches!(op.as_str(), "=" | "+=" | "-=" | "*=" | "/=" | "|=") {
                return Err(ResolveError::UdfInViewFoldBody {
                    view_name: view_name.to_string(),
                    offending_construct: format!("self-update operator `{op}`"),
                    span: *span,
                });
            }
            validate_fold_expr(view_name, value)
        }
        IrStmt::SelfAppend { fields, .. } => {
            // Plan G G3b/G3c — struct ring append. Each field's bound
            // expression must be a fold-legal expression (no Emit, no
            // ApplyAbility, etc.); validate per-field. The per-cell
            // struct layout is inferred at lowering time.
            for f in fields {
                validate_fold_expr(view_name, &f.value)?;
            }
            Ok(())
        }
        IrStmt::If { cond, then_body, else_body, .. } => {
            validate_fold_expr(view_name, cond)?;
            for ts in then_body {
                validate_fold_stmt(view_name, ts)?;
            }
            if let Some(eb) = else_body {
                for es in eb {
                    validate_fold_stmt(view_name, es)?;
                }
            }
            Ok(())
        }
        IrStmt::Match { span, .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "`match` statement (use if/else in fold bodies)".into(),
            span: *span,
        }),
        IrStmt::Expr(e) => validate_fold_expr(view_name, e),
        IrStmt::For { span, .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "unbounded `for` loop".into(),
            span: *span,
        }),
        IrStmt::ForEachAgent { span, .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "`for_each_agent` body-shape (only valid in physics)".into(),
            span: *span,
        }),
        IrStmt::Emit(IrEmit { span, .. }) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "`emit` inside fold body (only physics cascades emit events)"
                .into(),
            span: *span,
        }),
        IrStmt::ApplyAbility { span, .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct:
                "`apply_ability` inside fold body (registry dispatch only valid in physics)"
                    .into(),
            span: *span,
        }),
        IrStmt::BeliefObserve { span, .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct:
                "`beliefs().observe()` inside fold body (belief mutations only valid in physics)"
                    .into(),
            span: *span,
        }),
    }
}

fn validate_fold_expr(view_name: &str, e: &IrExprNode) -> Result<(), ResolveError> {
    match &e.kind {
        // Literals, locals, and resolved name references — trivially allowed.
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::EnumVariant { .. }
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. } => Ok(()),

        // Stdlib 1-hop method calls (e.g. `rng.uniform(0, 1)`, `query.*`)
        // are allowed. These are the only "call" shape permitted.
        IrExpr::NamespaceCall { args, .. } => {
            for a in args {
                validate_fold_expr(view_name, &a.value)?;
            }
            Ok(())
        }

        // Built-in math / aggregation primitives are allowed.
        IrExpr::BuiltinCall(_, args) => {
            for a in args {
                validate_fold_expr(view_name, &a.value)?;
            }
            Ok(())
        }

        // Cross-view composition rejected — views-calling-views inside a
        // fold body would break the one-pass commutative-update contract.
        IrExpr::ViewCall(_, _) | IrExpr::View(_) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct:
                "call to another view (cross-view composition forbidden in fold bodies)"
                    .into(),
            span: e.span,
        }),

        // Verb calls are not fold-body primitives.
        IrExpr::VerbCall(_, _) | IrExpr::Verb(_) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "verb call".into(),
            span: e.span,
        }),

        IrExpr::UnresolvedCall(name, _) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: format!(
                "unresolved call `{name}` (user-defined helpers are forbidden in fold bodies)"
            ),
            span: e.span,
        }),

        // Field / index / tuple / list are pure projections.
        IrExpr::Field { base, .. } => validate_fold_expr(view_name, base),
        IrExpr::Index(a, b) => {
            validate_fold_expr(view_name, a)?;
            validate_fold_expr(view_name, b)
        }
        IrExpr::Tuple(xs) | IrExpr::List(xs) => {
            for x in xs {
                validate_fold_expr(view_name, x)?;
            }
            Ok(())
        }

        // Arithmetic / comparison / logical operators.
        IrExpr::Binary(_, lhs, rhs) | IrExpr::In(lhs, rhs) | IrExpr::Contains(lhs, rhs) => {
            validate_fold_expr(view_name, lhs)?;
            validate_fold_expr(view_name, rhs)
        }
        IrExpr::Unary(_, rhs) => validate_fold_expr(view_name, rhs),

        // Bounded folds are allowed; quantifiers too (spec §2.3's closed
        // set already includes `forall`/`exists` via the logical surface).
        IrExpr::Fold { iter, body, .. } => {
            if let Some(i) = iter {
                validate_fold_expr(view_name, i)?;
            }
            validate_fold_expr(view_name, body)
        }
        IrExpr::Quantifier { iter, body, .. } => {
            validate_fold_expr(view_name, iter)?;
            validate_fold_expr(view_name, body)
        }

        // Struct literals / ctors are data shapes, not calls.
        IrExpr::StructLit { fields, .. } => {
            for f in fields {
                validate_fold_expr(view_name, &f.value)?;
            }
            Ok(())
        }
        IrExpr::Ctor { args, .. } => {
            for a in args {
                validate_fold_expr(view_name, a)?;
            }
            Ok(())
        }

        IrExpr::Match { .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "`match` expression (use if/else in fold bodies)".into(),
            span: e.span,
        }),
        IrExpr::If { cond, then_expr, else_expr } => {
            validate_fold_expr(view_name, cond)?;
            validate_fold_expr(view_name, then_expr)?;
            if let Some(eb) = else_expr {
                validate_fold_expr(view_name, eb)?;
            }
            Ok(())
        }

        IrExpr::PerUnit { expr, delta } => {
            validate_fold_expr(view_name, expr)?;
            validate_fold_expr(view_name, delta)
        }

        IrExpr::Raw(_) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "unrecognised expression shape".into(),
            span: e.span,
        }),

        // GPU ability evaluation Phase 2 primitives — scoring-only
        // surface. A view fold body has no currently-scored ability,
        // so reading ability tags / hints / ranges / cooldowns here
        // is meaningless.
        IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::AbilityOnCooldown(_) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct:
                "ability-evaluation primitive (`ability::tag(...)`, `ability::hint`, \
                 `ability::range`, `ability::on_cooldown(...)`) — scoring-only surface"
                    .into(),
            span: e.span,
        }),

        // Plan ToM Task 8 — belief read expressions. These carry observer/
        // target sub-expressions; we reject them from fold bodies (belief
        // reads need the full SimState cold_beliefs lookup that the fold
        // body context doesn't provide). T9 closes the scoring/physics
        // lowering path.
        IrExpr::BeliefsAccessor { .. }
        | IrExpr::BeliefsConfidence { .. }
        | IrExpr::BeliefsView { .. } => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct:
                "belief-read primitive (`beliefs(...).about(...).<field>` etc.) — \
                 not valid inside a view fold body"
                    .into(),
            span: e.span,
        }),
    }
}

// ---------------------------------------------------------------------------
// Physics body GPU-emittable validator (compiler/spec.md §1.2)
// ---------------------------------------------------------------------------
//
// Task 155 gave physics bodies `for` + `match` — a richer surface than the
// fold-body context carries. This validator locks that surface to the
// GPU-emittable subset documented in `compiler/spec.md` §1.2:
//
//   - POD discipline (`T: Pod`, `AggregatePool<T>`), no heap collections.
//   - Fixed-size inline-array / bounded spatial-query iteration sources —
//     never a runtime-sized `Vec<T>` or `HashMap<K, V>`.
//   - Self-emission recursion capped by `@terminating_in(N)` (spec §2.4)
//     or by a body check against `cascade.max_iterations` (the cascade
//     framework's global iteration ceiling). Arbitrary physics-rule
//     recursion beyond that is forbidden.
//   - No user-defined helpers / closures / trait objects.
//   - No `String` bindings inside the body — chronicle prose lives on
//     host-side template expansion, and replayable events already refuse
//     `String` fields.
//
// The validator runs AFTER `resolve_bodies` on the resolved `Compilation`
// so it has the full cross-rule picture for indirect-cycle detection.

/// Enforce that every physics rule body is emittable to SPIR-V.
pub(crate) fn validate_physics_bodies(comp: &Compilation) -> Result<(), ResolveError> {
    // Pre-collect the set of event names declared `@non_replayable`.
    // String-typed fields on these events are LEGAL chronicle prose
    // (the non-replayable ring is host-side only, P10), so an
    // `emit <NonReplayableEvent> { utterance: "..." }` inside a
    // physics body must NOT trigger the heap-allocation rejection.
    // The body validator threads this set through so its emit-arm
    // handler can recognise the case.
    let mut non_replayable_events: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for ev in &comp.events {
        if ev.annotations.iter().any(|a| a.name == "non_replayable") {
            non_replayable_events.insert(ev.name.clone());
        }
    }

    // Cross-rule recursion bookkeeping: for every physics rule, collect
    // the set of event names it handles and the set of event names it
    // emits. A rule is "recursive" if any event it emits is handled by
    // itself (direct) or by a rule that transitively emits back into it
    // (indirect). Both are rejected unless the rule is annotated
    // `@terminating_in(N)` or checks `cascade.max_iterations` (spec §2.4).
    let mut handled: Vec<Vec<String>> = Vec::with_capacity(comp.physics.len());
    let mut emitted: Vec<Vec<String>> = Vec::with_capacity(comp.physics.len());
    for p in &comp.physics {
        let mut h: Vec<String> = Vec::new();
        let mut e: Vec<String> = Vec::new();
        for handler in &p.handlers {
            match &handler.pattern {
                IrPhysicsPattern::Kind(kp) => {
                    if !h.iter().any(|n| n == &kp.name) {
                        h.push(kp.name.clone());
                    }
                }
                IrPhysicsPattern::Tag { name, tag, .. } => {
                    if let Some(tref) = tag {
                        for ev in &comp.events {
                            if ev.tags.contains(tref) && !h.iter().any(|n| n == &ev.name) {
                                h.push(ev.name.clone());
                            }
                        }
                    } else if !h.iter().any(|n| n == name) {
                        h.push(name.clone());
                    }
                }
            }
            collect_emitted_events(&handler.body, &mut e);
        }
        handled.push(h);
        emitted.push(e);
    }

    for (idx, p) in comp.physics.iter().enumerate() {
        // `@cpu_only` rules are intentionally-CPU-only; they'll never be
        // lowered to WGSL, so primitives like strings, unbounded
        // allocations, recursion, etc. inside their bodies don't need to
        // pass the GPU-emittable check. Short-circuit accept — the CPU
        // handler path runs arbitrary Rust and has no such restrictions.
        if p.cpu_only {
            continue;
        }

        // Two escape hatches for self-emission bounded recursion:
        //
        //   1. `@terminating_in(N)` annotation — spec §2.4's explicit
        //      bound-the-depth marker.
        //   2. A body that guards the self-emission against
        //      `cascade.max_iterations` — the cascade framework's global
        //      iteration ceiling (currently `MAX_CASCADE_ITERATIONS = 8`).
        //      The `physics cast` rule uses this shape: it checks
        //      `new_depth >= cascade.max_iterations` and emits
        //      `CastDepthExceeded` instead of the nested event once the
        //      depth reaches the ceiling. That's a bounded SPIR-V-
        //      emittable recursion; the cascade framework itself enforces
        //      the cap.
        let has_terminator = p.annotations.iter().any(|a| a.name == "terminating_in")
            || handlers_guard_on_cascade_ceiling(&p.handlers);

        if !has_terminator {
            // Direct self-recursion: this rule emits an event it also handles.
            for ev in &emitted[idx] {
                if handled[idx].iter().any(|h| h == ev) {
                    let span = find_emit_span(&p.handlers, ev).unwrap_or(p.span);
                    return Err(ResolveError::NotGpuEmittable {
                        physics_name: p.name.clone(),
                        construct: format!("recursive self-emission of `{ev}`"),
                        reason: format!(
                            "rule `{}` emits `{ev}` which retriggers itself; \
                             bound the recursion with `@terminating_in(N)` \
                             (spec §2.4), or guard the self-emission against \
                             `cascade.max_iterations` so the SPIR-V kernel \
                             has a compile-time iteration ceiling",
                            p.name
                        ),
                        span,
                    });
                }
            }
            // Indirect recursion: a cycle through other rules back to self.
            if emits_cycle_back(idx, &handled, &emitted) {
                return Err(ResolveError::NotGpuEmittable {
                    physics_name: p.name.clone(),
                    construct: "indirect recursion via emitted events".into(),
                    reason: format!(
                        "rule `{}` sits on an event-emission cycle that \
                         returns to itself; break the cycle or annotate \
                         every participating rule with `@terminating_in(N)` \
                         (spec §2.4)",
                        p.name
                    ),
                    span: p.span,
                });
            }
        }

        // Per-handler body walk: heap types, unbounded iter sources, UDF
        // calls, `String` let-bindings.
        for h in &p.handlers {
            validate_physics_body(&p.name, &h.body, &non_replayable_events)?;
        }
    }

    Ok(())
}

/// Recursive walker on resolved physics-handler statements.
pub(crate) fn validate_physics_body(
    physics_name: &str,
    body: &[IrStmt],
    non_replayable_events: &std::collections::HashSet<String>,
) -> Result<(), ResolveError> {
    for s in body {
        validate_physics_stmt(physics_name, s, non_replayable_events)?;
    }
    Ok(())
}

fn validate_physics_stmt(
    physics_name: &str,
    s: &IrStmt,
    non_replayable_events: &std::collections::HashSet<String>,
) -> Result<(), ResolveError> {
    match s {
        IrStmt::Let { name, value, span, .. } => {
            // `String` bindings defeat the POD discipline — every hot /
            // cold field that persists state has to be `Pod`, and the
            // only `String` surface on events is `@non_replayable`
            // metadata.
            if expr_mentions_string_literal(value) {
                return Err(ResolveError::NotGpuEmittable {
                    physics_name: physics_name.to_string(),
                    construct: format!("`String` let-binding `{name}`"),
                    reason:
                        "heap-backed `String` isn't `Pod` and can't round-trip \
                         through an `AggregatePool<T>` or a SPIR-V storage \
                         buffer"
                            .into(),
                    span: *span,
                });
            }
            validate_physics_expr(physics_name, value)
        }
        IrStmt::Emit(IrEmit { event_name, fields, .. }) => {
            // Spec: `String` field values are LEGAL when the emitted
            // event is declared `@non_replayable` — the chronicle ring
            // is host-side only (compiler/spec.md §1.2), so the heap
            // allocation never has to round-trip through a Pod
            // storage buffer or replay log. For replayable events the
            // POD discipline still applies; recurse through the
            // standard validator to surface the same error message
            // pre-existing fixtures see.
            let allow_strings = non_replayable_events.contains(event_name);
            for f in fields {
                if allow_strings {
                    // Permissive walk: skip the String-literal check on
                    // bare values (the validator's only String-rejection
                    // path), but still recurse into composite shapes so
                    // unsupported call constructs surface normally.
                    validate_physics_expr_allowing_strings(physics_name, &f.value)?;
                } else {
                    validate_physics_expr(physics_name, &f.value)?;
                }
            }
            Ok(())
        }
        IrStmt::For { iter, filter, body, span, .. } => {
            validate_physics_iter_source(physics_name, iter, *span)?;
            validate_physics_expr(physics_name, iter)?;
            if let Some(f) = filter {
                validate_physics_expr(physics_name, f)?;
            }
            for bs in body {
                validate_physics_stmt(physics_name, bs, non_replayable_events)?;
            }
            Ok(())
        }
        IrStmt::ForEachAgent { body, .. } => {
            // No iter / filter to validate — the iter is implicit
            // (every alive agent slot in 0..agent_cap order). Recurse
            // into the body so any per-statement physics constraints
            // (string literals, unsupported namespaces, etc.) still
            // surface from inside the for_each_agent block.
            for bs in body {
                validate_physics_stmt(physics_name, bs, non_replayable_events)?;
            }
            Ok(())
        }
        IrStmt::If { cond, then_body, else_body, .. } => {
            validate_physics_expr(physics_name, cond)?;
            for ts in then_body {
                validate_physics_stmt(physics_name, ts, non_replayable_events)?;
            }
            if let Some(eb) = else_body {
                for es in eb {
                    validate_physics_stmt(physics_name, es, non_replayable_events)?;
                }
            }
            Ok(())
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            validate_physics_expr(physics_name, scrutinee)?;
            for arm in arms {
                for stmt in &arm.body {
                    validate_physics_stmt(physics_name, stmt, non_replayable_events)?;
                }
            }
            Ok(())
        }
        IrStmt::SelfUpdate { value, .. } => validate_physics_expr(physics_name, value),
        IrStmt::SelfAppend { fields, .. } => {
            // Plan G G3b/G3c — physics rules don't have a `self` cell to
            // ring-append into; the resolver normally rejects this via
            // the fold-vs-physics body separation, but defensively
            // validate the bound exprs here so synthetic ASTs still get
            // surfaced cleanly.
            for f in fields {
                validate_physics_expr(physics_name, &f.value)?;
            }
            Ok(())
        }
        IrStmt::Expr(e) => validate_physics_expr(physics_name, e),
        IrStmt::ApplyAbility { ability, caster, target, .. } => {
            // Slice ε: validate the new optional caster/target operands
            // alongside the ability. A typo'd `apply_ability a by xx`
            // where `xx` references an unknown name would otherwise
            // ship through the resolver and surface as a confusing
            // CG-level error far from the source.
            validate_physics_expr(physics_name, ability)?;
            if let Some(c) = caster {
                validate_physics_expr(physics_name, c)?;
            }
            if let Some(t) = target {
                validate_physics_expr(physics_name, t)?;
            }
            Ok(())
        }
        IrStmt::BeliefObserve { observer, target, fields, .. } => {
            validate_physics_expr(physics_name, observer)?;
            validate_physics_expr(physics_name, target)?;
            for f in fields {
                validate_physics_expr(physics_name, &f.value)?;
            }
            Ok(())
        }
    }
}

fn validate_physics_expr(physics_name: &str, e: &IrExprNode) -> Result<(), ResolveError> {
    match &e.kind {
        // Literals / bare name references / resolved namespaces — all OK.
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::EnumVariant { .. }
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. } => Ok(()),

        // `String` literals are only legal on `@non_replayable` event
        // field assignments (chronicle prose). Inside a physics body,
        // they signal a heap allocation escaping into the POD layer.
        IrExpr::LitString(_) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: "`String` literal in body".into(),
            reason:
                "heap-backed `String` values aren't `Pod`; chronicle prose \
                 rendering is host-side only (compiler/spec.md §1.2), and \
                 replayable events already reject `String` fields"
                    .into(),
            span: e.span,
        }),

        // Stdlib 1-hop / builtin calls: recurse into args only.
        IrExpr::NamespaceCall { args, .. } | IrExpr::BuiltinCall(_, args) => {
            for a in args {
                validate_physics_expr(physics_name, &a.value)?;
            }
            Ok(())
        }

        // View calls are bounded by the view's storage hint; materialized
        // views ship with a GPU-resident output buffer (§1.2).
        IrExpr::ViewCall(_, args) => {
            for a in args {
                validate_physics_expr(physics_name, &a.value)?;
            }
            Ok(())
        }
        IrExpr::View(_) | IrExpr::Verb(_) => Ok(()),

        // Verb call args: recurse only — verbs lower to scoring-row lookups.
        IrExpr::VerbCall(_, args) => {
            for a in args {
                validate_physics_expr(physics_name, &a.value)?;
            }
            Ok(())
        }

        // User-defined helper: physics bodies can only call stdlib + emit.
        IrExpr::UnresolvedCall(name, _) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: format!("unresolved call `{name}`"),
            reason: format!(
                "`{name}` is neither a stdlib method nor a declared view / \
                 verb; physics bodies can only call stdlib accessors \
                 (`agents.*`, `abilities.*`, `query.*`, `view::*`, ...), \
                 built-in math, or `emit <Event> {{ ... }}`"
            ),
            span: e.span,
        }),

        // Field / index projections and struct / ctor shapes are pure data.
        IrExpr::Field { base, .. } => validate_physics_expr(physics_name, base),
        IrExpr::Index(a, b) => {
            validate_physics_expr(physics_name, a)?;
            validate_physics_expr(physics_name, b)
        }
        IrExpr::Tuple(xs) | IrExpr::List(xs) => {
            for x in xs {
                validate_physics_expr(physics_name, x)?;
            }
            Ok(())
        }

        IrExpr::Binary(_, lhs, rhs) | IrExpr::In(lhs, rhs) | IrExpr::Contains(lhs, rhs) => {
            validate_physics_expr(physics_name, lhs)?;
            validate_physics_expr(physics_name, rhs)
        }
        IrExpr::Unary(_, rhs) => validate_physics_expr(physics_name, rhs),

        IrExpr::Fold { iter, body, .. } => {
            if let Some(i) = iter {
                validate_physics_expr(physics_name, i)?;
            }
            validate_physics_expr(physics_name, body)
        }
        IrExpr::Quantifier { iter, body, .. } => {
            validate_physics_expr(physics_name, iter)?;
            validate_physics_expr(physics_name, body)
        }

        IrExpr::StructLit { fields, .. } => {
            for f in fields {
                validate_physics_expr(physics_name, &f.value)?;
            }
            Ok(())
        }
        IrExpr::Ctor { args, .. } => {
            for a in args {
                validate_physics_expr(physics_name, a)?;
            }
            Ok(())
        }
        IrExpr::Match { scrutinee, arms } => {
            validate_physics_expr(physics_name, scrutinee)?;
            for arm in arms {
                validate_physics_expr(physics_name, &arm.body)?;
            }
            Ok(())
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            validate_physics_expr(physics_name, cond)?;
            validate_physics_expr(physics_name, then_expr)?;
            if let Some(eb) = else_expr {
                validate_physics_expr(physics_name, eb)?;
            }
            Ok(())
        }
        IrExpr::PerUnit { expr, delta } => {
            validate_physics_expr(physics_name, expr)?;
            validate_physics_expr(physics_name, delta)
        }

        IrExpr::Raw(_) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: "unrecognised expression shape".into(),
            reason:
                "the compiler couldn't lower this construct to a typed IR \
                 node; physics bodies compile to SPIR-V so every expression \
                 must live in the closed surface (literals, locals, stdlib \
                 calls, operators, bounded for / match, emit)"
                    .into(),
            span: e.span,
        }),

        // GPU ability evaluation Phase 2 primitives. Scoring-only
        // surface; a physics body has no currently-scored ability.
        IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::AbilityOnCooldown(_) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: "ability-evaluation primitive in physics body".into(),
            reason:
                "`ability::tag(...)`, `ability::hint`, `ability::range`, \
                 and `ability::on_cooldown(...)` read state tied to the \
                 currently-scored ability inside a `per_ability` scoring \
                 row; they have no meaning inside a physics handler"
                    .into(),
            span: e.span,
        }),

        // Plan ToM Task 8 — belief read expressions. Physics bodies CAN
        // legally read belief state (they know the full SimState); we allow
        // them through here and recurse into sub-expressions so that T9's
        // CPU lowering (emit_physics.rs) gains the surface cleanly. GPU
        // lowering is deferred so they're still rejected by the WGSL emitter
        // until a matching WGSL T9 lands.
        IrExpr::BeliefsAccessor { observer, target, .. } => {
            validate_physics_expr(physics_name, observer)?;
            validate_physics_expr(physics_name, target)
        }
        IrExpr::BeliefsConfidence { observer, target } => {
            validate_physics_expr(physics_name, observer)?;
            validate_physics_expr(physics_name, target)
        }
        IrExpr::BeliefsView { observer, .. } => {
            validate_physics_expr(physics_name, observer)
        }
    }
}

/// Permissive variant of [`validate_physics_expr`] used when the
/// expression is a field-value position on an `emit
/// <NonReplayableEvent> { ... }` statement. Bare `String` literals
/// (chronicle prose) are LEGAL in this position because the
/// non-replayable ring lives host-side only — no `Pod` round-trip.
/// Everything else is checked under the standard
/// [`validate_physics_expr`] rules so the relaxation is scoped to
/// the literal at the field's bare value position; nested function
/// calls / arithmetic that compute Strings (none today) would still
/// flow through the strict check via composite-shape recursion.
fn validate_physics_expr_allowing_strings(
    physics_name: &str,
    e: &IrExprNode,
) -> Result<(), ResolveError> {
    match &e.kind {
        // The ONE relaxation: a bare String literal at a field-value
        // position on a `@non_replayable` emit.
        IrExpr::LitString(_) => Ok(()),
        // Everything else delegates to the standard validator. Composite
        // shapes (List, Tuple, StructLit, ...) recurse through the
        // strict path — this is intentional: the chronicle prose
        // exemption is for the bare literal, not for arbitrary
        // String-producing call shapes that don't exist in the DSL
        // anyway.
        _ => validate_physics_expr(physics_name, e),
    }
}

/// Physics `for` iteration sources must have a compile-time cap. Accept:
///
/// - Stdlib namespace calls (`query.nearby_agents`, `abilities.effects`,
///   `voxel.neighbors_above`, ...). Every stdlib method that yields a
///   list returns a bounded `SmallVec` / fixed-size array (§1.2).
/// - Field projections (`agent.memberships`, `agent.creditor_ledger`) —
///   entity fields are declared as `SortedVec<T, N>` / `Array<T, N>` /
///   `RingBuffer<T, N>` / `SmallVec<[T; N]>`, all capped at compile time.
/// - Materialized view reads — the storage hint pins the shape.
/// - `Local` binder — assumed bounded because a prior `let` / `for` /
///   handler-binding vetted the upstream source.
/// - `Index(...)` — a single element of a capped collection.
/// - Literal `List` / `Tuple` — length is a compile-time constant.
/// - `BuiltinCall` — stdlib math / aggregates.
/// - Bare `Namespace` (e.g. `agents`) — legacy collection accessor,
///   capped by the global agent slot pool.
///
/// Reject:
///
/// - `UnresolvedCall` — indistinguishable from a UDF helper.
/// - `VerbCall` / bare `View` / `Verb` — not iterables.
/// - `Raw` — unlowered expression; shape unknown.
/// - Literals / operators — not iterables.
fn validate_physics_iter_source(
    physics_name: &str,
    iter: &IrExprNode,
    for_span: Span,
) -> Result<(), ResolveError> {
    match &iter.kind {
        // Bounded iterable sources.
        IrExpr::NamespaceCall { .. }
        | IrExpr::Field { .. }
        | IrExpr::ViewCall(_, _)
        | IrExpr::Namespace(_)
        | IrExpr::Local(_, _)
        | IrExpr::Index(_, _)
        | IrExpr::List(_)
        | IrExpr::Tuple(_)
        | IrExpr::BuiltinCall(_, _) => Ok(()),

        IrExpr::UnresolvedCall(name, _) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: format!("for-loop over user-defined helper `{name}`"),
            reason: format!(
                "`{name}` is not a stdlib accessor; `for` iteration sources \
                 must be bounded (a spatial query like `query.nearby_agents`, \
                 an ability program via `abilities.effects`, a capped entity \
                 field like `agent.memberships`, or a materialized view)"
            ),
            span: for_span,
        }),

        IrExpr::VerbCall(_, _) | IrExpr::Verb(_) | IrExpr::View(_) => {
            Err(ResolveError::NotGpuEmittable {
                physics_name: physics_name.to_string(),
                construct: "for-loop over verb / bare view reference".into(),
                reason:
                    "verbs and bare view references aren't iterables; use a \
                     bounded spatial query, a capped entity field, or call \
                     the view (`view::<name>(...)`) instead"
                        .into(),
                span: for_span,
            })
        }

        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Binary(_, _, _)
        | IrExpr::Unary(_, _)
        | IrExpr::In(_, _)
        | IrExpr::Contains(_, _)
        | IrExpr::EnumVariant { .. }
        | IrExpr::NamespaceField { .. }
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::StructLit { .. }
        | IrExpr::Ctor { .. }
        | IrExpr::If { .. }
        | IrExpr::Match { .. }
        | IrExpr::Fold { .. }
        | IrExpr::Quantifier { .. }
        | IrExpr::PerUnit { .. }
        | IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::AbilityOnCooldown(_)
        // Plan ToM Task 8 — belief reads aren't iterable collections.
        | IrExpr::BeliefsAccessor { .. }
        | IrExpr::BeliefsConfidence { .. }
        | IrExpr::BeliefsView { .. } => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: "for-loop over non-iterable / unbounded expression".into(),
            reason:
                "`for` iteration sources must be bounded collections (a \
                 spatial query, a capped entity field, an ability program, \
                 or a materialized view); literal / computed expressions \
                 can't be proved bounded at compile time"
                    .into(),
            span: for_span,
        }),

        IrExpr::Raw(_) => Err(ResolveError::NotGpuEmittable {
            physics_name: physics_name.to_string(),
            construct: "for-loop over unrecognised expression".into(),
            reason: "iteration source didn't lower to a typed IR node".into(),
            span: for_span,
        }),
    }
}

/// Shallow check: does this resolved expression directly carry a `String`
/// literal? Used to flag `let name = "foo"` in a physics body. Full
/// `IrType` carrying would need pass 1b's type checker — until then,
/// catch the overwhelmingly common case.
fn expr_mentions_string_literal(e: &IrExprNode) -> bool {
    matches!(&e.kind, IrExpr::LitString(_))
}

fn collect_emitted_events(body: &[IrStmt], out: &mut Vec<String>) {
    for s in body {
        match s {
            IrStmt::Emit(IrEmit { event_name, .. }) => {
                if !out.iter().any(|n| n == event_name) {
                    out.push(event_name.clone());
                }
            }
            IrStmt::For { body, .. } => collect_emitted_events(body, out),
            IrStmt::ForEachAgent { body, .. } => collect_emitted_events(body, out),
            IrStmt::If { then_body, else_body, .. } => {
                collect_emitted_events(then_body, out);
                if let Some(eb) = else_body {
                    collect_emitted_events(eb, out);
                }
            }
            IrStmt::Match { arms, .. } => {
                for arm in arms {
                    collect_emitted_events(&arm.body, out);
                }
            }
            IrStmt::Let { .. }
            | IrStmt::SelfUpdate { .. }
            | IrStmt::SelfAppend { .. }
            | IrStmt::Expr(_)
            | IrStmt::BeliefObserve { .. }
            | IrStmt::ApplyAbility { .. } => {}
        }
    }
}

fn find_emit_span(handlers: &[PhysicsHandlerIR], target: &str) -> Option<Span> {
    for h in handlers {
        if let Some(sp) = find_emit_span_in_stmts(&h.body, target) {
            return Some(sp);
        }
    }
    None
}

fn find_emit_span_in_stmts(body: &[IrStmt], target: &str) -> Option<Span> {
    for s in body {
        match s {
            IrStmt::Emit(IrEmit { event_name, span, .. }) if event_name == target => {
                return Some(*span);
            }
            IrStmt::For { body, .. } | IrStmt::ForEachAgent { body, .. } => {
                if let Some(sp) = find_emit_span_in_stmts(body, target) {
                    return Some(sp);
                }
            }
            IrStmt::If { then_body, else_body, .. } => {
                if let Some(sp) = find_emit_span_in_stmts(then_body, target) {
                    return Some(sp);
                }
                if let Some(eb) = else_body {
                    if let Some(sp) = find_emit_span_in_stmts(eb, target) {
                        return Some(sp);
                    }
                }
            }
            IrStmt::Match { arms, .. } => {
                for arm in arms {
                    if let Some(sp) = find_emit_span_in_stmts(&arm.body, target) {
                        return Some(sp);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Detect an indirect emission cycle. Seeds the search from every event
/// `start` emits, walks every rule that handles that event, and reports
/// a hit when the walk reaches a rule whose emissions land back on one
/// of `start`'s handled events. Direct self-recursion (`start` -> `start`)
/// is diagnosed separately by the caller so this path doesn't double-fire.
fn emits_cycle_back(
    start: usize,
    handled: &[Vec<String>],
    emitted: &[Vec<String>],
) -> bool {
    use std::collections::VecDeque;
    let mut seen = vec![false; handled.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    // Seed with every rule (other than `start`) that handles one of
    // `start`'s emitted events — the first hops away from `start`.
    for ev in &emitted[start] {
        for (j, handled_j) in handled.iter().enumerate() {
            if j == start || seen[j] {
                continue;
            }
            if handled_j.iter().any(|h| h == ev) {
                seen[j] = true;
                queue.push_back(j);
            }
        }
    }
    while let Some(j) = queue.pop_front() {
        for ev in &emitted[j] {
            // Cycle back: someone we visit emits an event that `start` handles.
            if handled[start].iter().any(|h| h == ev) {
                return true;
            }
            for (k, handled_k) in handled.iter().enumerate() {
                if k == start || seen[k] {
                    continue;
                }
                if handled_k.iter().any(|h| h == ev) {
                    seen[k] = true;
                    queue.push_back(k);
                }
            }
        }
    }
    false
}

/// Does any handler in this rule reference `cascade.max_iterations`?
/// This is the cascade framework's global iteration ceiling
/// (`MAX_CASCADE_ITERATIONS`); a rule that checks against it has a
/// compile-time bound on recursion even without `@terminating_in`.
fn handlers_guard_on_cascade_ceiling(handlers: &[PhysicsHandlerIR]) -> bool {
    for h in handlers {
        if stmts_reference_cascade_ceiling(&h.body) {
            return true;
        }
    }
    false
}

fn stmts_reference_cascade_ceiling(body: &[IrStmt]) -> bool {
    for s in body {
        if stmt_references_cascade_ceiling(s) {
            return true;
        }
    }
    false
}

fn stmt_references_cascade_ceiling(s: &IrStmt) -> bool {
    match s {
        IrStmt::Let { value, .. } => expr_references_cascade_ceiling(value),
        IrStmt::Emit(IrEmit { fields, .. }) => {
            fields.iter().any(|f| expr_references_cascade_ceiling(&f.value))
        }
        IrStmt::For { iter, filter, body, .. } => {
            expr_references_cascade_ceiling(iter)
                || filter.as_ref().is_some_and(expr_references_cascade_ceiling)
                || stmts_reference_cascade_ceiling(body)
        }
        IrStmt::ForEachAgent { body, .. } => stmts_reference_cascade_ceiling(body),
        IrStmt::If { cond, then_body, else_body, .. } => {
            expr_references_cascade_ceiling(cond)
                || stmts_reference_cascade_ceiling(then_body)
                || else_body
                    .as_ref()
                    .is_some_and(|eb| stmts_reference_cascade_ceiling(eb))
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            expr_references_cascade_ceiling(scrutinee)
                || arms.iter().any(|a| stmts_reference_cascade_ceiling(&a.body))
        }
        IrStmt::SelfUpdate { value, .. } => expr_references_cascade_ceiling(value),
        IrStmt::SelfAppend { fields, .. } => {
            fields.iter().any(|f| expr_references_cascade_ceiling(&f.value))
        }
        IrStmt::Expr(e) => expr_references_cascade_ceiling(e),
        IrStmt::BeliefObserve { observer, target, fields, .. } => {
            expr_references_cascade_ceiling(observer)
                || expr_references_cascade_ceiling(target)
                || fields.iter().any(|f| expr_references_cascade_ceiling(&f.value))
        }
        IrStmt::ApplyAbility { ability, caster, target, .. } => {
            // Slice ε: cascade-ceiling check considers all three
            // operands. A `cascade_depth` reference inside the
            // caster/target operand should also surface (otherwise
            // the cascade-ceiling guard could miss legitimate
            // recursion-control patterns).
            expr_references_cascade_ceiling(ability)
                || caster.as_ref().is_some_and(expr_references_cascade_ceiling)
                || target.as_ref().is_some_and(expr_references_cascade_ceiling)
        }
    }
}

fn expr_references_cascade_ceiling(e: &IrExprNode) -> bool {
    match &e.kind {
        IrExpr::NamespaceField { ns, field, .. } => {
            *ns == NamespaceId::Cascade && field == "max_iterations"
        }
        IrExpr::Binary(_, a, b) | IrExpr::In(a, b) | IrExpr::Contains(a, b) => {
            expr_references_cascade_ceiling(a) || expr_references_cascade_ceiling(b)
        }
        IrExpr::Unary(_, x) => expr_references_cascade_ceiling(x),
        IrExpr::Field { base, .. } => expr_references_cascade_ceiling(base),
        IrExpr::Index(a, b) => {
            expr_references_cascade_ceiling(a) || expr_references_cascade_ceiling(b)
        }
        IrExpr::Tuple(xs) | IrExpr::List(xs) => xs.iter().any(expr_references_cascade_ceiling),
        IrExpr::NamespaceCall { args, .. }
        | IrExpr::BuiltinCall(_, args)
        | IrExpr::ViewCall(_, args)
        | IrExpr::VerbCall(_, args)
        | IrExpr::UnresolvedCall(_, args) => {
            args.iter().any(|a| expr_references_cascade_ceiling(&a.value))
        }
        IrExpr::Fold { iter, body, .. } => {
            iter.as_ref().is_some_and(|i| expr_references_cascade_ceiling(i))
                || expr_references_cascade_ceiling(body)
        }
        IrExpr::Quantifier { iter, body, .. } => {
            expr_references_cascade_ceiling(iter) || expr_references_cascade_ceiling(body)
        }
        IrExpr::StructLit { fields, .. } => {
            fields.iter().any(|f| expr_references_cascade_ceiling(&f.value))
        }
        IrExpr::Ctor { args, .. } => args.iter().any(expr_references_cascade_ceiling),
        IrExpr::Match { scrutinee, arms } => {
            expr_references_cascade_ceiling(scrutinee)
                || arms.iter().any(|a| expr_references_cascade_ceiling(&a.body))
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            expr_references_cascade_ceiling(cond)
                || expr_references_cascade_ceiling(then_expr)
                || else_expr
                    .as_ref()
                    .is_some_and(|eb| expr_references_cascade_ceiling(eb))
        }
        IrExpr::PerUnit { expr, delta } => {
            expr_references_cascade_ceiling(expr) || expr_references_cascade_ceiling(delta)
        }
        _ => false,
    }
}


// ---------------------------------------------------------------------------
// Mask / scoring body operator-set validators (spec §2.5)
// ---------------------------------------------------------------------------
//
// Mask predicates and scoring rows both lower into the same GPU-friendly
// scalar surface (SPIR-V boolean / f32 kernels). The closed operator set
// mirrors the fold-body restriction minus the `self +=` family: pure
// expressions over stdlib accessors + bounded aggregates + quantifiers +
// view calls, with `if/else` as the only control flow. Physics bodies allow
// `for` and `match`, but mask/scoring contexts stay restricted by design —
// they compile to per-row GPU kernels where unbounded iteration and variant
// dispatch aren't available. The validators are expression-only; `for`
// statements can't reach these slots (the parser rejects `for` in expression
// position), so the validators primarily catch `match` expressions — the one
// forbidden shape that *does* parse as an expr.

fn validate_mask_body(mask_name: &str, e: &IrExprNode) -> Result<(), ResolveError> {
    match &e.kind {
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::EnumVariant { .. }
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. } => Ok(()),

        IrExpr::NamespaceCall { args, .. } | IrExpr::BuiltinCall(_, args) => {
            for a in args {
                validate_mask_body(mask_name, &a.value)?;
            }
            Ok(())
        }

        IrExpr::Quantifier { iter, body, .. } => {
            validate_mask_body(mask_name, iter)?;
            validate_mask_body(mask_name, body)
        }
        IrExpr::Fold { iter, body, .. } => {
            if let Some(i) = iter {
                validate_mask_body(mask_name, i)?;
            }
            validate_mask_body(mask_name, body)
        }

        IrExpr::ViewCall(_, args) => {
            for a in args {
                validate_mask_body(mask_name, &a.value)?;
            }
            Ok(())
        }
        IrExpr::View(_) => Ok(()),

        IrExpr::VerbCall(_, args) => {
            for a in args {
                validate_mask_body(mask_name, &a.value)?;
            }
            Ok(())
        }
        IrExpr::Verb(_) => Ok(()),

        IrExpr::UnresolvedCall(_, args) => {
            for a in args {
                validate_mask_body(mask_name, &a.value)?;
            }
            Ok(())
        }

        IrExpr::Field { base, .. } => validate_mask_body(mask_name, base),
        IrExpr::Index(a, b) => {
            validate_mask_body(mask_name, a)?;
            validate_mask_body(mask_name, b)
        }
        IrExpr::Tuple(xs) | IrExpr::List(xs) => {
            for x in xs {
                validate_mask_body(mask_name, x)?;
            }
            Ok(())
        }

        IrExpr::Binary(_, lhs, rhs) | IrExpr::In(lhs, rhs) | IrExpr::Contains(lhs, rhs) => {
            validate_mask_body(mask_name, lhs)?;
            validate_mask_body(mask_name, rhs)
        }
        IrExpr::Unary(_, rhs) => validate_mask_body(mask_name, rhs),

        IrExpr::If { cond, then_expr, else_expr } => {
            validate_mask_body(mask_name, cond)?;
            validate_mask_body(mask_name, then_expr)?;
            if let Some(eb) = else_expr {
                validate_mask_body(mask_name, eb)?;
            }
            Ok(())
        }

        IrExpr::StructLit { fields, .. } => {
            for f in fields {
                validate_mask_body(mask_name, &f.value)?;
            }
            Ok(())
        }
        IrExpr::Ctor { args, .. } => {
            for a in args {
                validate_mask_body(mask_name, a)?;
            }
            Ok(())
        }

        IrExpr::PerUnit { expr, delta } => {
            validate_mask_body(mask_name, expr)?;
            validate_mask_body(mask_name, delta)
        }

        IrExpr::Match { .. } => Err(ResolveError::UdfInMaskBody {
            mask_name: mask_name.to_string(),
            offending_construct: "`match` expression (use if/else or view dispatch)".into(),
            span: e.span,
        }),

        // GPU ability evaluation Phase 2 primitives. Reachable from
        // mask bodies would mean a mask predicate depends on the
        // currently-scored ability — which isn't the mask kernel's
        // slot; masks run before scoring. Reject.
        IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::AbilityOnCooldown(_) => Err(ResolveError::UdfInMaskBody {
            mask_name: mask_name.to_string(),
            offending_construct:
                "ability-evaluation primitive (`ability::tag(...)`, `ability::hint`, \
                 `ability::range`, `ability::on_cooldown(...)`) — scoring-only surface; \
                 not visible to mask predicates"
                    .into(),
            span: e.span,
        }),

        // Plan ToM Task 8 — belief reads in mask bodies. Mask predicates
        // may legitimately read belief state (e.g. filter targets by known
        // hp). Allow through — recurse into sub-expressions; the mask
        // emitter's catch-all returns Unsupported until T9.
        IrExpr::BeliefsAccessor { observer, target, .. } => {
            validate_mask_body(mask_name, observer)?;
            validate_mask_body(mask_name, target)
        }
        IrExpr::BeliefsConfidence { observer, target } => {
            validate_mask_body(mask_name, observer)?;
            validate_mask_body(mask_name, target)
        }
        IrExpr::BeliefsView { observer, .. } => validate_mask_body(mask_name, observer),

        IrExpr::Raw(_) => Ok(()),
    }
}

fn validate_scoring_body(e: &IrExprNode) -> Result<(), ResolveError> {
    match &e.kind {
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::EnumVariant { .. }
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. } => Ok(()),

        IrExpr::NamespaceCall { args, .. } | IrExpr::BuiltinCall(_, args) => {
            for a in args {
                validate_scoring_body(&a.value)?;
            }
            Ok(())
        }

        IrExpr::Quantifier { iter, body, .. } => {
            validate_scoring_body(iter)?;
            validate_scoring_body(body)
        }
        IrExpr::Fold { iter, body, .. } => {
            if let Some(i) = iter {
                validate_scoring_body(i)?;
            }
            validate_scoring_body(body)
        }

        IrExpr::ViewCall(_, args) => {
            for a in args {
                validate_scoring_body(&a.value)?;
            }
            Ok(())
        }
        IrExpr::View(_) => Ok(()),

        IrExpr::VerbCall(_, args) => {
            for a in args {
                validate_scoring_body(&a.value)?;
            }
            Ok(())
        }
        IrExpr::Verb(_) => Ok(()),

        IrExpr::UnresolvedCall(_, args) => {
            for a in args {
                validate_scoring_body(&a.value)?;
            }
            Ok(())
        }

        IrExpr::Field { base, .. } => validate_scoring_body(base),
        IrExpr::Index(a, b) => {
            validate_scoring_body(a)?;
            validate_scoring_body(b)
        }
        IrExpr::Tuple(xs) | IrExpr::List(xs) => {
            for x in xs {
                validate_scoring_body(x)?;
            }
            Ok(())
        }

        IrExpr::Binary(_, lhs, rhs) | IrExpr::In(lhs, rhs) | IrExpr::Contains(lhs, rhs) => {
            validate_scoring_body(lhs)?;
            validate_scoring_body(rhs)
        }
        IrExpr::Unary(_, rhs) => validate_scoring_body(rhs),

        IrExpr::If { cond, then_expr, else_expr } => {
            validate_scoring_body(cond)?;
            validate_scoring_body(then_expr)?;
            if let Some(eb) = else_expr {
                validate_scoring_body(eb)?;
            }
            Ok(())
        }

        IrExpr::StructLit { fields, .. } => {
            for f in fields {
                validate_scoring_body(&f.value)?;
            }
            Ok(())
        }
        IrExpr::Ctor { args, .. } => {
            for a in args {
                validate_scoring_body(a)?;
            }
            Ok(())
        }

        IrExpr::PerUnit { expr, delta } => {
            validate_scoring_body(expr)?;
            validate_scoring_body(delta)
        }

        IrExpr::Match { .. } => Err(ResolveError::UdfInScoringBody {
            offending_construct: "`match` expression (use if/else or gradient terms)".into(),
            span: e.span,
        }),

        // GPU ability evaluation Phase 2 primitives. Legal inside
        // `per_ability` rows (the currently-scored ability is the
        // row's implicit binder); inside a standard row the primitive
        // has no meaningful binder. We permit it here at the resolve
        // stage — Phase 3's CPU emitter decides which row shape it's
        // in and errors if misused — so the DSL-compiler-level
        // validator stays permissive for Phase 2.
        //
        // `AbilityOnCooldown` carries a nested slot expression; walk
        // it too so an illegal subexpression (e.g. a `match`) inside
        // the slot argument still surfaces.
        IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange => Ok(()),
        IrExpr::AbilityOnCooldown(slot) => validate_scoring_body(slot),

        // Plan ToM Task 8 — belief reads in scoring bodies. A scoring row
        // may legitimately condition on known enemy HP etc. Allow through;
        // the scoring emitter's catch-all returns Unsupported until T9.
        IrExpr::BeliefsAccessor { observer, target, .. } => {
            validate_scoring_body(observer)?;
            validate_scoring_body(target)
        }
        IrExpr::BeliefsConfidence { observer, target } => {
            validate_scoring_body(observer)?;
            validate_scoring_body(target)
        }
        IrExpr::BeliefsView { observer, .. } => validate_scoring_body(observer),

        IrExpr::Raw(_) => Ok(()),
    }
}
