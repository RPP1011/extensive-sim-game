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

use crate::ast::{self, ActionHeadShape, AssertExpr, Decl, Expr, ExprKind, Program, Span, Stmt};
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
        symbols.builtins.insert("saturating_add".into(), Builtin::SaturatingAdd);

        // Typed namespaces. Each has its own field / method schema below.
        for (name, id) in [
            ("world", NamespaceId::World),
            ("cascade", NamespaceId::Cascade),
            ("event", NamespaceId::Event),
            ("mask", NamespaceId::Mask),
            ("action", NamespaceId::Action),
            ("rng", NamespaceId::Rng),
            ("query", NamespaceId::Query),
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
            // Singular alias for `abilities`. Added 2026-04-22 (ability-
            // cooldowns subsystem, Task 7) so designers can write
            // `ability::on_cooldown(<slot>)` in mask / physics predicates
            // with the natural singular form. Shares the same method
            // schema as `abilities::`.
            ("ability", NamespaceId::Abilities),
            // Roadmap §1 — Memberships. Grammar stub (no runtime state
            // yet); predicates return bool and emitters return
            // `Unsupported`. See `docs/superpowers/roadmap.md:161-211`.
            ("membership", NamespaceId::Membership),
            // Roadmap §3 — Relationships. Grammar stub (no runtime state
            // yet). See `docs/superpowers/roadmap.md:279-311`.
            ("relationship", NamespaceId::Relationship),
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
            (NamespaceId::Rng, "uniform_int") => Some((2, IrType::I32)),
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
            // blocks" reading). Added 2026-04-22 (ability-cooldowns
            // subsystem, Task 7). The implicit subject is the rule's
            // `self`; the slot arg coerces to `u8` via the argument
            // lowering in the emitter.
            (NamespaceId::Abilities, "on_cooldown") => Some((1, IrType::Bool)),
            (NamespaceId::Abilities, "hostile_only") => Some((1, IrType::Bool)),
            (NamespaceId::Abilities, "range") => Some((1, IrType::F32)),
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
            _ => None,
        }
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
    pub invariants: HashMap<String, InvariantRef>,
    pub probes: HashMap<String, ProbeRef>,
    pub metrics: HashMap<String, MetricRef>,
    /// `config` block name → `(ConfigRef, field-name → field-type)`. Populated
    /// in pass 1 so pass-2 body lowering can resolve `config.<block>.<field>`
    /// into a typed `NamespaceField { ns: Config, field: "<block>.<field>" }`.
    pub configs: HashMap<String, (ConfigRef, HashMap<String, IrType>)>,
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
                comp.events.push(EventIR {
                    name: d.name.clone(),
                    fields: Vec::new(),
                    tags: tag_refs,
                    annotations: non_tag_anns,
                    span: d.span,
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
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::View(d) => {
                check_dup(symbols, "view", &d.name, d.span, |s| s.views.contains_key(&d.name))?;
                let idx = push_idx(comp.views.len(), "view")?;
                symbols.views.insert(d.name.clone(), ViewRef(idx));
                symbols.record_first("view", &d.name, d.span);
                comp.views.push(ViewIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    return_ty: IrType::Unknown,
                    body: ViewBodyIR::Expr(IrExprNode { kind: IrExpr::LitBool(true), span: d.span }),
                    annotations: d.annotations.clone(),
                    kind: ViewKind::Lazy,
                    decay: None,
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
                    emits: Vec::new(),
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
            Decl::Query(_) => {
                // Queries are not a milestone-1a surface yet. Skip silently;
                // 1b will handle.
            }
        }
    }
    Ok(())
}

fn push_idx(len: usize, kind: &'static str) -> Result<u16, ResolveError> {
    u16::try_from(len).map_err(|_| ResolveError::TooManyDecls { kind })
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
                // predicates compile to GPU boolean kernels; task 155
                // (commit 9ba805c6) kept this restriction intentional
                // even as physics bodies gained `for`/`match`.
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
                comp.scoring[scoring_idx].entries = entries;
                scoring_idx += 1;
            }
            Decl::View(d) => {
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let return_ty = resolve_type(&d.return_ty, symbols);
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
                let decay = lower_decay_hint(&d.annotations, &d.body)?;
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
                let emits = d
                    .emits
                    .iter()
                    .map(|e| resolve_emit(e, &mut scope, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                let scoring = d
                    .scoring
                    .as_ref()
                    .map(|e| resolve_expr(e, &mut scope, symbols))
                    .transpose()?;
                comp.verbs[verb_idx].params = params;
                comp.verbs[verb_idx].action = action;
                comp.verbs[verb_idx].when = when;
                comp.verbs[verb_idx].emits = emits;
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
                    // (scalar metrics), `max_bin` (histograms). Bind both as
                    // Unknown so 1b can specialize.
                    let mut alert_scope = LocalScope::new();
                    alert_scope.bind("value", IrType::Unknown);
                    alert_scope.bind("max_bin", IrType::Unknown);
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
        Stmt::Expr(e) => Ok(IrStmt::Expr(resolve_expr(e, scope, symbols)?)),
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
        ExprKind::Binary { op, lhs, rhs } => IrExpr::Binary(
            *op,
            Box::new(resolve_expr(lhs, scope, symbols)?),
            Box::new(resolve_expr(rhs, scope, symbols)?),
        ),
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

/// Walk the view's annotations; if a `@decay(rate=R, per=tick)` annotation
/// exists, validate it and return a typed `DecayHint`. Validates:
///
/// - Paired with `@materialized` (errors otherwise — v1 only supports
///   anchor-pattern decay on event-folded views).
/// - Host body is a `Fold` (lazy views have no persistent state to decay).
/// - `rate` argument is a float literal in the open interval `(0.0, 1.0)`.
/// - `per` argument is the identifier `tick`. Other time bases are parsed
///   but rejected here.
/// - No extra unknown keys.
fn lower_decay_hint(
    annotations: &[ast::Annotation],
    body: &ast::ViewBody,
) -> Result<Option<DecayHint>, ResolveError> {
    let ann = match annotations.iter().find(|a| a.name == "decay") {
        Some(a) => a,
        None => return Ok(None),
    };

    // Must coexist with `@materialized`.
    let has_materialized = annotations.iter().any(|a| a.name == "materialized");
    if !has_materialized {
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
            other => {
                return Err(ResolveError::InvalidDecayHint {
                    detail: format!(
                        "unknown `@decay` argument `{other}`; expected `rate` and `per`"
                    ),
                    span: arg.span,
                });
            }
        }
    }

    let rate = rate.ok_or_else(|| ResolveError::InvalidDecayHint {
        detail: "missing required argument `rate`".into(),
        span: ann.span,
    })?;
    let per = per.ok_or_else(|| ResolveError::InvalidDecayHint {
        detail: "missing required argument `per`".into(),
        span: ann.span,
    })?;

    if !(rate > 0.0 && rate < 1.0) || !rate.is_finite() {
        return Err(ResolveError::InvalidDecayHint {
            detail: format!(
                "`rate` must be a finite float in the open interval (0.0, 1.0); got {rate}"
            ),
            span: ann.span,
        });
    }

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

    Ok(Some(DecayHint {
        rate: rate as f32,
        per: per_unit,
        span: ann.span,
    }))
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

fn validate_fold_stmt(view_name: &str, s: &IrStmt) -> Result<(), ResolveError> {
    match s {
        IrStmt::Let { value, .. } => validate_fold_expr(view_name, value),
        IrStmt::SelfUpdate { op, value, span } => {
            if !matches!(op.as_str(), "=" | "+=" | "-=" | "*=" | "/=") {
                return Err(ResolveError::UdfInViewFoldBody {
                    view_name: view_name.to_string(),
                    offending_construct: format!("self-update operator `{op}`"),
                    span: *span,
                });
            }
            validate_fold_expr(view_name, value)
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
        IrStmt::Emit(IrEmit { span, .. }) => Err(ResolveError::UdfInViewFoldBody {
            view_name: view_name.to_string(),
            offending_construct: "`emit` inside fold body (only physics cascades emit events)"
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
            validate_physics_body(&p.name, &h.body)?;
        }
    }

    Ok(())
}

/// Recursive walker on resolved physics-handler statements.
pub(crate) fn validate_physics_body(
    physics_name: &str,
    body: &[IrStmt],
) -> Result<(), ResolveError> {
    for s in body {
        validate_physics_stmt(physics_name, s)?;
    }
    Ok(())
}

fn validate_physics_stmt(physics_name: &str, s: &IrStmt) -> Result<(), ResolveError> {
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
        IrStmt::Emit(IrEmit { fields, .. }) => {
            for f in fields {
                validate_physics_expr(physics_name, &f.value)?;
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
                validate_physics_stmt(physics_name, bs)?;
            }
            Ok(())
        }
        IrStmt::If { cond, then_body, else_body, .. } => {
            validate_physics_expr(physics_name, cond)?;
            for ts in then_body {
                validate_physics_stmt(physics_name, ts)?;
            }
            if let Some(eb) = else_body {
                for es in eb {
                    validate_physics_stmt(physics_name, es)?;
                }
            }
            Ok(())
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            validate_physics_expr(physics_name, scrutinee)?;
            for arm in arms {
                for stmt in &arm.body {
                    validate_physics_stmt(physics_name, stmt)?;
                }
            }
            Ok(())
        }
        IrStmt::SelfUpdate { value, .. } => validate_physics_expr(physics_name, value),
        IrStmt::Expr(e) => validate_physics_expr(physics_name, e),
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
        | IrExpr::PerUnit { .. } => Err(ResolveError::NotGpuEmittable {
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
            IrStmt::Let { .. } | IrStmt::SelfUpdate { .. } | IrStmt::Expr(_) => {}
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
            IrStmt::For { body, .. } => {
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
        IrStmt::Expr(e) => expr_references_cascade_ceiling(e),
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
// view calls, with `if/else` as the only control flow. Task 155
// (commit 9ba805c6) expanded the *physics* surface to include `for` and
// `match`, but mask/scoring contexts stayed restricted on purpose — they
// compile to per-row GPU kernels where unbounded iteration and variant
// dispatch aren't available. The validators are expression-only; `for`
// statements can't reach these slots (the parser rejects `for` in
// expression position), so the validators primarily catch `match`
// expressions — the one forbidden shape that *does* parse as an expr.

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

        IrExpr::Raw(_) => Ok(()),
    }
}
