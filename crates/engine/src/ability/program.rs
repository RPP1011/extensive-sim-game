//! `AbilityProgram` IR — the intermediate representation that the ability DSL
//! compiler targets and that the `CastHandler` (Task 9) interprets at runtime.
//!
//! One program = one ability. It describes **how** the cast is delivered
//! (`Delivery`), **where** its effect lands (`Area`), **when** it is allowed
//! to fire (`Gate`), and **what** happens on resolve (a bounded smallvec of
//! `EffectOp` atoms).
//!
//! Design notes:
//! - `EffectOp` is `Copy`, explicit-repr u8 discriminant, and pinned to a
//!   16-byte size budget (see `ability_program_shape.rs`). Keeping it small
//!   matters because the cast-dispatch hot-path iterates the effects smallvec
//!   once per cast; fitting 4 effects in 64 bytes keeps them cache-local.
//! - The Meta effect (`CastAbility`) is what makes the subsystem recursive.
//!   Recursion is bounded by the engine's `MAX_CASCADE_ITERATIONS = 8` —
//!   Task 18 pins the depth budget.
//! - `TransferGold.amount` is `i32` (signed) so debt / refunds round-trip
//!   through the same op. `Inventory.gold` is also `i32` (narrowed 2026-04-22
//!   for GPU atomic compatibility; was i64).

use crate::ability::AbilityId;
use smallvec::SmallVec;

/// Maximum number of effects a single `AbilityProgram` may carry. Bounded so
/// the effects smallvec stays stack-resident and cast-dispatch touches a
/// fixed-size window per cast. Pinned by the size-budget test. Changing this
/// is a schema-hash bump.
pub const MAX_EFFECTS_PER_PROGRAM: usize = 6;

/// Maximum number of ability slots per agent. Governs the inner
/// dimension of `SimState::ability_cooldowns`, which carries a
/// per-(agent, ability-slot) local cooldown cursor alongside the
/// per-agent global GCD in `hot_cooldown_next_ready_tick`.
///
/// Added 2026-04-22 with the ability-cooldowns subsystem. 8 matches
/// the Ability Registry's per-unit slot budget from prior hero
/// templates; raise if/when a subsystem needs more per-agent slots
/// (storage cost is `8 × agent_cap × 4B` for the u32 cursor array).
pub const MAX_ABILITIES: usize = 8;

/// How a cast is delivered to its target.
///
/// MVP ships `Instant` only — the effect applies on the tick the cast resolves.
/// Wave 2 piece 5/6 lands `Method { kind, raw }` capturing the
/// .ability surface's `deliver <method> { params } { body }` block;
/// runtime semantics (multi-tick travel for projectile, persistent
/// zone tick, etc.) wire later via registry-driven dispatch (#125).
///
/// Note: dropped `Copy`/`Eq`/`Hash` derives when `Method` landed —
/// the `raw: String` payload precludes them. `Clone + Debug +
/// PartialEq` still hold.
#[derive(Clone, Debug, PartialEq)]
pub enum Delivery {
    /// Effects resolve in the same tick the cast was decided. Default
    /// for the Wave 1 corpus (Strike, Bleed, etc.) and for any
    /// `.ability` that does not declare a `deliver` block.
    Instant,
    /// `deliver <method> { params } { body }` block from the .ability
    /// surface. Wave 1.4 parser captures the raw source verbatim;
    /// this variant just promotes that capture into engine IR. Apply
    /// handlers (multi-tick travel for projectile, hold-over-time for
    /// beam, persistent zone tick for zone, etc.) wire the runtime
    /// semantics later (deferred infrastructure — task #124 + #125).
    Method {
        kind: DeliveryMethodKind,
        /// Verbatim source slice from `deliver` keyword through the
        /// closing `}` of the body block. Retained for backward
        /// compatibility — the structured `hooks` field below is the
        /// preferred read for new apply paths.
        raw:  String,
        /// Wave 2 piece 5/6 follow-up (#139): structured per-hook
        /// effect lists. Each entry corresponds to one
        /// `<hook_ident> { … }` stanza in the deliver body
        /// (`on_hit`, `on_tick`, `on_arrival`, `on_complete`,
        /// `on_kill`, etc.). Apply handlers fire each hook's
        /// effects when the corresponding state-machine signal
        /// resolves (projectile hits, channel ticks, etc.).
        ///
        /// Inner-effect modifiers (chance/stacking/scaling/etc.) on
        /// hook stmts are silently dropped at lowering today, the
        /// same gotcha as `program.nested_per_effect` — the per-hook
        /// aggregator is a future architectural lift. Authors who
        /// need rich hook-stmt modifiers should pull the effect to
        /// the ability's top level until then.
        hooks: SmallVec<[DeliveryHook; 4]>,
    },
}

/// One `<hook_ident> { … }` entry inside a `deliver` body. `kind` is
/// the engine's enumerated hook vocabulary; `effects` is the lowered
/// effect list for that hook.
#[derive(Clone, Debug, PartialEq)]
pub struct DeliveryHook {
    pub kind:    DeliveryHookKind,
    pub effects: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
}

/// Enumerated hook vocabulary. Matches the LoL-corpus survey of all
/// hook idents found inside deliver bodies. Unknown hook idents
/// surface as `LowerError::UnknownDeliveryHook` at lowering time.
///
/// Numeric discriminants are pinned for future SoA packing; renaming
/// or reordering bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeliveryHookKind {
    OnHit          = 0,  // projectile / chain / tether impact
    OnTick         = 1,  // channel / zone per-tick body
    OnArrival      = 2,  // projectile reaches max range
    OnComplete     = 3,  // channel finishes uninterrupted
    OnTrigger      = 4,  // trap activation
    OnKill         = 5,  // damage step kills the target
    OnDamage       = 6,  // generic on-damage observer
    OnDamageDealt  = 7,  // self deals damage
    OnDamageTaken  = 8,  // self takes damage
    OnDeath        = 9,  // self dies
    OnAutoAttack   = 10, // proc-on-basic-attack passive
    OnAbilityUsed  = 11, // proc-on-ability-cast passive
}

impl DeliveryHookKind {
    /// Parse the hook ident from the .ability surface. Returns `None`
    /// for unknown idents so upstream surfaces UnknownDeliveryHook.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "on_hit"          => Some(Self::OnHit),
            "on_tick"         => Some(Self::OnTick),
            "on_arrival"      => Some(Self::OnArrival),
            "on_complete"     => Some(Self::OnComplete),
            "on_trigger"      => Some(Self::OnTrigger),
            "on_kill"         => Some(Self::OnKill),
            "on_damage"       => Some(Self::OnDamage),
            "on_damage_dealt" => Some(Self::OnDamageDealt),
            "on_damage_taken" => Some(Self::OnDamageTaken),
            "on_death"        => Some(Self::OnDeath),
            "on_auto_attack"  => Some(Self::OnAutoAttack),
            "on_ability_used" => Some(Self::OnAbilityUsed),
            _ => None,
        }
    }
}

/// Spec §9 enumerates six delivery methods. The vocabulary lives in
/// engine so unknown method idents from the .ability surface surface
/// loudly at lowering time rather than being silently dropped.
///
/// Numeric discriminants are pinned so future packing work reads the
/// kind off `PackedAbilityRegistry` cleanly. Renaming or reordering
/// bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeliveryMethodKind {
    Projectile = 0,
    Channel    = 1,
    Zone       = 2,
    Chain      = 3,
    Tether     = 4,
    Trap       = 5,
}

impl DeliveryMethodKind {
    /// Parse the method ident from the `deliver <method> { … }`
    /// surface. Returns `None` for unknown idents so upstream can
    /// surface the original token in a typo diagnostic.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "projectile" => Some(Self::Projectile),
            "channel"    => Some(Self::Channel),
            "zone"       => Some(Self::Zone),
            "chain"      => Some(Self::Chain),
            "tether"     => Some(Self::Tether),
            "trap"       => Some(Self::Trap),
            _ => None,
        }
    }
}

/// Where the effect lands relative to the cast's target.
///
/// MVP ships `SingleTarget`; Plan-2 adds `Cone`, `Circle`, and full AoE.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Area {
    SingleTarget { range: f32 },
}

/// Gate predicates — cast-time filters evaluated by `evaluate_cast_gate`
/// (Task 9). `cooldown_ticks` is the number of ticks after a cast resolves
/// before the caster may cast again; `hostile_only` forces targets to pass
/// `CreatureType::is_hostile_to`; `line_of_sight` reserves a bit that the
/// pathfinding plan wires up later (MVP: unused).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Gate {
    pub cooldown_ticks: u32,
    pub hostile_only:   bool,
    pub line_of_sight:  bool,
}

/// For meta (recursive) effects: which agent does the nested cast target?
/// `Target` forwards the outer cast's target; `Caster` redirects to the
/// caster (e.g. a self-buff that fires alongside a damage op).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TargetSelector {
    Target = 0,
    Caster = 1,
}

/// One effect atom. Cast-dispatch (Task 9) pattern-matches and emits one
/// `Effect*Applied` event per op, handed to a per-op cascade handler in
/// Tasks 10–18.
///
/// Explicit discriminants pin ordinals for the schema hash.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EffectOp {
    // --- Combat (5) ---
    /// Apply `amount` hp damage to the target. Shield absorbs first
    /// (Task 10); overflow lands on hp. Negative `amount` is undefined
    /// (use `Heal` for that direction).
    Damage { amount: f32 } = 0,

    /// Raise target hp by `amount`, clamped at max_hp. Emits the real applied
    /// delta so replays observe post-clamp values.
    Heal { amount: f32 } = 1,

    /// Add `amount` to target's `hot_shield_hp`. Stackable; no decay in MVP.
    Shield { amount: f32 } = 2,

    /// Set `hot_stun_remaining_ticks` on target. Longer stun wins (max rule).
    Stun { duration_ticks: u32 } = 3,

    /// Set `hot_slow_remaining_ticks` and `hot_slow_factor_q8`. Combining
    /// rule: longer-or-stronger wins (see Task 14).
    Slow { duration_ticks: u32, factor_q8: i16 } = 4,

    // --- World (2) ---
    /// Move gold between caster and target via `cold_inventory[slot].gold`.
    /// Signed i32 — debt is allowed; negative `amount` reverses the flow.
    /// Narrowed from i64 on 2026-04-22 for GPU atomic compatibility.
    TransferGold { amount: i32 } = 5,

    /// Adjust the symmetric `state.views.standing` edge between caster and
    /// target by `delta`. Clamped to `[-1000, 1000]` inside the view.
    ModifyStanding { delta: i16 } = 6,

    // --- Meta (1) ---
    /// Re-cast another ability. `selector` picks whose slot becomes the nested
    /// target. Recursion is bounded by `MAX_CASCADE_ITERATIONS = 8`.
    CastAbility { ability: AbilityId, selector: TargetSelector } = 7,

    // --- Control / status verbs (Wave 2 piece 1) ---
    // These mirror `Stun`'s shape exactly: one absolute-tick expiry stored
    // in a per-agent SoA mirror (`hot_<verb>_expires_at_tick`). The kernels
    // that read those mirrors as cast / movement / target gates land in
    // later Wave 2 pieces; this slice only adds the variants + payload
    // packing + DSL lowering arms so authoring sites can compile against
    // the new verbs.
    /// Set `hot_root_expires_at_tick` on target. Longer root wins (max rule).
    /// Runtime semantic: target cannot move while `state.tick < expires_at`.
    Root    { duration_ticks: u32 } = 8,

    /// Set `hot_silence_expires_at_tick` on target. Longer silence wins.
    /// Runtime semantic: target cannot cast abilities while
    /// `state.tick < expires_at`.
    Silence { duration_ticks: u32 } = 9,

    /// Set `hot_fear_expires_at_tick` on target. Longer fear wins.
    /// Runtime semantic: target movement intent flips to flee-from-source
    /// while `state.tick < expires_at`.
    Fear    { duration_ticks: u32 } = 10,

    /// Set `hot_taunt_expires_at_tick` on target. Longer taunt wins.
    /// Runtime semantic: target's target-selector is forced to the cascade's
    /// `source` (carried on the Damaged event) while
    /// `state.tick < expires_at`. The "force-attack-source" coupling
    /// requires a richer payload than `{ duration_ticks }` alone — this
    /// slice ships the duration mirror; a follow-up wave wires the
    /// per-agent `taunt_source: Option<AgentId>` field once the runtime
    /// gate lands.
    Taunt   { duration_ticks: u32 } = 11,

    // --- Movement verbs (Wave 2 piece 2) ---
    // These four variants share a uniform `{ distance: f32 }` payload —
    // 4B + 1B tag, identical encoding to `Damage`/`Heal`/`Shield`. Apply
    // handlers (compute facing direction / away-from-caster / toward-caster
    // vectors and call `set_pos` on the existing `hot_pos` SoA column)
    // land in a follow-up Wave 2 piece. This slice only wires the
    // discriminants + payload packing + DSL lowering arms so authoring
    // sites can compile against the new verbs.
    //
    // No new SoA fields: position is already carried by `hot_pos`; the
    // movement verbs read / write that column via the same `set_pos`
    // setter the existing `dsl_compiler::cg::lower::physics` paths use.
    /// Caster moves up to `distance` along their current facing.
    /// Self-target verb: the cast slot supplies the caster id and the
    /// runtime resolves a unit-vector along the caster's facing then
    /// updates `hot_pos` accordingly.
    Dash      { distance: f32 } = 12,

    /// Caster instantly teleports up to `distance` toward the target.
    /// Targeted verb: clamp at the target's position when
    /// `dist(caster, target) < distance`.
    Blink     { distance: f32 } = 13,

    /// Target is pushed `distance` away from the caster (away vector =
    /// `normalize(target_pos - caster_pos)`). Targeted verb. Runtime
    /// must guard against zero-length vectors when caster and target
    /// occupy the same cell.
    Knockback { distance: f32 } = 14,

    /// Target is pulled `distance` toward the caster (toward vector =
    /// `normalize(caster_pos - target_pos)`). Targeted verb. Runtime
    /// must guard against zero-length vectors when caster and target
    /// occupy the same cell.
    Pull      { distance: f32 } = 15,

    // --- Advanced verbs (Wave 2 piece 3) ---
    // Two simple "advanced" verbs from spec §6 / §6.5 that each carry a
    // single 4B payload (4B + 1B tag, well under the 16B EffectOp size
    // budget). NEITHER variant adds new SoA fields:
    //   * `Execute` reads the existing `hot_hp` / `hot_max_hp` columns and
    //     compares against `hp_threshold`; the apply handler emits a
    //     Defeated/AgentDied event when the predicate trips. No new
    //     per-agent state.
    //   * `SelfDamage` translates 1:1 into a `Damaged{source: caster,
    //     target: caster, amount}` event the existing `ApplyDamage`
    //     chronicle already drains. No new per-agent state.
    // Per-fixture apply handlers are Wave 2 piece N (runtime use); this
    // slice only ships the IR variants + payload packing + DSL lowering.
    /// `execute <hp_threshold>` — instantly kill target if `target.hp <
    /// hp_threshold`. Spec §6.5 "advanced verbs". Apply-handler logic
    /// (compare and emit Defeated) is per-fixture work — Wave 2 piece N.
    Execute    { hp_threshold: f32 } = 16,

    /// `self_damage <amount>` — caster takes `amount` damage. The lowering
    /// emits this verb verbatim; per-fixture handlers translate it to a
    /// Damaged{source: caster, target: caster, amount} event the existing
    /// ApplyDamage chronicle then drains.
    SelfDamage { amount: f32 } = 17,

    // --- Buff verbs (Wave 2 piece 4) ---
    // Two duration-bearing buff variants that each mirror `Slow`'s payload
    // shape exactly: `{ duration_ticks: u32, <factor>_q8: i16 }` (4 + 2 = 6B
    // payload + 1B tag, well under the 16B EffectOp ceiling). Each variant
    // owns a per-agent SoA slot (`hot_lifesteal_*` / `hot_damage_taken_*`)
    // that the apply handler will populate with the project's
    // max-with-duration-tiebreak rule (see
    // `~/.claude/projects/-home-ricky-Projects-game/memory/project_buff_stacking_rule.md`):
    // single per-agent slot — incoming wins iff strictly greater magnitude
    // OR equal-magnitude with longer-remaining duration. Apply handlers
    // are per-fixture work in a follow-up Wave 2 piece; this slice ships
    // the IR variants + payload packing + DSL lowering only.
    /// `lifesteal <fraction> <duration>` — caster heals for `fraction` of
    /// damage dealt over the next `duration` window. `fraction` is lowered
    /// to q8 fixed-point (`1.0 == 256`). Apply-handler stacking semantics:
    /// max-with-duration-tiebreak per project_buff_stacking_rule memo
    /// (single per-agent SoA slot — incoming wins iff strictly greater
    /// fraction OR equal-fraction with longer remaining).
    LifeSteal    { duration_ticks: u32, fraction_q8: i16 } = 18,

    /// `damage_modify <multiplier> <duration>` — multiplier applied to
    /// damage TAKEN by the target over `duration`. q8 fixed-point: `1.0
    /// == 256`, `0.5` (take half damage) `== 128`, `1.5` (take 50% extra)
    /// `== 384`. Same max-with-tiebreak stacking as `LifeSteal`.
    DamageModify { duration_ticks: u32, multiplier_q8: i16 } = 19,

    /// `damage <amount> for <duration>` — DoT (damage-over-time).
    /// Applies `amount` to the target every tick for `duration_ticks`
    /// total. The total damage delivered is `amount * duration_ticks`,
    /// allowing authors to think in instantaneous-rate terms (e.g.
    /// `damage 5 for 2s` = 5 dmg/tick × 20 ticks = 100 dmg total).
    /// Apply handlers fire one Damaged event per tick over the window
    /// (deferred — DoT scheduler not wired).
    DamageOverTime { amount: f32, duration_ticks: u32 } = 20,

    /// `heal <amount> for <duration>` — HoT (heal-over-time). Mirror
    /// of `DamageOverTime` but for healing. Per-tick `amount` applied
    /// to the target's HP for `duration_ticks` (deferred apply
    /// handler — HoT scheduler not wired).
    HealOverTime { amount: f32, duration_ticks: u32 } = 21,

    /// `shield <amount> for <duration>` — timed shield. Same as
    /// `Shield` but the absorption pool auto-expires after
    /// `duration_ticks`. Apply handlers schedule a shield-clear
    /// chronicle at `cast_tick + duration_ticks` (deferred — shield
    /// expiry scheduler not wired). Distinct from `Shield` which is
    /// permanent until consumed.
    TimedShield { amount: f32, duration_ticks: u32 } = 22,

    /// `buff <stat> <magnitude> for <duration>` — temporary stat boost.
    /// `magnitude_q8` is q8 fixed-point matching the Slow/DamageModify
    /// convention (1.0 → 256; `+ 30%` → 76; `+ 100%` → 256). Apply
    /// handlers schedule a stat-clear chronicle at
    /// `cast_tick + duration_ticks` (deferred — buff scheduler not
    /// wired). 12 bytes with tag, well under the P4 ≤16-byte budget.
    Buff { stat: BuffStat, magnitude_q8: i16, duration_ticks: u32 } = 23,

    /// `summon "<template>" [N] [for <duration>]` — spawn `count` minions
    /// of the named template for `lifetime_ticks`. The template ident is
    /// stored as a 32-bit FxHash (deferred resolution — engine doesn't
    /// need to know template names; runtime side resolves the hash to a
    /// spawner via a registry follow-up). `count == 0` means the parser
    /// did not specify one (apply handlers default to 1);
    /// `lifetime_ticks == 0` means no duration was authored (handlers
    /// pick a sensible default, e.g. permanent until owner dies). Payload
    /// is 4 + 1 + 4 = 9 bytes + tag = 10 bytes; well under the P4 ≤16B
    /// EffectOp size budget. No apply handler in this slice — the LoL
    /// corpus needs the verb to lower so the per-hero programs build.
    Summon { template_hash: u32, count: u8, lifetime_ticks: u32 } = 24,

    // --- Non-combat verbs phase 1 (world primitives) ---
    // Two world-primitive verbs that finally cross the engine's existing
    // event boundary into the non-combat catalog (`AgentHarvested = 9`,
    // `AgentPlacedVoxel = 11`, `AgentHarvestedVoxel = 12`). Both share
    // the `Summon` shape's deferred-resolution discipline: the resource /
    // voxel ident from the .ability surface is FxHashed to a u32 so the
    // engine payload stays compact and authoritative resource resolution
    // happens on the runtime side via a registry follow-up.
    //
    // Payload budgets:
    //   * Harvest     — `kind_hash: u32 + amount: u16` = 6B + 1B tag = 7B.
    //   * PlaceVoxel  — `kind_hash: u32`               = 4B + 1B tag = 5B.
    // Both fit easily under the P4 ≤16-byte EffectOp ceiling.
    //
    // The engine ships THREE non-combat events but only TWO new
    // EffectOps: `Harvest` covers BOTH `AgentHarvested` (organic / surface
    // resource — berries, hides, fish) AND `AgentHarvestedVoxel` (ore /
    // stone / wood block). Apply handlers route the dispatch by looking
    // up `kind_hash` in the resource registry — voxel resources emit
    // `AgentHarvestedVoxel`, surface resources emit `AgentHarvested`. A
    // single EffectOp with split apply-side dispatch lets the .ability
    // surface stay terse (`harvest "berries"` / `mine "iron"`) without
    // forcing authors to know whether the resource is voxel-backed.
    //
    // The DSL surface exposes `mine "<kind>"` as an alias for
    // `harvest "<kind>"` purely for authoring ergonomics — both lower to
    // `EffectOp::Harvest`. The kind_hash distinguishes the resource via
    // the runtime registry side.
    /// `harvest "<kind>" [<amount>]` — caster gathers `amount` units of
    /// the named resource. `kind_hash` is the FxHash of the resource
    /// ident from the .ability source (deferred resolution — apply
    /// handlers map the hash to a concrete resource via a registry
    /// follow-up). Same pattern as `Summon::template_hash`. `amount`
    /// defaults to 1 when omitted in the DSL surface; u16 ceiling is
    /// 65535 units per cast — well past any sane single-tick gather
    /// rate. Apply handlers emit `AgentHarvested` for organic / surface
    /// resources and `AgentHarvestedVoxel` for voxel-backed resources;
    /// the runtime registry's lookup decides which.
    Harvest    { kind_hash: u32, amount: u16 } = 25,

    /// `place_voxel "<kind>"` — caster places one voxel of the named
    /// kind at the cast's target position. `kind_hash` is the FxHash of
    /// the voxel ident from the .ability source (deferred resolution —
    /// apply handlers map the hash to a concrete voxel kind via a
    /// registry follow-up). Same pattern as `Summon::template_hash`.
    /// Apply handlers emit `AgentPlacedVoxel` and write the voxel into
    /// world state. No amount — voxel placement is one-per-cast (multi-
    /// voxel constructions compose multiple `place_voxel` ops or wire a
    /// future area modifier).
    PlaceVoxel { kind_hash: u32 } = 26,

    /// `stealth for <duration> [break_on_damage]` — self-cast invisibility
    /// for `duration_ticks`. The LoL corpus shape is uniformly
    /// `stealth for 3s break_on_damage` (25 hero files: Akali, Diana,
    /// Khazix, MonkeyKing, Rengar, Talon, Vayne, Akshan, Aurora, …). No
    /// magnitude argument — stealth is a binary state. The
    /// `break_on_damage` lifetime modifier rides the existing per-effect
    /// lifetime SoA (LifetimeMode::BreakOnDamage); apply handlers tie
    /// the stealth flag to the lifetime expiry the same way TimedShield
    /// does. Payload is 4 bytes + 1 tag = 5 bytes; well under the P4
    /// ≤16-byte ceiling. Apply handler is deferred — stealth flag SoA
    /// + targeting-mask exclusion arrives with the registry-driven
    /// dispatch (#125 family).
    Stealth { duration_ticks: u32 } = 27,

    // ---- Wave 2 piece 8 — remaining LoL-corpus CC vocabulary. ----
    // Same single-`<duration>` shape as the Wave 2 piece 1 control
    // verbs (root/silence/fear/taunt). All target-cast; apply
    // handlers wire per-agent expiry tick-stamps the same way Stun
    // does. Payloads are 4 bytes + 1 tag = 5 bytes.
    /// `charm <duration>` — Ahri-style "target moves toward caster"
    /// CC. Combines movement-control + brief flee-suppression in
    /// LoL; for the engine a single duration field is enough — apply
    /// handlers later layer the move-toward semantics. 1 LoL file.
    Charm    { duration_ticks: u32 } = 28,
    /// `grounded <duration>` — Singed/Cassiopeia/Caitlyn AOE that
    /// disables dashes/blinks for the duration. Distinct from
    /// `Root` (which immobilizes against ALL movement); target keeps
    /// walk speed. 6 LoL files.
    Grounded { duration_ticks: u32 } = 29,
    /// `suppress <duration>` — Warwick/Malzahar/Skarner/Sett/Ambessa
    /// hard-CC that disables ALL actions and ignores tenacity. Apply
    /// handlers may treat as a stronger Stun (no tenacity diminish).
    /// 5 LoL files.
    Suppress { duration_ticks: u32 } = 30,
    /// `reflect <fraction> for <duration>` — Mel-style damage-bounce
    /// passive: incoming damage scales by `(1 - fraction)` and
    /// `fraction` is mirrored back to the source. `fraction_q8` is
    /// q8 fixed-point matching the Slow/DamageModify convention
    /// (0.3 → 76; 1.0 → 256). Mirrors `DamageModify`'s payload
    /// shape; apply handlers can layer on top of the existing
    /// `DamageModify` infrastructure. 1 LoL file (Mel).
    Reflect  { duration_ticks: u32, fraction_q8: i16 } = 31,
}

/// Stat targeted by `buff`. Vocabulary is small today (just the two
/// LoL-corpus uses); extend as more apply paths land. Stable repr(u8)
/// for future SoA packing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BuffStat {
    MoveSpeed   = 0,
    AttackSpeed = 1,
}

impl BuffStat {
    /// Parse the stat ident from the .ability surface. Unknown idents
    /// return None; lowering surfaces them as UnknownStatRef.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "move_speed"   => Some(Self::MoveSpeed),
            "attack_speed" => Some(Self::AttackSpeed),
            _ => None,
        }
    }
}

/// Coarse ability-category hint, per `.ability` DSL `hint:` field.
///
/// Exposes one hint per ability to scoring — scoring rows read the hint
/// via the DSL grammar addition `ability::hint` (landing in Phase 2 of
/// the GPU ability-evaluation subsystem). The sentinel `None` is used
/// for abilities authored without a `hint:` line; scoring expressions
/// that compare against a specific hint treat `None` as not-a-match.
///
/// Numeric discriminants are pinned so GPU packing (`PackedAbilityRegistry::hints`)
/// and WGSL `const` comparisons align without a runtime lookup. Renaming or
/// reordering variants bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AbilityHint {
    Damage = 0,
    Defense = 1,
    CrowdControl = 2,
    Utility = 3,
    /// Heal-shaped abilities (Mend / Lifeweaver heals / etc.). Distinct
    /// from Defense because scoring weights for heals (target ally with
    /// missing hp) and defenses (self-protect with shields) diverge —
    /// AI evaluators that bucket on hint were silently miscategorizing
    /// heals as defenses prior to #142.
    Heal = 4,
    /// Stat-buff abilities (move_speed / attack_speed / damage_modify
    /// boosts). Distinct from Utility because buffs are duration-bound
    /// stat boosts on a chosen target while Utility is a catchall for
    /// teleport / vision / cleanse / dispel — different scoring axes.
    Buff = 5,
}

impl AbilityHint {
    /// Parse the coarse category from its DSL token form (`damage`,
    /// `defense`, `crowd_control`, `utility`, `heal`, `buff`). Returns
    /// `None` for an unknown spelling so upstream can surface the
    /// original token in its error.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "damage" => Some(Self::Damage),
            "defense" => Some(Self::Defense),
            "crowd_control" => Some(Self::CrowdControl),
            "utility" => Some(Self::Utility),
            "heal" => Some(Self::Heal),
            "buff" => Some(Self::Buff),
            _ => None,
        }
    }

    /// Stable discriminant, matching the `#[repr(u8)]` ordinal. Used by
    /// the GPU packer to drop `Option<AbilityHint>` into a `u32` slot
    /// alongside `HINT_NONE_SENTINEL`.
    #[inline]
    pub fn discriminant(self) -> u8 {
        self as u8
    }
}

/// Fixed initial tag vocabulary for per-effect `[TAG: value]` power
/// ratings surfaced through the `.ability` DSL.
///
/// v1 ships a fixed enum (per the spec's "fixed enum for v1" decision
/// in `docs/spec/engine.md §11`
/// "Open questions"). A user-extensible symbol table is deferred — the
/// fixed enum lowers each tag to a known GPU buffer index without a
/// per-scenario rebind.
///
/// Numeric discriminants double as the column index into the packed
/// `tag_values` buffer (`tag_values[ab * NUM_TAGS + tag as usize]`).
/// Renaming or reordering bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AbilityTag {
    Physical = 0,
    Magical = 1,
    CrowdControl = 2,
    Heal = 3,
    Defense = 4,
    Utility = 5,
}

impl AbilityTag {
    /// Total count — pinned to match the `NUM_ABILITY_TAGS` stride used
    /// by `PackedAbilityRegistry::tag_values`. Bump in lockstep with any
    /// enum addition.
    pub const COUNT: usize = 6;

    /// Iterate every variant in declaration order. Useful for packers
    /// that need to fill a fixed-width row of per-tag values.
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::Physical,
            Self::Magical,
            Self::CrowdControl,
            Self::Heal,
            Self::Defense,
            Self::Utility,
        ]
        .into_iter()
    }

    /// Parse the tag from its DSL token form (`PHYSICAL`, `MAGICAL`,
    /// `CROWD_CONTROL`, `HEAL`, `DEFENSE`, `UTILITY`). Returns `None`
    /// for unknown tags so upstream can surface the original token.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            // PHYSICAL covers physical damage. TRUE is LoL's
            // "true damage" — bypasses armor/MR. Closest engine bucket
            // is Physical (both go through the damage scoring path);
            // a dedicated `True` variant would need a new tag column,
            // which apply handlers don't yet differentiate. Alias keeps
            // the LoL corpus parsing without a schema-hash bump.
            "PHYSICAL" | "TRUE" => Some(Self::Physical),
            // MAGIC is an authoring-time alias for the canonical
            // MAGICAL spelling. The LoL corpus uses both; accept
            // either so authors don't need to remember which.
            "MAGICAL" | "MAGIC" => Some(Self::Magical),
            "CROWD_CONTROL" => Some(Self::CrowdControl),
            "HEAL" => Some(Self::Heal),
            "DEFENSE" => Some(Self::Defense),
            "UTILITY" => Some(Self::Utility),
            _ => None,
        }
    }

    /// Column index into the packed `tag_values` row. Matches the
    /// `#[repr(u8)]` ordinal.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Maximum tag entries per `AbilityProgram`. Bounded so the tag
/// smallvec stays stack-resident; `AbilityTag::COUNT` is the natural
/// upper bound since each tag appears at most once per ability.
pub const MAX_TAGS_PER_PROGRAM: usize = AbilityTag::COUNT;

/// Per-effect stacking mode controlling apply-handler behavior when
/// the same effect is dispatched against an agent that already has it
/// active. See `project_buff_stacking_rule.md` for the canonical rule
/// (max-with-tiebreak default for pure single-slot buffs, additive for
/// damage-leaning multipliers).
///
/// Numeric discriminants are pinned so `PackedAbilityRegistry`'s
/// `stackings` column packs cleanly. Renaming or reordering bumps the
/// schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StackingMode {
    /// Re-application resets duration to full magnitude. Default for
    /// effects that didn't carry an explicit `stacking` modifier.
    Refresh = 0,
    /// Each application increments a counter; magnitude scales with
    /// stack count.
    Stack = 1,
    /// New duration = remaining + new (additive duration extension).
    Extend = 2,
}

/// #129 / Wave 1.4 `recast: <int|dur>` payload. Two shapes per spec
/// §4.2:
///   * `Count(n)` — `recast: N` integer form, capping the number of
///     consecutive recasts allowed within the `recast_window_ticks`.
///     Most LoL `.ability` files use this form (Aatrox `recast: 1`,
///     Ahri `recast: 3`, Janna/Darius `recast: 1`).
///   * `CooldownTicks(t)` — `recast: <duration>` form, the
///     ticks-between-recasts cooldown. Rare in the corpus, captured
///     here for completeness with the spec.
///
/// Numeric discriminants are pinned so a future SoA packing column
/// reads cleanly. Renaming or reordering bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RecastKind {
    /// `recast: N` — max consecutive recasts inside the window.
    Count(u32),
    /// `recast: <duration>` — recast cooldown in ticks (100 ms).
    CooldownTicks(u32),
}

/// Per-effect lifetime modifier — when does this effect end?
///
/// `UntilCasterDies` ties the effect to the caster's `hot_alive` flag;
/// `DamageableHp(N)` is a separate HP pool that depletes as the effect
/// absorbs damage (voxel-style); `BreakOnDamage` ends on the next
/// caster-damage event (LoL-style stealth pattern — Akali / Elise /
/// MonkeyKing all use the "stealth-for-3s, break_on_damage" idiom).
///
/// The discriminant is exposed via [`LifetimeMode::discriminant`] so
/// the SoA packer can drop it into a `u8` column. Variant ordinals
/// (0/1/2) are pinned by `crates/engine/src/schema_hash.rs` (the
/// `LifetimeMode:` line); reordering or renaming any variant bumps
/// the schema hash.
///
/// `DamageableHp` carries a payload (initial hp), which is the first
/// per-effect SoA column we have with variant data — see
/// `PackedAbilityRegistry::lifetime_payloads`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LifetimeMode {
    /// `until_caster_dies` — apply handlers tear the effect down when
    /// `hot_alive[caster] == false`.
    UntilCasterDies,
    /// `damageable_hp(N)` — the effect carries its own HP pool of
    /// initial value `N`; apply handlers decrement it as the effect
    /// absorbs damage and tear the effect down at zero.
    DamageableHp(f32),
    /// `break_on_damage` — apply handlers tear the effect down on the
    /// next `Damaged{target=caster, ...}` event the caster receives.
    BreakOnDamage,
}

impl LifetimeMode {
    /// Stable u8 discriminant for the SoA `lifetime_kinds` column.
    /// Pinned by `crates/engine/src/schema_hash.rs` (the
    /// `LifetimeMode:` line). Apply handlers + the GPU dispatch
    /// kernel switch on this value to pick the lifetime semantic.
    #[inline]
    pub fn discriminant(self) -> u8 {
        match self {
            LifetimeMode::UntilCasterDies => 0,
            LifetimeMode::DamageableHp(_) => 1,
            LifetimeMode::BreakOnDamage   => 2,
        }
    }

    /// Payload value for the SoA `lifetime_payloads` column. Only
    /// `DamageableHp` carries a meaningful value — the other two
    /// variants pack `0.0` so the column is dense + has no gap.
    #[inline]
    pub fn payload_f32(self) -> f32 {
        match self {
            LifetimeMode::DamageableHp(hp) => hp,
            _ => 0.0,
        }
    }
}

/// Per-effect area-of-effect shape — the modifier `in <shape>(args)`.
/// Spec §8 catalog: 5 disc-family (1-voxel default thickness) + 7
/// volume-family. Numeric discriminants pinned for PackedAbilityRegistry
/// packing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShapeKind {
    // Disc family (spec §8.1)
    Circle   = 0,   // circle(r)
    Cone     = 1,   // cone(r, angle_deg)
    Line     = 2,   // line(len, width)
    Ring     = 3,   // ring(inner, outer)
    Spread   = 4,   // spread(r, max_targets)
    // Volume family (spec §8.2)
    Box      = 5,   // box(wx, wy, wz)
    Sphere   = 6,   // sphere(r)
    Column   = 7,   // column(r, h)
    Wall     = 8,   // wall(len, h, thick) [facing: deg] — 4 args
    Cylinder = 9,   // cylinder(r, h)
    Dome     = 10,  // dome(r)
    Hull     = 11,  // hull(r)
}

impl ShapeKind {
    /// Parse from the source-form name. Returns None for unknown shapes.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "circle"   => Some(ShapeKind::Circle),
            "cone"     => Some(ShapeKind::Cone),
            "line"     => Some(ShapeKind::Line),
            "ring"     => Some(ShapeKind::Ring),
            "spread"   => Some(ShapeKind::Spread),
            "box"      => Some(ShapeKind::Box),
            "sphere"   => Some(ShapeKind::Sphere),
            "column"   => Some(ShapeKind::Column),
            "wall"     => Some(ShapeKind::Wall),
            "cylinder" => Some(ShapeKind::Cylinder),
            "dome"     => Some(ShapeKind::Dome),
            "hull"     => Some(ShapeKind::Hull),
            _ => None,
        }
    }

    /// Stable u8 discriminant matching the `#[repr(u8)]` ordinal.
    #[inline]
    pub fn discriminant(self) -> u8 { self as u8 }
}

/// Per-effect AOE shape with up to 4 f32 args. `Wall` takes 4
/// (len/h/thick/facing); other shapes use fewer and zero-pad. `args`
/// is parallel to the source-order positional args of the shape's
/// constructor in spec §8.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EffectAreaShape {
    pub kind: ShapeKind,
    pub args: [f32; 4],
}

/// `target: <mode>` header (spec §4.3). The eight modes the spec
/// table lists, captured here so apply handlers know how to bind the
/// cast's target arity (single agent, ground point, direction vector,
/// global no-target, etc.). Mirrors `dsl_ast::ast::TargetMode` 1:1.
///
/// Today's apply path only reads `Enemy`/`Self_`/`Ally` (the
/// per-pair-targeted modes); the position/directional/global modes
/// are captured at the IR boundary so the corpus lowers, with apply
/// dispatch arriving alongside registry-driven dispatch (#125).
///
/// Numeric discriminants are pinned for future SoA packing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TargetModeKind {
    Enemy     = 0,
    SelfCast  = 1,  // `Self_` in the AST; `self` is a Rust keyword
    Ally      = 2,
    SelfAoe   = 3,
    Ground    = 4,
    Direction = 5,
    Vector    = 6,
    Global    = 7,
}

/// Top-level `cost: <amount> <resource>` header (spec §4.2). Captured
/// here so apply handlers can debit the resource at cast-decide time
/// (deferred infrastructure today). The resource vocabulary is fixed:
/// mana / stamina / hp / gold.
///
/// Stored on `AbilityProgram.cost`; `None` means the .ability did not
/// declare a cost header. Wave 1 corpus shape (Strike / ShieldUp /
/// Mend / etc.) leaves it `None` — keeps existing program output
/// bit-stable.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AbilityCost {
    pub resource: CostResource,
    pub amount:   CostAmount,
}

/// Resource a `cost:` header debits from. Mirrors `dsl_ast::ast::CostResource`
/// but lives in engine so apply handlers don't pull a dsl_ast dep.
/// Numeric discriminants pinned for future packing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CostResource {
    Mana    = 0,
    Stamina = 1,
    Hp      = 2,
    Gold    = 3,
}

/// Magnitude of a `cost:` header — flat scalar or percent of the
/// resource's max. Mirrors `dsl_ast::ast::CostAmount`. The percent
/// scalar uses the Wave 1 convention (`25% mana` stores `25.0`, NOT
/// `0.25`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CostAmount {
    Flat(f32),
    PercentOfMax(f32),
}

/// Per-effect conditional gate from the `when <cond> [else <cond>]`
/// modifier (spec §6.1 slot 7). The verbatim `when_cond` source text is
/// retained for diagnostics; `when_compiled` carries the structured
/// predicate apply / GPU dispatch evaluates at runtime.
///
/// `when_cond` is the always-required positive predicate (verbatim source).
/// `else_cond` is the optional fallback — REJECTED at lower time today
/// (deferred — open task #163-followup). `when_compiled` is the lowered
/// structured form, populated when the predicate matches the restricted
/// vocab in [`EffectPredicate`]; `None` indicates lower-time rejection
/// fell back to a "no compiled predicate" state.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectWhenCondition {
    pub when_cond: String,
    pub else_cond: Option<String>,
    pub when_compiled: Option<EffectPredicate>,
}

/// Predicate binder — `self` or `target` from the source-form
/// `<binder>.<field> <op> <literal>`. Numeric discriminants pinned for
/// SoA packing (`when_pred_binder` column); 0 = self, 1 = target.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EffectPredicateBinder {
    SelfBinder = 0,
    Target     = 1,
}

/// Predicate comparison operator — one of `< <= > >= == !=`. Numeric
/// discriminants pinned for SoA packing (`when_pred_op` column).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EffectPredicateOp {
    Lt = 0,
    Le = 1,
    Gt = 2,
    Ge = 3,
    Eq = 4,
    Ne = 5,
}

/// Structured per-effect when-predicate. Restricted vocab:
///   * Form: `<binder>.<field> <op> <literal>`
///   * binder ∈ {self, target}
///   * field: discriminant of [`ScalingStatRef`] (the 8 f32 stat fields
///     already exposed for scaling). The lowering pass maps the source
///     field name through `AgentFieldId::from_snake` for vocabulary
///     validation, then narrows to the ScalingStatRef-shaped subset
///     (CasterStats's get-by-discriminant codomain).
///   * op ∈ {Lt, Le, Gt, Ge, Eq, Ne}
///   * literal: f32 (parser produces LitFloat or LitInt; int promotes).
///
/// `else <cond>`, compound predicates (`&&`/`||`/`!`), and field-vs-field
/// comparisons are REJECTED at lower time (open task #163-followup).
///
/// Doesn't impl `Eq`/`Hash` because of the `f32` literal — that's fine,
/// hashing is at higher granularity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EffectPredicate {
    pub binder:  EffectPredicateBinder,
    /// Discriminant of [`ScalingStatRef`] — 0..=7 maps to AttackDamage /
    /// AbilityPower / MaxHp / Hp / Armor / MagicResist / MoveSpeed /
    /// Mana. Pinned by the schema hash.
    pub field:   u8,
    pub op:      EffectPredicateOp,
    pub literal: f32,
}

// Pin the predicate at ≤8 bytes (1 + 1 + 1 + 1pad + 4 = 8). Any future
// growth (e.g. promoting field to u16 or adding a binder operand) bumps
// the schema hash + this guard.
const _: () = assert!(std::mem::size_of::<EffectPredicate>() <= 8);

/// Per-effect stat-scaling reference for the `+ N% stat_ref` modifier
/// (spec §6.1 slot 8). 8 common combat stats; an unknown stat_ref token
/// errors at lowering time (no enum variant for unknown — surface a
/// LowerError so authors get a typo diagnostic).
///
/// Numeric discriminants are pinned so `PackedAbilityRegistry`'s
/// `scaling_stat_refs` column packs cleanly. Renaming or reordering
/// bumps the schema hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScalingStatRef {
    AttackDamage  = 0,
    AbilityPower  = 1, // "AP"
    MaxHp         = 2,
    Hp            = 3,
    Armor         = 4,
    MagicResist   = 5,
    MoveSpeed     = 6,
    Mana          = 7,
}

impl ScalingStatRef {
    /// Parse from the source-form name. Accepts both spec long-form
    /// and the LoL/MOBA short-form aliases. Returns `None` for unknown.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "attack_damage" | "AD"     => Some(ScalingStatRef::AttackDamage),
            "ability_power" | "AP"     => Some(ScalingStatRef::AbilityPower),
            "max_hp"        | "MaxHP"  => Some(ScalingStatRef::MaxHp),
            "hp"            | "HP"     => Some(ScalingStatRef::Hp),
            "armor"                    => Some(ScalingStatRef::Armor),
            "magic_resist"  | "MR"     => Some(ScalingStatRef::MagicResist),
            "move_speed"               => Some(ScalingStatRef::MoveSpeed),
            "mana"                     => Some(ScalingStatRef::Mana),
            _ => None,
        }
    }

    /// Stable u8 discriminant matching the `#[repr(u8)]` ordinal.
    #[inline]
    pub fn discriminant(self) -> u8 {
        self as u8
    }
}

/// Caster's stat snapshot at cast-decide time. Threaded into
/// [`crate::ability::apply::apply_program`] so per-effect
/// `scalings_per_effect` slots can compute `base + Σ percent * stat`.
///
/// Eight fields parallel `ScalingStatRef`'s vocabulary (one per
/// variant). `Default::default()` is all-zeros — call sites that don't
/// care about scaling (e.g. legacy unit tests) pass `&CasterStats::default()`
/// to keep behavior bit-identical to the pre-scaling apply path
/// (effect amounts pass through unchanged when every stat is zero).
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct CasterStats {
    pub attack_damage: f32,
    pub ability_power: f32,
    pub max_hp:        f32,
    pub hp:            f32,
    pub armor:         f32,
    pub magic_resist:  f32,
    pub move_speed:    f32,
    pub mana:          f32,
}

impl CasterStats {
    /// Look up a stat by `ScalingStatRef`. Apply handlers use this to
    /// project a scaling slot's `stat_ref` to its scalar value before
    /// multiplying by `percent`.
    #[inline]
    pub fn get(&self, stat: ScalingStatRef) -> f32 {
        match stat {
            ScalingStatRef::AttackDamage => self.attack_damage,
            ScalingStatRef::AbilityPower => self.ability_power,
            ScalingStatRef::MaxHp        => self.max_hp,
            ScalingStatRef::Hp           => self.hp,
            ScalingStatRef::Armor        => self.armor,
            ScalingStatRef::MagicResist  => self.magic_resist,
            ScalingStatRef::MoveSpeed    => self.move_speed,
            ScalingStatRef::Mana         => self.mana,
        }
    }

    /// Look up a stat by `ScalingStatRef` discriminant (0..=7). Used
    /// by the per-effect when-predicate evaluator
    /// (`EffectPredicate::field` is a u8 discriminant). Out-of-range
    /// discriminants return `None` (predicate evaluator treats as
    /// false — defensive).
    #[inline]
    pub fn get_by_field_id(&self, discriminant: u8) -> Option<f32> {
        Some(match discriminant {
            0 => self.attack_damage, // AttackDamage
            1 => self.ability_power, // AbilityPower
            2 => self.max_hp,        // MaxHp
            3 => self.hp,            // Hp
            4 => self.armor,         // Armor
            5 => self.magic_resist,  // MagicResist
            6 => self.move_speed,    // MoveSpeed
            7 => self.mana,          // Mana
            _ => return None,
        })
    }
}

/// Per-effect scaling entry — one `+ N% stat_ref` slot. Multiple per
/// effect allowed (e.g. `damage 50 + 30% AP + 20% AD`); the per-effect
/// vector is bounded at `MAX_SCALINGS_PER_EFFECT`.
///
/// `percent` is stored as a fraction of the stat (e.g. `0.30` for
/// "+ 30% AP") so apply handlers can multiply directly without an
/// extra `/ 100.0`. The DSL parser stores `N` (e.g. `30` for `+ 30% AP`);
/// the lowering pass normalises by dividing by `100.0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EffectScaling {
    pub stat_ref: ScalingStatRef,
    pub percent:  f32,
}

/// Max scalings per effect. 2 fits LoL/MOBA convention (rare to scale
/// on more than two stats). Bounded so the per-effect inner SmallVec
/// stays stack-resident; the SoA pack column is sized
/// `n_abilities * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT`.
pub const MAX_SCALINGS_PER_EFFECT: usize = 2;

/// Max nested follow-up effects per outer effect (Wave 1.5#9).
/// Bounded so each `nested_per_effect` slot stays stack-resident in
/// the SmallVec inline buffer. 2 is enough for typical "damage + stun
/// + slow" combo abilities; richer cascades should compose multiple
/// outer effects instead. Overflow panics at lowering time today; a
/// future `LowerError::NestedBudgetExceeded` can soften that.
pub const MAX_NESTED_PER_EFFECT: usize = 2;

/// Compiled ability — the unit `AbilityRegistry` stores and `CastHandler`
/// dispatches.
///
/// Carries the scoring-surface fields (`hint`, `tags`) exposed to GPU
/// ability evaluation via `PackedAbilityRegistry::pack`. Both default
/// to empty so legacy test sites (which construct abilities without a
/// hint or tag vector) keep compiling unchanged — a program with no
/// tags scores 0 under `ability::tag(...)`, matching the spec's
/// "returns 0 silently" contract for missing tags.
#[derive(Clone, Debug)]
pub struct AbilityProgram {
    pub delivery: Delivery,
    pub area:     Area,
    pub gate:     Gate,
    pub effects:  SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
    /// Coarse hint per the `.ability` DSL's `hint:` field. `None` means
    /// the source file did not specify one.
    pub hint:     Option<AbilityHint>,
    /// Per-tag numeric power ratings aggregated across the ability's
    /// effects. One entry per `(tag, value)` pair — lookup is linear;
    /// max length is `MAX_TAGS_PER_PROGRAM == AbilityTag::COUNT`.
    pub tags:     SmallVec<[(AbilityTag, f32); MAX_TAGS_PER_PROGRAM]>,
    /// Wave 1.5#3: per-effect stacking mode. `None` for effects that
    /// didn't carry a `stacking <mode>` modifier — apply handlers
    /// should default to `StackingMode::Refresh` per the
    /// `project_buff_stacking_rule.md` contract. Index parallel to
    /// `effects`; a populated `stackings` slice is either empty (no
    /// effect carried the modifier) or has one slot per effect (None
    /// for the unmarked ones).
    pub stackings: SmallVec<[Option<StackingMode>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#5: per-effect probability gate. Q16 fixed-point:
    /// `0` = 0% (never fires), `65534` ≈ 100% (always fires; one less
    /// than `u16::MAX` so the SoA sentinel `CHANCE_NONE_SENTINEL =
    /// 0xFFFF` stays unambiguous). Apply handlers compare
    /// `per_agent_u32(seed, agent_id, tick, purpose) & 0xFFFF` against
    /// this value to decide whether the effect fires this tick. `None`
    /// for effects that didn't carry a `chance N%` modifier — apply
    /// handlers should default to "always fires" (no RNG gate). Index
    /// parallel to `effects`; a populated `chances` slice is either
    /// empty (no effect carried the modifier) or has one slot per
    /// effect (None for the unmarked ones).
    pub chances: SmallVec<[Option<u16>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#8: per-effect lifetime modifier. `None` for effects
    /// that didn't carry one of the three lifetime modifiers
    /// (`until_caster_dies` / `damageable_hp(N)` / `break_on_damage`)
    /// — apply handlers should treat the slot as "effect persists
    /// indefinitely (or for the verb's own duration if the verb has
    /// one)". Index parallel to `effects`; a populated `lifetimes`
    /// slice is either empty (no effect carried the modifier) or has
    /// one slot per effect (`None` for the unmarked ones).
    /// `LifetimeMode::DamageableHp` is the first per-effect modifier
    /// with variant data (an `f32` hp pool), so the SoA packer
    /// requires TWO columns (`lifetime_kinds` + `lifetime_payloads`).
    pub lifetimes: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#2: per-effect AOE shape from `in <shape>(args)`. `None`
    /// = single-target effect (default — uses program.area). Index
    /// parallel to `effects`; a populated `per_effect_areas` slice is
    /// either empty (no effect carried the modifier) or has one slot
    /// per effect (`None` for the unmarked ones). The SoA packer emits
    /// TWO companion columns (`area_kinds` + `area_args`) — kinds is a
    /// u8 discriminant column with sentinel `SHAPE_KIND_NONE_SENTINEL`
    /// (0xFF) for `None` slots; args is a flat `4×f32` per slot.
    pub per_effect_areas: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#4: per-effect scaling modifiers from `+ N% stat_ref`.
    /// Outer SmallVec is parallel to `effects`; inner SmallVec is the
    /// per-effect scaling list bounded at `MAX_SCALINGS_PER_EFFECT`
    /// (2 — fits LoL/MOBA convention of one or two scaling stats per
    /// effect, e.g. `damage 50 + 30% AP + 20% AD`). An empty inner
    /// SmallVec means no scaling was authored on that effect; an empty
    /// outer SmallVec means no effect on the ability carried the
    /// modifier (Wave 1 corpus shape — keeps output bit-stable).
    /// Apply handlers should treat empty + per-slot-empty identically
    /// (the effect's flat amount is the final value with no scaling
    /// added). The SoA packer emits TWO companion columns
    /// (`scaling_stat_refs` + `scaling_percents`) at a stride of
    /// `MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT` — stat_refs
    /// is a u8 discriminant column with sentinel
    /// `SCALING_STAT_NONE_SENTINEL` (0xFF) for unused slots; percents
    /// is f32 with `0.0` for unused slots.
    pub scalings_per_effect:
        SmallVec<[SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#7: per-effect conditional gate from `when <cond>
    /// [else <cond>]`. `None` for effects that didn't carry the
    /// modifier — apply handlers should treat the slot as "no gate"
    /// (the effect always passes the conditional check). The predicate
    /// body is stored as verbatim source text; downstream apply
    /// handlers parse + evaluate it (deferred infrastructure). Index
    /// parallel to `effects`; a populated `when_per_effect` slice is
    /// either empty (no effect carried the modifier) or has one slot
    /// per effect (`None` for the unmarked ones). Not GPU-packed —
    /// `String` payload is CPU-only metadata for now; the SoA packer
    /// in `PackedAbilityRegistry` skips this column.
    pub when_per_effect: SmallVec<[Option<EffectWhenCondition>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.5#9: per-effect nested follow-up effects from
    /// `<outer_verb> <args> { <inner_stmt>; <inner_stmt>; ... }`.
    /// Outer SmallVec is parallel to `effects`; inner SmallVec is
    /// the per-outer-effect nested list bounded at
    /// `MAX_NESTED_PER_EFFECT`. Each inner entry is a fully-lowered
    /// `EffectOp` — the outer cast resolves first, then apply
    /// handlers fire each nested op (deferred infrastructure;
    /// today's lowering only captures the IR, no apply dispatch).
    /// An empty inner SmallVec means no nested op was authored on
    /// that effect; an empty outer SmallVec means no effect on the
    /// ability carried the modifier (Wave 1 corpus shape — keeps
    /// output bit-stable). Inner-effect modifiers (the nested
    /// EffectStmt's own tags/chance/stacking/etc.) are SILENTLY
    /// DROPPED at lowering today — recursive aggregator capture is
    /// later infrastructure; authors who need rich nested modifiers
    /// should compose multiple outer effects instead.
    pub nested_per_effect:
        SmallVec<[SmallVec<[EffectOp; MAX_NESTED_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM]>,
    /// Wave 1.1 `cost: <amount> <resource>` header. `None` for
    /// abilities that declare no cost header (Wave 1 corpus +
    /// abilities that gate purely on cooldown). Apply handlers
    /// debit the resource at cast-decide time (deferred —
    /// resource SoA fields not all wired yet, e.g. stamina).
    pub cost: Option<AbilityCost>,
    /// Wave 1.1 `charges: <int>` header — maximum stored charges
    /// for charge-based recast abilities. `None` = no charge pool
    /// (every cast goes straight to cooldown). Apply handlers wire
    /// the per-agent charge SoA and recharge timer (deferred).
    pub charges: Option<u32>,
    /// Wave 1.1 `recharge: <duration>` header — time per charge to
    /// regenerate when the pool is below max. Stored as ticks (100ms
    /// each). Meaningless without `charges`; `None` if not declared.
    pub recharge_ticks: Option<u32>,
    /// Wave 1.1 `toggle` marker — declares this ability as a toggle
    /// (cast to enable, recast to disable, optional per-tick drain).
    /// `false` for one-shot abilities. Apply handlers wire toggle
    /// state SoA + per-tick drain accounting (deferred).
    pub is_toggle: bool,
    /// #129 / Wave 1.4 `recast: <int|dur>` header — multi-stage cast
    /// state. Two shapes per spec §4.2:
    ///   * `Some(RecastKind::Count(n))` — up to `n` consecutive
    ///     recasts within the recast window (e.g. Aatrox `recast: 1`,
    ///     Ahri `recast: 3`).
    ///   * `Some(RecastKind::CooldownTicks(t))` — `recast: 4s` form,
    ///     ticks-between-recasts cooldown (rare in the corpus today).
    /// `None` for plain abilities. Apply handlers wire the per-agent
    /// recast counter / window timer SoA (deferred — #125 family).
    pub recast: Option<RecastKind>,
    /// #129 / Wave 1.4 `recast_window: <duration>` — how long after
    /// the initial cast the recast state stays open before being
    /// dropped. Stored as ticks (100ms each). Meaningless without
    /// `recast`; `None` if not declared.
    pub recast_window_ticks: Option<u32>,
    /// `target: <mode>` header — captured from the eight modes per
    /// spec §4.3. Apply handlers only read Enemy/Self/Ally today
    /// (the per-pair-targeted modes); the position/directional/global
    /// modes (Ground, Direction, Vector, Global, SelfAoe) are
    /// captured here so the corpus lowers, with apply dispatch
    /// arriving via registry-driven infrastructure (#125). Default
    /// is `Enemy` to match historical `Area::SingleTarget` +
    /// `gate.hostile_only=true` shape.
    pub target_mode: TargetModeKind,
}

impl AbilityProgram {
    /// Convenience: a single-target, instant ability with the given gate
    /// and effect list. Most hand-authored test programs use this shape.
    ///
    /// Constructs a program with no scoring hint and no tags; callers
    /// that need them should set `hint` + `tags` on the returned value
    /// (or use `with_hint` / `with_tags`).
    pub fn new_single_target(
        range:   f32,
        gate:    Gate,
        effects: impl IntoIterator<Item = EffectOp>,
    ) -> Self {
        let mut v: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        for e in effects { v.push(e); }
        Self {
            delivery:  Delivery::Instant,
            area:      Area::SingleTarget { range },
            gate,
            effects:   v,
            hint:      None,
            tags:      SmallVec::new(),
            stackings: SmallVec::new(),
            chances:   SmallVec::new(),
            lifetimes: SmallVec::new(),
            per_effect_areas:    SmallVec::new(),
            scalings_per_effect: SmallVec::new(),
            when_per_effect:     SmallVec::new(),
            nested_per_effect:   SmallVec::new(),
            cost:                None,
            charges:             None,
            recharge_ticks:      None,
            is_toggle:           false,
            recast:              None,
            recast_window_ticks: None,
            // Default Enemy keeps historical convenience-constructor shape
            // (single-target, hostile_only=true on the gate).
            target_mode:         TargetModeKind::Enemy,
        }
    }

    /// Builder-style setter for `hint`. Returns `self` for chaining.
    pub fn with_hint(mut self, hint: AbilityHint) -> Self {
        self.hint = Some(hint);
        self
    }

    /// Builder-style setter for `tags`. Any prior tag entries are
    /// replaced. Returns `self` for chaining.
    pub fn with_tags<I>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = (AbilityTag, f32)>,
    {
        self.tags = SmallVec::new();
        for t in tags { self.tags.push(t); }
        self
    }

    /// Look up the value of a specific tag on this ability. Returns
    /// `0.0` when absent — the DSL contract for `ability::tag(TAG)` on
    /// an ability without the tag.
    #[inline]
    pub fn tag_value(&self, tag: AbilityTag) -> f32 {
        for &(t, v) in self.tags.iter() {
            if t == tag { return v; }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_op_size_budget() {
        // 16-byte budget: largest payload is CastAbility at ~6 bytes plus a
        // u8 discriminant, padded to 16. Anything larger means we've
        // accidentally grown a variant — catch it at the first site.
        // (TransferGold was i64 pre-2026-04-22; now i32.)
        let sz = std::mem::size_of::<EffectOp>();
        assert!(sz <= 16, "EffectOp grew past 16B budget: {sz}");
    }

    #[test]
    fn construct_damage_program() {
        let p = AbilityProgram::new_single_target(
            6.0,
            Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        assert_eq!(p.effects.len(), 1);
        assert!(matches!(p.delivery, Delivery::Instant));
        assert!(matches!(p.area, Area::SingleTarget { range: 6.0 }));
    }

    #[test]
    fn construct_multi_effect_program() {
        let p = AbilityProgram::new_single_target(
            4.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 20.0 },
                EffectOp::Stun { duration_ticks: 5 },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[1] {
            EffectOp::Stun { duration_ticks } => assert_eq!(duration_ticks, 5),
            _ => panic!("expected Stun at index 1"),
        }
    }

    #[test]
    fn construct_world_effect_program() {
        let p = AbilityProgram::new_single_target(
            3.0,
            Gate { cooldown_ticks: 100, hostile_only: false, line_of_sight: false },
            [
                EffectOp::TransferGold { amount: -50 },
                EffectOp::ModifyStanding { delta: -20 },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[0] {
            EffectOp::TransferGold { amount } => assert_eq!(amount, -50),
            _ => panic!("expected TransferGold at index 0"),
        }
    }

    #[test]
    fn construct_recursive_chain_program() {
        let nested = AbilityId::new(7).unwrap();
        let p = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 1, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 10.0 },
                EffectOp::CastAbility { ability: nested, selector: TargetSelector::Target },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[1] {
            EffectOp::CastAbility { ability, selector } => {
                assert_eq!(ability.raw(), 7);
                assert_eq!(selector, TargetSelector::Target);
            }
            _ => panic!("expected CastAbility at index 1"),
        }
    }
}
