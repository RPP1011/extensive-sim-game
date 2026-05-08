//! TOML loader for `AbilityRegistry` — collapses hand-rolled
//! `binding_check.rs` boilerplate (e.g. `duel_25v25_runtime`'s ~500 LOC
//! of `AbilityProgram::new_single_target` calls) to a single
//! `AbilityRegistry::from_toml(...)` call.
//!
//! Schema mirrors `AbilityProgram` 1:1. One TOML file owns the full
//! registry; `[[ability]]` array-of-tables drives append-only
//! registration in source order. The first `[[ability]]` lands at slot
//! `AbilityId(1)`, the second at `AbilityId(2)`, etc. — same ordering
//! discipline as `AbilityRegistryBuilder::register()`.
//!
//! ```toml
//! [[ability]]
//! name = "Strike"            # advisory; not required at runtime
//! range = 1.5
//! gate  = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
//! target_mode = "Enemy"      # Enemy | SelfCast | Ally | SelfAoe |
//!                            # Ground | Direction | Vector | Global
//! hint  = "damage"           # damage | defense | crowd_control |
//!                            # utility | heal | buff (optional)
//!
//! [[ability.effects]]
//! op     = "Damage"          # discriminator — see EffectOp variants
//! amount = 5.0
//!
//! [[ability]]
//! name = "Cleave"
//! range = 3.0
//! gate  = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
//!
//! [[ability.effects]]
//! op     = "Damage"
//! amount = 2.0
//! # Per-effect AOE shape (`per_effect_areas[i]` slot). Each effect
//! # may carry one shape; omit for single-target.
//! area   = { kind = "Circle", args = [1.0, 0.0, 0.0, 0.0] }
//! ```
//!
//! Coverage today (the **subset** the existing hand-rolled registries
//! exercise — additional EffectOp variants and modifiers can be wired
//! in follow-up slices on demand):
//!
//! - **Top-level fields:** `range`, `gate`, `target_mode`, `hint`,
//!   `tags`, `cost`, `charges`, `recharge_ticks`, `is_toggle`,
//!   `recast`, `recast_window_ticks`.
//! - **EffectOp variants (per-effect `op = "..."`):** Damage, Heal,
//!   Shield, Stun, Slow, TransferGold, ModifyStanding, Root,
//!   Silence, Fear, Taunt, Dash, Blink, Knockback, Pull, Execute,
//!   SelfDamage, LifeSteal, DamageModify, Stealth, Charm,
//!   Grounded, Suppress, PlantBelief, Observe, Scry, Reveal,
//!   Disguise, Decoy, EraseBelief.  (`CastAbility` requires a
//!   resolved `AbilityId` — wire later when call sites need it.
//!   `Buff`, `Reflect`, the DOT/HOT family, `TimedShield`,
//!   `Summon`, `Harvest`, `PlaceVoxel` aren't exercised by any
//!   current hand-rolled registry — wire on demand.)
//! - **Per-effect modifiers:** `area` (one `EffectAreaShape` →
//!   `per_effect_areas[i]`), `chance` (q16 int 0..=65534 →
//!   `chances[i]`), `stacking` (Refresh|Stack|Extend →
//!   `stackings[i]`), `lifetime` (UntilCasterDies|DamageableHp|
//!   BreakOnDamage → `lifetimes[i]`), `scalings`
//!   (`[{stat_ref, percent}]` → `scalings_per_effect[i]`).
//!
//! q8 fixed-point fields (`Slow.factor_q8`, `LifeSteal.fraction_q8`,
//! `DamageModify.multiplier_q8`) accept the raw `i16` literal; the
//! loader does NOT auto-convert from `0.5` → `128`. Callers preserve
//! their mental model (the .ability surface does the float→q8
//! conversion at lower time; TOML callers stay closer to the IR).
//!
//! Errors return [`AbilityTomlError`] with a contextual message
//! pointing at the offending `[[ability]]` index and field name. No
//! panics — this is the build-time path for runtime crates that ship
//! `assets/abilities/<name>.toml` alongside their `.sim` source.

use std::path::Path;

use serde::Deserialize;
use smallvec::SmallVec;

use super::program::{
    AbilityCost, AbilityHint, AbilityProgram, AbilityTag, Area, BuffStat, CostAmount,
    CostResource, Delivery, EffectAreaShape, EffectOp, EffectScaling, Gate, LifetimeMode,
    RecastKind, ScalingStatRef, ShapeKind, StackingMode, TargetModeKind,
    MAX_EFFECTS_PER_PROGRAM,
};
use super::{AbilityRegistry, AbilityRegistryBuilder};

/// Errors surfaced by [`AbilityRegistry::from_toml`] /
/// [`AbilityRegistry::from_toml_str`]. Each variant carries enough
/// context (ability index, field name, offending value) that a build
/// failure points straight at the source defect.
#[derive(Debug)]
pub enum AbilityTomlError {
    /// I/O failure reading the TOML file.
    Io(std::io::Error),
    /// `toml::de` returned a parse / type error.
    Parse(toml::de::Error),
    /// An ability's `target_mode = "..."` was not in the canonical
    /// vocabulary. Carries the bad token + ability index.
    UnknownTargetMode { ability_index: usize, value: String },
    /// `hint = "..."` not in the canonical vocabulary.
    UnknownHint { ability_index: usize, value: String },
    /// A `tags` entry's tag name was not in the canonical vocabulary.
    UnknownTag { ability_index: usize, value: String },
    /// An effect's `op = "..."` discriminator was not in the supported
    /// vocabulary (see module docs for the supported subset).
    UnknownEffectOp {
        ability_index: usize,
        effect_index: usize,
        value: String,
    },
    /// An effect's `op` is supported in the IR but not yet wired in
    /// the loader — call out the specific effect so the call site can
    /// either extend the loader or fall back to the constructor.
    UnsupportedEffectOp {
        ability_index: usize,
        effect_index: usize,
        op: &'static str,
    },
    /// A required field was missing for a given `op`.
    MissingEffectField {
        ability_index: usize,
        effect_index: usize,
        op: &'static str,
        field: &'static str,
    },
    /// An effect's `area.kind` not in the canonical shape vocabulary.
    UnknownShapeKind {
        ability_index: usize,
        effect_index: usize,
        value: String,
    },
    /// A scaling slot's `stat_ref` not in the canonical vocabulary.
    UnknownScalingStat {
        ability_index: usize,
        effect_index: usize,
        scaling_index: usize,
        value: String,
    },
    /// An effect's `stacking = "..."` not Refresh / Stack / Extend.
    UnknownStackingMode {
        ability_index: usize,
        effect_index: usize,
        value: String,
    },
    /// An effect's `lifetime.kind` not Until / Damageable / Break.
    UnknownLifetimeKind {
        ability_index: usize,
        effect_index: usize,
        value: String,
    },
    /// `cost.resource = "..."` not Mana / Stamina / Hp / Gold.
    UnknownCostResource { ability_index: usize, value: String },
    /// `[[ability.effects]]` overflowed `MAX_EFFECTS_PER_PROGRAM`.
    TooManyEffects {
        ability_index: usize,
        got: usize,
        max: usize,
    },
}

impl std::fmt::Display for AbilityTomlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use AbilityTomlError::*;
        match self {
            Io(e) => write!(f, "I/O error reading abilities TOML: {e}"),
            Parse(e) => write!(f, "TOML parse error: {e}"),
            UnknownTargetMode { ability_index, value } => write!(
                f,
                "ability[{ability_index}].target_mode = {value:?}: unknown target mode \
                 (expected one of Enemy/SelfCast/Ally/SelfAoe/Ground/Direction/Vector/Global)"
            ),
            UnknownHint { ability_index, value } => write!(
                f,
                "ability[{ability_index}].hint = {value:?}: unknown hint \
                 (expected one of damage/defense/crowd_control/utility/heal/buff)"
            ),
            UnknownTag { ability_index, value } => write!(
                f,
                "ability[{ability_index}].tags entry {value:?}: unknown tag \
                 (expected PHYSICAL/MAGICAL/CROWD_CONTROL/HEAL/DEFENSE/UTILITY)"
            ),
            UnknownEffectOp { ability_index, effect_index, value } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].op = {value:?}: unknown effect op"
            ),
            UnsupportedEffectOp { ability_index, effect_index, op } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].op = {op:?}: \
                 IR variant exists but the TOML loader doesn't wire it yet — \
                 either extend the loader or use the AbilityProgram constructor"
            ),
            MissingEffectField { ability_index, effect_index, op, field } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}] (op = {op:?}): missing field {field:?}"
            ),
            UnknownShapeKind { ability_index, effect_index, value } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].area.kind = {value:?}: \
                 unknown shape (expected one of circle/cone/line/ring/spread/box/sphere/column/wall/cylinder/dome/hull)"
            ),
            UnknownScalingStat {
                ability_index, effect_index, scaling_index, value,
            } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].scalings[{scaling_index}].stat_ref = {value:?}: \
                 unknown stat ref (expected attack_damage/ability_power/max_hp/hp/armor/magic_resist/move_speed/mana)"
            ),
            UnknownStackingMode { ability_index, effect_index, value } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].stacking = {value:?}: \
                 unknown stacking mode (expected Refresh/Stack/Extend)"
            ),
            UnknownLifetimeKind { ability_index, effect_index, value } => write!(
                f,
                "ability[{ability_index}].effects[{effect_index}].lifetime.kind = {value:?}: \
                 unknown lifetime (expected UntilCasterDies/DamageableHp/BreakOnDamage)"
            ),
            UnknownCostResource { ability_index, value } => write!(
                f,
                "ability[{ability_index}].cost.resource = {value:?}: \
                 unknown resource (expected Mana/Stamina/Hp/Gold)"
            ),
            TooManyEffects { ability_index, got, max } => write!(
                f,
                "ability[{ability_index}] has {got} effects (max {max} per program)"
            ),
        }
    }
}

impl std::error::Error for AbilityTomlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AbilityTomlError::Io(e) => Some(e),
            AbilityTomlError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AbilityTomlError {
    fn from(e: std::io::Error) -> Self { AbilityTomlError::Io(e) }
}
impl From<toml::de::Error> for AbilityTomlError {
    fn from(e: toml::de::Error) -> Self { AbilityTomlError::Parse(e) }
}

// ---- TOML schema types (private — host-side only) ----

#[derive(Deserialize)]
struct RegistryDoc {
    #[serde(default)]
    ability: Vec<AbilityDoc>,
}

#[derive(Deserialize)]
struct AbilityDoc {
    /// Optional human-readable label. Not required by the loader (the
    /// registry is slot-indexed); useful for debug output and as a
    /// search anchor when reading the TOML. Read out of the file (not
    /// `#[serde(skip)]`) so a `name = "..."` line in the TOML doesn't
    /// surface as `unknown field` and break otherwise-valid registries.
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
    range: f32,
    gate: GateDoc,
    #[serde(default)]
    target_mode: Option<String>,
    #[serde(default)]
    hint: Option<String>,
    #[serde(default)]
    tags: Vec<TagDoc>,
    #[serde(default)]
    cost: Option<CostDoc>,
    #[serde(default)]
    charges: Option<u32>,
    #[serde(default)]
    recharge_ticks: Option<u32>,
    #[serde(default)]
    is_toggle: Option<bool>,
    #[serde(default)]
    recast: Option<RecastDoc>,
    #[serde(default)]
    recast_window_ticks: Option<u32>,
    #[serde(default)]
    effects: Vec<EffectDoc>,
}

#[derive(Deserialize)]
struct GateDoc {
    cooldown_ticks: u32,
    hostile_only: bool,
    line_of_sight: bool,
}

#[derive(Deserialize)]
struct TagDoc {
    tag: String,
    value: f32,
}

#[derive(Deserialize)]
struct CostDoc {
    resource: String,
    /// One of `amount` (Flat) or `percent_of_max` must be supplied.
    #[serde(default)]
    amount: Option<f32>,
    #[serde(default)]
    percent_of_max: Option<f32>,
}

#[derive(Deserialize)]
struct RecastDoc {
    /// Use `count` (Count) OR `cooldown_ticks` (CooldownTicks).
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    cooldown_ticks: Option<u32>,
}

#[derive(Deserialize)]
struct EffectDoc {
    op: String,
    // ---- Common payload fields (which apply depends on `op`) ----
    #[serde(default)] amount: Option<f32>,
    #[serde(default)] duration_ticks: Option<u32>,
    #[serde(default)] factor_q8: Option<i16>,
    #[serde(default)] fraction_q8: Option<i16>,
    #[serde(default)] multiplier_q8: Option<i16>,
    #[serde(default)] hp_threshold: Option<f32>,
    #[serde(default)] distance: Option<f32>,
    #[serde(default)] delta: Option<i16>,
    /// ToM payloads
    #[serde(default)] subject_idx: Option<u32>,
    #[serde(default)] fact_bit: Option<u8>,
    #[serde(default)] target_observer: Option<u8>,
    #[serde(default)] fake_type: Option<u8>,
    #[serde(default)] fake_pos: Option<u32>,
    #[serde(default)] fields: Option<u8>,
    // ---- Per-effect modifiers ----
    #[serde(default)] area: Option<AreaDoc>,
    /// Q16 fixed-point. None ≡ "always fires"; Some(x) compares
    /// `(rng & 0xFFFF) < x`. 0..=65534 valid (65535 reserved as
    /// sentinel inside the GPU packer).
    #[serde(default)] chance: Option<u16>,
    #[serde(default)] stacking: Option<String>,
    #[serde(default)] lifetime: Option<LifetimeDoc>,
    #[serde(default)] scalings: Vec<ScalingDoc>,
}

#[derive(Deserialize)]
struct AreaDoc {
    kind: String,
    /// 4-element positional args matching the spec §8 shape signature.
    /// Authors zero-pad unused slots (e.g. `circle(r)` writes
    /// `[r, 0, 0, 0]`).
    args: [f32; 4],
}

#[derive(Deserialize)]
struct LifetimeDoc {
    kind: String,
    /// Required only for `kind = "DamageableHp"`.
    #[serde(default)] hp: Option<f32>,
}

#[derive(Deserialize)]
struct ScalingDoc {
    stat_ref: String,
    /// Stored as a fraction (0.30 == "+30%"). Matches
    /// `EffectScaling::percent`'s convention — caller-side.
    percent: f32,
}

// ---- Public entry points (impls on AbilityRegistry below) ----

/// Parse a TOML document into an `AbilityRegistry`. Builds
/// append-only — first `[[ability]]` lands at `AbilityId(1)`, second
/// at `AbilityId(2)`, … (same as `AbilityRegistryBuilder::register`).
pub fn from_toml_str(src: &str) -> Result<AbilityRegistry, AbilityTomlError> {
    let doc: RegistryDoc = toml::from_str(src)?;
    let mut builder = AbilityRegistryBuilder::new();
    for (i, a) in doc.ability.iter().enumerate() {
        builder.register(lower_ability(i, a)?);
    }
    Ok(builder.build())
}

/// Read + parse + lower the TOML at `path`.
pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<AbilityRegistry, AbilityTomlError> {
    let src = std::fs::read_to_string(path)?;
    from_toml_str(&src)
}

// ---- Lowering ----

fn lower_ability(i: usize, a: &AbilityDoc) -> Result<AbilityProgram, AbilityTomlError> {
    if a.effects.len() > MAX_EFFECTS_PER_PROGRAM {
        return Err(AbilityTomlError::TooManyEffects {
            ability_index: i,
            got: a.effects.len(),
            max: MAX_EFFECTS_PER_PROGRAM,
        });
    }

    let gate = Gate {
        cooldown_ticks: a.gate.cooldown_ticks,
        hostile_only: a.gate.hostile_only,
        line_of_sight: a.gate.line_of_sight,
    };

    let target_mode = match a.target_mode.as_deref() {
        None | Some("Enemy") | Some("enemy") => TargetModeKind::Enemy,
        Some("SelfCast") | Some("self_cast") | Some("self") => TargetModeKind::SelfCast,
        Some("Ally") | Some("ally") => TargetModeKind::Ally,
        Some("SelfAoe") | Some("self_aoe") => TargetModeKind::SelfAoe,
        Some("Ground") | Some("ground") => TargetModeKind::Ground,
        Some("Direction") | Some("direction") => TargetModeKind::Direction,
        Some("Vector") | Some("vector") => TargetModeKind::Vector,
        Some("Global") | Some("global") => TargetModeKind::Global,
        Some(s) => {
            return Err(AbilityTomlError::UnknownTargetMode {
                ability_index: i,
                value: s.to_string(),
            });
        }
    };

    let hint = match a.hint.as_deref() {
        None => None,
        Some(s) => match AbilityHint::parse(s) {
            Some(h) => Some(h),
            None => return Err(AbilityTomlError::UnknownHint {
                ability_index: i,
                value: s.to_string(),
            }),
        },
    };

    let mut tags: SmallVec<[(AbilityTag, f32); super::program::MAX_TAGS_PER_PROGRAM]> =
        SmallVec::new();
    for t in &a.tags {
        let parsed = AbilityTag::parse(&t.tag).ok_or_else(|| AbilityTomlError::UnknownTag {
            ability_index: i,
            value: t.tag.clone(),
        })?;
        tags.push((parsed, t.value));
    }

    // Walk effects + parallel modifier slots simultaneously. Per
    // AbilityProgram contract, modifier vectors must be either empty
    // or have one slot per effect — any effect that supplies the
    // modifier promotes the whole vector to "populated, with None for
    // unmarked slots".
    let mut effects: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
    let mut per_effect_areas: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut chances: SmallVec<[Option<u16>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
    let mut stackings: SmallVec<[Option<StackingMode>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut lifetimes: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut scalings_per_effect: SmallVec<
        [SmallVec<[EffectScaling; super::program::MAX_SCALINGS_PER_EFFECT]>;
            MAX_EFFECTS_PER_PROGRAM],
    > = SmallVec::new();

    let mut any_area = false;
    let mut any_chance = false;
    let mut any_stacking = false;
    let mut any_lifetime = false;
    let mut any_scaling = false;

    for (j, e) in a.effects.iter().enumerate() {
        effects.push(lower_effect_op(i, j, e)?);

        let area = match &e.area {
            None => None,
            Some(d) => Some(lower_area(i, j, d)?),
        };
        if area.is_some() { any_area = true; }
        per_effect_areas.push(area);

        if e.chance.is_some() { any_chance = true; }
        chances.push(e.chance);

        let stacking = match e.stacking.as_deref() {
            None => None,
            Some("Refresh") | Some("refresh") => Some(StackingMode::Refresh),
            Some("Stack")   | Some("stack")   => Some(StackingMode::Stack),
            Some("Extend")  | Some("extend")  => Some(StackingMode::Extend),
            Some(s) => {
                return Err(AbilityTomlError::UnknownStackingMode {
                    ability_index: i,
                    effect_index: j,
                    value: s.to_string(),
                });
            }
        };
        if stacking.is_some() { any_stacking = true; }
        stackings.push(stacking);

        let lifetime = match &e.lifetime {
            None => None,
            Some(d) => Some(lower_lifetime(i, j, d)?),
        };
        if lifetime.is_some() { any_lifetime = true; }
        lifetimes.push(lifetime);

        let mut inner: SmallVec<[EffectScaling; super::program::MAX_SCALINGS_PER_EFFECT]> =
            SmallVec::new();
        for (k, s) in e.scalings.iter().enumerate() {
            let stat_ref = ScalingStatRef::parse(&s.stat_ref).ok_or_else(|| {
                AbilityTomlError::UnknownScalingStat {
                    ability_index: i,
                    effect_index: j,
                    scaling_index: k,
                    value: s.stat_ref.clone(),
                }
            })?;
            inner.push(EffectScaling { stat_ref, percent: s.percent });
        }
        if !inner.is_empty() { any_scaling = true; }
        scalings_per_effect.push(inner);
    }

    // Trim empty modifier vectors to match the AbilityProgram
    // convention: an unmarked corpus emits an *empty* vector, not a
    // vector full of `None`s. Keeps output bit-identical to
    // hand-rolled programs that never touched the modifier slot.
    if !any_area     { per_effect_areas.clear(); }
    if !any_chance   { chances.clear(); }
    if !any_stacking { stackings.clear(); }
    if !any_lifetime { lifetimes.clear(); }
    // Inner SmallVecs may still be non-empty inside outer; but if
    // nobody scaled at all, drop the whole outer.
    if !any_scaling { scalings_per_effect.clear(); }

    let cost = match &a.cost {
        None => None,
        Some(c) => Some(lower_cost(i, c)?),
    };
    let recast = match &a.recast {
        None => None,
        Some(r) => match (r.count, r.cooldown_ticks) {
            (Some(n), None) => Some(RecastKind::Count(n)),
            (None, Some(t)) => Some(RecastKind::CooldownTicks(t)),
            (None, None) | (Some(_), Some(_)) => {
                return Err(AbilityTomlError::MissingEffectField {
                    ability_index: i,
                    effect_index: 0,
                    op: "ability.recast",
                    field: "exactly one of `count` / `cooldown_ticks`",
                });
            }
        },
    };

    Ok(AbilityProgram {
        delivery: Delivery::Instant,
        area: Area::SingleTarget { range: a.range },
        gate,
        effects,
        hint,
        tags,
        stackings,
        chances,
        lifetimes,
        per_effect_areas,
        scalings_per_effect,
        when_per_effect: SmallVec::new(),
        nested_per_effect: SmallVec::new(),
        cost,
        charges: a.charges,
        recharge_ticks: a.recharge_ticks,
        is_toggle: a.is_toggle.unwrap_or(false),
        recast,
        recast_window_ticks: a.recast_window_ticks,
        target_mode,
    })
}

fn lower_effect_op(i: usize, j: usize, e: &EffectDoc) -> Result<EffectOp, AbilityTomlError> {
    macro_rules! field {
        ($field:ident, $op:literal) => {
            e.$field.ok_or(AbilityTomlError::MissingEffectField {
                ability_index: i,
                effect_index: j,
                op: $op,
                field: stringify!($field),
            })?
        };
    }
    Ok(match e.op.as_str() {
        // Combat
        "Damage"    => EffectOp::Damage    { amount: field!(amount, "Damage") },
        "Heal"      => EffectOp::Heal      { amount: field!(amount, "Heal") },
        "Shield"    => EffectOp::Shield    { amount: field!(amount, "Shield") },
        "Stun"      => EffectOp::Stun      { duration_ticks: field!(duration_ticks, "Stun") },
        "Slow"      => EffectOp::Slow {
            duration_ticks: field!(duration_ticks, "Slow"),
            factor_q8:      field!(factor_q8, "Slow"),
        },
        // World
        "TransferGold"   => EffectOp::TransferGold   { amount: e.amount.map(|x| x as i32)
            .ok_or(AbilityTomlError::MissingEffectField {
                ability_index: i, effect_index: j, op: "TransferGold", field: "amount" })? },
        "ModifyStanding" => EffectOp::ModifyStanding { delta: field!(delta, "ModifyStanding") },
        // Control verbs
        "Root"     => EffectOp::Root     { duration_ticks: field!(duration_ticks, "Root") },
        "Silence"  => EffectOp::Silence  { duration_ticks: field!(duration_ticks, "Silence") },
        "Fear"     => EffectOp::Fear     { duration_ticks: field!(duration_ticks, "Fear") },
        "Taunt"    => EffectOp::Taunt    { duration_ticks: field!(duration_ticks, "Taunt") },
        // Movement verbs
        "Dash"      => EffectOp::Dash      { distance: field!(distance, "Dash") },
        "Blink"     => EffectOp::Blink     { distance: field!(distance, "Blink") },
        "Knockback" => EffectOp::Knockback { distance: field!(distance, "Knockback") },
        "Pull"      => EffectOp::Pull      { distance: field!(distance, "Pull") },
        // Advanced verbs
        "Execute"    => EffectOp::Execute    { hp_threshold: field!(hp_threshold, "Execute") },
        "SelfDamage" => EffectOp::SelfDamage { amount: field!(amount, "SelfDamage") },
        "LifeSteal"  => EffectOp::LifeSteal  {
            duration_ticks: field!(duration_ticks, "LifeSteal"),
            fraction_q8:    field!(fraction_q8, "LifeSteal"),
        },
        "DamageModify" => EffectOp::DamageModify {
            duration_ticks: field!(duration_ticks, "DamageModify"),
            multiplier_q8: field!(multiplier_q8, "DamageModify"),
        },
        // CC vocabulary (Wave 2 piece 8)
        "Stealth"   => EffectOp::Stealth   { duration_ticks: field!(duration_ticks, "Stealth") },
        "Charm"     => EffectOp::Charm     { duration_ticks: field!(duration_ticks, "Charm") },
        "Grounded"  => EffectOp::Grounded  { duration_ticks: field!(duration_ticks, "Grounded") },
        "Suppress"  => EffectOp::Suppress  { duration_ticks: field!(duration_ticks, "Suppress") },
        // ToM verbs
        "PlantBelief" => EffectOp::PlantBelief {
            subject_idx: field!(subject_idx, "PlantBelief"),
            fact_bit:    field!(fact_bit, "PlantBelief"),
        },
        "Observe" => EffectOp::Observe { target_observer: e.target_observer.unwrap_or(0) },
        "Scry"    => EffectOp::Scry    {
            target_observer: field!(target_observer, "Scry"),
            subject_idx:     field!(subject_idx, "Scry"),
        },
        "Reveal"  => EffectOp::Reveal  { subject_idx: field!(subject_idx, "Reveal") },
        // Deception verbs
        "Disguise" => EffectOp::Disguise {
            fake_type:      field!(fake_type, "Disguise"),
            duration_ticks: field!(duration_ticks, "Disguise"),
        },
        "Decoy" => EffectOp::Decoy {
            subject_idx: field!(subject_idx, "Decoy"),
            fake_pos:    field!(fake_pos, "Decoy"),
        },
        "EraseBelief" => EffectOp::EraseBelief {
            subject_idx: field!(subject_idx, "EraseBelief"),
            fields:      field!(fields, "EraseBelief"),
        },

        // IR variants the loader doesn't wire yet — surface a clear
        // error so the caller knows to extend (or stay hand-rolled).
        "CastAbility"   => return Err(AbilityTomlError::UnsupportedEffectOp {
            ability_index: i, effect_index: j, op: "CastAbility" }),
        "Buff"          => return Err(AbilityTomlError::UnsupportedEffectOp {
            ability_index: i, effect_index: j, op: "Buff" }),
        "Reflect"       => return Err(AbilityTomlError::UnsupportedEffectOp {
            ability_index: i, effect_index: j, op: "Reflect" }),
        "DamageOverTime" | "HealOverTime" | "TimedShield"
        | "Summon" | "Harvest" | "PlaceVoxel" => return Err(AbilityTomlError::UnsupportedEffectOp {
            ability_index: i, effect_index: j,
            op: match e.op.as_str() {
                "DamageOverTime" => "DamageOverTime",
                "HealOverTime"   => "HealOverTime",
                "TimedShield"    => "TimedShield",
                "Summon"         => "Summon",
                "Harvest"        => "Harvest",
                "PlaceVoxel"     => "PlaceVoxel",
                _ => unreachable!(),
            }
        }),

        other => return Err(AbilityTomlError::UnknownEffectOp {
            ability_index: i,
            effect_index: j,
            value: other.to_string(),
        }),
    })
}

fn lower_area(i: usize, j: usize, d: &AreaDoc) -> Result<EffectAreaShape, AbilityTomlError> {
    let kind = ShapeKind::parse(&d.kind.to_lowercase()).ok_or_else(|| {
        AbilityTomlError::UnknownShapeKind {
            ability_index: i,
            effect_index: j,
            value: d.kind.clone(),
        }
    })?;
    Ok(EffectAreaShape { kind, args: d.args })
}

fn lower_lifetime(i: usize, j: usize, d: &LifetimeDoc) -> Result<LifetimeMode, AbilityTomlError> {
    Ok(match d.kind.as_str() {
        "UntilCasterDies" | "until_caster_dies" => LifetimeMode::UntilCasterDies,
        "DamageableHp"    | "damageable_hp"      => {
            let hp = d.hp.ok_or(AbilityTomlError::MissingEffectField {
                ability_index: i, effect_index: j, op: "lifetime", field: "hp"
            })?;
            LifetimeMode::DamageableHp(hp)
        }
        "BreakOnDamage"   | "break_on_damage"    => LifetimeMode::BreakOnDamage,
        s => return Err(AbilityTomlError::UnknownLifetimeKind {
            ability_index: i, effect_index: j, value: s.to_string(),
        }),
    })
}

fn lower_cost(i: usize, c: &CostDoc) -> Result<AbilityCost, AbilityTomlError> {
    let resource = match c.resource.as_str() {
        "Mana"    | "mana"    => CostResource::Mana,
        "Stamina" | "stamina" => CostResource::Stamina,
        "Hp"      | "hp"      => CostResource::Hp,
        "Gold"    | "gold"    => CostResource::Gold,
        s => return Err(AbilityTomlError::UnknownCostResource {
            ability_index: i, value: s.to_string(),
        }),
    };
    let amount = match (c.amount, c.percent_of_max) {
        (Some(x), None) => CostAmount::Flat(x),
        (None, Some(p)) => CostAmount::PercentOfMax(p),
        // Default to flat-zero rather than erroring, so authors can
        // declare just `cost.resource = "..."` for a 0-cost
        // placeholder. The .ability surface accepts this shape too.
        (None, None) => CostAmount::Flat(0.0),
        (Some(_), Some(_)) => {
            return Err(AbilityTomlError::MissingEffectField {
                ability_index: i, effect_index: 0,
                op: "ability.cost",
                field: "exactly one of `amount` / `percent_of_max` (got both)",
            });
        }
    };
    // Ignore `BuffStat` import warnings; not used in lower_cost.
    let _ = BuffStat::MoveSpeed;
    Ok(AbilityCost { resource, amount })
}

// ---- impl AbilityRegistry ----

impl AbilityRegistry {
    /// Read + parse + lower a registry from a TOML file at `path`.
    /// See module docs for the schema.
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self, AbilityTomlError> {
        from_toml(path)
    }

    /// Parse + lower a registry from an in-memory TOML string. Useful
    /// for tests and for runtime crates that embed the source via
    /// `include_str!`.
    pub fn from_toml_str(src: &str) -> Result<Self, AbilityTomlError> {
        from_toml_str(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::AbilityId;

    fn parse(src: &str) -> AbilityRegistry {
        AbilityRegistry::from_toml_str(src).unwrap_or_else(|e| panic!("from_toml_str: {e}"))
    }

    #[test]
    fn empty_registry() {
        let reg = parse("");
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn single_target_damage() {
        let src = r#"
            [[ability]]
            name = "Strike"
            range = 5.0
            gate = { cooldown_ticks = 10, hostile_only = true, line_of_sight = false }

            [[ability.effects]]
            op = "Damage"
            amount = 30.0
        "#;
        let reg = parse(src);
        assert_eq!(reg.len(), 1);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert_eq!(p.gate.cooldown_ticks, 10);
        assert!(p.gate.hostile_only);
        match p.area {
            Area::SingleTarget { range } => assert_eq!(range, 5.0),
        }
        assert_eq!(p.effects.len(), 1);
        match p.effects[0] {
            EffectOp::Damage { amount } => assert_eq!(amount, 30.0),
            _ => panic!("expected Damage"),
        }
        // Modifier vectors should all be empty when unused.
        assert!(p.per_effect_areas.is_empty());
        assert!(p.chances.is_empty());
        assert!(p.stackings.is_empty());
        assert!(p.lifetimes.is_empty());
        assert!(p.scalings_per_effect.is_empty());
        assert_eq!(p.target_mode, TargetModeKind::Enemy);
    }

    #[test]
    fn slot_ids_are_source_order() {
        let src = r#"
            [[ability]]
            name = "First"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = false, line_of_sight = false }
            [[ability.effects]]
            op = "Heal"
            amount = 1.0

            [[ability]]
            name = "Second"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
            [[ability.effects]]
            op = "Damage"
            amount = 1.0

            [[ability]]
            name = "Third"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = false, line_of_sight = false }
            [[ability.effects]]
            op = "Shield"
            amount = 5.0
        "#;
        let reg = parse(src);
        assert_eq!(reg.len(), 3);
        // First slot = AbilityId(1) (Heal)
        match reg.get(AbilityId::new(1).unwrap()).unwrap().effects[0] {
            EffectOp::Heal { .. } => {}
            ref e => panic!("slot 1 expected Heal, got {e:?}"),
        }
        match reg.get(AbilityId::new(2).unwrap()).unwrap().effects[0] {
            EffectOp::Damage { .. } => {}
            ref e => panic!("slot 2 expected Damage, got {e:?}"),
        }
        match reg.get(AbilityId::new(3).unwrap()).unwrap().effects[0] {
            EffectOp::Shield { .. } => {}
            ref e => panic!("slot 3 expected Shield, got {e:?}"),
        }
    }

    #[test]
    fn multi_effect_with_per_effect_area() {
        let src = r#"
            [[ability]]
            name = "ConcussiveCleave"
            range = 3.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }

            [[ability.effects]]
            op = "Damage"
            amount = 3.0
            area = { kind = "circle", args = [1.0, 0.0, 0.0, 0.0] }

            [[ability.effects]]
            op = "Stun"
            duration_ticks = 15
            area = { kind = "circle", args = [1.0, 0.0, 0.0, 0.0] }
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert_eq!(p.effects.len(), 2);
        match p.effects[0] {
            EffectOp::Damage { amount } => assert_eq!(amount, 3.0),
            _ => panic!("expected Damage at 0"),
        }
        match p.effects[1] {
            EffectOp::Stun { duration_ticks } => assert_eq!(duration_ticks, 15),
            _ => panic!("expected Stun at 1"),
        }
        assert_eq!(p.per_effect_areas.len(), 2);
        for slot in 0..2 {
            assert_eq!(
                p.per_effect_areas[slot],
                Some(EffectAreaShape { kind: ShapeKind::Circle, args: [1.0, 0.0, 0.0, 0.0] }),
            );
        }
    }

    #[test]
    fn per_effect_modifiers_chance_stacking_lifetime_scalings() {
        let src = r#"
            [[ability]]
            name = "Bleed"
            range = 0.0
            gate = { cooldown_ticks = 50, hostile_only = false, line_of_sight = false }
            target_mode = "SelfCast"
            hint = "damage"

            [[ability.tags]]
            tag = "PHYSICAL"
            value = 50.0

            [[ability.effects]]
            op = "SelfDamage"
            amount = 5.0
            chance = 32768
            stacking = "Stack"
            lifetime = { kind = "DamageableHp", hp = 25.0 }
            scalings = [
                { stat_ref = "max_hp", percent = 0.05 },
            ]
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert!(matches!(p.effects[0], EffectOp::SelfDamage { amount } if amount == 5.0));
        assert_eq!(p.target_mode, TargetModeKind::SelfCast);
        assert_eq!(p.hint, Some(AbilityHint::Damage));
        assert_eq!(p.tags.len(), 1);
        assert_eq!(p.tags[0], (AbilityTag::Physical, 50.0));
        assert_eq!(p.chances, smallvec::smallvec![Some(32768u16)] as SmallVec<[Option<u16>; MAX_EFFECTS_PER_PROGRAM]>);
        assert_eq!(p.stackings.len(), 1);
        assert_eq!(p.stackings[0], Some(StackingMode::Stack));
        assert_eq!(p.lifetimes.len(), 1);
        assert_eq!(p.lifetimes[0], Some(LifetimeMode::DamageableHp(25.0)));
        assert_eq!(p.scalings_per_effect.len(), 1);
        assert_eq!(p.scalings_per_effect[0].len(), 1);
        assert_eq!(
            p.scalings_per_effect[0][0],
            EffectScaling { stat_ref: ScalingStatRef::MaxHp, percent: 0.05 },
        );
    }

    #[test]
    fn unused_modifier_vectors_stay_empty() {
        let src = r#"
            [[ability]]
            name = "Plain"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }

            [[ability.effects]]
            op = "Damage"
            amount = 1.0
            stacking = "Refresh"
            # area / chance / lifetime / scalings deliberately omitted

            [[ability.effects]]
            op = "Heal"
            amount = 1.0
            # no modifiers
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        // stacking populated → 2 slots, second is None
        assert_eq!(p.stackings.len(), 2);
        assert_eq!(p.stackings[0], Some(StackingMode::Refresh));
        assert_eq!(p.stackings[1], None);
        // The other modifier vectors stay empty (no effect set them).
        assert!(p.per_effect_areas.is_empty());
        assert!(p.chances.is_empty());
        assert!(p.lifetimes.is_empty());
        assert!(p.scalings_per_effect.is_empty());
    }

    #[test]
    fn reads_all_top_level_optional_fields() {
        let src = r#"
            [[ability]]
            name = "Toggleable"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = false, line_of_sight = false }
            target_mode = "Ally"
            hint = "buff"
            charges = 3
            recharge_ticks = 100
            is_toggle = true
            recast_window_ticks = 30
            recast = { count = 2 }
            cost = { resource = "Mana", amount = 25.0 }

            [[ability.effects]]
            op = "Heal"
            amount = 10.0
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert_eq!(p.target_mode, TargetModeKind::Ally);
        assert_eq!(p.hint, Some(AbilityHint::Buff));
        assert_eq!(p.charges, Some(3));
        assert_eq!(p.recharge_ticks, Some(100));
        assert!(p.is_toggle);
        assert_eq!(p.recast_window_ticks, Some(30));
        assert!(matches!(p.recast, Some(RecastKind::Count(2))));
        assert!(matches!(p.cost, Some(AbilityCost { resource: CostResource::Mana, amount: CostAmount::Flat(x) }) if x == 25.0));
    }

    #[test]
    fn errors_on_unknown_op() {
        let src = r#"
            [[ability]]
            name = "Bad"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = false, line_of_sight = false }
            [[ability.effects]]
            op = "Nonsense"
        "#;
        let err = match AbilityRegistry::from_toml_str(src) {
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => e,
        };
        match err {
            AbilityTomlError::UnknownEffectOp { ability_index: 0, effect_index: 0, .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn errors_on_missing_payload_field() {
        let src = r#"
            [[ability]]
            name = "Bad"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
            [[ability.effects]]
            op = "Damage"
            # amount missing
        "#;
        let err = match AbilityRegistry::from_toml_str(src) {
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => e,
        };
        match err {
            AbilityTomlError::MissingEffectField {
                ability_index: 0, effect_index: 0, op: "Damage", field: "amount",
            } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn errors_on_unsupported_op() {
        let src = r#"
            [[ability]]
            name = "Bad"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = false, line_of_sight = false }
            [[ability.effects]]
            op = "CastAbility"
        "#;
        let err = match AbilityRegistry::from_toml_str(src) {
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => e,
        };
        match err {
            AbilityTomlError::UnsupportedEffectOp { op: "CastAbility", .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn slow_factor_q8_round_trips() {
        // q8 packing stays caller-side (closer to IR) — the loader
        // takes the raw i16. 0.5 fraction == 128 q8.
        let src = r#"
            [[ability]]
            name = "ChillingTouch"
            range = 4.0
            gate = { cooldown_ticks = 50, hostile_only = true, line_of_sight = false }
            [[ability.effects]]
            op = "Slow"
            duration_ticks = 30
            factor_q8 = 128
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        match p.effects[0] {
            EffectOp::Slow { duration_ticks, factor_q8 } => {
                assert_eq!(duration_ticks, 30);
                assert_eq!(factor_q8, 128);
            }
            _ => panic!("expected Slow"),
        }
    }

    #[test]
    fn lower_disguise_op() {
        // Spy_network preview: Disguise verb wires through.
        let src = r#"
            [[ability]]
            name = "Disguise"
            range = 0.0
            gate = { cooldown_ticks = 300, hostile_only = false, line_of_sight = false }
            target_mode = "SelfCast"

            [[ability.effects]]
            op = "Disguise"
            fake_type = 7
            duration_ticks = 600
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        match p.effects[0] {
            EffectOp::Disguise { fake_type, duration_ticks } => {
                assert_eq!(fake_type, 7);
                assert_eq!(duration_ticks, 600);
            }
            _ => panic!("expected Disguise"),
        }
    }

    #[test]
    fn target_mode_aliases_lowercase() {
        let src = r#"
            [[ability]]
            name = "X"
            range = 1.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
            target_mode = "enemy"
            [[ability.effects]]
            op = "Damage"
            amount = 1.0
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert_eq!(p.target_mode, TargetModeKind::Enemy);
    }

    #[test]
    fn shape_kind_is_case_insensitive() {
        // ShapeKind::parse expects lowercase; the loader normalises.
        let src = r#"
            [[ability]]
            name = "X"
            range = 3.0
            gate = { cooldown_ticks = 0, hostile_only = true, line_of_sight = false }
            [[ability.effects]]
            op = "Damage"
            amount = 2.0
            area = { kind = "Circle", args = [1.5, 0.0, 0.0, 0.0] }
        "#;
        let reg = parse(src);
        let p = reg.get(AbilityId::new(1).unwrap()).unwrap();
        assert_eq!(
            p.per_effect_areas[0],
            Some(EffectAreaShape { kind: ShapeKind::Circle, args: [1.5, 0.0, 0.0, 0.0] }),
        );
    }
}
