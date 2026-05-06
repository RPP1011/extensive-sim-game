//! Wave 1.6 — lower the parsed `.ability` AST (`dsl_ast::AbilityFile` /
//! `AbilityDecl`) into engine-runtime `engine::ability::program::AbilityProgram`
//! values.
//!
//! Scope of this slice (per `docs/spec/ability_dsl_unified.md` §4 / §6 /
//! §7):
//!
//! * **Headers covered:** `target` (enemy/self only), `range`, `cooldown`,
//!   `cast` (TODO — `Gate` carries no `cast_ticks` field today; logged
//!   then ignored), `hint` (damage/defense/crowd_control/utility/heal —
//!   `economic` is reserved per §4.2).
//!
//! * **Effect verbs covered (20 of the 27 catalog entries):** `damage`,
//!   `heal`, `shield`, `stun`, `slow`, `transfer_gold`,
//!   `modify_standing`, `cast`, the Wave 2 piece 1 control verbs
//!   `root`, `silence`, `fear`, `taunt`, the Wave 2 piece 2
//!   movement verbs `dash`, `blink`, `knockback`, `pull`, the
//!   Wave 2 piece 3 advanced verbs `execute`, `self_damage`, plus the
//!   Wave 2 piece 4 buff verbs `lifesteal`, `damage_modify`. These
//!   match the 20 `EffectOp` variants on the engine side. Unknown
//!   verbs / arity mismatches are surfaced as errors.
//!
//! * **Out of scope (deferred to later waves):**
//!     - `template` / `structure` top-level blocks — Waves 1.2 / 1.3.
//!     - Other target modes (ally/self_aoe/ground/direction/vector/global)
//!       and `economic` hint — error today, wired by their respective
//!       waves.
//!     - The remaining 7 EffectOp variants (Teleport, ApplyStatus,
//!       SummonAlly, etc.) — Waves 2-5.
//!     - Two-phase split validator + ability-name resolution for
//!       `cast <Name>` — Wave 1.7 (registry wiring).
//!
//! Wave 1.4 surfaces (parser-only — lowering deferred):
//! The parser now accepts:
//!   * `recast: <int|dur>` and `recast_window: <duration>` ability
//!     headers.
//!   * `deliver <method> { params } { body }` body blocks (captured
//!     opaquely as a verbatim source slice — spec §9 hooks are Wave
//!     2+).
//!   * `morph { effects } into <Other>` body blocks.
//! Lowering of all five surfaces requires engine-side schema work
//! (multi-stage cast state, delivery-method SoA + on_*/on_arrival hook
//! dispatch, form-swap state machinery). Until then this module surfaces
//! `LowerError::HeaderNotImplemented { header: "recast" | "recast_window" }`,
//! `LowerError::DeliverBlockNotImplemented`, and
//! `LowerError::MorphBlockNotImplemented` respectively. Spec §4.4
//! states deliver and bare effects are mutually exclusive; the parser
//! deliberately admits coexistence to maximise corpus parse rate, and
//! this module enforces the spec rule via `LowerError::MixedBody`
//! BEFORE the deliver-block error fires.
//!
//! Wave 1.5 surfaces (parser-only — lowering deferred):
//! The `.ability` parser now lifts the nine effect-statement modifier
//! slots from spec §6.1 (`in <shape>`, `[TAG: value]`, `for <dur>`,
//! `when <cond>`, `chance N%`, `stacking <mode>`, `+ N% stat_ref`,
//! `until_caster_dies` / `damageable_hp(N)`, nested `{ … }` blocks)
//! into typed `EffectStmt` fields. None of these lower yet — engine
//! schema work for area expansion, status durations, conditional gates,
//! RNG gates, stack tracking, scaling stat references, voxel
//! lifetimes, and nested-effect dispatch all sit downstream of this
//! parser surface. Until then `lower_effect_stmt` surfaces
//! `LowerError::ModifierNotImplemented` for each populated modifier
//! slot — a deliberate "errors not silent drop" choice so authors don't
//! run with `damage 50 in circle(5)` quietly degrading to a single-
//! target hit. The Wave 1 corpus (Strike / ShieldUp / Mend) uses no
//! modifiers and continues to lower cleanly.
//!
//! Wave 1.1 surfaces (parser-only — lowering deferred):
//! The `.ability` parser now accepts four additional `ability`-block
//! headers (`cost`, `charges`, `recharge`, `toggle`) plus top-level
//! `passive` blocks (spec §4.2 / §5). Lowering of all five surfaces
//! requires engine-side schema changes (cost gates, per-agent charge
//! SoA fields, toggle state, PerEvent dispatch keyed on trigger
//! kinds) and is the work of Wave 2+. Until then this module surfaces
//! `LowerError::HeaderNotImplemented` / `PassiveBlockNotImplemented`
//! when it encounters those parsed surfaces — a deliberate choice
//! over silent acceptance so callers don't quietly miss header
//! semantics. Hero templates that use Wave 1.1 surfaces fail loudly
//! at the lowering boundary rather than running with degraded gates.
//!
//! Constitution touch-points:
//! * P1 (compiler-first): this module IS the compiler step that takes
//!   parser AST -> engine runtime. No interpretation.
//! * P2 (schema-hash): no engine type changes; pure consumer.
//! * P4 (16B EffectOp): no new variants; existing budget intact.

use dsl_ast::ast::{
    AbilityDecl, AbilityFile, AbilityHeader, EffectArg, EffectStmt, HintName, Span, TargetMode,
};
use engine::ability::program::{
    AbilityCost, AbilityHint, AbilityProgram, AbilityTag, Area, CostAmount, CostResource, Delivery, EffectAreaShape, EffectOp, EffectWhenCondition, MAX_NESTED_PER_EFFECT,
    EffectScaling, Gate, LifetimeMode, ScalingStatRef, ShapeKind, StackingMode, TargetSelector,
    MAX_EFFECTS_PER_PROGRAM, MAX_SCALINGS_PER_EFFECT, MAX_TAGS_PER_PROGRAM,
};
use engine::ability::AbilityId;
use smallvec::SmallVec;

/// Errors surfaced by `lower_ability_decl` / `lower_ability_file`.
///
/// Spans point into the original `.ability` source so callers can render
/// the same caret diagnostics the parser emits. `suggestion` on
/// `UnknownEffectVerb` is intentionally `Option<String>` — Wave 1.6 ships
/// without fuzzy-match heuristics; later waves can populate it without an
/// API churn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// `target: <mode>` named a mode the lowering pass does not yet
    /// implement (anything other than `enemy` or `self`).
    TargetModeReserved { mode: String, span: Span },
    /// `hint: <name>` named a category the lowering pass does not yet
    /// implement (today: only `economic` triggers this — the other five
    /// hints map onto `AbilityHint` variants).
    HintReserved { hint: String, span: Span },
    /// Effect verb is not in the Wave 1.6 catalog. `suggestion` is
    /// reserved for a future Levenshtein hint.
    UnknownEffectVerb { verb: String, span: Span, suggestion: Option<String> },
    /// Effect verb received a wrong number of positional arguments
    /// (Wave 1.0 parser drops modifier-tail tokens, so this counts only
    /// the leading scalar args).
    EffectArgMismatch { verb: String, expected: usize, got: usize, span: Span },
    /// Body holds more than `MAX_EFFECTS_PER_PROGRAM` effects.
    BudgetExceeded { ability: String, count: usize, max: usize, span: Span },
    /// Body mixes bare effects with a `deliver { … }` block. Today the
    /// parser rejects `deliver` blocks outright (Wave 1.4 work), so this
    /// is a defensive check kept in place to land alongside that wave.
    MixedBody { ability: String, span: Span },
    /// Wave 1.1 parser accepted a header (`cost`, `charges`, `recharge`,
    /// or `toggle`) whose lowering requires engine-side schema changes
    /// not yet landed. `header` is the literal source key. The error is
    /// surfaced rather than swallowed so authors don't run with silently
    /// degraded gates.
    HeaderNotImplemented { header: &'static str, span: Span },
    /// Wave 1.1 parser accepted a top-level `passive` block; lowering
    /// requires PerEvent dispatch + trigger catalog wiring (Wave 2+).
    PassiveBlockNotImplemented { name: String, span: Span },
    /// Wave 1.5 parser captured one of the nine effect-statement
    /// modifier slots (spec §6.1) into a typed AST field. Lowering of
    /// each slot requires distinct engine work (area expansion, status
    /// durations, conditional gates, RNG gates, stack tracking,
    /// scaling stat refs, voxel lifetimes, nested dispatch) — all
    /// downstream of this parser surface. The error is surfaced rather
    /// than swallowed so authors don't run with silently-degraded
    /// effects.
    ModifierNotImplemented {
        verb:     String,
        /// Slot identifier — one of "tags" / "for" / "when" / "nested".
        /// "stacking" was retired in Wave 1.5#3 (now lowered into
        /// `program.stackings`). "chance" was retired in Wave 1.5#5
        /// (now lowered into `program.chances`). "lifetime" was retired
        /// in Wave 1.5#8 (now lowered into `program.lifetimes`). "in"
        /// was retired in Wave 1.5#2 (now lowered into
        /// `program.per_effect_areas` — unknown shape names surface as
        /// `UnknownShape` instead). "scaling" was retired in Wave
        /// 1.5#4 (now lowered into `program.scalings_per_effect`;
        /// unknown stat-ref names surface as `UnknownStatRef` and
        /// per-effect overflow surfaces as `ScalingBudgetExceeded`).
        modifier: &'static str,
        span:     Span,
    },
    /// Wave 1.5#2 (`in <shape>(args)` modifier): the shape name is not
    /// in the engine's `ShapeKind` vocabulary (12 entries today, spec
    /// §8 catalog: circle/cone/line/ring/spread/box/sphere/column/wall/
    /// cylinder/dome/hull). Surfaced so authors don't silently lose AOE
    /// expansion on a typoed shape name.
    UnknownShape { shape: String, span: Span },
    /// Wave 1.5#4 (`+ N% stat_ref` modifier): the stat_ref token is not
    /// in the engine's `ScalingStatRef` vocabulary (8 entries today:
    /// attack_damage/AD, ability_power/AP, max_hp/MaxHP, hp/HP, armor,
    /// magic_resist/MR, move_speed, mana). Surfaced so authors don't
    /// silently lose scaling on a typoed stat name.
    UnknownStatRef { stat: String, span: Span },
    /// Wave 1.5#4 (`+ N% stat_ref` modifier): an effect declared more
    /// scalings than the per-effect budget allows
    /// (`MAX_SCALINGS_PER_EFFECT == 2` today — fits LoL/MOBA convention
    /// of one or two scaling stats per effect).
    ScalingBudgetExceeded {
        ability: String,
        count:   usize,
        max:     usize,
        span:    Span,
    },
    /// Wave 1.5#9 (nested-effect modifier): an outer effect declared
    /// more nested follow-up effects than the per-effect budget allows
    /// (`MAX_NESTED_PER_EFFECT == 2` today — typical "damage + stun"
    /// combos fit; richer cascades should compose multiple outer
    /// effects instead). Surfaced loudly rather than silently truncating.
    NestedBudgetExceeded {
        ability: String,
        count:   usize,
        max:     usize,
        span:    Span,
    },
    /// Wave 1.5 modifier lowering #1 (tag vocabulary): a `[TAG: value]`
    /// modifier named a tag not in the engine's `AbilityTag` vocabulary
    /// (PHYSICAL/MAGICAL/CROWD_CONTROL/HEAL/DEFENSE/UTILITY today).
    /// Surfaced so authors don't silently lose tag-based scoring weight.
    UnknownTag { tag: String, span: Span },
    /// Wave 1.5 modifier lowering #1 (tag vocabulary): an ability's
    /// effects collectively declared more distinct tags than the
    /// `MAX_TAGS_PER_PROGRAM` budget. Today this matches
    /// `AbilityTag::COUNT == 6` so the only way to trip is duplicating
    /// tag names — but that path lands as a duplicate at the per-effect
    /// site instead. Future-proof for a wider tag vocabulary.
    TagBudgetExceeded { ability: String, count: usize, max: usize, span: Span },
    /// Wave 2 piece 5/6 (`deliver <method> { … }` block): the method
    /// ident is not in the engine's `DeliveryMethodKind` vocabulary
    /// (projectile/channel/zone/chain/tether/trap today). Surfaced
    /// so a typoed method name doesn't silently lower to a no-op.
    UnknownDeliveryMethod {
        ability: String,
        method:  String,
        span:    Span,
    },
    /// Wave 1.4 parser accepted a `morph { effects } into <Other>`
    /// body block. Lowering requires form-swap state + cross-decl
    /// resolution (Wave 2+).
    MorphBlockNotImplemented {
        ability: String,
        into:    String,
        span:    Span,
    },
    /// Wave 1.2 parser accepted a top-level `template <Name>(<params>) { ... }`
    /// block. Template expansion (parameter substitution into `$ident`
    /// references in the body, depth-bounded recursion per spec §11.3)
    /// is Wave 2+ work. Surfaced loudly so authors don't run with a
    /// silently-dropped template definition.
    TemplateBlockNotImplemented {
        name: String,
        span: Span,
    },
    /// Wave 1.2 parser accepted an ability with a `: TemplateName(args)`
    /// instantiation clause. Without the template-expansion engine the
    /// lowering layer can't substitute the args into the template body,
    /// so this surfaces here.
    TemplateInstantiationNotImplemented {
        ability:  String,
        template: String,
        span:     Span,
    },
    /// Wave 1.3 parser accepted a top-level `structure <Name>(<params>) { ... }`
    /// block. Lowering requires voxel storage + rasterization +
    /// `StructureRegistry` (spec §12.2 GPU work) — all Wave 2+ work.
    /// Surfaced loudly so authors don't run with a silently-dropped
    /// structure definition.
    StructureBlockNotImplemented {
        name: String,
        span: Span,
    },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::TargetModeReserved { mode, .. } => write!(
                f,
                "target mode '{mode}' is planned/reserved; not yet supported by lowering"
            ),
            LowerError::HintReserved { hint, .. } => write!(
                f,
                "hint '{hint}' is planned/reserved; not yet supported by lowering"
            ),
            LowerError::UnknownEffectVerb { verb, suggestion, .. } => {
                write!(
                    f,
                    "unknown effect verb '{verb}'; valid verbs at this stage: damage / heal / shield / stun / slow / transfer_gold / modify_standing / cast / root / silence / fear / taunt / dash / blink / knockback / pull / execute / self_damage / lifesteal / damage_modify"
                )?;
                if let Some(s) = suggestion {
                    write!(f, " (did you mean '{s}'?)")?;
                }
                Ok(())
            }
            LowerError::EffectArgMismatch { verb, expected, got, .. } => write!(
                f,
                "effect verb '{verb}' expects {expected} positional arg(s); got {got}"
            ),
            LowerError::BudgetExceeded { ability, count, max, .. } => write!(
                f,
                "ability '{ability}' has {count} effects but the per-program budget is {max} (MAX_EFFECTS_PER_PROGRAM)"
            ),
            LowerError::MixedBody { ability, .. } => write!(
                f,
                "ability '{ability}' mixes bare effect statements with a deliver block; pick one body shape"
            ),
            LowerError::HeaderNotImplemented { header, .. } => write!(
                f,
                "`{header}:` header is parsed but lowering is Wave 2+ (engine schema does not yet carry the field)"
            ),
            LowerError::PassiveBlockNotImplemented { name, .. } => write!(
                f,
                "`passive {name}` is parsed but lowering is Wave 2+ (PerEvent dispatch + trigger catalog not yet wired)"
            ),
            LowerError::ModifierNotImplemented { verb, modifier, .. } => write!(
                f,
                "effect verb `{verb}` carries a `{modifier}` modifier slot that is parsed but lowering is Wave 2+"
            ),
            LowerError::UnknownTag { tag, .. } => write!(
                f,
                "unknown power tag `[{tag}: …]`; valid tags: PHYSICAL / MAGICAL / CROWD_CONTROL / HEAL / DEFENSE / UTILITY"
            ),
            LowerError::TagBudgetExceeded { ability, count, max, .. } => write!(
                f,
                "ability `{ability}` declares {count} distinct power tags but the per-program budget is {max}"
            ),
            LowerError::UnknownDeliveryMethod { ability, method, .. } => write!(
                f,
                "ability `{ability}` uses `deliver {method} {{…}}` — `{method}` is not a known delivery method; valid methods (spec §9): projectile / channel / zone / chain / tether / trap"
            ),
            LowerError::MorphBlockNotImplemented { ability, into, .. } => write!(
                f,
                "ability `{ability}` morphs into `{into}` — parsed by Wave 1.4 but lowering is Wave 2+ (form-swap state machinery not yet wired)"
            ),
            LowerError::TemplateBlockNotImplemented { name, .. } => write!(
                f,
                "`template {name}(…)` is parsed by Wave 1.2 but lowering is Wave 2+ (template expansion engine + `$param` substitution not yet wired)"
            ),
            LowerError::TemplateInstantiationNotImplemented { ability, template, .. } => write!(
                f,
                "ability `{ability}` instantiates `{template}(…)` — parsed by Wave 1.2 but lowering is Wave 2+ (template expansion engine not yet wired)"
            ),
            LowerError::StructureBlockNotImplemented { name, .. } => write!(
                f,
                "`structure {name}` is parsed by Wave 1.3 but lowering is Wave 2+ (voxel rasterization + StructureRegistry per spec §12.2 not yet wired)"
            ),
            LowerError::UnknownShape { shape, .. } => write!(
                f,
                "unknown AOE shape `in {shape}(…)`; valid shapes (spec §8): circle / cone / line / ring / spread / box / sphere / column / wall / cylinder / dome / hull"
            ),
            LowerError::UnknownStatRef { stat, .. } => write!(
                f,
                "unknown scaling stat-ref `+ N% {stat}`; valid stats: attack_damage/AD, ability_power/AP, max_hp/MaxHP, hp/HP, armor, magic_resist/MR, move_speed, mana"
            ),
            LowerError::ScalingBudgetExceeded { ability, count, max, .. } => write!(
                f,
                "ability `{ability}` declares {count} scalings on a single effect but the per-effect budget is {max} (MAX_SCALINGS_PER_EFFECT)"
            ),
            LowerError::NestedBudgetExceeded { ability, count, max, .. } => write!(
                f,
                "ability `{ability}` declares {count} nested effects on a single outer effect but the per-effect budget is {max} (MAX_NESTED_PER_EFFECT)"
            ),
        }
    }
}

impl std::error::Error for LowerError {}

/// Lower every `ability` decl inside an `AbilityFile`. The output preserves
/// declaration order so callers wiring a registry slot table see the same
/// indexing as the source file.
///
/// Errors short-circuit on the first failure — call `lower_ability_decl`
/// directly if you need per-decl error accumulation.
///
/// Wave 1.1: if `file.passives` is non-empty, the first passive is
/// surfaced as `LowerError::PassiveBlockNotImplemented`. Lowering of
/// passives requires PerEvent dispatch wiring (Wave 2+); silent skip
/// would mean an author's `passive Riposte { … }` block compiled away to
/// nothing, which is a worse outcome than a loud error.
pub fn lower_ability_file(file: &AbilityFile) -> Result<Vec<AbilityProgram>, LowerError> {
    if let Some(passive) = file.passives.first() {
        return Err(LowerError::PassiveBlockNotImplemented {
            name: passive.name.clone(),
            span: passive.span,
        });
    }
    // Wave 1.2: top-level `template` blocks parse but expansion lives
    // at Wave 2+. Surface the first one so a silently-dropped template
    // never reaches the registry.
    if let Some(template) = file.templates.first() {
        return Err(LowerError::TemplateBlockNotImplemented {
            name: template.name.clone(),
            span: template.span,
        });
    }
    // Wave 1.3: top-level `structure` blocks parse but voxel
    // rasterization + StructureRegistry (spec §12.2) lives at Wave 2+.
    // Surface the first one so a silently-dropped structure never
    // reaches the registry.
    if let Some(structure) = file.structures.first() {
        return Err(LowerError::StructureBlockNotImplemented {
            name: structure.name.clone(),
            span: structure.span,
        });
    }
    let mut out = Vec::with_capacity(file.abilities.len());
    for decl in &file.abilities {
        out.push(lower_ability_decl(decl)?);
    }
    Ok(out)
}

/// Lower a single `ability <Name> { ... }` decl.
///
/// Header semantics:
/// * `target: enemy` -> `Area::SingleTarget { range: 0.0 }`,
///   `gate.hostile_only = true`. The range is overwritten by a later
///   `range:` header.
/// * `target: self` -> same Area shape, `hostile_only = false`.
/// * Any other `target:` value -> `LowerError::TargetModeReserved`.
/// * `range: <f32>` -> overwrites the SingleTarget range. No bounds check
///   in this slice (negative ranges parse fine — runtime `evaluate_cast_gate`
///   treats them as "always out of range").
/// * `cooldown: <duration>` -> `gate.cooldown_ticks =
///   ceil(millis / 100)` (10 Hz tick).
/// * `cast: <duration>` -> currently dropped; engine `Gate` does not yet
///   carry a `cast_ticks` field. The status matrix in
///   `docs/spec/ability_dsl_unified.md §5.4` flags `cast` as `planned`.
/// * `hint: <name>` -> `Some(AbilityHint::*)`; `economic` is reserved.
///
/// Body semantics: see crate-level docs.
pub fn lower_ability_decl(decl: &AbilityDecl) -> Result<AbilityProgram, LowerError> {
    // -- Wave 1.2: ability with `: TemplateName(args)` instantiation.
    // Without the expansion engine the lowering layer can't substitute
    // args into the template body. Surface BEFORE the header / body
    // passes so authors of `ability Fireball : ElementalBolt(fire, 4.0)
    // { target: ground … }` see the template-not-implemented diagnostic
    // immediately rather than a misleading TargetModeReserved error on
    // a body shape they didn't pick.
    if let Some(inst) = &decl.instantiates {
        return Err(LowerError::TemplateInstantiationNotImplemented {
            ability:  decl.name.clone(),
            template: inst.name.clone(),
            span:     inst.span,
        });
    }

    // -- Header pass: collect the gate / area / hint into mutable scratch
    // values. We resolve `target:` first so a later `range:` can overwrite
    // the SingleTarget's range field. The Wave 1.0 parser already rejects
    // duplicate header keys at parse time, so we don't have to.

    let mut gate = Gate {
        cooldown_ticks: 0,
        hostile_only:   false,
        line_of_sight:  false,
    };
    // Default: zero-range single-target on the caster. If neither
    // `target:` nor `range:` is set, this matches a self-buff with no
    // proximity check.
    let mut area = Area::SingleTarget { range: 0.0 };
    let mut hint: Option<AbilityHint> = None;
    // Wave 1.1 cost header — captured into program.cost. None when
    // the .ability declared no cost header.
    let mut lowered_cost: Option<AbilityCost> = None;
    let mut lowered_charges: Option<u32> = None;
    let mut lowered_recharge_ticks: Option<u32> = None;
    let mut lowered_is_toggle: bool = false;

    for header in &decl.headers {
        match header {
            AbilityHeader::Target(mode) => {
                let (hostile, mode_str) = match mode {
                    TargetMode::Enemy => (true, "enemy"),
                    TargetMode::Self_ => (false, "self"),
                    TargetMode::Ally => return Err(target_reserved("ally", decl)),
                    TargetMode::SelfAoe => return Err(target_reserved("self_aoe", decl)),
                    TargetMode::Ground => return Err(target_reserved("ground", decl)),
                    TargetMode::Direction => return Err(target_reserved("direction", decl)),
                    TargetMode::Vector => return Err(target_reserved("vector", decl)),
                    TargetMode::Global => return Err(target_reserved("global", decl)),
                };
                gate.hostile_only = hostile;
                let _ = mode_str; // kept for future error surfacing.
            }
            AbilityHeader::Range(r) => {
                // Preserve the (currently-only) Area shape and overwrite
                // its range field.
                area = Area::SingleTarget { range: *r };
            }
            AbilityHeader::Cooldown(d) => {
                gate.cooldown_ticks = duration_to_ticks(d.millis);
            }
            AbilityHeader::Cast(_d) => {
                // TODO(wave-1.7+): `Gate` doesn't carry `cast_ticks` yet.
                // The `cast:` header parses but is silently dropped here;
                // the status matrix flags this as `planned` (spec §5.4).
                // When the field lands, store
                // `gate.cast_ticks = duration_to_ticks(d.millis)`.
            }
            AbilityHeader::Hint(h) => {
                hint = Some(map_hint(h, decl)?);
            }
            // Wave 1.1: parser surfaces — lowering is Wave 2+. Each
            // arm carries its own span (where available) so the
            // diagnostic points at the offending source line.
            AbilityHeader::Cost(spec) => {
                // Wave 1.1 cost header (#74 / Wave 2 follow-on): map
                // the AST CostSpec into the engine's AbilityCost slot.
                // Apply handlers debit at cast-decide later (deferred —
                // resource SoA fields like stamina not all wired yet).
                lowered_cost = Some(AbilityCost {
                    resource: match spec.resource {
                        dsl_ast::ast::CostResource::Mana    => CostResource::Mana,
                        dsl_ast::ast::CostResource::Stamina => CostResource::Stamina,
                        dsl_ast::ast::CostResource::Hp      => CostResource::Hp,
                        dsl_ast::ast::CostResource::Gold    => CostResource::Gold,
                    },
                    amount: match spec.amount {
                        dsl_ast::ast::CostAmount::Flat(v)         => CostAmount::Flat(v),
                        dsl_ast::ast::CostAmount::PercentOfMax(v) => CostAmount::PercentOfMax(v),
                    },
                });
            }
            AbilityHeader::Charges(n) => {
                lowered_charges = Some(*n);
            }
            AbilityHeader::Recharge(d) => {
                lowered_recharge_ticks = Some(duration_to_ticks(d.millis));
            }
            AbilityHeader::Toggle => {
                lowered_is_toggle = true;
            }
            // Wave 1.4: parser surfaces — lowering is Wave 2+.
            AbilityHeader::Recast(_) => {
                return Err(LowerError::HeaderNotImplemented {
                    header: "recast",
                    span:   decl.span,
                });
            }
            AbilityHeader::RecastWindow(_) => {
                return Err(LowerError::HeaderNotImplemented {
                    header: "recast_window",
                    span:   decl.span,
                });
            }
        }
    }

    // -- Body-block guard: Wave 1.4 surfaces. Per spec §4.4 / §23.1
    // deliver and bare effects are mutually exclusive — the parser
    // deliberately admits coexistence, lowering enforces. The order
    // here matters:
    //   1. MixedBody (loudest signal — author confused two body
    //      shapes; flag before either deliver or morph errors).
    //   2. DeliverBlockNotImplemented.
    //   3. MorphBlockNotImplemented.
    if decl.deliver.is_some() && !decl.effects.is_empty() {
        return Err(LowerError::MixedBody {
            ability: decl.name.clone(),
            span:    decl.span,
        });
    }
    // Wave 2 piece 5/6: capture the deliver block as
    // Delivery::Method { kind, raw } — the `kind` discriminant lives
    // in engine, the `raw` payload stays opaque. Apply handlers
    // (projectile travel, channel hold-over-time, persistent zone
    // tick) wire later via registry-driven dispatch (#125). Unknown
    // method idents surface as `UnknownDeliveryMethod` so typos
    // don't silently lower to a no-op.
    let lowered_delivery = if let Some(block) = &decl.deliver {
        let kind = engine::ability::program::DeliveryMethodKind::parse(&block.method)
            .ok_or_else(|| LowerError::UnknownDeliveryMethod {
                ability: decl.name.clone(),
                method:  block.method.clone(),
                span:    block.span,
            })?;
        Delivery::Method { kind, raw: block.raw.clone() }
    } else {
        Delivery::Instant
    };
    if let Some(block) = &decl.morph {
        return Err(LowerError::MorphBlockNotImplemented {
            ability: decl.name.clone(),
            into:    block.into.clone(),
            span:    block.span,
        });
    }

    // -- Effect pass.
    if decl.effects.len() > MAX_EFFECTS_PER_PROGRAM {
        return Err(LowerError::BudgetExceeded {
            ability: decl.name.clone(),
            count:   decl.effects.len(),
            max:     MAX_EFFECTS_PER_PROGRAM,
            span:    decl.span,
        });
    }

    let mut effects: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
    // Wave 1.5 modifier lowering #1: aggregate `[TAG: value]` slots
    // across all effects into program.tags (sum-per-tag).
    let mut tag_acc: SmallVec<[(AbilityTag, f32); MAX_TAGS_PER_PROGRAM]> = SmallVec::new();
    // Wave 1.5 modifier lowering #3: per-effect stacking mode, index
    // parallel to `effects`. We ALWAYS push one slot per effect (even
    // `None`) when ANY effect carried a `stacking` modifier — but to
    // avoid serialising an extra column on programs that don't use the
    // modifier at all, we keep the smallvec empty in the all-`None`
    // case. Apply handlers MUST treat empty `program.stackings` and a
    // populated slice with `None` slots both as
    // `StackingMode::Refresh` per `project_buff_stacking_rule.md`.
    let mut stackings_acc: SmallVec<[Option<StackingMode>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut any_stacking = false;
    // Wave 1.5#5: per-effect chance gate, index parallel to `effects`.
    // Same "all-`None` → empty smallvec" optimization the stackings
    // aggregator uses, so Wave 1 corpus output stays bit-stable.
    let mut chances_acc: SmallVec<[Option<u16>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
    let mut any_chance = false;
    // Wave 1.5#8: per-effect lifetime modifier, index parallel to
    // `effects`. Same "all-`None` → empty smallvec" optimization the
    // chances/stackings aggregators use, so Wave 1 corpus output stays
    // bit-stable. The `damageable_hp(N)` variant is the first per-
    // effect modifier we lower with variant data — its f32 hp pool
    // round-trips through `LifetimeMode::DamageableHp(hp)` straight
    // into `program.lifetimes`, then into the SoA pair
    // (`lifetime_kinds` + `lifetime_payloads`) at pack time.
    let mut lifetimes_acc: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut any_lifetime = false;
    // Wave 1.5#2: per-effect AOE shape, index parallel to `effects`.
    // Same "all-`None` → empty smallvec" optimization the other
    // aggregators use, so Wave 1 corpus output stays bit-stable. The
    // shape name is validated against `ShapeKind::parse` here — unknown
    // names surface as `LowerError::UnknownShape` (loud failure rather
    // than a silently-dropped AOE).
    let mut per_effect_areas_acc: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
        SmallVec::new();
    let mut any_area = false;
    // Wave 1.5#4: per-effect scaling list, OUTER-index parallel to
    // `effects`. Each inner SmallVec is bounded at
    // `MAX_SCALINGS_PER_EFFECT`. Same "all-empty → empty outer" optimisation
    // as the other aggregators so the Wave 1 corpus output stays
    // bit-stable. Stat-ref tokens are validated against
    // `ScalingStatRef::parse` here — unknown names surface as
    // `LowerError::UnknownStatRef` (loud failure rather than a
    // silently-dropped scaling). Per-effect overflow surfaces as
    // `LowerError::ScalingBudgetExceeded`.
    // Wave 1.5#7: per-effect when-condition modifier, index parallel to
    // `effects`. Same "all-`None` → empty smallvec" optimization the
    // other aggregators use, so Wave 1 corpus output stays bit-stable.
    // Predicate body is captured as verbatim source text (the AST
    // already stores it that way via `EffectCondition::when_cond:
    // String`); engine code does not parse it here. Apply handlers
    // wire the parse + evaluate later — for now the slot proves the
    // lowering captures the modifier without erroring.
    let mut when_per_effect_acc: SmallVec<
        [Option<EffectWhenCondition>; MAX_EFFECTS_PER_PROGRAM],
    > = SmallVec::new();
    let mut any_when = false;
    // Wave 1.5#9: per-effect nested follow-up effects, OUTER index
    // parallel to `effects`. Each inner SmallVec is bounded at
    // `MAX_NESTED_PER_EFFECT`. Same "all-empty → empty outer"
    // optimisation as scalings, so Wave 1 corpus output stays
    // bit-stable. Each nested EffectStmt is recursively lowered to a
    // bare EffectOp via lower_effect_stmt — its own modifiers
    // (tags/chance/stacking/etc.) are SILENTLY DROPPED today because
    // the per-ability aggregators only capture outer-stmt modifiers.
    // Apply handlers wire the deferred dispatch later (outer resolves
    // first, then nested ops fire); recursive aggregator capture is
    // separate later infrastructure.
    let mut nested_per_effect_acc: SmallVec<
        [SmallVec<[EffectOp; MAX_NESTED_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
    > = SmallVec::new();
    let mut any_nested = false;
    let mut scalings_per_effect_acc: SmallVec<
        [SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
    > = SmallVec::new();
    let mut any_scaling = false;
    for stmt in &decl.effects {
        // Per-effect tags first — fail fast on unknown tag names so the
        // verb-dispatch error doesn't hide them.
        for tag in &stmt.tags {
            let parsed = AbilityTag::parse(&tag.name).ok_or_else(|| LowerError::UnknownTag {
                tag:  tag.name.clone(),
                span: tag.span,
            })?;
            // Sum-per-tag aggregation. Linear scan is fine — at most
            // MAX_TAGS_PER_PROGRAM == 6 entries.
            if let Some(slot) = tag_acc.iter_mut().find(|(t, _)| *t == parsed) {
                slot.1 += tag.value;
            } else {
                if tag_acc.len() >= MAX_TAGS_PER_PROGRAM {
                    return Err(LowerError::TagBudgetExceeded {
                        ability: decl.name.clone(),
                        count:   tag_acc.len() + 1,
                        max:     MAX_TAGS_PER_PROGRAM,
                        span:    tag.span,
                    });
                }
                tag_acc.push((parsed, tag.value));
            }
        }
        // Wave 1.5#3: capture the per-effect stacking mode BEFORE the
        // verb dispatch (which would otherwise discard it). Map the
        // parser AST enum onto the engine enum 1:1.
        let mapped = stmt.stacking.map(map_stacking_mode);
        if mapped.is_some() {
            any_stacking = true;
        }
        stackings_acc.push(mapped);
        // Wave 1.5#5: capture the per-effect `chance N%` BEFORE the
        // verb dispatch. Q16 fixed-point: clamp to `0..=65534` so
        // `u16::MAX` stays reserved as `CHANCE_NONE_SENTINEL`. Authors
        // who want "always" should omit the modifier entirely; an
        // explicit `chance 100%` lowers to `Some(65534)` (one less
        // than the sentinel), which the apply-handler RNG gate
        // (`per_agent_u32(...) & 0xFFFF < q16`) treats as "fires
        // 65534/65536 of the time" — indistinguishable from "always"
        // at the 16-bit RNG resolution.
        let chance = stmt.chance.as_ref().map(|c| {
            (c.p * 65535.0).round().clamp(0.0, 65534.0) as u16
        });
        if chance.is_some() {
            any_chance = true;
        }
        chances_acc.push(chance);
        // Wave 1.5#8: capture the per-effect lifetime modifier BEFORE
        // the verb dispatch (which would otherwise need to ignore it).
        // Map the parser AST enum onto the engine enum 1:1; only the
        // `DamageableHp` variant carries data, threaded through as a
        // bare f32.
        let lifetime = stmt.lifetime.as_ref().map(|lt| match lt {
            dsl_ast::ast::EffectLifetime::UntilCasterDies { .. } => {
                LifetimeMode::UntilCasterDies
            }
            dsl_ast::ast::EffectLifetime::DamageableHp { hp, .. } => {
                LifetimeMode::DamageableHp(*hp)
            }
            dsl_ast::ast::EffectLifetime::BreakOnDamage { .. } => {
                LifetimeMode::BreakOnDamage
            }
        });
        if lifetime.is_some() {
            any_lifetime = true;
        }
        lifetimes_acc.push(lifetime);
        // Wave 1.5#7: capture the per-effect `when <cond>` BEFORE the
        // verb dispatch (which would otherwise reject it via the
        // `ModifierNotImplemented{when}` arm — now removed). Clone
        // both source-text predicates onto the engine slot; engine
        // code does not parse them. Apply handlers wire the parse +
        // evaluate later (deferred infrastructure).
        let when = stmt.condition.as_ref().map(|c| EffectWhenCondition {
            when_cond: c.when_cond.clone(),
            else_cond: c.else_cond.clone(),
        });
        if when.is_some() {
            any_when = true;
        }
        when_per_effect_acc.push(when);
        // Wave 1.5#9: capture the per-effect nested block BEFORE the
        // verb dispatch (which would otherwise reject it via the
        // `ModifierNotImplemented{nested}` arm — now removed).
        // Recursively call lower_effect_stmt on each nested stmt to
        // get its bare EffectOp; inner modifiers silently drop today
        // (see field doc on `nested_per_effect`). Bounded at
        // MAX_NESTED_PER_EFFECT — overflow surfaces as
        // `LowerError::NestedBudgetExceeded`.
        let mut inner_nested: SmallVec<[EffectOp; MAX_NESTED_PER_EFFECT]> = SmallVec::new();
        for nested_stmt in &stmt.nested {
            if inner_nested.len() >= MAX_NESTED_PER_EFFECT {
                return Err(LowerError::NestedBudgetExceeded {
                    ability: decl.name.clone(),
                    count:   inner_nested.len() + 1,
                    max:     MAX_NESTED_PER_EFFECT,
                    span:    nested_stmt.span,
                });
            }
            let nested_op = lower_effect_stmt(nested_stmt)?;
            inner_nested.push(nested_op);
        }
        if !inner_nested.is_empty() {
            any_nested = true;
        }
        nested_per_effect_acc.push(inner_nested);
        // Wave 1.5#2: capture the per-effect `in <shape>(args)`
        // BEFORE the verb dispatch (which would otherwise discard
        // it). Validate the shape name against the engine's
        // `ShapeKind` vocabulary; unknown names surface as
        // `UnknownShape` so a typoed shape doesn't silently degrade
        // to a single-target hit. Wall consumes 4 args; other shapes
        // take fewer and zero-pad — `args.iter().take(4)` clips any
        // overlong author input rather than erroring (spec §8 arity
        // mismatches are surfaced at parse time when implemented).
        let area = if let Some(a) = &stmt.area {
            let kind = ShapeKind::parse(&a.shape).ok_or_else(|| LowerError::UnknownShape {
                shape: a.shape.clone(),
                span:  a.span,
            })?;
            let mut args = [0.0_f32; 4];
            for (i, v) in a.args.iter().take(4).enumerate() {
                args[i] = *v;
            }
            Some(EffectAreaShape { kind, args })
        } else {
            None
        };
        if area.is_some() {
            any_area = true;
        }
        per_effect_areas_acc.push(area);
        // Wave 1.5#4: capture the per-effect scaling list BEFORE the
        // verb dispatch (which would otherwise discard it). Validate
        // each stat_ref token against the engine's `ScalingStatRef`
        // vocabulary; unknown names surface as `UnknownStatRef`.
        // Bounded at `MAX_SCALINGS_PER_EFFECT` per effect — overflow
        // surfaces as `ScalingBudgetExceeded`. The parser stores
        // `percent` as `N` (e.g. `30` for `+ 30% AP`); the engine
        // stores it as a fraction (`0.30`) so apply handlers can
        // multiply directly without an extra `/ 100.0`.
        let mut inner: SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]> = SmallVec::new();
        for sc in &stmt.scalings {
            let stat_ref = ScalingStatRef::parse(&sc.stat_ref).ok_or_else(|| {
                LowerError::UnknownStatRef {
                    stat: sc.stat_ref.clone(),
                    span: sc.span,
                }
            })?;
            if inner.len() >= MAX_SCALINGS_PER_EFFECT {
                return Err(LowerError::ScalingBudgetExceeded {
                    ability: decl.name.clone(),
                    count:   inner.len() + 1,
                    max:     MAX_SCALINGS_PER_EFFECT,
                    span:    sc.span,
                });
            }
            inner.push(EffectScaling {
                stat_ref,
                percent: sc.percent / 100.0,
            });
        }
        if !inner.is_empty() {
            any_scaling = true;
        }
        scalings_per_effect_acc.push(inner);
        let op = lower_effect_stmt(stmt)?;
        effects.push(op);
    }

    // Discard the all-`None` aggregator so the resulting program looks
    // identical to the Wave 1 corpus output for sources that don't use
    // the modifier. Apply handlers + the SoA packer treat empty +
    // all-`None` identically.
    if !any_stacking {
        stackings_acc.clear();
    }
    if !any_chance {
        chances_acc.clear();
    }
    if !any_lifetime {
        lifetimes_acc.clear();
    }
    if !any_area {
        per_effect_areas_acc.clear();
    }
    if !any_scaling {
        scalings_per_effect_acc.clear();
    }
    if !any_when {
        when_per_effect_acc.clear();
    }
    if !any_nested {
        nested_per_effect_acc.clear();
    }

    Ok(AbilityProgram {
        delivery: lowered_delivery,
        area,
        gate,
        effects,
        hint,
        tags: tag_acc,
        stackings: stackings_acc,
        chances: chances_acc,
        lifetimes: lifetimes_acc,
        per_effect_areas: per_effect_areas_acc,
        scalings_per_effect: scalings_per_effect_acc,
        when_per_effect: when_per_effect_acc,
        nested_per_effect: nested_per_effect_acc,
        cost: lowered_cost,
        charges: lowered_charges,
        recharge_ticks: lowered_recharge_ticks,
        is_toggle: lowered_is_toggle,
    })
}

/// Lower a single `EffectStmt` to one `EffectOp`. The verb dispatch is
/// hand-rolled because the cast-to-`EffectOp` shape varies per verb
/// (different arity, different argument types).
///
/// Wave 1.5 modifier slots: any populated modifier slot (spec §6.1)
/// produces `LowerError::ModifierNotImplemented` BEFORE the verb
/// dispatch fires, so authors get the same surface diagnostic
/// regardless of verb. Slot-check order matches the spec §6.1 list so
/// the error message is stable.
///
/// Unknown verbs and verb/arity mismatches surface via `LowerError`.
fn lower_effect_stmt(stmt: &EffectStmt) -> Result<EffectOp, LowerError> {
    // Wave 1.5: short-circuit on the first populated modifier slot.
    // The slot order mirrors spec §6.1's evaluation order so the
    // diagnostic an author sees is the "lowest-numbered" unimplemented
    // slot, not whichever the dispatch happens to trip over.
    // Wave 1.5#2: `in <shape>(args)` is consumed by the per-ability
    // aggregator in `lower_ability_decl` (one slot per effect, parallel
    // to `program.effects`). The verb dispatch below stays oblivious —
    // apply handlers read `program.per_effect_areas[i]` to expand the
    // effect to AOE. Unknown shape names surface from the aggregator
    // as `LowerError::UnknownShape`. No short-circuit here.
    // Tags handled by the per-ability aggregator in `lower_ability_decl`
    // (Wave 1.5 modifier lowering #1) — no short-circuit here.
    // Wave 1.5 modifier #6 (`for <duration>`): for the eight
    // duration-bearing verbs (stun/slow/root/silence/fear/taunt/
    // lifesteal/damage_modify) the modifier acts as a duration source —
    // their per-verb arms below detect `stmt.duration.is_some()` and
    // pull the duration from there instead of a positional. For the
    // other verbs (damage/heal/shield/cast/transfer_gold/modify_standing)
    // the modifier means DoT/HoT semantics — a NEW EffectOp variant is
    // needed (DamageOverTime / HealOverTime), so still surface the
    // ModifierNotImplemented for those.
    if let Some(d) = &stmt.duration {
        if !is_duration_bearing_verb(&stmt.verb) {
            return Err(LowerError::ModifierNotImplemented {
                verb:     stmt.verb.clone(),
                modifier: "for",
                span:     d.span,
            });
        }
    }
    // Wave 1.5#7: `when <cond> [else <cond>]` is consumed by the
    // per-ability aggregator in `lower_ability_decl` (one slot per
    // effect, parallel to `program.effects`). The verb dispatch
    // below stays oblivious — apply handlers later parse + evaluate
    // `program.when_per_effect[i]` as a runtime predicate. No
    // short-circuit here.
    // Wave 1.5#5: `chance N%` is consumed by the per-ability
    // aggregator in `lower_ability_decl` (one slot per effect, parallel
    // to `program.effects`). The verb dispatch below stays oblivious —
    // apply handlers read `program.chances[i]` to decide whether to fire
    // the effect this tick. No short-circuit here.
    // Wave 1.5#3: `stacking <mode>` is consumed by the per-ability
    // aggregator in `lower_ability_decl` (one slot per effect, parallel
    // to `program.effects`). The verb dispatch below stays oblivious —
    // apply handlers read `program.stackings[i]` to pick a policy. No
    // short-circuit here.
    // Wave 1.5#4: `+ N% stat_ref` is consumed by the per-ability
    // aggregator in `lower_ability_decl` (one per-effect SmallVec slot
    // parallel to `program.effects`, each holding up to
    // MAX_SCALINGS_PER_EFFECT entries). The verb dispatch below stays
    // oblivious — apply handlers read `program.scalings_per_effect[i]`
    // to add `stat * percent` to the effect's flat amount. Unknown
    // stat-ref names surface from the aggregator as
    // `LowerError::UnknownStatRef`; per-effect overflow surfaces as
    // `LowerError::ScalingBudgetExceeded`. No short-circuit here.
    // Wave 1.5#8: `until_caster_dies` / `damageable_hp(N)` /
    // `break_on_damage` are consumed by the per-ability aggregator in
    // `lower_ability_decl` (one slot per effect, parallel to
    // `program.effects`). The verb dispatch below stays oblivious —
    // apply handlers read `program.lifetimes[i]` to pick the lifetime
    // semantic. No short-circuit here.
    // Wave 1.5#9: `{ <inner_stmt>; ... }` nested follow-up effects
    // are consumed by the per-ability aggregator in
    // `lower_ability_decl` (one inner SmallVec per outer effect,
    // parallel to `program.effects`). The verb dispatch below stays
    // oblivious — apply handlers later read
    // `program.nested_per_effect[i]` and fire each inner op after
    // the outer one resolves. No short-circuit here.
    match stmt.verb.as_str() {
        "damage" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Damage { amount })
        }
        "heal" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Heal { amount })
        }
        "shield" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Shield { amount })
        }
        "stun" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Stun { duration_ticks: duration_to_ticks(dur) })
        }
        // Wave 2 piece 1 — four new control verbs. Each takes a single
        // `<duration>` arg and lowers to the matching `EffectOp::*`
        // variant; runtime gating (move/cast suppression, flee/taunt
        // intent rerouting) lands in later Wave 2 pieces alongside the
        // mask-builder updates.
        "root" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Root { duration_ticks: duration_to_ticks(dur) })
        }
        "silence" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Silence { duration_ticks: duration_to_ticks(dur) })
        }
        "fear" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Fear { duration_ticks: duration_to_ticks(dur) })
        }
        "taunt" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Taunt { duration_ticks: duration_to_ticks(dur) })
        }
        // Wave 2 piece 2 — four new movement verbs. Each takes a single
        // `<distance:f32>` arg (mirrors `damage`'s shape, NOT the
        // duration shape of the control verbs) and lowers to the
        // matching `EffectOp::*` variant. Apply handlers (compute
        // facing direction / away-from-caster / toward-caster vectors
        // and update `hot_pos`) land in a follow-up Wave 2 piece.
        "dash" => {
            let dist = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Dash { distance: dist })
        }
        "blink" => {
            let dist = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Blink { distance: dist })
        }
        "knockback" => {
            let dist = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Knockback { distance: dist })
        }
        "pull" => {
            let dist = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Pull { distance: dist })
        }
        // Wave 2 piece 3 — two new advanced verbs. Each takes a single
        // `<f32>` arg (mirrors `damage`'s shape exactly). NEITHER adds
        // SoA fields:
        //   * `execute <hp_threshold>` reads existing `hot_hp` and
        //     emits a Defeated/AgentDied event when target.hp <
        //     hp_threshold. Apply handler is per-fixture work — Wave 2
        //     piece N.
        //   * `self_damage <amount>` re-emits a Damaged{source: caster,
        //     target: caster, amount} event the existing ApplyDamage
        //     chronicle already drains. No new state.
        "execute" => {
            let threshold = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Execute { hp_threshold: threshold })
        }
        "self_damage" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::SelfDamage { amount })
        }
        // Wave 2 piece 4 — two new buff verbs. Both mirror `slow`'s shape:
        // `<f32 magnitude> <duration>` → `EffectOp::* { duration_ticks,
        // <field>_q8 }` with the same q8 pack rule (`magnitude * 256`,
        // clamped into `i16::MIN..=i16::MAX`). Apply-handler stacking
        // semantics (single per-agent slot, max-with-duration-tiebreak)
        // are documented on the SoA fields and follow the project
        // buff-stacking rule.
        "lifesteal" => {
            let fraction = require_number_arg(stmt, 0)?;
            let (dur, arity) = extract_duration(stmt, 1, 2)?;
            require_arity(stmt, arity)?;
            let fraction_q8 = (fraction * 256.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::LifeSteal {
                duration_ticks: duration_to_ticks(dur),
                fraction_q8,
            })
        }
        "damage_modify" => {
            let mult = require_number_arg(stmt, 0)?;
            let (dur, arity) = extract_duration(stmt, 1, 2)?;
            require_arity(stmt, arity)?;
            let multiplier_q8 = (mult * 256.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::DamageModify {
                duration_ticks: duration_to_ticks(dur),
                multiplier_q8,
            })
        }
        "slow" => {
            // `slow <factor:f32> <duration>` — two positional args. The
            // engine packs `factor` into a Q8 fixed-point i16 (factor *
            // 256) so 1.0 == 256. Wave 1.5#6: `slow 0.5 for 4s` is also
            // accepted — duration comes from the `for` modifier instead.
            let factor = require_number_arg(stmt, 0)?;
            let (dur, arity) = extract_duration(stmt, 1, 2)?;
            require_arity(stmt, arity)?;
            let factor_q8 = (factor * 256.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::Slow {
                duration_ticks: duration_to_ticks(dur),
                factor_q8,
            })
        }
        "transfer_gold" => {
            let amt = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            // Gold transfers are integer; round-half-to-even style cast
            // is fine because the parser already rejects fractional
            // tokens that aren't ints (the `EffectArg::Number` branch
            // accepts both). Preserve the sign.
            Ok(EffectOp::TransferGold { amount: amt.round() as i32 })
        }
        "modify_standing" => {
            let delta = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            // The current EffectOp variant is the legacy single-i16
            // shape. Wave 3 evolves to {a_sel, b_sel, delta}; that
            // requires a schema-hash bump and lives with the verb-pair
            // refactor. Clamp here so the cast can't overflow at
            // parse-time corner cases.
            let clamped = delta.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::ModifyStanding { delta: clamped })
        }
        "cast" => {
            // `cast <ability_name>` — the inner ability is resolved at
            // registry-wiring time (Wave 1.7). We accept either a bare
            // identifier or a string here so the parser's `Ident` /
            // `String` distinction does not leak into the lowering API.
            //
            // Selector is fixed to `Caster` for now per the Wave 1.6
            // brief — the spec leaves selector control for the future
            // `cast_on <selector>` modifier (Wave 2+).
            //
            // TODO(wave-1.7): take the resolved `AbilityId` from the
            // registry name table here. Until then we emit the smallest
            // valid id (`AbilityId::new(1)`) as a placeholder so the
            // program survives the size-budget test.
            let _name = require_name_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            let placeholder = AbilityId::new(1).expect("AbilityId::new(1) is always Some");
            Ok(EffectOp::CastAbility {
                ability: placeholder,
                selector: TargetSelector::Caster,
            })
        }
        _ => Err(LowerError::UnknownEffectVerb {
            verb:       stmt.verb.clone(),
            span:       stmt.span,
            suggestion: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Small helpers — kept private; the public surface is the two `lower_*`
// functions above plus `LowerError`.
// ---------------------------------------------------------------------------

/// Convert a duration in milliseconds to ticks at the engine's 10 Hz
/// (100 ms / tick) cadence. Rounds up so a 1 ms cooldown still costs at
/// least one tick — matches the spec's "ceil(millis / 100)" rule.
fn duration_to_ticks(millis: u32) -> u32 {
    if millis == 0 {
        0
    } else {
        // ceil(m / 100) without floats.
        (millis.saturating_add(99)) / 100
    }
}

fn map_hint(h: &HintName, decl: &AbilityDecl) -> Result<AbilityHint, LowerError> {
    match h {
        HintName::Damage => Ok(AbilityHint::Damage),
        HintName::Defense => Ok(AbilityHint::Defense),
        HintName::CrowdControl => Ok(AbilityHint::CrowdControl),
        HintName::Utility => Ok(AbilityHint::Utility),
        // The engine `AbilityHint` does not carry a `Heal` variant today;
        // the closest scoring bucket is `Defense` per `docs/spec/ability_dsl_unified.md §4.2`.
        // Routing `heal` -> `Defense` keeps scoring rows that read the
        // hint deterministic; if/when the engine grows a `Heal` variant
        // (schema-hash bump) update both arms.
        HintName::Heal => Ok(AbilityHint::Defense),
        // The engine `AbilityHint` does not carry a `Buff` variant today;
        // route to `Utility` per `docs/spec/ability_dsl_unified.md §4.2`
        // (buffs and other ally-empowering effects share the utility
        // scoring bucket). If/when the engine grows a dedicated `Buff`
        // variant (schema-hash bump) update both arms.
        HintName::Buff => Ok(AbilityHint::Utility),
        HintName::Economic => Err(LowerError::HintReserved {
            hint: "economic".to_string(),
            span: decl.span,
        }),
    }
}

fn target_reserved(mode: &str, decl: &AbilityDecl) -> LowerError {
    LowerError::TargetModeReserved { mode: mode.to_string(), span: decl.span }
}

/// Wave 1.5#3: map the parser AST `StackingMode` (Refresh / Stack /
/// Extend) onto the engine runtime `StackingMode` enum. The variants
/// are 1:1 today; this helper exists so a future divergence (the spec
/// hints at `additive` / `max` aliases mapping to `Stack` / `Refresh`
/// per the project_buff_stacking_rule memo) lands in one place.
#[inline]
fn map_stacking_mode(m: dsl_ast::ast::StackingMode) -> StackingMode {
    match m {
        dsl_ast::ast::StackingMode::Refresh => StackingMode::Refresh,
        dsl_ast::ast::StackingMode::Stack => StackingMode::Stack,
        dsl_ast::ast::StackingMode::Extend => StackingMode::Extend,
    }
}

fn require_arity(stmt: &EffectStmt, expected: usize) -> Result<(), LowerError> {
    if stmt.args.len() != expected {
        return Err(LowerError::EffectArgMismatch {
            verb:     stmt.verb.clone(),
            expected,
            got:      stmt.args.len(),
            span:     stmt.span,
        });
    }
    Ok(())
}

fn require_number_arg(stmt: &EffectStmt, idx: usize) -> Result<f32, LowerError> {
    match stmt.args.get(idx) {
        Some(EffectArg::Number(v)) => Ok(*v),
        // `transfer_gold 50` parses as `Number`. `damage 30%` would parse
        // as `Percent` — Wave 1.6 doesn't accept percents on these verbs
        // (the spec catalog locks them to absolute scalars), so fall
        // through to a clean error.
        Some(_) | None => Err(LowerError::EffectArgMismatch {
            verb:     stmt.verb.clone(),
            // `expected` reports the total positional arg count from the
            // caller's perspective — the missing/wrong-typed arg will
            // surface via the per-verb arity check just after.
            expected: stmt.args.len().max(idx + 1),
            got:      stmt.args.len(),
            span:     stmt.span,
        }),
    }
}

fn require_duration_arg(stmt: &EffectStmt, idx: usize) -> Result<u32, LowerError> {
    match stmt.args.get(idx) {
        Some(EffectArg::Duration(d)) => Ok(d.millis),
        Some(_) | None => Err(LowerError::EffectArgMismatch {
            verb:     stmt.verb.clone(),
            expected: stmt.args.len().max(idx + 1),
            got:      stmt.args.len(),
            span:     stmt.span,
        }),
    }
}

/// Wave 1.5#6: extract a duration arg for a stateful verb, preferring
/// the `for <duration>` modifier if present; fall back to the
/// `positional_idx`th positional arg. Returns the duration in millis
/// AND the arity to enforce on the remaining positional args.
///
/// `full_arity` is the verb's positional-arg count when no modifier is
/// present (e.g. 1 for Stun, 2 for Slow). When the modifier IS present
/// the verb consumes one fewer positional, so arity drops by 1.
fn extract_duration(
    stmt:           &EffectStmt,
    positional_idx: usize,
    full_arity:     usize,
) -> Result<(u32, usize), LowerError> {
    if let Some(d) = &stmt.duration {
        // Caller's `is_duration_bearing_verb` short-circuit guards
        // upstream — by the time we land here for one of the 8 verbs,
        // the modifier IS the duration source. Saturating-sub for
        // robustness against future zero-arity verbs.
        Ok((d.duration.millis, full_arity.saturating_sub(1)))
    } else {
        let dur = require_duration_arg(stmt, positional_idx)?;
        Ok((dur, full_arity))
    }
}

/// Wave 1.5#6: returns true iff the verb consumes a `for <duration>`
/// modifier as its duration source instead of erroring with
/// ModifierNotImplemented{for}. The other verbs (damage/heal/shield/
/// cast/transfer_gold/modify_standing) get DoT/HoT semantics from
/// `for` — those need new EffectOp variants and stay errored.
fn is_duration_bearing_verb(verb: &str) -> bool {
    matches!(
        verb,
        "stun" | "slow" | "root" | "silence" | "fear" | "taunt"
        | "lifesteal" | "damage_modify"
    )
}

fn require_name_arg(stmt: &EffectStmt, idx: usize) -> Result<String, LowerError> {
    match stmt.args.get(idx) {
        Some(EffectArg::Ident(n)) => Ok(n.clone()),
        Some(EffectArg::String(s)) => Ok(s.clone()),
        Some(_) | None => Err(LowerError::EffectArgMismatch {
            verb:     stmt.verb.clone(),
            expected: stmt.args.len().max(idx + 1),
            got:      stmt.args.len(),
            span:     stmt.span,
        }),
    }
}
