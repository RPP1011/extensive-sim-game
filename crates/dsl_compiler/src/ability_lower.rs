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
    AbilityHint, AbilityProgram, AbilityTag, Area, Delivery, EffectOp, Gate,
    MAX_EFFECTS_PER_PROGRAM, MAX_TAGS_PER_PROGRAM, TargetSelector,
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
        /// Slot identifier — one of "in" / "tags" / "for" / "when" /
        /// "chance" / "stacking" / "scaling" / "lifetime" / "nested".
        modifier: &'static str,
        span:     Span,
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
    /// Wave 1.4 parser accepted a `deliver <method> { params } { body }`
    /// body block. Lowering requires delivery-method SoA buffers +
    /// on_hit / on_arrival / on_tick hook dispatch (spec §9), all Wave
    /// 2+ work. Surfaced loudly so authors don't run with silently
    /// dropped delivery semantics.
    DeliverBlockNotImplemented {
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
            LowerError::DeliverBlockNotImplemented { ability, method, .. } => write!(
                f,
                "ability `{ability}` uses `deliver {method} {{…}}` — parsed by Wave 1.4 but lowering is Wave 2+ (delivery-method SoA + on_hit/on_arrival/on_tick hook dispatch not yet wired)"
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
                return Err(LowerError::HeaderNotImplemented {
                    header: "cost",
                    span:   spec.span,
                });
            }
            AbilityHeader::Charges(_) => {
                return Err(LowerError::HeaderNotImplemented {
                    header: "charges",
                    span:   decl.span,
                });
            }
            AbilityHeader::Recharge(_) => {
                return Err(LowerError::HeaderNotImplemented {
                    header: "recharge",
                    span:   decl.span,
                });
            }
            AbilityHeader::Toggle => {
                return Err(LowerError::HeaderNotImplemented {
                    header: "toggle",
                    span:   decl.span,
                });
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
    if let Some(block) = &decl.deliver {
        return Err(LowerError::DeliverBlockNotImplemented {
            ability: decl.name.clone(),
            method:  block.method.clone(),
            span:    block.span,
        });
    }
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
        let op = lower_effect_stmt(stmt)?;
        effects.push(op);
    }

    Ok(AbilityProgram {
        delivery: Delivery::Instant,
        area,
        gate,
        effects,
        hint,
        tags: tag_acc,
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
    if let Some(area) = &stmt.area {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "in",
            span:     area.span,
        });
    }
    // Tags handled by the per-ability aggregator in `lower_ability_decl`
    // (Wave 1.5 modifier lowering #1) — no short-circuit here.
    if let Some(d) = &stmt.duration {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "for",
            span:     d.span,
        });
    }
    if let Some(cond) = &stmt.condition {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "when",
            span:     cond.span,
        });
    }
    if let Some(ch) = &stmt.chance {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "chance",
            span:     ch.span,
        });
    }
    if stmt.stacking.is_some() {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "stacking",
            span:     stmt.span,
        });
    }
    if let Some(s) = stmt.scalings.first() {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "scaling",
            span:     s.span,
        });
    }
    if let Some(lt) = &stmt.lifetime {
        let span = match lt {
            dsl_ast::ast::EffectLifetime::UntilCasterDies { span } => *span,
            dsl_ast::ast::EffectLifetime::DamageableHp { span, .. } => *span,
            dsl_ast::ast::EffectLifetime::BreakOnDamage { span } => *span,
        };
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "lifetime",
            span,
        });
    }
    if !stmt.nested.is_empty() {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "nested",
            span:     stmt.nested[0].span,
        });
    }
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
            let dur = require_duration_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Stun { duration_ticks: duration_to_ticks(dur) })
        }
        // Wave 2 piece 1 — four new control verbs. Each takes a single
        // `<duration>` arg and lowers to the matching `EffectOp::*`
        // variant; runtime gating (move/cast suppression, flee/taunt
        // intent rerouting) lands in later Wave 2 pieces alongside the
        // mask-builder updates.
        "root" => {
            let dur = require_duration_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Root { duration_ticks: duration_to_ticks(dur) })
        }
        "silence" => {
            let dur = require_duration_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Silence { duration_ticks: duration_to_ticks(dur) })
        }
        "fear" => {
            let dur = require_duration_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Fear { duration_ticks: duration_to_ticks(dur) })
        }
        "taunt" => {
            let dur = require_duration_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
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
            let dur = require_duration_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let fraction_q8 = (fraction * 256.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::LifeSteal {
                duration_ticks: duration_to_ticks(dur),
                fraction_q8,
            })
        }
        "damage_modify" => {
            let mult = require_number_arg(stmt, 0)?;
            let dur = require_duration_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let multiplier_q8 = (mult * 256.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::DamageModify {
                duration_ticks: duration_to_ticks(dur),
                multiplier_q8,
            })
        }
        "slow" => {
            // `slow <factor:f32> <duration>` — two positional args. The
            // engine packs `factor` into a Q8 fixed-point i16 (factor *
            // 256) so 1.0 == 256.
            let factor = require_number_arg(stmt, 0)?;
            let dur = require_duration_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
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
