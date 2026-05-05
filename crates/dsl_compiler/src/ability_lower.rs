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
//! * **Effect verbs covered (8 of the 27 catalog entries):** `damage`,
//!   `heal`, `shield`, `stun`, `slow`, `transfer_gold`,
//!   `modify_standing`, `cast`. These match the existing 8 `EffectOp`
//!   variants on the engine side. Unknown verbs / arity mismatches are
//!   surfaced as errors.
//!
//! * **Out of scope (deferred to later waves):**
//!     - `deliver` / `recast` / `morph` body blocks — Wave 1.4.
//!     - `template` / `structure` top-level blocks — Waves 1.2 / 1.3.
//!     - Other target modes (ally/self_aoe/ground/direction/vector/global)
//!       and `economic` hint — error today, wired by their respective
//!       waves.
//!     - The remaining 19 EffectOp variants (Knockback, Teleport, ApplyStatus,
//!       SummonAlly, etc.) — Waves 2-5.
//!     - Two-phase split validator + ability-name resolution for
//!       `cast <Name>` — Wave 1.7 (registry wiring).
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
    AbilityHint, AbilityProgram, Area, Delivery, EffectOp, Gate, MAX_EFFECTS_PER_PROGRAM,
    TargetSelector,
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
                    "unknown effect verb '{verb}'; valid verbs at this stage: damage / heal / shield / stun / slow / transfer_gold / modify_standing / cast"
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
        }
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
    for stmt in &decl.effects {
        let op = lower_effect_stmt(stmt)?;
        effects.push(op);
    }

    Ok(AbilityProgram {
        delivery: Delivery::Instant,
        area,
        gate,
        effects,
        hint,
        tags: SmallVec::new(),
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
    if let Some(tag) = stmt.tags.first() {
        return Err(LowerError::ModifierNotImplemented {
            verb:     stmt.verb.clone(),
            modifier: "tags",
            span:     tag.span,
        });
    }
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
