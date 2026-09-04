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
//! * **Effect verbs covered (21 of the 27 catalog entries):** `damage`,
//!   `heal`, `shield`, `stun`, `slow`, `transfer_gold`,
//!   `modify_standing`, `cast`, the Wave 2 piece 1 control verbs
//!   `root`, `silence`, `fear`, `taunt`, the Wave 2 piece 2
//!   movement verbs `dash`, `blink`, `knockback`, `pull`, the
//!   Wave 2 piece 3 advanced verbs `execute`, `self_damage`, the
//!   Wave 2 piece 4 buff verbs `lifesteal`, `damage_modify`, plus the
//!   LoL-corpus `summon "<template>"` verb. These match the 21
//!   amount-bearing / status / movement / summon EffectOp variants on
//!   the engine side. Unknown verbs / arity mismatches are surfaced
//!   as errors.
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
    AbilityDecl, AbilityFile, AbilityHeader, AbilityProgramStep, CastSpec, EffectArg, EffectStmt,
    HintName, InterruptKind as AstInterruptKind, InterruptSet, Span, TargetMode,
};
use engine::ability::interrupt::{InterruptKind, InterruptMask};
use engine::ability::program::{
    AbilityCost, AbilityHint, AbilityProgram, AbilityTag, Area, CostAmount, CostResource, Delivery, EffectAreaShape, EffectOp, EffectPredicate, EffectPredicateBinder, EffectPredicateOp, EffectWhenCondition, MAX_NESTED_PER_EFFECT, MAX_PRED_NODES_PER_EFFECT, TargetModeKind,
    EffectScaling, Gate, LifetimeMode, ScalingStatRef, ShapeKind, StackingMode, TargetSelector, TelegraphKind, TELEGRAPH_KIND_NONE, WhenPredicate,
    MAX_EFFECTS_PER_PROGRAM, MAX_SCALINGS_PER_EFFECT, MAX_TAGS_PER_PROGRAM,
};
use engine::ability::AbilityId;
use smallvec::SmallVec;

/// Resolve a parsed [`InterruptSet`] to the engine-side packed
/// [`InterruptMask`]. Plan G G2.5 — the lowering captures the
/// declared interrupt set at compile time so the busy-resolution
/// kernel sees a fixed bitmask per ability rather than re-evaluating
/// the AST shape at runtime.
fn lower_interrupt_set(set: &InterruptSet) -> InterruptMask {
    fn ast_to_engine(k: AstInterruptKind) -> InterruptKind {
        match k {
            AstInterruptKind::Damage     => InterruptKind::Damage,
            AstInterruptKind::Stun       => InterruptKind::Stun,
            AstInterruptKind::CasterDied => InterruptKind::CasterDied,
            AstInterruptKind::TargetDied => InterruptKind::TargetDied,
            AstInterruptKind::Movement   => InterruptKind::Movement,
        }
    }
    match set {
        InterruptSet::None     => InterruptMask::none(),
        InterruptSet::Standard => InterruptMask::standard(),
        InterruptSet::Subset(kinds) => {
            let mut m = InterruptMask::none();
            for k in kinds { m = m.with(ast_to_engine(*k)); }
            m
        }
        InterruptSet::StandardPlus(kinds) => {
            let mut m = InterruptMask::standard();
            for k in kinds { m = m.with(ast_to_engine(*k)); }
            m
        }
        InterruptSet::StandardMinus(kinds) => {
            let mut m = InterruptMask::standard();
            for k in kinds { m = m.without(ast_to_engine(*k)); }
            m
        }
    }
}

/// Parse the verbatim telegraph source slice (e.g.
/// `"circle(self.pos, radius: 4)"` or `"line(self.pos, target.pos,
/// width: 2)"`) into a `(TelegraphKind, [f32; 4])` tuple ready for the
/// packed registry's `telegraph_kind` / `telegraph_params` columns.
///
/// Recognises only `circle(...)` and `line(...)` shape names and pulls
/// the single named numeric arg (`radius:` for circles, `width:` for
/// lines) into `params[0]`. Position arguments (`self.pos`, `target.pos`)
/// are NOT parsed here — they're implicit in the threats fold's
/// projection step (centre = caster pos for circle; endpoints = caster +
/// target pos for line).
///
/// Returns `None` for unparseable text (unknown shape, missing named
/// arg, malformed literal). Caller treats `None` as "no telegraph"
/// (sentinel + zeros). A future slice could promote this to a typed
/// `LowerError::UnknownTelegraphShape` once the threats view
/// (G3g) hard-depends on the metadata.
fn parse_telegraph_text(text: &str) -> Option<(TelegraphKind, [f32; 4])> {
    let text = text.trim();
    let (head, rest) = text.split_once('(')?;
    let head = head.trim();
    let rest = rest.strip_suffix(')')?.trim();
    let kind = match head {
        "circle" => TelegraphKind::Circle,
        "line"   => TelegraphKind::Line,
        _ => return None,
    };
    let key = match kind {
        TelegraphKind::Circle => "radius",
        TelegraphKind::Line   => "width",
    };
    // Find `<key>:` then parse the value up to the next `,` / end.
    // Args may appear in any order — we only need the named numeric.
    for arg in rest.split(',') {
        let arg = arg.trim();
        if let Some((k, v)) = arg.split_once(':') {
            if k.trim() == key {
                if let Ok(n) = v.trim().parse::<f32>() {
                    let mut params = [0.0_f32; 4];
                    params[0] = n;
                    return Some((kind, params));
                }
            }
        }
    }
    None
}

/// Errors surfaced by `lower_ability_decl` / `lower_ability_file`.
///
/// Spans point into the original `.ability` source so callers can render
/// the same caret diagnostics the parser emits. `suggestion` on
/// `UnknownEffectVerb` is intentionally `Option<String>` — Wave 1.6 ships
/// without fuzzy-match heuristics; later waves can populate it without an
/// API churn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
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
    /// A status-effect verb that wants a `Duration` arg (with a time
    /// suffix — `1s`, `500ms`) received a bare integer instead. The
    /// parser captures `stun 8` as `EffectArg::Number(8.0)`, which is
    /// structurally correct (one positional arg) but semantically wrong
    /// — the lowering needs millis, and an unsuffixed integer is
    /// ambiguous (ticks? seconds? millis?). Pre-this-variant the
    /// failure surfaced as `EffectArgMismatch { expected: 1, got: 1 }`,
    /// which is structurally false (both sides are 1) and gave the
    /// designer no hint about the missing time suffix. The new variant
    /// is purely a diagnostic improvement — no engine impact.
    ///
    /// `got_value` is the parser's f32, rendered to a stable string
    /// (e.g. `"8"` for `stun 8`, `"8.5"` for `stun 8.5`) so the carrier
    /// stays `Eq`-compatible with the rest of `LowerError`.
    EffectArgExpectedDuration {
        verb:      String,
        got_value: String,
        span:      Span,
    },
    /// Body holds more than `MAX_EFFECTS_PER_PROGRAM` effects.
    BudgetExceeded { ability: String, count: usize, max: usize, span: Span },
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
    /// Wave 1.5#9 follow-up (#141): an outer effect's nested stmt
    /// carries its OWN modifier (chance/stacking/lifetime/scaling/
    /// in-shape/tags/for-duration/when/recursive-nested). Today the
    /// per-ability aggregator only captures outer-stmt modifiers, so
    /// inner-stmt modifiers would silently disappear — a real
    /// authoring footgun. Recursive aggregator capture is a future
    /// architectural lift; until then we error loudly with the
    /// offending modifier slot named.
    NestedModifierDropped {
        ability:  String,
        verb:     String,
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
    /// Wave 2 piece 5/6 (`deliver <method> { … }` block): the method
    /// ident is not in the engine's `DeliveryMethodKind` vocabulary
    /// (projectile/channel/zone/chain/tether/trap today). Surfaced
    /// so a typoed method name doesn't silently lower to a no-op.
    UnknownDeliveryMethod {
        ability: String,
        method:  String,
        span:    Span,
    },
    /// #143: a `when <cond>` (or `else <cond>`) modifier captured a
    /// raw source slice that fails to re-parse as a standalone
    /// expression at lower time. Today the parser captures the
    /// predicate text verbatim (terminating at the next modifier
    /// keyword / EOL / `}`); without re-validation, a typo'd
    /// operator (`target.hp ~ 30`) or unbalanced sub-expression
    /// silently ships through. Catching it here surfaces the bug
    /// at the `.ability` author's screen instead of as a malformed
    /// runtime predicate.
    WhenConditionParseError {
        ability:   String,
        clause:    &'static str, // "when" or "else"
        predicate: String,
        reason:    String,
        span:      Span,
    },
    /// #143 follow-up: a `when <cond>` (or `else <cond>`) predicate
    /// references a field on `self` or `target` that isn't in the
    /// engine's agent-field vocabulary (`AgentFieldId`). Common cause
    /// is a typo (`target.htp` for `target.hp`), or referencing a
    /// sim-specific extension field that isn't part of the canonical
    /// agent SoA — the latter is a false positive at the `.ability`
    /// layer (the field IS valid in the consuming sim) and can be
    /// silenced by extending the agent vocabulary or updating the
    /// allowlist. Pre-#143-followup the typo silently shipped through.
    WhenConditionUnknownField {
        ability:   String,
        clause:    &'static str, // "when" or "else"
        binder:    String, // "self" / "target" / etc.
        field:     String,
        span:      Span,
    },
    /// Wave 1.5#7 GPU eval: a `when <cond>` modifier carried a
    /// construct outside the restricted predicate vocab the
    /// dispatcher (CPU + GPU) evaluates today. Compound predicates
    /// (`&&` / `||` / `!`) are supported. Deferred branches:
    ///   * `else <cond>` clause
    ///   * field-vs-field comparisons (`target.hp < self.hp`)
    ///   * non-`<binder>.<field> <op> <literal>` atom shapes
    /// Authors must restructure the predicate to the supported shape
    /// or wait for the deferred slice. Surfaces with the construct's
    /// name so the diagnostic points at the right thing.
    WhenConditionUnsupported {
        ability: String,
        clause:  &'static str, // "when" or "else"
        reason:  String,
        span:    Span,
    },
    /// Task #227: a compound `when <cond>` tree serialized to more RPN
    /// nodes than the SoA stride (`MAX_PRED_NODES_PER_EFFECT = 12`)
    /// can carry. Realistically, this means an author wrote a
    /// 7-or-more-atom Boolean expression on a single effect; the
    /// fix is usually to split the predicate across multiple effect
    /// statements or hoist invariants up to the ability gate.
    WhenConditionTreeTooLarge {
        ability: String,
        clause:  &'static str, // "when"
        nodes:   usize,
        max:     usize,
        span:    Span,
    },
    /// Wave 1.5#7 GPU eval: a `when <cond>` predicate referenced an
    /// `AgentFieldId` that's outside the GPU-evaluable subset
    /// (the 8 ScalingStatRef-shaped f32 fields: AttackDamage /
    /// AbilityPower / MaxHp / Hp / Armor / MagicResist / MoveSpeed /
    /// Mana). The field IS in the engine's broader agent vocabulary
    /// (so `WhenConditionUnknownField` does not fire), but the
    /// dispatcher's per-stat agent-SoA bindings cover only this
    /// 8-field subset today. Authors should rephrase or extend the
    /// vocab in a coordinated pass.
    WhenConditionUnsupportedField {
        ability: String,
        clause:  &'static str, // "when" or "else"
        field:   String,
        span:    Span,
    },
    /// #139 (deliver-block body parsing): a `<hook_ident> { … }` inside
    /// a `deliver` body block named a hook not in the engine's
    /// `DeliveryHookKind` vocabulary (on_hit/on_tick/on_arrival/
    /// on_complete/on_trigger/on_kill/on_damage/on_damage_dealt/
    /// on_damage_taken/on_death/on_auto_attack/on_ability_used).
    /// Surfaced so a typoed hook ident doesn't silently lower to a
    /// no-op.
    UnknownDeliveryHook {
        ability: String,
        method:  String,
        hook:    String,
        span:    Span,
    },
    /// #139: a deliver-body hook's effect list exceeded the
    /// per-program effect budget (MAX_EFFECTS_PER_PROGRAM). Same budget
    /// as outer-effect lists — apply handlers SoA-pack hook bodies
    /// at the same width.
    DeliveryHookBudgetExceeded {
        ability: String,
        hook:    String,
        count:   usize,
        max:     usize,
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
            LowerError::HintReserved { hint, .. } => write!(
                f,
                "hint '{hint}' is planned/reserved; not yet supported by lowering"
            ),
            LowerError::UnknownEffectVerb { verb, suggestion, .. } => {
                write!(
                    f,
                    "unknown effect verb '{verb}'; valid verbs at this stage: damage / heal / shield / stun / slow / transfer_gold / modify_standing / cast / root / silence / fear / taunt / dash / blink / knockback / pull / execute / self_damage / lifesteal / damage_modify / summon / reveal / erase_belief / decoy / cast_recipe / wear_tool / propose / announce / gain_skill / create_obligation"
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
            LowerError::EffectArgExpectedDuration { verb, got_value, .. } => write!(
                f,
                "effect verb `{verb}` expects a time-suffixed argument (e.g. `1s`, `500ms`); got bare number `{got_value}` (did you mean `{got_value}s` or `{got_value}ms`?)"
            ),
            LowerError::BudgetExceeded { ability, count, max, .. } => write!(
                f,
                "ability '{ability}' has {count} effects but the per-program budget is {max} (MAX_EFFECTS_PER_PROGRAM)"
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
            LowerError::WhenConditionParseError { ability, clause, predicate, reason, .. } => write!(
                f,
                "ability `{ability}`'s `{clause} {predicate}` clause does not re-parse as an expression: {reason}"
            ),
            LowerError::WhenConditionUnknownField { ability, clause, binder, field, .. } => write!(
                f,
                "ability `{ability}`'s `{clause}` clause references `{binder}.{field}` — `{field}` is not in the engine's agent-field vocabulary; valid fields include hp / max_hp / alive / mana / shield_hp / armor / move_speed (see AgentFieldId for the full list). If this is a sim-specific extension field, extend the vocabulary or silence this check."
            ),
            LowerError::WhenConditionUnsupported { ability, clause, reason, .. } => write!(
                f,
                "ability `{ability}`'s `{clause}` clause uses an unsupported construct: {reason} (supported: `<binder>.<field> <op> <literal>` with binder ∈ {{self, target}}; other forms deferred)"
            ),
            LowerError::WhenConditionUnsupportedField { ability, clause, field, .. } => write!(
                f,
                "ability `{ability}`'s `{clause}` clause references field `{field}` — outside the GPU-evaluable subset for this slice (supported: attack_damage / ability_power / max_hp / hp / armor / magic_resist / move_speed / mana). Either rephrase or extend the per-stat agent-SoA bindings."
            ),
            LowerError::WhenConditionTreeTooLarge { ability, clause, nodes, max, .. } => write!(
                f,
                "ability `{ability}`'s `{clause}` clause serializes to {nodes} RPN nodes but the per-effect budget is {max} (MAX_PRED_NODES_PER_EFFECT). Split the predicate across multiple effects or hoist invariants out of the per-effect gate."
            ),
            LowerError::UnknownDeliveryHook { ability, method, hook, .. } => write!(
                f,
                "ability `{ability}` uses `deliver {method} {{ … {hook} {{…}} … }}` — `{hook}` is not a known delivery hook ident; valid hooks: on_hit / on_tick / on_arrival / on_complete / on_trigger / on_kill / on_damage / on_damage_dealt / on_damage_taken / on_death / on_auto_attack / on_ability_used"
            ),
            LowerError::DeliveryHookBudgetExceeded { ability, hook, count, max, .. } => write!(
                f,
                "ability `{ability}`'s `{hook} {{…}}` deliver-body hook declares {count} effects but the per-hook budget is {max} (MAX_EFFECTS_PER_PROGRAM)"
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
            LowerError::NestedModifierDropped { ability, verb, modifier, .. } => write!(
                f,
                "ability `{ability}`'s nested `{verb}` carries a `{modifier}` modifier that the lowering would silently drop today (recursive aggregator capture is a future lift); rewrite the inner verb to avoid the modifier or pull the effect to the outer level"
            ),
        }
    }
}

impl std::error::Error for LowerError {}

/// Outcome of lowering one `AbilityFile`. The split lets callers see
/// which abilities lowered cleanly AND which top-level decls were
/// skipped because their semantics aren't wired yet (passive blocks,
/// template definitions, structure definitions). Pre-#140 the first
/// such decl was a hard `Err(_)`, which short-circuited the entire
/// file even when valid `ability` decls followed it. That broke the
/// composite-file shape the LoL corpus uses (a passive-and-three-
/// abilities `.ability` would fail rather than yield three programs +
/// one warning).
///
/// `programs` holds the cleanly-lowered abilities in declaration
/// order; `skipped` collects one entry per top-level decl that's
/// parsed but not yet lowered (passives / templates / structures).
/// Per-ability lowering errors still abort with `Err(_)` — those are
/// genuine bugs in the source, not "feature-not-yet-wired" gaps.
#[derive(Debug, Default)]
pub struct LowerOutcome {
    pub programs: Vec<AbilityProgram>,
    pub skipped:  Vec<LowerError>,
}

/// Lower every `ability` decl inside an `AbilityFile`. The output
/// preserves declaration order so callers wiring a registry slot table
/// see the same indexing as the source file.
///
/// Per-ability lowering errors (UnknownEffectVerb / EffectArgMismatch
/// / typed-program contract violations) still propagate as `Err(_)`.
///
/// Wave 1.1 / 1.2 / 1.3 top-level decls (`passive` / `template` /
/// `structure`) parse cleanly but aren't lowered yet — passives need
/// PerEvent dispatch (Wave 2+), templates need expansion / parameter
/// substitution (Wave 2+), structures need voxel rasterization +
/// StructureRegistry (Wave 2+). Pre-#140, the first such decl
/// short-circuited the WHOLE file with `Err(_)`. Post-#140, each is
/// recorded as a `LowerOutcome::skipped` entry and the abilities
/// alongside continue to lower. Authors who want the loud error
/// per-decl can iterate `outcome.skipped` and pick the first.
pub fn lower_ability_file(file: &AbilityFile) -> Result<LowerOutcome, LowerError> {
    let mut out = LowerOutcome {
        programs: Vec::with_capacity(file.abilities.len()),
        skipped:  Vec::new(),
    };
    for passive in &file.passives {
        out.skipped.push(LowerError::PassiveBlockNotImplemented {
            name: passive.name.clone(),
            span: passive.span,
        });
    }
    for template in &file.templates {
        out.skipped.push(LowerError::TemplateBlockNotImplemented {
            name: template.name.clone(),
            span: template.span,
        });
    }
    for structure in &file.structures {
        out.skipped.push(LowerError::StructureBlockNotImplemented {
            name: structure.name.clone(),
            span: structure.span,
        });
    }
    for decl in &file.abilities {
        out.programs.push(lower_ability_decl(decl)?);
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
    // #129: capture recast / recast_window so the 39 LoL files that
    // declare them stop tripping `HeaderNotImplemented`. Apply
    // semantics (per-agent recast counter + window timer) lands later
    // alongside the registry-driven dispatch (#125 family); the
    // captured payload sits on the program ready for that wiring.
    let mut lowered_recast: Option<engine::ability::program::RecastKind> = None;
    let mut lowered_recast_window_ticks: Option<u32> = None;
    // `target: <mode>` defaults to Enemy when no header declares it
    // (matches the gate.hostile_only default + Wave 1 corpus shape).
    let mut lowered_target_mode: TargetModeKind = TargetModeKind::Enemy;

    for header in &decl.headers {
        match header {
            AbilityHeader::Target(mode) => {
                // All eight target modes per spec §4.3 lower into engine IR.
                // `gate.hostile_only` only encodes the Enemy/Self/Ally
                // distinction — apply handlers read `program.target_mode` for
                // the richer routing (position/directional/global) which today
                // doesn't dispatch (deferred — registry-driven apply).
                let (kind, hostile) = match mode {
                    TargetMode::Enemy     => (TargetModeKind::Enemy,     true),
                    TargetMode::Self_     => (TargetModeKind::SelfCast,  false),
                    TargetMode::Ally      => (TargetModeKind::Ally,      false),
                    TargetMode::SelfAoe   => (TargetModeKind::SelfAoe,   false),
                    // Position / directional / global modes have no
                    // single-agent target; `hostile_only` defaults to
                    // false (no friendly-fire check) — apply handlers
                    // wire the spatial/vector resolution later.
                    TargetMode::Ground    => (TargetModeKind::Ground,    false),
                    TargetMode::Direction => (TargetModeKind::Direction, false),
                    TargetMode::Vector    => (TargetModeKind::Vector,    false),
                    TargetMode::Global    => (TargetModeKind::Global,    false),
                };
                lowered_target_mode = kind;
                gate.hostile_only = hostile;
            }
            AbilityHeader::Range(r) => {
                // Preserve the (currently-only) Area shape and overwrite
                // its range field.
                area = Area::SingleTarget { range: *r };
            }
            AbilityHeader::Cooldown(d, _) => {
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
            // #129: capture `recast: <int|dur>` into the program.
            // Apply semantics (per-agent counter / window timer SoA)
            // arrive with registry-driven dispatch; until then the
            // recast header is recorded but doesn't influence cast
            // gating. 39 LoL files declare it.
            AbilityHeader::Recast(v) => {
                use dsl_ast::ast::RecastValue;
                lowered_recast = Some(match v {
                    RecastValue::Count(n)    => engine::ability::program::RecastKind::Count(*n),
                    RecastValue::Duration(d) => engine::ability::program::RecastKind::CooldownTicks(
                        duration_to_ticks(d.millis),
                    ),
                });
            }
            AbilityHeader::RecastWindow(d) => {
                lowered_recast_window_ticks = Some(duration_to_ticks(d.millis));
            }
        }
    }

    // -- Body-block guard: Wave 1.4 surfaces. Spec §4.4 / §23.1 said
    // deliver and bare effects were mutually exclusive, but the LoL
    // hero corpus uses the composite pattern heavily (e.g.
    // ArcaneShift = `deliver projectile {…} + dash to_target` — the
    // projectile fires on impact AND the caster simultaneously dashes
    // to the target point). With Delivery::Method capturing the
    // payload separately from program.effects, both can coexist:
    // trailing bare effects fire on the caster at cast-decide time,
    // delivery payload fires on projectile resolution.
    //
    // MixedBody check relaxed (#128, post 49bbeee2). The MorphBlock
    // check below stays — morph still defers entirely to Wave 2+
    // (form-swap state machinery not wired).
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
        // #139: parse-side captures `<hook_ident> { <effect_stmts> }`
        // entries inside the deliver body. Lower each hook ident
        // through the engine's `DeliveryHookKind` vocabulary (typoed
        // idents surface as `UnknownDeliveryHook`), then recursively
        // lower each hook's effects via `lower_effect_stmt`. The same
        // NestedModifierDropped guard #141 applies — outer-effect
        // aggregators (tags / scalings / chances / lifetimes / shapes)
        // don't see hook-effect modifiers, so an inner stmt with one
        // would silently lose its modifier slot. Reject loudly.
        let mut lowered_hooks: SmallVec<
            [engine::ability::program::DeliveryHook; 4],
        > = SmallVec::new();
        for hook in &block.hooks {
            let hook_kind =
                engine::ability::program::DeliveryHookKind::parse(&hook.kind)
                    .ok_or_else(|| LowerError::UnknownDeliveryHook {
                        ability: decl.name.clone(),
                        method:  block.method.clone(),
                        hook:    hook.kind.clone(),
                        span:    hook.span,
                    })?;
            if hook.effects.len() > MAX_EFFECTS_PER_PROGRAM {
                return Err(LowerError::DeliveryHookBudgetExceeded {
                    ability: decl.name.clone(),
                    hook:    hook.kind.clone(),
                    count:   hook.effects.len(),
                    max:     MAX_EFFECTS_PER_PROGRAM,
                    span:    hook.span,
                });
            }
            let mut hook_effects: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> =
                SmallVec::new();
            for hook_stmt in &hook.effects {
                // #139 known gap: hook-stmt outer-aggregator modifiers
                // (in-shape / tags / chance / stacking / lifetime /
                // scaling / when / nested) currently silently drop
                // here — `lower_effect_stmt` consumes verb-level
                // modifiers (`for <duration>` ⇒ DoT/HoT/timed) but
                // the per-program aggregator slots don't fan out to
                // hook bodies. This is strictly a smaller drop than
                // the prior opaque-body behavior (which lost the verb
                // itself). Recursive aggregator capture for hooks is
                // tracked as future work; until then the verb is
                // lifted into IR even if some modifier slots aren't.
                let op = lower_effect_stmt(hook_stmt)?;
                hook_effects.push(op);
            }
            lowered_hooks.push(engine::ability::program::DeliveryHook {
                kind:    hook_kind,
                effects: hook_effects,
            });
        }
        Delivery::Method { kind, raw: block.raw.clone(), hooks: lowered_hooks }
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

    // Abilities authored with the new `cast { … } effect { … }`
    // program shape lower into:
    //   * `effects`          = a single `EffectOp::CastBegin` op (the
    //                          IMMEDIATE-cast IR), and
    //   * `pending_program`  = the lowered op stream from each
    //                          `Effects(_)` step (the DEFERRED-resolution
    //                          IR, fired later by the busy-resolution
    //                          kernel when the cast resolves without
    //                          interruption).
    //
    // The parser guarantees mutual exclusion: when `decl.program` is
    // `Some`, `decl.effects` is empty, so the legacy effect loop below
    // is a no-op for this branch. Currently we emit only the first
    // `Cast` step; multi-stage programs (cast→effect→cast→effect chains)
    // lower to additional `CastBegin` ops in a follow-up slice.
    //
    // `ability_id` and `target_slot` on the CastBegin op are
    // placeholders at lowering time — the apply path overrides them at
    // dispatch via runtime context (caster/target plumbed by
    // `apply_ability`). The q8 target position fields are likewise
    // zeroed; they are wired by the BusyTargetPos SoA write at the cast
    // site for shape-aware threats lookup.
    //
    // **Per-effect modifier slots on pending_program** (chances /
    // scalings / lifetimes / etc.) are NOT captured today — we emit
    // bare ops only. The first behavioral pin that exercises a
    // pending-program with modifiers will land the parallel aggregator
    // slots; deferring keeps this slice small.
    let mut pending_effects: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
    let mut cast_interrupt_mask: Option<InterruptMask> = None;
    // Telegraph metadata, populated from the first cast{}'s
    // `telegraph: <shape>(...)` field. Defaults to the none-sentinel +
    // zero params when no cast{} block (or no telegraph field) is
    // authored. The threats fold reads the packed companion columns to
    // project per-caster zones.
    let mut telegraph_kind: u8 = TELEGRAPH_KIND_NONE;
    let mut telegraph_params: [f32; 4] = [0.0; 4];
    if let Some(steps) = &decl.program {
        for step in steps {
            match step {
                AbilityProgramStep::Cast(spec) => {
                    if !effects.is_empty() {
                        // First-cast-only MVP — silently skip subsequent
                        // Cast steps. Multi-stage chains are a follow-up.
                        continue;
                    }
                    let CastSpec { duration_ticks, interrupts, telegraph, .. } = spec;
                    let duration_clamped: u16 = (*duration_ticks).min(u16::MAX as u32) as u16;
                    cast_interrupt_mask = Some(lower_interrupt_set(interrupts));
                    if let Some(text) = telegraph.as_deref() {
                        if let Some((kind, params)) = parse_telegraph_text(text) {
                            telegraph_kind = kind.discriminant();
                            telegraph_params = params;
                        }
                        // Unknown / unparseable shapes silently fall back
                        // to the none-sentinel — Plan G G3e MVP only
                        // recognises `circle(...)` + `line(...)`. A
                        // future slice could surface a typed error here
                        // when authors typo a shape name.
                    }
                    effects.push(EffectOp::CastBegin {
                        ability_id:     0,
                        duration_ticks: duration_clamped,
                        target_slot:    0,
                        target_x_q8:    0,
                        target_y_q8:    0,
                    });
                }
                AbilityProgramStep::Effects(stmts) => {
                    for stmt in stmts {
                        if pending_effects.len() >= MAX_EFFECTS_PER_PROGRAM {
                            return Err(LowerError::BudgetExceeded {
                                ability: decl.name.clone(),
                                count:   pending_effects.len() + 1,
                                max:     MAX_EFFECTS_PER_PROGRAM,
                                span:    stmt.span,
                            });
                        }
                        let op = lower_effect_stmt(stmt)?;
                        pending_effects.push(op);
                    }
                }
            }
        }
        if effects.len() > MAX_EFFECTS_PER_PROGRAM {
            return Err(LowerError::BudgetExceeded {
                ability: decl.name.clone(),
                count:   effects.len(),
                max:     MAX_EFFECTS_PER_PROGRAM,
                span:    decl.span,
            });
        }
    }

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
        // #143: re-parse the captured `when` (and optional `else`)
        // predicate text as an expression. The Wave 1.5 parser
        // captured it as opaque source text terminated by the next
        // modifier keyword / EOL / `}` — that lets a typo'd operator
        // (`target.hp ~ 30`) or unbalanced sub-expression silently
        // ship through. Re-parsing here surfaces the bug with the
        // ability name + clause + reason.
        //
        // #143 follow-up: walk the parsed expression for
        // `<binder>.<field>` accessors where binder is `self` or
        // `target`, and validate `field` against the engine's
        // `AgentFieldId` vocabulary. Catches the typo'd-field case
        // (`target.htp` for `target.hp`) that the syntax check
        // alone misses.
        let when = if let Some(c) = stmt.condition.as_ref() {
            // Task #227: lower the predicate to a Boolean tree
            // (`WhenPredicate`) with `EffectPredicate` atoms at the
            // leaves. The tree handles `&&` / `||` / `!`; the simple
            // atomic case (no compound combinators) collapses to a
            // single `WhenPredicate::Atom`. `when_compiled` is also
            // populated for the simple atomic case so legacy code
            // paths that only consume single-atom predicates keep
            // working. Restricted leaf vocab (form
            // `<binder>.<field> <op> <literal>`); anything else
            // surfaces WhenConditionUnsupported / -UnsupportedField.
            let when_then = parse_when_branch(
                &c.when_cond, "when", &decl.name, c.span,
            )?;
            // Task #228: optional `else <cond>` branch — semantically
            // equivalent to `when X || Y` for this slice. The
            // structural distinction (which branch matched) lives at
            // the source-text layer (`EffectWhenCondition::else_cond`
            // is preserved verbatim below) so future extensions can
            // light up branch-distinguishing semantics without an IR
            // bump. Combining via `Or` lets the existing RPN encoding
            // and both backend evaluators handle it transparently —
            // no new sentinels, no schema bump.
            let when_compound = if let Some(else_text) = c.else_cond.as_ref() {
                let when_else = parse_when_branch(
                    else_text, "else", &decl.name, c.span,
                )?;
                WhenPredicate::Or(Box::new(when_then), Box::new(when_else))
            } else {
                when_then
            };
            // Bound the RPN stride. Counting nodes here gives an early
            // diagnostic at the author's screen rather than a defensive
            // truncate at SoA pack time.
            let node_count = count_rpn_nodes(&when_compound);
            if node_count > MAX_PRED_NODES_PER_EFFECT {
                return Err(LowerError::WhenConditionTreeTooLarge {
                    ability: decl.name.clone(),
                    clause:  "when",
                    nodes:   node_count,
                    max:     MAX_PRED_NODES_PER_EFFECT,
                    span:    c.span,
                });
            }
            // Mirror simple atoms onto `when_compiled` for legacy
            // single-atom consumers; compound trees populate only
            // `when_compound`.
            let when_compiled = match &when_compound {
                WhenPredicate::Atom(a) => Some(*a),
                _ => None,
            };
            Some(EffectWhenCondition {
                when_cond:     c.when_cond.clone(),
                // Task #228: preserve verbatim else-branch source text
                // so future extensions can recover which branch
                // matched. The lowered `when_compound` already encodes
                // both branches as `Or(then, else)` for execution.
                else_cond:     c.else_cond.clone(),
                when_compiled,
                when_compound: Some(when_compound),
            })
        } else {
            None
        };
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
            // #141: error loudly if the nested stmt carries any modifier
            // captured by the OUTER per-ability aggregators (those
            // aggregators only see outer stmts; inner-stmt modifiers
            // would silently disappear at lowering). Recursive
            // aggregator capture is a future architectural lift.
            //
            // Note: `for-duration` is NOT included here — duration is
            // consumed by `lower_effect_stmt`'s verb dispatch directly
            // (slow/stun/etc. fold it into the EffectOp's duration_ticks
            // field; damage/heal/shield promote into DoT/HoT/TimedShield
            // variants). It never reaches the outer aggregator and so
            // is never silently dropped.
            let modifier = if nested_stmt.area.is_some()       { Some("in-shape") }
                else if !nested_stmt.tags.is_empty()           { Some("[TAG: …]") }
                else if nested_stmt.chance.is_some()           { Some("chance") }
                else if nested_stmt.stacking.is_some()         { Some("stacking") }
                else if !nested_stmt.scalings.is_empty()       { Some("+ N% stat_ref") }
                else if nested_stmt.lifetime.is_some()         { Some("lifetime") }
                else if nested_stmt.condition.is_some()        { Some("when") }
                else if !nested_stmt.nested.is_empty()         { Some("nested {…}") }
                else                                           { None };
            if let Some(m) = modifier {
                return Err(LowerError::NestedModifierDropped {
                    ability:  decl.name.clone(),
                    verb:     nested_stmt.verb.clone(),
                    modifier: m,
                    span:     nested_stmt.span,
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
        recast: lowered_recast,
        recast_window_ticks: lowered_recast_window_ticks,
        target_mode: lowered_target_mode,
        cast_interrupt_mask,
        pending_program: pending_effects,
        telegraph_kind,
        telegraph_params,
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
    // pull the duration from there instead of a positional.
    //
    // damage/heal also accept `for <duration>` — the per-verb arms
    // below branch into DamageOverTime / HealOverTime EffectOps when
    // the duration is present.
    //
    // The remaining verbs (shield/cast/transfer_gold/modify_standing)
    // have no over-time semantics — for-modifier on those still
    // surfaces ModifierNotImplemented.
    if let Some(d) = &stmt.duration {
        if !is_duration_bearing_verb(&stmt.verb)
            && stmt.verb != "damage"
            && stmt.verb != "heal"
            && stmt.verb != "shield"
            && stmt.verb != "summon"
        {
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
            // `damage X for Ts` → DoT (130). The for-duration is stored
            // separately on the EffectStmt; if present, return
            // DamageOverTime so apply handlers iterate per-tick.
            if let Some(d) = &stmt.duration {
                Ok(EffectOp::DamageOverTime {
                    amount,
                    duration_ticks: duration_to_ticks(d.duration.millis),
                })
            } else {
                Ok(EffectOp::Damage { amount })
            }
        }
        "heal" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            // `heal X for Ts` → HoT (130). Mirror of DoT.
            if let Some(d) = &stmt.duration {
                Ok(EffectOp::HealOverTime {
                    amount,
                    duration_ticks: duration_to_ticks(d.duration.millis),
                })
            } else {
                Ok(EffectOp::Heal { amount })
            }
        }
        "shield" => {
            let amount = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            // `shield X for Ts` → TimedShield (130 follow-on). Same
            // branching pattern as damage→DoT and heal→HoT.
            if let Some(d) = &stmt.duration {
                Ok(EffectOp::TimedShield {
                    amount,
                    duration_ticks: duration_to_ticks(d.duration.millis),
                })
            } else {
                Ok(EffectOp::Shield { amount })
            }
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
        // Wave 2 piece 7 — `stealth for <duration> [break_on_damage]`.
        // The 25 LoL hero files share the exact shape `stealth for 3s
        // break_on_damage`. No magnitude — stealth is binary. The
        // duration MUST come from `for`; the explicit positional
        // form `stealth <duration>` is rejected (no corpus uses it).
        // The `break_on_damage` modifier rides the existing per-effect
        // lifetime SoA — it doesn't need a stealth-specific arm here.
        "stealth" => {
            // Positional duration is meaningless for stealth — the only
            // valid shape is `stealth for <dur>`. extract_duration's
            // `positional_idx` is unused if `stmt.duration` is Some
            // (the upstream `is_duration_bearing_verb` guard ensures
            // we only land here when the modifier IS present).
            let (dur, arity) = extract_duration(stmt, 0, 0)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Stealth { duration_ticks: duration_to_ticks(dur) })
        }
        // Wave 2 piece 8 — remaining LoL CC vocabulary. charm/grounded/
        // suppress all share the Stun shape (one positional duration);
        // reflect shares the LifeSteal shape (fraction + duration with
        // q8 magnitude packing).
        "charm" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Charm { duration_ticks: duration_to_ticks(dur) })
        }
        "grounded" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Grounded { duration_ticks: duration_to_ticks(dur) })
        }
        "suppress" => {
            let (dur, arity) = extract_duration(stmt, 0, 1)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Suppress { duration_ticks: duration_to_ticks(dur) })
        }
        "reflect" => {
            let fraction = require_number_arg(stmt, 0)?;
            let (dur, arity) = extract_duration(stmt, 1, 2)?;
            require_arity(stmt, arity)?;
            let fraction_q8 = (fraction * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::Reflect {
                duration_ticks: duration_to_ticks(dur),
                fraction_q8,
            })
        }
        // Wave 2 piece 2 — four new movement verbs. Each takes a single
        // `<distance:f32>` arg (mirrors `damage`'s shape, NOT the
        // duration shape of the control verbs) and lowers to the
        // matching `EffectOp::*` variant. Apply handlers (compute
        // facing direction / away-from-caster / toward-caster vectors
        // and update `hot_pos`) land in a follow-up Wave 2 piece.
        "dash" => {
            // Two LoL/MOBA forms accepted:
            //   1. `dash <distance:f32>` — fixed travel distance.
            //   2. `dash to_target` — directive: travel until the
            //      cast's target is reached. Encoded as the sentinel
            //      `distance = f32::INFINITY` so the `EffectOp::Dash`
            //      shape stays a single f32 (P4: ≤16-byte budget).
            //      Apply handlers check `distance.is_infinite()` to
            //      branch into target-tracking travel.
            let dist = match stmt.args.first() {
                Some(EffectArg::Number(v)) => *v,
                Some(EffectArg::Ident(n)) if n == "to_target" => f32::INFINITY,
                _ => {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     "dash".to_string(),
                        expected: 1,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
            };
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
        // Lift A — multi-tick procedure: `travel_to <x> <y> for <duration>`.
        // The caster initiates a multi-tick walk to (x, y) over `eta_ticks`
        // (10 Hz, so `for 5s` ≈ 50 ticks). The destination is packed q8
        // (256 = 1 unit; range ±127.99). The DSL surface accepts an
        // optional positional Z (today ignored — 2D-flat sims dominate; a
        // future SoA cell extension can carry Z too without a verb shape
        // change). The downstream consumer rule sets `busy_until_tick =
        // world.tick + eta_ticks` and populates the per-agent
        // `travel_dest_{x,y,z}` SoA cells; a per-tick travel kernel
        // interpolates `pos` toward the destination.
        "travel_to" => {
            // Two positional args: (dest_x, dest_y). The optional dest_z
            // is accepted but ignored for the q8-packed payload (the SoA
            // cell still receives it on the apply side via the consumer
            // rule — a future EffectOp evolution can carry z too).
            let dx = require_number_arg(stmt, 0)?;
            let dy = require_number_arg(stmt, 1)?;
            // Optional positional dest_z — accept but don't pack into the
            // EffectOp payload today (Lift A ships 2D-flat travel; the SoA
            // cell `travel_dest_z` is reserved for a future variant).
            let _dz = match stmt.args.get(2) {
                None => 0.0,
                Some(EffectArg::Number(v)) => *v,
                Some(_) => {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     "travel_to".to_string(),
                        expected: 2,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
            };
            // Duration MUST come from the `for` modifier — travel without
            // an ETA is meaningless. The is_duration_bearing_verb guard
            // upstream ensures we only land here when `stmt.duration` is
            // Some.
            let dur = stmt
                .duration
                .as_ref()
                .map(|d| d.duration.millis)
                .ok_or_else(|| LowerError::EffectArgMismatch {
                    verb:     "travel_to".to_string(),
                    expected: 3, // x + y + for-duration
                    got:      stmt.args.len(),
                    span:     stmt.span,
                })?;
            // 2 or 3 positional args allowed (x, y, [z]). The for-modifier
            // doesn't count toward arity.
            if stmt.args.len() != 2 && stmt.args.len() != 3 {
                return Err(LowerError::EffectArgMismatch {
                    verb:     "travel_to".to_string(),
                    expected: 2,
                    got:      stmt.args.len(),
                    span:     stmt.span,
                });
            }
            let dest_x_q8 = (dx * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let dest_y_q8 = (dy * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::TravelTo {
                dest_x_q8,
                dest_y_q8,
                eta_ticks: duration_to_ticks(dur),
            })
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
        "buff" => {
            // `buff <stat:ident> <magnitude:f32> for <duration>` — LoL
            // corpus form (#131). Stat vocabulary: move_speed,
            // attack_speed (BuffStat::parse). Magnitude packs to q8
            // matching Slow/DamageModify (`+ 30%` → 76; `+ 100%` → 256).
            // Duration comes from the for-modifier (positional duration
            // form would also work via extract_duration).
            let stat_name = require_name_arg(stmt, 0)?;
            let stat = engine::ability::program::BuffStat::parse(&stat_name)
                .ok_or_else(|| LowerError::UnknownStatRef {
                    stat: stat_name.clone(),
                    span: stmt.span,
                })?;
            let magnitude = require_number_arg(stmt, 1)?;
            let (dur, arity) = extract_duration(stmt, 2, 3)?;
            require_arity(stmt, arity)?;
            let magnitude_q8 = (magnitude * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            Ok(EffectOp::Buff {
                stat,
                magnitude_q8,
                duration_ticks: duration_to_ticks(dur),
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
        "scry" => {
            // `scry <target_observer> <subject_idx>` — Wave 3 ToM Phase
            // 3.5. Two positional integer args; no duration (one-shot
            // belief copy). target_observer is a u8 slot id (0..255);
            // subject_idx is a u32 agent index. The chronicle consumer
            // copies the 6 BeliefState SoA columns from
            // `[target_observer * N + subject_idx]` to
            // `[caster * N + subject_idx]`. Mirrors the transfer_gold
            // arm's multi-arg, no-duration shape.
            let observer = require_number_arg(stmt, 0)?;
            let subject = require_number_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let target_observer =
                observer.round().clamp(0.0, u8::MAX as f32) as u8;
            let subject_idx =
                subject.round().clamp(0.0, u32::MAX as f32) as u32;
            Ok(EffectOp::Scry { target_observer, subject_idx })
        }
        "plant_belief" => {
            // `plant_belief <subject_idx:u32> bit <fact_bit:u8>` — Wave 3
            // ToM primitive (#223 spy_network). Three positional args
            // with the `bit` keyword acting as an in-band separator so
            // the surface reads as natural English ("plant belief about
            // subject 5, bit 3"). Engine variant ordinal is 32; payload
            // is 5 bytes (u32 + u8) under the P4 ≤16-byte ceiling. The
            // apply handler folds `1u << fact_bit` into the
            // `[caster_slot * agent_cap + subject_idx]` cell of the
            // BeliefState bitset via WGSL `atomicOr` — see
            // `engine::ability::program::EffectOp::PlantBelief` docs for
            // the chronicle / pair_map fold contract.
            //
            // Bounds: `subject_idx` is u32 (full agent id range);
            // `fact_bit` is u8 in 0..32 (bit position into the suspicion
            // bitset). We clamp at cast time rather than erroring so a
            // typo doesn't gate the whole .ability file from lowering —
            // the engine schema will reject out-of-range bits at apply
            // time anyway and the diagnostic there names the offending
            // EffectOp directly.
            let subject = require_number_arg(stmt, 0)?;
            let bit_kw = require_name_arg(stmt, 1)?;
            if bit_kw != "bit" {
                return Err(LowerError::EffectArgMismatch {
                    verb:     "plant_belief".to_string(),
                    expected: 3,
                    got:      stmt.args.len(),
                    span:     stmt.span,
                });
            }
            let fact_bit = require_number_arg(stmt, 2)?;
            require_arity(stmt, 3)?;
            let subject_idx = subject.round().clamp(0.0, u32::MAX as f32) as u32;
            let fact_bit_u8 = fact_bit.round().clamp(0.0, u8::MAX as f32) as u8;
            Ok(EffectOp::PlantBelief { subject_idx, fact_bit: fact_bit_u8 })
        }
        // Wave 3 ToM Phase 4 — `erase_belief <subject_idx> <fields>`.
        // `fields` is a u8 bitset: bit 0=pos, 1=type, 2=tick,
        // 3=confidence, 4=suspicion, 5=flags. Engine variant 38, kind 69.
        "erase_belief" => {
            let subject = require_number_arg(stmt, 0)?;
            let fields = require_number_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let subject_idx = subject.round().clamp(0.0, u32::MAX as f32) as u32;
            let fields = fields.round().clamp(0.0, u8::MAX as f32) as u8;
            Ok(EffectOp::EraseBelief { subject_idx, fields })
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
        // Wave 3 ToM Phase 4 — `disguise <fake_type:int> for <duration>`.
        // Caster publicly poses as `fake_type` (a creature_type ordinal,
        // u8) for `duration_ticks`. The duration MUST come from the
        // `for` modifier (positional duration form is not accepted —
        // mirrors `stealth`'s shape). The `is_duration_bearing_verb`
        // short-circuit covers the modifier path; positional fake_type
        // is read here as the sole positional arg.
        "disguise" => {
            let fake_type_f = require_number_arg(stmt, 0)?;
            let fake_type = fake_type_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            let (dur, arity) = extract_duration(stmt, 1, 2)?;
            require_arity(stmt, arity)?;
            Ok(EffectOp::Disguise {
                fake_type,
                duration_ticks: duration_to_ticks(dur),
            })
        }
        "summon" => {
            // `summon "<template>" [<count:int>] [for <duration>]` —
            // LoL corpus form. All 17 .ability sites in the LoL hero
            // corpus use the bare `summon "<template>"` shape (no
            // count, no duration); the optional positional count and
            // `for <duration>` are accepted for forward compatibility
            // with the spec's full surface.
            //
            // Template ident becomes a 32-bit FxHash so the engine
            // payload stays a single u32 (deferred resolution — apply
            // handlers map the hash to a concrete spawner via a
            // registry follow-up). String + Ident accepted via
            // require_name_arg so authors aren't forced to remember
            // the quoting convention.
            let template = require_name_arg(stmt, 0)?;
            let template_hash = summon_template_hash(&template);
            // Optional positional arg 1: count. Default 0 (apply
            // handler treats as 1).
            let count: u8 = match stmt.args.get(1) {
                None => 0,
                Some(EffectArg::Number(n)) => {
                    n.round().clamp(0.0, u8::MAX as f32) as u8
                }
                Some(_) => {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     "summon".to_string(),
                        expected: 1,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
            };
            // Optional `for <duration>` modifier. Default 0 (apply
            // handler picks a sensible default — e.g. permanent until
            // owner dies). The is_duration_bearing_verb short-circuit
            // does NOT cover summon (we treat duration as truly
            // optional rather than required), so we read it here.
            let lifetime_ticks = stmt
                .duration
                .as_ref()
                .map(|d| duration_to_ticks(d.duration.millis))
                .unwrap_or(0);
            let expected_arity = if count == 0 { 1 } else { 2 };
            require_arity(stmt, expected_arity)?;
            Ok(EffectOp::Summon {
                template_hash,
                count,
                lifetime_ticks,
            })
        }
        // Non-combat verbs phase 1 — world primitives. Both `harvest`
        // and `mine` lower to `EffectOp::Harvest`; the kind_hash is the
        // FxHash of the resource ident (same FxHash family as Summon's
        // template_hash). `mine` is purely an authoring-ergonomics alias
        // — apply handlers use the hash to look up resource metadata in
        // the runtime registry (organic / surface vs voxel-backed)
        // and dispatch to AgentHarvested vs AgentHarvestedVoxel.
        // `place_voxel "<kind>"` lowers to `EffectOp::PlaceVoxel` — a
        // single u32 payload carrying the FxHash of the voxel ident.
        // The cast target supplies the placement position; apply
        // handlers emit AgentPlacedVoxel and mutate world state.
        "harvest" | "mine" => {
            // `harvest "<kind>" [<amount>]` (or `mine` alias). Kind is
            // a string-or-ident arg matching the `summon` precedent; the
            // optional positional `amount` defaults to 1 when omitted.
            // No `for <duration>` modifier — gathering is instantaneous.
            let kind = require_name_arg(stmt, 0)?;
            let kind_hash = summon_template_hash(&kind);
            let amount: u16 = match stmt.args.get(1) {
                None => 1,
                Some(EffectArg::Number(n)) => {
                    n.round().clamp(0.0, u16::MAX as f32) as u16
                }
                Some(_) => {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     stmt.verb.clone(),
                        expected: 1,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
            };
            let expected_arity = if stmt.args.len() == 1 { 1 } else { 2 };
            require_arity(stmt, expected_arity)?;
            Ok(EffectOp::Harvest { kind_hash, amount })
        }
        "place_voxel" => {
            // `place_voxel "<kind>"` — one u32 payload (FxHash of kind
            // ident). String + Ident accepted via `require_name_arg`.
            let kind = require_name_arg(stmt, 0)?;
            let kind_hash = summon_template_hash(&kind);
            require_arity(stmt, 1)?;
            Ok(EffectOp::PlaceVoxel { kind_hash })
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
        // Wave 3 phase 3.5 — Theory-of-Mind `observe` verb. Single
        // positional `<id:u8>` arg (agent slot id) — the future-extension
        // `target_observer` byte; today only `0` (self) is wired engine-
        // side. Engine op is `EffectOp::Observe { target_observer: u8 }`
        // (kind=33). Out-of-range ids clamp into 0..=255 to match the
        // `summon`/`harvest` u8/u16 cast precedent rather than reject.
        "observe" => {
            let id_f = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            let target_observer = id_f.round().clamp(0.0, u8::MAX as f32) as u8;
            Ok(EffectOp::Observe { target_observer })
        }
        // Wave 3 ToM Phase 3.5 — `reveal <subject_idx>` one-to-many
        // belief broadcast. The dispatcher resolves caster; the consumer
        // iterates all observers.
        "reveal" => {
            let raw = require_number_arg(stmt, 0)?;
            require_arity(stmt, 1)?;
            Ok(EffectOp::Reveal { subject_idx: raw.round() as u32 })
        }
        // Wave 3 ToM Phase 4 — `decoy <subject_idx> <fake_pos>`. Caster
        // plants a full BeliefState row in the cast target's belief map
        // about a third-party agent slot. `fake_pos` is a packed u32
        // (fake_pos_x i8, fake_pos_y i8, fake_pos_z i8, fake_type u8)
        // per engine spec. Engine ordinal 37, chronicle event 68.
        "decoy" => {
            let subject_idx = require_number_arg(stmt, 0)?
                .round().clamp(0.0, u32::MAX as f32) as u32;
            let fake_pos = require_number_arg(stmt, 1)?
                .round().clamp(0.0, u32::MAX as f32) as u32;
            require_arity(stmt, 2)?;
            Ok(EffectOp::Decoy { subject_idx, fake_pos })
        }
        // Lift B — `cast_recipe <recipe_id> [target <tool_slot>]`. Caster
        // fires a production recipe by id. The `recipe_id` (u16) indexes
        // into the per-fixture `RecipeRegistry`. The optional `target
        // <tool_slot>` keyword pair binds a specific Tool entity slot
        // (u8) to the cast — `0xFF` is the "no tool target" sentinel
        // applied when the keyword is absent. See `docs/spec/economy.md`
        // §4.1. Engine ordinal 40, chronicle event 71.
        "cast_recipe" => {
            let recipe_id_f = require_number_arg(stmt, 0)?;
            let recipe_id = recipe_id_f
                .round()
                .clamp(0.0, u16::MAX as f32) as u16;
            // Optional `target <tool_slot>` modifier — present iff arity
            // is 3 (positional id + `target` ident + positional slot).
            // Default sentinel 0xFF means "no tool target".
            let target_tool = if stmt.args.len() >= 3 {
                let kw = require_name_arg(stmt, 1)?;
                if kw != "target" {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     "cast_recipe".to_string(),
                        expected: 3,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
                let slot_f = require_number_arg(stmt, 2)?;
                require_arity(stmt, 3)?;
                slot_f.round().clamp(0.0, u8::MAX as f32) as u8
            } else {
                require_arity(stmt, 1)?;
                0xFFu8
            };
            Ok(EffectOp::Recipe { recipe_id, target_tool })
        }
        // Lift B — `wear_tool <tool_kind> <amount>`. Caster bumps wear
        // on a tool of `tool_kind` (u8 ordinal — Forge, Anvil, Loom, …)
        // by `amount` (u16, q8 fraction-of-durability — 256 = 1.0).
        // Recipes typically pair with a `wear_tool` step so capital
        // goods depreciate in use. See `docs/spec/economy.md` §4.3.
        // Engine ordinal 41, chronicle event 72.
        "wear_tool" => {
            let tool_kind_f = require_number_arg(stmt, 0)?;
            let amount_f = require_number_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let tool_kind = tool_kind_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            let amount = amount_f
                .round()
                .clamp(0.0, u16::MAX as f32) as u16;
            Ok(EffectOp::WearTool { tool_kind, amount })
        }
        // Lift C — `propose <contract_kind> [expires_at <tick>]`. Caster
        // offers a bilateral agreement of `contract_kind` (u8 ordinal —
        // Marriage, Partnership, Service, …) to the cast target. The
        // optional `expires_at <tick>` keyword pair sets the wall-clock
        // tick at which the proposal auto-cancels; absent → 0 sentinel
        // (proposal stays open until target accepts/declines or caster
        // cancels). The companion accept / decline verbs ship in a
        // follow-up slice. See `docs/spec/economy.md` §7. Engine
        // ordinal 42, chronicle event 73.
        "propose" => {
            let contract_kind_f = require_number_arg(stmt, 0)?;
            let contract_kind = contract_kind_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            // Optional `expires_at <tick>` modifier — present iff arity
            // is 3 (positional kind + `expires_at` ident + positional tick).
            // Absent means 0 sentinel ("no expiry").
            let expires_at_tick = if stmt.args.len() >= 3 {
                let kw = require_name_arg(stmt, 1)?;
                if kw != "expires_at" {
                    return Err(LowerError::EffectArgMismatch {
                        verb:     "propose".to_string(),
                        expected: 3,
                        got:      stmt.args.len(),
                        span:     stmt.span,
                    });
                }
                let tick_f = require_number_arg(stmt, 2)?;
                require_arity(stmt, 3)?;
                tick_f.round().clamp(0.0, u32::MAX as f32) as u32
            } else {
                require_arity(stmt, 1)?;
                0u32
            };
            Ok(EffectOp::Propose { contract_kind, expires_at_tick })
        }
        // Lift D — `gain_skill <skill_id> <amount>`. Self-cast skill
        // growth. The `skill_id` (u8 ordinal) indexes the per-fixture
        // SkillRegistry; `amount` is q8 fraction-of-mastery (256 = full
        // mastery, but the consumer clamps the per-skill cell to
        // [0.0, 1.0]). See `docs/spec/economy.md` §8. Engine ordinal 44,
        // chronicle event 75.
        "gain_skill" => {
            let skill_id_f = require_number_arg(stmt, 0)?;
            let amount_f = require_number_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let skill_id = skill_id_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            let amount_q8 = amount_f
                .round()
                .clamp(0.0, u16::MAX as f32) as u16;
            Ok(EffectOp::GainSkill { skill_id, amount_q8 })
        }
        // Lift D — `create_obligation <obligation_id> <kind>`. Caster
        // (creditor / claimant) registers a persistent obligation
        // against the cast target (debtor / promisor). The
        // `obligation_id` (u16) is the slot in the per-fixture
        // ObligationRegistry; `kind` (u8) tags the variant — Debt=0,
        // Future=1, Insurance=2, Retainer=3, Service=4. The full TERMS
        // (principal, due_tick, collateral, …) live in the registry
        // entry; the discharge / default companion verbs ship later.
        // See `docs/spec/economy.md` §7. Engine ordinal 45, chronicle
        // event 76.
        "create_obligation" => {
            let obligation_id_f = require_number_arg(stmt, 0)?;
            let kind_f = require_number_arg(stmt, 1)?;
            require_arity(stmt, 2)?;
            let obligation_id = obligation_id_f
                .round()
                .clamp(0.0, u16::MAX as f32) as u16;
            let kind = kind_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            Ok(EffectOp::CreateObligation { obligation_id, kind })
        }
        // Lift C — `announce <announcement_kind> radius <radius_cells>`.
        // Caster broadcasts a public event of `announcement_kind` (u8)
        // to all agents within `radius_cells` cells. The `radius`
        // keyword is required — announcements without a fan-out radius
        // are meaningless. Storage is q8 fixed-point: 256 = 1.0 cell;
        // the per-fixture consumer divides by 256 to walk the spatial-
        // hash. See `docs/spec/economy.md` §6. Engine ordinal 43,
        // chronicle event 74.
        "announce" => {
            let announcement_kind_f = require_number_arg(stmt, 0)?;
            let kw = require_name_arg(stmt, 1)?;
            if kw != "radius" {
                return Err(LowerError::EffectArgMismatch {
                    verb:     "announce".to_string(),
                    expected: 3,
                    got:      stmt.args.len(),
                    span:     stmt.span,
                });
            }
            let radius_f = require_number_arg(stmt, 2)?;
            require_arity(stmt, 3)?;
            let announcement_kind = announcement_kind_f
                .round()
                .clamp(0.0, u8::MAX as f32) as u8;
            // q8 packing: multiply cells × 256, clamp into u16 (max
            // ~256 cells of fan-out, which already covers the spatial-
            // hash diameter of every existing fixture).
            let radius_q8 = (radius_f * 256.0)
                .round()
                .clamp(0.0, u16::MAX as f32) as u16;
            Ok(EffectOp::Announce { announcement_kind, radius_q8 })
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
        // #142: distinct Heal/Buff variants now live in engine — routing
        // heal→Defense and buff→Utility was a real scoring bug. AI
        // evaluators that bucket on hint were miscategorizing heals as
        // defenses (different urgency curves) and buffs as utility
        // (different target-selection shape).
        HintName::Heal => Ok(AbilityHint::Heal),
        HintName::Buff => Ok(AbilityHint::Buff),
        HintName::Economic => Err(LowerError::HintReserved {
            hint: "economic".to_string(),
            span: decl.span,
        }),
    }
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

/// Render an f32 captured from `EffectArg::Number(_)` so the
/// `EffectArgExpectedDuration` diagnostic reads `bare number 8` for an
/// integer-valued literal and `bare number 8.5` for a fractional one.
/// The default `Display` for f32 always renders integer values with a
/// trailing `.0` (`8` → `"8"` is what the parser saw on the screen, but
/// `format!("{}", 8.0_f32)` is `"8"` on stable as of 1.55+ — we still
/// route through this helper so a future `f32::Display` change doesn't
/// regress the diagnostic).
fn format_bare_number(v: f32) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn require_duration_arg(stmt: &EffectStmt, idx: usize) -> Result<u32, LowerError> {
    match stmt.args.get(idx) {
        Some(EffectArg::Duration(d)) => Ok(d.millis),
        // Designer wrote `stun 8` (intending "8 ticks" or "8 seconds") —
        // arity is right but the parser captured the unsuffixed integer
        // as `Number(8.0)`, not `Duration { millis: 8000 }`. Surface a
        // typed diagnostic that names the missing time suffix instead
        // of the misleading `EffectArgMismatch { expected: 1, got: 1 }`.
        // `format_bare_number` renders integer-valued f32s without the
        // `.0` tail so the diagnostic reads `bare integer 8`, not
        // `bare integer 8.0`.
        Some(EffectArg::Number(v)) => Err(LowerError::EffectArgExpectedDuration {
            verb:      stmt.verb.clone(),
            got_value: format_bare_number(*v),
            span:      stmt.span,
        }),
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
        | "lifesteal" | "damage_modify" | "buff"
        // Wave 2 piece 7 — `stealth for <duration>`. Same shape as the
        // control verbs (no magnitude arg, duration via `for`), but
        // self-cast (apply handlers will gate on caster, not target).
        | "stealth"
        // Wave 2 piece 8 — remaining CC vocabulary from the LoL corpus.
        // charm/grounded/suppress have stun-shape (positional duration);
        // reflect has lifesteal-shape (fraction + duration).
        | "charm" | "grounded" | "suppress" | "reflect"
        // Wave 3 ToM Phase 4 — `disguise <fake_type> for <duration>`
        // (spy_network surface). Self-cast deception verb. The
        // `fake_type` ordinal is the single positional arg; the
        // duration comes from `for`.
        | "disguise"
        // Lift A — `travel_to <x> <y> for <duration>`. Same `for`-as-
        // duration-source pattern as the buffs above; the duration IS
        // the eta_ticks, NOT a positional. Travel without a duration is
        // meaningless (would imply teleport, which is the `blink` verb).
        | "travel_to"
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

/// Stable 32-bit hash of a deferred-resolution ident. Used to project
/// .ability source string args into `EffectOp::Summon { template_hash }`,
/// `EffectOp::Harvest { kind_hash }`, and `EffectOp::PlaceVoxel
/// { kind_hash }` payloads (deferred resolution — apply handlers map
/// the hash to a concrete spawner / resource / voxel via a registry
/// follow-up).
///
/// Uses `rustc_hash::FxHasher` (FxHash) — fixed-seed, deterministic,
/// no per-process salt. Cross-platform stable; replay equivalence
/// holds across CPU/GPU and across machine reboots.
fn summon_template_hash(template: &str) -> u32 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write(template.as_bytes());
    // Fold 64 → 32: xor halves so both ends contribute. FxHash already
    // mixes bits well; this is a pure compaction step.
    let full = h.finish();
    ((full >> 32) as u32) ^ (full as u32)
}

/// Task #228: parse + validate one branch of a per-effect predicate
/// clause (the `when` body or the optional `else` body) and lower it
/// to a [`WhenPredicate`] tree. Re-parses the verbatim source slice
/// the AST captured (since Wave 1.5's parser stops at the next
/// modifier keyword without validating the predicate grammar), then
/// runs the same vocabulary check the legacy `when` path uses, so the
/// `else` branch surfaces typos with a `clause: "else"` diagnostic.
fn parse_when_branch(
    source:  &str,
    clause:  &'static str,
    ability: &str,
    span:    Span,
) -> Result<WhenPredicate, LowerError> {
    let expr = dsl_ast::parser::parse_expression(source).map_err(|e| {
        LowerError::WhenConditionParseError {
            ability:   ability.to_string(),
            clause,
            predicate: source.to_string(),
            reason:    e.message.clone(),
            span,
        }
    })?;
    if let Some((binder, field)) = first_unknown_agent_field(&expr) {
        return Err(LowerError::WhenConditionUnknownField {
            ability: ability.to_string(),
            clause,
            binder,
            field,
            span,
        });
    }
    extract_when_predicate(&expr, ability, clause, span)
}

/// Task #227: recursive extraction of a [`WhenPredicate`] tree from
/// the parsed when-condition expression. Compound combinators
/// (`&&` / `||` / `!`) lower to `WhenPredicate::And` / `Or` / `Not`;
/// leaf comparisons lower via [`extract_predicate_atom`] to
/// `WhenPredicate::Atom`. Parenthesized sub-expressions follow the
/// parser's precedence (`!` > `&&` > `||`).
///
/// Out-of-scope leaf shapes (field-vs-field comparisons,
/// non-`<binder>.<field> <op> <literal>` shapes) surface as
/// `WhenConditionUnsupported`. Out-of-vocab agent fields surface as
/// `WhenConditionUnsupportedField`. Both errors fire from the leaf
/// extractor — the recursive walk preserves the "fail-at-the-typo"
/// diagnostic shape.
fn extract_when_predicate(
    expr:    &dsl_ast::ast::Expr,
    ability: &str,
    clause:  &'static str,
    span:    Span,
) -> Result<WhenPredicate, LowerError> {
    use dsl_ast::ast::{BinOp, ExprKind, UnOp};
    match &expr.kind {
        ExprKind::Binary { op: BinOp::And, lhs, rhs } => {
            let l = extract_when_predicate(lhs, ability, clause, span)?;
            let r = extract_when_predicate(rhs, ability, clause, span)?;
            Ok(WhenPredicate::And(Box::new(l), Box::new(r)))
        }
        ExprKind::Binary { op: BinOp::Or, lhs, rhs } => {
            let l = extract_when_predicate(lhs, ability, clause, span)?;
            let r = extract_when_predicate(rhs, ability, clause, span)?;
            Ok(WhenPredicate::Or(Box::new(l), Box::new(r)))
        }
        ExprKind::Unary { op: UnOp::Not, rhs } => {
            let inner = extract_when_predicate(rhs, ability, clause, span)?;
            Ok(WhenPredicate::Not(Box::new(inner)))
        }
        // Anything else: try the leaf comparison extractor.
        _ => extract_predicate_atom(expr, ability, clause, span)
            .map(WhenPredicate::Atom),
    }
}

/// Count the number of RPN nodes a `WhenPredicate` tree serializes
/// to (atoms + operators). Used by the lower path to surface
/// `WhenConditionTreeTooLarge` early when the tree exceeds the SoA
/// stride.
fn count_rpn_nodes(tree: &WhenPredicate) -> usize {
    match tree {
        WhenPredicate::Atom(_) => 1,
        WhenPredicate::And(l, r) | WhenPredicate::Or(l, r) => {
            count_rpn_nodes(l) + count_rpn_nodes(r) + 1
        }
        WhenPredicate::Not(inner) => count_rpn_nodes(inner) + 1,
    }
}

/// Wave 1.5#7 GPU eval: extract a single leaf [`EffectPredicate`]
/// atom from the parsed when-condition expression. Restricted vocab
/// (matched as a flat top-level binary expression):
///   * Form: `<binder>.<field> <op> <literal>` (or `<literal> <op>
///     <binder>.<field>` — flipped via op-flip below).
///   * binder ∈ {self, target} → [`EffectPredicateBinder`].
///   * field name → maps via [`AgentFieldId::from_snake`] for vocabulary
///     validation, then narrows to a [`ScalingStatRef`] discriminant
///     (the GPU dispatcher's per-stat agent-SoA bindings cover only
///     this 8-field subset).
///   * op ∈ {`<`, `<=`, `>`, `>=`, `==`, `!=`} → [`EffectPredicateOp`].
///   * literal: `LitFloat` or `LitInt` (int promotes to f32).
///
/// Compound combinators (`&&` / `||` / `!`) are handled by
/// [`extract_when_predicate`] and never reach this path. Field-vs-field
/// comparisons and non-binary expressions surface as
/// `WhenConditionUnsupported`; out-of-vocab agent fields surface as
/// `WhenConditionUnsupportedField`.
///
/// Pre-condition: `first_unknown_agent_field` already rejected any
/// unknown-to-`AgentFieldId` field — so the field name resolves
/// through `AgentFieldId::from_snake`. This helper additionally
/// narrows to the `ScalingStatRef`-shaped subset.
fn extract_predicate_atom(
    expr:    &dsl_ast::ast::Expr,
    ability: &str,
    clause:  &'static str,
    span:    Span,
) -> Result<EffectPredicate, LowerError> {
    use dsl_ast::ast::{BinOp, ExprKind};

    // Top-level must be a Binary expression with a comparison op.
    let (op_ast, lhs, rhs) = match &expr.kind {
        ExprKind::Binary { op, lhs, rhs } => (*op, lhs.as_ref(), rhs.as_ref()),
        _ => return Err(LowerError::WhenConditionUnsupported {
            ability: ability.to_string(),
            clause,
            reason:  "expected `<binder>.<field> <op> <literal>` (top-level binary comparison)".to_string(),
            span,
        }),
    };
    let op = match op_ast {
        BinOp::Lt   => EffectPredicateOp::Lt,
        BinOp::LtEq => EffectPredicateOp::Le,
        BinOp::Gt   => EffectPredicateOp::Gt,
        BinOp::GtEq => EffectPredicateOp::Ge,
        BinOp::Eq   => EffectPredicateOp::Eq,
        BinOp::NotEq=> EffectPredicateOp::Ne,
        // Compound predicates (`&&`/`||`) reach this leaf path only
        // when an author wrote something like `(a && b) < 5` —
        // structurally invalid as an atom. Surface a pointed
        // diagnostic.
        BinOp::And | BinOp::Or => return Err(LowerError::WhenConditionUnsupported {
            ability: ability.to_string(),
            clause,
            reason:  "compound predicate (&&/||) is not a valid leaf comparison — only `<binder>.<field> <op> <literal>` is".to_string(),
            span,
        }),
        _ => return Err(LowerError::WhenConditionUnsupported {
            ability: ability.to_string(),
            clause,
            reason:  format!("operator `{op_ast:?}` not supported in restricted predicate vocab (use one of < <= > >= == !=)"),
            span,
        }),
    };
    // Match `<binder>.<field>` on one side, literal on the other. If
    // literal is on LHS, flip the op so the canonical form has
    // (binder.field) on the LHS for the evaluator.
    let (binder_ast, field_name, literal, op) = match (
        extract_binder_field(lhs),
        extract_literal_f32(rhs),
    ) {
        (Some((b, f)), Some(lit)) => (b, f, lit, op),
        _ => match (extract_binder_field(rhs), extract_literal_f32(lhs)) {
            (Some((b, f)), Some(lit)) => (b, f, lit, flip_op(op)),
            _ => return Err(LowerError::WhenConditionUnsupported {
                ability: ability.to_string(),
                clause,
                reason:  "expected `<binder>.<field> <op> <literal>` — got field-vs-field or non-literal operand".to_string(),
                span,
            }),
        },
    };
    let binder = match binder_ast.as_str() {
        "self"   => EffectPredicateBinder::SelfBinder,
        "target" => EffectPredicateBinder::Target,
        _ => return Err(LowerError::WhenConditionUnsupported {
            ability: ability.to_string(),
            clause,
            reason:  format!("binder `{binder_ast}` not supported (only `self` and `target`)"),
            span,
        }),
    };
    // Map field name → ScalingStatRef discriminant. ScalingStatRef
    // exposes the canonical 8-field f32 subset the GPU dispatcher's
    // per-stat agent-SoA bindings cover; anything outside that errors
    // as `WhenConditionUnsupportedField` (distinct from
    // `WhenConditionUnknownField` — the latter rejects fields outside
    // the broader agent vocabulary entirely).
    let stat_ref = match ScalingStatRef::parse(&field_name) {
        Some(s) => s,
        None => return Err(LowerError::WhenConditionUnsupportedField {
            ability: ability.to_string(),
            clause,
            field:   field_name,
            span,
        }),
    };
    Ok(EffectPredicate {
        binder,
        field:   stat_ref.discriminant(),
        op,
        literal,
    })
}

/// Match a `<binder>.<field>` shape, returning `Some((binder_ident,
/// field_name))` when the expression is a `Field(Ident(binder),
/// field_name)`. Other shapes return `None`.
fn extract_binder_field(expr: &dsl_ast::ast::Expr) -> Option<(String, String)> {
    use dsl_ast::ast::ExprKind;
    if let ExprKind::Field(base, field_name) = &expr.kind {
        if let ExprKind::Ident(binder) = &base.kind {
            return Some((binder.clone(), field_name.clone()));
        }
    }
    None
}

/// Match a `LitFloat` or `LitInt`, returning the numeric value as f32.
/// Other shapes (idents, fields, calls) return `None`.
fn extract_literal_f32(expr: &dsl_ast::ast::Expr) -> Option<f32> {
    use dsl_ast::ast::ExprKind;
    match &expr.kind {
        ExprKind::Float(v) => Some(*v as f32),
        ExprKind::Int(v)   => Some(*v as f32),
        _ => None,
    }
}

/// Flip a comparison op so `(literal op binder.field)` becomes
/// `(binder.field flipped_op literal)`. Used when the parser captures
/// the literal on the LHS — the canonical evaluator expects the binder
/// on the LHS.
#[inline]
fn flip_op(op: EffectPredicateOp) -> EffectPredicateOp {
    match op {
        EffectPredicateOp::Lt => EffectPredicateOp::Gt,
        EffectPredicateOp::Le => EffectPredicateOp::Ge,
        EffectPredicateOp::Gt => EffectPredicateOp::Lt,
        EffectPredicateOp::Ge => EffectPredicateOp::Le,
        EffectPredicateOp::Eq => EffectPredicateOp::Eq,
        EffectPredicateOp::Ne => EffectPredicateOp::Ne,
    }
}

/// #143 follow-up: walk a parsed when-condition expression looking for
/// `<binder>.<field>` patterns where binder is `self` or `target` (the
/// two binders the `.ability` surface conventionally uses for cast-time
/// agent reads). Return the first one whose field name is NOT in the
/// engine's `AgentFieldId` vocabulary.
///
/// Returns `Some((binder, field))` for the first unknown access, or
/// `None` when every agent-binder access references a known field.
/// Non-agent binders (`world.tick`, `config.foo`, `ability::hint`,
/// builtins like `count`/`sum`/`forall`) are skipped — they have their
/// own validation paths and aren't expected to round-trip through the
/// AgentFieldId table.
///
/// The walk handles all `ExprKind` shapes recursively, not just the
/// flat `Field(Ident, _)` pattern, so nested expressions like
/// `(target.htp + 5) * 2` still surface the typo.
fn first_unknown_agent_field(expr: &dsl_ast::ast::Expr) -> Option<(String, String)> {
    use dsl_ast::ast::ExprKind;
    use crate::cg::data_handle::AgentFieldId;
    match &expr.kind {
        ExprKind::Field(base, field_name) => {
            // Recurse into the base first — `(a.b).c` should validate
            // `a.b` then `<base>.c`.
            if let Some(found) = first_unknown_agent_field(base) {
                return Some(found);
            }
            // Only validate accesses where the base is a bare
            // `self` / `target` ident. Other shapes (function calls,
            // index expressions, nested fields like `target.belief.x`)
            // aren't routed through AgentFieldId today.
            if let ExprKind::Ident(binder) = &base.kind {
                if (binder == "self" || binder == "target")
                    && AgentFieldId::from_snake(field_name).is_none()
                {
                    return Some((binder.clone(), field_name.clone()));
                }
            }
            None
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            first_unknown_agent_field(lhs).or_else(|| first_unknown_agent_field(rhs))
        }
        ExprKind::Unary { rhs, .. } => first_unknown_agent_field(rhs),
        ExprKind::In { item, set } | ExprKind::Contains { set, item } => {
            first_unknown_agent_field(item).or_else(|| first_unknown_agent_field(set))
        }
        ExprKind::Call(callee, args) => {
            if let Some(f) = first_unknown_agent_field(callee) {
                return Some(f);
            }
            args.iter().find_map(|a| first_unknown_agent_field(&a.value))
        }
        ExprKind::Index(base, idx) => {
            first_unknown_agent_field(base).or_else(|| first_unknown_agent_field(idx))
        }
        ExprKind::If { cond, then_expr, else_expr } => {
            first_unknown_agent_field(cond)
                .or_else(|| first_unknown_agent_field(then_expr))
                .or_else(|| else_expr.as_deref().and_then(first_unknown_agent_field))
        }
        ExprKind::Quantifier { iter, body, .. } => {
            first_unknown_agent_field(iter).or_else(|| first_unknown_agent_field(body))
        }
        ExprKind::Fold { iter, body, .. } => {
            iter.as_deref().and_then(first_unknown_agent_field)
                .or_else(|| first_unknown_agent_field(body))
        }
        ExprKind::List(items) | ExprKind::Tuple(items) => {
            items.iter().find_map(first_unknown_agent_field)
        }
        ExprKind::Struct { fields, .. } => {
            fields.iter().find_map(|f| first_unknown_agent_field(&f.value))
        }
        ExprKind::Ctor { args, .. } => {
            args.iter().find_map(first_unknown_agent_field)
        }
        ExprKind::Match { scrutinee, arms } => {
            first_unknown_agent_field(scrutinee)
                .or_else(|| arms.iter().find_map(|a| first_unknown_agent_field(&a.body)))
        }
        ExprKind::PerUnit { expr, delta } => {
            first_unknown_agent_field(expr).or_else(|| first_unknown_agent_field(delta))
        }
        ExprKind::BeliefsAccessor { observer, target, .. }
        | ExprKind::BeliefsConfidence { observer, target } => {
            first_unknown_agent_field(observer).or_else(|| first_unknown_agent_field(target))
        }
        ExprKind::BeliefsView { observer, .. } => first_unknown_agent_field(observer),
        ExprKind::Block { bindings, expr } => {
            bindings
                .iter()
                .find_map(|(_, v)| first_unknown_agent_field(v))
                .or_else(|| first_unknown_agent_field(expr))
        }
        // Leaves: literals + bare idents have no nested fields to check.
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)
        | ExprKind::String(_) | ExprKind::Ident(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_ast::parse_ability_file;

    /// Wave 3 ToM Phase 3.5: `scry <observer> <subject>` lowers to
    /// `EffectOp::Scry { target_observer, subject_idx }`.
    #[test]
    fn scry_lowers_two_positional_ints() {
        let src = "ability Spy { target: self cooldown: 1s hint: utility scry 3 5 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Scry { target_observer, subject_idx } => {
                assert_eq!(target_observer, 3);
                assert_eq!(subject_idx, 5);
            }
            ref other => panic!("expected Scry; got {other:?}"),
        }
    }

    /// Wave 3 ToM unit pin — `plant_belief 5 bit 3` lowers to
    /// `EffectOp::PlantBelief { subject_idx: 5, fact_bit: 3 }`. This is
    /// the smallest end-to-end check that the new arm wires up: it
    /// proves the parser accepts the surface syntax (positional number
    /// + `bit` ident + positional number) and that the lowering casts
    /// both args to the right widths (u32 / u8).
    #[test]
    fn lower_plant_belief_basic() {
        let src = "ability Spy { target: enemy range: 5.0 cooldown: 1s plant_belief 5 bit 3 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::PlantBelief { subject_idx, fact_bit } => {
                assert_eq!(subject_idx, 5);
                assert_eq!(fact_bit, 3);
            }
            ref other => panic!("expected PlantBelief; got {other:?}"),
        }
    }

    /// `decoy <subject_idx> <fake_pos>` lowers to `EffectOp::Decoy`.
    /// The packed `fake_pos` value sits under 2^24 so it round-trips
    /// through the parser's f32 numeric storage exactly — values above
    /// the f32 mantissa boundary would need a hex-literal lex extension
    /// to surface the high bits cleanly.
    #[test]
    fn lower_decoy_two_args() {
        let src = "ability Bait { target: enemy cooldown: 1s decoy 5 12345678 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("decoy must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Decoy { subject_idx, fake_pos } => {
                assert_eq!(subject_idx, 5);
                assert_eq!(fake_pos, 12_345_678);
            }
            ref other => panic!("expected EffectOp::Decoy; got {other:?}"),
        }
    }

    /// Lift B — `cast_recipe <recipe_id>` (no tool target) lowers to
    /// `EffectOp::Recipe { recipe_id, target_tool: 0xFF }`. The 0xFF
    /// sentinel signals "no specific tool slot bound to this cast".
    #[test]
    fn lower_cast_recipe_no_tool() {
        let src = "ability ForgeIron { target: self cooldown: 5s cast_recipe 7 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("cast_recipe must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Recipe { recipe_id, target_tool } => {
                assert_eq!(recipe_id, 7);
                assert_eq!(target_tool, 0xFF);
            }
            ref other => panic!("expected EffectOp::Recipe; got {other:?}"),
        }
    }

    /// Lift B — `cast_recipe <recipe_id> target <tool_slot>` binds a
    /// specific tool slot to the cast. Both numbers clamp into their
    /// respective integer widths (u16 + u8).
    #[test]
    fn lower_cast_recipe_with_tool_target() {
        let src = "ability ForgeSword { target: self cooldown: 5s cast_recipe 12 target 3 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("cast_recipe must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Recipe { recipe_id, target_tool } => {
                assert_eq!(recipe_id, 12);
                assert_eq!(target_tool, 3);
            }
            ref other => panic!("expected EffectOp::Recipe; got {other:?}"),
        }
    }

    /// Lift B — `wear_tool <tool_kind> <amount>` lowers to
    /// `EffectOp::WearTool { tool_kind: u8, amount: u16 }`. Amount is
    /// q8 fraction-of-durability (256 = 1.0 of full durability).
    #[test]
    fn lower_wear_tool_two_args() {
        let src = "ability HammerSwing { target: self cooldown: 1s wear_tool 4 64 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("wear_tool must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::WearTool { tool_kind, amount } => {
                assert_eq!(tool_kind, 4);
                assert_eq!(amount, 64);
            }
            ref other => panic!("expected EffectOp::WearTool; got {other:?}"),
        }
    }

    /// Lift C — `propose <contract_kind>` (no expiry) lowers to
    /// `EffectOp::Propose { contract_kind, expires_at_tick: 0 }`. The
    /// 0 sentinel signals "proposal stays open until target accepts /
    /// declines or caster cancels".
    #[test]
    fn lower_propose_no_expiry() {
        let src = "ability OfferMarriage { target: enemy range: 5.0 cooldown: 1s propose 1 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("propose must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Propose { contract_kind, expires_at_tick } => {
                assert_eq!(contract_kind, 1);
                assert_eq!(expires_at_tick, 0);
            }
            ref other => panic!("expected EffectOp::Propose; got {other:?}"),
        }
    }

    /// Lift C — `propose <contract_kind> expires_at <tick>` binds an
    /// auto-cancel deadline.
    #[test]
    fn lower_propose_with_expiry() {
        let src = "ability OfferContract { target: enemy range: 5.0 cooldown: 1s propose 2 expires_at 5000 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("propose must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Propose { contract_kind, expires_at_tick } => {
                assert_eq!(contract_kind, 2);
                assert_eq!(expires_at_tick, 5000);
            }
            ref other => panic!("expected EffectOp::Propose; got {other:?}"),
        }
    }

    /// Lift C — `announce <kind> radius <cells>` lowers to
    /// `EffectOp::Announce { announcement_kind, radius_q8 }`. Radius is
    /// stored q8 (cells × 256). 3.5 cells → 896.
    #[test]
    fn lower_announce_with_radius() {
        let src = "ability TownCryer { target: self cooldown: 1s announce 7 radius 3.5 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("announce must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::Announce { announcement_kind, radius_q8 } => {
                assert_eq!(announcement_kind, 7);
                assert_eq!(radius_q8, 896);
            }
            ref other => panic!("expected EffectOp::Announce; got {other:?}"),
        }
    }

    /// Lift D — `gain_skill <skill_id> <amount>` lowers to
    /// `EffectOp::GainSkill { skill_id, amount_q8 }`. Self-cast skill
    /// growth — q8 amount (256 = full mastery, but consumer clamps).
    #[test]
    fn lower_gain_skill_two_args() {
        let src = "ability Practice { target: self cooldown: 1s gain_skill 3 32 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("gain_skill must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::GainSkill { skill_id, amount_q8 } => {
                assert_eq!(skill_id, 3);
                assert_eq!(amount_q8, 32);
            }
            ref other => panic!("expected EffectOp::GainSkill; got {other:?}"),
        }
    }

    /// Lift D — `create_obligation <obligation_id> <kind>` lowers to
    /// `EffectOp::CreateObligation`. The u16 obligation_id round-trips
    /// through the f32 parser path (under 2^24).
    #[test]
    fn lower_create_obligation_two_args() {
        let src = "ability Lend { target: enemy range: 5.0 cooldown: 1s create_obligation 17 0 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("create_obligation must lower");
        assert_eq!(prog.effects.len(), 1);
        match prog.effects[0] {
            EffectOp::CreateObligation { obligation_id, kind } => {
                assert_eq!(obligation_id, 17);
                assert_eq!(kind, 0);
            }
            ref other => panic!("expected EffectOp::CreateObligation; got {other:?}"),
        }
    }

    /// Plan G G2.6 — `cast { duration: 3t }` lowers into a single
    /// `EffectOp::CastBegin` with the duration carried through. The
    /// `Effects(_)` step body lowers into `pending_program` (option D
    /// deferred-resolution slot) — verified by the
    /// `cast_block_effects_lower_into_pending_program` test below.
    #[test]
    fn cast_block_lowers_to_cast_begin() {
        let src = "ability Firebolt { \
            target: enemy range: 8.0 cooldown: 5s \
            cast { duration: 3t interrupts: standard } \
            effect { damage 25 } \
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.effects.len(), 1, "cast{{}} ability should emit exactly one CastBegin op");
        match prog.effects[0] {
            EffectOp::CastBegin { ability_id, duration_ticks, target_slot, target_x_q8, target_y_q8 } => {
                assert_eq!(ability_id, 0, "ability_id placeholder at lowering time");
                assert_eq!(duration_ticks, 3, "3t cast duration");
                assert_eq!(target_slot, 0, "target_slot placeholder");
                assert_eq!(target_x_q8, 0);
                assert_eq!(target_y_q8, 0);
            }
            ref other => panic!("expected EffectOp::CastBegin; got {other:?}"),
        }
    }

    /// Plan G option D — the `effect { … }` step body of a cast{}
    /// program lowers into `pending_program`, NOT into `effects`.
    /// `effects` carries only the immediate-cast IR (CastBegin); the
    /// busy-resolution kernel will fire `pending_program` later via
    /// `apply_pending_program`.
    #[test]
    fn cast_block_effects_lower_into_pending_program() {
        let src = "ability Firebolt { \
            target: enemy range: 8.0 cooldown: 5s \
            cast { duration: 3t interrupts: standard } \
            effect { damage 25 } \
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.pending_program.len(), 1,
            "effect{{}} body should populate pending_program with one op");
        match prog.pending_program[0] {
            EffectOp::Damage { amount } => {
                assert!((amount - 25.0).abs() < 1e-3, "damage amount should round-trip; got {amount}");
            }
            ref other => panic!("expected EffectOp::Damage; got {other:?}"),
        }
        // Sanity: legacy bare-effect abilities have empty pending_program.
        let bare_src = "ability Strike { target: enemy range: 1.5 cooldown: 1s damage 10 }";
        let bare_file = parse_ability_file(bare_src).expect("parser");
        let bare_prog = lower_ability_decl(&bare_file.abilities[0]).expect("lowering");
        assert!(bare_prog.pending_program.is_empty(),
            "bare-effect ability must have empty pending_program");
    }

    /// Plan G option D — pure-utility cast (`cast{}` with no
    /// `effect{}` sibling). Emits CastBegin in `effects` and leaves
    /// `pending_program` empty. The busy-resolution kernel sees the
    /// empty pending slot and clears busy state without firing damage
    /// — useful for non-damage casts (a 3-tick channel that just
    /// blocks the agent).
    #[test]
    fn cast_only_lowers_with_empty_pending_program() {
        let src = "ability Channel {
            target: self cooldown: 5s
            cast { duration: 5t interrupts: standard }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.effects.len(), 1, "CastBegin still emitted for cast-only programs");
        assert!(matches!(prog.effects[0], EffectOp::CastBegin { .. }));
        assert!(prog.pending_program.is_empty(),
            "pending_program empty when no effect{{}} follows");
    }

    /// Plan G option D MVP — multi-stage chains
    /// (cast → effect → cast → effect) are NOT yet supported. The
    /// lowering captures the FIRST `Cast` step's CastBegin and
    /// silently skips subsequent Cast steps; all `Effects` blocks
    /// (in source order) merge into a single pending_program. This
    /// pin documents the MVP behaviour so the future multi-stage
    /// slice can update the assertion in lockstep.
    #[test]
    fn multi_stage_cast_chain_takes_first_cast_only_today() {
        let src = "ability TwoStage {
            target: enemy range: 5.0 cooldown: 8s
            cast { duration: 3t interrupts: standard }
            effect { damage 10 }
            cast { duration: 5t interrupts: standard }
            effect { damage 20 }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("multi-stage must lower (first-cast-only MVP)");
        assert_eq!(prog.effects.len(), 1, "first Cast step's CastBegin only");
        // Both effect{} blocks merge into pending_program.
        assert_eq!(prog.pending_program.len(), 2);
        match prog.effects[0] {
            EffectOp::CastBegin { duration_ticks, .. } => {
                assert_eq!(duration_ticks, 3, "must take FIRST cast's duration, not subsequent");
            }
            ref other => panic!("expected CastBegin; got {other:?}"),
        }
    }

    /// Plan G — `interrupts: none` is a valid InterruptSet. The
    /// lowering should accept it; the busy-resolution kernel reads
    /// the set per-fixture so the lowering doesn't need to do
    /// anything special with it today.
    #[test]
    fn cast_block_with_interrupts_none_lowers() {
        let src = "ability BindSoul {
            target: self cooldown: 60s
            cast { duration: 10t interrupts: none }
            effect { heal 50 }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("uninterruptible cast must lower");
        assert_eq!(prog.effects.len(), 1);
        assert_eq!(prog.pending_program.len(), 1);
        assert!(matches!(prog.effects[0], EffectOp::CastBegin { .. }));
        assert!(matches!(prog.pending_program[0], EffectOp::Heal { .. }));
        assert_eq!(prog.cast_interrupt_mask, Some(InterruptMask::none()),
            "cast_interrupt_mask should resolve `interrupts: none` to InterruptMask::none()");
    }

    /// Plan G G2.5 — `cast_interrupt_mask` is populated for cast{}
    /// programs from the parsed `interrupts:` declaration. The
    /// engine-side helper `should_interrupt(...)` consults the same
    /// mask; this pin proves the lowering writes what the kernel
    /// will read.
    #[test]
    fn cast_interrupt_mask_round_trips_from_source() {
        // standard mask
        let std_src = "ability Firebolt {
            target: enemy range: 8.0 cooldown: 5s
            cast { duration: 3t interrupts: standard }
            effect { damage 25 }
        }";
        let file = parse_ability_file(std_src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.cast_interrupt_mask, Some(InterruptMask::standard()));

        // legacy bare-effect ability has no cast{} block → None.
        let bare_src = "ability Strike { target: enemy range: 1.5 cooldown: 1s damage 10 }";
        let bare_file = parse_ability_file(bare_src).expect("parser");
        let bare_prog = lower_ability_decl(&bare_file.abilities[0]).expect("lowering");
        assert_eq!(bare_prog.cast_interrupt_mask, None,
            "bare-effect abilities have no cast block → no interrupt mask");
    }

    /// Plan G G2.5 — `lower_interrupt_set` round-trips every AST
    /// shape into the engine packed mask. Pin the resolution at the
    /// helper level so the runtime kernel always sees a fixed
    /// bitmask (not the AST shape).
    #[test]
    fn interrupt_set_lowering_round_trip() {
        use dsl_ast::ast::{InterruptKind as AstK, InterruptSet};
        // standard = Damage | Stun | CasterDied | TargetDied (4 bits).
        assert_eq!(super::lower_interrupt_set(&InterruptSet::Standard),
                   InterruptMask::standard());
        // none — uninterruptible.
        assert_eq!(super::lower_interrupt_set(&InterruptSet::None),
                   InterruptMask::none());
        // Subset { Damage, Movement } — explicit pair, NOT standard.
        let subset = InterruptSet::Subset(vec![AstK::Damage, AstK::Movement]);
        let m = super::lower_interrupt_set(&subset);
        assert!(m.contains(InterruptKind::Damage));
        assert!(m.contains(InterruptKind::Movement));
        assert!(!m.contains(InterruptKind::Stun),
            "subset is exact — Stun NOT included");
        // standard + { movement } — every standard kind PLUS Movement.
        let plus = InterruptSet::StandardPlus(vec![AstK::Movement]);
        assert_eq!(super::lower_interrupt_set(&plus), InterruptMask::all());
        // standard - { damage } — forager-style "keep moving under fire".
        let minus = InterruptSet::StandardMinus(vec![AstK::Damage]);
        let mm = super::lower_interrupt_set(&minus);
        assert!(!mm.contains(InterruptKind::Damage));
        assert!(mm.contains(InterruptKind::Stun));
        assert!(mm.contains(InterruptKind::CasterDied));
        assert!(mm.contains(InterruptKind::TargetDied));
    }

    /// Plan G option D — multiple statements inside one `effect{}` block
    /// lower in order into pending_program, preserving authored order.
    #[test]
    fn cast_block_effects_preserve_order() {
        let src = "ability Combo {
            target: enemy range: 5.0 cooldown: 8s
            cast { duration: 5t interrupts: standard }
            effect {
                damage 30
                stun 1s
            }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.pending_program.len(), 2, "two pending ops expected");
        assert!(matches!(prog.pending_program[0], EffectOp::Damage { .. }),
            "first pending op should be Damage; got {:?}", prog.pending_program[0]);
        assert!(matches!(prog.pending_program[1], EffectOp::Stun { .. }),
            "second pending op should be Stun; got {:?}", prog.pending_program[1]);
    }

    /// Legacy abilities (no `cast {`/`effect {` blocks) keep lowering
    /// the bare effects list. Regression guard so the G2.6 program
    /// branch never silently swallows the legacy path.
    #[test]
    fn bare_effect_ability_still_lowers_legacy_path() {
        let src = "ability Strike { target: enemy range: 1.5 cooldown: 1s damage 10 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.effects.len(), 1);
        assert!(matches!(prog.effects[0], EffectOp::Damage { amount } if (amount - 10.0).abs() < 1e-3),
            "legacy bare-effect path should still emit EffectOp::Damage; got {:?}", prog.effects[0]);
    }

    /// Plan G G3e — `cast { telegraph: circle(self.pos, radius: R) }`
    /// populates `telegraph_kind` (= `TelegraphKind::Circle`) and
    /// `telegraph_params[0] = R` on the lowered program. The packed
    /// registry's `pack_telegraph_metadata_column` test in
    /// `engine::ability::packed::tests` then asserts these flow into
    /// the SoA columns the threats fold (G3g) reads.
    #[test]
    fn cast_block_with_circle_telegraph_lowers_to_program_fields() {
        let src = "ability Firebolt {
            target: enemy range: 8.0 cooldown: 5s
            cast { duration: 3t; telegraph: circle(self.pos, radius: 4) }
            effect { damage 25 }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.telegraph_kind, TelegraphKind::Circle.discriminant(),
            "circle telegraph should set kind to Circle (= 1)");
        assert!((prog.telegraph_params[0] - 4.0).abs() < 1e-6,
            "circle telegraph radius should land in params[0]; got {:?}",
            prog.telegraph_params);
        assert_eq!(&prog.telegraph_params[1..], &[0.0, 0.0, 0.0],
            "trailing params zero-pad");
    }

    /// Plan G G3e — same as the circle test but for the line shape;
    /// `width: W` lands in `params[0]`.
    #[test]
    fn cast_block_with_line_telegraph_lowers_to_program_fields() {
        let src = "ability Beam {
            target: enemy range: 12.0 cooldown: 10s
            cast { duration: 2t; telegraph: line(self.pos, target.pos, width: 3) }
            effect { damage 40 }
        }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.telegraph_kind, TelegraphKind::Line.discriminant(),
            "line telegraph should set kind to Line (= 2)");
        assert!((prog.telegraph_params[0] - 3.0).abs() < 1e-6,
            "line telegraph width should land in params[0]; got {:?}",
            prog.telegraph_params);
    }

    /// Plan G G3e — abilities with no telegraph (legacy bare-effect or
    /// cast{} without the field) leave the sentinel + zero defaults.
    #[test]
    fn ability_without_telegraph_carries_sentinel_default() {
        let src = "ability Strike { target: enemy range: 1.5 cooldown: 1s damage 10 }";
        let file = parse_ability_file(src).expect("parser");
        let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
        assert_eq!(prog.telegraph_kind, TELEGRAPH_KIND_NONE,
            "no telegraph → sentinel kind");
        assert_eq!(prog.telegraph_params, [0.0; 4],
            "no telegraph → zero params");
    }
}
