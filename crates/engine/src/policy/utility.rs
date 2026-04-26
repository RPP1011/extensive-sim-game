//! Utility-function policy backend.
//!
//! Task 138 reshaped the target-resolution layer. Scoring rows rank
//! action heads (base + Σ active-modifier deltas), and target-bound
//! heads (Attack / MoveToward) argmax over the per-agent candidate
//! lists produced by the compiler-emitted `mask_<name>_candidates`
//! enumerators (`TargetMask`). The old `nearest_other` heuristic — which
//! picked targets regardless of hostility or utility — is gone.
//!
//! Per-tick flow:
//!
//! 1. For every alive agent, walk `SCORING_TABLE`:
//!    - Self-only kinds (Hold / Flee / Eat / …) score once with
//!      `score_entry(entry, state, agent, None)`.
//!    - Target-bound kinds (Attack / MoveToward) score each candidate
//!      in `target_mask.candidates_for(agent, kind)` with
//!      `score_entry(entry, state, agent, Some(target))` and keep the
//!      highest-scoring (kind, target) pair.
//! 2. The winning (kind, agent, target) is turned into an `Action` via
//!    `build_action`. Kinds without a dedicated constructor fall back
//!    to `Action::hold`.
//!
//! Any field referenced from DSL as `self.<name>` or `target.<name>`
//! must have a matching arm in `read_field` below. The mapping is
//! documented in `docs/dsl/scoring_fields.md`; changing it is a schema
//! bump.

use super::{Action, ActionKind, MicroTarget, PolicyBackend};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind, TargetMask};
use crate::state::SimState;
use engine_data::scoring::{PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE};

pub struct UtilityBackend;

impl PolicyBackend for UtilityBackend {
    fn evaluate(
        &self,
        state: &SimState,
        mask: &MaskBuffer,
        target_mask: &TargetMask,
        out: &mut Vec<Action>,
    ) {
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let row_start = slot * MicroKind::ALL.len();
            let mut best: Option<(MicroKind, Option<AgentId>, f32)> = None;

            for entry in SCORING_TABLE {
                // Skip rows whose action head isn't a known MicroKind. The
                // compiler's emitter rejects unknowns at build time, so this
                // path is defensive against a future table we don't know
                // about yet (e.g. macro actions added before engine support).
                let kind = match micro_kind_from_u16(entry.action_head) {
                    Some(k) => k,
                    None => continue,
                };
                // Mask gates the kind. Unavailable kinds contribute no score,
                // matching the legacy behaviour.
                if !mask.micro_kind[row_start + kind as usize] {
                    continue;
                }

                if kind.target_slot().is_some() {
                    // Target-bound — argmax over candidate targets. Task 138
                    // retired `nearest_other`; this is the sole target-
                    // selection path for Attack / MoveToward. An empty
                    // candidate list means the mask never set the bit, so
                    // this branch no-ops per the mask check above.
                    for &target in target_mask.candidates_for(id, kind) {
                        let score = score_entry(entry, state, id, Some(target));
                        match best {
                            None => best = Some((kind, Some(target), score)),
                            Some((_, _, bs)) if score > bs => {
                                best = Some((kind, Some(target), score));
                            }
                            _ => {}
                        }
                    }
                } else {
                    // Self-only — score the row once with no target binding.
                    let score = score_entry(entry, state, id, None);
                    match best {
                        None => best = Some((kind, None, score)),
                        Some((_, _, bs)) if score > bs => {
                            best = Some((kind, None, score));
                        }
                        _ => {}
                    }
                }
            }

            // No mask-allowed entry? Fall through to Hold — same failsafe as
            // the legacy code. The mask builder always sets Hold for alive
            // agents so this branch should be unreachable in practice.
            match best {
                Some((kind, target, _)) => out.push(build_action(kind, id, target, state)),
                None => out.push(Action::hold(id)),
            }
        }
    }
}

/// Score one table row against the current agent / candidate target.
/// `base` always applies; each modifier's predicate is evaluated in turn
/// and the delta added when the predicate passes. Modifiers beyond
/// `modifier_count` are padding (`ModifierRow::EMPTY`) and are skipped
/// by the count gate.
///
/// `target` is `Some` for target-bound rows and `None` for self-only
/// rows. Scoring predicates that reference `target.<field>` observe
/// `Some(_)` here; self-only predicates ignore it.
fn score_entry(
    entry: &ScoringEntry,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> f32 {
    let mut score = entry.base;
    // Personality dot-product: not wired at milestone 5 (all weights are
    // zero in the emitted table). Kept explicit so the call shape matches
    // when personality vectors land.
    let personality = read_personality(state, agent);
    for i in 0..entry.personality_weights.len() {
        score += entry.personality_weights[i] * personality[i];
    }

    let count = entry.modifier_count as usize;
    let max = count.min(MAX_MODIFIERS);
    for i in 0..max {
        let row = &entry.modifiers[i];
        // Gradient rows evaluate to a scalar: `score += view_value * delta`.
        // Everything else is boolean: `score += (pred ? delta : 0)`.
        // `KIND_VIEW_GRADIENT` is the fuzzy-scoring path that lets
        // decisions scale smoothly with an accumulated view value
        // (threat_level, future aggression / relationship views, …)
        // rather than flipping at a fixed threshold.
        match row.predicate.kind {
            PredicateDescriptor::KIND_VIEW_GRADIENT => {
                let v = eval_view_call(state, agent, target, &row.predicate);
                if v.is_finite() {
                    score += v * row.delta;
                }
            }
            PredicateDescriptor::KIND_BELIEF_GRADIENT => {
                let v = eval_belief_scalar(state, agent, target, &row.predicate);
                if v.is_finite() {
                    score += v * row.delta;
                }
            }
            _ => {
                if eval_predicate(&row.predicate, state, agent, target) {
                    score += row.delta;
                }
            }
        }
    }

    // Task 81 — terrain height-bonus modifier. Fires on the Attack row
    // when the shooter sits at least 2 m above the target AND the
    // straight-line segment between them is unobstructed terrain.
    //
    // Implemented as a hardcoded post-step rather than a DSL-emitted
    // `ModifierRow` because the scoring-table lowering path has no
    // vec3-typed `self.pos` / `target.pos` field-ids and no TerrainQuery
    // call kind. The MVP slice deliberately keeps the compiler change
    // minimal (namespace + method only, previous sub-commit); once a
    // concrete predicate encoding for `terrain.line_of_sight(a, b)` in
    // the POD `PredicateDescriptor` lands, this branch collapses to an
    // emitter-side row and goes away.
    //
    // The terrain backend defaults to `FlatPlane`, where every LOS is
    // clear and every height is 0 — so this branch is a pure no-op for
    // the wolves+humans parity fixture (every Attack pair sits at
    // z=0 ± 0.0, failing the `> 2.0` gate) and for every legacy test
    // that inherits the default.
    //
    // Numeric delta (+0.35) sits below the +0.4 `my_enemies` /
    // `pack_focus` bumps so terrain advantage augments but doesn't
    // dominate existing targeting signals. Tunable; once terrain is
    // ubiquitous enough to hit a balance sweep, move to config.
    if entry.action_head == MicroKind::Attack as u16 {
        if let Some(t) = target {
            score += terrain_height_bonus(state, agent, t);
        }
    }

    score
}

/// Task 81 height-advantage threshold (m) — shooter must sit strictly
/// above target by more than this to earn the bonus.
pub const TERRAIN_HEIGHT_THRESHOLD_M: f32 = 2.0;
/// Task 81 height-advantage bonus delta added to the Attack row when
/// both the elevation and LOS gates fire.
pub const TERRAIN_HEIGHT_BONUS: f32 = 0.35;

/// Terrain height-advantage bonus for `MicroKind::Attack`. Returns
/// `TERRAIN_HEIGHT_BONUS` when the shooter has >2 m of elevation over
/// the target and a clear line-of-sight; `0.0` otherwise.
///
/// The elevation gate is a hard step so a shooter standing exactly at
/// `z = target.z + 2.0` doesn't accidentally fire it — defenders need
/// to commit real height. The LOS check prevents the bonus firing
/// through solid terrain (a shooter on the far side of a cliff can't
/// claim the advantage on something they can't see).
///
/// Both reads go through `SimState`; a missing agent surfaces as no
/// bonus (None → 0.0) rather than a NaN propagation. Exposed `pub` so
/// tests and examples can observe the gate directly without round-
/// tripping through the full scorer.
#[inline]
pub fn terrain_height_bonus(state: &SimState, agent: AgentId, target: AgentId) -> f32 {
    let from = match state.agent_pos(agent) {
        Some(p) => p,
        None => return 0.0,
    };
    let to = match state.agent_pos(target) {
        Some(p) => p,
        None => return 0.0,
    };
    if from.z > to.z + TERRAIN_HEIGHT_THRESHOLD_M && state.terrain.line_of_sight(from, to) {
        TERRAIN_HEIGHT_BONUS
    } else {
        0.0
    }
}

/// Read a scalar field referenced by the scoring table. `field_id` is
/// owned by the compiler — see `docs/dsl/scoring_fields.md`. Any drift
/// between the compiler's emission and this dispatch is a schema bug
/// (catches at the `SCORING_HASH` gate in CI).
///
/// Target-side field ids live in the `0x4000..0x8000` reserved range
/// (task 138). `target == None` on a target-side read surfaces as
/// `f32::NAN`, matching the "fail closed" convention for unknown
/// descriptors.
fn read_field(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    field_id: u16,
) -> f32 {
    // Target-side field ids (task 138). `target == None` means the
    // modifier referenced `target.<field>` on a self-only row — invalid;
    // surface as NaN so comparisons fail.
    if field_id >= 0x4000 && field_id < 0x8000 {
        let target = match target {
            Some(t) => t,
            None => return f32::NAN,
        };
        return match field_id {
            0x4000 => state.agent_hp(target).unwrap_or(0.0),
            0x4001 => state.agent_max_hp(target).unwrap_or(1.0),
            0x4002 => {
                let hp = state.agent_hp(target).unwrap_or(0.0);
                let max = state.agent_max_hp(target).unwrap_or(1.0);
                if max > 0.0 {
                    hp / max
                } else {
                    0.0
                }
            }
            0x4003 => state.agent_shield_hp(target).unwrap_or(0.0),
            _ => f32::NAN,
        };
    }

    match field_id {
        0 => state.agent_hp(agent).unwrap_or(0.0),
        1 => state.agent_max_hp(agent).unwrap_or(1.0),
        2 => {
            let hp = state.agent_hp(agent).unwrap_or(0.0);
            let max = state.agent_max_hp(agent).unwrap_or(1.0);
            if max > 0.0 {
                hp / max
            } else {
                0.0
            }
        }
        3 => state.agent_shield_hp(agent).unwrap_or(0.0),
        4 => state.agent_attack_range(agent).unwrap_or(2.0),
        // Psych-needs scalars (task 141). Absent-slot fallback is `0.0`
        // so a fresh agent with no needs state scores identically to one
        // whose needs read "fully sated".
        5 => state.agent_hunger(agent).unwrap_or(0.0),
        6 => state.agent_thirst(agent).unwrap_or(0.0),
        7 => state.agent_rest_timer(agent).unwrap_or(0.0),
        // Personality dims (aggression / social_drive / ambition /
        // altruism / curiosity). The SoA isn't wired yet — every read
        // is a constant `0.0` placeholder, matching `read_personality`.
        // Reserving the ids means a scoring row can reference
        // `self.personality.<dim>` today and pick up live values once
        // the SoA lands without a schema bump.
        8 => 0.0,  // self.personality.aggression
        9 => 0.0,  // self.personality.social_drive
        10 => 0.0, // self.personality.ambition
        11 => 0.0, // self.personality.altruism
        12 => 0.0, // self.personality.curiosity
        _ => f32::NAN,
    }
}

/// Placeholder personality vector. Returns zeros for every agent at
/// milestone 5; revisits when the personality SoA lands. Keeping the
/// function signature stable means the scorer's call site doesn't move.
fn read_personality(_state: &SimState, _agent: AgentId) -> [f32; 5] {
    [0.0; 5]
}

/// Evaluate a predicate descriptor. Unknown kinds return `false` — the
/// row contributes nothing, matching the "fail closed" convention for
/// unrecognised predicate shapes.
///
/// `KIND_VIEW_GRADIENT` is *not* a boolean predicate — `score_entry`
/// handles that kind specifically before calling this function.
fn eval_predicate(
    pred: &PredicateDescriptor,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> bool {
    match pred.kind {
        PredicateDescriptor::KIND_ALWAYS => true,
        PredicateDescriptor::KIND_SCALAR_COMPARE => {
            let lhs = read_field(state, agent, target, pred.field_id);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar(pred.op, lhs, rhs)
        }
        PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE => {
            let lhs = eval_view_call(state, agent, target, pred);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar(pred.op, lhs, rhs)
        }
        PredicateDescriptor::KIND_BELIEF_SCALAR_COMPARE => {
            let lhs = eval_belief_scalar(state, agent, target, pred);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar(pred.op, lhs, rhs)
        }
        _ => false,
    }
}

/// Evaluate a `@materialized` view call referenced by a scoring
/// predicate. `pred.field_id` carries the VIEW_ID (one of the
/// compile-time `VIEW_ID_*` constants in `engine_data::scoring`);
/// `pred.payload[4]` / `[5]` are arg-slot codes:
///
/// - `ARG_SELF = 0` → the scorer's current agent.
/// - `ARG_TARGET = 1` → the head's target binding (target-bound rows).
/// - `ARG_WILDCARD = 0xFE` → sum across the slot (e.g. Σ threat from
///   every recorded partner).
/// - `ARG_NONE = 0xFF` → unused (single-arg view).
///
/// One match arm per `VIEW_ID_*`. Unknown VIEW_IDs return NaN so a
/// forgotten arm fails closed (a scalar-compare against NaN is false;
/// a gradient times NaN is discarded by the `is_finite` gate in
/// `score_entry`).
fn eval_view_call(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let slot0 = pred.payload[4];
    let slot1 = pred.payload[5];
    match pred.field_id {
        // NOTE: VIEW_ID_THREAT_LEVEL, VIEW_ID_MY_ENEMIES, VIEW_ID_KIN_FEAR,
        // VIEW_ID_PACK_FOCUS, VIEW_ID_RALLY_BOOST — all read `state.views.*`
        // which is DELETED (Plan B1' Task 11). Return 0.0 (neutral) so the
        // engine crate compiles. Tests that exercise view-dependent scoring
        // are `#[ignore]`d. Re-enable after B1' Task 11 emits
        // engine_rules::step::step and the ViewRegistry moves to engine_rules.
        PredicateDescriptor::VIEW_ID_THREAT_LEVEL
        | PredicateDescriptor::VIEW_ID_MY_ENEMIES
        | PredicateDescriptor::VIEW_ID_KIN_FEAR
        | PredicateDescriptor::VIEW_ID_PACK_FOCUS
        | PredicateDescriptor::VIEW_ID_RALLY_BOOST => {
            let _ = (slot0, slot1, agent, target, state);
            0.0
        }
        _ => f32::NAN,
    }
}

/// Evaluate a belief-state scalar for use in `KIND_BELIEF_SCALAR_COMPARE`
/// and `KIND_BELIEF_GRADIENT` predicates.
///
/// Layout of `pred`:
/// - `field_id` = `BELIEF_FIELD_*` index.
/// - `payload[4]` = observer slot code (ARG_SELF=0 or ARG_TARGET=1).
/// - `payload[5]` = target slot code.
///
/// Returns the belief field value, or `f32::NAN` when the observer /
/// target cannot be resolved or no belief entry exists (fail-closed).
fn eval_belief_scalar(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let obs_slot = pred.payload[4];
    let tgt_slot = pred.payload[5];

    let observer = match obs_slot {
        PredicateDescriptor::ARG_SELF => agent,
        PredicateDescriptor::ARG_TARGET => match target {
            Some(t) => t,
            None => return f32::NAN,
        },
        _ => return f32::NAN,
    };
    let tgt = match tgt_slot {
        PredicateDescriptor::ARG_SELF => agent,
        PredicateDescriptor::ARG_TARGET => match target {
            Some(t) => t,
            None => return f32::NAN,
        },
        _ => return f32::NAN,
    };

    let beliefs = match state.agent_cold_beliefs(observer) {
        Some(b) => b,
        None => return f32::NAN,
    };
    let entry = match beliefs.get(&tgt) {
        Some(e) => e,
        None => return 0.0, // no belief entry → 0.0 (not NaN; absence is meaningful)
    };

    match pred.field_id {
        PredicateDescriptor::BELIEF_FIELD_LAST_KNOWN_HP => entry.last_known_hp,
        PredicateDescriptor::BELIEF_FIELD_LAST_KNOWN_MAX_HP => entry.last_known_max_hp,
        PredicateDescriptor::BELIEF_FIELD_CONFIDENCE => entry.confidence,
        PredicateDescriptor::BELIEF_FIELD_LAST_UPDATED_TICK => entry.last_updated_tick as f32,
        _ => f32::NAN,
    }
}

/// Map an arg-slot code to the concrete `AgentId` the view call should
/// receive. `None` for wildcard (handled specially by the caller) or
/// a target slot with no target bound.
#[allow(dead_code)]
fn resolve_slot(slot: u8, agent: AgentId, target: Option<AgentId>) -> Option<AgentId> {
    match slot {
        PredicateDescriptor::ARG_SELF => Some(agent),
        PredicateDescriptor::ARG_TARGET => target,
        _ => None,
    }
}

fn compare_scalar(op: u8, lhs: f32, rhs: f32) -> bool {
    // NaN short-circuits to false (consistent with IEEE comparisons), which
    // matches how an unknown `field_id` surfaces — no score change.
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    match op {
        PredicateDescriptor::OP_LT => lhs < rhs,
        PredicateDescriptor::OP_LE => lhs <= rhs,
        PredicateDescriptor::OP_EQ => lhs == rhs,
        PredicateDescriptor::OP_GE => lhs >= rhs,
        PredicateDescriptor::OP_GT => lhs > rhs,
        PredicateDescriptor::OP_NE => lhs != rhs,
        _ => false,
    }
}

/// Map a `u16` action-head tag back to a `MicroKind`. `None` for heads
/// the engine doesn't know — the scorer drops those rows, which is the
/// right behaviour once macro actions start appearing in the table.
fn micro_kind_from_u16(v: u16) -> Option<MicroKind> {
    // Keep the match exhaustive so a future MicroKind addition forces a
    // review here. We match `as u16` of the discriminant so this stays in
    // lockstep with `crates/engine/src/mask.rs`.
    let k = match v {
        0 => MicroKind::Hold,
        1 => MicroKind::MoveToward,
        2 => MicroKind::Flee,
        3 => MicroKind::Attack,
        4 => MicroKind::Cast,
        5 => MicroKind::UseItem,
        6 => MicroKind::Harvest,
        7 => MicroKind::Eat,
        8 => MicroKind::Drink,
        9 => MicroKind::Rest,
        10 => MicroKind::PlaceTile,
        11 => MicroKind::PlaceVoxel,
        12 => MicroKind::HarvestVoxel,
        13 => MicroKind::Converse,
        14 => MicroKind::ShareStory,
        15 => MicroKind::Communicate,
        16 => MicroKind::Ask,
        17 => MicroKind::Remember,
        _ => return None,
    };
    Some(k)
}

/// Turn a winning `(kind, agent, target)` tuple into a concrete `Action`.
/// Target-bound kinds expect `target == Some(_)`; self-only kinds expect
/// `None`. Kinds without a dedicated constructor fall back to Hold — the
/// scorer never picks them unless the DSL declares a row ranking them
/// above Hold, which doesn't happen until they land.
fn build_action(
    kind: MicroKind,
    id: AgentId,
    target: Option<AgentId>,
    state: &SimState,
) -> Action {
    match (kind, target) {
        (MicroKind::Hold, _) => Action::hold(id),
        (MicroKind::MoveToward, Some(t)) => match state.agent_pos(t) {
            Some(pos) => Action::move_toward(id, pos),
            None => Action::hold(id),
        },
        (MicroKind::Attack, Some(t)) => Action::attack(id, t),
        (MicroKind::Eat, _) => Action::eat(id),
        // Flee is self-only on the scorer's side (no candidate list —
        // the scoring row just ranks "should I run?"). The actual
        // threat that the engine's `step_full` Flee arm moves the agent
        // away from is resolved here: nearest hostile within
        // `config.combat.aggro_range`. Task 148. Without a hostile in
        // range we fall back to Hold — `step_full`'s Flee arm needs
        // `MicroTarget::Agent(threat)` to compute the away-vector, so
        // emitting Flee with no target would be a silent no-op anyway.
        (MicroKind::Flee, _) => match nearest_hostile(state, id) {
            Some(threat) => Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind:   MicroKind::Flee,
                    target: MicroTarget::Agent(threat),
                },
            },
            None => Action::hold(id),
        },
        // Target-bound kinds with no target, or domain-hook kinds without
        // a dedicated constructor — fall back to Hold. Mask shouldn't
        // have enabled them unless the scorer has a row ranking them
        // above Hold, which doesn't happen until the DSL declares one.
        _ => Action::hold(id),
    }
}

/// Pick the nearest hostile agent within `config.combat.aggro_range`.
/// Mirrors `mask::inferred_cast_target` — same "nearest hostile" shape,
/// just used by the Flee action builder instead of the Cast gate. Runs
/// once per Flee-picking agent per tick; cheap on small populations.
///
/// Task 148 — this is the runtime primitive that turns the scorer's
/// self-only Flee choice into an actual away-from-threat movement.
/// Lives as a hand-written helper rather than a scoring-row argmax
/// because Flee's scoring row (base + hp modifiers) doesn't enumerate
/// candidate threats — making Flee target-bound would require adding
/// it to `MicroKind::TARGET_BOUND` and growing a `mask_flee_candidates`
/// enumerator, which is tracked as a follow-up.
fn nearest_hostile(state: &SimState, self_id: AgentId) -> Option<AgentId> {
    let pos = state.agent_pos(self_id)?;
    let ct = state.agent_creature_type(self_id)?;
    let spatial = state.spatial();
    let mut best: Option<(AgentId, f32)> = None;
    for other in spatial.within_radius(state, pos, state.config.combat.aggro_range) {
        if other == self_id { continue; }
        let op = match state.agent_pos(other) { Some(p) => p, None => continue };
        let oc = match state.agent_creature_type(other) { Some(c) => c, None => continue };
        if !ct.is_hostile_to(oc) { continue; }
        let d = pos.distance(op);
        match best {
            None => best = Some((other, d)),
            Some((_, bd)) if d < bd => best = Some((other, d)),
            _ => {}
        }
    }
    best.map(|(id, _)| id)
}
