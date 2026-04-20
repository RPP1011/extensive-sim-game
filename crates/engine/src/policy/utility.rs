//! Utility-function policy backend.
//!
//! Milestone 5 (compiler-first scoring): the per-kind `match` that used
//! to live here — `utility_score(kind, hp, max_hp)` — is gone. Every
//! score now lives as a row in `engine_rules::scoring::SCORING_TABLE`,
//! emitted by `dsl_compiler` from `assets/sim/scoring.sim`. The scorer
//! here is a hand-written interpreter for the POD row format: it
//! iterates the table, evaluates each row's predicates against the
//! current agent, accumulates `base + Σ active-modifier deltas`, and
//! keeps the best score per action-head (filtered through the mask).
//!
//! Target-resolution logic (`nearest_other` + `Action::move_toward` /
//! `Action::attack`) stays hand-written — it's action construction, not
//! scoring. Adding a new scoring row touches only `scoring.sim`; adding
//! a new action-head with a new target-selection rule touches both.
//!
//! Any field referenced from DSL as `self.<name>` must have a matching
//! arm in `read_field` below. The mapping is documented in
//! `docs/dsl/scoring_fields.md` — changing it is a schema bump.

use super::{Action, PolicyBackend};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::state::SimState;
use engine_rules::scoring::{PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE};

pub struct UtilityBackend;

impl PolicyBackend for UtilityBackend {
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let row_start = slot * MicroKind::ALL.len();
            let mut best_kind = MicroKind::Hold;
            let mut best_score = f32::MIN;
            let mut have_best = false;

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

                let score = score_entry(entry, state, id);
                if !have_best || score > best_score {
                    best_score = score;
                    best_kind = kind;
                    have_best = true;
                }
            }

            // No mask-allowed entry? Fall through to Hold — same failsafe as
            // the legacy code. The mask builder always sets Hold for alive
            // agents so this branch should be unreachable in practice.
            if !have_best {
                out.push(Action::hold(id));
                continue;
            }

            out.push(build_action(best_kind, id, state));
        }
    }
}

/// Score one table row against the current agent. `base` always applies;
/// each modifier's predicate is evaluated in turn and the delta added when
/// the predicate passes. Modifiers beyond `modifier_count` are padding
/// (`ModifierRow::EMPTY`) and are skipped by the count gate.
fn score_entry(entry: &ScoringEntry, state: &SimState, agent: AgentId) -> f32 {
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
        if eval_predicate(&row.predicate, state, agent) {
            score += row.delta;
        }
    }
    score
}

/// Read a scalar field referenced by the scoring table. `field_id` is
/// owned by the compiler — see `docs/dsl/scoring_fields.md`. Any drift
/// between the compiler's emission and this dispatch is a schema bug
/// (catches at the `SCORING_HASH` gate in CI).
fn read_field(state: &SimState, agent: AgentId, field_id: u16) -> f32 {
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
fn eval_predicate(pred: &PredicateDescriptor, state: &SimState, agent: AgentId) -> bool {
    match pred.kind {
        PredicateDescriptor::KIND_ALWAYS => true,
        PredicateDescriptor::KIND_SCALAR_COMPARE => {
            let lhs = read_field(state, agent, pred.field_id);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar(pred.op, lhs, rhs)
        }
        _ => false,
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

/// Turn the winning `(kind, agent)` pair into a concrete `Action`.
/// Target-resolution lives here rather than in the scoring table — the
/// table ranks kinds, the action builder chooses where to point them.
fn build_action(kind: MicroKind, id: AgentId, state: &SimState) -> Action {
    match kind {
        MicroKind::Hold => Action::hold(id),
        MicroKind::MoveToward => {
            if let Some(target) = nearest_other(state, id) {
                let pos = state.agent_pos(target).unwrap();
                Action::move_toward(id, pos)
            } else {
                Action::hold(id)
            }
        }
        MicroKind::Attack => {
            if let Some(target) = nearest_other(state, id) {
                Action::attack(id, target)
            } else {
                Action::hold(id)
            }
        }
        MicroKind::Eat => Action::eat(id),
        // Domain hooks without a dedicated constructor fall back to Hold —
        // mask shouldn't have enabled them unless the scorer has a row
        // ranking them above Hold, which doesn't happen until the DSL
        // declares one.
        _ => Action::hold(id),
    }
}

fn nearest_other(state: &SimState, self_id: AgentId) -> Option<AgentId> {
    let self_pos = state.agent_pos(self_id)?;
    state
        .agents_alive()
        .filter(|id| *id != self_id)
        .min_by(|a, b| {
            let da = (state.agent_pos(*a).unwrap() - self_pos).length_squared();
            let db = (state.agent_pos(*b).unwrap() - self_pos).length_squared();
            da.total_cmp(&db)
        })
}
