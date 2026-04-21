//! Task 197 — replace CPU `step_phases_1_to_3` with GPU scoring.
//!
//! Historical shape (task 193, Phase 6g):
//! ```text
//! step() {
//!   engine::step::step_phases_1_to_3(state, scratch, policy);  // CPU mask build — THE BOTTLENECK
//!   engine::step::apply_actions(state, scratch, events);       // CPU
//!   ...GPU cascade...
//!   ...CPU finalize...
//!   run_scoring_sidecar(state);   // duplicate GPU mask + scoring for diagnostics
//! }
//! ```
//!
//! The CPU mask-build + policy-evaluate at N=1000 costs ~170-700 ms/tick
//! (task 195's `PhaseTimings` profiling). Meanwhile the GPU already
//! produces the same per-agent decisions via the mask + scoring kernels
//! (tasks 183/184 + Phase 6d view reads) — just consumed as a post-tick
//! sidecar rather than as the tick's action source.
//!
//! This module closes that gap:
//!
//! 1. [`scoring_outputs_to_actions`] converts a `Vec<ScoreOutput>` (the
//!    GPU scoring kernel's per-slot argmax) into `Vec<Action>` in slot
//!    order, mirroring `UtilityBackend::evaluate`'s per-alive-agent
//!    emission contract. Target-bound heads (Attack / MoveToward) pull
//!    the target slot from `ScoreOutput.chosen_target`; Flee resolves
//!    its threat via `nearest_hostile` on the CPU (same path the CPU
//!    backend uses — the scorer doesn't enumerate flee threats).
//! 2. The caller shuffles the resulting action list with
//!    [`engine::step::shuffle_actions_in_place`] so the per-tick order
//!    matches the CPU backend byte-for-byte, then calls
//!    `engine::step::apply_actions` unchanged.
//!
//! Net effect: the ~170 ms CPU mask + evaluate phase collapses to a
//! ~2 ms GPU dispatch + O(N) scoring→actions conversion on CPU. The
//! GPU kernels already ran (as the sidecar); we just moved them to the
//! top of the tick and consume the output.
//!
//! ## Not a full port
//!
//! A WGSL `apply_actions` kernel — one thread per agent, emitting
//! events directly into the GPU event ring — is the end-state for
//! Phase 7+. That requires a flee `nearest_hostile` precompute (the
//! spatial kernel from task 186 already exists — just needs to feed
//! `apply_actions`), a GPU-side opportunity-attack emit path, and a
//! movement kernel that updates `agent_pos` in place. The scope
//! estimate landed outside task 197's budget; the compromise here
//! keeps the simulation byte-compatible with CPU apply semantics
//! (engagement slow, opportunity attacks, announce broadcasting, …)
//! while cutting the dominant N=1000 cost.

#![cfg(feature = "gpu")]

use engine::ids::AgentId;
use engine::mask::MicroKind;
use engine::policy::{Action, ActionKind, MicroTarget};
use engine::state::SimState;

use crate::scoring::{ScoreOutput, NO_TARGET};

/// Convert a `Vec<ScoreOutput>` (one entry per agent slot, as returned
/// by `ScoringKernel::run_and_readback`) into a `Vec<Action>` ready for
/// `engine::step::apply_actions`. The emission order matches
/// `UtilityBackend::evaluate`:
///
/// * One `Action` per alive agent, in ascending slot order.
/// * Dead slots contribute nothing (no Hold sentinel) — matches the CPU
///   backend's `for id in state.agents_alive()` loop.
/// * Target-bound heads (Attack / MoveToward) read `chosen_target` as a
///   0-based slot and resolve to the live `AgentId`; if the target slot
///   is `NO_TARGET` or points at a dead agent, the action falls back
///   to Hold — same guard as `build_action`'s `match (kind, target)`
///   arms.
/// * Flee is self-only on the scorer side. The CPU backend resolves the
///   actual threat via `nearest_hostile(self, aggro_range)`; we mirror
///   that path here so the `MicroTarget::Agent(threat)` the
///   `apply_actions` Flee arm expects is populated correctly.
/// * MoveToward reads the target slot's position. If the target moved
///   / died this tick (between the GPU scoring dispatch that picked it
///   and this conversion), we fall back to Hold — the CPU backend
///   would hit the same guard via `state.agent_pos(t)`.
///
/// Heads the scorer might emit but `apply_actions` doesn't know about
/// (future macro actions, Cast without an ability registry pack, etc.)
/// also fall back to Hold. This is deliberately permissive — the scorer
/// only emits heads it scored a row for, and the CPU reference does the
/// same.
pub fn scoring_outputs_to_actions(
    state: &SimState,
    scoring: &[ScoreOutput],
    out: &mut Vec<Action>,
) {
    out.clear();
    for id in state.agents_alive() {
        let slot = (id.raw() - 1) as usize;
        let so = match scoring.get(slot) {
            Some(s) => s,
            None => {
                // Scoring buffer was sized for a smaller agent_cap —
                // shouldn't happen (the backend resizes view storage /
                // scoring pool before dispatch) but guard against it.
                out.push(Action::hold(id));
                continue;
            }
        };
        out.push(build_action_from_scoring(state, id, so));
    }
}

/// Translate a single `ScoreOutput` into a concrete `Action`. Mirrors
/// `utility::build_action` but reads the target slot / head from the
/// `ScoreOutput` instead of the scorer's in-process best-tracker.
fn build_action_from_scoring(
    state: &SimState,
    agent: AgentId,
    so: &ScoreOutput,
) -> Action {
    let kind = match micro_kind_from_u32(so.chosen_action) {
        Some(k) => k,
        None => return Action::hold(agent),
    };
    match kind {
        MicroKind::Hold => Action::hold(agent),
        MicroKind::MoveToward => {
            let target = resolve_target(so.chosen_target);
            match target.and_then(|t| state.agent_pos(t)) {
                Some(pos) => Action::move_toward(agent, pos),
                None => Action::hold(agent),
            }
        }
        MicroKind::Attack => {
            match resolve_target(so.chosen_target) {
                Some(t) if state.agent_alive(t) => Action::attack(agent, t),
                _ => Action::hold(agent),
            }
        }
        MicroKind::Flee => {
            match nearest_hostile(state, agent) {
                Some(threat) => Action {
                    agent,
                    kind: ActionKind::Micro {
                        kind:   MicroKind::Flee,
                        target: MicroTarget::Agent(threat),
                    },
                },
                None => Action::hold(agent),
            }
        }
        MicroKind::Eat => Action::eat(agent),
        MicroKind::Drink => Action {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::Drink,
                target: MicroTarget::None,
            },
        },
        MicroKind::Rest => Action {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::Rest,
                target: MicroTarget::None,
            },
        },
        // The scoring kernel only produces heads in {Hold, MoveToward,
        // Flee, Attack, Eat, Drink, Rest} (see `MASK_NAMES` / `action_
        // head_to_mask_idx` — every other head maps to `MASK_SLOT_NONE`
        // and never wins). Defensive Hold for heads the scorer shouldn't
        // emit.
        _ => Action::hold(agent),
    }
}

fn resolve_target(slot: u32) -> Option<AgentId> {
    if slot == NO_TARGET {
        return None;
    }
    AgentId::new(slot + 1)
}

/// Mirror of `engine::policy::utility::micro_kind_from_u16` (which is
/// `pub(crate)`). Keeps the heads the GPU scorer actually produces — the
/// others fall through to `None` and the caller emits Hold.
fn micro_kind_from_u32(v: u32) -> Option<MicroKind> {
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

/// Mirror of `utility::nearest_hostile` — picks the nearest hostile
/// inside `config.combat.aggro_range`. Only called for Flee actions,
/// so the per-tick cost scales with the flee-agent count, not the
/// whole population; in practice the GPU cascade's `engagement_on_move`
/// rule fires first so most aggressors never pick Flee. Ties on
/// distance break on raw-id ascending for determinism (same as the
/// CPU reference).
fn nearest_hostile(state: &SimState, self_id: AgentId) -> Option<AgentId> {
    let pos = state.agent_pos(self_id)?;
    let ct = state.agent_creature_type(self_id)?;
    let spatial = state.spatial();
    let mut best: Option<(AgentId, f32)> = None;
    for other in spatial.within_radius(state, pos, state.config.combat.aggro_range) {
        if other == self_id {
            continue;
        }
        let op = match state.agent_pos(other) {
            Some(p) => p,
            None => continue,
        };
        let oc = match state.agent_creature_type(other) {
            Some(c) => c,
            None => continue,
        };
        if !ct.is_hostile_to(oc) {
            continue;
        }
        let d = pos.distance(op);
        match best {
            None => best = Some((other, d)),
            Some((_, bd)) if d < bd => best = Some((other, d)),
            Some((b, bd)) if (d - bd).abs() < f32::EPSILON && other.raw() < b.raw() => {
                best = Some((other, d));
            }
            _ => {}
        }
    }
    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::creature::CreatureType;
    use engine::state::AgentSpawn;
    use glam::Vec3;

    #[test]
    fn hold_outputs_produce_hold_actions() {
        let mut state = SimState::new(4, 0xDEAD_BEEF);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::ZERO,
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
        let scoring = vec![
            ScoreOutput::default(), // (Hold, NO_TARGET)
            ScoreOutput::default(),
            ScoreOutput::default(),
            ScoreOutput::default(),
        ];
        let mut actions = Vec::new();
        scoring_outputs_to_actions(&state, &scoring, &mut actions);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].micro_kind(), Some(MicroKind::Hold));
    }

    #[test]
    fn attack_with_live_target_produces_attack_action() {
        let mut state = SimState::new(4, 0xDEAD_BEEF);
        // slot 0 (AgentId(1)) — attacker (Wolf)
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(0.0, 0.0, 0.0),
                hp: 50.0,
                ..Default::default()
            })
            .unwrap();
        // slot 1 (AgentId(2)) — target (Human)
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(1.0, 0.0, 0.0),
                hp: 50.0,
                ..Default::default()
            })
            .unwrap();
        let mut scoring = vec![ScoreOutput::default(); 4];
        scoring[0] = ScoreOutput {
            chosen_action: 3, // Attack
            chosen_target: 1, // slot 1 = AgentId(2)
            best_score_bits: 0,
            debug: 0,
        };
        let mut actions = Vec::new();
        scoring_outputs_to_actions(&state, &scoring, &mut actions);
        assert_eq!(actions.len(), 2);
        // Attacker action
        assert_eq!(actions[0].micro_kind(), Some(MicroKind::Attack));
        if let ActionKind::Micro { target: MicroTarget::Agent(t), .. } = actions[0].kind {
            assert_eq!(t.raw(), 2);
        } else {
            panic!("expected Attack w/ Agent target");
        }
        // Target picks Hold (default scoring output).
        assert_eq!(actions[1].micro_kind(), Some(MicroKind::Hold));
    }

    #[test]
    fn movetoward_reads_target_position() {
        let mut state = SimState::new(4, 0xDEAD_BEEF);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(0.0, 0.0, 0.0),
                hp: 50.0,
                ..Default::default()
            })
            .unwrap();
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(3.0, 4.0, 0.0),
                hp: 50.0,
                ..Default::default()
            })
            .unwrap();
        let mut scoring = vec![ScoreOutput::default(); 4];
        scoring[0] = ScoreOutput {
            chosen_action: 1, // MoveToward
            chosen_target: 1, // slot 1
            best_score_bits: 0,
            debug: 0,
        };
        let mut actions = Vec::new();
        scoring_outputs_to_actions(&state, &scoring, &mut actions);
        if let ActionKind::Micro { target: MicroTarget::Position(p), .. } = actions[0].kind {
            assert_eq!(p, Vec3::new(3.0, 4.0, 0.0));
        } else {
            panic!("expected MoveToward w/ Position target");
        }
    }

    #[test]
    fn attack_with_no_target_falls_back_to_hold() {
        let mut state = SimState::new(4, 0xDEAD_BEEF);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::ZERO,
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
        let mut scoring = vec![ScoreOutput::default(); 4];
        scoring[0] = ScoreOutput {
            chosen_action: 3, // Attack
            chosen_target: NO_TARGET,
            ..Default::default()
        };
        let mut actions = Vec::new();
        scoring_outputs_to_actions(&state, &scoring, &mut actions);
        assert_eq!(actions[0].micro_kind(), Some(MicroKind::Hold));
    }
}
