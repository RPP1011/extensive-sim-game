// crates/engine/src/step.rs
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, PolicyBackend};
use crate::rng::per_agent_u32;
use crate::state::SimState;

const MOVE_SPEED_MPS: f32 = 1.0;

pub fn step<B: PolicyBackend>(state: &mut SimState, events: &mut EventRing, backend: &B) {
    let mut mask = MaskBuffer::new(state.agent_cap() as usize);
    mask.mark_hold_allowed(state);
    mask.mark_move_allowed_if_others_exist(state);

    let mut actions: Vec<Action> = Vec::with_capacity(state.agent_cap() as usize);
    backend.evaluate(state, &mask, &mut actions);

    apply_actions(state, &actions, events);
    state.tick += 1;
}

/// Fisher-Yates shuffle of action indices using a deterministic PRNG seeded by
/// `(world_seed, tick)`. This makes action-application order depend on the world
/// seed (spec §7.2 — determinism contract / first-mover-bias prevention).
fn shuffled_order(n: usize, world_seed: u64, tick: u32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let tick64 = tick as u64;
    // Sentinel agent id 1 is used as a fixed stream discriminator for the
    // per-tick shuffle — distinct from any per-agent decision stream.
    let sentinel = AgentId::new(1).unwrap();
    for i in (1..n).rev() {
        let r = per_agent_u32(world_seed, sentinel, tick64 * 65536 + i as u64, b"shuffle");
        let j = (r as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

fn apply_actions(state: &mut SimState, actions: &[Action], events: &mut EventRing) {
    let order = shuffled_order(actions.len(), state.seed, state.tick);
    for &idx in &order {
        let action = &actions[idx];
        match action.micro_kind {
            MicroKind::Hold => {}
            MicroKind::MoveToward => {
                if let Some(target) = nearest_other(state, action.agent) {
                    let from = state.agent_pos(action.agent).unwrap();
                    let target_pos = state.agent_pos(target).unwrap();
                    let delta = target_pos - from;
                    if delta.length_squared() > 0.0 {
                        let to = from + delta.normalize() * MOVE_SPEED_MPS;
                        state.set_agent_pos(action.agent, to);
                        events.push(Event::AgentMoved {
                            agent_id: action.agent, from, to, tick: state.tick,
                        });
                    }
                }
            }
            MicroKind::Attack | MicroKind::Eat => {
                // Not implemented in MVP.
            }
        }
    }
}

fn nearest_other(state: &SimState, self_id: AgentId) -> Option<AgentId> {
    let self_pos = state.agent_pos(self_id)?;
    state.agents_alive()
        .filter(|id| *id != self_id)
        .min_by(|a, b| {
            let da = (state.agent_pos(*a).unwrap() - self_pos).length_squared();
            let db = (state.agent_pos(*b).unwrap() - self_pos).length_squared();
            da.total_cmp(&db)
        })
}
