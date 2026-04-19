// crates/engine/src/step.rs
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, PolicyBackend};
use crate::rng::per_agent_u32;
use crate::state::SimState;

const MOVE_SPEED_MPS: f32 = 1.0;

/// Per-tick scratch buffers hoisted out of `step` so a steady-state tick loop
/// allocates zero bytes. Caller constructs once (capacity = `state.agent_cap()`),
/// reuses across ticks. Buffers are reset/cleared at the top of each `step`.
pub struct SimScratch {
    pub mask:        MaskBuffer,
    pub actions:     Vec<Action>,
    pub shuffle_idx: Vec<u32>,
}

impl SimScratch {
    pub fn new(n_agents: usize) -> Self {
        Self {
            mask:        MaskBuffer::new(n_agents),
            actions:     Vec::with_capacity(n_agents),
            shuffle_idx: Vec::with_capacity(n_agents),
        }
    }
}

pub fn step<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing,
    backend: &B,
) {
    scratch.mask.reset();
    scratch.mask.mark_hold_allowed(state);
    scratch.mask.mark_move_allowed_if_others_exist(state);
    scratch.actions.clear();
    backend.evaluate(state, &scratch.mask, &mut scratch.actions);

    apply_actions(state, &scratch.actions, events, &mut scratch.shuffle_idx);
    state.tick += 1;
}

/// Fisher-Yates shuffle of action indices using a deterministic PRNG seeded by
/// `(world_seed, tick)`. This makes action-application order depend on the world
/// seed (spec §7.2 — determinism contract / first-mover-bias prevention).
///
/// Writes into the caller-owned `order` buffer (cleared + extended in place) so
/// the per-tick order vec does not re-allocate once `SimScratch` is warm.
fn shuffle_order_into(order: &mut Vec<u32>, n: usize, world_seed: u64, tick: u32) {
    order.clear();
    order.extend(0..n as u32);
    let tick64 = tick as u64;
    // Sentinel agent id 1 is used as a fixed stream discriminator for the
    // per-tick shuffle — distinct from any per-agent decision stream.
    let sentinel = AgentId::new(1).unwrap();
    for i in (1..n).rev() {
        let r = per_agent_u32(world_seed, sentinel, tick64 * 65536 + i as u64, b"shuffle");
        let j = (r as usize) % (i + 1);
        order.swap(i, j);
    }
}

fn apply_actions(
    state:   &mut SimState,
    actions: &[Action],
    events:  &mut EventRing,
    order:   &mut Vec<u32>,
) {
    shuffle_order_into(order, actions.len(), state.seed, state.tick);
    for &idx in order.iter() {
        let action = &actions[idx as usize];
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
            // New variants from the full 18-kind set. Dispatch lands in Tasks 9–12.
            _ => {}
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
