// crates/engine/src/step.rs
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
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
        match action.kind {
            ActionKind::Micro { kind: MicroKind::Hold, .. } => {}
            ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(target_pos),
            } => {
                let from = state.agent_pos(action.agent).unwrap();
                let delta = target_pos - from;
                if delta.length_squared() > 0.0 {
                    let to = from + delta.normalize() * MOVE_SPEED_MPS;
                    state.set_agent_pos(action.agent, to);
                    events.push(Event::AgentMoved {
                        agent_id: action.agent, from, to, tick: state.tick,
                    });
                }
            }
            ActionKind::Micro { kind: MicroKind::Attack, .. }
            | ActionKind::Micro { kind: MicroKind::Eat, .. } => {
                // Not implemented in MVP.
            }
            ActionKind::Micro { .. } => {
                // Other MicroKinds (Flee, UseItem, Ask, …) land in Tasks 9–12.
            }
            ActionKind::Macro(_) => {
                // Macro dispatch lands in Tasks 13–15.
            }
        }
    }
}

