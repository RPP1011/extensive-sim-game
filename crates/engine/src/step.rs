// crates/engine/src/step.rs
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, PolicyBackend};
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

fn apply_actions(state: &mut SimState, actions: &[Action], events: &mut EventRing) {
    for action in actions {
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
