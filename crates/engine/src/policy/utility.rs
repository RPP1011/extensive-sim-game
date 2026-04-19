// crates/engine/src/policy/utility.rs
use super::{Action, PolicyBackend};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::state::SimState;

pub struct UtilityBackend;

impl PolicyBackend for UtilityBackend {
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let row_start = slot * MicroKind::ALL.len();
            let hp = state.agent_hp(id).unwrap_or(0.0);
            let max_hp = state.agent_max_hp(id).unwrap_or(1.0);
            let mut best = (MicroKind::Hold, f32::MIN);
            for (i, &kind) in MicroKind::ALL.iter().enumerate() {
                if !mask.micro_kind[row_start + i] {
                    continue;
                }
                let score = utility_score(kind, hp, max_hp);
                if score > best.1 {
                    best = (kind, score);
                }
            }
            out.push(match best.0 {
                MicroKind::Hold => Action::hold(id),
                MicroKind::MoveToward => {
                    // Resolve the nearest-other target position at emission time —
                    // backend is responsible for picking "where to move". If no
                    // other agent exists, fall back to Hold (mask should have
                    // already excluded MoveToward in that case, but be defensive).
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
                // New variants (Tasks 9–12 will score them explicitly); fall
                // back to Hold for MVP since utility_score returns 0.0 for them
                // and Hold scores 0.1 — we only reach this arm if a variant
                // is both mask-allowed and somehow outscores Hold without a
                // dedicated constructor.
                _ => Action::hold(id),
            });
        }
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

fn utility_score(kind: MicroKind, hp: f32, max_hp: f32) -> f32 {
    match kind {
        MicroKind::Hold       => 0.1,
        MicroKind::MoveToward => 0.3,                      // Prefer moving when mask allows
        MicroKind::Attack     => if hp > max_hp * 0.5 { 0.5 } else { 0.0 },
        MicroKind::Eat        => if hp < max_hp * 0.3 { 0.8 } else { 0.0 },
        // New variants (Tasks 9–12 will score them explicitly); neutral for MVP.
        _ => 0.0,
    }
}
