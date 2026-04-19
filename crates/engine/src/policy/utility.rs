// crates/engine/src/policy/utility.rs
use super::{Action, PolicyBackend};
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
                if !mask.micro_kind[row_start + i] { continue; }
                let score = utility_score(kind, hp, max_hp);
                if score > best.1 { best = (kind, score); }
            }
            out.push(Action { agent: id, micro_kind: best.0, target: None });
        }
    }
}

fn utility_score(kind: MicroKind, hp: f32, max_hp: f32) -> f32 {
    match kind {
        MicroKind::Hold       => 0.1,
        MicroKind::MoveToward => 0.3,                      // Prefer moving when mask allows
        MicroKind::Attack     => if hp > max_hp * 0.5 { 0.5 } else { 0.0 },
        MicroKind::Eat        => if hp < max_hp * 0.3 { 0.8 } else { 0.0 },
    }
}
