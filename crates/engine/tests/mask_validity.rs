use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::mask::MicroKind;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn all_chosen_actions_pass_their_mask() {
    let mut state = SimState::new(50, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100_000);
    let cascade = CascadeRegistry::new();
    for i in 0..20 {
        let angle = (i as f32 / 20.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(30.0 * angle.cos(), 30.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }

    for _ in 0..500 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        // After step, scratch.actions holds the tick's actions and
        // scratch.mask holds the mask that was used to produce them.
        for action in &scratch.actions {
            let slot = (action.agent.raw() - 1) as usize;
            let micro = action
                .micro_kind()
                .expect("UtilityBackend never emits Macro actions yet");
            let offset = slot * MicroKind::ALL.len() + micro as usize;
            assert!(
                scratch.mask.micro_kind[offset],
                "tick={} action={:?} violated mask (slot={}, offset={})",
                state.tick, action, slot, offset
            );
        }
    }
}
