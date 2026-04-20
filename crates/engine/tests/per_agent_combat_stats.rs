//! Audit fix MEDIUM #10 — the kernel's Attack branch and
//! `OpportunityAttackHandler` must honour the per-agent `hot_attack_damage`
//! and `hot_attack_range` SoA fields. Setting them via `set_agent_attack_*`
//! changes observable next-attack behaviour.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::MicroKind;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct ForceAttack {
    attacker: AgentId,
    target:   AgentId,
}
impl PolicyBackend for ForceAttack {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.attacker,
            kind: ActionKind::Micro {
                kind: MicroKind::Attack,
                target: MicroTarget::Agent(self.target),
            },
        });
        for id in state.agents_alive() {
            if id != self.attacker { out.push(Action::hold(id)); }
        }
    }
}

#[test]
fn set_agent_attack_damage_is_honoured_by_next_attack() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();

    let attacker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    let target = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    // Raise attacker damage to 25.
    state.set_agent_attack_damage(attacker, 25.0);

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &ForceAttack { attacker, target },
        &cascade,
    );

    let dmg: f32 = events
        .iter()
        .find_map(|e| match e {
            Event::AgentAttacked { attacker: a, target: t, damage, .. }
                if *a == attacker && *t == target => Some(*damage),
            _ => None,
        })
        .expect("expected one AgentAttacked event");
    assert_eq!(dmg, 25.0,
        "set_agent_attack_damage(25.0) must translate into AgentAttacked.damage=25.0");
    // Target HP dropped by exactly 25.
    assert_eq!(state.agent_hp(target), Some(75.0));
}

#[test]
fn set_agent_attack_range_is_honoured_by_next_attack() {
    // Default ATTACK_RANGE = 2m. Place target at 2.5m and bump range to 3m.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();

    let attacker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    let target = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(2.5, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    // Without the fix, the kernel uses the 2.0 constant → 2.5m target
    // misses. With MEDIUM #10 applied, the per-agent range wins.
    state.set_agent_attack_range(attacker, 3.0);

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &ForceAttack { attacker, target },
        &cascade,
    );

    // The attack lands — look for AgentAttacked on the target.
    let attacked = events.iter().any(|e| matches!(e,
        Event::AgentAttacked { attacker: a, target: t, .. }
            if *a == attacker && *t == target));
    assert!(attacked,
        "per-agent attack range 3m must let the 2.5m attack land");
}
