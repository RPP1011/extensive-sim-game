use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn mask_attack_bit_pins_attack_range_at_2m_boundary() {
    // The mask-side `ATTACK_RANGE_FOR_MASK` constant in mask.rs is separate
    // from the kernel-side `ATTACK_RANGE` in step.rs. They MUST agree — or
    // the policy evaluates with a bit set for a distance the kernel will
    // refuse to resolve (and vice versa). Pin both sides at 2.0m with a
    // pair of 1.99m/2.01m fixtures.
    fn bit_set_for(state: &SimState, attacker: AgentId) -> bool {
        let mut mask = MaskBuffer::new(state.agent_cap() as usize);
        mask.mark_attack_allowed_if_target_in_range(state);
        let slot = (attacker.raw() - 1) as usize;
        let offset = slot * MicroKind::ALL.len() + MicroKind::Attack as usize;
        mask.micro_kind[offset]
    }

    // At 1.99m — mask bit must be set.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.99, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    assert!(bit_set_for(&state, attacker), "attack mask bit must be set at 1.99m");

    // At 2.01m — mask bit must be clear.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(2.01, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    assert!(!bit_set_for(&state, attacker), "attack mask bit must be clear at 2.01m");
}

struct AttackFixed(AgentId);
impl PolicyBackend for AttackFixed {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.0 { out.push(Action::hold(id)); continue; }
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind: MicroKind::Attack,
                    target: MicroTarget::Agent(self.0),
                },
            });
        }
    }
}

#[test]
fn attack_reduces_hp_within_range() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 1m away — within 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(90.0));

    // Assert damage payload on the event, not just the variant.
    let damage = events.iter().find_map(|e| match e {
        Event::AgentAttacked { damage, .. } => Some(*damage),
        _ => None,
    }).expect("AgentAttacked emitted");
    assert!((damage - 10.0).abs() < 1e-6, "damage should be 10.0, got {}", damage);
}

#[test]
fn attack_at_1_99m_hits_pinning_attack_range_upper_bound() {
    // Boundary: attacker at 1.99m — just inside ATTACK_RANGE=2.0 with `<=`.
    // If an impl rounded the range down to 1.5 this would fail.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.99, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(90.0), "hit at 1.99m");

    let damage = events.iter().find_map(|e| match e {
        Event::AgentAttacked { damage, .. } => Some(*damage),
        _ => None,
    }).expect("AgentAttacked emitted at 1.99m");
    assert!((damage - 10.0).abs() < 1e-6, "damage should be 10.0, got {}", damage);
}

#[test]
fn attack_at_2_01m_misses_pinning_attack_range_upper_bound() {
    // Boundary: attacker at 2.01m — just outside ATTACK_RANGE=2.0.
    // If an impl bumped the range up to 5.0 this would fail.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(2.01, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(100.0), "miss at 2.01m");
    assert!(!events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn attack_beyond_range_is_a_no_op() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 10m away — beyond 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(10.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(100.0));
    assert!(!events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn hp_hits_zero_kills_agent_and_emits_died_event() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 10.0,  // one-shot
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(0.0));
    assert!(!state.agent_alive(victim), "hp=0 agent is dead");
    assert!(events.iter().any(|e| matches!(e, Event::AgentDied { agent_id, .. } if *agent_id == victim)));

    // Damage payload must still be 10.0 even when the target was one-shot.
    let damage = events.iter().find_map(|e| match e {
        Event::AgentAttacked { damage, .. } => Some(*damage),
        _ => None,
    }).expect("AgentAttacked emitted");
    assert!((damage - 10.0).abs() < 1e-6, "damage should be 10.0, got {}", damage);
}

#[test]
fn attack_dead_target_is_no_op() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 5.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    // First tick kills the victim.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    let events_after_kill = events.len();
    // Second tick: attacker tries again on dead victim — should produce nothing new.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(events.len(), events_after_kill, "no new events against dead target");
}
