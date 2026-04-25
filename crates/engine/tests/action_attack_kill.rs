use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn mask_attack_bit_pins_attack_range_at_2m_boundary() {
    // Pin both sides of the 2.0m attack range with a pair of 1.99m/2.01m
    // fixtures. As of compiler milestone 4 the attack-mask predicate also
    // gates on hostility (`assets/sim/masks.sim` calls `is_hostile`), so
    // the fixtures use a hostile Human-vs-Wolf pair.
    fn bit_set_for(state: &SimState, attacker: AgentId) -> bool {
        let mut mask = MaskBuffer::new(state.agent_cap() as usize);
        let mut target_mask =
            engine::mask::TargetMask::new(state.agent_cap() as usize);
        mask.mark_attack_allowed_from_candidates(state, &mut target_mask);
        let slot = (attacker.raw() - 1) as usize;
        let offset = slot * MicroKind::ALL.len() + MicroKind::Attack as usize;
        mask.micro_kind[offset]
    }

    // At 1.99m — mask bit must be set.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.99, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert!(bit_set_for(&state, attacker), "attack mask bit must be set at 1.99m");

    // At 2.01m — mask bit must be clear.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(2.01, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert!(!bit_set_for(&state, attacker), "attack mask bit must be clear at 2.01m");
}

#[test]
fn mask_attack_bit_respects_hostility_gate() {
    // Compiler milestone 4 folded hostility into the attack-mask predicate.
    // Two Humans at melee range should no longer have the Attack bit set;
    // Human + Wolf at the same range should.
    fn bit_set_for(state: &SimState, attacker: AgentId) -> bool {
        let mut mask = MaskBuffer::new(state.agent_cap() as usize);
        let mut target_mask =
            engine::mask::TargetMask::new(state.agent_cap() as usize);
        mask.mark_attack_allowed_from_candidates(state, &mut target_mask);
        let slot = (attacker.raw() - 1) as usize;
        let offset = slot * MicroKind::ALL.len() + MicroKind::Attack as usize;
        mask.micro_kind[offset]
    }

    // Two Humans at 1m — not hostile, bit must be clear.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert!(!bit_set_for(&state, attacker), "same-species pair is not hostile");

    // Human vs Wolf at 1m — hostile per `CreatureType::is_hostile_to`.
    let mut state = SimState::new(4, 42);
    let _victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert!(bit_set_for(&state, attacker), "Wolf-vs-Human pair is hostile");
}

struct AttackFixed(AgentId);
impl PolicyBackend for AttackFixed {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
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
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 1m away — within 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
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
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.99, 0.0, 0.0), hp: 100.0,
        ..Default::default()
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
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(2.01, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(100.0), "miss at 2.01m");
    assert!(!events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn attack_beyond_range_is_a_no_op() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 10m away — beyond 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(10.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(100.0));
    assert!(!events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn hp_hits_zero_kills_agent_and_emits_died_event() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 10.0,  // one-shot,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
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
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 5.0,
        ..Default::default()
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    // First tick kills the victim.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    let events_after_kill = events.len();
    // Second tick: attacker tries again on dead victim — should produce nothing new.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(events.len(), events_after_kill, "no new events against dead target");
}
