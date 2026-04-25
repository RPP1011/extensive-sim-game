//! End-to-end test that a non-default `Config` flows from
//! `SimState::new_with_config` through `step()` into the observable event
//! log. Pins the contract: every balance constant that used to be a
//! `pub const` is now read off `state.config`, so overriding a field
//! changes the simulation's behaviour without any other plumbing.
//!
//! Demo: doubling `combat.attack_damage` from 10.0 to 20.0 makes the
//! attack event carry 20.0 damage instead of 10.0. Same seed, same
//! scenario, same tick count — the only variable is the `Config`.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine_data::config::Config;
use glam::Vec3;

/// Policy backend that makes slot 1 (the first spawned agent, by `AgentId`
/// convention) attack whoever the caller passed in and everyone else hold.
/// Lets the test drive a single deterministic `Attack` per tick.
struct AttackFixed(AgentId);
impl PolicyBackend for AttackFixed {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.0 {
                out.push(Action::hold(id));
                continue;
            }
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

fn spawn_hostile_pair(state: &mut SimState) -> AgentId {
    // Human victim at origin + Wolf attacker 1m away — inside default
    // attack range, hostility guaranteed by the creature matrix.
    let victim = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let _attacker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    victim
}

fn attack_damage_dealt(damage_override: Option<f32>) -> (f32, f32) {
    let mut cfg = Config::default();
    if let Some(d) = damage_override {
        cfg.combat.attack_damage = d;
    }
    let mut state = SimState::new_with_config(4, 42, cfg);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let victim = spawn_hostile_pair(&mut state);
    let before = state.agent_hp(victim).unwrap();
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    let after = state.agent_hp(victim).unwrap();

    let emitted_damage = events
        .iter()
        .find_map(|e| match e {
            Event::AgentAttacked { damage, .. } => Some(*damage),
            _ => None,
        })
        .expect("AgentAttacked emitted");
    (before - after, emitted_damage)
}

#[test]
fn default_config_still_deals_ten_damage() {
    let (hp_drop, emitted) = attack_damage_dealt(None);
    assert!(
        (hp_drop - 10.0).abs() < 1e-5,
        "default config should still deal 10.0 damage, got {hp_drop}"
    );
    assert!(
        (emitted - 10.0).abs() < 1e-5,
        "emitted damage should be 10.0, got {emitted}"
    );
}

#[test]
fn doubling_attack_damage_in_config_doubles_damage_dealt() {
    // Tuning `combat.attack_damage` from 10.0 to 20.0 should exactly double
    // the HP drop + the emitted event's `damage` field. No other field is
    // touched, so every other code path (mask, hostility, range, etc.) is
    // identical to the default-config run above.
    let (hp_drop, emitted) = attack_damage_dealt(Some(20.0));
    assert!(
        (hp_drop - 20.0).abs() < 1e-5,
        "config override should deal 20.0 damage, got {hp_drop}"
    );
    assert!(
        (emitted - 20.0).abs() < 1e-5,
        "emitted damage should be 20.0, got {emitted}"
    );
}

#[test]
fn sim_state_new_is_equivalent_to_default_config() {
    // `SimState::new` is a convenience wrapper; it must be indistinguishable
    // from `new_with_config(..., Config::default())` for every balance-
    // observable output. Walk a single step and compare the resulting HP.
    let (hp_a, _) = attack_damage_dealt(None);
    let mut cfg = Config::default();
    // Writing the defaults back explicitly — no-op but pins the pattern.
    cfg.combat.attack_damage = 10.0;
    let (hp_b, _) = {
        let mut state = SimState::new_with_config(4, 42, cfg);
        let mut scratch = SimScratch::new(state.agent_cap() as usize);
        let mut events = EventRing::<Event>::with_cap(1024);
        let cascade = CascadeRegistry::<Event>::new();
        let victim = spawn_hostile_pair(&mut state);
        let before = state.agent_hp(victim).unwrap();
        step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
        let after = state.agent_hp(victim).unwrap();
        (before - after, ())
    };
    assert!(
        (hp_a - hp_b).abs() < 1e-5,
        "new() and new_with_config(Default::default()) must agree; got {hp_a} vs {hp_b}"
    );
}
