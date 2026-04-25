//! End-to-end: 100 agents × 1000 ticks with a policy that emits a rich mix of
//! Hold / MoveToward / Flee / Attack / Eat / Drink / Rest / Communicate /
//! Announce. Same seed twice → identical replayable hash. Different seeds →
//! different hashes. Release build runs in ≤ 2 s.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, AnnounceAudience, MacroAction,
                     MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

/// A policy that emits a rotating mix of the universal action primitives.
/// Distinct agents pick different kinds based on id % N so the event stream
/// varies across the population — a silent uniform policy would make determinism
/// trivial.
struct MixedPolicy;

impl PolicyBackend for MixedPolicy {
    fn evaluate(&self, state: &SimState, _mask: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        let tick = state.tick;
        for id in state.agents_alive() {
            let pick = id.raw() % 7;
            let hunger = state.agent_hunger(id).unwrap_or(1.0);

            let action = if hunger < 0.3 {
                // Starving agents always eat.
                Action { agent: id, kind: ActionKind::Micro {
                    kind: MicroKind::Eat, target: MicroTarget::None,
                }}
            } else if pick == 0 && tick % 50 == 7 {
                // Every 50 ticks, this pick-class announces.
                Action { agent: id, kind: ActionKind::Macro(MacroAction::Announce {
                    speaker: id,
                    audience: AnnounceAudience::Area(
                        state.agent_pos(id).unwrap_or(Vec3::ZERO),
                        10.0,
                    ),
                    fact_payload: tick as u64,
                })}
            } else if pick == 1 {
                Action { agent: id, kind: ActionKind::Micro {
                    kind: MicroKind::Drink, target: MicroTarget::None,
                }}
            } else if pick == 2 {
                Action { agent: id, kind: ActionKind::Micro {
                    kind: MicroKind::Rest, target: MicroTarget::None,
                }}
            } else if pick == 3 {
                // Communicate with AgentId::new(1) if alive.
                if let Some(target) = engine::ids::AgentId::new(1) {
                    if state.agent_alive(target) && target != id {
                        Action { agent: id, kind: ActionKind::Micro {
                            kind: MicroKind::Communicate,
                            target: MicroTarget::Agent(target),
                        }}
                    } else {
                        Action::hold(id)
                    }
                } else { Action::hold(id) }
            } else if pick == 4 {
                Action { agent: id, kind: ActionKind::Micro {
                    kind: MicroKind::MoveToward,
                    target: MicroTarget::Position(Vec3::new(0.0, 0.0, 10.0)),
                }}
            } else if pick == 5 {
                // Attack nearest if any other agent is alive.
                let mut target: Option<engine::ids::AgentId> = None;
                let self_pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
                let mut best_d2 = f32::INFINITY;
                for other in state.agents_alive() {
                    if other == id { continue; }
                    if let Some(op) = state.agent_pos(other) {
                        let d2 = (op - self_pos).length_squared();
                        if d2 < best_d2 { best_d2 = d2; target = Some(other); }
                    }
                }
                match target {
                    Some(t) => Action { agent: id, kind: ActionKind::Micro {
                        kind: MicroKind::Attack, target: MicroTarget::Agent(t),
                    }},
                    None => Action::hold(id),
                }
            } else {
                Action::hold(id)
            };
            out.push(action);
        }
    }
}

fn run(seed: u64) -> [u8; 32] {
    let mut state = SimState::new(110, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1_000_000);
    let cascade = CascadeRegistry::<Event>::new();
    for i in 0..100u32 {
        let angle = (i as f32 / 100.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    for _ in 0..1000 {
        step(&mut state, &mut scratch, &mut events, &MixedPolicy, &cascade);
    }
    events.replayable_sha256()
}

#[test]
fn same_seed_same_hash() {
    let h1 = run(42);
    let h2 = run(42);
    assert_eq!(h1, h2, "deterministic replay");
}

#[test]
fn different_seed_different_hash() {
    let h1 = run(42);
    let h2 = run(43);
    assert_ne!(h1, h2, "different seeds must diverge");
}

#[test]
fn mixed_run_under_two_seconds_release() {
    let t0 = std::time::Instant::now();
    let _ = run(42);
    let elapsed = t0.elapsed();
    eprintln!("mixed-action 100×1000: {:?}", elapsed);
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed.as_secs_f64() <= 2.0,
        "mixed-action run took {:?}, over 2s release budget", elapsed
    );
}
