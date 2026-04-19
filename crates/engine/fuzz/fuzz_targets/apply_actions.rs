// crates/engine/fuzz/fuzz_targets/apply_actions.rs
#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FuzzSpawn {
    x: i16,
    y: i16,
    z: u8,
    hp: u8,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    seed:   u64,
    spawns: Vec<FuzzSpawn>,
    ticks:  u8,
}

/// A policy backend that emits an arbitrary-but-fixed action per agent,
/// drawn from `data` (the fuzzer-provided byte stream).
struct FuzzPolicy {
    bytes: Vec<u8>,
}

impl PolicyBackend for FuzzPolicy {
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>) {
        let n_kinds = MicroKind::ALL.len();
        for (i, id) in state.agents_alive().enumerate() {
            let b = self.bytes.get(i).copied().unwrap_or(0);
            let kind = MicroKind::ALL[(b as usize) % n_kinds];
            let slot = (id.raw() - 1) as usize;
            let bit  = slot * n_kinds + kind as usize;
            // Respect the mask: if bit is false, emit Hold (always true).
            let kind = if mask.micro_kind.get(bit).copied().unwrap_or(false) {
                kind
            } else {
                MicroKind::Hold
            };
            let target = match kind {
                MicroKind::MoveToward => MicroTarget::Position(Vec3::new(
                    (b as f32) * 0.1, 0.0, 10.0,
                )),
                MicroKind::Flee | MicroKind::Attack => {
                    // Target the previous alive agent (arbitrary).
                    let tgt = AgentId::new(((slot as u32) % state.agent_cap()) + 1).unwrap();
                    MicroTarget::Agent(tgt)
                }
                MicroKind::Cast => MicroTarget::AbilityIdx(b),
                MicroKind::UseItem => MicroTarget::ItemSlot(b),
                _ => MicroTarget::None,
            };
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro { kind, target },
            });
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(input) = FuzzInput::arbitrary(&mut u) else { return; };

    // Bound the work.
    let n_spawns = input.spawns.len().min(8);
    let cap = (n_spawns as u32).saturating_add(4).max(1);
    let ticks = (input.ticks as u32).min(20);
    if n_spawns == 0 || ticks == 0 { return; }

    let mut state = SimState::new(cap, input.seed);
    for s in &input.spawns[..n_spawns] {
        let pos = Vec3::new(s.x as f32 * 0.5, s.y as f32 * 0.5, (s.z as f32) % 100.0);
        let hp  = (s.hp as f32).max(1.0);
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human, pos, hp,
        });
    }

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let policy = FuzzPolicy { bytes: data.to_vec() };

    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &policy, &cascade);

        // Invariant #1: pool is self-consistent after every tick.
        assert!(state.pool_is_consistent(),
            "pool invariant violated after fuzz tick");

        // Invariant #2: no alive agent has negative HP.
        for id in state.agents_alive() {
            let hp = state.agent_hp(id).unwrap_or(0.0);
            assert!(hp >= 0.0, "alive agent {:?} has negative HP {}", id, hp);
        }

        // Invariant #3: needs are in [0.0, 1.0].
        for id in state.agents_alive() {
            let h = state.agent_hunger(id).unwrap_or(0.5);
            let t = state.agent_thirst(id).unwrap_or(0.5);
            let r = state.agent_rest_timer(id).unwrap_or(0.5);
            assert!((0.0..=1.0).contains(&h), "hunger out of range: {}", h);
            assert!((0.0..=1.0).contains(&t), "thirst out of range: {}", t);
            assert!((0.0..=1.0).contains(&r), "rest out of range: {}", r);
        }
    }

    // Invariant #4: replayable_sha256 is computable (no panic on hasher).
    let _hash = events.replayable_sha256();
});
