//! Plan 3, Task 12 — end-to-end acceptance.
//!
//! Drives the engine through 100 ticks, snapshots state to disk,
//! reloads, runs another 100 ticks, and asserts the final state
//! matches a continuous-run reference (200 ticks straight, same seed +
//! same agent layout).
//!
//! Notes:
//! - Final **state** equality is the contract (positions / hp / hunger /
//!   alive). The replay event hash is NOT comparable across the
//!   save→load boundary because Plan 3 v1 doesn't snapshot ring entries
//!   (see `snapshot/format.rs#coverage-gaps`).
//! - The seed is threaded through `SimState::new`; the world is
//!   otherwise deterministic via PCG.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::policy::UtilityBackend;
use engine::snapshot::{load_snapshot, save_snapshot};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

const SEED: u64 = 42;
const N_AGENTS: usize = 20;
const AGENT_CAP: u32 = 32;
const RING_CAP: usize = 65_536;

fn spawn_layout(state: &mut SimState) {
    for i in 0..N_AGENTS {
        let angle = (i as f32 / N_AGENTS as f32) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(30.0 * angle.cos(), 30.0 * angle.sin(), 10.0),
            hp: 100.0,
            max_hp: 100.0,
        });
    }
}

fn run_straight(ticks: u32) -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(RING_CAP);
    let cascade = CascadeRegistry::new();
    spawn_layout(&mut state);
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    state
}

fn run_save_reload(ticks_a: u32, ticks_b: u32) -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(RING_CAP);
    let cascade = CascadeRegistry::new();
    spawn_layout(&mut state);

    for _ in 0..ticks_a {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }

    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("engine_plan3_acc_{}_{}.bin", pid, nonce));
    save_snapshot(&state, &events, &path).unwrap();

    let (mut state2, mut events2) = load_snapshot(&path).unwrap();
    let mut scratch2 = SimScratch::new(state2.agent_cap() as usize);

    for _ in 0..ticks_b {
        step(&mut state2, &mut scratch2, &mut events2, &UtilityBackend, &cascade);
    }

    std::fs::remove_file(&path).ok();
    state2
}

#[test]
fn save_reload_yields_same_final_state() {
    let straight = run_straight(200);
    let interrupted = run_save_reload(100, 100);

    assert_eq!(straight.tick, 200);
    assert_eq!(interrupted.tick, 200);
    assert_eq!(
        straight.agents_alive().count(),
        interrupted.agents_alive().count(),
        "alive count diverged",
    );

    // Per-agent state equality across all spawned slots.
    for slot in 0..N_AGENTS {
        let id = AgentId::new((slot + 1) as u32).unwrap();
        assert_eq!(
            straight.agent_alive(id),
            interrupted.agent_alive(id),
            "alive[{}] diverged",
            slot,
        );
        if !straight.agent_alive(id) {
            continue;
        }
        let p1 = straight.agent_pos(id).unwrap();
        let p2 = interrupted.agent_pos(id).unwrap();
        assert_eq!(p1, p2, "pos[{}] diverged: {:?} vs {:?}", slot, p1, p2);

        let h1 = straight.agent_hp(id).unwrap();
        let h2 = interrupted.agent_hp(id).unwrap();
        assert_eq!(h1, h2, "hp[{}] diverged: {} vs {}", slot, h1, h2);

        assert_eq!(
            straight.agent_hunger(id),
            interrupted.agent_hunger(id),
            "hunger[{}] diverged",
            slot,
        );
        assert_eq!(
            straight.agent_thirst(id),
            interrupted.agent_thirst(id),
            "thirst[{}] diverged",
            slot,
        );
    }
}
