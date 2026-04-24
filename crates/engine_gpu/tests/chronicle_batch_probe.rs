//! Diagnostic probe kept post-fix as a regression canary.
//!
//! Dumps the internal batch-path state after a single `step_batch(1)`
//! on the Human+Wolf fight fixture and asserts the invariants that
//! were violated pre-fix of the `ResidentPhysicsCfg` uniform
//! `queue.write_buffer` collapse:
//!
//!   - `apply_event_ring.tail` > 0 (apply_actions ran)
//!   - `num_events_buf[0]` == `apply_event_ring.tail` (seed kernel ran)
//!   - `indirect_args[0]` has `x > 0` (seed computed workgroup count)
//!   - `chronicle_ring.tail` > 0 (physics iter 0 actually fired rules)
//!
//! The last invariant is the one the pre-fix implementation violated:
//! physics iter N read `num_events_buf[max_iters-1]` (=0 on this
//! fixture) because every iteration's uniform upload collapsed onto the
//! same byte range via `queue.write_buffer`, leaving iter 0 with
//! stale `read_slot`.

#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::event_ring::EventKindTag;
use engine_gpu::GpuBackend;
use glam::Vec3;

fn build_fight_state() -> SimState {
    let mut state = SimState::new(8, 0xC1_F1_A8);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf spawn");
    state
}

#[test]
fn probe_step_batch_1_internal_state() {
    let mut gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("probe_step_batch_1_internal_state: GPU init failed ({e})");
            return;
        }
    };
    let mut state = build_fight_state();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(4096);
    let cascade = CascadeRegistry::with_engine_builtins();

    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 1);

    let (apply_tail, apply_records) = gpu.apply_event_ring_debug_for_test(8);
    let num_events = gpu.num_events_buf_for_test(8);
    let indirect = gpu.indirect_args_for_test(8);
    let (chron_tail, chron_records) = gpu.chronicle_ring_debug_for_test(8);

    eprintln!("apply_ring.tail = {apply_tail}");
    for (i, r) in apply_records.iter().take(apply_tail as usize).enumerate() {
        let kind = EventKindTag::from_u32(r.kind);
        eprintln!("  apply[{i}] kind={} ({kind:?}) tick={}", r.kind, r.tick);
    }
    eprintln!("num_events_buf = {num_events:?}");
    eprintln!(
        "indirect_args = {:?}",
        indirect.iter().map(|a| (a.x, a.y, a.z)).collect::<Vec<_>>()
    );
    eprintln!("chronicle_ring.tail = {chron_tail}");
    for (i, r) in chron_records.iter().take(chron_tail as usize).enumerate() {
        eprintln!(
            "  chron[{i}] template_id={} agent={} target={} tick={}",
            r.template_id, r.agent, r.target, r.tick
        );
    }

    // Invariants. The chronicle_tail > 0 assertion is the load-bearing
    // one — it would have caught the ResidentPhysicsCfg uniform-collapse
    // bug on the day Phase 2 landed.
    assert!(apply_tail > 0, "apply_actions/movement wrote nothing");
    assert_eq!(
        num_events[0], apply_tail,
        "seed kernel should have written apply_tail into num_events[0]"
    );
    assert!(indirect[0].x > 0, "seed kernel wrote zero workgroups");
    assert!(
        chron_tail > 0,
        "physics iter 0 ran but didn't emit chronicle records — \
         likely a regression of the ResidentPhysicsCfg uniform-collapse bug"
    );
}
