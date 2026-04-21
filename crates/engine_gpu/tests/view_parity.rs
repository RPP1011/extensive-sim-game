//! Phase 4 parity harness — per-view CPU/GPU fold parity.
//!
//! For each materialized view shape (slot_map, pair_map, pair_map
//! @decay), generate a deterministic event stream, fold it on both
//! sides, and assert byte-equal storage + byte-equal decayed reads.
//!
//! This test doesn't drive a full `GpuBackend::step` — the view
//! storage isn't yet wired into the tick loop (that's the follow-up
//! integration task). Instead it uses the GPU view_storage API
//! directly: `ViewStorage::new` for setup, `fold_pair_events` /
//! `fold_slot_events` for dispatch, and the readback family for the
//! assert.
//!
//! CPU reference comes from the compiler-emitted Rust views
//! (`engine::generated::views::*`) — the same types the engine uses in
//! production. Feeding identical events through both sides ensures
//! the WGSL fold matches the Rust fold bit-for-bit (modulo the
//! HashMap-sparse vs dense-array layout difference, which the test
//! reconciles pair-by-pair).
//!
//! Run with `cargo test -p engine_gpu --features gpu`.

#![cfg(feature = "gpu")]

use engine::event::Event;
use engine::generated::views::{EngagedWith, KinFear, MyEnemies, PackFocus, RallyBoost, ThreatLevel};
use engine::ids::AgentId;
use engine_gpu::view_storage::{
    build_all_specs, emit::ViewShape, DecayCellCpu, FoldInputPair, FoldInputSlot, ViewStorage,
};
use glam::Vec3;

const TEST_AGENT_CAP: u32 = 16;
const TICKS_TO_SIM: u32 = 12;

/// Lazy, shared wgpu device — pollster-blocks the async setup. Keeping
/// this as a function (not a thread-local) because the test runner
/// spawns threads per test; each test builds and tears down its own
/// ViewStorage.
fn make_device() -> (wgpu::Device, wgpu::Queue, String) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("request_adapter");
        let label = format!("{:?}", adapter.get_info().backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("view_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device");
        (device, queue, label)
    })
}

fn agent(slot: u32) -> AgentId {
    AgentId::new(slot + 1).expect("slot -> agent id")
}

// ---------------------------------------------------------------------------
// Shape-coverage test 1: slot_map — engaged_with.
//
// Fold a sequence of EngagementCommitted / EngagementBroken events on
// both sides; assert the dense GPU `array<u32>` slot map matches the
// CPU `HashMap`. Both sides encode AgentId+1 (with 0 = empty).
// ---------------------------------------------------------------------------

#[test]
fn slot_map_engaged_with_parity() {
    let (device, queue, backend_label) = make_device();
    eprintln!("view_parity: wgpu backend = {backend_label}");

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = EngagedWith::default();

    // Scripted sequence: commit, commit (different), break first, commit again.
    //   tick 1: 0 ↔ 1   (slot)
    //   tick 2: 2 ↔ 3
    //   tick 3: break (0, 1)
    //   tick 4: 4 ↔ 5
    //   tick 5: commit 0 ↔ 6 (reuse 0)
    let events: Vec<(u32, Event)> = vec![
        (
            1,
            Event::EngagementCommitted {
                actor: agent(0),
                target: agent(1),
                tick: 1,
            },
        ),
        (
            2,
            Event::EngagementCommitted {
                actor: agent(2),
                target: agent(3),
                tick: 2,
            },
        ),
        (
            3,
            Event::EngagementBroken {
                actor: agent(0),
                former_target: agent(1),
                reason: 0,
                tick: 3,
            },
        ),
        (
            4,
            Event::EngagementCommitted {
                actor: agent(4),
                target: agent(5),
                tick: 4,
            },
        ),
        (
            5,
            Event::EngagementCommitted {
                actor: agent(0),
                target: agent(6),
                tick: 5,
            },
        ),
    ];

    // CPU fold.
    for (tick, ev) in &events {
        cpu.fold_event(ev, *tick);
    }

    // GPU fold — translate each engagement event to FoldInputSlot.
    let gpu_events: Vec<FoldInputSlot> = events
        .iter()
        .map(|(_tick, ev)| match ev {
            Event::EngagementCommitted { actor, target, .. } => FoldInputSlot {
                first: actor.raw() - 1,
                second: target.raw() - 1,
                kind: 0,
                _pad: 0,
            },
            Event::EngagementBroken {
                actor,
                former_target,
                ..
            } => FoldInputSlot {
                first: actor.raw() - 1,
                second: former_target.raw() - 1,
                kind: 1,
                _pad: 0,
            },
            _ => unreachable!(),
        })
        .collect();
    // Dispatch one event at a time so the fold is strictly ordered.
    // Atomic CAS + in-batch parallelism is non-deterministic when
    // multiple events target the same slot; the parity guarantee only
    // holds for a sequential dispatch stream (see module docstring
    // on determinism policy).
    for ev in &gpu_events {
        gpu.fold_slot_events(&device, &queue, "engaged_with", std::slice::from_ref(ev))
            .expect("fold slot");
    }

    let gpu_slots = gpu
        .readback_slot_map(&device, &queue, "engaged_with")
        .expect("readback");

    // Assert every slot matches.
    for slot in 0..TEST_AGENT_CAP {
        let id = agent(slot);
        let cpu_partner = cpu.get(id);
        let cpu_encoded = cpu_partner.map(|p| p.raw()).unwrap_or(0);
        let gpu_encoded = gpu_slots[slot as usize];
        assert_eq!(
            gpu_encoded, cpu_encoded,
            "slot_map mismatch at slot {slot}: cpu={cpu_encoded} gpu={gpu_encoded}"
        );
    }
}

// ---------------------------------------------------------------------------
// Shape-coverage test 2: pair_map non-decay — my_enemies.
//
// Fold a sequence of AgentAttacked events. Each fold is `self += 1.0`
// clamped at `[0.0, 1.0]`. Assert every populated CPU cell shows up in
// the dense GPU buffer at the right index, and every un-touched GPU
// cell is 0.0.
// ---------------------------------------------------------------------------

#[test]
fn pair_map_my_enemies_parity() {
    let (device, queue, _label) = make_device();

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = MyEnemies::default();

    // Several attacks including repeats (clamp to 1.0) and distinct pairs.
    // Pairs (attacker → observer): (0, 1), (0, 1) again, (2, 3), (4, 5),
    // (0, 3), (2, 1). The view stores key = (target, actor) = (observer, attacker).
    let pairs: &[(u32, u32, u32)] = &[
        // (tick, attacker, observer)
        (1, 0, 1),
        (2, 0, 1), // repeat — clamp should hold at 1.0
        (3, 2, 3),
        (4, 4, 5),
        (5, 0, 3),
        (6, 2, 1),
    ];

    // CPU.
    for (tick, attacker, observer) in pairs {
        let ev = Event::AgentAttacked {
            actor: agent(*attacker),
            target: agent(*observer),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
    }

    // GPU — the view's a_name is `observer`, b_name is `attacker`.
    // The handler pattern binds `actor -> attacker, target -> observer`,
    // so first_key_field (view arg 0 = observer) = "target" (event's
    // target field), second_key_field = "actor". The FoldInputPair
    // is (first=observer, second=attacker).
    let gpu_events: Vec<FoldInputPair> = pairs
        .iter()
        .map(|(tick, attacker, observer)| FoldInputPair {
            first: *observer,
            second: *attacker,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    // Serial dispatch for deterministic CAS ordering — see slot_map
    // test for the rationale.
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, "my_enemies", std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_cells = gpu
        .readback_pair_scalar(&device, &queue, "my_enemies")
        .expect("readback scalar");
    assert_eq!(
        gpu_cells.len() as u32,
        TEST_AGENT_CAP * TEST_AGENT_CAP,
        "pair_map readback size"
    );

    // Walk every possible pair; assert equality with CPU.
    let n = TEST_AGENT_CAP;
    for obs in 0..n {
        for att in 0..n {
            let cpu_val = cpu.get(agent(obs), agent(att));
            let gpu_val = gpu_cells[(obs * n + att) as usize];
            assert!(
                (cpu_val - gpu_val).abs() < 1e-6,
                "my_enemies cell mismatch at ({obs},{att}): cpu={cpu_val} gpu={gpu_val}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shape-coverage test 3: pair_map @decay — threat_level.
//
// Fold events spread across multiple ticks; assert both the raw
// `(base, anchor)` cell pair and the observable "decayed get" value
// match between CPU and GPU after the last fold.
// ---------------------------------------------------------------------------

#[test]
fn pair_map_decay_threat_level_parity() {
    let (device, queue, _label) = make_device();

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = ThreatLevel::default();

    // Spread across ticks so decay is non-trivial. Cell (a=target,
    // b=actor) matches the view's handler pattern `{ actor: b, target: a }`.
    let events: &[(u32, u32, u32)] = &[
        // (tick, actor (= attacker), target)
        (1, 0, 1),
        (2, 0, 1),
        (4, 0, 1), // same cell, 3 hits spread out
        (3, 2, 3),
        (6, 2, 3),
        (5, 4, 5),
    ];

    for (tick, actor, target) in events {
        let ev = Event::AgentAttacked {
            actor: agent(*actor),
            target: agent(*target),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
    }

    // GPU: view's a_name is `a`, b_name is `b`. Handler pattern binds
    // `actor -> b, target -> a`. first_key_field = "target" (= target),
    // second_key_field = "actor" (= actor). So FoldInputPair.first =
    // target, .second = actor.
    let gpu_events: Vec<FoldInputPair> = events
        .iter()
        .map(|(tick, actor, target)| FoldInputPair {
            first: *target,
            second: *actor,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    // Serial dispatch for deterministic CAS ordering.
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, "threat_level", std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_cells = gpu
        .readback_pair_decay(&device, &queue, "threat_level")
        .expect("readback decay");
    assert_eq!(
        gpu_cells.len() as u32,
        TEST_AGENT_CAP * TEST_AGENT_CAP,
        "decay readback size"
    );

    // For each (observer, attacker) pair, compare the observable get()
    // at a few query ticks. The raw (base, anchor) also has to match
    // byte-for-byte inside the CPU's anchor-pattern convention:
    //   cpu stores (base, anchor) where base is POST-fold at anchor_tick
    //   gpu stores same shape.
    let n = TEST_AGENT_CAP;
    let query_ticks = [6u32, 10, 50];
    for a in 0..n {
        for b in 0..n {
            let cpu_cell_populated = !(cpu.get(agent(a), agent(b), 6) == 0.0
                && cpu.get(agent(a), agent(b), 0) == 0.0);
            let gpu_cell = gpu_cells[(a * n + b) as usize];

            // Un-populated cells: GPU should be (0.0, 0). CPU has no
            // entry → get() returns 0.0.
            if !cpu_cell_populated {
                assert_eq!(gpu_cell.value, 0.0, "stale cell at ({a},{b}) value");
                assert_eq!(gpu_cell.anchor, 0, "stale cell at ({a},{b}) anchor");
                continue;
            }

            // Compare the observable value across multiple ticks.
            for &qt in &query_ticks {
                let cpu_val = cpu.get(agent(a), agent(b), qt);
                let gpu_val = decay_get(&gpu_cell, ThreatLevel::RATE, qt, 0.0, 1000.0);
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-5,
                    "threat_level get mismatch at ({a},{b}) tick {qt}: cpu={cpu_val} gpu={gpu_val}"
                );
            }
        }
    }
}

/// Apply the decay formula exactly as the CPU `get()` does — used to
/// map a readback `DecayCellCpu` to its observable value at a query
/// tick. Mirrors `threat_level::ThreatLevel::get` and the emitter's
/// WGSL formula: `base * rate^(tick - anchor)` clamped.
fn decay_get(cell: &DecayCellCpu, rate: f32, tick: u32, lo: f32, hi: f32) -> f32 {
    if cell.value == 0.0 {
        return 0.0;
    }
    let dt = tick.saturating_sub(cell.anchor);
    let decayed = cell.value * rate.powi(dt as i32);
    decayed.clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Extended shape-coverage — the other three @decay pair_maps each get
// a compact fold + single-tick read check. Repeats the same flow as
// threat_level but with its own rate / handler shape, so a regression
// in any of (kin_fear, pack_focus, rally_boost)'s classify → spec →
// WGSL path surfaces here.
// ---------------------------------------------------------------------------

#[test]
fn pair_map_decay_kin_fear_parity() {
    run_decay_parity_scenario(
        "kin_fear",
        KinFear::RATE,
        0.0,
        10.0,
        // FearSpread { observer, dead_kin }
        |tick, first, second| Event::FearSpread {
            observer: agent(first),
            dead_kin: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        KinFear::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

#[test]
fn pair_map_decay_pack_focus_parity() {
    run_decay_parity_scenario(
        "pack_focus",
        PackFocus::RATE,
        0.0,
        10.0,
        // PackAssist { observer, target }
        |tick, first, second| Event::PackAssist {
            observer: agent(first),
            target: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        PackFocus::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

#[test]
fn pair_map_decay_rally_boost_parity() {
    run_decay_parity_scenario(
        "rally_boost",
        RallyBoost::RATE,
        0.0,
        10.0,
        // RallyCall { observer, wounded_kin }
        |tick, first, second| Event::RallyCall {
            observer: agent(first),
            wounded_kin: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        RallyBoost::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

/// Generic scaffold — tests a pair_map @decay view by folding the same
/// set of (tick, first, second) events on both sides and asserting
/// the observable get() and the dense cell layout match.
fn run_decay_parity_scenario<M>(
    view_name: &str,
    rate: f32,
    clamp_lo: f32,
    clamp_hi: f32,
    make_event: impl Fn(u32, u32, u32) -> Event,
    mut fold: impl FnMut(M, Event, u32) -> M,
    initial: M,
    get: impl Fn(&M, u32, u32, u32) -> f32,
) {
    let (device, queue, _label) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);

    let events: &[(u32, u32, u32)] = &[
        (1, 0, 1),
        (2, 0, 1),
        (3, 2, 3),
        (5, 0, 1), // 3rd hit, later tick → different decay
        (4, 4, 5),
        (6, 2, 3),
    ];

    // CPU fold.
    let mut cpu = initial;
    for (tick, first, second) in events {
        let ev = make_event(*tick, *first, *second);
        cpu = fold(cpu, ev, *tick);
    }

    // GPU fold — for these three views the handler binds
    // (observer, dead_kin)/(observer, target)/(observer, wounded_kin)
    // with the same ordering as view args. So FoldInputPair.first =
    // first (= observer), .second = second.
    let gpu_events: Vec<FoldInputPair> = events
        .iter()
        .map(|(tick, first, second)| FoldInputPair {
            first: *first,
            second: *second,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, view_name, std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_cells = gpu
        .readback_pair_decay(&device, &queue, view_name)
        .expect("readback decay");

    let n = TEST_AGENT_CAP;
    let query_ticks = [6u32, 10, 30];
    for a in 0..n {
        for b in 0..n {
            let cpu_v6 = get(&cpu, a, b, 6);
            let cpu_populated = !(cpu_v6 == 0.0 && get(&cpu, a, b, 0) == 0.0);
            let gpu_cell = gpu_cells[(a * n + b) as usize];
            if !cpu_populated {
                assert_eq!(
                    gpu_cell.value, 0.0,
                    "{view_name} stale cell at ({a},{b}) value"
                );
                assert_eq!(
                    gpu_cell.anchor, 0,
                    "{view_name} stale cell at ({a},{b}) anchor"
                );
                continue;
            }
            for &qt in &query_ticks {
                let cpu_val = get(&cpu, a, b, qt);
                let gpu_val = decay_get(&gpu_cell, rate, qt, clamp_lo, clamp_hi);
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-5,
                    "{view_name} get mismatch at ({a},{b}) tick {qt}: cpu={cpu_val} gpu={gpu_val}"
                );
            }
        }
    }
    let _ = Vec3::ZERO; // touch glam so the dev-dep link stays live.
}

// ---------------------------------------------------------------------------
// Storage-layout smoke check — make sure every materialized view is
// visible through `build_all_specs()`, which the ViewStorage
// constructor iterates. A new materialized view added to views.sim
// without being mirrored into engine_gpu's IR shim (see
// `build_materialized_view_irs`) would fail this early.
// ---------------------------------------------------------------------------

#[test]
fn ships_every_materialized_view_shape() {
    let specs = build_all_specs();
    let names: Vec<_> = specs.iter().map(|s| s.view_name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "engaged_with",
            "my_enemies",
            "threat_level",
            "kin_fear",
            "pack_focus",
            "rally_boost",
        ],
        "materialized view list drift — update engine_gpu::view_storage::build_materialized_view_irs"
    );
    // Shape coverage: one slot_map, two no-decay or decay-free, four @decay.
    let slot_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::SlotMap { .. }))
        .count();
    let pair_scalar_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::PairMapScalar))
        .count();
    let pair_decay_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::PairMapDecay { .. }))
        .count();
    assert_eq!(slot_count, 1, "expect 1 slot_map (engaged_with)");
    assert_eq!(
        pair_scalar_count, 1,
        "expect 1 pair_map non-decay (my_enemies)"
    );
    assert_eq!(
        pair_decay_count, 4,
        "expect 4 pair_map @decay (threat_level, kin_fear, pack_focus, rally_boost)"
    );
    let _ = TICKS_TO_SIM;
}

// ---------------------------------------------------------------------------
// Byte-equal readback for the cell buffer — compare raw u32 bits so we
// catch any endian / clamp-on-write drift. At TEST_AGENT_CAP=16 the
// readback is 16*16*4 = 1024 bytes; this is cheap.
// ---------------------------------------------------------------------------

#[test]
fn pair_map_scalar_byte_equal_readback() {
    let (device, queue, _label) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);

    // Empty read after reset — every cell must be 0x00000000.
    let cells = gpu
        .readback_pair_scalar(&device, &queue, "my_enemies")
        .expect("readback after reset");
    for (i, v) in cells.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0u32,
            "cell {i} non-zero after reset: bits={:#x}",
            v.to_bits()
        );
    }

    // Single fold — one event, one populated cell. Byte-equal 1.0
    // (0x3f800000) at the target slot, zero elsewhere.
    gpu.fold_pair_events(
        &device,
        &queue,
        "my_enemies",
        &[FoldInputPair {
            first: 2,
            second: 3,
            tick: 1,
            _pad: 0,
        }],
    )
    .expect("fold");
    let cells = gpu
        .readback_pair_scalar(&device, &queue, "my_enemies")
        .expect("readback after fold");
    let n = TEST_AGENT_CAP;
    let idx_23 = (2 * n + 3) as usize;
    assert_eq!(
        cells[idx_23].to_bits(),
        1.0_f32.to_bits(),
        "expected +1.0 at (2,3); got {:?}",
        cells[idx_23]
    );
    for (i, v) in cells.iter().enumerate() {
        if i == idx_23 {
            continue;
        }
        assert_eq!(
            v.to_bits(),
            0u32,
            "stray non-zero cell at {i} after single fold"
        );
    }
}
