//! Task 196 — per_entity_topk(K=8) view parity.
//!
//! Three flavours of coverage for the 4 topk-migrated views:
//!
//!   1. **Per-view basic fold parity** — fold a small deterministic
//!      event stream on both CPU and GPU; assert every (observer,
//!      attacker) pair's stored value matches. Sorts by id inside each
//!      observer's K slots before comparison since the GPU fold's fill
//!      order may differ from the CPU (task 185's policy).
//!
//!   2. **Stress** — 1000 events across 100 observers for every topk
//!      view, verifying the two sides stay in agreement.
//!
//!   3. **Pinning / eviction** — 10 distinct attackers per observer at
//!      K=8: verify the 2 lowest-value slots get evicted and only the
//!      8 highest-value pairs survive.
//!
//! These tests exercise `fold_pair_events` + `readback_topk` on the
//! `ViewStorage` API directly — they don't drive the cascade integration
//! layer. That means the view-under-test is built by `ViewStorage::new`
//! and has no direct dependency on events.sim / cascade.rs; a topk
//! regression at the storage layer surfaces here even if no cascade
//! pipeline can fold the view yet.
//!
//! Run with `cargo test -p engine_gpu --features gpu --test topk_view_parity`.

#![cfg(feature = "gpu")]

use engine::event::Event;
use engine::generated::views::{MyEnemies, PackFocus, RallyBoost, ThreatLevel};
use engine::ids::AgentId;
use engine_gpu::view_storage::{FoldInputPair, TopkReadback, ViewStorage};

const TEST_AGENT_CAP: u32 = 128;

fn agent(slot: u32) -> AgentId {
    AgentId::new(slot + 1).expect("slot -> agent id")
}

fn make_device() -> (wgpu::Device, wgpu::Queue) {
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("topk_view_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device");
        (device, queue)
    })
}

/// Collect the populated (id, value) slots for observer `obs` from the
/// topk readback, sorted by id so CPU/GPU fill order doesn't matter.
fn populated_slots_by_id(rb: &TopkReadback, obs: u32) -> Vec<(u32, f32)> {
    let (ids, vals, _anchors) = rb.row(obs);
    let mut out: Vec<(u32, f32)> = ids
        .iter()
        .zip(vals.iter())
        .filter(|(id, _)| **id != 0)
        .map(|(id, v)| (*id, *v))
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

// ---------------------------------------------------------------------------
// Per-view fold parity
// ---------------------------------------------------------------------------

/// my_enemies — no decay, clamp [0, 1]. The second hit on the same
/// pair is idempotent (clamps back to 1.0); new pairs land in empty
/// slots. With <= K distinct attackers per observer no eviction runs.
#[test]
fn topk_my_enemies_basic_parity() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = MyEnemies::default();

    // (tick, attacker, observer). Some repeats, 4 distinct pairs, 2
    // distinct observers — stays well inside K=8.
    let events: &[(u32, u32, u32)] = &[
        (1, 0, 5),
        (1, 1, 5),
        (2, 0, 5), // repeat (observer=5, attacker=0) — value already 1.0
        (3, 2, 5),
        (4, 3, 5),
        (5, 0, 7),
        (6, 2, 7),
    ];
    for (tick, atk, obs) in events {
        let ev = Event::AgentAttacked {
            actor: agent(*atk),
            target: agent(*obs),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
        gpu.fold_pair_events(
            &device,
            &queue,
            "my_enemies",
            &[FoldInputPair {
                first: *obs,
                second: *atk,
                tick: *tick,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    let rb = gpu
        .readback_topk(&device, &queue, "my_enemies")
        .expect("readback topk");

    for obs in 0..TEST_AGENT_CAP {
        let gpu_pairs = populated_slots_by_id(&rb, obs);
        let mut cpu_pairs: Vec<(u32, f32)> = (0..TEST_AGENT_CAP)
            .map(|a| (agent(a).raw(), cpu.get(agent(obs), agent(a))))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        cpu_pairs.sort_by_key(|(id, _)| *id);
        assert_eq!(cpu_pairs, gpu_pairs, "obs={obs}");
    }
}

/// threat_level — decay 0.98, clamp [0, 1000]. Accumulates across
/// multiple ticks; the decay math on read has to match between CPU
/// and GPU. Sort-by-id then compare get() values across query ticks.
#[test]
fn topk_threat_level_basic_parity() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = ThreatLevel::default();

    // (tick, actor, target). Handler binds `actor: b, target: a`, so
    // view param `a` = target (observer), `b` = actor.
    let events: &[(u32, u32, u32)] = &[
        (1, 0, 5),
        (4, 0, 5), // same pair later → decay then +1.0
        (2, 1, 5),
        (3, 2, 6),
        (5, 2, 6),
        (6, 3, 5),
    ];
    for (tick, actor, target) in events {
        let ev = Event::AgentAttacked {
            actor: agent(*actor),
            target: agent(*target),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
        gpu.fold_pair_events(
            &device,
            &queue,
            "threat_level",
            &[FoldInputPair {
                first: *target,
                second: *actor,
                tick: *tick,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    let rb = gpu
        .readback_topk(&device, &queue, "threat_level")
        .expect("readback topk");

    let query_ticks = [6u32, 10, 50];
    for obs in 0..TEST_AGENT_CAP {
        for atk in 0..TEST_AGENT_CAP {
            let gpu_val = |qt: u32| -> f32 {
                let (ids, vals, anchors) = rb.row(obs);
                for i in 0..ids.len() {
                    if ids[i] == agent(atk).raw() {
                        let dt = qt.saturating_sub(anchors[i]);
                        let decayed = vals[i] * ThreatLevel::RATE.powi(dt as i32);
                        return decayed.clamp(0.0, 1000.0);
                    }
                }
                0.0
            };
            for &qt in &query_ticks {
                let cpu_v = cpu.get(agent(obs), agent(atk), qt);
                let gpu_v = gpu_val(qt);
                assert!(
                    (cpu_v - gpu_v).abs() < 1e-5,
                    "threat_level obs={obs} atk={atk} tick={qt} cpu={cpu_v} gpu={gpu_v}"
                );
            }
        }
    }
}

/// pack_focus — decay 0.933, clamp [0, 10]. Handler binds
/// `observer, target` with the same ordering as view args, so
/// FoldInputPair.first = observer, .second = target.
#[test]
fn topk_pack_focus_basic_parity() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = PackFocus::default();

    let events: &[(u32, u32, u32)] = &[
        (1, 5, 0),
        (2, 5, 0),
        (3, 5, 0), // 3 hits on the same pair
        (4, 5, 1),
        (5, 7, 2),
    ];
    for (tick, observer, target) in events {
        let ev = Event::PackAssist {
            observer: agent(*observer),
            target: agent(*target),
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
        gpu.fold_pair_events(
            &device,
            &queue,
            "pack_focus",
            &[FoldInputPair {
                first: *observer,
                second: *target,
                tick: *tick,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    let rb = gpu
        .readback_topk(&device, &queue, "pack_focus")
        .expect("readback topk");

    let query_ticks = [5u32, 10, 30];
    for obs in 0..TEST_AGENT_CAP {
        for atk in 0..TEST_AGENT_CAP {
            let gpu_val = |qt: u32| -> f32 {
                let (ids, vals, anchors) = rb.row(obs);
                for i in 0..ids.len() {
                    if ids[i] == agent(atk).raw() {
                        let dt = qt.saturating_sub(anchors[i]);
                        let decayed = vals[i] * PackFocus::RATE.powi(dt as i32);
                        return decayed.clamp(0.0, 10.0);
                    }
                }
                0.0
            };
            for &qt in &query_ticks {
                let cpu_v = cpu.get(agent(obs), agent(atk), qt);
                let gpu_v = gpu_val(qt);
                assert!(
                    (cpu_v - gpu_v).abs() < 1e-4,
                    "pack_focus obs={obs} atk={atk} tick={qt} cpu={cpu_v} gpu={gpu_v}"
                );
            }
        }
    }
}

#[test]
fn topk_rally_boost_basic_parity() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = RallyBoost::default();

    let events: &[(u32, u32, u32)] = &[
        (1, 3, 5),
        (2, 3, 6),
        (4, 3, 5),
        (5, 4, 7),
        (6, 4, 7),
    ];
    for (tick, observer, wounded_kin) in events {
        let ev = Event::RallyCall {
            observer: agent(*observer),
            wounded_kin: agent(*wounded_kin),
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
        gpu.fold_pair_events(
            &device,
            &queue,
            "rally_boost",
            &[FoldInputPair {
                first: *observer,
                second: *wounded_kin,
                tick: *tick,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    let rb = gpu
        .readback_topk(&device, &queue, "rally_boost")
        .expect("readback topk");

    let query_ticks = [6u32, 12, 40];
    for obs in 0..TEST_AGENT_CAP {
        for atk in 0..TEST_AGENT_CAP {
            let gpu_val = |qt: u32| -> f32 {
                let (ids, vals, anchors) = rb.row(obs);
                for i in 0..ids.len() {
                    if ids[i] == agent(atk).raw() {
                        let dt = qt.saturating_sub(anchors[i]);
                        let decayed = vals[i] * RallyBoost::RATE.powi(dt as i32);
                        return decayed.clamp(0.0, 10.0);
                    }
                }
                0.0
            };
            for &qt in &query_ticks {
                let cpu_v = cpu.get(agent(obs), agent(atk), qt);
                let gpu_v = gpu_val(qt);
                assert!(
                    (cpu_v - gpu_v).abs() < 1e-4,
                    "rally_boost obs={obs} atk={atk} tick={qt} cpu={cpu_v} gpu={gpu_v}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stress: 1000 events across 100 observers
// ---------------------------------------------------------------------------

/// Fold 1000 events through every topk view (my_enemies, threat_level,
/// pack_focus, rally_boost). Verify CPU↔GPU equality per observer as a
/// sorted-by-id slot list. With 100 distinct attackers possible and
/// K=8 this covers both the no-eviction path and the eviction path.
#[test]
fn topk_stress_1000_events_all_views() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu_my = MyEnemies::default();
    let mut cpu_threat = ThreatLevel::default();
    let mut cpu_pack = PackFocus::default();
    let mut cpu_rally = RallyBoost::default();

    // Deterministic PRNG — small LCG keeps the test self-contained.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };

    for _i in 0..1000 {
        let tick = rand() % 200 + 1;
        // 100 observers, 100 attackers. Skip self-pairs.
        let obs = rand() % 100;
        let atk = rand() % 100;
        if obs == atk {
            continue;
        }

        // Fold into every topk view. Each view has a slightly
        // different field binding — shared helper below.
        fold_all_views(
            &device,
            &queue,
            &mut gpu,
            &mut cpu_my,
            &mut cpu_threat,
            &mut cpu_pack,
            &mut cpu_rally,
            tick,
            obs,
            atk,
        );
    }

    // Sweep per-view, per-observer, per-attacker; check sorted-slot
    // equality (topk fill order may differ across CPU/GPU but the set
    // of populated (id, value) pairs must match).
    assert_topk_parity(&device, &queue, &gpu, "my_enemies", |obs, atk| {
        cpu_my.get(agent(obs), agent(atk))
    });
    assert_topk_decay_parity(
        &device,
        &queue,
        &gpu,
        "threat_level",
        ThreatLevel::RATE,
        0.0,
        1000.0,
        |obs, atk, tick| cpu_threat.get(agent(obs), agent(atk), tick),
    );
    assert_topk_decay_parity(
        &device,
        &queue,
        &gpu,
        "pack_focus",
        PackFocus::RATE,
        0.0,
        10.0,
        |obs, atk, tick| cpu_pack.get(agent(obs), agent(atk), tick),
    );
    assert_topk_decay_parity(
        &device,
        &queue,
        &gpu,
        "rally_boost",
        RallyBoost::RATE,
        0.0,
        10.0,
        |obs, atk, tick| cpu_rally.get(agent(obs), agent(atk), tick),
    );
}

// Fold one (tick, observer, attacker) triple into every topk view on
// both CPU and GPU. Matches the field bindings for each view.
#[allow(clippy::too_many_arguments)]
fn fold_all_views(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut ViewStorage,
    cpu_my: &mut MyEnemies,
    cpu_threat: &mut ThreatLevel,
    cpu_pack: &mut PackFocus,
    cpu_rally: &mut RallyBoost,
    tick: u32,
    obs: u32,
    atk: u32,
) {
    // my_enemies + threat_level: attacker is actor, observer is target.
    let attack_ev = Event::AgentAttacked {
        actor: agent(atk),
        target: agent(obs),
        damage: 1.0,
        tick,
    };
    cpu_my.fold_event(&attack_ev, tick);
    cpu_threat.fold_event(&attack_ev, tick);
    gpu.fold_pair_events(
        device,
        queue,
        "my_enemies",
        &[FoldInputPair {
            first: obs,
            second: atk,
            tick,
            _pad: 0,
        }],
    )
    .expect("fold my_enemies");
    gpu.fold_pair_events(
        device,
        queue,
        "threat_level",
        &[FoldInputPair {
            first: obs,
            second: atk,
            tick,
            _pad: 0,
        }],
    )
    .expect("fold threat_level");

    // pack_focus: (observer, target) directly.
    let pack_ev = Event::PackAssist {
        observer: agent(obs),
        target: agent(atk),
        tick,
    };
    cpu_pack.fold_event(&pack_ev, tick);
    gpu.fold_pair_events(
        device,
        queue,
        "pack_focus",
        &[FoldInputPair {
            first: obs,
            second: atk,
            tick,
            _pad: 0,
        }],
    )
    .expect("fold pack_focus");

    // rally_boost: (observer, wounded_kin) directly.
    let rally_ev = Event::RallyCall {
        observer: agent(obs),
        wounded_kin: agent(atk),
        tick,
    };
    cpu_rally.fold_event(&rally_ev, tick);
    gpu.fold_pair_events(
        device,
        queue,
        "rally_boost",
        &[FoldInputPair {
            first: obs,
            second: atk,
            tick,
            _pad: 0,
        }],
    )
    .expect("fold rally_boost");
}

fn assert_topk_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &ViewStorage,
    view: &str,
    cpu_get: impl Fn(u32, u32) -> f32,
) {
    let rb = gpu
        .readback_topk(device, queue, view)
        .expect("readback topk");
    for obs in 0..100 {
        let gpu_pairs = populated_slots_by_id(&rb, obs);
        let mut cpu_pairs: Vec<(u32, f32)> = (0..100)
            .map(|a| (agent(a).raw(), cpu_get(obs, a)))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        cpu_pairs.sort_by_key(|(id, _)| *id);
        assert_eq!(
            cpu_pairs, gpu_pairs,
            "{view} obs={obs} slot mismatch"
        );
    }
}

fn assert_topk_decay_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &ViewStorage,
    view: &str,
    rate: f32,
    lo: f32,
    hi: f32,
    cpu_get: impl Fn(u32, u32, u32) -> f32,
) {
    let rb = gpu
        .readback_topk(device, queue, view)
        .expect("readback topk");
    // Query at a couple of future ticks so the decay formula runs.
    for tick in [250u32, 500] {
        for obs in 0..100 {
            let (ids, vals, anchors) = rb.row(obs);
            // Build a sorted GPU-observable slot list for (id, decayed).
            let mut gpu_pairs: Vec<(u32, f32)> = ids
                .iter()
                .zip(vals.iter().zip(anchors.iter()))
                .filter(|(id, _)| **id != 0)
                .map(|(id, (v, a))| {
                    let dt = tick.saturating_sub(*a);
                    let decayed = v * rate.powi(dt as i32);
                    (*id, decayed.clamp(lo, hi))
                })
                .collect();
            gpu_pairs.sort_by_key(|(id, _)| *id);
            let mut cpu_pairs: Vec<(u32, f32)> = (0..100)
                .map(|a| (agent(a).raw(), cpu_get(obs, a, tick)))
                .filter(|(_, v)| *v > 0.0)
                .collect();
            cpu_pairs.sort_by_key(|(id, _)| *id);
            assert_eq!(
                cpu_pairs.len(),
                gpu_pairs.len(),
                "{view} obs={obs} tick={tick} slot count: cpu={cpu_pairs:?} gpu={gpu_pairs:?}"
            );
            for ((cpu_id, cpu_v), (gpu_id, gpu_v)) in cpu_pairs.iter().zip(gpu_pairs.iter()) {
                assert_eq!(cpu_id, gpu_id, "{view} obs={obs} id mismatch");
                assert!(
                    (cpu_v - gpu_v).abs() < 1e-4,
                    "{view} obs={obs} id={cpu_id} tick={tick} cpu={cpu_v} gpu={gpu_v}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pinning / eviction
// ---------------------------------------------------------------------------

/// Eviction policy: at K=8 with 10 distinct attackers, the first 8
/// distinct attackers to land fill the slots in insertion order; the
/// next 2 (attackers 8 and 9) arrive while the slots are full. Every
/// fold is `+1.0` — the find-or-evict-else-drop contract requires
/// `delta > min_value` for eviction, so the 9th and 10th distinct
/// attackers at delta=1 CANNOT displace the existing slots which
/// already sit at value=1. They drop silently.
///
/// This is the "fold order shapes membership" property: the survivors
/// are the 8 attackers that arrived *first*, not the strongest by
/// any other metric. To get value-based eviction under the current
/// `+1.0` invariant, weaker attackers must have lower *accumulated*
/// values at the moment the newcomer arrives — which means the first
/// folds for the strong ones have to happen BEFORE the slot fills.
/// The test sets that up: pre-feed attacker 0 once (value 1),
/// attackers 1..=7 twice each (values 2..=8, filling K slots),
/// then send attackers 8 and 9 with 3 folds each. The first of
/// attacker 8's folds (delta=1, min=1) drops. Then attacker 8's
/// second fold still can't enter (no existing slot for 8, no empty
/// slot, min=1, delta=1). Attacker 8 never gets in.
///
/// The cleanest way to exercise eviction given delta=+1.0 is to
/// *accumulate* inside existing slots first (raising their baseline)
/// and then have a newcomer that is given repeated folds on the
/// same tick — but each individual +1.0 newcomer fold still hits
/// min=1 against the untouched slot and drops. So eviction under
/// the shipped `+1.0` invariant is only reachable when the min
/// slot's value has DECAYED below 1.0 (via time passing) before a
/// fresh attacker tries to land.
///
/// Test flow:
///   - Tick 1: fold attackers 0..=7 once each → K slots filled at
///     value 1 (anchors all = 1).
///   - Tick 100: fold attacker 8 once. By then every old slot has
///     decayed to `1 * 0.933^99 ≈ 0.001`, so delta=1 > min ≈ 0.001
///     and attacker 8 evicts slot-with-smallest-decayed = the first
///     inserted (attacker 0).
///   - Tick 101: fold attacker 9 once. Same story — slot-1 (attacker
///     1, inserted next) is now the smallest and gets evicted.
#[test]
fn topk_eviction_keeps_top_k_by_value() {
    let (device, queue) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = PackFocus::default();
    let obs: u32 = 42;

    // Stage 1: fill K slots at tick 1.
    for atk in 0..8u32 {
        let ev = Event::PackAssist {
            observer: agent(obs),
            target: agent(atk),
            tick: 1,
        };
        cpu.fold_event(&ev, 1);
        gpu.fold_pair_events(
            &device,
            &queue,
            "pack_focus",
            &[FoldInputPair {
                first: obs,
                second: atk,
                tick: 1,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    // Stage 2: tick 100 — slots decayed to ~0.001. Land attackers 8
    // and 9 at different ticks so their eviction math is stable.
    for (tick, atk) in [(100u32, 8u32), (101, 9)] {
        let ev = Event::PackAssist {
            observer: agent(obs),
            target: agent(atk),
            tick,
        };
        cpu.fold_event(&ev, tick);
        gpu.fold_pair_events(
            &device,
            &queue,
            "pack_focus",
            &[FoldInputPair {
                first: obs,
                second: atk,
                tick,
                _pad: 0,
            }],
        )
        .expect("fold");
    }

    // Readback the final state.
    let rb = gpu
        .readback_topk(&device, &queue, "pack_focus")
        .expect("readback topk");
    let slots = populated_slots_by_id(&rb, obs);
    assert_eq!(
        slots.len(),
        8,
        "expected K=8 populated slots, got {slots:?}"
    );

    // Attackers 0 and 1 should have been evicted.
    let ids: Vec<u32> = slots.iter().map(|(id, _)| *id).collect();
    assert!(
        !ids.contains(&agent(0).raw()),
        "attacker 0 should have been evicted; got {ids:?}"
    );
    assert!(
        !ids.contains(&agent(1).raw()),
        "attacker 1 should have been evicted; got {ids:?}"
    );
    // Attackers 8 and 9 must be present.
    assert!(
        ids.contains(&agent(8).raw()),
        "attacker 8 should have been inserted; got {ids:?}"
    );
    assert!(
        ids.contains(&agent(9).raw()),
        "attacker 9 should have been inserted; got {ids:?}"
    );

    // Parity with CPU get() at a late tick.
    for atk in 0..10u32 {
        let cpu_v = cpu.get(agent(obs), agent(atk), 200);
        let (ids_row, vals_row, anchors_row) = rb.row(obs);
        let gpu_v = ids_row
            .iter()
            .zip(vals_row.iter().zip(anchors_row.iter()))
            .find(|(id, _)| **id == agent(atk).raw())
            .map(|(_, (v, a))| {
                let dt = 200u32.saturating_sub(*a);
                let decayed = v * PackFocus::RATE.powi(dt as i32);
                decayed.clamp(0.0, 10.0)
            })
            .unwrap_or(0.0);
        assert!(
            (cpu_v - gpu_v).abs() < 1e-5,
            "pack_focus atk={atk} cpu={cpu_v} gpu={gpu_v}"
        );
    }
}
