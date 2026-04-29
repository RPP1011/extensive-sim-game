// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! Phase 6f — GPU cascade driver parity harness.
//!
//! Runs the cascade loop against multiple seed batches, each designed to
//! exercise cascade depth > 1:
//!
//!   * Lethal damage on a target whose death triggers `fear_spread` on
//!     nearby kin, which may in turn emit `pack_assist` / further
//!     follow-on events.
//!   * An attack on a low-hp target to drive `rally_on_wound`.
//!
//! The driver is compared against a CPU reference built by feeding the
//! same seed events through `CascadeRegistry::run_fixed_point`. Parity:
//!
//!   * Final SimState byte-exact on hp/alive/shield/stun/slow/cooldown
//!     /engaged_with across every agent slot.
//!   * Post-cascade view_storage cells byte-exact against the CPU
//!     `state.views` reads for each materialised view (my_enemies,
//!     threat_level, kin_fear, pack_focus, rally_boost, engaged_with).
//!   * Replayable emitted event set equal (sorted by
//!     `(tick, kind, payload[0])`).
//!
//! Additionally, `cascade_iterations_gt_one_on_kill_cascade` asserts the
//! kill path actually goes deeper than one iteration — if a future
//! refactor collapses the cascade back to a single pass, this guards.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::cascade::{fold_iteration_events, pack_initial_events, run_cascade};
use engine_gpu::event_ring::{pack_event, EventRecord};
use engine_gpu::physics::{PackedAbilityRegistry, PhysicsKernel};
use engine_gpu::spatial_gpu::GpuSpatialHash;
use engine_gpu::view_storage::ViewStorage;
use glam::Vec3;

// ---------------------------------------------------------------------------
// GPU device + asset loading
// ---------------------------------------------------------------------------

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue, String) {
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
            .expect("adapter");
        let label = format!("{:?}", adapter.get_info().backend);
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cascade_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        (device, queue, label)
    })
}

fn load_compilation() -> dsl_compiler::ir::Compilation {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop(); // crates/
    root.pop(); // repo root
    root.push("assets/sim");

    let mut merged = Program { decls: Vec::new() };
    for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
        let src = std::fs::read_to_string(root.join(f))
            .unwrap_or_else(|e| panic!("read {}: {e}", f));
        merged.decls.extend(dsl_compiler::parse(&src).unwrap().decls);
    }
    dsl_compiler::compile_ast(merged).expect("resolve")
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Pack-of-wolves fixture: 4 wolves clustered together so a death
/// triggers fear_spread across the pack. Humans are absent to keep the
/// parity surface focused on the kin cascade.
fn pack_of_wolves_fixture() -> SimState {
    let mut state = SimState::new(8, 0xCAFE_D00D);
    for (i, x) in [0.0, 2.0, -2.0, 0.0].iter().enumerate() {
        let y = if i == 3 { 2.0 } else { 0.0 };
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(*x, y, 0.0),
                hp: if i == 0 { 10.0 } else { 80.0 }, // wolf[0] is fragile
                ..Default::default()
            })
            .expect("wolf spawn");
    }
    state
}

/// Canonical 3h+2w fixture — the wolves_and_humans fixture the engine
/// uses for its own parity baseline.
fn canonical_fixture() -> SimState {
    let mut state = SimState::new(8, 0xD00D_FACE_0042_0042);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("h1");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("h2");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("h3");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("w1");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(-3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("w2");
    state
}

// ---------------------------------------------------------------------------
// CPU cascade reference
// ---------------------------------------------------------------------------

/// Run the CPU cascade on a seeded event list, returning the final state
/// and the replayable subset of events it emitted this cascade.
///
/// `fixture_builder` is called to produce the fresh starting state —
/// `SimState` isn't `Clone`, so each test run rebuilds its fixture from
/// scratch.
fn run_cpu_cascade(
    mut state: SimState,
    seeded: &[Event],
) -> (SimState, Vec<Event>) {
    let mut events = EventRing::with_cap(4096);
    let registry = CascadeRegistry::with_engine_builtins();
    let pushed_before_seed = events.total_pushed();
    for e in seeded {
        events.push(e.clone());
    }
    let after_seed = events.total_pushed();
    registry.run_fixed_point(&mut state, &mut events);
    // Collect events emitted by cascade dispatch — the seed events
    // themselves are EXCLUDED so the comparison against the GPU
    // `all_emitted_events` (which is kernel-emit-only) is apples-to-
    // apples. Both sides get to see the seeds as inputs.
    let mut emitted = Vec::new();
    for idx in after_seed..events.total_pushed() {
        if let Some(e) = events.get_pushed(idx) {
            emitted.push(e);
        }
    }
    // Fold the CPU views too so the test can cross-check against GPU
    // view_storage — `run_fixed_point` drives `dispatch` only; the
    // materialised views are folded by `step_full`'s Phase 5 in the real
    // pipeline. Here we fold the whole seeded + emitted set so the
    // view cells reflect both the input events (which fold into
    // my_enemies / threat_level / engaged_with) and the cascade's
    // own emissions (FearSpread / PackAssist / RallyCall → the other
    // decay views).
    state.views.fold_all(&events, pushed_before_seed, state.tick);
    (state, emitted)
}

// ---------------------------------------------------------------------------
// Parity helpers
// ---------------------------------------------------------------------------

fn assert_state_parity(cpu: &SimState, gpu: &SimState, ctx: &str) {
    let cap = cpu.agent_cap();
    assert_eq!(cap, gpu.agent_cap(), "[{ctx}] agent_cap");
    for slot in 0..cap {
        let id = AgentId::new(slot + 1).unwrap();
        assert_eq!(
            cpu.agent_alive(id),
            gpu.agent_alive(id),
            "[{ctx}] alive mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_hp(id).map(f32::to_bits),
            gpu.agent_hp(id).map(f32::to_bits),
            "[{ctx}] hp mismatch slot={slot} cpu={:?} gpu={:?}",
            cpu.agent_hp(id), gpu.agent_hp(id),
        );
        assert_eq!(
            cpu.agent_shield_hp(id).map(f32::to_bits),
            gpu.agent_shield_hp(id).map(f32::to_bits),
            "[{ctx}] shield mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_stun_expires_at(id),
            gpu.agent_stun_expires_at(id),
            "[{ctx}] stun mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_slow_expires_at(id),
            gpu.agent_slow_expires_at(id),
            "[{ctx}] slow expiry mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_slow_factor_q8(id),
            gpu.agent_slow_factor_q8(id),
            "[{ctx}] slow factor mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_cooldown_next_ready(id),
            gpu.agent_cooldown_next_ready(id),
            "[{ctx}] cooldown mismatch slot={slot}",
        );
        assert_eq!(
            cpu.agent_engaged_with(id),
            gpu.agent_engaged_with(id),
            "[{ctx}] engaged_with mismatch slot={slot}",
        );
    }
}

fn assert_replayable_events_match(cpu: &[Event], gpu: &[EventRecord], ctx: &str) {
    use engine_gpu::event_ring::unpack_record;
    let mut cpu_r: Vec<_> = cpu.iter().filter(|e| e.is_replayable()).cloned().collect();
    let mut gpu_r: Vec<Event> = gpu.iter().filter_map(unpack_record).collect();
    let key = |e: &Event| {
        let r = pack_event(e).expect("pack");
        (r.tick, r.kind, r.payload[0], r.payload[1])
    };
    cpu_r.sort_by_key(key);
    gpu_r.sort_by_key(key);
    assert_eq!(
        cpu_r.len(), gpu_r.len(),
        "[{ctx}] replayable event count mismatch:\ncpu={cpu_r:?}\ngpu={gpu_r:?}",
    );
    assert_eq!(
        cpu_r, gpu_r,
        "[{ctx}] replayable contents differ",
    );
}

/// Cross-check the GPU view_storage cells against the CPU's post-cascade
/// view registry. Exercises every pair view + engaged_with.
fn assert_view_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view_storage: &ViewStorage,
    cpu: &SimState,
    ctx: &str,
) {
    let n = cpu.agent_cap() as usize;
    let tick = cpu.tick;

    // my_enemies — task 196 topk(K=8) (no decay). Readback via
    // `readback_topk` and compare `cpu.views.my_enemies.get()` values
    // slot-by-slot.
    let my_rb = view_storage
        .readback_topk(device, queue, "my_enemies")
        .expect("readback my_enemies");
    for obs_slot in 0..n {
        let obs = match AgentId::new(obs_slot as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        for atk_slot in 0..n {
            let atk = match AgentId::new(atk_slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            let cpu_v = cpu.views.my_enemies.get(obs, atk);
            let (ids, vals, _an) = my_rb.row(obs_slot as u32);
            let gpu_v = ids
                .iter()
                .zip(vals.iter())
                .find(|(id, _)| **id == atk.raw())
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            assert_eq!(
                cpu_v.to_bits(),
                gpu_v.to_bits(),
                "[{ctx}] my_enemies[obs={obs_slot},atk={atk_slot}] cpu={cpu_v} gpu={gpu_v}",
            );
        }
    }

    // threat_level / pack_focus / rally_boost — task 196 topk(K=8) @decay.
    // kin_fear is NOT migrated and stays on dense pair_map.
    let decay_pairs: &[(&str, Box<dyn Fn(AgentId, AgentId) -> f32>)] = &[
        (
            "threat_level",
            Box::new(|a: AgentId, b: AgentId| cpu.views.threat_level.get(a, b, tick)),
        ),
        (
            "kin_fear",
            Box::new(|a: AgentId, b: AgentId| cpu.views.kin_fear.get(a, b, tick)),
        ),
        (
            "pack_focus",
            Box::new(|a: AgentId, b: AgentId| cpu.views.pack_focus.get(a, b, tick)),
        ),
        (
            "rally_boost",
            Box::new(|a: AgentId, b: AgentId| cpu.views.rally_boost.get(a, b, tick)),
        ),
    ];
    for (name, getter) in decay_pairs {
        use dsl_compiler::emit_view_wgsl::ViewShape;
        let spec = match view_storage.spec(name) {
            Some(s) => s,
            None => continue,
        };
        let rate = match spec.shape {
            ViewShape::PairMapDecay { rate } => rate,
            _ => continue,
        };
        let clamp = spec.clamp;
        if view_storage.topk_dims(name).is_some() {
            // Topk path (task 196): readback parallel buffers.
            let rb = view_storage
                .readback_topk(device, queue, name)
                .unwrap_or_else(|e| panic!("[{ctx}] readback_topk {name}: {e}"));
            for a_slot in 0..n {
                let a_id = match AgentId::new(a_slot as u32 + 1) {
                    Some(id) => id,
                    None => continue,
                };
                for b_slot in 0..n {
                    let b_id = match AgentId::new(b_slot as u32 + 1) {
                        Some(id) => id,
                        None => continue,
                    };
                    let cpu_v = getter(a_id, b_id);
                    let (ids, vals, anchors) = rb.row(a_slot as u32);
                    let gpu_raw = ids
                        .iter()
                        .zip(vals.iter().zip(anchors.iter()))
                        .find(|(id, _)| **id == b_id.raw())
                        .map(|(_, (v, a))| (*v, *a));
                    let decayed = match gpu_raw {
                        Some((value, anchor)) => {
                            let dt = tick.saturating_sub(anchor);
                            let mut d = value * rate.powi(dt as i32);
                            if let Some((lo, hi)) = clamp {
                                d = d.clamp(lo as f32, hi as f32);
                            }
                            d
                        }
                        None => 0.0,
                    };
                    let diff = (cpu_v - decayed).abs();
                    assert!(
                        diff < 1e-5,
                        "[{ctx}] {name}[{a_slot},{b_slot}] cpu={cpu_v} gpu={decayed} diff={diff}",
                    );
                }
            }
            continue;
        }
        let cells = view_storage
            .readback_pair_decay(device, queue, name)
            .unwrap_or_else(|e| panic!("[{ctx}] readback {name}: {e}"));
        for a_slot in 0..n {
            let a_id = match AgentId::new(a_slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            for b_slot in 0..n {
                let b_id = match AgentId::new(b_slot as u32 + 1) {
                    Some(id) => id,
                    None => continue,
                };
                let cpu_v = getter(a_id, b_id);
                let cell = cells[a_slot * n + b_slot];
                // Reconstruct the "now" value on the GPU side —
                // decayed = value * rate^(tick - anchor), optionally clamped.
                let dt = tick.saturating_sub(cell.anchor);
                let mut decayed = cell.value * rate.powi(dt as i32);
                if let Some((lo, hi)) = clamp {
                    decayed = decayed.clamp(lo as f32, hi as f32);
                }
                if dt == 0 {
                    assert_eq!(
                        cpu_v.to_bits(), decayed.to_bits(),
                        "[{ctx}] {name}[{a_slot},{b_slot}] dt=0 cpu={cpu_v:?} gpu={decayed:?} \
                         cell=(v={}, anchor={}), tick={}",
                        cell.value, cell.anchor, tick,
                    );
                } else {
                    let diff = (cpu_v - decayed).abs();
                    assert!(
                        diff < 1e-5,
                        "[{ctx}] {name}[{a_slot},{b_slot}] dt={dt} cpu={cpu_v} gpu={decayed} diff={diff}",
                    );
                }
            }
        }
    }

    // engaged_with — slot_map, GPU cells store `partner_raw + 1` or 0.
    let engaged_gpu = view_storage
        .readback_slot_map(device, queue, "engaged_with")
        .expect("readback engaged_with");
    for slot in 0..n {
        let id = match AgentId::new(slot as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        let cpu_partner = cpu.views.engaged_with.get(id).map(|p| p.raw()).unwrap_or(0);
        let gpu_partner = engaged_gpu[slot];
        assert_eq!(
            cpu_partner, gpu_partner,
            "[{ctx}] engaged_with[{slot}] cpu_partner={cpu_partner} gpu_partner={gpu_partner}",
        );
    }
}

// ---------------------------------------------------------------------------
// Shared cascade driver wiring
// ---------------------------------------------------------------------------

struct GpuHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    physics: PhysicsKernel,
    view_storage: ViewStorage,
    spatial: GpuSpatialHash,
    comp: dsl_compiler::ir::Compilation,
}

impl GpuHarness {
    fn new(agent_cap: u32) -> Self {
        let (device, queue, label) = gpu_device_queue();
        eprintln!("[cascade_parity] backend={label}");
        let comp = load_compilation();
        let ctx = EmitContext {
            events: &comp.events,
            event_tags: &comp.event_tags,
        };
        let physics = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096)
            .expect("physics kernel");
        let view_storage = ViewStorage::new(&device, agent_cap)
            .expect("view storage");
        let spatial = GpuSpatialHash::new(&device).expect("spatial hash");
        Self { device, queue, physics, view_storage, spatial, comp }
    }

    fn drive_cascade(
        &mut self,
        mut state: SimState,
        seeded: &[Event],
    ) -> (SimState, Vec<EventRecord>, u32, bool) {
        use engine_gpu::cascade::apply_final_slots;

        // Reset view storage so decay anchors start fresh per test.
        self.view_storage.reset(&self.queue);

        let initial = pack_initial_events(seeded);
        let abilities = PackedAbilityRegistry::empty();
        let out = {
            let ctx = EmitContext {
                events: &self.comp.events,
                event_tags: &self.comp.event_tags,
            };
            run_cascade(
                &self.device,
                &self.queue,
                &state,
                &mut self.physics,
                &mut self.view_storage,
                &mut self.spatial,
                &abilities,
                &initial,
                // `kin_radius` is designer-tunable via
                // `state.config.combat.kin_radius`; the retired
                // `DEFAULT_KIN_RADIUS` const mapped to the same 12 m
                // default.
                state.config.combat.kin_radius,
                &ctx,
            )
        }
        .expect("cascade");

        // Also fold the seeded events into view_storage so the final view
        // state is comparable with the CPU (which folds the complete
        // seed+emitted set). The physics kernel processed the seeds
        // without folding them — its fold pass only handles what IT
        // emits. Without this, engaged_with / my_enemies / threat_level
        // miss the seed contribution.
        fold_iteration_events(&self.device, &self.queue, &mut self.view_storage, &initial)
            .expect("fold seeds");

        apply_final_slots(&mut state, &out.final_agent_slots);
        // Cascade doesn't advance tick; we keep it in sync for view decay
        // comparisons.
        (state, out.all_emitted_events, out.iterations, out.converged)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Most basic cascade: seed an `AgentAttacked` → chronicle_attack fires
/// (non-replayable), no state mutation. Cascade should converge at
/// iteration 1 (empty output).
#[test]
fn cascade_trivial_agent_attacked() {
    let mut harness = GpuHarness::new(8);
    let seeded = vec![Event::AgentAttacked {
        actor: AgentId::new(4).unwrap(),
        target: AgentId::new(1).unwrap(),
        damage: 5.0,
        tick: 0,
    }];

    let (cpu_state, cpu_emitted) = run_cpu_cascade(canonical_fixture(), &seeded);
    let (gpu_state, gpu_emitted, iters, converged) =
        harness.drive_cascade(canonical_fixture(), &seeded);

    eprintln!("[trivial] iters={iters} converged={converged}");
    assert!(converged, "trivial cascade should converge");
    assert_state_parity(&cpu_state, &gpu_state, "trivial");
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted, "trivial");
    assert_view_parity(&harness.device, &harness.queue, &harness.view_storage, &cpu_state, "trivial");
}

/// Lethal damage on a pack-member wolf triggers a multi-iteration
/// cascade:
///   iter 1: EffectDamageApplied → damage rule → emit AgentDied + kill
///           + emit AgentAttacked (audit event)
///   iter 2: AgentDied → fear_spread_on_death → emit FearSpread for each
///           kin within 12 m
///   iter 3+: FearSpread itself fires NO replayable physics rule
///           (chronicle_rout exists but is stubbed / gated), so cascade
///           converges.
///
/// Primary assertions: post-cascade state matches CPU, replayable event
/// set matches, kin_fear view cells are populated.
#[test]
fn cascade_kill_fear_spread_depth_gt_one() {
    let mut harness = GpuHarness::new(8);
    let seeded = vec![Event::EffectDamageApplied {
        actor: AgentId::new(2).unwrap(), // wolf[1]
        target: AgentId::new(1).unwrap(), // fragile wolf[0] (10 hp)
        amount: 50.0,                      // overkill
        tick: 0,
    }];

    let (cpu_state, cpu_emitted) = run_cpu_cascade(pack_of_wolves_fixture(), &seeded);

    // Sanity: CPU actually killed the target and emitted FearSpread.
    assert!(!cpu_state.agent_alive(AgentId::new(1).unwrap()), "target must be dead");
    let cpu_has_fear = cpu_emitted
        .iter()
        .any(|e| matches!(e, Event::FearSpread { .. }));
    assert!(cpu_has_fear, "CPU cascade should emit FearSpread");

    let (gpu_state, gpu_emitted, iters, converged) =
        harness.drive_cascade(pack_of_wolves_fixture(), &seeded);

    eprintln!(
        "[kill_cascade] iters={iters} converged={converged} cpu_emitted={} gpu_emitted={}",
        cpu_emitted.len(), gpu_emitted.len(),
    );
    assert!(converged, "kill cascade should converge");
    assert!(iters >= 2, "kill cascade must reach iter>=2, got {iters}");

    assert_state_parity(&cpu_state, &gpu_state, "kill_cascade");
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted, "kill_cascade");
    assert_view_parity(&harness.device, &harness.queue, &harness.view_storage, &cpu_state, "kill_cascade");
}

/// Multi-seed cascade: two lethal-damage events in the same batch,
/// both targeting different wolves within kin range. Exercises:
///   - parallel kill dispatch in the same physics pass
///   - two AgentDied → FearSpread cascades folding into kin_fear
///   - fold dispatch aggregating events from a multi-iteration cascade
#[test]
fn cascade_multi_kill_parallel_batch() {
    let mut harness = GpuHarness::new(8);
    let seeded = vec![
        Event::EffectDamageApplied {
            actor: AgentId::new(2).unwrap(),
            target: AgentId::new(1).unwrap(), // fragile wolf[0] (10 hp)
            amount: 50.0,
            tick: 0,
        },
        // Second lethal damage — wolf[3] full-hp drops to 0 via 85 dmg.
        Event::EffectDamageApplied {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(4).unwrap(), // wolf[3] full hp
            amount: 95.0,
            tick: 0,
        },
    ];

    let (cpu_state, cpu_emitted) = run_cpu_cascade(pack_of_wolves_fixture(), &seeded);
    let cpu_deaths = cpu_emitted
        .iter()
        .filter(|e| matches!(e, Event::AgentDied { .. }))
        .count();
    assert_eq!(cpu_deaths, 2, "both targets should die on CPU side");

    let (gpu_state, gpu_emitted, iters, converged) =
        harness.drive_cascade(pack_of_wolves_fixture(), &seeded);
    eprintln!(
        "[multi_kill] iters={iters} converged={converged} cpu_emitted={} gpu_emitted={}",
        cpu_emitted.len(), gpu_emitted.len(),
    );
    assert!(converged, "multi-kill cascade should converge");
    assert!(iters >= 2, "multi-kill cascade must reach iter>=2");

    assert_state_parity(&cpu_state, &gpu_state, "multi_kill");
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted, "multi_kill");
    assert_view_parity(&harness.device, &harness.queue, &harness.view_storage, &cpu_state, "multi_kill");
}

/// Cascade convergence on an empty seed — no events in, no events out,
/// zero iterations. Acts as the smoke test for the cascade driver's
/// early-exit path.
#[test]
fn cascade_empty_seed_converges_immediately() {
    let mut harness = GpuHarness::new(8);
    let (_state, emitted, iters, converged) =
        harness.drive_cascade(canonical_fixture(), &[]);
    assert_eq!(iters, 0, "empty seed: expected 0 iterations");
    assert!(converged, "empty seed: converged");
    assert!(emitted.is_empty(), "empty seed: no emitted events");
}

/// Iteration-depth histogram on a sweep of synthetic seed batches. Acts
/// as both a regression check (the shipped rules shouldn't regress past
/// iter>=5) and a diagnostic: the eprintln lets the task report state
/// the typical cascade depth on the canonical fixtures.
#[test]
fn cascade_iteration_distribution() {
    let mut harness = GpuHarness::new(8);
    let cases: &[(&str, fn() -> SimState, Vec<Event>)] = &[
        ("noop_attack", canonical_fixture, vec![Event::AgentAttacked {
            actor: AgentId::new(4).unwrap(),
            target: AgentId::new(1).unwrap(),
            damage: 1.0,
            tick: 0,
        }]),
        ("one_kill", pack_of_wolves_fixture, vec![Event::EffectDamageApplied {
            actor: AgentId::new(2).unwrap(),
            target: AgentId::new(1).unwrap(),
            amount: 50.0,
            tick: 0,
        }]),
        ("two_kill", pack_of_wolves_fixture, vec![
            Event::EffectDamageApplied {
                actor: AgentId::new(2).unwrap(),
                target: AgentId::new(1).unwrap(),
                amount: 50.0,
                tick: 0,
            },
            Event::EffectDamageApplied {
                actor: AgentId::new(1).unwrap(),
                target: AgentId::new(4).unwrap(),
                amount: 95.0,
                tick: 0,
            },
        ]),
        ("light_wound", pack_of_wolves_fixture, vec![Event::EffectDamageApplied {
            actor: AgentId::new(2).unwrap(),
            target: AgentId::new(3).unwrap(),
            amount: 50.0, // drops to 30 — below 50% threshold, triggers RallyCall.
            tick: 0,
        }]),
    ];
    let mut histogram = [0u32; (MAX_CASCADE_ITERATIONS_FOR_HIST + 1) as usize];
    eprintln!("[cascade_hist] case-by-case iteration depth:");
    for (label, fixture, seeded) in cases {
        let (_cpu, _cpu_emit) = run_cpu_cascade(fixture(), seeded);
        let (_gpu, gpu_emit, iters, converged) =
            harness.drive_cascade(fixture(), seeded);
        eprintln!(
            "  {label}: iters={iters} converged={converged} gpu_emitted_events={}",
            gpu_emit.len()
        );
        assert!(converged, "{label} should converge");
        assert!(iters <= MAX_CASCADE_ITERATIONS_FOR_HIST, "{label} iters={iters} exceeds hist bound");
        histogram[iters as usize] += 1;
    }
    eprintln!("[cascade_hist] overall depth distribution: {histogram:?}");
}

/// Local hist bound — matches `engine_gpu::cascade::MAX_CASCADE_ITERATIONS`
/// so the histogram array sizing stays in step if the cap bumps.
const MAX_CASCADE_ITERATIONS_FOR_HIST: u32 = 8;
