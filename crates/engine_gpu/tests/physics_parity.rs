//! Phase 6e — GPU physics kernel parity harness.
//!
//! Runs a hand-crafted event batch through both:
//!   * the CPU cascade (authoritative `CascadeRegistry::dispatch`), and
//!   * the GPU physics kernel (`PhysicsKernel::run_batch`)
//!
//! Then asserts both produced the same `SimState` mutations and the
//! same new-event set. The batch is a single cascade iteration: two
//! `AgentAttacked` events. Each fires:
//!   * chronicle_attack  (stubbed on GPU — emits ChronicleEntry, non-
//!     replayable, ignored by parity)
//!   * chronicle_wound   (same, also gated on hp_pct < 0.5)
//!   * rally_on_wound    (emits RallyCall if target.hp_pct < 0.5)
//!
//! This keeps the surface small enough to debug end-to-end while still
//! exercising state mutation (none for AgentAttacked itself — it's an
//! audit event; no direct hp change) plus event emission (via wound
//! chronicle and rally). Piece 3 will cascade these further.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::pack_event;
use engine_gpu::physics::{
    pack_agent_slots, unpack_agent_slots, GpuKinList, PackedAbilityRegistry, PhysicsCfg,
    PhysicsKernel,
};
use glam::Vec3;

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
                label: Some("physics_parity::device"),
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

/// Spawn a small mixed fixture: 2 humans, 2 wolves, all with hp above
/// the 50%-wound threshold so `rally_on_wound` doesn't fire (keeps the
/// parity window narrow for the first integration pass).
fn build_fixture() -> SimState {
    let mut state = SimState::new(8, 0xDEAD_BEEF);
    for i in 0..2 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32 * 5.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
    }
    for i in 0..2 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(50.0 + i as f32 * 5.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("wolf spawn");
    }
    state
}

/// Load the physics rules in their canonical `PhysicsIR` form from
/// `assets/sim/`. Mirrors the coverage test's loader.
fn load_physics_and_events() -> dsl_compiler::ir::Compilation {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop(); // crates/
    root.pop(); // repo root
    root.push("assets/sim");

    let mut merged = Program { decls: Vec::new() };
    for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
        let src = std::fs::read_to_string(root.join(f)).unwrap_or_else(|e| {
            panic!("read {}: {e}", f);
        });
        merged.decls.extend(dsl_compiler::parse(&src).unwrap().decls);
    }
    dsl_compiler::compile_ast(merged).expect("resolve")
}

/// Run one physics iteration on a list of seeded events — CPU + GPU —
/// and return both sides' final state + emitted replayable events.
fn run_one_iteration(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    comp: &dsl_compiler::ir::Compilation,
    seeded: &[Event],
) -> (SimState, SimState, Vec<Event>, Vec<Event>) {
    // CPU side.
    let mut cpu_state = build_fixture();
    let mut cpu_events = EventRing::with_cap(4096);
    let registry = CascadeRegistry::with_engine_builtins();
    for e in seeded {
        cpu_events.push(e.clone());
    }
    let pre_total = cpu_events.total_pushed();
    for idx in cpu_events.dispatched()..pre_total {
        if let Some(e) = cpu_events.get_pushed(idx) {
            registry.dispatch(&e, &mut cpu_state, &mut cpu_events);
        }
    }
    cpu_events.set_dispatched(pre_total);
    let mut cpu_emitted: Vec<Event> = Vec::new();
    for idx in pre_total..cpu_events.total_pushed() {
        if let Some(e) = cpu_events.get_pushed(idx) {
            cpu_emitted.push(e);
        }
    }

    // GPU side.
    let mut gpu_state = build_fixture();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel =
        PhysicsKernel::new(device, &comp.physics, &ctx, 4096).expect("physics kernel init");
    let agent_slots_in = pack_agent_slots(&gpu_state);
    let empty_abilities = PackedAbilityRegistry::empty();
    let agent_cap = gpu_state.agent_cap();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    let cfg = PhysicsCfg {
        tick: gpu_state.tick,
        num_events: 0,
        combat_engagement_range: gpu_state.config.combat.engagement_range,
        cascade_max_iterations: 8,
        agent_cap,
        max_abilities: engine_gpu::physics::MAX_ABILITIES as u32,
        max_effects: engine_gpu::physics::MAX_EFFECTS as u32,
        _pad: 0,
    };
    let events_in: Vec<_> = seeded.iter().filter_map(pack_event).collect();
    let out = kernel
        .run_batch(
            device,
            queue,
            &agent_slots_in,
            &empty_abilities,
            &kin_lists,
            &nearest_hostile,
            &events_in,
            cfg,
        )
        .expect("run_batch");
    assert!(
        !out.drain.overflowed,
        "event ring overflowed: drain={:?}",
        out.drain
    );
    unpack_agent_slots(&mut gpu_state, &out.agent_slots_out);
    let gpu_emitted: Vec<Event> = out
        .events_out
        .iter()
        .filter_map(engine_gpu::event_ring::unpack_record)
        .collect();
    (cpu_state, gpu_state, cpu_emitted, gpu_emitted)
}

/// Assert state byte-exact on every SimState slot for the fields the
/// physics kernel updates.
fn assert_state_parity(cpu: &SimState, gpu: &SimState) {
    let cap = cpu.agent_cap();
    for slot in 0..cap {
        let id = AgentId::new(slot + 1).unwrap();
        assert_eq!(
            cpu.agent_hp(id).map(f32::to_bits),
            gpu.agent_hp(id).map(f32::to_bits),
            "hp mismatch slot {slot}: cpu={:?} gpu={:?}",
            cpu.agent_hp(id),
            gpu.agent_hp(id),
        );
        assert_eq!(
            cpu.agent_alive(id),
            gpu.agent_alive(id),
            "alive mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_shield_hp(id).map(f32::to_bits),
            gpu.agent_shield_hp(id).map(f32::to_bits),
            "shield mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_stun_expires_at(id),
            gpu.agent_stun_expires_at(id),
            "stun mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_slow_expires_at(id),
            gpu.agent_slow_expires_at(id),
            "slow-expiry mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_slow_factor_q8(id),
            gpu.agent_slow_factor_q8(id),
            "slow-factor mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_cooldown_next_ready(id),
            gpu.agent_cooldown_next_ready(id),
            "cooldown mismatch slot {slot}",
        );
        assert_eq!(
            cpu.agent_engaged_with(id),
            gpu.agent_engaged_with(id),
            "engaged_with mismatch slot {slot}",
        );
    }
}

/// Normalise + compare emitted event lists (replayable subset only).
/// Sort key mirrors `GpuEventRing::drain` + keeps the CPU side comparable.
fn assert_replayable_events_match(cpu: &[Event], gpu: &[Event]) {
    let mut cpu_r: Vec<_> = cpu.iter().filter(|e| e.is_replayable()).cloned().collect();
    let mut gpu_r: Vec<_> = gpu.iter().filter(|e| e.is_replayable()).cloned().collect();
    let key = |e: &Event| {
        let r = pack_event(e).expect("pack");
        (r.tick, r.kind, r.payload[0], r.payload[1])
    };
    cpu_r.sort_by_key(key);
    gpu_r.sort_by_key(key);
    assert_eq!(
        cpu_r.len(),
        gpu_r.len(),
        "replayable count mismatch:\ncpu={cpu_r:?}\ngpu={gpu_r:?}",
    );
    assert_eq!(
        cpu_r, gpu_r,
        "replayable contents mismatch:\ncpu={cpu_r:?}\ngpu={gpu_r:?}",
    );
}

#[test]
fn physics_batch_parity_agent_attacked() {
    // Two AgentAttacked events. Each fires chronicle_attack,
    // chronicle_wound (gated on hp_pct < 0.5 — doesn't fire with our
    // 100-hp fixture), and rally_on_wound (same gate).
    //
    // With targets at full HP none of the wound-gated rules fire —
    // so the only emitted events are two ChronicleEntry events (non-
    // replayable). The replayable-subset comparison should pass with
    // both sides emitting zero replayable events.
    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[physics_parity attacked] backend={backend_label}");
    let comp = load_physics_and_events();
    let tick = 0u32;
    let seeded = vec![
        Event::AgentAttacked {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(3).unwrap(),
            damage: 5.0,
            tick,
        },
        Event::AgentAttacked {
            actor: AgentId::new(2).unwrap(),
            target: AgentId::new(4).unwrap(),
            damage: 7.5,
            tick,
        },
    ];
    let (cpu_state, gpu_state, cpu_emitted, gpu_emitted) =
        run_one_iteration(&device, &queue, &comp, &seeded);
    assert_state_parity(&cpu_state, &gpu_state);
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted);
}

#[test]
fn physics_batch_parity_effect_damage_applied() {
    // Two EffectDamageApplied events → each calls the `damage` rule:
    //   * absorb via shield (no shield here — skips that branch)
    //   * emit AgentAttacked
    //   * subtract damage from hp, clamp at 0
    //   * if hp <= 0, emit AgentDied + kill agent
    //
    // With 100 hp + 5 damage, no-one dies, and each event mutates one
    // target's hp to 95.0. Both sides should agree byte-exact.
    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[physics_parity damage] backend={backend_label}");
    let comp = load_physics_and_events();
    let tick = 0u32;
    let seeded = vec![
        Event::EffectDamageApplied {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(3).unwrap(),
            amount: 5.0,
            tick,
        },
        Event::EffectDamageApplied {
            actor: AgentId::new(2).unwrap(),
            target: AgentId::new(4).unwrap(),
            amount: 7.5,
            tick,
        },
    ];
    let (cpu_state, gpu_state, cpu_emitted, gpu_emitted) =
        run_one_iteration(&device, &queue, &comp, &seeded);

    // Sanity — the damage actually landed on both sides.
    let target_3 = AgentId::new(3).unwrap();
    let target_4 = AgentId::new(4).unwrap();
    assert_eq!(cpu_state.agent_hp(target_3), Some(95.0));
    assert_eq!(cpu_state.agent_hp(target_4), Some(92.5));

    assert_state_parity(&cpu_state, &gpu_state);
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted);
}

#[test]
fn physics_batch_parity_effect_heal_applied() {
    // Heal doesn't fire when target is at full HP — the rule body
    // reads `agents.max_hp(t)` and clamps `hp + amount` to it. With
    // 100 / 100 hp both sides become 100 again (no-op).
    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[physics_parity heal] backend={backend_label}");
    let comp = load_physics_and_events();
    let tick = 0u32;
    let seeded = vec![Event::EffectHealApplied {
        actor: AgentId::new(1).unwrap(),
        target: AgentId::new(3).unwrap(),
        amount: 10.0,
        tick,
    }];
    let (cpu_state, gpu_state, cpu_emitted, gpu_emitted) =
        run_one_iteration(&device, &queue, &comp, &seeded);
    assert_state_parity(&cpu_state, &gpu_state);
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted);
}

#[test]
fn physics_batch_parity_effect_shield_stun_slow() {
    // Exercise shield / stun / slow rules in one batch.
    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[physics_parity shield/stun/slow] backend={backend_label}");
    let comp = load_physics_and_events();
    let tick = 0u32;
    let seeded = vec![
        Event::EffectShieldApplied {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(3).unwrap(),
            amount: 25.0,
            tick,
        },
        Event::EffectStunApplied {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(3).unwrap(),
            expires_at_tick: 50,
            tick,
        },
        Event::EffectSlowApplied {
            actor: AgentId::new(1).unwrap(),
            target: AgentId::new(4).unwrap(),
            expires_at_tick: 40,
            factor_q8: 128,
            tick,
        },
    ];
    let (cpu_state, gpu_state, cpu_emitted, gpu_emitted) =
        run_one_iteration(&device, &queue, &comp, &seeded);

    let t3 = AgentId::new(3).unwrap();
    let t4 = AgentId::new(4).unwrap();
    assert_eq!(cpu_state.agent_shield_hp(t3), Some(25.0));
    assert_eq!(cpu_state.agent_stun_expires_at(t3), Some(50));
    assert_eq!(cpu_state.agent_slow_expires_at(t4), Some(40));
    assert_eq!(cpu_state.agent_slow_factor_q8(t4), Some(128));

    assert_state_parity(&cpu_state, &gpu_state);
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted);
}

#[test]
fn physics_batch_parity_lethal_damage() {
    // Damage larger than target HP — triggers the kill path:
    // emit AgentDied + kill_agent (sets alive=false).
    //
    // CPU cascade emits AgentDied, which feeds into
    // engagement_on_death + fear_spread_on_death. The GPU kernel only
    // runs ONE iteration, so the AgentDied gets emitted but its
    // cascade-children don't fire (that's Piece 3's job). The
    // replayable emitted set matches if we only compare the first
    // cascade pass — which is what our parity helper does (it calls
    // dispatch on the seed events, not run_fixed_point).
    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[physics_parity lethal] backend={backend_label}");
    let comp = load_physics_and_events();
    let tick = 0u32;
    let seeded = vec![Event::EffectDamageApplied {
        actor: AgentId::new(1).unwrap(),
        target: AgentId::new(3).unwrap(),
        amount: 150.0, // overshoots 100 hp
        tick,
    }];
    let (cpu_state, gpu_state, cpu_emitted, gpu_emitted) =
        run_one_iteration(&device, &queue, &comp, &seeded);

    let t3 = AgentId::new(3).unwrap();
    assert!(!cpu_state.agent_alive(t3), "CPU: target should be dead");
    // HP field is `Some(0.0)` after the kill — `agent_hp` doesn't
    // filter by alive. The alive flag is the authoritative signal.
    assert_eq!(cpu_state.agent_hp(t3), Some(0.0));

    assert_state_parity(&cpu_state, &gpu_state);
    assert_replayable_events_match(&cpu_emitted, &gpu_emitted);
}
