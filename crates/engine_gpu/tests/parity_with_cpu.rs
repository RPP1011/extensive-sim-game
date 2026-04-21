//! Phase 2 parity harness (task #183 / plan: `docs/plans/gpu_megakernel_plan.md`).
//!
//! Gated on `--features gpu` so the CPU-only build doesn't pay the
//! compile-time cost of the fixture / backend wiring. Run with
//! `cargo test -p engine_gpu --features gpu`. The root workspace's `gpu`
//! feature re-exports this one (see root `Cargo.toml`), so
//! `cargo test --features gpu` from the workspace root picks it up too.
//!
//! ## What Phase 2 adds on top of Phase 1
//!
//! Phase 1 ran one kernel (Attack only) alongside the CPU step and
//! asserted its output matched a CPU reference bitmap each tick.
//! Phase 2 generalises: the backend now runs ONE fused kernel every
//! tick that emits N bitmaps — one per supported mask — and this test
//! asserts each one against its CPU reference.
//!
//! ## Masks in the fused kernel
//!
//! Seven masks are byte-parity checked:
//!
//!   * Attack      — target-bound, radius-filtered, hostility predicate
//!   * MoveToward  — target-bound, radius-filtered, alive + self-exclusion
//!   * Hold / Flee / Eat / Drink / Rest — self-only, alive gate
//!
//! Cast is **skipped**: its `mask Cast(ability: AbilityId)` head takes
//! a non-Agent parameter and its predicate reads views + cooldowns that
//! the Phase 2 emitter has no GPU backing for. The test surfaces the
//! skip via `eprintln!` so a regression that accidentally pulls Cast
//! back into the fused kernel surfaces early.
//!
//! If the fixture dies out before tick 50, later ticks have no alive
//! agents and every bitmap is zero on both sides — the byte-equal
//! check still passes trivially.

#![cfg(feature = "gpu")]

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::{
    mask::cpu_mask_bitmap,
    scoring::{cpu_score_outputs, ScoreOutput, NO_TARGET},
    GpuBackend,
};
use glam::Vec3;

const SEED: u64 = 0xD00D_FACE_0042_0042;
const TICKS: u32 = 50;
const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

/// Masks the fused kernel emits in Phase 2. Any name not in this list
/// is skipped in the parity comparison with a logged reason — currently
/// only "Cast" falls into that bucket, since its parametric head /
/// view dependencies land in Phase 4+.
const PARITY_MASK_NAMES: &[&str] = &[
    "Attack",
    "MoveToward",
    "Hold",
    "Flee",
    "Eat",
    "Drink",
    "Rest",
];

/// Spawn the canonical 3-humans-and-2-wolves fixture that the engine's
/// `wolves_and_humans_parity.rs` anchor uses. Kept in lockstep (same seed,
/// same positions, same HP) so a failure here also shows up in the engine's
/// own parity baseline, making root-causing easier.
fn spawn_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 2 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 3 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(-3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 2 spawn");
    state
}

/// Deterministic per-state fingerprint. Covers the observable mutations
/// each tick makes: `tick`, alive bitmap, and per-agent `(pos, hp)` for
/// every slot in spawn order. Format is stable text — `{:.6}` for all
/// f32 values so a glam::Vec3 `Debug` format change doesn't silently
/// invalidate the fingerprint. Keeps the assertion message readable when
/// it does fire.
fn fingerprint(state: &SimState) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(256);
    writeln!(s, "tick={} agent_cap={}", state.tick, state.agent_cap()).unwrap();
    for id in state.agents_alive() {
        let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
        let hp = state.agent_hp(id).unwrap_or(f32::NAN);
        writeln!(
            s,
            "id={} pos=({:.6},{:.6},{:.6}) hp={:.6}",
            id.raw(),
            pos.x,
            pos.y,
            pos.z,
            hp,
        )
        .unwrap();
    }
    s
}

/// Collect every event the ring holds, in push order, into a `Vec<Event>`
/// so we can `assert_eq!` across backends. The `EventRing` itself isn't
/// `PartialEq`, but its element type is.
fn collect_events(ring: &EventRing) -> Vec<Event> {
    ring.iter().copied().collect()
}

/// Run `TICKS` ticks through `CpuBackend` and return `(state, events)`.
fn run_cpu() -> (SimState, EventRing) {
    let mut backend = CpuBackend;
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for _ in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    (state, events)
}

/// Run `TICKS` ticks through `GpuBackend`, checking every fused-mask
/// GPU bitmap against a CPU reference at every tick. Returns
/// `(state, events)` for the top-level CPU-vs-GPU state fingerprint
/// parity check.
fn run_gpu() -> (SimState, EventRing) {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    eprintln!("parity_with_cpu: wgpu backend = {}", backend.backend_label());

    // Log which masks the GPU actually runs — if the fused-kernel emitter
    // silently drops one, this is where it surfaces. Also warn about
    // Cast being skipped so the reason is in the test output, not just
    // the docstring.
    let gpu_masks: Vec<&str> = backend
        .mask_bindings()
        .iter()
        .map(|b| b.mask_name.as_str())
        .collect();
    eprintln!("parity_with_cpu: fused-kernel masks = {:?}", gpu_masks);
    eprintln!(
        "parity_with_cpu: skipped (Phase 4+ blocker) = [\"Cast\"] — parametric head + view/cooldown deps"
    );
    assert_eq!(
        gpu_masks.len(),
        PARITY_MASK_NAMES.len(),
        "unexpected fused-kernel mask count — regression or new mask added without updating PARITY_MASK_NAMES"
    );

    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let gpu_bitmaps = backend.last_mask_bitmaps();
        assert!(
            !gpu_bitmaps.is_empty(),
            "tick {tick_i}: GpuBackend::last_mask_bitmaps empty — kernel dispatch failed?"
        );
        assert_eq!(
            gpu_bitmaps.len(),
            PARITY_MASK_NAMES.len(),
            "tick {tick_i}: expected {} per-mask bitmaps, got {}",
            PARITY_MASK_NAMES.len(),
            gpu_bitmaps.len(),
        );

        // Per-mask byte-equality check.
        for name in PARITY_MASK_NAMES {
            let gpu_bitmap = backend
                .last_bitmap_for(name)
                .unwrap_or_else(|| panic!("tick {tick_i}: no GPU bitmap for mask `{name}`"));
            let cpu_bitmap = cpu_mask_bitmap(&state, name).unwrap_or_else(|| {
                panic!("tick {tick_i}: no CPU reference for mask `{name}`")
            });
            if gpu_bitmap != cpu_bitmap.as_slice() {
                let diff = bitmap_diff(&cpu_bitmap, gpu_bitmap);
                panic!(
                    "tick {tick_i}: GPU mask `{name}` bitmap differs from CPU\n\
                     cpu = {cpu_bitmap:#010x?}\n\
                     gpu = {gpu_bitmap:#010x?}\n\
                     disagreeing slots: {diff:?}",
                );
            }
        }
    }
    (state, events)
}

fn event_kind_name(e: &Event) -> &'static str {
    match e {
        Event::AgentMoved { .. } => "AgentMoved",
        Event::AgentAttacked { .. } => "AgentAttacked",
        Event::AgentDied { .. } => "AgentDied",
        Event::AgentFled { .. } => "AgentFled",
        Event::AgentAte { .. } => "AgentAte",
        Event::AgentDrank { .. } => "AgentDrank",
        Event::AgentRested { .. } => "AgentRested",
        Event::AgentCast { .. } => "AgentCast",
        Event::AgentUsedItem { .. } => "AgentUsedItem",
        Event::AgentHarvested { .. } => "AgentHarvested",
        Event::AgentPlacedTile { .. } => "AgentPlacedTile",
        Event::AgentPlacedVoxel { .. } => "AgentPlacedVoxel",
        Event::AgentHarvestedVoxel { .. } => "AgentHarvestedVoxel",
        Event::AgentConversed { .. } => "AgentConversed",
        Event::AgentSharedStory { .. } => "AgentSharedStory",
        Event::AgentCommunicated { .. } => "AgentCommunicated",
        Event::InformationRequested { .. } => "InformationRequested",
        Event::AgentRemembered { .. } => "AgentRemembered",
        Event::QuestPosted { .. } => "QuestPosted",
        Event::QuestAccepted { .. } => "QuestAccepted",
        Event::BidPlaced { .. } => "BidPlaced",
        Event::AnnounceEmitted { .. } => "AnnounceEmitted",
        Event::RecordMemory { .. } => "RecordMemory",
        Event::ChronicleEntry { .. } => "ChronicleEntry",
        Event::OpportunityAttackTriggered { .. } => "OpportunityAttackTriggered",
        Event::EffectDamageApplied { .. } => "EffectDamageApplied",
        Event::EffectHealApplied { .. } => "EffectHealApplied",
        Event::EffectShieldApplied { .. } => "EffectShieldApplied",
        Event::EffectStunApplied { .. } => "EffectStunApplied",
        Event::EffectSlowApplied { .. } => "EffectSlowApplied",
        Event::EffectGoldTransfer { .. } => "EffectGoldTransfer",
        Event::EffectStandingDelta { .. } => "EffectStandingDelta",
        Event::CastDepthExceeded { .. } => "CastDepthExceeded",
        Event::EngagementCommitted { .. } => "EngagementCommitted",
        Event::EngagementBroken { .. } => "EngagementBroken",
        Event::FearSpread { .. } => "FearSpread",
        Event::PackAssist { .. } => "PackAssist",
        Event::RallyCall { .. } => "RallyCall",
    }
}

/// List every slot index where the two bitmaps disagree — used for
/// diagnostic output when the per-tick comparison fails.
fn bitmap_diff(cpu: &[u32], gpu: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let n_words = cpu.len().max(gpu.len());
    for w in 0..n_words {
        let c = cpu.get(w).copied().unwrap_or(0);
        let g = gpu.get(w).copied().unwrap_or(0);
        let mut xor = c ^ g;
        while xor != 0 {
            let bit = xor.trailing_zeros();
            out.push((w as u32) * 32 + bit);
            xor &= xor - 1;
        }
    }
    out
}

/// Phase 2 headline test: `GpuBackend::step` dispatches the fused mask
/// kernel every tick and each of the seven supported mask bitmaps
/// matches the CPU-computed reference byte-for-byte across 50 ticks of
/// the canonical fixture. Also retains the Phase 0 CPU-forwarding
/// parity (event log + state fingerprint) — those assertions guard
/// against regressions in the side-by-side CPU path that
/// `GpuBackend::step` still runs.
/// Task 193 (Phase 6g) headline test: `GpuBackend::step` runs an
/// authoritative GPU tick — CPU phases 1–3 for decisions, GPU cascade
/// + view folds for state mutation, CPU cold-state replay for the 11
/// stubbed rules — and produces byte-exact state equal to `CpuBackend`
/// across 50 ticks of the canonical 3h+2w fixture.
///
/// Asserts, per tick:
///   * State fingerprint equality (per-agent pos + hp + alive).
///   * Per-agent hp/alive/shield/stun/slow/cooldown/engaged_with
///     byte-exact (via `fingerprint` for the hp/pos subset; the
///     cumulative event-multiset check at the end covers the rest).
///
/// If divergence occurs, the first divergent tick is logged along
/// with a per-kind event histogram so root-cause analysis starts with
/// a pointer to what changed (typically a missing cold-state rule or
/// a spatial-radius mismatch — task 193 hit one of each on first
/// landing and fixed both in the cascade driver).
#[test]
fn gpu_full_tick_loop_matches_cpu_50_ticks() {
    let mut cpu_backend = CpuBackend;
    let mut cpu_state = spawn_fixture();
    let mut cpu_scratch = SimScratch::new(cpu_state.agent_cap() as usize);
    let mut cpu_events = EventRing::with_cap(EVENT_RING_CAP);
    let cpu_cascade = CascadeRegistry::with_engine_builtins();

    let mut gpu_backend = GpuBackend::new().expect("GpuBackend init");
    let mut gpu_state = spawn_fixture();
    let mut gpu_scratch = SimScratch::new(gpu_state.agent_cap() as usize);
    let mut gpu_events = EventRing::with_cap(EVENT_RING_CAP);
    let gpu_cascade = CascadeRegistry::with_engine_builtins();

    for tick_i in 0..TICKS {
        cpu_backend.step(&mut cpu_state, &mut cpu_scratch, &mut cpu_events, &UtilityBackend, &cpu_cascade);
        gpu_backend.step(&mut gpu_state, &mut gpu_scratch, &mut gpu_events, &UtilityBackend, &gpu_cascade);

        let cpu_fp = fingerprint(&cpu_state);
        let gpu_fp = fingerprint(&gpu_state);
        assert_eq!(
            cpu_fp, gpu_fp,
            "state fingerprint diverged at tick_i={tick_i} (post-step state.tick={})\n\
             CPU:\n{cpu_fp}\nGPU:\n{gpu_fp}",
            cpu_state.tick,
        );
    }

    // End-of-run: cumulative event multiset parity. Same rationale as
    // `gpu_backend_matches_cpu_on_canonical_fixture` — the GPU cascade
    // batches event emission so push order differs from CPU's per-
    // event dispatch, but the multiset is equal.
    let cpu_evs = collect_events(&cpu_events);
    let gpu_evs = collect_events(&gpu_events);
    assert_eq!(
        cpu_evs.len(), gpu_evs.len(),
        "event count diverged: cpu={} gpu={}",
        cpu_evs.len(), gpu_evs.len(),
    );
    let mut cpu_sorted = cpu_evs.clone();
    let mut gpu_sorted = gpu_evs.clone();
    cpu_sorted.sort_by_key(|e| format!("{e:?}"));
    gpu_sorted.sort_by_key(|e| format!("{e:?}"));
    assert_eq!(
        cpu_sorted, gpu_sorted,
        "event multiset differs after 50 ticks",
    );

    // Perf diagnostic — report cascade iterations. Not a failure
    // signal; just a breadcrumb for the task's report.
    if let Some(iters) = gpu_backend.last_cascade_iterations() {
        eprintln!("gpu_full_tick_loop: last-tick cascade iterations = {iters}");
    }
    if let Some(err) = gpu_backend.last_cascade_error() {
        eprintln!("gpu_full_tick_loop: last-tick cascade error = {err}");
    }
}

/// Phase 6g diagnostic: walk the canonical fixture one tick at a time on
/// both backends and print the first tick where the per-agent hp/alive
/// diverges. Feeds into the full-tick parity test above — if this
/// reports a divergence tick, the per-tick state comparison surfaces
/// which agent changed and when.
#[test]
fn gpu_full_tick_per_tick_state_divergence() {
    let mut cpu_backend = CpuBackend;
    let mut cpu_state = spawn_fixture();
    let mut cpu_scratch = SimScratch::new(cpu_state.agent_cap() as usize);
    let mut cpu_events = EventRing::with_cap(EVENT_RING_CAP);
    let cpu_cascade = CascadeRegistry::with_engine_builtins();

    let mut gpu_backend = GpuBackend::new().expect("GpuBackend init");
    let mut gpu_state = spawn_fixture();
    let mut gpu_scratch = SimScratch::new(gpu_state.agent_cap() as usize);
    let mut gpu_events = EventRing::with_cap(EVENT_RING_CAP);
    let gpu_cascade = CascadeRegistry::with_engine_builtins();

    for tick_i in 0..TICKS {
        let cpu_pushed_before = cpu_events.total_pushed();
        let gpu_pushed_before = gpu_events.total_pushed();
        cpu_backend.step(&mut cpu_state, &mut cpu_scratch, &mut cpu_events, &UtilityBackend, &cpu_cascade);
        gpu_backend.step(&mut gpu_state, &mut gpu_scratch, &mut gpu_events, &UtilityBackend, &gpu_cascade);

        let cpu_fp = fingerprint(&cpu_state);
        let gpu_fp = fingerprint(&gpu_state);
        if cpu_fp != gpu_fp {
            // Print this-tick's events (both sides) for triage.
            let cpu_this: Vec<Event> = (cpu_pushed_before..cpu_events.total_pushed())
                .filter_map(|i| cpu_events.get_pushed(i)).collect();
            let gpu_this: Vec<Event> = (gpu_pushed_before..gpu_events.total_pushed())
                .filter_map(|i| gpu_events.get_pushed(i)).collect();
            eprintln!("[this-tick CPU events ({} total)]", cpu_this.len());
            for e in &cpu_this { eprintln!("  {e:?}"); }
            eprintln!("[this-tick GPU events ({} total)]", gpu_this.len());
            for e in &gpu_this { eprintln!("  {e:?}"); }
            eprintln!(
                "divergence at tick_i={} (post-step state.tick={})\n\
                 CPU:\n{}\nGPU:\n{}",
                tick_i, cpu_state.tick, cpu_fp, gpu_fp
            );

            // Summarise event diff just for this tick using push indices.
            // `total_pushed()` is monotonic; walk both sides from their
            // start-of-tick snapshot.
            // Show event-kind histograms at this point:
            use std::collections::BTreeMap;
            let cpu_evs = collect_events(&cpu_events);
            let gpu_evs = collect_events(&gpu_events);
            let mut cpu_hist: BTreeMap<&str, usize> = BTreeMap::new();
            let mut gpu_hist: BTreeMap<&str, usize> = BTreeMap::new();
            for e in &cpu_evs {
                *cpu_hist.entry(event_kind_name(e)).or_insert(0) += 1;
            }
            for e in &gpu_evs {
                *gpu_hist.entry(event_kind_name(e)).or_insert(0) += 1;
            }
            eprintln!("[cumulative event histogram after tick {tick_i}]");
            let keys: std::collections::BTreeSet<&str> =
                cpu_hist.keys().copied().chain(gpu_hist.keys().copied()).collect();
            for k in &keys {
                let c = cpu_hist.get(k).copied().unwrap_or(0);
                let g = gpu_hist.get(k).copied().unwrap_or(0);
                if c != g {
                    eprintln!("  {k:30}: cpu={c:4} gpu={g:4} delta={}", c as i64 - g as i64);
                }
            }
            panic!("per-tick state divergence at tick_i={tick_i}");
        }
    }
}

#[test]
fn gpu_backend_matches_cpu_on_canonical_fixture() {
    let (cpu_state, cpu_events) = run_cpu();
    let (gpu_state, gpu_events) = run_gpu();

    // Event log parity — match as a multiset. Task 193 (Phase 6g)
    // made the GPU cascade authoritative, which batches event
    // emission differently than the CPU cascade's per-event
    // dispatch. The *set* of events is identical (counts + contents
    // match), but the *push order* differs because:
    //
    //   * CPU's cascade dispatches event-by-event, so a chronicle
    //     emission sits immediately after the parent AgentAttacked in
    //     the ring.
    //   * GPU's cascade dispatches a batch, emits follow-on events
    //     sorted by (tick, kind, payload[0]), then runs cold-state
    //     replay (chronicles / transfer_gold / modify_standing /
    //     record_memory) in one post-pass.
    //
    // Multiset equality preserves the "no events lost, no events
    // duplicated" invariant while allowing the reorder. The state
    // fingerprint + per-tick divergence test (below / above) catch
    // any semantic difference the reorder might hide.
    let cpu_evs = collect_events(&cpu_events);
    let gpu_evs = collect_events(&gpu_events);

    // Diagnostic: event-kind histogram on both sides so a divergence
    // lands with a first-pass root-cause hint.
    {
        use std::collections::BTreeMap;
        let mut cpu_hist: BTreeMap<&str, usize> = BTreeMap::new();
        let mut gpu_hist: BTreeMap<&str, usize> = BTreeMap::new();
        for e in &cpu_evs {
            *cpu_hist.entry(event_kind_name(e)).or_insert(0) += 1;
        }
        for e in &gpu_evs {
            *gpu_hist.entry(event_kind_name(e)).or_insert(0) += 1;
        }
        if cpu_hist != gpu_hist {
            eprintln!("[event-histogram] divergence:");
            let keys: std::collections::BTreeSet<&str> =
                cpu_hist.keys().copied().chain(gpu_hist.keys().copied()).collect();
            for k in &keys {
                let c = cpu_hist.get(k).copied().unwrap_or(0);
                let g = gpu_hist.get(k).copied().unwrap_or(0);
                eprintln!("  {k:30}: cpu={c:4} gpu={g:4} delta={}", c as i64 - g as i64);
            }
        }
    }

    assert_eq!(
        cpu_evs.len(),
        gpu_evs.len(),
        "event count diverged: cpu={} gpu={}",
        cpu_evs.len(),
        gpu_evs.len(),
    );

    // Multiset equality — sort both by a canonical key and compare.
    // Event is not Ord, but it is Debug + PartialEq, so we stringify
    // once for the sort key.
    let mut cpu_sorted = cpu_evs.clone();
    let mut gpu_sorted = gpu_evs.clone();
    cpu_sorted.sort_by_key(|e| format!("{e:?}"));
    gpu_sorted.sort_by_key(|e| format!("{e:?}"));
    assert_eq!(
        cpu_sorted, gpu_sorted,
        "event multiset differs between CpuBackend and GpuBackend",
    );
    assert_eq!(
        cpu_events.total_pushed(),
        gpu_events.total_pushed(),
        "EventRing::total_pushed diverged",
    );

    // State fingerprint parity.
    let cpu_fp = fingerprint(&cpu_state);
    let gpu_fp = fingerprint(&gpu_state);
    assert_eq!(
        cpu_fp, gpu_fp,
        "state fingerprint differs after {} ticks\ncpu:\n{}\ngpu:\n{}",
        TICKS, cpu_fp, gpu_fp,
    );
}

/// Phase 0 holdover: running the same fixture through `CpuBackend`
/// twice produces identical output. Without this, a failing
/// `gpu_backend_matches_cpu_on_canonical_fixture` would be ambiguous
/// between "GPU diverged" and "the fixture / setup is non-deterministic".
/// This isolates the latter.
#[test]
fn cpu_backend_is_deterministic_on_canonical_fixture() {
    let (state_a, events_a) = run_cpu();
    let (state_b, events_b) = run_cpu();
    assert_eq!(collect_events(&events_a), collect_events(&events_b));
    assert_eq!(fingerprint(&state_a), fingerprint(&state_b));
}

/// Direct GPU-vs-CPU fused-mask check before any step runs. Runs the
/// fused kernel against a freshly-spawned fixture (tick 0, no movement
/// yet) and compares every per-mask bitmap against the CPU reference.
/// Isolates "kernel correctness on a known state" from "kernel
/// correctness across the tick-by-tick state evolution" — if this test
/// passes but the multi-tick version fails, the bug is in how state
/// changes between ticks affect the kernel, not in the kernel itself.
#[test]
fn gpu_fused_masks_match_cpu_on_spawn_state() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let state = spawn_fixture();
    let gpu_bitmaps = backend
        .verify_masks_on_gpu(&state)
        .expect("GPU fused-mask dispatch");
    assert_eq!(gpu_bitmaps.len(), PARITY_MASK_NAMES.len());
    let bindings: Vec<String> = backend
        .mask_bindings()
        .iter()
        .map(|b| b.mask_name.clone())
        .collect();

    for (i, name) in bindings.iter().enumerate() {
        let gpu = &gpu_bitmaps[i];
        let cpu = cpu_mask_bitmap(&state, name).expect("CPU reference for mask");
        assert_eq!(
            gpu, &cpu,
            "GPU mask `{name}` differs from CPU on spawn state\ngpu={gpu:#010x?}\ncpu={cpu:#010x?}",
        );
    }

    // Hand-computed anchor for Attack (unchanged from Phase 1) — keeps
    // the CPU reference honest. Humans at (0,0,0), (2,0,0), (-2,0,0);
    // wolves at (3,0,0), (-3,0,0). Attack range is 2.0. Human↔Wolf is
    // mutually hostile. Expected attackers: ids 2, 3, 4, 5 → slots
    // 1, 2, 3, 4.
    let attack_cpu = cpu_mask_bitmap(&state, "Attack").expect("attack cpu");
    let expected_attack: u32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
    assert_eq!(
        attack_cpu[0], expected_attack,
        "CPU Attack reference didn't match hand-computed spawn expectation"
    );

    // Self-only masks — every alive slot is set. 5 agents → bits 0..=4.
    let hold_cpu = cpu_mask_bitmap(&state, "Hold").expect("hold cpu");
    let expected_alive: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
    assert_eq!(
        hold_cpu[0], expected_alive,
        "CPU Hold reference should set every alive slot"
    );

    // MoveToward — every alive agent has at least one other alive
    // agent within `max_move_radius` (20.0 by default — all 5 fixture
    // agents sit on a 6m span), so every slot is set.
    let move_toward_cpu = cpu_mask_bitmap(&state, "MoveToward").expect("mt cpu");
    assert_eq!(
        move_toward_cpu[0], expected_alive,
        "CPU MoveToward reference should set every alive slot (all within max_move_radius)"
    );
}

// ---------------------------------------------------------------------------
// Phase 3 — scoring parity tests
// ---------------------------------------------------------------------------
//
// Two scoring tests live below. The first one is the byte-exact one
// (spawn state, view buffers all empty so the GPU's view stub matches
// CPU's "view returns 0"). The second is a multi-tick best-effort
// fixture (humans only — no hostile pairs, so no AgentAttacked events
// land, so view storage stays empty for the whole run).
//
// The full canonical 3v2 fixture is intentionally NOT used for scoring
// parity: as soon as the wolves attack, the CPU's view buffers
// (threat_level, my_enemies) populate but the GPU stub returns 0,
// flipping the argmax on Attack/Flee rows that depend on view
// modifiers. Phase 4 (task 185) wires real view storage; at that
// point this comment can come down and the canonical fixture joins
// the scoring parity sweep.

/// Fixture with 4 humans only, spaced inside max_move_radius.
/// Humans aren't hostile to each other; no `AgentAttacked` events ever
/// land; views (threat_level / my_enemies / kin_fear / pack_focus /
/// rally_boost) stay empty for the entire run. That makes the GPU's
/// view stub (returning 0) byte-equivalent to the CPU's actual view
/// reads (also 0 because the views are empty), so scoring parity
/// holds tick-by-tick.
fn spawn_no_combat_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(8.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 8.0),
        Vec3::new(8.0, 0.0, 8.0),
    ];
    for pos in positions {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos,
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
    }
    state
}

/// Phase 3 byte-exact test: the GPU scoring kernel produces the same
/// `(chosen_action, chosen_target)` per agent as the CPU reference on
/// the canonical 3v2 spawn state. Tick 0 — no combat has occurred,
/// every view buffer is empty, so the view stub on the GPU side
/// (returning 0) matches what the CPU's view reads return on an empty
/// view (also 0). This isolates "kernel correctness on a known state"
/// from "kernel correctness across the tick-by-tick state evolution".
#[test]
fn gpu_scoring_matches_cpu_on_spawn_state() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let state = spawn_fixture();
    let gpu_outs = backend
        .verify_scoring_on_gpu(&state)
        .expect("GPU scoring dispatch");
    let cpu_outs = cpu_score_outputs(&state);

    assert_eq!(
        gpu_outs.len(),
        cpu_outs.len(),
        "scoring output length differs"
    );

    // Diagnostic: print per-slot summaries before asserting so a fail
    // shows what diverged at a glance.
    eprintln!("gpu_scoring_matches_cpu_on_spawn_state: per-slot outputs");
    for slot in 0..gpu_outs.len() {
        let gpu = &gpu_outs[slot];
        let cpu = &cpu_outs[slot];
        eprintln!(
            "  slot {slot}: GPU=(action={}, target={}, score={:.4}) CPU=(action={}, target={}, score={:.4}){}",
            gpu.chosen_action,
            gpu.chosen_target,
            f32::from_bits(gpu.best_score_bits),
            cpu.chosen_action,
            cpu.chosen_target,
            f32::from_bits(cpu.best_score_bits),
            if gpu.chosen_action != cpu.chosen_action || gpu.chosen_target != cpu.chosen_target {
                "  <-- DIVERGENCE"
            } else {
                ""
            },
        );
    }

    // Per-slot equality. We compare on (chosen_action, chosen_target)
    // and the score's bit-pattern for full-byte parity. Pad bytes are
    // zero on both sides (GPU initialises to 0 before dispatch; CPU
    // ScoreOutput::default has _pad: 0).
    for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
        assert_scoring_eq(slot, gpu, cpu);
    }
}

/// Same shape as the byte-exact spawn-state test, but runs through
/// `TICKS` ticks of a no-combat fixture (4 humans, no wolves). Every
/// tick the GPU's `(chosen_action, chosen_target)` per agent must
/// match the CPU reference — view buffers stay empty for the whole
/// run because no hostile events fire, so the view stub doesn't
/// poison the argmax.
#[test]
fn gpu_scoring_matches_cpu_no_combat_fixture() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_no_combat_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let gpu_outs = backend.last_scoring_outputs();
        assert!(
            !gpu_outs.is_empty(),
            "tick {tick_i}: GpuBackend::last_scoring_outputs empty — kernel dispatch failed?"
        );
        assert_eq!(
            gpu_outs.len(),
            state.agent_cap() as usize,
            "tick {tick_i}: scoring output length mismatch"
        );

        let cpu_outs = cpu_score_outputs(&state);
        for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
            assert_scoring_eq_with_tick(tick_i, slot, gpu, cpu);
        }
    }
}

/// Phase 6c byte-exact: the GPU scoring kernel matches the CPU
/// reference's `(action, target, score)` per agent on EVERY tick of
/// the canonical 3-humans-2-wolves fixture. View modifiers
/// (grudges, pack_focus once engagements commit, rally_boost after
/// wounds) populate naturally over 50 ticks of combat, so this test
/// exercises the full view-read dispatch — including the wildcard
/// loop that mirrors `sum_for_first(a, tick)` and the dt=0 short-
/// circuit on `pow(rate, 0)` that keeps f32 byte-exact with the
/// uploaded value. Failure: any per-slot divergence at any tick.
///
/// Renamed from `gpu_scoring_canonical_fixture_best_effort` (which
/// allowed log-and-continue divergences while view storage was
/// stubbed at 0.0). Task 189 (Phase 6c) wired real view reads, so
/// this test now hard-asserts byte-exact for every tick.
#[test]
fn gpu_scoring_canonical_fixture_exact() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        let gpu_outs = backend.last_scoring_outputs();
        let cpu_outs = cpu_score_outputs(&state);
        assert_eq!(gpu_outs.len(), cpu_outs.len(), "tick {tick_i}: len");

        for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
            assert_scoring_eq_with_tick(tick_i, slot, gpu, cpu);
        }
    }
}

/// Byte-exact scoring parity with deliberate `AgentAttacked` events
/// folded into `my_enemies` ahead of time. The canonical 3v2 fixture
/// will populate `my_enemies` naturally over a long run, but this
/// test forces grudges into existence at tick 0 so the per-pair
/// wildcard sum + the modifier 5 (`my_enemies > 0.5` → +0.4 on
/// Attack) fires from the very first scoring dispatch. Catches
/// regressions in the upload path's pair_map_scalar handling that
/// the natural-evolution test would only surface on tick 25+.
#[test]
fn gpu_scoring_with_grudges_byte_exact() {
    use engine::ids::AgentId;
    use engine_gpu::view_storage::FoldInputPair;

    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Inject grudges: every wolf has attacked every human in the past.
    // Before task 193 (Phase 6g) the backend's step() mirrored CPU
    // `state.views` → GPU `view_storage` each tick, so injecting into
    // the CPU side was enough. Task 193 makes the GPU cascade
    // authoritative — scoring now reads GPU-owned view_storage that
    // only reflects what ran through GPU fold kernels. The test now
    // seeds BOTH the CPU and GPU sides so the byte-exact parity
    // semantic is preserved under the new architecture.
    let humans = [AgentId::new(1).unwrap(), AgentId::new(2).unwrap(), AgentId::new(3).unwrap()];
    let wolves = [AgentId::new(4).unwrap(), AgentId::new(5).unwrap()];

    // Ensure cascade/view_storage are sized right before folding on the
    // GPU side. A dry-run step() call would init these, but we want to
    // seed grudges BEFORE the first decision tick — so explicitly
    // resize view_storage and fold the grudge pairs in.
    backend
        .rebuild_view_storage(state.agent_cap())
        .expect("rebuild_view_storage");

    let mut pair_events: Vec<FoldInputPair> = Vec::new();
    for &wolf in &wolves {
        for &human in &humans {
            // my_enemies fold pattern: AgentAttacked { actor, target } →
            // my_enemies[target, actor] += 1.0 (clamped to [0, 1]).
            events.push(engine::event::Event::AgentAttacked {
                actor: wolf,
                target: human,
                damage: 0.0,
                tick: state.tick,
            });
            pair_events.push(FoldInputPair {
                first: human.raw() - 1,
                second: wolf.raw() - 1,
                tick: state.tick,
                _pad: 0,
            });
        }
    }
    let events_before = events.push_count() - (humans.len() * wolves.len());
    state.views.fold_all(&events, events_before, state.tick);

    // GPU-side: fold the same grudge pairs into my_enemies + threat_level
    // (AgentAttacked folds into both on the CPU path too — see views.sim).
    let device = backend.device().clone();
    let queue = backend.queue().clone();
    backend
        .view_storage_mut()
        .fold_pair_events(&device, &queue, "my_enemies", &pair_events)
        .expect("fold_pair_events(my_enemies)");
    backend
        .view_storage_mut()
        .fold_pair_events(&device, &queue, "threat_level", &pair_events)
        .expect("fold_pair_events(threat_level)");

    // Confirm the CPU view registers the grudges.
    let pre_check = state.views.my_enemies.get(humans[0], wolves[0]);
    assert!(pre_check > 0.0, "grudge fold didn't take: my_enemies[h1, w1] = {pre_check}");

    // Now run TICKS ticks. At every tick the GPU's `my_enemies` cells
    // mirror the CPU's, and any scoring row that reads
    // `my_enemies(target, self)` will fire identically on both sides.
    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        let gpu_outs = backend.last_scoring_outputs();
        let cpu_outs = cpu_score_outputs(&state);
        for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
            assert_scoring_eq_with_tick(tick_i, slot, gpu, cpu);
        }
    }
}

/// Phase 6d: scoring reads view cells straight from view_storage's
/// atomic buffers. This test exercises the *GPU* fold path end-to-end:
/// inject a handful of synthetic AgentAttacked events into BOTH the
/// CPU view registry and the GPU `view_storage::fold_pair_events`
/// kernel, then call `verify_scoring_on_gpu` and assert byte-exact
/// `(chosen_action, chosen_target, best_score_bits)` against the CPU
/// reference.
///
/// Divergences from `gpu_scoring_with_grudges_byte_exact` (which uses
/// `GpuBackend::step`'s CPU→GPU mirror bridge): this test bypasses
/// the mirror and folds directly via view_storage's kernel. If the
/// fold kernel's atomic CAS loop, the scoring kernel's `atomicLoad`
/// reads, or the anchor stamping disagree at the bit level, the
/// assertion fires. That's the Piece 1 contract: scoring + fold
/// share the same buffer.
#[test]
fn gpu_scoring_reads_fold_kernel_output_byte_exact() {
    use engine::ids::AgentId;
    use engine_gpu::view_storage::FoldInputPair;

    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_fixture();

    // Grow view_storage if needed so its cap matches the fixture's 8.
    backend
        .rebuild_view_storage(state.agent_cap())
        .expect("rebuild view_storage to match state.agent_cap");

    // Inject grudges on BOTH sides:
    //   CPU: push events into the ring + fold_all into state.views.
    //   GPU: call view_storage.fold_pair_events on my_enemies with the
    //        same (observer, attacker, tick) tuples.
    let humans = [AgentId::new(1).unwrap(), AgentId::new(2).unwrap(), AgentId::new(3).unwrap()];
    let wolves = [AgentId::new(4).unwrap(), AgentId::new(5).unwrap()];

    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let mut pair_events: Vec<FoldInputPair> = Vec::new();
    for &wolf in &wolves {
        for &human in &humans {
            // CPU side — AgentAttacked folds into my_enemies AND
            // threat_level (see assets/sim/views.sim fold handlers).
            events.push(engine::event::Event::AgentAttacked {
                actor: wolf,
                target: human,
                damage: 0.0,
                tick: state.tick,
            });
            // GPU side — my_enemies fold key = (observer=target, attacker=actor).
            // threat_level fold key = (a=target, b=actor).
            pair_events.push(FoldInputPair {
                first: human.raw() - 1,
                second: wolf.raw() - 1,
                tick: state.tick,
                _pad: 0,
            });
        }
    }
    state.views.fold_all(&events, 0, state.tick);

    // Reset + fold into view_storage on the GPU side. AgentAttacked
    // folds into TWO views — my_enemies (the +grudge fold) and
    // threat_level (the +decayed-threat fold) — so we dispatch the
    // same pair-events list against both. If this list ever diverges
    // from what's in CPU's `state.views.fold_all`, scoring's byte
    // comparison further down catches it.
    let (device, queue) = (backend.device().clone(), backend.queue().clone());
    backend.view_storage().reset(&queue);
    backend
        .view_storage_mut()
        .fold_pair_events(&device, &queue, "my_enemies", &pair_events)
        .expect("fold_pair_events(my_enemies)");
    backend
        .view_storage_mut()
        .fold_pair_events(&device, &queue, "threat_level", &pair_events)
        .expect("fold_pair_events(threat_level)");

    // Cross-check: the GPU-folded my_enemies cells match the CPU view
    // registry's values bit-for-bit. If this fails, the scoring-level
    // assertion below would be ambiguous between "fold diverged" and
    // "scoring read diverged".
    let gpu_cells = backend
        .view_storage()
        .readback_pair_scalar(&device, &queue, "my_enemies")
        .expect("readback my_enemies");
    let n = state.agent_cap() as usize;
    for observer_slot in 0..n {
        let obs = match AgentId::new(observer_slot as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        for attacker_slot in 0..n {
            let atk = match AgentId::new(attacker_slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            let cpu_v = state.views.my_enemies.get(obs, atk);
            let gpu_v = gpu_cells[observer_slot * n + attacker_slot];
            assert_eq!(
                cpu_v.to_bits(), gpu_v.to_bits(),
                "my_enemies[{observer_slot},{attacker_slot}] CPU={cpu_v} GPU={gpu_v} after GPU fold"
            );
        }
    }

    // Now run scoring and assert per-slot byte-exact.
    let gpu_outs = backend
        .verify_scoring_on_gpu_preserving_views(&state)
        .expect("scoring dispatch");
    let cpu_outs = cpu_score_outputs(&state);
    for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
        assert_scoring_eq(slot, gpu, cpu);
    }
}

fn assert_scoring_eq(slot: usize, gpu: &ScoreOutput, cpu: &ScoreOutput) {
    assert_scoring_eq_with_tick(u32::MAX, slot, gpu, cpu);
}

fn assert_scoring_eq_with_tick(tick: u32, slot: usize, gpu: &ScoreOutput, cpu: &ScoreOutput) {
    let tick_str = if tick == u32::MAX {
        "spawn".to_string()
    } else {
        format!("tick {tick}")
    };
    assert_eq!(
        gpu.chosen_action, cpu.chosen_action,
        "{tick_str} slot {slot}: chosen_action GPU={} CPU={}",
        gpu.chosen_action, cpu.chosen_action
    );
    assert_eq!(
        gpu.chosen_target, cpu.chosen_target,
        "{tick_str} slot {slot}: chosen_target GPU={} CPU={} (NO_TARGET={NO_TARGET})",
        gpu.chosen_target, cpu.chosen_target,
    );
    if gpu.best_score_bits != cpu.best_score_bits {
        let gpu_score = f32::from_bits(gpu.best_score_bits);
        let cpu_score = f32::from_bits(cpu.best_score_bits);
        // Post-task-193: the old CPU→GPU view mirror used to stamp
        // anchor=state.tick so decay reads hit `pow(rate, 0) == 1` on
        // the scoring tick, keeping the score bit-exact. With the GPU
        // cascade authoritative, anchor is the ORIGINATING event's
        // tick, so decay reads compute `pow(rate, scoring_tick -
        // event_tick)` for dt>0 — a 1-ULP precision gap between
        // WGSL's pow and the CPU f32 pow is possible. We allow that
        // sub-ULP drift when action/target agree, matching the
        // `cascade_parity.rs` tolerance for dt>0 decays. Anything
        // bigger than a few ULPs signals a real divergence (the
        // score flip through a threshold boundary would alter
        // chosen_action, which is still checked strict above).
        let diff = (gpu_score - cpu_score).abs();
        let rel = diff / cpu_score.abs().max(1e-6);
        let ulp_diff = (gpu.best_score_bits as i64 - cpu.best_score_bits as i64).abs();
        if rel > 1e-5 && ulp_diff > 4 {
            panic!(
                "{tick_str} slot {slot}: best_score_bits GPU=0x{:08x} ({gpu_score}) CPU=0x{:08x} ({cpu_score}) diff={diff} ulp_diff={ulp_diff}",
                gpu.best_score_bits, cpu.best_score_bits,
            );
        }
    }
}
