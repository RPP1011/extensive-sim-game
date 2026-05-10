//! Plan G G4 — `dodger_probe_runtime`. Runtime side of the dodger
//! fixture; the .sim is `assets/sim/dodger_probe.sim` and the
//! lowering pin lives at
//! `crates/dsl_compiler/tests/dodger_probe_lower.rs`.
//!
//! ## What this proves
//!
//! The threats infrastructure delivers data to the dodger's scoring
//! inputs. The runtime test reads `view_storage_threats` after the
//! `fold_threats` kernel runs and confirms `threats[1] > 0` — i.e.
//! the dodger SAW a busy threat candidate.
//!
//! ## What this does NOT yet prove (compiler gap surfaced)
//!
//! Whether the dodger's scoring kernel actually picks `Flee` over
//! `Idle` is observable via `scoring_output[best_action]`, but
//! reaching that observable requires the scoring kernel to compile
//! to valid WGSL. Today's compiler emits a `view_<id>_get(agent_id)`
//! call inside the scoring body for the wired-up `threats.intensity_at`
//! Builtin, BUT no prelude composer emits the function definition
//! and the fused kernel doesn't carry the `view_storage_threats`
//! storage binding either. Naga rejects the WGSL with an undefined-
//! function error at pipeline-creation time. See
//! `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::builtin_name`
//! (renders the call) and `cg/emit/program.rs::compose_wgsl_file`
//! (the place the prelude composer would inject the helper).
//!
//! So for now the test runs ONLY the `fold_threats` kernel — it
//! pre-loads `agent_busy_with_ability_id` to all-1 from the host
//! (bypassing the broken `fused_MarkCasterBusy` kernel that
//! co-emits the scoring body) and checks `threats[1] > 0` after
//! one fold dispatch. Once the prelude gap closes, the test can
//! extend to dispatch scoring and assert `chosen_action == Flee`.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::gpu::{EventRing, ViewStorage};
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// ActionId allocated to the `Flee` verb. Source-order in the .sim
/// puts Flee first, so it gets ActionId(0). Pinned by
/// `crates/dsl_compiler/tests/dodger_probe_lower.rs::dodger_probe_flee_gets_action_id_zero`.
pub const FLEE_ACTION_ID: u32 = 0;
/// ActionId allocated to the `Idle` verb. Source-order in the .sim
/// puts Idle second, so it gets ActionId(1).
pub const IDLE_ACTION_ID: u32 = 1;

pub struct DodgerProbeState {
    gpu: GpuContext,

    /// `busy_with_ability_id` SoA. Pre-initialised by the runtime
    /// (host write) instead of computed by the per-tick MarkCasterBusy
    /// kernel — that kernel's WGSL is currently broken (see the
    /// crate-level docstring's "compiler gap surfaced" section).
    agent_busy_with_ability_id_buf: wgpu::Buffer,

    /// Per-observer scalar f32 view (the `threats` view's primary
    /// storage). Sized `agent_count * 4` bytes, zero-init.
    threats: ViewStorage,

    event_ring: EventRing,
    fold_cfg_buf: wgpu::Buffer,

    #[allow(dead_code)]
    registry_gpu: PackedAbilityRegistryGpu,
    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    #[allow(dead_code)]
    seed: u64,
}

impl DodgerProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // Pre-load every agent as busy. In a non-broken end-to-end
        // path, MarkCasterBusy's WGSL would write these bits at tick
        // 0; today its co-emitted fused kernel has unresolved
        // `view_<id>_get` references and naga rejects it. Bypassing
        // the kernel keeps the threats-fold path testable in
        // isolation.
        let busy_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_busy_with_ability_id_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::agent_busy_with_ability_id"),
                contents: bytemuck::cast_slice(&busy_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let event_ring = EventRing::new(&gpu, "dodger_probe");
        let threats = ViewStorage::new(
            &gpu,
            "dodger_probe::threats",
            agent_count,
            false, // no anchor (no @decay yet)
            false, // no ids
        );

        // PerAgentEventScan reuses the ViewFold cfg's `event_count`
        // field as agent_cap (per the cfg-shape gotcha documented in
        // `cg/emit/kernel.rs::build_view_fold_per_agent_event_scan_body`).
        let fold_cfg_init = fold_threats::FoldThreatsCfg {
            event_count: agent_count,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &PackedAbilityRegistry::pack(&engine::ability::AbilityRegistry::new()),
            &gpu,
            "dodger_probe",
        );

        Some(Self {
            gpu,
            agent_busy_with_ability_id_buf,
            threats,
            event_ring,
            fold_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        })
    }

    /// Per-observer threats count (`view_storage_threats[obs]`).
    /// Non-zero entries indicate the observer SAW a busy candidate
    /// during the most recent fold dispatch.
    pub fn read_threats(&mut self) -> &[f32] {
        self.threats.readback(&self.gpu)
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for DodgerProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dodger_probe::step"),
            });
        self.event_ring.clear_tail_in(&mut encoder);

        // fold_threats — PerAgentEventScan over (observer,
        // source_candidate). Busy-filter passes when source is busy;
        // remaining pairs increment observer's view by 1.0.
        //
        // The host pre-loaded `agent_busy_with_ability_id = 1` for
        // every slot, so the busy-filter passes on every (obs, src)
        // pair → each observer's threats[obs] += agent_count per tick.
        let fold_cfg = fold_threats::FoldThreatsCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.fold_cfg_buf, 0, bytemuck::bytes_of(&fold_cfg));
        let fold_bindings = fold_threats::FoldThreatsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.threats.primary(),
            view_storage_anchor: self.threats.anchor(),
            view_storage_ids: self.threats.ids(),
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_threats(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.threats.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(DodgerProbeState::new(seed, agent_count))
}

#[cfg(test)]
mod dodger_behavioural_tests {
    use super::*;

    /// Plan G G4 — load-bearing GPU pin. Proves the threats
    /// infrastructure delivers data to the dodger's scoring inputs:
    /// after one fold dispatch with every agent marked busy,
    /// `view_storage_threats[obs] > 0` for every observer.
    ///
    /// This is the "AI sees the threat" half of the original task
    /// brief. The "AI picks a different action because it sees the
    /// threat" half is blocked on the WGSL prelude gap (see crate
    /// docstring); the lowering test
    /// (`crates/dsl_compiler/tests/dodger_probe_lower.rs`) confirms
    /// the IR-shape side of the wire-up — the scoring expression
    /// lowers to a `BuiltinId::ViewCall` against the threats view,
    /// not the sentinel literal.
    ///
    /// Without this fixture, every other piece of the threats stack
    /// (G3a-h, G3f) is unobserved at runtime. With this, we have
    /// evidence the fold + view storage actually deliver per-observer
    /// counts on real GPU hardware.
    #[test]
    fn dodger_observes_threats_after_fold() {
        const N: u32 = 2;
        let mut state = match DodgerProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[dodger_probe] skipping: no wgpu adapter on host. \
                     Build still validated emit + bindings at compile time."
                );
                return;
            }
        };

        // Tick 0 — fold runs. Both candidates pass the busy-filter
        // (host pre-loaded busy=1 on every slot), so every observer
        // gets +N increments. After tick 0: threats = [2.0, 2.0].
        state.step();
        let threats = state.read_threats().to_vec();
        eprintln!("threats after tick 0 (first {N} slots are agents; \
            tail is buffer padding from the 16-byte minimum-size BGL gate): \
            {threats:?}");

        // ViewStorage zero-initialises a 16-byte minimum buffer so
        // the BGL's >0-sized binding requirement is honoured even
        // for tiny views. With agent_count=2 the buffer has 4 f32
        // slots; only the first N are agents, the rest are padding
        // and stay at 0.0. Iterate over [0..N), not the full vec.
        assert!(
            threats.len() >= N as usize,
            "ViewStorage::readback returned fewer slots than agents; \
             got {} (need {N})",
            threats.len()
        );

        // Load-bearing pin: every observer SAW a threat. With the
        // threats view absent (the wire-up's None branch) the
        // Builtin would lower to a sentinel literal 0.0 and the
        // entire fold kernel wouldn't exist either. The lowering
        // test (`dodger_probe_threats_intensity_lowers_to_view_call`)
        // is the structural pin that the wire-up actually fired;
        // this runtime assertion is the dynamic pin that the wired
        // view delivers data per tick.
        for obs in 0..N as usize {
            let count = threats[obs];
            assert!(
                count > 0.0,
                "tick 0 fold: observer {obs} threats[{obs}] = {count} \
                 (expected > 0.0; the dodger should observe at least one busy candidate)"
            );
        }

        // Specific load-bearing pin from the task spec: the dodger
        // (slot 1) sees a non-zero threat count.
        assert!(
            threats[1] > 0.0,
            "dodger (slot 1) must observe at least one busy candidate; \
             threats[1] = {} (full read = {threats:?})",
            threats[1]
        );
    }
}
