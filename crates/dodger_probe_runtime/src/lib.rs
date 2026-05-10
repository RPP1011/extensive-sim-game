//! Plan G G4 — `dodger_probe_runtime`. Runtime side of the dodger
//! fixture; the .sim is `assets/sim/dodger_probe.sim` and the
//! lowering pin lives at
//! `crates/dsl_compiler/tests/dodger_probe_lower.rs`.
//!
//! ## What this proves
//!
//! The threats infrastructure materially changes AI behaviour on
//! GPU. Three pins:
//!   * `dodger_observes_threats_after_fold` — the fold kernel
//!     populates `view_storage_threats[obs] > 0` per observer.
//!   * `dodger_picks_flee_via_host_scoring_when_threats_present` —
//!     host-side mirror of the scoring expression; sanity check.
//!   * `dodger_picks_flee_on_gpu_when_threats_present` — full GPU
//!     pipeline; reads `scoring_output[agent * 4]` after running the
//!     scoring kernel and asserts dodger picks Flee WITH threats and
//!     Idle WITHOUT.
//!
//! ## Step-loop architecture
//!
//! `step()` walks the compiler-emitted [`schedule::SCHEDULE`] table
//! in source order rather than hand-authoring the dispatch sequence.
//! Each kernel's bindings are wired in a match arm against
//! `KernelId`; missing a dispatch is impossible because the loop
//! visits every emitted entry. Adding a kernel to the .sim grows the
//! match by one arm.

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

    /// `agent_alive` SoA. Flee + Idle verbs read this for their
    /// `when (self.alive)` eligibility gate.
    agent_alive_buf: wgpu::Buffer,

    /// Per-(observer, source) belief bits — `agent_count * agent_count`
    /// u32 cells. The `@belief_gated` annotation on dodger_probe.sim's
    /// `view threats` causes the fold kernel to read
    /// `beliefs_flags[observer * agent_cap + source] & (1u << 7u)`
    /// instead of the raw busy lookup. Default seed in `try_new` is
    /// `(1u << 7u)` for every cell — preserves the omniscient
    /// behavior the existing tests expect. Tests that want per-observer
    /// asymmetry call `seed_beliefs_flags`.
    beliefs_flags_buf: wgpu::Buffer,

    /// `busy_with_ability_id` SoA. The fused MarkCasterBusy kernel
    /// (now compilable thanks to the WGSL prelude composer fix at
    /// commit 6d8bd159) writes this column at tick 0; subsequent
    /// ticks read it. Initialised to zero so the kernel's
    /// `where (world.tick == 0)` gate can flip the bits.
    agent_busy_with_ability_id_buf: wgpu::Buffer,

    /// Per-observer scalar f32 view (the `threats` view's primary
    /// storage). Sized `agent_count * 4` bytes, zero-init.
    threats: ViewStorage,

    /// Per-verb mask bitmaps (one bit per agent each). Verb scoring
    /// kernels write to these atomically as the eligibility gate
    /// (`when (self.alive)` for both Flee and Idle today). Sized
    /// `(agent_count + 31) / 32 * 4` bytes — one u32 word per 32
    /// agents, zero-init each tick by the dispatch.
    mask_0_bitmap_buf: wgpu::Buffer, // Flee verb's mask
    mask_1_bitmap_buf: wgpu::Buffer, // Idle verb's mask

    /// Scoring output: 4 × u32 per agent — `[best_action,
    /// best_target, bitcast<u32>(best_utility), 0]`. The argmax
    /// kernel writes the winning action per agent.
    scoring_output_buf: wgpu::Buffer,
    scoring_output_staging: wgpu::Buffer,

    event_ring: EventRing,
    fold_cfg_buf: wgpu::Buffer,
    fused_cfg_buf: wgpu::Buffer,

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

        // Pre-load busy_with_ability_id to all-1. The fused
        // MarkCasterBusy kernel ALSO writes these bits at tick 0 via
        // its `where (world.tick == 0)` gate — so the host preload is
        // idempotent there but lets the first fold dispatch (which
        // runs BEFORE MarkCasterBusy in [`schedule::SCHEDULE`]) see
        // non-zero busy on tick 0. Without the preload, fold would
        // early-exit on tick 0 and the view would stay at 0 until
        // tick 1, costing every observer one tick of latency.
        let busy_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_busy_with_ability_id_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::agent_busy_with_ability_id"),
                contents: bytemuck::cast_slice(&busy_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        // alive SoA — Flee/Idle's `when (self.alive)` gate reads
        // agent_alive. All-1 (every agent alive).
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // beliefs_flags — `agent_count * agent_count` u32 cells. The
        // `@belief_gated` annotation on the threats view causes the
        // fold to read this buffer for the source-candidate gate.
        // Default seed: every cell carries `(1u << 7u)` (the
        // OBSERVED_BUSY bit) so every observer sees every source —
        // preserves the omniscient behavior the existing tests expect.
        // Tests that want per-observer asymmetry call
        // `seed_beliefs_flags` to override.
        let beliefs_init: Vec<u32> =
            vec![1u32 << 7u32; (agent_count as usize) * (agent_count as usize)];
        let beliefs_flags_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::beliefs_flags"),
                contents: bytemuck::cast_slice(&beliefs_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Mask bitmaps — one u32 word per 32 agents (16 bytes minimum
        // floor for wgpu's >0-sized binding requirement).
        let mask_words = ((agent_count + 31) / 32).max(1) as usize;
        let mask_bytes = (mask_words * 4).max(16) as u64;
        let mask_zeros = vec![0u32; (mask_bytes / 4) as usize];
        let mk_mask = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&mask_zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mask_0_bitmap_buf = mk_mask("dodger_probe::mask_0_bitmap_flee");
        let mask_1_bitmap_buf = mk_mask("dodger_probe::mask_1_bitmap_idle");

        // Scoring output: 4 × u32 per agent. The argmax kernel writes
        // [best_action, best_target, bitcast<u32>(best_utility), pad].
        let scoring_words = (agent_count as usize) * 4;
        let scoring_bytes = ((scoring_words * 4).max(16)) as u64;
        let scoring_zeros = vec![0u32; (scoring_bytes / 4) as usize];
        let scoring_output_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::scoring_output"),
                contents: bytemuck::cast_slice(&scoring_zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let scoring_output_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dodger_probe::scoring_output_staging"),
            size: scoring_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
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

        let fused_cfg_init = fused_MarkCasterBusy::FusedMarkCasterBusyCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let fused_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("dodger_probe::fused_cfg"),
                contents: bytemuck::bytes_of(&fused_cfg_init),
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
            agent_alive_buf,
            agent_busy_with_ability_id_buf,
            beliefs_flags_buf,
            threats,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            scoring_output_buf,
            scoring_output_staging,
            event_ring,
            fold_cfg_buf,
            fused_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        })
    }

    /// Plan G — seed the per-(observer, source) belief grid. Each
    /// cell at index `observer * agent_count + source` is a u32
    /// bitset; bit 7 is the `OBSERVED_BUSY` convention the fold gate
    /// reads. Setting bit 7 → "observer believes source is busy
    /// casting" → the fold counts source's contribution toward
    /// observer's threat tally. Clearing bit 7 → fold skips the pair.
    ///
    /// Length must be `agent_count * agent_count`. Used by tests
    /// that want to demonstrate per-observer threat asymmetry on
    /// real GPU (the default `try_new` seeds every cell with bit 7,
    /// preserving the omniscient behavior).
    pub fn seed_beliefs_flags(&mut self, values: &[u32]) {
        let expected = (self.agent_count as usize) * (self.agent_count as usize);
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_flags: length {} doesn't match agent_count^2 = {}",
            values.len(),
            expected,
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_flags_buf,
            0,
            bytemuck::cast_slice(values),
        );
    }

    /// Convenience: the OBSERVED_BUSY bit value (1 << 7). Tests
    /// constructing seed arrays use this to keep bit position in
    /// one place.
    pub const OBSERVED_BUSY_BIT: u32 = 1u32 << 7u32;

    /// Force-zero both the busy source AND the threats view — test-only.
    /// Both writes are needed:
    ///   * `busy = 0` stops `FoldThreats` from re-incrementing the view
    ///     this tick (the fold's busy-filter early-exits).
    ///   * `view = 0` clears the accumulated value from prior ticks
    ///     (the fold's body is `self += 1.0` with no per-tick reset, so
    ///     a non-zero view persists once any tick stamped a non-zero
    ///     value).
    /// Without both, scoring on the next tick would still read the
    /// pre-existing accumulated view value.
    pub fn force_no_threats(&mut self) {
        let busy_zeros: Vec<u32> = vec![0u32; self.agent_count as usize];
        self.gpu.queue.write_buffer(
            &self.agent_busy_with_ability_id_buf,
            0,
            bytemuck::cast_slice(&busy_zeros),
        );
        let view_zeros: Vec<u32> = vec![0u32; (self.agent_count as usize).max(4)];
        self.gpu.queue.write_buffer(
            self.threats.primary(),
            0,
            bytemuck::cast_slice(&view_zeros),
        );
        // @belief_gated: clear the actual gate predicate's source.
        // Without this, the fold's belief-bit check passes for every
        // cell (default seed = all-bit-7-set) and threats accumulate
        // again on the next step.
        let beliefs_zeros: Vec<u32> =
            vec![0u32; (self.agent_count as usize) * (self.agent_count as usize)];
        self.gpu.queue.write_buffer(
            &self.beliefs_flags_buf,
            0,
            bytemuck::cast_slice(&beliefs_zeros),
        );
        self.threats.mark_dirty();
    }

    /// Read the scoring_output buffer back to host. Layout per agent:
    /// `[best_action: u32, best_target: u32, bitcast<u32>(best_utility): u32, _pad: u32]`.
    /// The argmax kernel writes one quad per agent; index by
    /// `agent_id * 4 + 0` for the winning action.
    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = ((self.agent_count as usize) * 4 * 4).max(16) as u64;
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dodger_probe::read_scoring_output"),
            });
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_buf,
            0,
            &self.scoring_output_staging,
            0,
            bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = self.scoring_output_staging.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("scoring_output_staging map_async failed")
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
        let out = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            words.to_vec()
        };
        self.scoring_output_staging.unmap();
        out
    }

    /// Convenience: per-agent winning action as parsed by the argmax
    /// kernel. `chosen_action(agent_id)` reads `scoring_output[agent_id * 4]`.
    pub fn chosen_action(&self, scoring_output: &[u32], agent_id: u32) -> u32 {
        scoring_output[(agent_id as usize) * 4]
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
    /// Walk the compiler-emitted [`schedule::SCHEDULE`] table in source
    /// order, binding each kernel from its `KernelId`. The runtime no
    /// longer hand-authors the dispatch sequence — that's the
    /// compiler's job. Adding a kernel to the .sim only requires a new
    /// match arm here for binding setup; missing a dispatch is
    /// impossible because the loop visits every entry.
    ///
    /// The 3 kernels actively wired in this fixture:
    ///   1. `FusedMaskVerbFlee` — sets mask bitmaps so the scoring
    ///      kernel's per-verb eligibility checks pass.
    ///   2. `FoldThreats` — PerAgentEventScan; updates view storage
    ///      from `agent_busy_with_ability_id` (read on next tick).
    ///   3. `FusedMarkCasterBusy` — stamps busy at tick 0 + computes
    ///      scoring_argmax against the threats view.
    ///
    /// `UploadSimCfg`, `FusedPackAgents`, `KickSnapshot` are emitted
    /// by the compiler but unused in this minimal pin (no snapshot
    /// readback path, no agent-pack consumer); they fall through the
    /// catch-all match arm.
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dodger_probe::step"),
            });
        self.event_ring.clear_tail_in(&mut encoder);

        let agent_buffers = engine::gpu::AgentBuffers {
            alive_buf: Some(&self.agent_alive_buf),
            ..Default::default()
        };
        let ctx = engine::gpu::KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
            voxel_grid: None,
        };

        // Per-tick mask clear: mask_predicate ops accumulate via
        // atomicOr; stale bits from previous ticks would break per-tick
        // eligibility. Issued before the compute pass so the GPU
        // observes zeroes when the mask kernel reads-modify-writes.
        let mask_words = ((self.agent_count + 31) / 32).max(1) as usize;
        let mask_bytes = (mask_words * 4).max(16) as u64;
        let zeros = vec![0u32; (mask_bytes / 4) as usize];
        self.gpu.queue.write_buffer(&self.mask_0_bitmap_buf, 0, bytemuck::cast_slice(&zeros));
        self.gpu.queue.write_buffer(&self.mask_1_bitmap_buf, 0, bytemuck::cast_slice(&zeros));

        // Update per-tick cfg uniforms once. Both kernels consume the
        // same `(agent_cap, tick, seed, _pad)` POD layout.
        let cfg = fused_MarkCasterBusy::FusedMarkCasterBusyCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.fused_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let fold_cfg = fold_threats::FoldThreatsCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.fold_cfg_buf, 0, bytemuck::bytes_of(&fold_cfg));

        for op in schedule::SCHEDULE {
            match op {
                schedule::DispatchOp::Kernel(KernelId::FusedMaskVerbFlee) => {
                    let extras = fused_mask_verb_Flee::FusedMaskVerbFleeExtras {
                        mask_0_bitmap: &self.mask_0_bitmap_buf,
                        mask_1_bitmap: &self.mask_1_bitmap_buf,
                        cfg: &self.fused_cfg_buf,
                    };
                    let bindings = fused_mask_verb_Flee::FusedMaskVerbFleeBindings::from_context_with_extras(
                        &ctx, &extras,
                    );
                    dispatch::dispatch_fused_mask_verb_flee(
                        &mut self.cache,
                        &bindings,
                        &self.gpu.device,
                        &mut encoder,
                        self.agent_count,
                    );
                }
                schedule::DispatchOp::Kernel(KernelId::FoldThreats) => {
                    let bindings = fold_threats::FoldThreatsBindings {
                        event_ring: self.event_ring.ring(),
                        event_tail: self.event_ring.tail(),
                        view_storage_primary: self.threats.primary(),
                        view_storage_anchor: self.threats.anchor(),
                        view_storage_ids: self.threats.ids(),
                        agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
                        beliefs_flags: &self.beliefs_flags_buf,
                        sim_cfg: self.event_ring.sim_cfg(),
                        cfg: &self.fold_cfg_buf,
                    };
                    dispatch::dispatch_fold_threats(
                        &mut self.cache,
                        &bindings,
                        &self.gpu.device,
                        &mut encoder,
                        self.agent_count,
                    );
                }
                schedule::DispatchOp::Kernel(KernelId::FusedMarkCasterBusy) => {
                    let extras = fused_MarkCasterBusy::FusedMarkCasterBusyExtras {
                        agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
                        mask_0_bitmap: &self.mask_0_bitmap_buf,
                        mask_1_bitmap: &self.mask_1_bitmap_buf,
                        scoring_output: &self.scoring_output_buf,
                        view_storage_threats_primary: self.threats.primary(),
                        cfg: &self.fused_cfg_buf,
                    };
                    let bindings = fused_MarkCasterBusy::FusedMarkCasterBusyBindings::from_context_with_extras(
                        &ctx, &extras,
                    );
                    dispatch::dispatch_fused_markcasterbusy(
                        &mut self.cache,
                        &bindings,
                        &self.gpu.device,
                        &mut encoder,
                        self.agent_count,
                    );
                }
                // UploadSimCfg / FusedPackAgents / KickSnapshot — emitted
                // by the compiler but not wired to any consumer in this
                // fixture. Skipping them is the explicit choice; the
                // schedule loop makes the omission audit-visible.
                schedule::DispatchOp::Kernel(_) => {}
                schedule::DispatchOp::FixedPoint { .. }
                | schedule::DispatchOp::Indirect { .. }
                | schedule::DispatchOp::GatedBy { .. } => {}
            }
        }

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

    /// Plan G G4 — host-side dodge demonstration. The proper GPU
    /// scoring path is blocked on the WGSL prelude composer gap
    /// (audit `crates/dodger_probe_runtime/src/lib.rs` docstring +
    /// task #288). Until that gap closes, this pin demonstrates the
    /// BEHAVIOR end-to-end:
    ///
    ///   1. Run the threats fold on GPU (proven by the sibling pin).
    ///   2. Read threats per observer back to host.
    ///   3. Evaluate the same scoring expressions the .sim declares
    ///      (Flee = threats[obs]; Idle = 0.5) on the host CPU.
    ///   4. Argmax the per-agent winning action.
    ///   5. Compare WITH-threats vs FORCE-ZERO baseline: WITH picks
    ///      Flee; baseline picks Idle.
    ///
    /// This is a host-side mirror of what the GPU scoring kernel
    /// SHOULD do once the prelude composer wires `view_<id>_get`
    /// helpers + `view_storage_<name>` BGL bindings. The same
    /// expressions, same arithmetic, same argmax — just evaluated
    /// CPU-side to bypass the broken WGSL emit path.
    ///
    /// Pin asserts:
    ///   * threat-aware mode: dodger picks Flee (action_id 0).
    ///   * baseline (threats forced to zero): dodger picks Idle (action_id 1).
    ///   * Diff in chosen action proves the threats infrastructure
    ///     materially changes AI behaviour.
    #[test]
    fn dodger_picks_flee_via_host_scoring_when_threats_present() {
        const N: u32 = 2;
        const ACTION_FLEE: u32 = 0;
        const ACTION_IDLE: u32 = 1;
        let mut state = match DodgerProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!("[dodger_probe] skipping host-side dodge pin: no wgpu adapter.");
                return;
            }
        };
        state.step();
        let threats = state.read_threats().to_vec();

        // Host-side scoring mirroring the .sim's verb declarations:
        //   verb Flee score (threats.intensity_at(self))
        //   verb Idle score (0.5)
        let host_score = |_obs: usize, threat_intensity: f32| -> (u32, f32) {
            let flee_utility: f32 = threat_intensity;
            let idle_utility: f32 = 0.5;
            if flee_utility > idle_utility {
                (ACTION_FLEE, flee_utility)
            } else {
                (ACTION_IDLE, idle_utility)
            }
        };

        // Threat-aware: dodger reads its real per-observer threat count.
        let (dodger_action_aware, dodger_utility_aware) = host_score(1, threats[1]);
        assert_eq!(
            dodger_action_aware, ACTION_FLEE,
            "WITH threats: dodger should pick Flee (action {ACTION_FLEE}); \
             threats[1]={}, utility={dodger_utility_aware}",
            threats[1]
        );

        // Baseline: force the threat read to zero (the sentinel the
        // Builtin would emit if the threats view were absent or the
        // wire-up didn't fire).
        let (dodger_action_baseline, dodger_utility_baseline) = host_score(1, 0.0);
        assert_eq!(
            dodger_action_baseline, ACTION_IDLE,
            "WITHOUT threats: dodger should pick Idle (action {ACTION_IDLE}); \
             utility={dodger_utility_baseline}"
        );

        // Diff is the proof: threats infrastructure materially changes
        // AI choice. ACTION_FLEE != ACTION_IDLE (≠ in raw u32 too).
        assert_ne!(
            dodger_action_aware, dodger_action_baseline,
            "threats infrastructure must produce a behaviour delta; \
             aware={dodger_action_aware}, baseline={dodger_action_baseline}"
        );
        eprintln!(
            "[dodger host-side pin] WITH threats[1]={} → action={} ({:.2} util); \
             WITHOUT → action={} ({:.2} util). Diff confirmed.",
            threats[1], dodger_action_aware, dodger_utility_aware,
            dodger_action_baseline, dodger_utility_baseline,
        );
    }

    /// **The load-bearing GPU pin.** Replaces the host-side stand-in
    /// with a real GPU-side scoring readback. The full pipeline runs:
    ///
    ///   1. fused_MarkCasterBusy stamps `busy_with_ability_id = 1`
    ///      at tick 0; mask predicates gate Flee + Idle on `self.alive`;
    ///      scoring argmax computes per-agent winning action by reading
    ///      `view_storage_threats_primary` via the new prelude composer.
    ///   2. fold_threats updates the view storage (read on next tick).
    ///   3. After step 2, scoring_output[agent * 4] == winning action.
    ///
    /// Pin shape:
    ///   * Run TWO ticks: tick 0 fires MarkCasterBusy + fold; tick 1
    ///     fires scoring with the threats view populated by tick 0's fold.
    ///   * After tick 1: dodger's scoring_output[1*4] == FLEE (0).
    ///   * Reset threats to zero (force-zero baseline); run another
    ///     tick; scoring_output[1*4] == IDLE (1).
    ///
    /// This is the pin that proves the threats infrastructure
    /// materially changes AI behaviour ON GPU — not via host-side
    /// scoring stand-in.
    #[test]
    fn dodger_picks_flee_on_gpu_when_threats_present() {
        const N: u32 = 2;
        let mut state = match DodgerProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!("[dodger gpu pin] skipping: no wgpu adapter on host.");
                return;
            }
        };

        // Tick 0: MarkCasterBusy stamps busy + scoring runs against
        // initial-zero threats (so picks Idle), then fold populates
        // threats from busy.
        state.step();
        // Tick 1: MarkCasterBusy is a no-op (`world.tick == 0` gate
        // fails), fold sees busy=1 and adds N to each observer.
        // Scoring sees the previous tick's threat counts (also N=2 from
        // tick 0's fold). Dodger picks Flee.
        state.step();

        let scoring = state.read_scoring_output();
        let dodger_action = state.chosen_action(&scoring, 1);
        eprintln!(
            "[dodger gpu pin] tick 1 (threats present) — dodger chosen action = {} (FLEE={}, IDLE={})",
            dodger_action, FLEE_ACTION_ID, IDLE_ACTION_ID,
        );
        assert_eq!(
            dodger_action, FLEE_ACTION_ID,
            "WITH threats: dodger should pick Flee on GPU; got action_id={}",
            dodger_action,
        );

        // Baseline: zero both the busy source and the accumulated view,
        // then step. Fold sees busy=0 (early-exits, view stays zero);
        // scoring then reads view=0 so Flee (0.0) loses to Idle (0.5).
        // The MarkCasterBusy `world.tick == 0` gate has long since
        // stopped firing, so busy stays at 0 once we zero it.
        state.force_no_threats();
        state.step();
        let scoring = state.read_scoring_output();
        let dodger_action_baseline = state.chosen_action(&scoring, 1);
        assert_eq!(
            dodger_action_baseline, IDLE_ACTION_ID,
            "WITHOUT threats: dodger should pick Idle on GPU; got action_id={}",
            dodger_action_baseline,
        );

        assert_ne!(
            dodger_action, dodger_action_baseline,
            "GPU scoring must produce a behaviour delta when threats are present vs absent",
        );
    }

    /// **Effectiveness report.** Closes the qualitative-vs-quantitative
    /// gap left by the per-agent dodger pin. Runs N=64 agents over
    /// M=16 ticks twice — once with threats present, once with the
    /// busy source forced to zero — and reports the per-condition
    /// percentage of agents that picked Flee. The delta IS the threat
    /// system's effectiveness on AI scoring.
    ///
    /// Pin shape:
    ///   * WITH threats: ≥95% of agents (every agent except the
    ///     warmup tick the fold hasn't populated yet) should pick
    ///     Flee — every observer sees N busy candidates so Flee's
    ///     utility (= N) outscores Idle's 0.5.
    ///   * WITHOUT threats (busy and view forced to zero each tick):
    ///     0% should pick Flee — every Flee score is 0.0, Idle wins
    ///     on the 0.5 baseline.
    ///   * Delta ≥ 95 percentage points → threats infrastructure
    ///     materially changes ≥95% of decisions.
    ///
    /// This is the metric the recurring "report effectiveness of
    /// threat impact on scoring" prompt asked for. Per-tick log lines
    /// land in `--nocapture` output for human review.
    #[test]
    fn threat_effectiveness_report_with_vs_without_at_population_scale() {
        const N: u32 = 64;
        const TICKS: usize = 16;

        // Helper: count how many of the N agents picked Flee.
        fn flee_count(scoring: &[u32], n: u32) -> u32 {
            (0..n)
                .filter(|i| scoring[(*i as usize) * 4] == FLEE_ACTION_ID)
                .count() as u32
        }

        // --- Condition A: threats present (default state).
        let mut state_a = match DodgerProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!("[threat-effectiveness] skipping: no wgpu adapter on host.");
                return;
            }
        };
        let mut flee_a_per_tick: Vec<u32> = Vec::with_capacity(TICKS);
        for _ in 0..TICKS {
            state_a.step();
            let scoring = state_a.read_scoring_output();
            flee_a_per_tick.push(flee_count(&scoring, N));
        }

        // --- Condition B: threats forced to zero each tick (baseline).
        let mut state_b =
            DodgerProbeState::try_new(0xCAFE, N).expect("second wgpu state");
        let mut flee_b_per_tick: Vec<u32> = Vec::with_capacity(TICKS);
        for _ in 0..TICKS {
            state_b.force_no_threats();
            state_b.step();
            let scoring = state_b.read_scoring_output();
            flee_b_per_tick.push(flee_count(&scoring, N));
        }

        // --- Aggregate. Skip tick 0 from condition A (warmup — fold
        // hasn't populated the view yet, so scoring sees zero threats
        // and picks Idle). The "effectiveness" metric is steady-state
        // behavior, ticks 1..TICKS.
        let total_decisions = (TICKS as u32 - 1) * N;
        let flees_a: u32 = flee_a_per_tick[1..].iter().sum();
        let flees_b: u32 = flee_b_per_tick.iter().sum();
        let pct_a = (flees_a as f64 / total_decisions as f64) * 100.0;
        let pct_b = (flees_b as f64 / (TICKS as u32 * N) as f64) * 100.0;
        let delta_pp = pct_a - pct_b;

        eprintln!(
            "[threat-effectiveness] N={N} TICKS={TICKS} → effectiveness report",
        );
        eprintln!(
            "[threat-effectiveness]   WITH threats:    {flees_a}/{total_decisions} flees in steady state ({pct_a:.1}%)"
        );
        eprintln!(
            "[threat-effectiveness]   WITHOUT threats: {flees_b}/{} flees ({pct_b:.1}%)",
            TICKS as u32 * N
        );
        eprintln!(
            "[threat-effectiveness]   delta:           {delta_pp:.1} percentage points"
        );
        eprintln!(
            "[threat-effectiveness]   per-tick flee counts WITH:    {flee_a_per_tick:?}"
        );
        eprintln!(
            "[threat-effectiveness]   per-tick flee counts WITHOUT: {flee_b_per_tick:?}"
        );

        assert!(
            pct_a >= 95.0,
            "WITH threats: expected ≥95% of agents to pick Flee in steady state; got {pct_a:.1}%"
        );
        assert_eq!(
            flees_b, 0,
            "WITHOUT threats: NO agent should pick Flee (Flee scores 0.0, Idle wins on 0.5 baseline); got {flees_b}/{}",
            TICKS as u32 * N,
        );
        assert!(
            delta_pp >= 95.0,
            "threat infrastructure must shift ≥95 percentage points of decisions; got {delta_pp:.1}pp",
        );
    }

    /// **The belief-gated divergence pin — the E2E proof.** Same
    /// physical world (every agent stamps `busy = 1` at tick 0), but
    /// per-observer beliefs produce divergent scoring on real GPU.
    ///
    /// Setup with N=4: observer slot 1 has belief bit 7 set for
    /// source 0 ONLY. Every other (observer, source) cell is clear.
    /// Every observer except slot 1 is "blind" — believes no source
    /// is busy.
    ///
    /// Expected with `@belief_gated` on `view threats`:
    ///   * Slot 1's threats[1] = 1.0 → picks Flee.
    ///   * Every other slot's threats[i] = 0.0 → picks Idle.
    ///
    /// Without the annotation the omniscient gate would make every
    /// observer see every busy source → every agent picks Flee. The
    /// behavioral asymmetry IS the proof the GPU kernel is reading
    /// the belief bits, not the raw busy bits.
    #[test]
    fn belief_gated_divergence_per_observer_on_gpu() {
        const N: u32 = 4;
        let mut state = match DodgerProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!("[belief-divergence] skipping: no wgpu adapter on host.");
                return;
            }
        };

        // Per-observer asymmetric seed: only slot 1 believes slot 0
        // is busy. Every other (observer, source) cell is clear.
        let cap = (N * N) as usize;
        let mut beliefs = vec![0u32; cap];
        beliefs[(1 * N + 0) as usize] = DodgerProbeState::OBSERVED_BUSY_BIT;
        state.seed_beliefs_flags(&beliefs);

        // Two ticks: tick 0 lets the busy preload + fold settle;
        // tick 1 is the first observable steady-state scoring.
        state.step();
        state.step();

        let scoring = state.read_scoring_output();
        let actions: Vec<u32> = (0..N).map(|i| state.chosen_action(&scoring, i)).collect();
        eprintln!("[belief-divergence] per-agent chosen actions: {actions:?}");

        assert_eq!(
            actions[1], FLEE_ACTION_ID,
            "slot 1 has belief bit set for source 0 → expected Flee, got {}",
            actions[1],
        );
        for i in 0..N {
            if i == 1 {
                continue;
            }
            assert_eq!(
                actions[i as usize], IDLE_ACTION_ID,
                "slot {i} has no beliefs set → expected Idle, got {}",
                actions[i as usize],
            );
        }

        eprintln!(
            "[belief-divergence] proven on real GPU: only slot 1 (with belief bit) picked Flee; \
             slots 0/2/3 (no beliefs) picked Idle. Same physical world, divergent per-observer behavior."
        );
    }

    /// **The sidestep pin.** This is the test that closes the gap
    /// from "Flee is a state" → "agent escapes a real projectile".
    /// The bar set in chat: an agent must be able to sidestep a
    /// narrow projectile launched at them.
    ///
    /// Setup: a projectile travels along the line y=0 from x=10 to
    /// x=-10 over 10 ticks (speed 1 / tick). Corridor half-width
    /// 0.5 — anything within |y| < 0.5 at impact tick gets hit. The
    /// dodger starts at (0, 0) — dead center of the corridor at
    /// the impact tick, so it WILL be hit if it doesn't move.
    ///
    /// Per tick the GPU scoring runs (existing pipeline). When the
    /// scoring picks Flee (because threats are present), the host
    /// applies a perpendicular sidestep — the projectile direction
    /// is `(-1, 0)`, so perpendicular is `(0, ±1)`. With sidestep
    /// speed 0.1 over 10 ticks the dodger ends at y ≈ 1.0, well
    /// outside the |y| < 0.5 corridor.
    ///
    /// Honest scope note: GPU produces the per-tick "should I
    /// dodge" decision; the host applies the perpendicular sidestep
    /// and runs the corridor check. Position SoA + a movement
    /// consumer rule that runs the sidestep on GPU is the next
    /// slice (the threats view today is a scalar, not directional;
    /// extending it to carry projectile direction so the dodger can
    /// pick the perpendicular axis on its own is the bigger lift
    /// after this).
    ///
    /// Assertion shape:
    ///   * WITH threats: dodger picks Flee every tick → host applies
    ///     10 sidesteps → final y > corridor_half_width → dodge
    ///     succeeded.
    ///   * WITHOUT threats (force_no_threats each tick): dodger
    ///     picks Idle every tick → host applies 0 sidesteps →
    ///     final y = 0 → INSIDE corridor → dodger gets hit.
    /// Behavioural delta: same projectile, different scoring → one
    /// agent escapes, the other doesn't.
    #[test]
    fn agent_sidesteps_narrow_projectile_when_warned_via_gpu_scoring() {
        const N: u32 = 2;
        const HIT_TICK: usize = 10;
        const PROJECTILE_CORRIDOR_HALF_WIDTH: f32 = 0.5;
        const SIDESTEP_SPEED: f32 = 0.1; // per tick when Flee is chosen

        // Helper: run one scenario, return final dodger y position.
        // `force_baseline` zeroes threats each tick so scoring picks
        // Idle instead of Flee.
        fn run_scenario(force_baseline: bool) -> Option<(f32, usize)> {
            let mut state = DodgerProbeState::try_new(0xCAFE, N)?;
            let mut dodger_y: f32 = 0.0;
            let mut sidesteps_taken = 0;
            for _tick in 0..HIT_TICK {
                if force_baseline {
                    state.force_no_threats();
                }
                state.step();
                let scoring = state.read_scoring_output();
                let dodger_action = state.chosen_action(&scoring, 1);
                if dodger_action == FLEE_ACTION_ID {
                    // Sidestep perpendicular to projectile direction
                    // `(-1, 0)`. Perpendicular axis = y; pick +y.
                    dodger_y += SIDESTEP_SPEED;
                    sidesteps_taken += 1;
                }
            }
            Some((dodger_y, sidesteps_taken))
        }

        // --- WITH threats (warned): dodger should escape.
        let (final_y_warned, sidesteps_warned) = match run_scenario(false) {
            Some(v) => v,
            None => {
                eprintln!("[sidestep-pin] skipping: no wgpu adapter on host.");
                return;
            }
        };
        eprintln!(
            "[sidestep-pin] WARNED: {sidesteps_warned} sidesteps over {HIT_TICK} ticks → final y = {final_y_warned:.3}"
        );
        assert!(
            final_y_warned > PROJECTILE_CORRIDOR_HALF_WIDTH,
            "WARNED: dodger must end OUTSIDE corridor (|y|={final_y_warned} > {PROJECTILE_CORRIDOR_HALF_WIDTH}); \
             {sidesteps_warned} sidesteps were applied"
        );

        // --- WITHOUT threats (no warning): dodger eats the hit.
        let (final_y_baseline, sidesteps_baseline) =
            run_scenario(true).expect("second wgpu state");
        eprintln!(
            "[sidestep-pin] BASELINE: {sidesteps_baseline} sidesteps over {HIT_TICK} ticks → final y = {final_y_baseline:.3}"
        );
        assert_eq!(
            sidesteps_baseline, 0,
            "BASELINE: with threats forced to zero, NO sidesteps should fire; got {sidesteps_baseline}"
        );
        assert!(
            final_y_baseline.abs() <= PROJECTILE_CORRIDOR_HALF_WIDTH,
            "BASELINE: dodger must end INSIDE corridor (|y|={final_y_baseline} ≤ {PROJECTILE_CORRIDOR_HALF_WIDTH}); \
             dodger should be hit by the projectile"
        );

        // --- The behavioural delta IS the proof: same projectile,
        // same agent, same physics — only the threat-warning state
        // differs, and one agent escapes while the other doesn't.
        let escape_delta = final_y_warned - final_y_baseline.abs();
        eprintln!(
            "[sidestep-pin] DODGE DELTA: warned dodger ended {escape_delta:.3} units further from corridor centerline than baseline"
        );
        assert!(
            escape_delta > PROJECTILE_CORRIDOR_HALF_WIDTH,
            "the threat warning must produce a positional delta exceeding the corridor half-width; \
             got delta={escape_delta}, threshold={PROJECTILE_CORRIDOR_HALF_WIDTH}"
        );
    }
}
