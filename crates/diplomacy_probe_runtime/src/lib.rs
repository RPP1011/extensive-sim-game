//! Per-fixture runtime for `assets/sim/diplomacy_probe.sim` — the
//! smallest end-to-end probe combining the recently-landed surfaces
//! into a real diplomacy / coalition-formation pattern.
//!
//! Mirrors `abilities_runtime` (multi-verb cascade + scoring + 2
//! chronicles + 2 folds) AND `tom_probe_runtime` (pair_map u32 view
//! with atomicOr fold) AND `cooldown_probe_runtime` (single
//! per-agent physics + 1 fold). The composition is novel:
//!
//! ## Per-tick chain
//!
//!  1. Clears — event_tail = 0; mask_0_bitmap + mask_1_bitmap zeroed.
//!  2. ObserveAndAct physics (per_agent, no `on Tick` event source) —
//!     each alive Diplomat emits one `Observed { observer=self,
//!     target=self, signal=1 }` per tick into the ring (kind 1u).
//!  3. Mask round — `fused_mask_verb_ProposeAlliance` writes BOTH
//!     mask_0 (ProposeAlliance, `tick % 3 == 0`) AND mask_1 (Betray,
//!     `tick % 3 != 0`) bitmaps in one PerAgent pass. EXACTLY ONE
//!     bit is set per agent per tick (the predicates are disjoint).
//!  4. Scoring — argmax over the 2 competing rows. The mask gate
//!     filters: only the active verb's row contributes. Whichever
//!     verb is masked-in wins (utility = 1.0 vs 0.5; only one row
//!     ever passes the mask check). Emits one `ActionSelected`
//!     (kind 4u) per agent.
//!  5. ProposeAlliance chronicle — reads ActionSelected slots, gates
//!     on `action_id == 0u`, emits `AllianceProposed` (kind 2u).
//!     Fires on ticks 0, 3, 6, ... (34/100 ticks).
//!  6. Betray chronicle — reads ActionSelected slots, gates on
//!     `action_id == 1u`, emits `Betrayed` (kind 3u). Fires on ticks
//!     1, 2, 4, 5, 7, 8, ... (66/100 ticks).
//!  7. seed_indirect_0 — keeps indirect-args buffer warm.
//!  8. fold_trust — atomicOr u32 fold on `Observed` (kind 1u). Per-
//!     (observer, target) pair_map; observer == target == self →
//!     diagonal slots converge to 1u.
//!  9. fold_alliances_proposed — RMW per AllianceProposed (kind 2u).
//! 10. fold_betrayals_committed — RMW per Betrayed (kind 3u).
//!
//! ## Observable
//!
//!  - `trust(i, i)` = 1u for every i; off-diagonal = 0u (placeholder
//!    self-routing per tom_probe shape).
//!  - `alliances_proposed[i]` = ceil(TICKS / 3) = 34.0 for every slot
//!    at TICKS=100, observation_tick_mod=3. (Ticks 0,3,...,99 → 34.)
//!  - `betrayals_committed[i]` = TICKS - 34 = 66.0 for every slot.
//!
//! See `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the diplomacy probe. Owns:
///   - Diplomat SoA (alive only — pos/vel are declaration-only here)
///   - Two mask bitmaps (one per verb)
///   - Scoring output (4 × u32 per agent)
///   - Shared event ring (3 producer kinds: Observed, AllianceProposed,
///     Betrayed; plus ActionSelected from scoring; plus the chronicles
///     consume ActionSelected and emit the verb-event kinds)
///   - `trust` pair_map u32 storage (agent_count × agent_count)
///     allocated locally (mirrors tom_probe — readback returns &[u32])
///   - `alliances_proposed` + `betrayals_committed` ViewStorage (f32)
///   - Per-kernel cfg uniforms
pub struct DiplomacyProbeState {
    gpu: GpuContext,

    // -- Agent SoA (only `alive` is read by any compiled kernel) --
    agent_alive_buf: wgpu::Buffer,

    // -- Mask bitmaps (Strike-style: one per verb in source order) --
    mask_0_bitmap_buf: wgpu::Buffer, // ProposeAlliance
    mask_1_bitmap_buf: wgpu::Buffer, // Betray
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,

    // -- Shared event ring (Observed / AllianceProposed / Betrayed /
    //    ActionSelected — 4 kinds threading through one ring) --
    event_ring: EventRing,

    // -- trust (pair_map u32 — atomicOr fold).
    //    Allocated locally so the host-side readback is &[u32], not the
    //    f32 bitcast round-trip ViewStorage forces.
    trust_primary: wgpu::Buffer,
    trust_staging: wgpu::Buffer,
    trust_cache: Vec<u32>,
    trust_dirty: bool,
    trust_cfg_buf: wgpu::Buffer,

    // -- alliances_proposed (f32, no decay) --
    alliances_proposed: ViewStorage,
    alliances_proposed_cfg_buf: wgpu::Buffer,

    // -- betrayals_committed (f32, no decay) --
    betrayals_committed: ViewStorage,
    betrayals_committed_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    observe_cfg_buf: wgpu::Buffer,
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_propose_cfg_buf: wgpu::Buffer,
    chronicle_betray_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl DiplomacyProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Diplomat SoA — only `alive` is read by ObserveAndAct's
        // `where (self.alive)` gate. All-1 init so every slot fires.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Two mask bitmaps — cleared each tick before the fused mask
        // kernel runs.
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mk_mask = |label: &str| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: mask_bitmap_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mask_0_bitmap_buf = mk_mask("diplomacy_probe_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("diplomacy_probe_runtime::mask_1_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_bytes = (agent_count as u64) * 4 * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diplomacy_probe_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared event ring.
        let event_ring = EventRing::new(&gpu, "diplomacy_probe_runtime");

        // trust pair_map storage — `agent_count × agent_count × u32`.
        // Allocated locally (NOT through ViewStorage) so the host-side
        // readback returns &[u32] without a bitcast round-trip. Same
        // shape as tom_probe_runtime::beliefs_primary.
        let trust_slot_count = (agent_count as u64) * (agent_count as u64);
        let trust_bytes = (trust_slot_count * 4).max(16);
        let trust_primary = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diplomacy_probe_runtime::trust_primary"),
            size: trust_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trust_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diplomacy_probe_runtime::trust_staging"),
            size: trust_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // alliances_proposed + betrayals_committed — f32, no decay,
        // no top-K → no anchor / no ids.
        let alliances_proposed = ViewStorage::new(
            &gpu,
            "diplomacy_probe_runtime::alliances_proposed",
            agent_count,
            false,
            false,
        );
        let betrayals_committed = ViewStorage::new(
            &gpu,
            "diplomacy_probe_runtime::betrayals_committed",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let observe_cfg_init = physics_ObserveAndAct::PhysicsObserveAndActCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let observe_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::observe_cfg"),
                contents: bytemuck::bytes_of(&observe_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let mask_cfg_init =
            fused_mask_verb_ProposeAlliance::FusedMaskVerbProposeAllianceCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0, _pad: 0,
            };
        let mask_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::mask_cfg"),
                contents: bytemuck::bytes_of(&mask_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_propose_cfg_init =
            physics_verb_chronicle_ProposeAlliance::PhysicsVerbChronicleProposeAllianceCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let chronicle_propose_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::chronicle_propose_cfg"),
                contents: bytemuck::bytes_of(&chronicle_propose_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_betray_cfg_init =
            physics_verb_chronicle_Betray::PhysicsVerbChronicleBetrayCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let chronicle_betray_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::chronicle_betray_cfg"),
                contents: bytemuck::bytes_of(&chronicle_betray_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let trust_cfg_init = fold_trust::FoldTrustCfg {
            event_count: 0,
            tick: 0,
            // pair_map → second_key_pop = agent_count so the fold body
            // composes `view_storage_primary[k1 * second_key_pop + k2]`
            // landing the diagonal at index `i * N + i`.
            second_key_pop: agent_count,
            _pad: 0,
        };
        let trust_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::trust_cfg"),
                contents: bytemuck::bytes_of(&trust_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let alliances_cfg_init = fold_alliances_proposed::FoldAlliancesProposedCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let alliances_proposed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::alliances_cfg"),
                contents: bytemuck::bytes_of(&alliances_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let betrayals_cfg_init = fold_betrayals_committed::FoldBetrayalsCommittedCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let betrayals_committed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("diplomacy_probe_runtime::betrayals_cfg"),
                contents: bytemuck::bytes_of(&betrayals_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            event_ring,
            trust_primary,
            trust_staging,
            trust_cache: vec![0u32; trust_slot_count as usize],
            trust_dirty: false,
            trust_cfg_buf,
            alliances_proposed,
            alliances_proposed_cfg_buf,
            betrayals_committed,
            betrayals_committed_cfg_buf,
            observe_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_propose_cfg_buf,
            chronicle_betray_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-(observer, target) trust bitset, flattened row-major: slot
    /// `[observer * agent_count + target]` holds the OR-folded trust
    /// signal bits the observer has seen about the target. Length =
    /// `agent_count × agent_count`.
    pub fn trust(&mut self) -> &[u32] {
        if self.trust_dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("diplomacy_probe_runtime::trust::readback"),
                },
            );
            let bytes = (self.trust_cache.len() as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.trust_primary,
                0,
                &self.trust_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.trust_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.trust_cache.copy_from_slice(raw);
            drop(mapped);
            self.trust_staging.unmap();
            self.trust_dirty = false;
        }
        &self.trust_cache
    }

    /// Per-Diplomat alliance proposal counter (one f32 per slot). With
    /// observation_tick_mod = 3 and TICKS = 100, ticks 0,3,6,...,99
    /// are observation ticks → 34 fires per slot → 34.0.
    pub fn alliances_proposed(&mut self) -> &[f32] {
        self.alliances_proposed.readback(&self.gpu)
    }

    /// Per-Diplomat betrayal counter. TICKS - 34 = 66 fires per slot
    /// → 66.0 at TICKS = 100.
    pub fn betrayals_committed(&mut self) -> &[f32] {
        self.betrayals_committed.readback(&self.gpu)
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

impl CompiledSim for DiplomacyProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("diplomacy_probe_runtime::step"),
                });

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_0_bitmap_buf,
            0,
            mask_bytes.max(4),
        );
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_1_bitmap_buf,
            0,
            mask_bytes.max(4),
        );

        // (2) ObserveAndAct — per-Diplomat; emits one Observed event
        // per tick into the ring (kind 1u). observer == target == self
        // placeholder routing (real spatial broadcast is a follow-up).
        let observe_cfg = physics_ObserveAndAct::PhysicsObserveAndActCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.observe_cfg_buf,
            0,
            bytemuck::bytes_of(&observe_cfg),
        );
        let observe_bindings = physics_ObserveAndAct::PhysicsObserveAndActBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_alive: &self.agent_alive_buf,
            cfg: &self.observe_cfg_buf,
        };
        dispatch::dispatch_physics_observeandact(
            &mut self.cache,
            &observe_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        // ObserveAndAct's emit-Observed body adds one slot per alive
        // agent (worst case: all alive). The host-side EventRing tail
        // estimate tracks this so downstream chronicle / fold dispatches
        // can size their `event_count` from `event_ring.tail_value()`
        // without hard-coding `agent_count * N_producers`. See Gap #2 in
        // `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`.
        self.event_ring.note_emits(self.agent_count);

        // (3) Mask round — `fused_mask_verb_ProposeAlliance` writes
        // BOTH mask_0 (ProposeAlliance, `tick % 3 == 0`) AND mask_1
        // (Betray, `tick % 3 != 0`) bitmaps. Predicates are disjoint
        // so EXACTLY ONE bit is set per agent per tick.
        let mask_cfg =
            fused_mask_verb_ProposeAlliance::FusedMaskVerbProposeAllianceCfg {
                agent_cap: self.agent_count,
                tick: self.tick as u32,
                seed: 0, _pad: 0,
            };
        self.gpu
            .queue
            .write_buffer(&self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg));
        let mask_bindings =
            fused_mask_verb_ProposeAlliance::FusedMaskVerbProposeAllianceBindings {
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                cfg: &self.mask_cfg_buf,
            };
        dispatch::dispatch_fused_mask_verb_proposealliance(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) Scoring — argmax over the 2 competing rows. Mask gates
        // filter out masked-out rows, so whichever verb's predicate
        // is true wins. Emits one ActionSelected (kind 4u) per agent.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4b) Note that the scoring kernel emits one ActionSelected
        // per agent into the ring; bump the host-side EventRing tail
        // estimate so the downstream chronicles' `event_count` covers
        // both the Observed slots (from step 2) and the ActionSelected
        // slots (from step 4). Closes Gap #2 from
        // `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` —
        // replaces the manual `agent_count * 2` constant that would
        // silently miss events if a future probe added a third
        // producer round.
        self.event_ring.note_emits(self.agent_count);

        // (5) ProposeAlliance chronicle — reads ActionSelected slots,
        // gates on `action_id == 0u`, emits AllianceProposed (kind 2u).
        // event_count is sized via `event_ring.tail_value()`, the
        // host-tracked sum of every prior `note_emits` call this tick
        // (Observed + ActionSelected = `agent_count * 2` so far).
        // The per-handler tag check inside the body skips
        // non-ActionSelected slots; passing the upper bound is safe
        // and adapts automatically to new producers added upstream.
        let chronicle_event_count = self.event_ring.tail_value();
        let propose_chronicle_cfg =
            physics_verb_chronicle_ProposeAlliance::PhysicsVerbChronicleProposeAllianceCfg {
                event_count: chronicle_event_count,
                tick: self.tick as u32,
                seed: 0,
                _pad0: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_propose_cfg_buf,
            0,
            bytemuck::bytes_of(&propose_chronicle_cfg),
        );
        let propose_chronicle_bindings =
            physics_verb_chronicle_ProposeAlliance::PhysicsVerbChronicleProposeAllianceBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_propose_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_proposealliance(
            &mut self.cache,
            &propose_chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count.saturating_mul(2),
        );
        // The ProposeAlliance chronicle's gate-passes slots produce
        // up to one AllianceProposed event per agent (whichever
        // ActionSelected matched `action_id == 0u`). Bump the tail
        // estimate so the downstream Betray chronicle + folds see
        // these slots when sizing their `event_count`.
        self.event_ring.note_emits(self.agent_count);

        // (6) Betray chronicle — reads ActionSelected slots, gates on
        // `action_id == 1u`, emits Betrayed (kind 3u). event_count is
        // again sourced from `event_ring.tail_value()` to cover every
        // upstream emit (Observed + ActionSelected + AllianceProposed).
        let betray_chronicle_event_count = self.event_ring.tail_value();
        let betray_chronicle_cfg =
            physics_verb_chronicle_Betray::PhysicsVerbChronicleBetrayCfg {
                event_count: betray_chronicle_event_count,
                tick: self.tick as u32,
                seed: 0,
                _pad0: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_betray_cfg_buf,
            0,
            bytemuck::bytes_of(&betray_chronicle_cfg),
        );
        let betray_chronicle_bindings =
            physics_verb_chronicle_Betray::PhysicsVerbChronicleBetrayBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_betray_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_betray(
            &mut self.cache,
            &betray_chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count.saturating_mul(2),
        );
        // Betray chronicle's gate-passes slots produce up to one
        // Betrayed event per agent. Tail estimate after this round
        // covers everything the folds will see (Observed +
        // ActionSelected + AllianceProposed + Betrayed = up to 4N).
        self.event_ring.note_emits(self.agent_count);

        // (7) seed_indirect_0 — keeps indirect args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (8-10) Three folds reading the SAME ring, partitioned by
        // per-handler tag filter (kind 1u/2u/3u). event_count sourced
        // from `event_ring.tail_value()` — the host-side sum of every
        // prior `note_emits` call this tick (Observed + ActionSelected
        // + AllianceProposed + Betrayed, up to 4N for the diplomacy
        // probe). Closes Gap #2 from
        // `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`:
        // pre-fix this was a hard-coded `agent_count * 4` constant
        // that would silently miss events if a future cascade round
        // added a producer; post-fix the estimate adapts automatically
        // to every `note_emits` call upstream.
        let event_count_estimate = self.event_ring.tail_value();

        // (8) fold_trust — atomicOr u32 fold filtered on tag 1u
        // (Observed). second_key_pop = agent_count so the
        // `[observer * N + target]` index lands at the pair_map flat
        // offset.
        let trust_cfg = fold_trust::FoldTrustCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.trust_cfg_buf,
            0,
            bytemuck::bytes_of(&trust_cfg),
        );
        let trust_bindings = fold_trust::FoldTrustBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.trust_primary,
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.trust_cfg_buf,
        };
        dispatch::dispatch_fold_trust(
            &mut self.cache,
            &trust_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // (9) fold_alliances_proposed — RMW per AllianceProposed
        // (kind 2u). Per-handler tag filter rejects all other kinds.
        let alliances_cfg = fold_alliances_proposed::FoldAlliancesProposedCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.alliances_proposed_cfg_buf,
            0,
            bytemuck::bytes_of(&alliances_cfg),
        );
        let alliances_bindings =
            fold_alliances_proposed::FoldAlliancesProposedBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.alliances_proposed.primary(),
                view_storage_anchor: self.alliances_proposed.anchor(),
                view_storage_ids: self.alliances_proposed.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.alliances_proposed_cfg_buf,
            };
        dispatch::dispatch_fold_alliances_proposed(
            &mut self.cache,
            &alliances_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // (10) fold_betrayals_committed — RMW per Betrayed (kind 3u).
        // Per-handler tag filter rejects all other kinds.
        let betrayals_cfg = fold_betrayals_committed::FoldBetrayalsCommittedCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.betrayals_committed_cfg_buf,
            0,
            bytemuck::bytes_of(&betrayals_cfg),
        );
        let betrayals_bindings =
            fold_betrayals_committed::FoldBetrayalsCommittedBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.betrayals_committed.primary(),
                view_storage_anchor: self.betrayals_committed.anchor(),
                view_storage_ids: self.betrayals_committed.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.betrayals_committed_cfg_buf,
            };
        dispatch::dispatch_fold_betrayals_committed(
            &mut self.cache,
            &betrayals_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // kick_snapshot intentionally skipped — host-side artefact,
        // same skip pattern as verb_probe_runtime / abilities_runtime.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.trust_dirty = true;
        self.alliances_proposed.mark_dirty();
        self.betrayals_committed.mark_dirty();
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
    Box::new(DiplomacyProbeState::new(seed, agent_count))
}
