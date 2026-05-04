//! Per-fixture runtime for `assets/sim/pair_scoring_probe.sim` —
//! end-to-end probe of PAIR-FIELD SCORING (spec §8.3).
//!
//! ## Outcome (a) FULL FIRE
//!
//! As of the 2026-05-04 close of Gap A (per-pair candidate read
//! folding in `cg::lower::expr::lower_namespace_call`) and Gap B
//! (binder LocalRef shadowing in `cg::lower::scoring::lower_standard_row`),
//! the verb `Heal(self, target: Agent) = ... score (1000.0 -
//! agents.cooldown_next_ready_tick(target))` lowers to a per-pair
//! scoring kernel that:
//!
//!   - Iterates `per_pair_candidate: u32 = 0u..cfg.agent_cap` inside
//!     the row body
//!   - Reads `agent_cooldown_next_ready_tick[per_pair_candidate]`
//!     directly (no `target_expr_<N>` indirection — the structural
//!     fold collapsed the per-pair-id read to `AgentRef::PerPairCandidate`)
//!   - Sets `best_target = per_pair_candidate` on argmax wins
//!   - Emits `ActionSelected{actor=agent_id, action_id=0, target=best_target}`
//!     per gated agent slot
//!
//! With `cooldown_next_ready_tick[N] = N * 10`, the inverted score
//! `1000.0 - cooldown_next_ready_tick(target)` peaks at `target=0`
//! (lowest cooldown value → highest utility). Every healer therefore
//! picks slot 0 each tick. Mask gating (`target != self`) suppresses
//! agent 0's own action (its only checked candidate `cand=0` fails
//! the predicate — the mask kernel still has the `mask_0_k=1u` TODO
//! placeholder), so agents 1..7 emit one `ActionSelected` per tick.
//! Chronicle physics emits one `Healed{healer, target=0, amount=5.0}`
//! per ActionSelected. Fold accumulates `received[0] += amount` per
//! event. Expected after `TICKS=100`:
//!
//!   received[0] = TICKS * (AGENT_COUNT - 1) * heal.amount
//!               = 100 * 7 * 5.0 = 3500.0
//!   received[N] = 0.0 for N > 0.
//!
//! ## Per-tick chain
//!
//!   1. clear event_tail
//!   2. clear mask_0_bitmap (stale bits would gate the wrong actors)
//!   3. mask_verb_Heal — atomic-OR per (agent, cand=0) pair where
//!      `target != self`. Sets bits 1..7 with the `mask_0_k=1u`
//!      placeholder in place; bit 0 stays clear.
//!   4. scoring — per-actor argmax over per_pair_candidate, atomic-
//!      append one ActionSelected per gated agent slot.
//!   5. seed_indirect_0 — populates indirect_args_0 from event_tail.
//!   6. physics_verb_chronicle_Heal — gates on `action_id == 0u`,
//!      emits `Healed{healer=actor, target=target, amount=config.heal.amount}`.
//!   7. fold_received — per-Healed accumulator into the per-target
//!      `received` view storage.
//!
//! See `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md` for
//! the full gap-chain history.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the pair-scoring probe. Owns the agent SoA,
/// the event ring, the view storage, the per-mask bitmap, the
/// scoring_output buffer, and per-kernel cfg uniforms.
pub struct PairScoringProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. Initialised all-1. Not consumed by any
    /// emitted kernel today (the verb's mask predicate doesn't gate
    /// on `self.alive`), but kept for parity with sibling probes.
    #[allow(dead_code)]
    agent_alive_buf: wgpu::Buffer,
    /// `cooldown_next_ready_tick` SoA — read by the scoring kernel
    /// via `agent_cooldown_next_ready_tick[per_pair_candidate]`.
    /// Initialised to `ready_at[N] = N * 10` so slot 0 wins the
    /// inverted-score argmax (lowest cooldown → highest utility).
    agent_cooldown_next_ready_tick_buf: wgpu::Buffer,

    // -- Mask bitmap (one bit per agent in u32 words) --
    /// `mask_0_bitmap`: u32 array of length `ceil(agent_count / 32)`.
    /// `mask_verb_Heal` atomicOrs into the appropriate word/bit per
    /// agent slot whose mask predicate fires for any candidate.
    mask_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    /// Layout: `[best_action, best_target, bitcast<u32>(best_utility),
    /// 0]` per agent slot. Sized at `agent_count * 4`. Not read back
    /// from the host; observable surfaces via the chronicle + fold
    /// downstream.
    scoring_output_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-target healing-received accumulator (f32). Fed by `Healed`
    /// (kind tag = 1u). Under FULL FIRE, slot 0 absorbs every healer's
    /// emission per tick.
    received: ViewStorage,
    received_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl PairScoringProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` unused by any compiled kernel today,
        // kept for parity with sibling probes. Initialised all-1.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Cooldown SoA — staggered init `ready_at[N] = N * 10` so
        // the per-pair argmax picks slot 0 (lowest cooldown → highest
        // inverted score under `1000.0 - cooldown_next_ready_tick`).
        let cooldown_init: Vec<u32> = (0..agent_count).map(|n| n * 10).collect();
        let agent_cooldown_next_ready_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::agent_cooldown_next_ready_tick"),
                contents: bytemuck::cast_slice(&cooldown_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Mask bitmap — `ceil(agent_count / 32)` u32 words. Cleared
        // each tick before mask_verb_Heal runs so stale bits don't
        // gate the wrong actors.
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair_scoring_probe_runtime::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::mask_0_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_bytes = (agent_count as u64) * 4 * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair_scoring_probe_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Event ring + received view storage (per-agent f32, no anchor,
        // no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "pair_scoring_probe_runtime");
        let received = ViewStorage::new(
            &gpu,
            "pair_scoring_probe_runtime::received",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mk_cfg = |label: &str| -> wgpu::Buffer {
            let init = mask_verb_Heal::MaskVerbHealCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mask_cfg_buf = mk_cfg("pair_scoring_probe_runtime::mask_cfg");
        let scoring_cfg_buf = mk_cfg("pair_scoring_probe_runtime::scoring_cfg");
        let seed_cfg_buf = mk_cfg("pair_scoring_probe_runtime::seed_cfg");

        let chronicle_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let chronicle_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::chronicle_cfg"),
                contents: bytemuck::bytes_of(&chronicle_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let received_cfg_init = fold_received::FoldReceivedCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let received_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::received_cfg"),
                contents: bytemuck::bytes_of(&received_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            agent_cooldown_next_ready_tick_buf,
            mask_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            event_ring,
            received,
            received_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-target healing-received accumulator (one f32 per slot).
    /// Under FULL FIRE: `received[0] = TICKS * (AGENT_COUNT - 1) *
    /// heal.amount`, all other slots stay at 0.0.
    pub fn received(&mut self) -> &[f32] {
        self.received.readback(&self.gpu)
    }

    /// Read back the per-actor `[best_action, best_target,
    /// bitcast<u32>(best_utility), 0]` scoring output. Useful for
    /// diagnosing which intermediate stage drops events: under FULL
    /// FIRE every slot N>0 should have `(action_id=0, target=0)`;
    /// slot 0 stays at `(NO_ACTION, NO_AGENT)` because its mask bit
    /// never sets (the `cand=0, agent=0` pair fails `target != self`
    /// and `mask_0_k=1u` doesn't iterate further candidates).
    pub fn scoring_output(&mut self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair_scoring_probe_runtime::scoring_output_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pair_scoring_probe_runtime::scoring_readback"),
                });
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_buf,
            0,
            &staging,
            0,
            bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("poll");
        let _ = receiver.recv().expect("scoring_output map_async result");
        let mapped = slice.get_mapped_range();
        let words: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        words
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

impl CompiledSim for PairScoringProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pair_scoring_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1b) Per-tick clear of mask_0_bitmap so stale bits from the
        // last tick don't gate the wrong actors.
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_bitmap_buf,
            0,
            mask_bytes.max(4),
        );

        // (2) mask_verb_Heal — per-pair (agent, cand=0) atomicOr into
        // mask_0_bitmap when `target != self`. Note: the emitted body
        // hardcodes `mask_0_k = 1u` (TODO task-5.7), so only cand=0
        // is checked per actor; agent 0's bit therefore stays clear
        // (predicate fails for agent=0, cand=0).
        let mask_cfg = mask_verb_Heal::MaskVerbHealCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = mask_verb_Heal::MaskVerbHealBindings {
            mask_0_bitmap: &self.mask_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_mask_verb_heal(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) scoring — per-actor argmax over per_pair_candidate,
        // emits one ActionSelected{actor, action_id=0, target=best}
        // per gated agent slot (agents 1..7 under the mask gate).
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_cooldown_next_ready_tick: &self
                .agent_cooldown_next_ready_tick_buf,
            mask_0_bitmap: &self.mask_bitmap_buf,
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

        // (4) physics_verb_chronicle_Heal — per-event handler. Reads
        // each ActionSelected slot the scoring kernel just emitted,
        // gates on `action_id == 0u` (Heal's id), and emits
        // `Healed{healer=actor, target=target, amount=config.heal.amount}`.
        // Dispatched with `event_count = agent_count` because the
        // scoring kernel emits at most one ActionSelected per actor
        // per tick.
        let chronicle_cfg =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad0: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_heal(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) seed_indirect_0 — populates indirect_args_0 from
        // event_tail (kept for parity).
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
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

        // (6) fold_received — RMW per Healed event (kind tag = 1u).
        // event_count is sized to cover both event kinds (one
        // ActionSelected + up to one Healed per actor): ~ 2 ×
        // agent_count slots/tick. The per-handler tag check ignores
        // ActionSelected.
        let event_count_estimate = self.agent_count * 2;
        let received_cfg = fold_received::FoldReceivedCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.received_cfg_buf,
            0,
            bytemuck::bytes_of(&received_cfg),
        );
        let received_bindings = fold_received::FoldReceivedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.received.primary(),
            view_storage_anchor: self.received.anchor(),
            view_storage_ids: self.received.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.received_cfg_buf,
        };
        dispatch::dispatch_fold_received(
            &mut self.cache,
            &received_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.received.mark_dirty();
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
    Box::new(PairScoringProbeState::new(seed, agent_count))
}
