//! Per-fixture runtime for `assets/sim/megaswarm_10000.sim` —
//! the SIXTEENTH real gameplay-shaped fixture and the third SCALE-UP
//! for pair-field scoring (10000 agents → 100M mask cells per tick
//! per verb, 100x mass_battle_100v100 and 10x megaswarm_1000).
//!
//! Composition: 5000 Red (level=1) + 5000 Blue (level=2) on a single
//! Agent SoA. TWO team-mirrored verbs (RedStrike + BlueStrike) so
//! the per-actor argmax can score-out same-team targets via the
//! `if (target.level == <enemy>) ... else { -1e9 }` pattern.
//!
//! ## Why chunked PerPair dispatches
//!
//! The compiler-emitted PerPair mask kernel dispatches one thread per
//! (actor, candidate) pair = `agent_cap²` threads. At agent_cap=10000
//! that's 100 000 000 threads = 1 562 500 workgroups (workgroup_size=64).
//! That EXCEEDS the typical wgpu/Vulkan/Metal/D3D12 hardware limit
//! `max_compute_workgroups_per_dimension` (65535) by ~24x.
//!
//! Workaround: split the 1D pair-grid dispatch into ~32 chunks of
//! `MASK_CHUNK_WORKGROUPS = 50000` workgroups each (= 3 200 000
//! pairs per chunk). Each chunk uses a dedicated cfg uniform with its
//! own `_pad0` value (repurposed by the dsl_compiler PerPair preamble
//! as a `pair_offset` chunk base — see
//! `mask_predicate_per_pair_body_with_prefix` in `kernel.rs`). All
//! chunks of all verbs are recorded into one encoder + one submit per
//! tick — no extra round-trips.
//!
//! ## Per-tick chain
//!
//!   1. clear_tail + clear both mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_RedStrike — PerPair, ~32 chunked dispatches
//!      of 50000 workgroups each, walking pair_offset 0..agent_cap²,
//!      writes both mask_0 (RedStrike) + mask_1 (BlueStrike).
//!   3. scoring — PerAgent argmax over 2 candidate verbs per actor;
//!      inner loop over `agent_cap` candidates per pair-field row.
//!      Emits one ActionSelected{actor, action_id, target} per
//!      gated agent.
//!   4. physics_verb_chronicle_RedStrike — gates action_id==0u,
//!      emits Damaged (Red attacks Blue).
//!   5. physics_verb_chronicle_BlueStrike — gates action_id==1u,
//!      emits Damaged (Blue attacks Red).
//!   6. physics_ApplyDamage — PerEvent kernel.
//!   7. seed_indirect_0 — keeps indirect-args buffer warm.
//!   8. fold_damage_dealt — per-source f32 accumulator.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-team agent populations. 5000 + 5000 = 10000.
pub const PER_TEAM: u32 = 5000;
pub const TOTAL_AGENTS: u32 = PER_TEAM * 2;

/// All combatants spawn with HP=100. Strike does 1000 dmg (single-shot
/// kill) — see fixture header for why.
pub const COMBATANT_HP: f32 = 100.0;

/// Workgroups per chunked PerPair dispatch. The wgpu/Vulkan limit for
/// `max_compute_workgroups_per_dimension` is typically 65535. 50000
/// leaves headroom for adapters that report lower limits (e.g. older
/// integrated GPUs cap at 32768). At workgroup_size=64 each chunk
/// processes 50000 * 64 = 3 200 000 pairs.
pub const MASK_CHUNK_WORKGROUPS: u32 = 50_000;

/// Pairs processed per chunk = MASK_CHUNK_WORKGROUPS * 64 (the
/// workgroup_size baked into every PerPair kernel by the compiler).
pub const PAIRS_PER_CHUNK: u32 = MASK_CHUNK_WORKGROUPS * 64;

/// Encode team as the per-agent `level` slot.
/// Red = 1u, Blue = 2u. Matches the encoding documented in the .sim.
fn level_for_team(team: u32) -> u32 {
    team + 1
}

/// Per-fixture state for the megaswarm.
pub struct Megaswarm10000State {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    RedStrike=0, BlueStrike=1) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    // Mask cfg: one buffer per chunk slot. `mask_chunk_cfgs[i]` carries
    // `_pad0 = i * PAIRS_PER_CHUNK` (the chunk's pair_offset). `tick`
    // is rewritten every tick for every chunk slot.
    mask_chunk_cfgs: Vec<wgpu::Buffer>,
    /// Number of chunks = ceil(agent_cap² / PAIRS_PER_CHUNK).
    num_mask_chunks: u32,

    scoring_cfg_buf: wgpu::Buffer,
    chronicle_red_strike_cfg_buf: wgpu::Buffer,
    chronicle_blue_strike_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl Megaswarm10000State {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Per-agent SoA inits. Layout:
        //   slots 0..PER_TEAM         → Red team   (level=1)
        //   slots PER_TEAM..2*PER_TEAM → Blue team (level=2)
        let mut hp_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        let mut alive_init: Vec<u32> = Vec::with_capacity(agent_count as usize);
        let mut level_init: Vec<u32> = Vec::with_capacity(agent_count as usize);

        for team in 0..2u32 {
            for _ in 0..PER_TEAM {
                hp_init.push(COMBATANT_HP);
                alive_init.push(1);
                level_init.push(level_for_team(team));
            }
        }
        debug_assert_eq!(hp_init.len(), agent_count as usize);

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Two mask bitmaps (RedStrike + BlueStrike). Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("megaswarm_10000::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("megaswarm_10000::mask_1_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("megaswarm_10000::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "megaswarm_10000");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "megaswarm_10000::damage_dealt",
            agent_count,
            false,
            false,
        );

        // -- Mask cfg chunks. One per ~PAIRS_PER_CHUNK slice of the
        //    agent_cap² pair grid. `_pad0` is the per-chunk
        //    pair_offset; the compiler-emitted PerPair preamble does
        //    `let pair = gid.x + cfg._pad0;`.
        let total_pairs = agent_count as u64 * agent_count as u64;
        let num_mask_chunks =
            ((total_pairs + (PAIRS_PER_CHUNK as u64) - 1) / PAIRS_PER_CHUNK as u64) as u32;
        let mut mask_chunk_cfgs: Vec<wgpu::Buffer> = Vec::with_capacity(num_mask_chunks as usize);
        for chunk_idx in 0..num_mask_chunks {
            let pair_offset = chunk_idx * PAIRS_PER_CHUNK;
            // NOTE: Rust struct field is `_pad` (Generic cfg shape) but
            // it maps by-position to the WGSL struct's `_pad0` field —
            // see dsl_compiler `build_generic_cfg_struct_decl`. The
            // PerPair preamble reads `cfg._pad0` from WGSL.
            let cfg_init = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: pair_offset,
            };
            let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("megaswarm_10000::mask_cfg_chunk_{chunk_idx}")),
                contents: bytemuck::bytes_of(&cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            mask_chunk_cfgs.push(buf);
        }

        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_red_strike_cfg_init =
            physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_red_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::chronicle_red_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_red_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_blue_strike_cfg_init =
            physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_blue_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::chronicle_blue_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_blue_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_10000::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_level_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            mask_chunk_cfgs,
            num_mask_chunks,
            scoring_cfg_buf,
            chronicle_red_strike_cfg_buf,
            chronicle_blue_strike_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn read_level(&self) -> Vec<u32> {
        self.read_u32(&self.agent_level_buf, "level")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("megaswarm_10000::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_10000::read_f32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("megaswarm_10000::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_10000::read_u32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
    pub fn num_mask_chunks(&self) -> u32 { self.num_mask_chunks }

    /// CPU-side sweep that resets HP=sentinel for any agent whose
    /// alive=0 but whose HP is NOT already at sentinel. Required to
    /// sidestep a deeper-layer gap: the compiler-emitted apply
    /// kernel uses non-atomic read-modify-write on agent_hp, so when
    /// many concurrent threads target the same agent the death
    /// branch's sentinel-HP write can be overwritten by an else-
    /// branch's stale-value write. The result is "dead but HP=10"
    /// agents that continue to win the next tick's argmax, freezing
    /// the simulation. Same as megaswarm_1000 — see that fixture's
    /// header for full discussion.
    ///
    /// Coalesces contiguous dead slots into a single write_buffer
    /// call to keep CPU→GPU bandwidth manageable at 10000 agents.
    pub fn sweep_dead_to_sentinel(&mut self) {
        let alive = self.read_alive();
        let hp = self.read_hp();
        // Collect (slot, sentinel) updates. For 10000 agents with
        // ~1 kill/tick under symmetric pile-on, this is at most
        // ~few hundred entries during the asymmetric tail.
        let sentinel: f32 = 1.0e9;
        let sentinel_bytes = bytemuck::bytes_of(&sentinel).to_vec();
        let mut wrote_any = false;
        let mut i = 0usize;
        while i < self.agent_count as usize {
            if alive[i] == 0 && hp[i] < 1.0e8 {
                // Find the run of contiguous dead-not-sentinel slots
                // starting at i, then write them in one go.
                let start = i;
                while i < self.agent_count as usize && alive[i] == 0 && hp[i] < 1.0e8 {
                    i += 1;
                }
                let run_len = i - start;
                let mut run_bytes: Vec<u8> = Vec::with_capacity(run_len * 4);
                for _ in 0..run_len {
                    run_bytes.extend_from_slice(&sentinel_bytes);
                }
                self.gpu.queue.write_buffer(
                    &self.agent_hp_buf,
                    (start as u64) * 4,
                    &run_bytes,
                );
                wrote_any = true;
            } else {
                i += 1;
            }
        }
        if wrote_any {
            // Force the writes to flush before the next step's
            // scoring kernel reads agent_hp.
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        }
    }
}

impl CompiledSim for Megaswarm10000State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_10000::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per tick: ~agent_count ActionSelected + ~agent_count Damaged
        // + up to ~agent_count Defeated ≈ 3 * agent_count. The shared
        // event ring is fixed at EVENT_RING_CAP_SLOTS=65536 across the
        // engine, so at agent_cap=10000 we cap clears at agent_count*4
        // = 40000 (well under 65536) instead of the ×8 headroom
        // megaswarm_1000 used. This is the upper bound the consumer
        // walks; per-handler tag filters reject non-matching slots so
        // overestimating is safe but undercounting is not — 4x is
        // already 33% headroom over the realistic ~3x.
        let max_slots_per_tick = (self.agent_count * 4).min(60_000);
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — chunked PerPair dispatch over the entire
        //     agent_cap² pair grid. Each chunk:
        //       a) write its cfg uniform with the current tick
        //       b) call the dispatch helper with the per-chunk pair
        //          count = MASK_CHUNK_WORKGROUPS * 64
        //     The fused kernel writes BOTH mask_0 (RedStrike) and
        //     mask_1 (BlueStrike) per pair.
        //
        // We update each chunk's cfg uniform's `tick` field via a
        // queue.write_buffer outside the encoder — they all flush
        // before the encoder's submit.
        //
        // SUBMIT-PER-CHUNK: at agent_cap≥9000 (26+ mask chunks)
        // wgpu reports "Encoder is invalid" if all chunks plus the
        // downstream scoring/chronicle/apply/fold passes are encoded
        // into a single command buffer. Empirically the single-encoder
        // budget on this driver tops out around ~22-25 dispatches +
        // copies. We split the mask round into N small command
        // buffers (one per chunk) so each finish() validates against
        // a reasonable pass count, then submit the post-mask pipeline
        // in a fresh encoder. The current encoder is finished/submitted
        // after the per-tick clears, then re-created for the chunked
        // dispatches loop, then again for the rest of the tick.
        self.gpu.queue.submit(Some(encoder.finish()));

        let total_pairs = (self.agent_count as u64) * (self.agent_count as u64);
        for chunk_idx in 0..self.num_mask_chunks {
            let pair_offset = chunk_idx * PAIRS_PER_CHUNK;
            // Recompute remaining pairs in this chunk so the final
            // chunk doesn't request more workgroups than needed.
            let remaining = (total_pairs - pair_offset as u64).min(PAIRS_PER_CHUNK as u64) as u32;
            let cfg = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeCfg {
                agent_cap: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad: pair_offset,
            };
            self.gpu.queue.write_buffer(
                &self.mask_chunk_cfgs[chunk_idx as usize],
                0,
                bytemuck::bytes_of(&cfg),
            );
            let bindings = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeBindings {
                agent_alive: &self.agent_alive_buf,
                agent_level: &self.agent_level_buf,
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                cfg: &self.mask_chunk_cfgs[chunk_idx as usize],
            };
            let mut chunk_encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("megaswarm_10000::mask_chunk"),
                },
            );
            // The dispatch helper does `(agent_cap + 63) / 64`
            // workgroups in X. Passing `remaining` (= up to
            // PAIRS_PER_CHUNK) yields up to MASK_CHUNK_WORKGROUPS
            // workgroups — under the 65535 hardware limit.
            dispatch::dispatch_fused_mask_verb_redstrike(
                &mut self.cache, &bindings, &self.gpu.device, &mut chunk_encoder,
                remaining,
            );
            self.gpu.queue.submit(Some(chunk_encoder.finish()));
        }

        // Re-open the encoder for the rest of the tick.
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_10000::step_tail") },
        );

        // (3) Scoring — argmax over both rows. Inner loop over
        // `cfg.agent_cap` candidates per pair-field row. Emits one
        // ActionSelected per gated agent.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_level: &self.agent_level_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) RedStrike chronicle — gates action_id==0u, emits Damaged.
        let red_strike_cfg = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_red_strike_cfg_buf, 0, bytemuck::bytes_of(&red_strike_cfg),
        );
        let red_strike_bindings = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_red_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_redstrike(
            &mut self.cache, &red_strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4b) BlueStrike chronicle — gates action_id==1u, emits Damaged.
        let blue_strike_cfg = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_blue_strike_cfg_buf, 0, bytemuck::bytes_of(&blue_strike_cfg),
        );
        let blue_strike_bindings = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_blue_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bluestrike(
            &mut self.cache, &blue_strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) Apply damage (PerEvent).
        let event_count_estimate = self.agent_count * 8;
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg),
        );
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) fold_damage_dealt — RMW per Damaged event.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache, &damage_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    // agent_count is fixed by the per-team layout (5000+5000=10000).
    Box::new(Megaswarm10000State::new(seed))
}
