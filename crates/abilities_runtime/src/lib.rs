//! Per-fixture runtime for `assets/sim/abilities_probe.sim` — the
//! smallest end-to-end probe of the ABILITY SYSTEM.
//!
//! Mirrors `verb_probe_runtime`'s shape verbatim: compiler-emitted
//! kernels live in `OUT_DIR/generated.rs`, the hand-written
//! `AbilitiesProbeState` owns the GPU context, per-buffer storage,
//! kernel cache, event ring, and per-tick dispatch chain.
//!
//! ## Per-tick chain (orchestrated)
//!
//! Six rounds inside one tick:
//!
//! 1. **Clears** — `event_tail = 0`, both mask bitmaps zeroed.
//! 2. **Mask round** — `fused_mask_verb_Strike` writes BOTH the
//!    Strike mask (mask_0) AND the Heal mask (mask_1) bitmaps in
//!    one PerAgent pass. `self.alive` is always true so every alive
//!    agent's bit is set in both bitmaps.
//! 3. **Scoring** — argmax over the two competing rows. Strike's
//!    score = `config.combat.strike_damage` = 10.0; Heal's score =
//!    `config.combat.heal_amount` = 5.0. Strike always wins, so
//!    every agent emits `ActionSelected{ actor=agent, action_id=0,
//!    target=NO_TARGET }` (kind tag = 3u). The losing-row mask
//!    (mask_1) is loaded but Heal's utility never beats Strike's.
//! 4. **Strike chronicle** — *blocked by Gap #1.* The Strike
//!    chronicle physics rule got fused into
//!    `fused_fold_healing_done_healed`, whose WGSL body references
//!    an undeclared `view_storage_primary` (only `view_1_primary`
//!    is bound). The shader fails naga validation, so this runtime
//!    SKIPS that fused-kernel dispatch entirely. Consequence: no
//!    `DamageDealt` events ever land in the ring, so
//!    `damage_total` stays at its `initial: 0.0`.
//! 5. **Heal chronicle** — `physics_verb_chronicle_Heal` reads the
//!    ActionSelected slots emitted in round 3, gates on
//!    `action_id == 1u` (Heal's id). Heal NEVER wins argmax — every
//!    slot has `action_id = 0u` — so the gate is always false and
//!    no `Healed` events are emitted. Dispatched anyway to exercise
//!    the kernel's binding shape.
//! 6. **seed_indirect_0** — keeps the indirect-args buffer warm.
//! 7. **fold_damage_total** — dispatched to exercise the per-handler
//!    tag-filter codepath. The body gates on `event_ring[..tag..]
//!    == 1u` (DamageDealt tag); since no DamageDealt events landed,
//!    the fold accumulates zero into every slot.
//!
//! ## Observable
//!
//! - `damage_total[i]` = 0.0 for every slot — Gap #1 blocks the
//!   Strike chronicle.
//! - `healing_done` is NOT directly readable today; the fold kernel
//!   fails to load. The runtime exposes only `damage_total` for the
//!   sim_app to read back.
//!
//! See `docs/superpowers/notes/2026-05-04-abilities_probe.md` for the
//! discovery doc + gap punch list.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            _pad: 0.0,
        }
    }
}

/// Per-fixture state for the abilities probe. Carries one Combatant
/// SoA (alive only — pos/vel are declaration-only in the .sim and
/// no kernel reads them), the event ring, the per-view storage, the
/// scoring_output buffer, two mask bitmaps (one per verb), and the
/// per-kernel cfg uniforms.
pub struct AbilitiesProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. Initialised all-1 so every slot's mask
    /// predicate (`self.alive`) evaluates true.
    agent_alive_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: Strike=0, Heal=1). --
    /// `mask_0_bitmap` (Strike) — atomicOr-set when `self.alive`.
    mask_0_bitmap_buf: wgpu::Buffer,
    /// `mask_1_bitmap` (Heal) — same predicate, distinct bitmap.
    mask_1_bitmap_buf: wgpu::Buffer,
    /// Pre-zeroed source for per-tick clears via copy_buffer_to_buffer.
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-attacker damage_total view storage. Fed by `DamageDealt`
    /// events (kind tag = 1u). Today's runtime never sees a
    /// DamageDealt event because the Strike chronicle was fused into
    /// the broken `fused_fold_healing_done_healed` kernel (Gap #1)
    /// — so damage_total stays at 0.0.
    damage_total: ViewStorage,
    damage_total_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,
    #[allow(dead_code)]
    snapshot_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl AbilitiesProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — only `alive` is read by any compiled kernel
        // (fused_mask_verb_Strike). Sized to agent_count u32 slots.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Two mask bitmaps — one per verb. Cleared each tick before
        // the fused mask kernel runs so stale bits don't leak.
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
        let mask_0_bitmap_buf = mk_mask("abilities_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("abilities_runtime::mask_1_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent
        let scoring_output_bytes = (agent_count as u64) * 4 * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("abilities_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Event ring + damage_total view storage (per-agent f32, no
        // anchor, no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "abilities_runtime");
        let damage_total = ViewStorage::new(
            &gpu,
            "abilities_runtime::damage_total",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::mask_cfg"),
                contents: bytemuck::bytes_of(&mask_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0,
                tick: 0,
                _pad0: 0,
                _pad1: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::chronicle_heal_cfg"),
                contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let snapshot_cfg_init = kick_snapshot::KickSnapshotCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let snapshot_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::snapshot_cfg"),
                contents: bytemuck::bytes_of(&snapshot_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let damage_cfg_init = fold_damage_total::FoldDamageTotalCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let damage_total_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abilities_runtime::damage_total_cfg"),
                contents: bytemuck::bytes_of(&damage_cfg_init),
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
            damage_total,
            damage_total_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_heal_cfg_buf,
            seed_cfg_buf,
            snapshot_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-attacker damage_total accumulator (one f32 per slot).
    /// Today (Gap #1 active): zero for every slot — the Strike
    /// chronicle was fused into the broken
    /// `fused_fold_healing_done_healed` kernel and is never
    /// dispatched, so no `DamageDealt` events land in the ring.
    pub fn damage_total(&mut self) -> &[f32] {
        self.damage_total.readback(&self.gpu)
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

impl CompiledSim for AbilitiesProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("abilities_runtime::step"),
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

        // (2) Mask round — `fused_mask_verb_Strike` runs both
        // mask predicates (mask_0 + mask_1) in one PerAgent pass.
        // Both predicates are `self.alive` so every alive slot's
        // bit is set in both bitmaps.
        let mask_cfg = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu
            .queue
            .write_buffer(&self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg));
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) Scoring — argmax over the 2 competing rows. Strike's
        // utility = 10.0 wins over Heal's 5.0 every tick, so every
        // agent atomically emits one `ActionSelected{ actor=agent,
        // action_id=0, target=NO_TARGET }` (kind tag = 3u).
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
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

        // (4) Strike chronicle — *blocked by Gap #1.* The Strike
        // chronicle was fused into `fused_fold_healing_done_healed`,
        // whose WGSL body references an undeclared
        // `view_storage_primary` (only `view_1_primary` is in the
        // bindings). Naga rejects the shader at module-creation
        // time, so we DO NOT instantiate that kernel. Consequence:
        // no DamageDealt events ever reach the ring; damage_total
        // stays at 0.0 for every slot. See discovery doc Gap #1.

        // (5) Heal chronicle — exercised even though it never fires.
        // Reads ActionSelected slots emitted in round 3, gates on
        // `action_id == 1u` (Heal's id). Heal NEVER wins argmax, so
        // every slot's action_id is 0u and the gate is always false.
        // Zero `Healed` events emitted; the kernel still binds and
        // dispatches cleanly.
        let chronicle_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_heal_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_heal_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_heal(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) seed_indirect_0 — reads tail count, populates
        // indirect_args_0 with workgroup count. Kept for parity.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
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

        // (7) fold_damage_total — RMW per `DamageDealt` (kind=1u).
        // event_count sized generously to cover both event kinds in
        // the ring (ActionSelected from scoring + any DamageDealt
        // from the Strike chronicle, which today is zero — Gap #1).
        // The per-handler tag check inside the fold body guards
        // against ActionSelected slots, so the loop accumulates
        // zero into every slot.
        let event_count_estimate = self.agent_count * 2;
        let damage_cfg = fold_damage_total::FoldDamageTotalCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_total_cfg_buf,
            0,
            bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_total::FoldDamageTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_total.primary(),
            view_storage_anchor: self.damage_total.anchor(),
            view_storage_ids: self.damage_total.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_total_cfg_buf,
        };
        dispatch::dispatch_fold_damage_total(
            &mut self.cache,
            &damage_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (8) `fused_fold_healing_done_healed` — *NOT DISPATCHED.*
        // Naga rejects the shader (Gap #1: undeclared
        // `view_storage_primary`). Even if dispatched and the shader
        // somehow compiled, the kernel fuses the Strike chronicle
        // (which we DO want) with the healing_done fold (which is
        // empty because Heal never wins). The right fix is at the
        // emit/fusion layer; runtime-side workarounds would mask
        // the real defect.

        // (9) kick_snapshot — host-side artefact; skipped, same
        // pattern as verb_probe_runtime.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_total.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. Same skip as
        // verb_probe_runtime.
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(AbilitiesProbeState::new(seed, agent_count))
}
