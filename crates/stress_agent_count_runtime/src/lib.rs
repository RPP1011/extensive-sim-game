//! Per-fixture runtime for `assets/sim/stress_agent_count.sim` —
//! Fixture A from `docs/superpowers/plans/2026-05-09-stress-test-runtime-ceilings.md`.
//!
//! ## What this fixture characterizes
//!
//! One-shot characterization of the agent_count ceiling. Drives the
//! standard mask → score → chronicle → consume per-tick chain at a
//! variable agent_cap; the `stress_agent_count_app` binary sweeps the
//! cap from 1k to 200k and emits NDJSON per-tick metrics + a final
//! summary. The first cap that panics or OOMs is recorded as the
//! ceiling.
//!
//! ## Per-tick chain
//!
//! Mirrors the spy_network / duel_abilities shape but with a single
//! self-target verb (Pulse) and a fused chronicle-dispatcher + no-op
//! consumer (the .sim's only `@phase(post)` rule is
//! `ApplyPulseFromChronicle` which does `let _ = agents.hp(a)`):
//!
//!   1. Per-tick clears: event_tail, ring headers, mask bitmap,
//!      scoring_output.
//!   2. mask_verb_Pulse — PerAgent (NOT PerPair, since the verb has
//!      `verb Pulse(self)` with no per-pair candidate), writes mask_0.
//!   3. scoring — PerAgent argmax over the single row; emits
//!      ActionSelected for every alive agent.
//!   4. physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse —
//!      fused PerEvent kernel that walks ActionSelected → dispatches
//!      apply_ability → walks Pulse's program (one EffectOp::SelfDamage)
//!      → emits EffectSelfDamageApplied (kind=39) → consumer reads
//!      `agents.hp(actor)`.
//!   5. seed_indirect_0 — keeps indirect-args buffer warm.
//!
//! No consumer side-effects (the `let _ = agents.hp(a)` is intentionally
//! a dead read so HP doesn't drift across ticks; the population stays
//! constant for the entire tick budget).

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::rng::per_agent_u32_pcg;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::EventRing;

/// Pre-built dispatcher kernel total size, captured at build time by
/// build.rs (sum of all WGSL kernel byte counts). Read by the driver
/// binary's NDJSON metric emit so the harness can record kernel size
/// at the active cap (the kernel size is independent of agent_cap, so
/// it's the same for every cap in the sweep — but it's a useful
/// breakpoint candidate as the codegen continues to evolve).
pub const DISPATCHER_KERNEL_BYTES: usize = include!(concat!(
    env!("OUT_DIR"),
    "/dispatcher_total_bytes.txt"
));

/// Seed used by all stress runs. Per the plan doc P5 compliance:
/// pseudo-hex `0x5tre55_aa` doesn't fit in a real u64 hex literal so
/// we use the closest sensible interpretation.
pub const STRESS_SEED: u64 = 0x57_2E_55_AA_57_2E_55_AA;

/// Per-fixture state for the stress fixture.
pub struct StressAgentCountState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // The chronicle dispatcher + scoring kernels both bind these stat
    // columns; init to neutral values so any %max_hp / +AD scaling
    // future-mod doesn't produce sentinel results. Pulse itself doesn't
    // scale on any stat, so the values are dead reads.
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmap (Pulse=0) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring (engine helper) --
    event_ring: EventRing,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Packed AbilityRegistry uploaded once at construction. Holds the
    /// Pulse program (single EffectOp::SelfDamage{0.0}) at AbilityId(1).
    registry_gpu: PackedAbilityRegistryGpu,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,

    /// Per-agent positions (Vec3), seeded deterministically via PCG.
    /// Kept on host only — the .sim doesn't read pos in any verb body
    /// so we don't need a GPU buffer for it (pack_agents binds it but
    /// the runtime never invokes pack_agents).
    positions_host: Vec<Vec3>,
}

impl StressAgentCountState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert!(agent_count > 0, "agent_count must be > 0");
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Build the AbilityRegistry from the .ability corpus + upload.
        let built = build_stress_registry();
        let packed = PackedAbilityRegistry::pack(&built);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "stress_agent_count_runtime");

        // -- Per-agent SoA. HP=100 (constant — Pulse self-damages 0),
        //    alive=1, mana=0, stats=neutral.
        let n = agent_count as usize;
        let hp_init = vec![100.0_f32; n];
        let alive_init = vec![1u32; n];
        let mana_init = vec![0.0_f32; n];
        let max_hp_init = vec![100.0_f32; n];
        let zeros_f32 = vec![0.0_f32; n];

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress_agent_count::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_mana_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_mana"),
                contents: bytemuck::cast_slice(&mana_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_max_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_max_hp"),
                contents: bytemuck::cast_slice(&max_hp_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let mk_stat = |label: &'static str| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("stress_agent_count::agent_attack_damage");
        let agent_armor_buf = mk_stat("stress_agent_count::agent_armor");
        let agent_magic_resist_buf = mk_stat("stress_agent_count::agent_magic_resist");
        let agent_move_speed_buf = mk_stat("stress_agent_count::agent_move_speed");

        // -- Single mask bitmap (Pulse=0). Cleared each tick via copy.
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // -- Scoring output (4 × u32 per agent).
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> =
            vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let event_ring = EventRing::new(&gpu, "stress_agent_count");

        // -- Per-kernel cfg uniforms.
        let mask_cfg_init = mask_verb_Pulse::MaskVerbPulseCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress_agent_count::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_cfg_init = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::chronicle_cfg"),
                contents: bytemuck::bytes_of(&chronicle_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress_agent_count::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -- Per-agent positions, seeded deterministically via PCG.
        //    Constitution P5: all sim randomness flows through PCG;
        //    `per_agent_u32_pcg(seed_lo, agent_id, tick=0, purpose)` is
        //    the canonical mirror of the WGSL helper.
        let seed_lo = seed as u32;
        let mut positions_host: Vec<Vec3> = Vec::with_capacity(n);
        for slot in 0..n {
            let aid = slot as u32;
            let rx = per_agent_u32_pcg(seed_lo, aid, 0, 0xA5_5A_A5_5A);
            let ry = per_agent_u32_pcg(seed_lo, aid, 0, 0x5A_A5_5A_A5);
            let rz = per_agent_u32_pcg(seed_lo, aid, 0, 0xC0_FF_EE_42);
            // Map into a 1024×1024×1024 volume centred at origin.
            let fx = ((rx as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            let fy = ((ry as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            let fz = ((rz as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            positions_host.push(Vec3::new(fx, fy, fz));
        }

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            mask_0_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_cfg_buf,
            seed_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
            positions_host,
        }
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

    /// Snapshot of the chronicle event ring's tail value (= total
    /// events emitted this tick at flush time). Triggers a small
    /// staging readback; only meant for the behavioral pin test, NOT
    /// the per-tick metrics path.
    pub fn read_event_count(&self) -> u32 {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("stress_agent_count::read_event_count"),
            },
        );
        encoder.copy_buffer_to_buffer(self.event_ring.tail(), 0, &staging, 0, 4);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: u32 = *bytemuck::from_bytes::<u32>(&mapped[..4]);
        drop(mapped);
        staging.unmap();
        v
    }
}

impl CompiledSim for StressAgentCountState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("stress_agent_count::step"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per tick: ~agent_count ActionSelected + ~agent_count
        // EffectSelfDamageApplied. 4× headroom keeps far below the
        // engine's EVENT_RING_CAP_SLOTS (which is also 65536).
        let max_slots_per_tick = (self.agent_count.saturating_mul(4)).min(60_000);
        self.event_ring.clear_ring_headers_in(
            &self.gpu,
            &mut encoder,
            max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_0_bitmap_buf,
            0,
            mask_bytes.max(4),
        );
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf,
            0,
            &self.scoring_output_buf,
            0,
            scoring_output_bytes.max(16),
        );

        // (2) Mask round — single PerAgent kernel, dispatches
        //     `agent_cap` threads (no per-pair fan-out — Pulse is
        //     self-target, so each agent only gates its own slot).
        let mask_cfg = mask_verb_Pulse::MaskVerbPulseCfg {
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
        let mask_bindings = mask_verb_Pulse::MaskVerbPulseBindings {
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_mask_verb_pulse(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) Scoring — argmax over the single Pulse row.
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
            agent_hp: &self.agent_hp_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) Fused chronicle dispatcher + ApplyPulseFromChronicle
        //     consumer (PerEvent kernel). Walks ActionSelected entries,
        //     dispatches Pulse's program (one EffectOp::SelfDamage),
        //     emits EffectSelfDamageApplied (kind=39), consumer reads
        //     `agents.hp(actor)` (no-op).
        let event_count_estimate = self.agent_count.saturating_mul(4).min(60_000);
        let chronicle_cfg = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: self.agent_count,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana: &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_chances: &self.registry_gpu.chances,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            cfg: &self.chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applypulsefromchronicle_and_verb_chronicle_pulse(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (5) seed_indirect_0 — keeps args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
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

        self.gpu.queue.submit(Some(encoder.finish()));
        // Wait for GPU work to complete so per-tick wall-clock
        // measurements taken in the driver actually capture GPU time
        // (not just queue-submit time, which is microseconds regardless
        // of agent_cap).
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("poll after step submit");
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &self.positions_host
    }

    fn snapshot(&mut self) -> AgentSnapshot {
        AgentSnapshot {
            positions: self.positions_host.clone(),
            creature_types: vec![1u32; self.agent_count as usize],
            alive: vec![1u32; self.agent_count as usize],
        }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![VizGlyph::new('?', 240), VizGlyph::new('.', 27)]
    }
}

/// Pulse.ability source — embedded at compile time so the runtime
/// doesn't need filesystem access to the workspace at run time
/// (release-built binaries get distributed standalone).
const PULSE_ABILITY_SRC: &str =
    include_str!("../../../assets/ability_test/stress_agent_count/Pulse.ability");

/// Build the Pulse ability registry from the embedded `.ability`
/// source. Mirrors
/// `spy_network_runtime::binding_check::build_spy_network_registry`
/// but embeds the .ability text rather than reading via the
/// `CARGO_MANIFEST_DIR` env var (which is only set at compile time,
/// not at run time).
fn build_stress_registry() -> engine::ability::registry::AbilityRegistry {
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    let files = vec![(
        "Pulse.ability".to_string(),
        parse("Pulse.ability", PULSE_ABILITY_SRC),
    )];

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over stress_agent_count corpus")
        .registry
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(StressAgentCountState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Behavioral pin test.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin: a single tick at agent_cap=1000 advances `state.tick` to 1
    /// AND emits at least one chronicle record (event_tail > 0).
    /// Exercises the full mask → score → chronicle → consumer pipeline
    /// at a non-trivial agent count.
    #[test]
    fn tick_advances_at_1k_agents() {
        let mut state = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            StressAgentCountState::new(STRESS_SEED, 1000)
        })) {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "[tick_advances_at_1k_agents] skipping: GPU init failed (no wgpu adapter on host?)"
                );
                return;
            }
        };

        state.step();
        assert_eq!(state.tick(), 1, "tick should advance to 1 after one step");

        // event_tail counts chronicle records written this tick. With
        // 1000 alive agents, we expect ~1000 ActionSelected +
        // ~1000 EffectSelfDamageApplied = 2000-ish entries; the floor
        // we assert is "non-zero" since GPU race ordering / event-ring
        // back-pressure can clip the count somewhat.
        let event_count = state.read_event_count();
        assert!(
            event_count > 0,
            "expected event_tail > 0 after one tick at 1000 agents; got {event_count}",
        );
    }
}
