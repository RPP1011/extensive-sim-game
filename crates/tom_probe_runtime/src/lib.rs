//! Per-fixture runtime for `assets/sim/tom_probe.sim` — Theory-of-Mind
//! end-to-end probe (post-fix shape).
//!
//! ## What lights up
//!
//! - The Knower SoA (`agent_alive`).
//! - The `BeliefAcquired` event ring + per-tick tail clear.
//! - The `physics_WhatIBelieve` kernel (per-Knower; emits one
//!   `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }`
//!   per tick into the ring).
//! - The `fold_beliefs_flags` kernel (per-event; OR's `fact_bit` into
//!   `view_storage_primary[observer * agent_cap + subject]` via
//!   WGSL native `atomicOr` — no CAS retry, P11 trivial). Renamed
//!   from `fold_beliefs` in Wave 3 ToM Phase 2 (see below).
//! - Per-(observer, subject) belief storage: see "BeliefState SoA"
//!   below for the multi-field shape introduced by Wave 3 ToM
//!   Phase 2. The `flags` column is the same `array<atomic<u32>>`
//!   BGL shape the fold kernel expects, so binding parity holds.
//!
//! ## BeliefState SoA (Wave 3 ToM Phase 2)
//!
//! Each (observer, target) pair is described by 6 parallel SoA
//! columns mirroring the spec's BeliefState block
//! (`docs/spec/ability_dsl_test_sims.md` spy_network):
//!
//! | Field        | Type       | Buffer                          | Notes                                  |
//! |--------------|------------|---------------------------------|----------------------------------------|
//! | flags        | u32        | `beliefs_flags_primary`         | Wave 3 Phase 1 bit-OR slot (kept)      |
//! | last_known_pos        | vec3 (4×f32 padded) | `beliefs_pos_primary`           | observer's last sighting               |
//! | last_known_creature_type | u8 → u32     | `beliefs_type_primary`          | observer's last sighted classification |
//! | last_seen_tick        | u32       | `beliefs_tick_primary`          | tick of last update; powers decay      |
//! | confidence            | u8 (q8)   | `beliefs_confidence_primary`    | 0..255 (≈ 0.0..1.0); how sure observer |
//! | suspicion             | u8 (q8)   | `beliefs_suspicion_primary`     | 0..255; how much observer suspects     |
//!
//! Phase 2 ships the storage shape + readback accessors only — the
//! writer verbs (`scry` / `disguise` / `observe` / etc.) are
//! Phase 3, and the spec's `agents.beliefs(o, s).<field>`
//! struct-then-field DSL access syntax is also deferred there. The
//! 5 non-`flags` columns are NOT folded over any event today; they
//! exist as runtime-allocated GPU storage so test fixtures can
//! pre-seed them with `seed_beliefs_<field>()` and assert
//! per-(observer, target) cell isolation via the readback accessors.
//!
//! Cell indexing across all 6 columns is `observer * agent_count +
//! target` — identical to `beliefs_flags`, so a fold kernel that
//! eventually folds into one of these columns can reuse the existing
//! `pair_map` index expression unchanged.
//!
//! ## Expected outcome (FULL FIRE)
//!
//! After N ticks at agent_count = N: `beliefs_flags(i, i) = 1u` for
//! every alive Knower; every off-diagonal `beliefs_flags(i, j != i)
//! = 0u`. The tom_probe_app driver asserts both halves and reports
//! OUTCOME (a) FULL FIRE on success.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{bgl_storage, bgl_uniform, EventRing};

// =====================================================================
// Wave 3 ToM Phase 3.6 — GPU consumer kernels for observe/scry/reveal +
// per-tick decay (close the Phase 3 deferred items).
//
// The Phase 3/3.5 consumer methods (`observe` / `scry` / `reveal`) and
// the per-tick `decay_step` shipped as runtime-side CPU sweeps that
// patched the BeliefState SoA columns via `queue.write_buffer`. The
// "column-rewrite trick" (rewrite the whole u8 column to dodge wgpu's
// 4-byte-multiple write constraint) was O(N²) per call.
//
// Phase 3.6 replaces those CPU sweeps with hand-written WGSL compute
// kernels invoked from the runtime each call (observe/scry/reveal) or
// each tick (decay). The DSL `@decay(beliefs)` annotation + chronicle-
// stream PerEvent emission for the consumers stay deferred (they would
// require new compiler IR variants and a schema-hash bump). Hand-
// written WGSL gives the same CPU→GPU move with zero impact on the
// schema hash and zero new DSL surface — see the `# Alternative` note
// in the Phase 3.6 brief.
//
// **Storage convention.** The 6 BeliefState columns live in
// `(primary, staging)` buffer pairs (see `alloc_belief_column_pair`).
// Five columns map cleanly onto WGSL types:
//
//   * `flags` / `tick`             — `array<u32>` (4 B per cell)
//   * `pos`                        — `array<vec4<f32>>` (16 B per cell)
//
// The three u8 columns (`type`, `confidence`, `suspicion`) are
// allocated as byte buffers rounded up to a 16-byte multiple. WGSL has
// no `u8`, so kernels treat them as `array<atomic<u32>>` (4 packed
// little-endian bytes per word) and read/write a single byte via the
// `read_packed_byte` / `write_packed_byte` helpers below. The two-step
// `atomicAnd` + `atomicOr` write IS NOT atomic across the pair, which
// is fine here because:
//
//   - observe / scry: each dispatch runs ONE thread that touches at
//     most one byte per word.
//   - reveal: N threads, each writing cell `[obs * N + subject]`.
//     Different observers' bytes land in different words for any
//     `N >= 4` (the only fixture size used today is N = 4).
//   - decay: kernel dispatches `(cell_count + 3) / 4` threads, each
//     thread owns one full word and processes its 4 cells in registers
//     before writing the word back — no inter-thread word contention.
//
// **Cell indexing.** All columns use `observer * agent_count + subject`
// row-major flat indexing — same as Phase 1 / Phase 2 / 3 / 3.5.
// =====================================================================

/// Uniform payload for the observe / scry / reveal kernels. The same
/// 16-byte struct backs all three; unused fields are set to 0 by
/// `dispatch_belief_consumer`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BeliefConsumerCfg {
    /// `observe.observer` / `scry.observer` / `reveal.caster`.
    observer: u32,
    /// `observe.target` / `scry.target_observer` / `reveal.subject`.
    target_or_subject: u32,
    /// `scry.subject` (only used by scry; 0 elsewhere).
    subject: u32,
    /// Width of the 2-D pair grid = agent_count. Drives the
    /// `observer * agent_count + subject` cell-index expression and
    /// (for reveal) the per-thread observer index.
    agent_count: u32,
    /// Current sim tick. observe stamps this into `last_seen_tick`.
    /// decay reads it to compute `last == current_tick` (skip same-tick
    /// cells so the observe→decay pipeline doesn't immediately undo
    /// observe's confidence=255 writeback).
    tick: u32,
    /// Total cell count = `agent_count * agent_count`. Decay's only
    /// upper-bound check (`cell_idx >= cell_count` early-return) reads
    /// this to filter the round-up tail of the dispatch.
    cell_count: u32,
    /// Pad to 32 B so every `BeliefConsumerCfg` upload is a clean
    /// 16-byte-multiple write (wgpu min-binding-size + std140 friendly).
    _pad0: u32,
    _pad1: u32,
}

/// Pipelines + bind-group layouts for the four hand-written GPU
/// consumer kernels. Built once at `TomProbeState::new` time and
/// reused per call. The cfg uniform buffer is shared across kernels —
/// each dispatch overwrites it via `write_buffer` before binding.
struct BeliefConsumerGpu {
    /// Single 32-byte uniform buffer reused across all 4 kernels.
    cfg_buf: wgpu::Buffer,
    /// observe kernel — 1 workgroup × 1 thread; binds all 6 BeliefState
    /// columns + agent_pos / agent_creature_type.
    observe_pipeline: wgpu::ComputePipeline,
    observe_bgl: wgpu::BindGroupLayout,
    /// scry kernel — 1 workgroup × 1 thread; binds all 6 columns
    /// (read-modify-write since src and dst are both in the same
    /// buffers) + cfg.
    scry_pipeline: wgpu::ComputePipeline,
    scry_bgl: wgpu::BindGroupLayout,
    /// reveal kernel — N workgroups × 1 thread (one thread per
    /// observer slot); binds all 6 columns + cfg.
    reveal_pipeline: wgpu::ComputePipeline,
    reveal_bgl: wgpu::BindGroupLayout,
    /// decay kernel — `(cell_count + 3) / 4 / 64` workgroups × 64
    /// threads (one thread per packed-byte word in the confidence
    /// column); binds confidence + tick columns + cfg.
    decay_pipeline: wgpu::ComputePipeline,
    decay_bgl: wgpu::BindGroupLayout,
}

/// Shared WGSL helpers used by every belief-consumer kernel. Treats a
/// `array<atomic<u32>>` u8-column buffer as 4 little-endian bytes per
/// word and exposes byte-granularity read/write. The two-step write
/// (`atomicAnd` mask-clear → `atomicOr` value-set) is safe under the
/// per-kernel concurrency rules documented at the module header.
const BELIEF_HELPERS_WGSL: &str = r#"
fn read_packed_byte(buf_idx: u32, byte_in_word: u32, word: u32) -> u32 {
    let shift = byte_in_word * 8u;
    return (word >> shift) & 0xFFu;
}

fn write_packed_byte_atomic(buf: ptr<storage, array<atomic<u32>>, read_write>, cell_idx: u32, value: u32) {
    let word_idx = cell_idx / 4u;
    let byte_in_word = cell_idx % 4u;
    let shift = byte_in_word * 8u;
    let mask = 0xFFu << shift;
    let payload = (value & 0xFFu) << shift;
    // Two-step write: clear the byte slot, then set the new bits. Safe
    // under the per-kernel concurrency rules (see module-header notes).
    atomicAnd(&((*buf)[word_idx]), ~mask);
    atomicOr(&((*buf)[word_idx]), payload);
}
"#;

/// observe: copy target's CURRENT pos / creature_type from the agent
/// SoA + stamp `last_seen_tick = cfg.tick` and `confidence = 255` into
/// the (observer, target) cell of the 6 BeliefState columns.
///
///   beliefs_pos[obs * N + tgt]        = agent_pos[tgt]
///   beliefs_type[obs * N + tgt]       = agent_creature_type[tgt]
///   beliefs_tick[obs * N + tgt]       = cfg.tick
///   beliefs_confidence[obs * N + tgt] = 255u
///   beliefs_flags[obs * N + tgt]      = unchanged (Phase 1 bit-OR slot)
///   beliefs_suspicion[obs * N + tgt]  = unchanged (Phase 4 deception)
///
/// Single-thread dispatch — the work is one cell per call.
const OBSERVE_WGSL: &str = r#"
struct Cfg {
    observer: u32,
    target_or_subject: u32,
    subject: u32,
    agent_count: u32,
    tick: u32,
    cell_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read>            agent_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>            agent_creature_type: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>      beliefs_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write>      beliefs_type: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write>      beliefs_tick: array<u32>;
@group(0) @binding(6) var<storage, read_write>      beliefs_confidence: array<atomic<u32>>;

%HELPERS%

@compute @workgroup_size(1)
fn cs_observe(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let observer = cfg.observer;
    let target = cfg.target_or_subject;
    let cell = observer * cfg.agent_count + target;

    // pos: direct vec4<f32> copy.
    beliefs_pos[cell] = agent_pos[target];

    // creature_type: read packed byte at `target` from agent SoA, write
    // packed byte at `cell` into beliefs SoA.
    let agent_type_word_idx = target / 4u;
    let agent_type_word = atomicLoad(&agent_creature_type[agent_type_word_idx]);
    let target_type = read_packed_byte(agent_type_word_idx, target % 4u, agent_type_word);
    write_packed_byte_atomic(&beliefs_type, cell, target_type);

    // tick: direct u32 store.
    beliefs_tick[cell] = cfg.tick;

    // confidence: peg to 255 (q8 max — fresh observation).
    write_packed_byte_atomic(&beliefs_confidence, cell, 255u);
}
"#;

/// scry: cross-observer 6-field copy. caster (`cfg.observer`) reads
/// target_observer (`cfg.target_or_subject`)'s belief row about
/// `cfg.subject` and writes the same tuple into caster's belief row
/// about subject.
///
///   For each field f in {pos, type, tick, confidence, suspicion, flags}:
///     beliefs_f[observer * N + subject] = beliefs_f[target_obs * N + subject]
///
/// Single-thread dispatch — same shape as observe.
const SCRY_WGSL: &str = r#"
struct Cfg {
    observer: u32,
    target_or_subject: u32,
    subject: u32,
    agent_count: u32,
    tick: u32,
    cell_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> beliefs_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> beliefs_type: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> beliefs_tick: array<u32>;
@group(0) @binding(4) var<storage, read_write> beliefs_confidence: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> beliefs_suspicion: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> beliefs_flags: array<u32>;

%HELPERS%

@compute @workgroup_size(1)
fn cs_scry(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let observer = cfg.observer;
    let target_obs = cfg.target_or_subject;
    let subject = cfg.subject;
    let n = cfg.agent_count;
    let src_cell = target_obs * n + subject;
    let dst_cell = observer  * n + subject;

    // Read all 6 source-cell values BEFORE writing any destination —
    // when src == dst (self-scry) this prevents the per-byte writes
    // from clobbering the source mid-copy.
    let src_pos        = beliefs_pos[src_cell];
    let src_type_word  = atomicLoad(&beliefs_type[src_cell / 4u]);
    let src_type       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_type_word);
    let src_tick       = beliefs_tick[src_cell];
    let src_conf_word  = atomicLoad(&beliefs_confidence[src_cell / 4u]);
    let src_conf       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_conf_word);
    let src_susp_word  = atomicLoad(&beliefs_suspicion[src_cell / 4u]);
    let src_susp       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_susp_word);
    let src_flags      = beliefs_flags[src_cell];

    beliefs_pos[dst_cell]   = src_pos;
    beliefs_tick[dst_cell]  = src_tick;
    beliefs_flags[dst_cell] = src_flags;
    write_packed_byte_atomic(&beliefs_type,       dst_cell, src_type);
    write_packed_byte_atomic(&beliefs_confidence, dst_cell, src_conf);
    write_packed_byte_atomic(&beliefs_suspicion,  dst_cell, src_susp);
}
"#;

/// reveal: one-to-many fan-out. caster broadcasts its beliefs about
/// subject to every observer slot. Each thread owns one observer slot
/// and writes the (observer, subject) cell.
///
///   For obs in 0..N: beliefs_f[obs * N + subject] = beliefs_f[caster * N + subject]
///
/// Phase 3.5 ships the "all observers" fan-out shape; range-gated
/// reveal (`reveal subject in <volume>`) defers until spatial-query
/// infrastructure for "observers within radius of caster" matures.
const REVEAL_WGSL: &str = r#"
struct Cfg {
    observer: u32,
    target_or_subject: u32,
    subject: u32,
    agent_count: u32,
    tick: u32,
    cell_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> beliefs_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> beliefs_type: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> beliefs_tick: array<u32>;
@group(0) @binding(4) var<storage, read_write> beliefs_confidence: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> beliefs_suspicion: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> beliefs_flags: array<u32>;

%HELPERS%

@compute @workgroup_size(64)
fn cs_reveal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let observer = gid.x;
    let n = cfg.agent_count;
    if (observer >= n) { return; }
    let caster  = cfg.observer;
    let subject = cfg.target_or_subject;
    let src_cell = caster   * n + subject;
    let dst_cell = observer * n + subject;

    // Every thread reads the same src_cell — the GPU's L1/SMEM coalesces
    // the broadcast. No write-after-write hazard since we never write
    // src_cell from observers != caster.
    let src_pos        = beliefs_pos[src_cell];
    let src_type_word  = atomicLoad(&beliefs_type[src_cell / 4u]);
    let src_type       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_type_word);
    let src_tick       = beliefs_tick[src_cell];
    let src_conf_word  = atomicLoad(&beliefs_confidence[src_cell / 4u]);
    let src_conf       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_conf_word);
    let src_susp_word  = atomicLoad(&beliefs_suspicion[src_cell / 4u]);
    let src_susp       = read_packed_byte(src_cell / 4u, src_cell % 4u, src_susp_word);
    let src_flags      = beliefs_flags[src_cell];

    beliefs_pos[dst_cell]   = src_pos;
    beliefs_tick[dst_cell]  = src_tick;
    beliefs_flags[dst_cell] = src_flags;
    write_packed_byte_atomic(&beliefs_type,       dst_cell, src_type);
    write_packed_byte_atomic(&beliefs_confidence, dst_cell, src_conf);
    write_packed_byte_atomic(&beliefs_suspicion,  dst_cell, src_susp);
}
"#;

/// decay: per-tick confidence decrement. Walks every (observer, target)
/// cell, decrements `confidence` by 1 saturating, with two skip rules:
///
///   - cells observed THIS tick (`last_seen_tick == cfg.tick`) skip
///     so observe→decay in the same tick doesn't immediately undo
///     observe's confidence=255 writeback.
///   - never-observed cells (`last == 0 && conf == 0`) skip — the
///     saturating arithmetic would no-op anyway; the guard is intent +
///     a tiny perf hint.
///
/// Each thread owns one packed-byte word (4 cells). It loads the word
/// once, processes 4 cells in registers, and writes the word back —
/// avoiding inter-thread word contention. No atomic needed: only one
/// thread ever owns a given word.
const DECAY_WGSL: &str = r#"
struct Cfg {
    observer: u32,
    target_or_subject: u32,
    subject: u32,
    agent_count: u32,
    tick: u32,
    cell_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> beliefs_confidence: array<u32>;
@group(0) @binding(2) var<storage, read>       beliefs_tick:       array<u32>;

@compute @workgroup_size(64)
fn cs_decay(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    // word_count = ceil(cell_count / 4). We dispatch with an integer-
    // divided ceiling so the trailing thread might own a partial word
    // (cell_count not a multiple of 4); skip cells beyond cell_count
    // inside the loop.
    let cell_count = cfg.cell_count;
    let word_count = (cell_count + 3u) / 4u;
    if (word_idx >= word_count) { return; }

    let word = beliefs_confidence[word_idx];
    var new_word: u32 = 0u;

    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let cell = word_idx * 4u + b;
        let shift = b * 8u;
        let conf = (word >> shift) & 0xFFu;
        var new_conf: u32 = conf;
        if (cell < cell_count) {
            let last = beliefs_tick[cell];
            // Skip never-observed cells: leaves new_conf == conf == 0.
            // Skip cells observed this tick: same-tick observe pegged
            // confidence at 255; don't drop it back to 254 immediately.
            let observed_this_tick = last == cfg.tick;
            let never_observed = last == 0u && conf == 0u;
            if (!observed_this_tick && !never_observed) {
                if (conf > 0u) {
                    new_conf = conf - 1u;
                } else {
                    new_conf = 0u;
                }
            }
        }
        new_word = new_word | ((new_conf & 0xFFu) << shift);
    }
    beliefs_confidence[word_idx] = new_word;
}
"#;

impl BeliefConsumerGpu {
    fn new(device: &wgpu::Device) -> Self {
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tom_probe_runtime::belief_consumer::cfg"),
            size: std::mem::size_of::<BeliefConsumerCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let observe_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tom_probe_runtime::observe::bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, true),  // agent_pos
                bgl_storage(2, false), // agent_creature_type (atomic)
                bgl_storage(3, false), // beliefs_pos
                bgl_storage(4, false), // beliefs_type (atomic)
                bgl_storage(5, false), // beliefs_tick
                bgl_storage(6, false), // beliefs_confidence (atomic)
            ],
        });
        let observe_pipeline =
            build_pipeline(device, &observe_bgl, "observe", OBSERVE_WGSL, "cs_observe");

        let scry_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tom_probe_runtime::scry::bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, false), // beliefs_pos
                bgl_storage(2, false), // beliefs_type (atomic)
                bgl_storage(3, false), // beliefs_tick
                bgl_storage(4, false), // beliefs_confidence (atomic)
                bgl_storage(5, false), // beliefs_suspicion (atomic)
                bgl_storage(6, false), // beliefs_flags
            ],
        });
        let scry_pipeline =
            build_pipeline(device, &scry_bgl, "scry", SCRY_WGSL, "cs_scry");

        // reveal binds the same 6 columns as scry — same BGL layout but
        // separate object so wgpu's pipeline-cache lookup keys cleanly.
        let reveal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tom_probe_runtime::reveal::bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, false),
                bgl_storage(2, false),
                bgl_storage(3, false),
                bgl_storage(4, false),
                bgl_storage(5, false),
                bgl_storage(6, false),
            ],
        });
        let reveal_pipeline =
            build_pipeline(device, &reveal_bgl, "reveal", REVEAL_WGSL, "cs_reveal");

        let decay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tom_probe_runtime::decay::bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage(1, false), // beliefs_confidence
                bgl_storage(2, true),  // beliefs_tick (read-only)
            ],
        });
        let decay_pipeline =
            build_pipeline(device, &decay_bgl, "decay", DECAY_WGSL, "cs_decay");

        Self {
            cfg_buf,
            observe_pipeline,
            observe_bgl,
            scry_pipeline,
            scry_bgl,
            reveal_pipeline,
            reveal_bgl,
            decay_pipeline,
            decay_bgl,
        }
    }
}

/// Helper: substitute `%HELPERS%` for `BELIEF_HELPERS_WGSL`, build the
/// shader module + pipeline. Centralised so each kernel definition stays
/// a single `const &str` literal.
fn build_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
    wgsl: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    let resolved = wgsl.replace("%HELPERS%", BELIEF_HELPERS_WGSL);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("tom_probe_runtime::{label}::wgsl")),
        source: wgpu::ShaderSource::Wgsl(resolved.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("tom_probe_runtime::{label}::pl")),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("tom_probe_runtime::{label}::pipeline")),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Per-fixture state for the ToM probe. Carries the Knower SoA
/// (`agent_alive`), the `BeliefAcquired` event ring, the 6-column
/// per-(observer, subject) BeliefState SoA storage (Wave 3 ToM
/// Phase 2 — see module docstring for the column inventory), and
/// the cfg uniforms for the producer (`physics_WhatIBelieve`) +
/// consumer (`fold_beliefs_flags`) kernels.
pub struct TomProbeState {
    gpu: GpuContext,

    // -- Agent SoA (read by physics_WhatIBelieve to gate `self.alive`) --
    /// 1 = alive, 0 = dead. Initialised all-1 so every Knower fires
    /// the producer rule each tick.
    agent_alive_buf: wgpu::Buffer,

    // -- Agent SoA (Wave 3 ToM Phase 3) — addressed by the `observe`
    // consumer to read target's CURRENT pos / creature_type at consume
    // tick. These columns aren't read by any DSL kernel today (the
    // existing Knower entity declares `pos: vec3, vel: vec3` but the
    // .sim consumer doesn't bind them); they're allocated runtime-side
    // so the observe round-trip pin can pre-seed target's state and
    // verify the consumer copies it into the BeliefState SoA.
    //
    // `agent_pos`: vec4 f32 padded (vec3 padded to vec4 for std430 +
    // future-proofing parity with the existing `beliefs_pos` column).
    // `agent_creature_type`: u8 (one byte per slot). Both share the
    // same `agent_count`-sized footprint.
    agent_pos_buf: wgpu::Buffer,
    agent_pos_staging: wgpu::Buffer,
    agent_pos_cache: Vec<[f32; 4]>,
    agent_pos_dirty: bool,

    agent_creature_type_buf: wgpu::Buffer,
    agent_creature_type_staging: wgpu::Buffer,
    agent_creature_type_cache: Vec<u8>,
    agent_creature_type_dirty: bool,

    // -- BeliefState SoA: `flags` column (Wave 3 ToM Phase 1 + Phase 2) --
    //
    // `pair_map`-keyed: `agent_cap × agent_cap × u32`. The fold body
    // indexes `view_storage_primary[observer * cfg.second_key_pop +
    // subject]`. We allocate this locally (instead of via
    // `engine::gpu::ViewStorage`) so the host-side readback can
    // surface a `&[u32]` directly without an f32 bitcast round-trip.
    // Renamed from `beliefs_*` → `beliefs_flags_*` in Phase 2 to
    // disambiguate from the other 5 BeliefState columns below.
    beliefs_flags_primary: wgpu::Buffer,
    beliefs_flags_staging: wgpu::Buffer,
    beliefs_flags_cache: Vec<u32>,
    beliefs_flags_dirty: bool,

    // -- BeliefState SoA: `last_known_pos` column (Phase 2) --
    //
    // 4×f32 per cell (vec3 padded to vec4 for std430 alignment) so
    // `agent_cap × agent_cap` cells take `agent_cap^2 * 16` bytes.
    // The host-side cache is `Vec<[f32; 4]>` so the readback can
    // surface positions without a per-row reshape.
    beliefs_pos_primary: wgpu::Buffer,
    beliefs_pos_staging: wgpu::Buffer,
    beliefs_pos_cache: Vec<[f32; 4]>,
    beliefs_pos_dirty: bool,

    // -- BeliefState SoA: `last_known_creature_type` column (Phase 2) --
    //
    // u8 per cell, packed as one byte per slot. Host cache is
    // `Vec<u8>` (no padding); GPU buffer rounds size to a 16-byte
    // multiple to satisfy std430 + the wgpu min-binding-size guards.
    beliefs_type_primary: wgpu::Buffer,
    beliefs_type_staging: wgpu::Buffer,
    beliefs_type_cache: Vec<u8>,
    beliefs_type_dirty: bool,

    // -- BeliefState SoA: `last_seen_tick` column (Phase 2) --
    //
    // u32 per cell — same shape as `flags` minus the `atomic<>`
    // qualifier (Phase 2 has no per-event writers contending on this
    // column; Phase 3 will introduce them and may need to switch to
    // `atomic<u32>` per usage pattern).
    beliefs_tick_primary: wgpu::Buffer,
    beliefs_tick_staging: wgpu::Buffer,
    beliefs_tick_cache: Vec<u32>,
    beliefs_tick_dirty: bool,

    // -- BeliefState SoA: `confidence` column (Phase 2) --
    //
    // u8 per cell (q8 fixed-point in 0..255 representing 0.0..1.0).
    // Same packing strategy as `creature_type`.
    beliefs_confidence_primary: wgpu::Buffer,
    beliefs_confidence_staging: wgpu::Buffer,
    beliefs_confidence_cache: Vec<u8>,
    beliefs_confidence_dirty: bool,

    // -- BeliefState SoA: `suspicion` column (Phase 2) --
    //
    // u8 per cell (q8 — observer's hostility / lying suspicion of
    // the target). Same packing strategy as `creature_type`.
    beliefs_suspicion_primary: wgpu::Buffer,
    beliefs_suspicion_staging: wgpu::Buffer,
    beliefs_suspicion_cache: Vec<u8>,
    beliefs_suspicion_dirty: bool,

    // -- Event ring + per-kernel cfg uniforms --
    event_ring: EventRing,
    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,
    /// Wave 3 ToM Phase 3.7 — cfg uniform for the compiler-emitted
    /// `physics_ApplyTickBeliefStamp` kernel (PerEventEmit shape:
    /// `{ event_count, tick, seed, agent_cap }`). The kernel's body
    /// references `cfg.agent_cap` from inside the
    /// `agents_set_beliefs_last_seen_tick` setter stub for the cell-
    /// index expression.
    apply_tick_belief_stamp_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    /// Wave 3 ToM Phase 3.6 — hand-written GPU pipelines for the
    /// observe / scry / reveal consumers + per-tick decay sweep.
    /// Replaces the Phase 3/3.5 runtime-side CPU `queue.write_buffer`
    /// column-rewrite trick with proper compute dispatches.
    consumer_gpu: BeliefConsumerGpu,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Allocate a paired (primary, staging) buffer of `bytes` bytes for
/// a single BeliefState SoA column. The primary carries
/// `STORAGE | COPY_SRC | COPY_DST` (kernel can write, host can
/// readback + seed); staging is `MAP_READ | COPY_DST`. The pair is
/// the standard wgpu shape for "GPU-resident readable buffer with
/// off-line CPU mapping".
fn alloc_belief_column_pair(
    device: &wgpu::Device,
    label_primary: &str,
    label_staging: &str,
    bytes: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let bytes = bytes.max(16); // wgpu rejects sub-16B storage allocations
    let primary = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label_primary),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label_staging),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (primary, staging)
}

impl TomProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Knower SoA — `agent_alive` is the only field the producer
        // rule reads (`where (self.alive)`). Every slot starts alive
        // so every tick fires.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Wave 3 ToM Phase 3 agent SoA — `agent_pos` (vec4 padded) and
        // `agent_creature_type` (u8). Allocated paired (primary +
        // staging) so the observe consumer can read them off-line into
        // host memory and write the result into the BeliefState SoA's
        // 6 columns. Both start zero-filled; tests pre-seed via
        // `seed_agent_pos` / `seed_agent_creature_type`.
        let agent_pos_bytes = (agent_count as u64) * 16;
        let (agent_pos_buf, agent_pos_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::agent_pos_primary",
            "tom_probe_runtime::agent_pos_staging",
            agent_pos_bytes,
        );
        let agent_type_bytes = (((agent_count as u64) + 15) / 16 * 16).max(16);
        let (agent_creature_type_buf, agent_creature_type_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::agent_creature_type_primary",
                "tom_probe_runtime::agent_creature_type_staging",
                agent_type_bytes,
            );

        // -- BeliefState SoA columns --
        //
        // All 6 columns share the same `agent_count × agent_count`
        // cell-count footprint and the `observer * agent_count +
        // target` indexing convention. The byte sizes differ per
        // column type:
        //   - flags / tick      : 4 B per cell  (u32)
        //   - pos               : 16 B per cell (vec4f, vec3 padded)
        //   - type / confidence / suspicion : 1 B per cell (u8)
        let cell_count = (agent_count as u64) * (agent_count as u64);

        let flags_bytes = cell_count * 4;
        let (beliefs_flags_primary, beliefs_flags_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_flags_primary",
            "tom_probe_runtime::beliefs_flags_staging",
            flags_bytes,
        );

        let pos_bytes = cell_count * 16;
        let (beliefs_pos_primary, beliefs_pos_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_pos_primary",
            "tom_probe_runtime::beliefs_pos_staging",
            pos_bytes,
        );

        // u8 columns share an "rounded up to a 16-byte multiple"
        // size so std430 alignment + min-binding-size guards stay
        // happy even at small agent counts. The CPU cache is sized
        // to the exact cell count (no padding); readback only copies
        // `cell_count` bytes back into the cache.
        let u8_buf_bytes = ((cell_count + 15) / 16 * 16).max(16);
        let (beliefs_type_primary, beliefs_type_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_type_primary",
            "tom_probe_runtime::beliefs_type_staging",
            u8_buf_bytes,
        );

        let tick_bytes = cell_count * 4;
        let (beliefs_tick_primary, beliefs_tick_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_tick_primary",
            "tom_probe_runtime::beliefs_tick_staging",
            tick_bytes,
        );

        let (beliefs_confidence_primary, beliefs_confidence_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::beliefs_confidence_primary",
                "tom_probe_runtime::beliefs_confidence_staging",
                u8_buf_bytes,
            );

        let (beliefs_suspicion_primary, beliefs_suspicion_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::beliefs_suspicion_primary",
                "tom_probe_runtime::beliefs_suspicion_staging",
                u8_buf_bytes,
            );

        let event_ring = EventRing::new(&gpu, "tom_probe_runtime");

        let physics_cfg_init =
            physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: 0,
            };
        let physics_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::physics_WhatIBelieve_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let fold_cfg_init = fold_beliefs_flags::FoldBeliefsFlagsCfg {
            event_count: 0,
            tick: 0,
            // `beliefs_flags(observer: Agent, subject: Agent)` —
            // both keys are Agent, so second_key_pop = agent_count.
            // The fold body composes
            // `view_storage_primary[k1 * second_key_pop + k2]` so
            // this MUST equal agent_count for the diagonal to land
            // at index `i * N + i`.
            second_key_pop: agent_count,
            _pad: 0,
        };
        let fold_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::fold_beliefs_flags_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let consumer_gpu = BeliefConsumerGpu::new(&gpu.device);

        // Wave 3 ToM Phase 3.7 — cfg buf for the compiler-emitted
        // `physics_ApplyTickBeliefStamp` consumer kernel. Per-tick
        // updates write `event_count + tick + agent_cap` into this
        // buffer before the dispatch in `step()`.
        let apply_tick_belief_stamp_cfg_init =
            physics_ApplyTickBeliefStamp::PhysicsApplyTickBeliefStampCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: agent_count,
            };
        let apply_tick_belief_stamp_cfg_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(
                    "tom_probe_runtime::physics_ApplyTickBeliefStamp_cfg",
                ),
                contents: bytemuck::bytes_of(&apply_tick_belief_stamp_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            agent_pos_buf,
            agent_pos_staging,
            agent_pos_cache: vec![[0.0f32; 4]; agent_count as usize],
            agent_pos_dirty: false,
            agent_creature_type_buf,
            agent_creature_type_staging,
            agent_creature_type_cache: vec![0u8; agent_count as usize],
            agent_creature_type_dirty: false,
            beliefs_flags_primary,
            beliefs_flags_staging,
            beliefs_flags_cache: vec![0u32; cell_count as usize],
            beliefs_flags_dirty: false,
            beliefs_pos_primary,
            beliefs_pos_staging,
            beliefs_pos_cache: vec![[0.0f32; 4]; cell_count as usize],
            beliefs_pos_dirty: false,
            beliefs_type_primary,
            beliefs_type_staging,
            beliefs_type_cache: vec![0u8; cell_count as usize],
            beliefs_type_dirty: false,
            beliefs_tick_primary,
            beliefs_tick_staging,
            beliefs_tick_cache: vec![0u32; cell_count as usize],
            beliefs_tick_dirty: false,
            beliefs_confidence_primary,
            beliefs_confidence_staging,
            beliefs_confidence_cache: vec![0u8; cell_count as usize],
            beliefs_confidence_dirty: false,
            beliefs_suspicion_primary,
            beliefs_suspicion_staging,
            beliefs_suspicion_cache: vec![0u8; cell_count as usize],
            beliefs_suspicion_dirty: false,
            event_ring,
            physics_cfg_buf,
            fold_cfg_buf,
            apply_tick_belief_stamp_cfg_buf,
            cache: dispatch::KernelCache::default(),
            consumer_gpu,
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-(observer, subject) belief flag bitset, flattened
    /// row-major: slot `[observer * agent_count + subject]` holds
    /// the OR-folded fact bits the observer believes about the
    /// subject. Length = `agent_count × agent_count`. This is the
    /// `flags` column of the BeliefState SoA — the only column
    /// folded over a chronicle event in Phase 2 (Phase 1's
    /// `BeliefAcquired` shape preserved).
    pub fn beliefs_flags(&mut self) -> &[u32] {
        if self.beliefs_flags_dirty {
            let bytes = (self.beliefs_flags_cache.len() as u64) * 4;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_flags::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_flags_primary,
                0,
                &self.beliefs_flags_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_flags_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.beliefs_flags_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_flags_staging.unmap();
            self.beliefs_flags_dirty = false;
        }
        &self.beliefs_flags_cache
    }

    /// Per-(observer, subject) `last_known_pos` column. Each cell is
    /// a `[f32; 4]` (vec3 padded to vec4). Wave 3 ToM Phase 2
    /// runtime-only storage; no fold writer today (writer lands in
    /// Phase 3).
    pub fn beliefs_pos(&mut self) -> &[[f32; 4]] {
        if self.beliefs_pos_dirty {
            let bytes = (self.beliefs_pos_cache.len() as u64) * 16;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_pos::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_pos_primary,
                0,
                &self.beliefs_pos_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            self.beliefs_pos_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_pos_staging.unmap();
            self.beliefs_pos_dirty = false;
        }
        &self.beliefs_pos_cache
    }

    /// Per-(observer, subject) `last_known_creature_type` column.
    /// Each cell is a `u8` classification ordinal. Wave 3 ToM
    /// Phase 2 runtime-only storage.
    pub fn beliefs_type(&mut self) -> &[u8] {
        if self.beliefs_type_dirty {
            let cell_count = self.beliefs_type_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_type::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_type_primary,
                0,
                &self.beliefs_type_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_type_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_type_cache.copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_type_staging.unmap();
            self.beliefs_type_dirty = false;
        }
        &self.beliefs_type_cache
    }

    /// Per-(observer, subject) `last_seen_tick` column. Each cell
    /// is a `u32` tick number powering decay computations. Wave 3
    /// ToM Phase 2 runtime-only storage.
    pub fn beliefs_tick(&mut self) -> &[u32] {
        if self.beliefs_tick_dirty {
            let bytes = (self.beliefs_tick_cache.len() as u64) * 4;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_tick::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_tick_primary,
                0,
                &self.beliefs_tick_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_tick_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.beliefs_tick_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_tick_staging.unmap();
            self.beliefs_tick_dirty = false;
        }
        &self.beliefs_tick_cache
    }

    /// Per-(observer, subject) `confidence` column. Each cell is a
    /// `u8` q8 value in 0..255 representing 0.0..1.0. Wave 3 ToM
    /// Phase 2 runtime-only storage.
    pub fn beliefs_confidence(&mut self) -> &[u8] {
        if self.beliefs_confidence_dirty {
            let cell_count = self.beliefs_confidence_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_confidence::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_confidence_primary,
                0,
                &self.beliefs_confidence_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_confidence_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_confidence_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_confidence_staging.unmap();
            self.beliefs_confidence_dirty = false;
        }
        &self.beliefs_confidence_cache
    }

    /// Per-(observer, subject) `suspicion` column. Each cell is a
    /// `u8` q8 value in 0..255 representing how much the observer
    /// suspects the target is hostile / lying. Wave 3 ToM Phase 2
    /// runtime-only storage.
    pub fn beliefs_suspicion(&mut self) -> &[u8] {
        if self.beliefs_suspicion_dirty {
            let cell_count = self.beliefs_suspicion_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_suspicion::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_suspicion_primary,
                0,
                &self.beliefs_suspicion_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_suspicion_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_suspicion_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_suspicion_staging.unmap();
            self.beliefs_suspicion_dirty = false;
        }
        &self.beliefs_suspicion_cache
    }

    // -- Test-only seed helpers (Wave 3 ToM Phase 2) --
    //
    // The pre-fix storage path (Phase 1) was driven entirely by the
    // `BeliefAcquired` event flowing through `fold_beliefs_flags`.
    // Phase 2 adds 5 more columns with no event-driven writers
    // today, so the regression pin needs an off-band channel to seed
    // each column with known bytes and assert the readback shape.
    //
    // Each `seed_*` method takes a host-side slice sized to
    // `agent_count^2` and writes it into the corresponding primary
    // GPU buffer via `Queue::write_buffer`. The dirty flag is
    // toggled so the next read-back picks up the seeded bytes.

    /// Test-only: seed the entire `flags` column.
    pub fn seed_beliefs_flags(&mut self, values: &[u32]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_flags expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_flags_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_flags_dirty = true;
    }

    /// Test-only: seed the entire `last_known_pos` column. Each
    /// cell is a `[f32; 4]` (vec3 padded to vec4).
    pub fn seed_beliefs_pos(&mut self, values: &[[f32; 4]]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_pos expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_pos_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_pos_dirty = true;
    }

    /// Test-only: seed the entire `last_known_creature_type`
    /// column.
    pub fn seed_beliefs_type(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_type expected {expected} cells, got {}",
            values.len(),
        );
        // Pad to the underlying buffer size (16-byte aligned) with
        // zero bytes so write_buffer doesn't reject a sub-buffer
        // write.
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_type_primary, 0, &padded);
        self.beliefs_type_dirty = true;
    }

    /// Test-only: seed the entire `last_seen_tick` column.
    pub fn seed_beliefs_tick(&mut self, values: &[u32]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_tick expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_tick_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_tick_dirty = true;
    }

    /// Test-only: seed the entire `confidence` column (q8 in
    /// 0..255).
    pub fn seed_beliefs_confidence(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_confidence expected {expected} cells, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_confidence_primary, 0, &padded);
        self.beliefs_confidence_dirty = true;
    }

    /// Test-only: seed the entire `suspicion` column (q8 in
    /// 0..255).
    pub fn seed_beliefs_suspicion(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_suspicion expected {expected} cells, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_suspicion_primary, 0, &padded);
        self.beliefs_suspicion_dirty = true;
    }

    /// Number of (observer, target) cells in each BeliefState SoA
    /// column = `agent_count^2`. Convenience for test code that
    /// pre-allocates seed vectors.
    pub fn cell_count(&self) -> u32 {
        self.agent_count * self.agent_count
    }

    // -- Wave 3 ToM Phase 3 agent SoA seeders + readers --
    //
    // The observe consumer reads target's CURRENT pos / creature_type
    // from the agent SoA at consume tick. These are runtime-only
    // helpers (the .sim consumer doesn't bind these columns today —
    // the existing Knower entity declares `pos: vec3, vel: vec3` but
    // no kernel reads them). The closed-loop `observe_round_trip_pin`
    // pre-seeds target B's state via `seed_agent_pos` /
    // `seed_agent_creature_type`, calls `observe(observer_a, target_b,
    // tick)`, then asserts the BeliefState SoA's 6 columns at
    // `[a * N + b]` reflect what was seeded.

    /// Test-only: seed agent SoA `pos` column. Each cell is a
    /// `[f32; 4]` (vec3 padded to vec4 for std430 alignment).
    pub fn seed_agent_pos(&mut self, values: &[[f32; 4]]) {
        let expected = self.agent_count as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_agent_pos expected {expected} agents, got {}",
            values.len(),
        );
        self.gpu
            .queue
            .write_buffer(&self.agent_pos_buf, 0, bytemuck::cast_slice(values));
        self.agent_pos_dirty = true;
    }

    /// Test-only: seed agent SoA `creature_type` column. Each cell is
    /// a `u8` classification ordinal.
    pub fn seed_agent_creature_type(&mut self, values: &[u8]) {
        let expected = self.agent_count as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_agent_creature_type expected {expected} agents, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.agent_creature_type_buf, 0, &padded);
        self.agent_creature_type_dirty = true;
    }

    /// Read the agent SoA `pos` column back to host memory. The cache
    /// is staged on first read and reused until any seeder dirties it.
    pub fn agent_pos(&mut self) -> &[[f32; 4]] {
        if self.agent_pos_dirty {
            let bytes = (self.agent_pos_cache.len() as u64) * 16;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::agent_pos::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.agent_pos_buf,
                0,
                &self.agent_pos_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.agent_pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            self.agent_pos_cache.copy_from_slice(raw);
            drop(mapped);
            self.agent_pos_staging.unmap();
            self.agent_pos_dirty = false;
        }
        &self.agent_pos_cache
    }

    /// Read the agent SoA `creature_type` column back to host memory.
    pub fn agent_creature_type(&mut self) -> &[u8] {
        if self.agent_creature_type_dirty {
            let cell_count = self.agent_creature_type_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::agent_creature_type::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.agent_creature_type_buf,
                0,
                &self.agent_creature_type_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.agent_creature_type_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.agent_creature_type_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.agent_creature_type_staging.unmap();
            self.agent_creature_type_dirty = false;
        }
        &self.agent_creature_type_cache
    }

    /// Wave 3 ToM Phase 3.6 — GPU observe consumer. Dispatches a single
    /// hand-written WGSL workgroup that reads target's CURRENT pos +
    /// creature_type from the agent SoA and writes into ALL 6
    /// BeliefState SoA columns at `[observer * agent_count + target]`:
    ///
    ///   - `last_known_pos[idx]` ← target's current pos
    ///   - `last_known_creature_type[idx]` ← target's current type
    ///   - `last_seen_tick[idx]` ← `current_tick`
    ///   - `confidence[idx]` ← 255 (max — fresh observation)
    ///   - `flags[idx]` ← unchanged (Phase 1 bit-OR slot; observe
    ///     doesn't set bits)
    ///   - `suspicion[idx]` ← unchanged (Phase 4 deception verbs
    ///     mutate this; observe leaves it alone)
    ///
    /// Replaces the Phase 3 CPU consumer (which read both agent columns
    /// off-line, patched 6 host-side buffers, and re-uploaded the whole
    /// u8 columns to dodge the 4-byte-multiple write constraint —
    /// O(N²) per call). The GPU consumer is one workgroup × 1 thread,
    /// one dispatch — no staging readback, no host-side column rewrite.
    pub fn observe(&mut self, observer: u32, target: u32) {
        assert!(
            observer < self.agent_count,
            "observe: observer {observer} >= agent_count {}",
            self.agent_count,
        );
        assert!(
            target < self.agent_count,
            "observe: target {target} >= agent_count {}",
            self.agent_count,
        );

        let cfg = BeliefConsumerCfg {
            observer,
            target_or_subject: target,
            subject: 0,
            agent_count: self.agent_count,
            tick: self.tick as u32,
            cell_count: self.cell_count(),
            _pad0: 0,
            _pad1: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.consumer_gpu.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tom_probe_runtime::observe::bg"),
                layout: &self.consumer_gpu.observe_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.consumer_gpu.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.agent_pos_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.agent_creature_type_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.beliefs_pos_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.beliefs_type_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.beliefs_tick_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.beliefs_confidence_primary.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::observe::encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tom_probe_runtime::observe::pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.consumer_gpu.observe_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        self.beliefs_pos_dirty = true;
        self.beliefs_type_dirty = true;
        self.beliefs_tick_dirty = true;
        self.beliefs_confidence_dirty = true;
    }

    /// Wave 3 ToM Phase 3.6 — GPU scry consumer. Single workgroup × 1
    /// thread; 6-field cross-observer copy. The kernel reads all 6
    /// source-cell values into registers BEFORE writing the destination
    /// so self-scry (src == dst) preserves the source-cell payload.
    pub fn scry(&mut self, observer: u32, target_observer: u32, subject: u32) {
        assert!(
            observer < self.agent_count,
            "scry: observer {observer} >= agent_count {}",
            self.agent_count,
        );
        assert!(
            target_observer < self.agent_count,
            "scry: target_observer {target_observer} >= agent_count {}",
            self.agent_count,
        );
        assert!(
            subject < self.agent_count,
            "scry: subject {subject} >= agent_count {}",
            self.agent_count,
        );

        let cfg = BeliefConsumerCfg {
            observer,
            target_or_subject: target_observer,
            subject,
            agent_count: self.agent_count,
            tick: self.tick as u32,
            cell_count: self.cell_count(),
            _pad0: 0,
            _pad1: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.consumer_gpu.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tom_probe_runtime::scry::bg"),
                layout: &self.consumer_gpu.scry_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.consumer_gpu.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.beliefs_pos_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.beliefs_type_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.beliefs_tick_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.beliefs_confidence_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.beliefs_suspicion_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.beliefs_flags_primary.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::scry::encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tom_probe_runtime::scry::pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.consumer_gpu.scry_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        self.beliefs_pos_dirty = true;
        self.beliefs_type_dirty = true;
        self.beliefs_tick_dirty = true;
        self.beliefs_confidence_dirty = true;
        self.beliefs_suspicion_dirty = true;
        self.beliefs_flags_dirty = true;
    }

    /// Wave 3 ToM Phase 3.6 — GPU reveal consumer. One thread per
    /// observer slot; each thread reads `[caster * N + subject]` and
    /// writes its own `[observer_slot * N + subject]` cell (the caster's
    /// own slot is included as a self-write — idempotent).
    ///
    /// Phase 3.5 ships the "all observers" fan-out shape; range-gated
    /// reveal lands when spatial-query infrastructure for "observers
    /// within radius of caster" matures.
    pub fn reveal(&mut self, caster: u32, subject: u32) {
        assert!(
            caster < self.agent_count,
            "reveal: caster {caster} >= agent_count {}",
            self.agent_count,
        );
        assert!(
            subject < self.agent_count,
            "reveal: subject {subject} >= agent_count {}",
            self.agent_count,
        );

        let cfg = BeliefConsumerCfg {
            observer: caster,
            target_or_subject: subject,
            subject: 0,
            agent_count: self.agent_count,
            tick: self.tick as u32,
            cell_count: self.cell_count(),
            _pad0: 0,
            _pad1: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.consumer_gpu.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tom_probe_runtime::reveal::bg"),
                layout: &self.consumer_gpu.reveal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.consumer_gpu.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.beliefs_pos_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.beliefs_type_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.beliefs_tick_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.beliefs_confidence_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.beliefs_suspicion_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.beliefs_flags_primary.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::reveal::encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tom_probe_runtime::reveal::pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.consumer_gpu.reveal_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // workgroup_size = 64 in the kernel — one workgroup covers
            // every fixture today (max agent_count = 16 in the round-trip
            // pins). div_ceil keeps shape correct for larger fixtures
            // if/when they appear.
            let workgroups = self.agent_count.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        self.beliefs_pos_dirty = true;
        self.beliefs_type_dirty = true;
        self.beliefs_tick_dirty = true;
        self.beliefs_confidence_dirty = true;
        self.beliefs_suspicion_dirty = true;
        self.beliefs_flags_dirty = true;
    }

    /// Wave 3 ToM Phase 3.6 — GPU per-tick decay phase. Replaces the
    /// runtime-side CPU sweep with a hand-written compute kernel that
    /// processes one packed-byte word (4 cells) per thread:
    ///
    ///   - Loads the word from `beliefs_confidence` and the 4
    ///     corresponding `beliefs_tick` values.
    ///   - For each of the 4 cells: skip if observed-this-tick or
    ///     never-observed, else `saturating_sub(conf, 1)`.
    ///   - Repacks and stores the new word — no atomic needed since
    ///     each thread owns a unique word.
    ///
    /// **Decay slope rationale.** The spec sketch in the Phase 3 brief
    /// initially read `new_confidence = saturating_sub(confidence,
    /// (world.tick - last_seen_tick) * decay_per_tick)` — but applying
    /// that formula every tick double-counts the delta (`tick 1` =
    /// drop 1, `tick 2` = drop 2 more, etc., summing to N(N+1)/2 instead
    /// of the linear N). The companion pin
    /// (`decay_pin.rs::decay_slope_is_one_per_tick`) explicitly asserts
    /// `confidence == 155` after 100 ticks (= 255 - 100), so the
    /// implementation must produce a *linear* slope. The simplest
    /// linear formulation is "decrement by 1 per call, skipping cells
    /// observed this tick" — which matches the observable target slope
    /// without needing a baseline-confidence register.
    ///
    /// **Why a separate `decay_step` API instead of folding into
    /// `step()`.** The `decay_pin.rs` regression intentionally exercises
    /// the order `step() → decay_step()`; collapsing the two into a
    /// single `step()` call would make the semantics implicit and break
    /// the test's "fresh observation resets decay" assertion (which
    /// calls observe between ticks before the next decay sweep). Phase
    /// 3.6 keeps the API split; the `@decay(beliefs)` annotation that
    /// would auto-trigger the kernel from the .sim surface is deferred.
    ///
    /// `decay_per_tick = 1` is the canonical rate, hardcoded in
    /// `DECAY_WGSL`. Pinned by `decay_pin.rs`.
    pub fn decay_step(&mut self) {
        let cfg = BeliefConsumerCfg {
            observer: 0,
            target_or_subject: 0,
            subject: 0,
            agent_count: self.agent_count,
            tick: self.tick as u32,
            cell_count: self.cell_count(),
            _pad0: 0,
            _pad1: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.consumer_gpu.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tom_probe_runtime::decay::bg"),
                layout: &self.consumer_gpu.decay_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.consumer_gpu.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.beliefs_confidence_primary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.beliefs_tick_primary.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::decay::encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tom_probe_runtime::decay::pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.consumer_gpu.decay_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One thread per packed-byte word in the confidence column;
            // workgroup_size = 64 in the kernel.
            let word_count = self.cell_count().div_ceil(4);
            let workgroups = word_count.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        self.beliefs_confidence_dirty = true;
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

impl CompiledSim for TomProbeState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("tom_probe_runtime::step"),
            },
        );

        // (1) Per-tick clear of event_tail. The producer rule
        // atomicAdd's against it during physics_WhatIBelieve to
        // acquire write slots; the count accumulates over the tick
        // and the fold kernel reads it via cfg.event_count. Clearing
        // here guarantees a fresh per-tick slot count even though
        // event slots from prior ticks linger in the ring (the fold
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters stale slots).
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) physics_WhatIBelieve — per-Knower; emits one
        // `BeliefAcquired { observer: self, subject: self, fact_bit:
        // 1 }` per tick when `self.alive`.
        let physics_cfg = physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );
        let physics_bindings =
            physics_WhatIBelieve::PhysicsWhatIBelieveBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_alive: &self.agent_alive_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_whatibelieve(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_beliefs_flags — per-event; OR's `fact_bit` into
        // `beliefs_flags_primary[observer * agent_cap + subject]`
        // via `atomicOr`. We size event_count = agent_count: every
        // alive Knower emits exactly one event per tick, so this is
        // the exact upper bound (no skip / no over-dispatch). The
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters anything beyond the producer's per-tick batch even
        // if leftover slots from prior ticks remain in the ring.
        let event_count_estimate = self.agent_count;
        let fold_cfg = fold_beliefs_flags::FoldBeliefsFlagsCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.fold_cfg_buf,
            0,
            bytemuck::bytes_of(&fold_cfg),
        );
        let fold_bindings = fold_beliefs_flags::FoldBeliefsFlagsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.beliefs_flags_primary,
            // No `@decay` and no top-K → no anchor / no ids; the
            // generated `record()` body falls back to primary via
            // `unwrap_or(primary_buf)` per `kernel.rs`'s slot-aliasing
            // convention.
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_beliefs_flags(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // (4) Wave 3 ToM Phase 3.7 — compiler-emitted
        // `physics_ApplyTickBeliefStamp` consumer. Reads
        // `BeliefAcquired` events from the event_ring (same ring the
        // fold above just folded over) and writes
        // `agents.set_beliefs_last_seen_tick(o, s, b)` for each event.
        // The setter WGSL body (compiler-emitted from the .sim source)
        // does `beliefs_tick[o * cfg.agent_cap + s] = b`. End-to-end
        // proof of the setter pipe — no hand-written WGSL on this path.
        let apply_tick_cfg =
            physics_ApplyTickBeliefStamp::PhysicsApplyTickBeliefStampCfg {
                event_count: event_count_estimate,
                tick: self.tick as u32,
                seed: 0,
                agent_cap: self.agent_count,
            };
        self.gpu.queue.write_buffer(
            &self.apply_tick_belief_stamp_cfg_buf,
            0,
            bytemuck::bytes_of(&apply_tick_cfg),
        );
        let apply_tick_bindings =
            physics_ApplyTickBeliefStamp::PhysicsApplyTickBeliefStampBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                beliefs_tick: &self.beliefs_tick_primary,
                cfg: &self.apply_tick_belief_stamp_cfg_buf,
            };
        dispatch::dispatch_physics_applytickbeliefstamp(
            &mut self.cache,
            &apply_tick_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.beliefs_flags_dirty = true;
        self.beliefs_tick_dirty = true;
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. Same shape
        // as verb_probe_runtime (which has the same comment).
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TomProbeState::new(seed, agent_count))
}
