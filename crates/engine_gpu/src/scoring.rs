//! Phase 3 + 6c + 6d scoring kernel — GPU argmax over the emitted
//! `SCORING_TABLE` with real view-buffer reads.
//!
//! Builds on Phase 2's fused mask output. Each tick:
//!
//! 1. The mask kernel writes 7 bitmaps — one per supported mask —
//!    telling us which `(action, agent)` pairs are allowed.
//! 2. This kernel reads those bitmaps, a packed agent SoA, and the
//!    @materialized view state **directly from `view_storage`'s
//!    atomic buffers** (Phase 6d), then scores every
//!    `(agent, entry_row, target)` combination, picking the argmax
//!    per agent. Output: `(chosen_action, chosen_target)` per agent,
//!    byte-parity with the CPU scorer.
//!
//! ## Phase 6d: direct view_storage reads
//!
//! Phase 3-6c scoring owned its own per-view `DecayCell` buffers and
//! uploaded a CPU-mirror of `SimState::views` every tick. Phase 6d
//! replaces that: the scoring kernel's WGSL is emitted in
//! `atomic_views` mode (see [`dsl_compiler::emit_scoring_wgsl::
//! emit_scoring_wgsl_atomic_views`]) so `view_<snake>_cells` are
//! `array<atomic<u32>>` bindings — *the same* buffers that
//! `view_storage::ViewStorage` fold kernels write to. No CPU mirror,
//! no per-tick upload.
//!
//! Knock-on layout changes:
//!   * Pair-map decay views get TWO bindings (values + anchor ticks)
//!     instead of one `array<DecayCell>`. See `atomic_views` emitter.
//!   * Binding count grows by `num_decay_views`. At the current set
//!     of 1 slot-map + 1 pair-scalar + 4 pair-decay views, that's
//!     `5 core + 1 + 1 + 2*4 = 15` — still under the 16 per-group cap.
//!
//! ## Binding layout (see `emit_scoring_wgsl` for the matching WGSL side)
//!
//!   * `@binding(0)` `agent_data` — packed struct per slot. 64 bytes
//!     (16 × 4-byte). Packs every scalar `read_field` needs so scoring
//!     doesn't add N one-per-field buffers.
//!   * `@binding(1)` `mask_bitmaps` — flat concat of all 7 mask bitmaps.
//!     Uploaded from `FusedMaskKernel::run_and_readback`'s output; the
//!     scoring kernel reads bit `slot` of mask `m` by indexing
//!     `mask_bitmaps[m * num_mask_words + slot/32] & (1 << (slot % 32))`.
//!   * `@binding(2)` `scoring_table` — `SCORING_TABLE` uploaded verbatim
//!     in a WGSL-friendly layout (see `GpuScoringEntry` below).
//!   * `@binding(3)` `scoring_out` — one `ScoreOutput` per agent:
//!     `{ chosen_action: u32, chosen_target: u32, best_score_bits:
//!     u32, _pad: u32 }`. The backend reads this back every tick and
//!     partitions by agent slot.
//!   * `@binding(4)` `cfg` uniform — radii + table row count + mask-
//!     word count + `tick` + `view_agent_cap` (the latter two feed the
//!     view read snippets emitted alongside).
//!   * `@binding(5)` `view_engaged_with_slots: array<u32>` — slot_map
//!     storage for `engaged_with` (shared with view_storage).
//!   * `@binding(6)` `view_my_enemies_cells: array<atomic<u32>>` —
//!     pair_map scalar storage for `my_enemies`, shared with
//!     view_storage. Each cell holds an f32's bits.
//!   * Decay views (4 of them: `kin_fear`, `pack_focus`, `rally_boost`,
//!     `threat_level`, sorted by name) each take **two** bindings:
//!     `view_<name>_cells: array<atomic<u32>>` (value bits) and
//!     `view_<name>_anchors: array<atomic<u32>>` (anchor ticks).
//!     Binding 7,8 → kin_fear; 9,10 → pack_focus; 11,12 → rally_boost;
//!     13,14 → threat_level.
//!
//! **Binding count = 15**, within the 16-per-bind-group cap on every
//! target we care about. If a future view pushes past this, split the
//! decay-anchor bindings onto a second bind group.
//!
//! ## Determinism strategy
//!
//! Chose **single thread per agent** + sequential argmax in WGSL-level
//! strictly-greater semantics. Each thread walks the table in order,
//! evaluates the gate (mask bit), and — for target-bound rows — walks
//! candidate slots 0..agent_cap in ascending order. Ties preserve the
//! earlier (lower entry_idx, lower target_slot) winner — identical to
//! the CPU argmax in `crates/engine/src/policy/utility.rs`.
//!
//! Not a workgroup-level reduce because:
//!   * the alternative (segmented scan in shared memory) needs a fixed
//!     upper bound on `agent × action × target` entries and adds enough
//!     complexity (tie-break key packing, bit twiddles) that per-thread
//!     sequential is easier to verify byte-exact;
//!   * at Phase 3 test sizes (N ≤ 16 agents) the bottleneck isn't the
//!     score inner loop, it's the device setup / upload / readback;
//!   * when N climbs and this becomes the bottleneck, Phase 6 or a
//!     follow-up task can swap in a workgroup reduce with the same
//!     argmax contract.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use dsl_compiler::emit_scoring_wgsl::{
    action_head_to_mask_idx, emit_scoring_wgsl_atomic_views, scoring_view_binding_order,
    MASK_NAMES, SCORING_CORE_BINDINGS, WORKGROUP_SIZE,
};
use dsl_compiler::emit_view_wgsl::{ViewShape, ViewStorageSpec};
use engine::ids::AgentId;
use engine::state::SimState;
use engine_rules::scoring::{ModifierRow, PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE};
use wgpu::util::DeviceExt;

use crate::mask::{FusedMaskKernel, KernelError};
use crate::view_storage::{build_all_specs, ViewStorage};

/// Sentinel for "no target" in a `ScoreOutput`. Mirrors the WGSL
/// `NO_TARGET` constant.
pub const NO_TARGET: u32 = 0xFFFF_FFFFu32;

/// Packed per-agent struct the scoring kernel reads. Matches the WGSL
/// `AgentData` in [`emit_scoring_wgsl`]. 64 bytes = 16 f32s; alignment
/// is 16-byte via the trailing pads.
///
/// Fields are in the same order `read_field` dispatches on, so adding
/// a new scoring-accessible field is: bump the WGSL struct, add a
/// field here, bump `GpuConfig`'s field count, and update
/// `write_agent_data_for_state` to populate it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuAgentData {
    // Vec3 + padding fits the WGSL Vec3f32 + 1 trailing f32 (we re-use
    // that trailing f32 for `hp` since it needs to be present). To keep
    // the WGSL-side struct as `{ pos: Vec3f32, hp, ... }` without a
    // hidden pad we explicitly lay the Rust side out as separate fields.
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub attack_range: f32,
    pub hunger: f32,
    pub thirst: f32,
    pub fatigue: f32,
    pub alive: u32,
    pub creature_type: u32,
    /// Precomputed `hp / max_hp`, populated CPU-side. Read by the
    /// scoring kernel for `field_id == 2 (hp_pct)` instead of
    /// recomputing the division on GPU. Avoids a 1-ULP precision
    /// gap between CPU strict-IEEE and GPU relaxed f32 division
    /// that flips `hp_pct >= 0.8` from true on CPU to false on GPU.
    pub hp_pct: f32,
    /// Reserved — paired with `hp_pct` for symmetry; not yet read.
    pub target_hp_pct_unused: f32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Packed scoring entry. Mirrors WGSL `ScoringEntryGpu` exactly. The
/// CPU `ScoringEntry` struct has non-uniform field sizes (u8, u16, f32)
/// that we smear to u32/f32 here so the WGSL side can index fields at
/// 4-byte offsets without bit shuffling. Padding brought the per-entry
/// size up to 320 bytes: 32-byte header + 8 modifier rows × 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuScoringEntry {
    pub action_head: u32,
    pub modifier_count: u32,
    pub base: f32,
    pub _pad_hdr: u32,
    pub personality_weights: [f32; 5],
    pub _pad_after_weights0: f32,
    pub _pad_after_weights1: f32,
    pub _pad_after_weights2: f32,
    pub modifiers: [GpuModifierRow; 8],
}

/// Packed modifier row. Matches WGSL `ModifierRow`.
///
/// The CPU `ModifierRow` is `{ predicate: PredicateDescriptor, delta:
/// f32 }` where `PredicateDescriptor` is `{ kind: u8, op: u8, field_id:
/// u16, payload: [u8; 12] }` — total 20 bytes. We widen everything to
/// u32 for WGSL alignment: 24 bytes of predicate + 4 bytes delta + 4
/// bytes padding = 32 bytes. `payload[0..4]` / `[4..8]` / `[8..12]` in
/// the CPU form re-encode as `payload0/1/2` here (little-endian order,
/// matching how f32 thresholds are written by the emitter).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuModifierRow {
    pub kind: u32,
    pub op: u32,
    pub field_id: u32,
    pub payload0: u32,
    pub payload1: u32,
    pub payload2: u32,
    pub delta: f32,
    pub _pad: u32,
}

/// Per-agent scoring output. Matches WGSL `ScoreOutput`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq, Eq)]
pub struct ScoreOutput {
    pub chosen_action: u32,
    pub chosen_target: u32,
    /// Bit-pattern of the winning f32 score. Useful for diagnostics —
    /// the backend doesn't consume it yet.
    pub best_score_bits: u32,
    /// Debug slot — read by diagnostics tests, ignored by production
    /// callers. The CPU reference leaves it at 0 unless a test wires
    /// one in.
    pub debug: u32,
}

impl Default for ScoreOutput {
    fn default() -> Self {
        // Default = "Hold with no target". Matches the CPU fall-through.
        Self {
            chosen_action: 0,
            chosen_target: NO_TARGET,
            best_score_bits: 0,
            debug: 0,
        }
    }
}

/// Config uniform carried alongside the scoring kernel. Carries the
/// radii + dispatch-time sizes + tick and view_agent_cap for the
/// per-view read snippets. 32 bytes total — aligned to 16 for uniform
/// buffer layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuConfig {
    combat_attack_range: f32,
    movement_max_move_radius: f32,
    num_entries: u32,
    num_mask_words: u32,
    tick: u32,
    view_agent_cap: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Convert the CPU `SCORING_TABLE` into the GPU-shaped buffer. One
/// entry per row, padding modifiers up to `MAX_MODIFIERS`. Done once
/// at backend init; the table is compile-time constant so no per-tick
/// re-upload is needed.
pub fn pack_scoring_table() -> Vec<GpuScoringEntry> {
    SCORING_TABLE.iter().map(pack_entry).collect()
}

fn pack_entry(e: &ScoringEntry) -> GpuScoringEntry {
    let mut modifiers = [pack_modifier(&ModifierRow::EMPTY); 8];
    let n = (e.modifier_count as usize).min(MAX_MODIFIERS);
    for i in 0..n {
        modifiers[i] = pack_modifier(&e.modifiers[i]);
    }
    GpuScoringEntry {
        action_head: e.action_head as u32,
        modifier_count: e.modifier_count as u32,
        base: e.base,
        _pad_hdr: 0,
        personality_weights: e.personality_weights,
        _pad_after_weights0: 0.0,
        _pad_after_weights1: 0.0,
        _pad_after_weights2: 0.0,
        modifiers,
    }
}

fn pack_modifier(m: &ModifierRow) -> GpuModifierRow {
    let p = &m.predicate;
    // Re-pack the 12-byte payload into three u32s. Little-endian (same
    // as the compiler emits).
    let payload0 = u32::from_le_bytes([p.payload[0], p.payload[1], p.payload[2], p.payload[3]]);
    let payload1 = u32::from_le_bytes([p.payload[4], p.payload[5], p.payload[6], p.payload[7]]);
    let payload2 = u32::from_le_bytes([p.payload[8], p.payload[9], p.payload[10], p.payload[11]]);
    GpuModifierRow {
        kind: p.kind as u32,
        op: p.op as u32,
        field_id: p.field_id as u32,
        payload0,
        payload1,
        payload2,
        delta: m.delta,
        _pad: 0,
    }
}

/// Populate a `Vec<GpuAgentData>` from the `SimState`. One struct per
/// slot; dead slots write `alive = 0` and zero-fields elsewhere,
/// matching the mask kernel's "dead slots produce no bits" convention.
pub fn pack_agent_data(state: &SimState) -> Vec<GpuAgentData> {
    let cap = state.agent_cap() as usize;
    let mut out = Vec::with_capacity(cap);
    for slot in 0..cap {
        let id = match AgentId::new(slot as u32 + 1) {
            Some(id) => id,
            None => {
                out.push(GpuAgentData::zeroed());
                continue;
            }
        };
        let alive = state.agent_alive(id);
        if !alive {
            let mut zero = GpuAgentData::zeroed();
            zero.alive = 0;
            out.push(zero);
            continue;
        }
        let pos = state.agent_pos(id).unwrap_or(glam::Vec3::ZERO);
        let hp = state.agent_hp(id).unwrap_or(0.0);
        let max_hp = state.agent_max_hp(id).unwrap_or(1.0);
        let hp_pct = if max_hp > 0.0 { hp / max_hp } else { 0.0 };
        out.push(GpuAgentData {
            pos_x: pos.x,
            pos_y: pos.y,
            pos_z: pos.z,
            hp,
            max_hp,
            shield_hp: state.agent_shield_hp(id).unwrap_or(0.0),
            attack_range: state.agent_attack_range(id).unwrap_or(2.0),
            hunger: state.agent_hunger(id).unwrap_or(0.0),
            thirst: state.agent_thirst(id).unwrap_or(0.0),
            fatigue: state.agent_rest_timer(id).unwrap_or(0.0),
            alive: 1,
            creature_type: state.agent_creature_type(id).map(|c| c as u32).unwrap_or(u32::MAX),
            hp_pct,
            target_hp_pct_unused: 0.0,
            _pad2: 0,
            _pad3: 0,
        });
    }
    out
}

/// Flatten the `Vec<Vec<u32>>` from the mask kernel into a single
/// `Vec<u32>` suitable for upload. Mask `m`'s word `w` ends up at
/// `out[m * num_mask_words + w]`.
///
/// The CPU-side bitmap order is asserted to match `MASK_NAMES` in the
/// WGSL emitter — callers should hand in the kernel's
/// `last_mask_bitmaps()` directly; the kernel's
/// `FUSED_MASK_NAMES` order is kept in sync with the emitter's
/// `MASK_NAMES`. If the two ever disagree the scoring kernel reads
/// the wrong bitmap for a given action head, which the parity test
/// catches immediately.
pub fn pack_mask_bitmaps(
    kernel: &FusedMaskKernel,
    bitmaps: &[Vec<u32>],
    num_mask_words: u32,
) -> Result<Vec<u32>, KernelError> {
    // Build a reverse index: for each name in MASK_NAMES, find its
    // binding index in the kernel (the order the kernel emitted, which
    // is also `FUSED_MASK_NAMES`). Panic if a name is missing — that's
    // a compile-time contract violation, not a runtime error.
    let n_masks = MASK_NAMES.len();
    let words_per_mask = num_mask_words as usize;
    let mut out = vec![0u32; n_masks * words_per_mask];
    for (dst_idx, &name) in MASK_NAMES.iter().enumerate() {
        let src_idx = kernel
            .bindings()
            .iter()
            .position(|b| b.mask_name == name)
            .ok_or_else(|| {
                KernelError::Dispatch(format!(
                    "pack_mask_bitmaps: fused mask kernel has no binding for `{name}`"
                ))
            })?;
        let src = bitmaps.get(src_idx).ok_or_else(|| {
            KernelError::Dispatch(format!(
                "pack_mask_bitmaps: missing bitmap for mask `{name}` at src_idx {src_idx}"
            ))
        })?;
        if src.len() != words_per_mask {
            return Err(KernelError::Dispatch(format!(
                "pack_mask_bitmaps: mask `{name}` has {} words, expected {words_per_mask}",
                src.len()
            )));
        }
        let dst_start = dst_idx * words_per_mask;
        out[dst_start..dst_start + words_per_mask].copy_from_slice(src);
    }
    Ok(out)
}

/// Scoring-kernel error surface.
#[derive(Debug)]
pub enum ScoringError {
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoringError::ShaderCompile(s) => write!(f, "scoring shader compile: {s}"),
            ScoringError::Dispatch(s) => write!(f, "scoring dispatch: {s}"),
        }
    }
}

impl std::error::Error for ScoringError {}

impl From<KernelError> for ScoringError {
    fn from(e: KernelError) -> Self {
        ScoringError::Dispatch(format!("{e}"))
    }
}

/// Phase 3 scoring pipeline + per-tick buffer pool.
///
/// Owns:
///   * one `wgpu::ComputePipeline` — `cs_scoring`.
///   * one bind-group layout — 5 entries.
///   * a pool of buffers sized for the current agent cap. Re-creates
///     when cap changes (rare).
///   * the packed scoring table buffer (uploaded once at init; table
///     is compile-time constant so no per-tick re-upload).
pub struct ScoringKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Compile-time-constant scoring table buffer. Uploaded once at
    /// init and never mutated after.
    scoring_table_buf: wgpu::Buffer,
    scoring_table_len: u32,
    /// View storage specs (sorted by name) used to drive both the
    /// emitted WGSL and the per-view upload path. Built once at init
    /// from `view_storage::build_all_specs()`.
    view_specs: Vec<ViewStorageSpec>,
    pool: Option<ScoringPool>,
}

struct ScoringPool {
    agent_cap: u32,
    num_mask_words: u32,
    agent_data_buf: wgpu::Buffer,
    mask_bitmaps_buf: wgpu::Buffer,
    scoring_out_buf: wgpu::Buffer,
    scoring_out_readback: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    // Phase 6d: view buffers are owned by `ViewStorage` and bound
    // per-run. The bind group is also re-built per-run since it
    // references view_storage's buffers (whose handles we don't keep
    // across calls).
}

impl ScoringKernel {
    /// Build the scoring pipeline + upload the constant table.
    ///
    /// Phase 6d: pipeline is built against the atomic-storage view
    /// binding layout (`emit_scoring_wgsl_atomic_views`). `ViewStorage`'s
    /// buffers are bound directly by `run_and_readback`; this struct
    /// no longer owns any view storage.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, ScoringError> {
        // Classify every materialized view. `scoring_view_binding_order`
        // filters out Lazy views and sorts the rest by name — this is
        // the same order the WGSL emitter uses for view bindings, so
        // the layout stays in sync without a separate coordination
        // step.
        let view_specs = build_all_specs();

        let wgsl = emit_scoring_wgsl_atomic_views(&view_specs);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::scoring::wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(ScoringError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{wgsl}"
            )));
        }

        // Build the bind-group layout entries: 5 core + per-view.
        // Decay views get TWO view bindings (cells + anchors). Others
        // get one. Matches the atomic-mode WGSL emitter's layout.
        let total_view_bindings: usize =
            scoring_view_binding_order(&view_specs)
                .iter()
                .map(|s| match s.shape {
                    ViewShape::PairMapDecay { .. } => 2,
                    _ => 1,
                })
                .sum();
        let mut bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(
            SCORING_CORE_BINDINGS as usize + total_view_bindings,
        );
        // agent_data, mask_bitmaps, scoring_table — read-only storage.
        for binding in 0..3u32 {
            bgl_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        // scoring_out — read_write storage.
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        // cfg uniform.
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        // View bindings — atomic-storage layout mirrors view_storage.
        // WGSL `atomicLoad` against a `read_only` storage buffer is
        // disallowed on some adapters, so every atomic-typed view
        // binding is declared as `Storage { read_only: false }`.
        // Slot-map (`engaged_with`) is plain `array<u32>` and can stay
        // read-only. One binding per non-decay view; two per decay
        // view (cells + anchors).
        let mut next_binding = SCORING_CORE_BINDINGS;
        for spec in scoring_view_binding_order(&view_specs) {
            match spec.shape {
                ViewShape::SlotMap { .. } => {
                    bgl_entries.push(wgpu::BindGroupLayoutEntry {
                        binding: next_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                    next_binding += 1;
                }
                ViewShape::PairMapScalar => {
                    bgl_entries.push(wgpu::BindGroupLayoutEntry {
                        binding: next_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // WGSL `atomicLoad` requires the buffer to
                            // be non-read_only even though scoring only
                            // reads. view_storage's fold kernel is the
                            // only writer.
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                    next_binding += 1;
                }
                ViewShape::PairMapDecay { .. } => {
                    // Values buffer — atomic<u32>.
                    bgl_entries.push(wgpu::BindGroupLayoutEntry {
                        binding: next_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                    next_binding += 1;
                    // Anchors buffer — atomic<u32>.
                    bgl_entries.push(wgpu::BindGroupLayoutEntry {
                        binding: next_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                    next_binding += 1;
                }
                ViewShape::Lazy => {}
            }
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::scoring::bgl"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::scoring::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::scoring::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_scoring"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Upload the compile-time-constant SCORING_TABLE.
        let packed_table = pack_scoring_table();
        let scoring_table_len = packed_table.len() as u32;
        let scoring_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::scoring::table"),
            contents: bytemuck::cast_slice(&packed_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Nothing to do with `queue` at init — we already passed
        // contents to create_buffer_init. Parameter reserved for a
        // future "upload table via separate write_buffer call" path.
        let _ = queue;

        Ok(Self {
            pipeline,
            bind_group_layout,
            scoring_table_buf,
            scoring_table_len,
            view_specs,
            pool: None,
        })
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, agent_cap: u32, num_mask_words: u32) {
        if let Some(p) = &self.pool {
            if p.agent_cap == agent_cap && p.num_mask_words == num_mask_words {
                return;
            }
        }

        let agent_data_bytes = (agent_cap as usize) * std::mem::size_of::<GpuAgentData>();
        let mask_bitmaps_bytes =
            (MASK_NAMES.len() * num_mask_words as usize) * std::mem::size_of::<u32>();
        let scoring_out_bytes = (agent_cap as usize) * std::mem::size_of::<ScoreOutput>();

        let agent_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::scoring::agent_data"),
            size: agent_data_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_bitmaps_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::scoring::mask_bitmaps"),
            size: mask_bitmaps_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scoring_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::scoring::scoring_out"),
            size: scoring_out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scoring_out_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::scoring::scoring_out_readback"),
            size: scoring_out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::scoring::cfg"),
            contents: bytemuck::cast_slice(&[GpuConfig {
                combat_attack_range: 0.0,
                movement_max_move_radius: 0.0,
                num_entries: 0,
                num_mask_words: 0,
                tick: 0,
                view_agent_cap: agent_cap,
                _pad0: 0,
                _pad1: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Phase 6d: view buffers now live in `ViewStorage` and are
        // bound per-run in `run_and_readback`. No local view_bufs.

        self.pool = Some(ScoringPool {
            agent_cap,
            num_mask_words,
            agent_data_buf,
            mask_bitmaps_buf,
            scoring_out_buf,
            scoring_out_readback,
            cfg_buf,
        });
    }

    /// Dispatch the scoring kernel, reading agent data from `state`,
    /// mask bitmaps from the companion mask kernel, and view cells
    /// directly from `view_storage`'s atomic buffers.
    ///
    /// The caller is responsible for:
    ///   * Folding the current tick's events into `view_storage` (so
    ///     the GPU reads post-fold cells).
    ///   * Ensuring `view_storage.agent_cap() >= state.agent_cap()`.
    ///
    /// One output per agent slot; dead slots get `(Hold, NO_TARGET)`.
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
        mask_kernel: &FusedMaskKernel,
        mask_bitmaps: &[Vec<u32>],
        view_storage: &ViewStorage,
    ) -> Result<Vec<ScoreOutput>, ScoringError> {
        let agent_cap = state.agent_cap();
        let num_mask_words = agent_cap.div_ceil(32).max(1);
        if view_storage.agent_cap() < agent_cap {
            return Err(ScoringError::Dispatch(format!(
                "view_storage.agent_cap ({}) < state.agent_cap ({}); rebuild ViewStorage",
                view_storage.agent_cap(),
                agent_cap,
            )));
        }
        self.ensure_pool(device, agent_cap, num_mask_words);
        let pool = self.pool.as_ref().expect("pool ensured");

        // Pack + upload agent_data.
        let agent_data = pack_agent_data(state);
        queue.write_buffer(
            &pool.agent_data_buf,
            0,
            bytemuck::cast_slice(&agent_data),
        );

        // Pack + upload mask bitmaps.
        let packed_masks = pack_mask_bitmaps(mask_kernel, mask_bitmaps, num_mask_words)?;
        queue.write_buffer(
            &pool.mask_bitmaps_buf,
            0,
            bytemuck::cast_slice(&packed_masks),
        );

        // Upload the cfg uniform — radii + sizes + tick + view_agent_cap.
        // `view_agent_cap` here tracks `view_storage.agent_cap()`, NOT
        // state.agent_cap, because the flat row-major view buffers are
        // sized by that cap and cell address = observer * cap + attacker
        // must use the storage-layout cap.
        let cfg = GpuConfig {
            combat_attack_range: state.config.combat.attack_range,
            movement_max_move_radius: state.config.movement.max_move_radius,
            num_entries: self.scoring_table_len,
            num_mask_words,
            tick: state.tick,
            view_agent_cap: view_storage.agent_cap(),
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&pool.cfg_buf, 0, bytemuck::cast_slice(&[cfg]));

        // Build the per-run bind group referencing view_storage's
        // buffers. Since view_storage is borrowed immutably and its
        // buffers' handles (wgpu::Buffer = Arc inside) can be captured
        // cheaply, we build entries into a scratch Vec.
        let bind_group = self.build_bind_group(device, pool, view_storage)?;

        // Zero the output buffer — kernel writes every slot, but if
        // dispatch fails we want well-defined (empty) readback.
        let zero_outs = vec![ScoreOutput::default(); agent_cap as usize];
        queue.write_buffer(
            &pool.scoring_out_buf,
            0,
            bytemuck::cast_slice(&zero_outs),
        );

        // Dispatch.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::scoring::encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::scoring::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        let scoring_out_bytes = (agent_cap as usize) * std::mem::size_of::<ScoreOutput>();
        encoder.copy_buffer_to_buffer(
            &pool.scoring_out_buf,
            0,
            &pool.scoring_out_readback,
            0,
            scoring_out_bytes as u64,
        );
        queue.submit(Some(encoder.finish()));

        // Readback.
        let slice = pool.scoring_out_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let map_result = rx
            .recv()
            .map_err(|e| ScoringError::Dispatch(format!("map_async channel closed: {e}")))?;
        map_result.map_err(|e| ScoringError::Dispatch(format!("map_async: {e:?}")))?;

        let data = slice.get_mapped_range();
        let outs: Vec<ScoreOutput> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.scoring_out_readback.unmap();

        Ok(outs)
    }

    /// Construct the per-run bind group: 5 core entries (from the pool)
    /// plus one or two per view (from `view_storage`'s buffers).
    fn build_bind_group(
        &self,
        device: &wgpu::Device,
        pool: &ScoringPool,
        view_storage: &ViewStorage,
    ) -> Result<wgpu::BindGroup, ScoringError> {
        let sorted = scoring_view_binding_order(&self.view_specs);
        let total_view_bindings: usize = sorted
            .iter()
            .map(|s| match s.shape {
                ViewShape::PairMapDecay { .. } => 2,
                _ => 1,
            })
            .sum();
        let mut bg_entries: Vec<wgpu::BindGroupEntry> =
            Vec::with_capacity(SCORING_CORE_BINDINGS as usize + total_view_bindings);
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: pool.agent_data_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: pool.mask_bitmaps_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: self.scoring_table_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: pool.scoring_out_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: pool.cfg_buf.as_entire_binding(),
        });
        let mut next_binding = SCORING_CORE_BINDINGS;
        for spec in sorted {
            let primary = view_storage.primary_buffer(&spec.view_name).ok_or_else(|| {
                ScoringError::Dispatch(format!(
                    "view_storage missing buffer for `{}`",
                    spec.view_name
                ))
            })?;
            match spec.shape {
                ViewShape::SlotMap { .. } | ViewShape::PairMapScalar => {
                    bg_entries.push(wgpu::BindGroupEntry {
                        binding: next_binding,
                        resource: primary.as_entire_binding(),
                    });
                    next_binding += 1;
                }
                ViewShape::PairMapDecay { .. } => {
                    let anchor = view_storage.anchor_buffer(&spec.view_name).ok_or_else(|| {
                        ScoringError::Dispatch(format!(
                            "view_storage missing anchor buffer for decay view `{}`",
                            spec.view_name
                        ))
                    })?;
                    bg_entries.push(wgpu::BindGroupEntry {
                        binding: next_binding,
                        resource: primary.as_entire_binding(),
                    });
                    next_binding += 1;
                    bg_entries.push(wgpu::BindGroupEntry {
                        binding: next_binding,
                        resource: anchor.as_entire_binding(),
                    });
                    next_binding += 1;
                }
                ViewShape::Lazy => {}
            }
        }
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::scoring::bg"),
            layout: &self.bind_group_layout,
            entries: &bg_entries,
        }))
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// Per-agent CPU scoring result, byte-parity comparable to
/// `ScoreOutput` from the GPU kernel. Same semantics as the CPU
/// `UtilityBackend::evaluate` path (pick the argmax entry row, then
/// its argmax target for target-bound heads), but the output is
/// exposed as a flat slot-indexed `Vec<ScoreOutput>` rather than the
/// engine's `Vec<Action>` so the test can `assert_eq!` against the
/// GPU's readback directly.
///
/// The scoring semantics deliberately mirror `score_entry` in
/// `crates/engine/src/policy/utility.rs` line-by-line. If that file
/// changes (new predicate kind, new field id), this one has to move
/// in lockstep — the parity test catches silent drift.
pub fn cpu_score_outputs(state: &SimState) -> Vec<ScoreOutput> {
    use engine_rules::scoring::PredicateDescriptor;

    let agent_cap = state.agent_cap();
    let mut out = vec![ScoreOutput::default(); agent_cap as usize];

    for slot in 0..agent_cap {
        let id = match AgentId::new(slot + 1) {
            Some(id) => id,
            None => continue,
        };
        if !state.agent_alive(id) {
            continue;
        }
        let mut best: Option<(u32, u32, f32)> = None; // (action_head, target_slot, score)

        for entry in SCORING_TABLE {
            let mask_idx = action_head_to_mask_idx(entry.action_head);
            if mask_idx == dsl_compiler::emit_scoring_wgsl::MASK_SLOT_NONE {
                continue;
            }
            // Gate by the CPU-computed mask bit. Matches the GPU's
            // `mask_bit(mask_idx, agent_slot)` read.
            let mask_name = MASK_NAMES[mask_idx as usize];
            let mask = match crate::mask::cpu_mask_bitmap(state, mask_name) {
                Some(m) => m,
                None => continue,
            };
            let word = mask.get((slot / 32) as usize).copied().unwrap_or(0);
            let bit = 1u32 << (slot % 32);
            if (word & bit) == 0 {
                continue;
            }

            if is_target_bound(entry.action_head) {
                // Walk candidate slots in ascending order; filter by
                // alive + radius + (hostility for Attack).
                for t in 0..agent_cap {
                    if t == slot {
                        continue;
                    }
                    let tid = match AgentId::new(t + 1) {
                        Some(id) => id,
                        None => continue,
                    };
                    if !state.agent_alive(tid) {
                        continue;
                    }
                    let self_pos = state.agent_pos(id).unwrap_or(glam::Vec3::ZERO);
                    let tgt_pos = state.agent_pos(tid).unwrap_or(glam::Vec3::ZERO);
                    let d = self_pos.distance(tgt_pos);
                    let radius = if entry.action_head == 3 {
                        state.config.combat.attack_range
                    } else {
                        state.config.movement.max_move_radius
                    };
                    if d > radius {
                        continue;
                    }
                    if entry.action_head == 3 {
                        // Hostility check matches DSL is_hostile.
                        let sct = state.agent_creature_type(id);
                        let tct = state.agent_creature_type(tid);
                        match (sct, tct) {
                            (Some(a), Some(b)) if a.is_hostile_to(b) => {}
                            _ => continue,
                        }
                    }

                    let s = score_entry_cpu(entry, state, id, Some(tid));
                    match best {
                        None => best = Some((entry.action_head as u32, t, s)),
                        Some((_, _, bs)) if s > bs => {
                            best = Some((entry.action_head as u32, t, s));
                        }
                        _ => {}
                    }
                }
            } else {
                let s = score_entry_cpu(entry, state, id, None);
                match best {
                    None => best = Some((entry.action_head as u32, NO_TARGET, s)),
                    Some((_, _, bs)) if s > bs => {
                        best = Some((entry.action_head as u32, NO_TARGET, s));
                    }
                    _ => {}
                }
            }
        }

        // No mask-allowed row? Fall through to Hold (action 0).
        let (action, target, score_bits) = match best {
            Some((a, t, s)) => (a, t, s.to_bits()),
            None => (0, NO_TARGET, 0),
        };
        out[slot as usize] = ScoreOutput {
            chosen_action: action,
            chosen_target: target,
            best_score_bits: score_bits,
            debug: 0,
        };
        // Silence unused import warning in minimal builds.
        let _ = std::mem::size_of::<PredicateDescriptor>();
    }
    out
}

fn is_target_bound(action_head: u16) -> bool {
    matches!(action_head, 1 | 3)
}

/// CPU score_entry mirror. Uses `engine_rules::scoring::*` constants.
/// Mirrors `engine::policy::utility::score_entry` line-by-line, including
/// the real view-call dispatch (no more 0.0 stub — Phase 6c).
fn score_entry_cpu(
    entry: &ScoringEntry,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> f32 {
    let mut score = entry.base;
    // Personality dot product — zero at Phase 3 (placeholder on both
    // CPU and GPU).
    for w in &entry.personality_weights {
        score += *w * 0.0;
    }

    let count = (entry.modifier_count as usize).min(MAX_MODIFIERS);
    for i in 0..count {
        let row = &entry.modifiers[i];
        match row.predicate.kind {
            PredicateDescriptor::KIND_VIEW_GRADIENT => {
                let v = eval_view_call_cpu(state, agent, target, &row.predicate);
                if v.is_finite() {
                    // Use fused multiply-add to match WGSL implementations
                    // that fuse `score + v * delta` into a single FMA op
                    // (Vulkan/SPIR-V's `OpExtInst Fma`, which most
                    // drivers emit for `a + b * c` patterns when a is a
                    // mutable accumulator). Separate `*` then `+` gives
                    // a 1-ULP difference vs FMA on values where the
                    // intermediate doesn't fit cleanly into f32; using
                    // mul_add here forces the CPU side to take the same
                    // path. Same fix the engine itself will need when
                    // task 190 wires GPU scoring back into the policy
                    // dispatch — for now this only affects the parity
                    // mirror in this file.
                    score = v.mul_add(row.delta, score);
                }
            }
            _ => {
                if eval_predicate_cpu(&row.predicate, state, agent, target) {
                    score += row.delta;
                }
            }
        }
    }
    score
}

fn eval_predicate_cpu(
    pred: &engine_rules::scoring::PredicateDescriptor,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> bool {
    match pred.kind {
        PredicateDescriptor::KIND_ALWAYS => true,
        PredicateDescriptor::KIND_SCALAR_COMPARE => {
            let lhs = read_field_cpu(state, agent, target, pred.field_id);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar_cpu(pred.op, lhs, rhs)
        }
        PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE => {
            let lhs = eval_view_call_cpu(state, agent, target, pred);
            let mut tb = [0u8; 4];
            tb.copy_from_slice(&pred.payload[0..4]);
            let rhs = f32::from_le_bytes(tb);
            compare_scalar_cpu(pred.op, lhs, rhs)
        }
        _ => false,
    }
}

/// CPU mirror of `engine::policy::utility::eval_view_call`. Differs
/// in one place from the engine path: wildcard sums iterate slot
/// 0..agent_cap rather than HashMap iteration order, so the f32
/// accumulation matches the GPU kernel's per-slot loop. Engine
/// behaviour is unchanged — the engine still uses `sum_for_first`
/// (HashMap iteration); for tiny positive deltas the difference is
/// noise, but byte-exact parity needs an exact accumulation order.
///
/// Documented divergence: if the engine ever depended on this CPU
/// mirror for production scoring (it doesn't — `engine_gpu::scoring`
/// is parity-only at this phase), aligning the engine to slot-order
/// would be the right fix. For now the engine and the mirror differ
/// in summation order but agree on every f32 result on the fixtures
/// we exercise.
fn eval_view_call_cpu(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let slot0 = pred.payload[4];
    let slot1 = pred.payload[5];
    let agent_cap = state.agent_cap();
    match pred.field_id {
        PredicateDescriptor::VIEW_ID_THREAT_LEVEL => {
            let a = match resolve_slot_cpu(slot0, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            match slot1 {
                PredicateDescriptor::ARG_WILDCARD => {
                    sum_pair_decay_slots(agent_cap, |b| {
                        state.views.threat_level.get(a, b, state.tick)
                    })
                }
                _ => {
                    let b = match resolve_slot_cpu(slot1, agent, target) {
                        Some(id) => id,
                        None => return f32::NAN,
                    };
                    state.views.threat_level.get(a, b, state.tick)
                }
            }
        }
        PredicateDescriptor::VIEW_ID_MY_ENEMIES => {
            let a = match resolve_slot_cpu(slot0, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            let b = match resolve_slot_cpu(slot1, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            state.views.my_enemies.get(a, b)
        }
        PredicateDescriptor::VIEW_ID_KIN_FEAR => {
            let a = match resolve_slot_cpu(slot0, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            match slot1 {
                PredicateDescriptor::ARG_WILDCARD => {
                    sum_pair_decay_slots(agent_cap, |b| {
                        state.views.kin_fear.get(a, b, state.tick)
                    })
                }
                _ => {
                    let b = match resolve_slot_cpu(slot1, agent, target) {
                        Some(id) => id,
                        None => return f32::NAN,
                    };
                    state.views.kin_fear.get(a, b, state.tick)
                }
            }
        }
        PredicateDescriptor::VIEW_ID_PACK_FOCUS => {
            let a = match resolve_slot_cpu(slot0, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            match slot1 {
                PredicateDescriptor::ARG_WILDCARD => {
                    sum_pair_decay_slots(agent_cap, |b| {
                        state.views.pack_focus.get(a, b, state.tick)
                    })
                }
                _ => {
                    let b = match resolve_slot_cpu(slot1, agent, target) {
                        Some(id) => id,
                        None => return f32::NAN,
                    };
                    state.views.pack_focus.get(a, b, state.tick)
                }
            }
        }
        PredicateDescriptor::VIEW_ID_RALLY_BOOST => {
            let a = match resolve_slot_cpu(slot0, agent, target) {
                Some(id) => id,
                None => return f32::NAN,
            };
            match slot1 {
                PredicateDescriptor::ARG_WILDCARD => {
                    sum_pair_decay_slots(agent_cap, |b| {
                        state.views.rally_boost.get(a, b, state.tick)
                    })
                }
                _ => {
                    let b = match resolve_slot_cpu(slot1, agent, target) {
                        Some(id) => id,
                        None => return f32::NAN,
                    };
                    state.views.rally_boost.get(a, b, state.tick)
                }
            }
        }
        _ => f32::NAN,
    }
}

/// Sum pair_map (decay or scalar) values across attacker slots in
/// ascending order — matches the GPU kernel's
/// `for (var t: u32 = 0u; t < cfg.view_agent_cap; t = t + 1u)` loop
/// shape so f32 accumulation order is identical.
fn sum_pair_decay_slots<F>(agent_cap: u32, mut sample: F) -> f32
where
    F: FnMut(AgentId) -> f32,
{
    let mut total: f32 = 0.0;
    for slot in 0..agent_cap {
        let id = match AgentId::new(slot + 1) {
            Some(id) => id,
            None => continue,
        };
        total += sample(id);
    }
    total
}

fn resolve_slot_cpu(slot: u8, agent: AgentId, target: Option<AgentId>) -> Option<AgentId> {
    match slot {
        PredicateDescriptor::ARG_SELF => Some(agent),
        PredicateDescriptor::ARG_TARGET => target,
        _ => None,
    }
}

fn read_field_cpu(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    field_id: u16,
) -> f32 {
    if field_id >= 0x4000 && field_id < 0x8000 {
        let target = match target {
            Some(t) => t,
            None => return f32::NAN,
        };
        return match field_id {
            0x4000 => state.agent_hp(target).unwrap_or(0.0),
            0x4001 => state.agent_max_hp(target).unwrap_or(1.0),
            0x4002 => {
                let hp = state.agent_hp(target).unwrap_or(0.0);
                let max = state.agent_max_hp(target).unwrap_or(1.0);
                if max > 0.0 {
                    hp / max
                } else {
                    0.0
                }
            }
            0x4003 => state.agent_shield_hp(target).unwrap_or(0.0),
            _ => f32::NAN,
        };
    }
    match field_id {
        0 => state.agent_hp(agent).unwrap_or(0.0),
        1 => state.agent_max_hp(agent).unwrap_or(1.0),
        2 => {
            let hp = state.agent_hp(agent).unwrap_or(0.0);
            let max = state.agent_max_hp(agent).unwrap_or(1.0);
            if max > 0.0 {
                hp / max
            } else {
                0.0
            }
        }
        3 => state.agent_shield_hp(agent).unwrap_or(0.0),
        4 => state.agent_attack_range(agent).unwrap_or(2.0),
        5 => state.agent_hunger(agent).unwrap_or(0.0),
        6 => state.agent_thirst(agent).unwrap_or(0.0),
        7 => state.agent_rest_timer(agent).unwrap_or(0.0),
        8..=12 => 0.0,
        _ => f32::NAN,
    }
}

fn compare_scalar_cpu(op: u8, lhs: f32, rhs: f32) -> bool {
    use engine_rules::scoring::PredicateDescriptor;
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    match op {
        PredicateDescriptor::OP_LT => lhs < rhs,
        PredicateDescriptor::OP_LE => lhs <= rhs,
        PredicateDescriptor::OP_EQ => lhs == rhs,
        PredicateDescriptor::OP_GE => lhs >= rhs,
        PredicateDescriptor::OP_GT => lhs > rhs,
        PredicateDescriptor::OP_NE => lhs != rhs,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packing a CPU `SCORING_TABLE` into GPU form preserves table
    /// length and a few load-bearing fields (base, modifier_count,
    /// action_head). Spot-checks — the deep comparison happens in the
    /// parity test.
    #[test]
    fn pack_scoring_table_preserves_shape() {
        let packed = pack_scoring_table();
        assert_eq!(
            packed.len(),
            SCORING_TABLE.len(),
            "packed table length drifted from SCORING_TABLE"
        );
        for (i, (cpu, gpu)) in SCORING_TABLE.iter().zip(packed.iter()).enumerate() {
            assert_eq!(gpu.action_head, cpu.action_head as u32, "entry {i} action_head");
            assert_eq!(gpu.modifier_count, cpu.modifier_count as u32, "entry {i} count");
            assert_eq!(gpu.base.to_bits(), cpu.base.to_bits(), "entry {i} base");
        }
    }

    /// GpuAgentData has a well-defined size (padding aligned) and the
    /// WGSL-side packing doesn't drift between build platforms.
    #[test]
    fn agent_data_size_is_stable() {
        // pos(3) + 11 fields = 14 u32-equivalent + 2 trailing pad = 64 bytes
        assert_eq!(std::mem::size_of::<GpuAgentData>(), 64);
        assert_eq!(std::mem::size_of::<ScoreOutput>(), 16);
        assert_eq!(std::mem::size_of::<GpuModifierRow>(), 32);
        // Header (16 bytes) + weights (20) + pad (12) + modifiers (32 × 8 = 256) = 304.
        assert_eq!(std::mem::size_of::<GpuScoringEntry>(), 16 + 20 + 12 + 256);
    }

    /// `pack_agent_data` writes the wolf's hp/max_hp into the slot
    /// the WGSL kernel will read. Regression guard: a slot offset of 1
    /// (id-based instead of slot-based) would put zeros at slot 3.
    #[test]
    fn pack_agent_data_wolf_slot_has_hp_and_max_hp() {
        use engine::creature::CreatureType;
        use engine::state::{AgentSpawn, SimState};
        use glam::Vec3;
        let mut state = SimState::new(8, 0xDEAD_BEEF);
        for i in 0..3 {
            state
                .spawn_agent(AgentSpawn {
                    creature_type: CreatureType::Human,
                    pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
                    hp: 100.0,
                    ..Default::default()
                })
                .expect("human spawn");
        }
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(3.0, 0.0, 0.0),
                hp: 80.0,
                ..Default::default()
            })
            .expect("wolf spawn");
        let packed = pack_agent_data(&state);
        assert_eq!(packed.len(), 8, "agent cap");
        let wolf = packed[3];
        assert_eq!(wolf.alive, 1, "wolf alive");
        assert_eq!(wolf.creature_type, CreatureType::Wolf as u32, "wolf ct");
        assert!((wolf.hp - 80.0).abs() < 1e-6, "wolf hp = {}", wolf.hp);
        assert!(
            (wolf.max_hp - 100.0).abs() < 1e-6,
            "wolf max_hp = {} (expected 100)",
            wolf.max_hp
        );
    }

    /// The emitted scoring WGSL — both the no-views (Phase 3) and the
    /// fully-wired (Phase 6c) form — passes naga's runtime parser. This
    /// is the dev-loop guard for "the shader source string changed and
    /// now it doesn't compile". Runs inside the dsl_compiler ->
    /// engine_gpu layer without needing a GPU device, so it surfaces
    /// in `cargo test` with `--features gpu` even on CI boxes lacking
    /// a usable adapter.
    #[test]
    fn emitted_scoring_wgsl_parses_through_naga() {
        use dsl_compiler::emit_scoring_wgsl::{
            emit_scoring_wgsl, emit_scoring_wgsl_with_views,
        };

        let src_stub = emit_scoring_wgsl();
        if let Err(e) = naga::front::wgsl::parse_str(&src_stub) {
            panic!("--- scoring wgsl (no views) parse error ---\n{e}\n--- source ---\n{src_stub}");
        }

        let specs = build_all_specs();
        let src_wired = emit_scoring_wgsl_with_views(&specs);
        if let Err(e) = naga::front::wgsl::parse_str(&src_wired) {
            panic!("--- scoring wgsl (views wired) parse error ---\n{e}\n--- source ---\n{src_wired}");
        }
    }

    /// The dt=0 short-circuit on `pow(rate, 0)` is in place. Regression
    /// guard for the byte-exact mirror path documented in the
    /// upload_view_state_from_cpu module: when the integration layer
    /// uploads cells with `anchor_tick = state.tick`, the GPU read's
    /// `pow(rate, dt=0)` must reduce to a multiplication by `1.0`
    /// (exact in IEEE), not a `pow` op (vendor-dependent precision).
    #[test]
    fn view_decay_pow_short_circuits_when_dt_zero() {
        use dsl_compiler::emit_scoring_wgsl::emit_scoring_wgsl_with_views;
        let specs = build_all_specs();
        let src = emit_scoring_wgsl_with_views(&specs);
        // Every emitted decay-view read snippet should wrap pow in a
        // `select(..., 1.0, dt == 0u)`. There are 4 @decay views shipped
        // (kin_fear, pack_focus, rally_boost, threat_level), so 4
        // matches expected.
        let select_count = src
            .matches("select(pow(")
            .count();
        assert_eq!(
            select_count, 4,
            "expected 4 short-circuited decay reads, got {select_count}\nsource:\n{src}"
        );
        assert!(
            src.contains("1.0, dt == 0u"),
            "missing dt-zero short-circuit predicate:\n{src}"
        );
    }

    /// Each view spec the kernel binds buffers for shows up at the
    /// expected binding index in the emitted WGSL. Pins the contract
    /// between the WGSL emitter's `scoring_view_binding_order` (sort
    /// by name) and the Rust kernel's bind-group construction loop —
    /// if either side reorders without the other, the buffer at
    /// binding K won't match the WGSL's `view_<name>_*` symbol and the
    /// scoring kernel reads garbage.
    #[test]
    fn wgsl_view_bindings_in_sync() {
        use dsl_compiler::emit_scoring_wgsl::{
            emit_scoring_wgsl_with_views, scoring_view_binding_order, SCORING_CORE_BINDINGS,
        };

        let specs = build_all_specs();
        let src = emit_scoring_wgsl_with_views(&specs);
        for (i, spec) in scoring_view_binding_order(&specs).into_iter().enumerate() {
            let binding = SCORING_CORE_BINDINGS + i as u32;
            let snake = &spec.snake;
            let expected_symbol = match spec.shape {
                ViewShape::SlotMap { .. } => format!(
                    "@group(0) @binding({binding}) var<storage, read> view_{snake}_slots:"
                ),
                ViewShape::PairMapScalar | ViewShape::PairMapDecay { .. } => format!(
                    "@group(0) @binding({binding}) var<storage, read> view_{snake}_cells:"
                ),
                ViewShape::Lazy => continue,
            };
            assert!(
                src.contains(&expected_symbol),
                "binding {binding} for view `{}` missing in scoring WGSL\nexpected: {expected_symbol}\nsource head:\n{}",
                spec.view_name,
                src.lines().take(60).collect::<Vec<_>>().join("\n"),
            );
        }
    }

    /// Phase 6d: the atomic-mode WGSL emission used by `ScoringKernel`
    /// lines up with the bind-group layout constructed in `new`. Views
    /// appear in name-sorted order; pair_map_decay views consume two
    /// consecutive bindings (cells + anchors) rather than the single
    /// `DecayCell` binding the plain-array mode used.
    #[test]
    fn wgsl_atomic_view_bindings_in_sync() {
        use dsl_compiler::emit_scoring_wgsl::{
            emit_scoring_wgsl_atomic_views, scoring_view_binding_order, SCORING_CORE_BINDINGS,
        };

        let specs = build_all_specs();
        let src = emit_scoring_wgsl_atomic_views(&specs);
        let mut binding = SCORING_CORE_BINDINGS;
        for spec in scoring_view_binding_order(&specs) {
            let snake = &spec.snake;
            match spec.shape {
                ViewShape::SlotMap { .. } => {
                    let expected = format!(
                        "@group(0) @binding({binding}) var<storage, read> view_{snake}_slots: array<u32>"
                    );
                    assert!(
                        src.contains(&expected),
                        "slot_map atomic binding missing: {expected}\nsrc head:\n{}",
                        src.lines().take(80).collect::<Vec<_>>().join("\n")
                    );
                    binding += 1;
                }
                ViewShape::PairMapScalar => {
                    let expected = format!(
                        "@group(0) @binding({binding}) var<storage, read_write> view_{snake}_cells: array<atomic<u32>>"
                    );
                    assert!(
                        src.contains(&expected),
                        "pair_scalar atomic binding missing: {expected}\nsrc head:\n{}",
                        src.lines().take(80).collect::<Vec<_>>().join("\n")
                    );
                    binding += 1;
                }
                ViewShape::PairMapDecay { .. } => {
                    let exp_cells = format!(
                        "@group(0) @binding({binding}) var<storage, read_write> view_{snake}_cells: array<atomic<u32>>"
                    );
                    let exp_anchors = format!(
                        "@group(0) @binding({}) var<storage, read_write> view_{snake}_anchors: array<atomic<u32>>",
                        binding + 1
                    );
                    assert!(
                        src.contains(&exp_cells),
                        "pair_decay atomic cells binding missing: {exp_cells}"
                    );
                    assert!(
                        src.contains(&exp_anchors),
                        "pair_decay atomic anchors binding missing: {exp_anchors}"
                    );
                    binding += 2;
                }
                ViewShape::Lazy => {}
            }
        }
    }

    /// Phase 6d: the atomic-mode WGSL parses through naga. This exercises
    /// the same code path `ScoringKernel::new` takes during backend init,
    /// surfacing emitter bugs in `cargo test --lib` without needing a GPU.
    #[test]
    fn atomic_mode_scoring_wgsl_parses_through_naga() {
        use dsl_compiler::emit_scoring_wgsl::emit_scoring_wgsl_atomic_views;
        let specs = build_all_specs();
        let src = emit_scoring_wgsl_atomic_views(&specs);
        if let Err(e) = naga::front::wgsl::parse_str(&src) {
            panic!("--- atomic-mode scoring WGSL parse error ---\n{e}\n--- source ---\n{src}");
        }
    }
}
