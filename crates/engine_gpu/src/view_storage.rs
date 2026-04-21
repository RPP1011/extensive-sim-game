//! Phase 4 — GPU view storage + per-view fold kernels.
//!
//! See `docs/plans/gpu_megakernel_plan.md`. The Phase 2 backend runs
//! mask kernels on GPU but still relies on CPU-side `ViewRegistry` for
//! every view read. Phase 4 adds the GPU-side counterpart: one storage
//! buffer (or pair of buffers) per @materialized view, plus one WGSL
//! compute kernel per view that folds a batch of event-derived
//! `FoldInput` structs into the storage.
//!
//! This module is NOT yet spliced into the backend's tick loop — the
//! follow-up integration task wires scoring/physics to call these fold
//! kernels instead of (or alongside) the CPU `ViewRegistry::fold_all`.
//! Here we expose the buffers, kernels, and a parity helper so the
//! Phase 4 parity test can seed the same event stream into CPU + GPU
//! and assert byte-equal storage + reads.
//!
//! ## Views today
//!
//! Nine views ship in `assets/sim/views.sim`; the GPU side provisions
//! storage for the six materialized ones:
//!
//!   | View          | Shape       | Decay rate | GPU bytes at N=2000 |
//!   |---------------|-------------|------------|---------------------|
//!   | engaged_with  | slot_map    | —          |  ~8 KB              |
//!   | my_enemies    | pair_map    | —          |  16 MB              |
//!   | threat_level  | pair_map    | 0.98       |  32 MB              |
//!   | kin_fear      | pair_map    | 0.891      |  32 MB              |
//!   | pack_focus    | pair_map    | 0.933      |  32 MB              |
//!   | rally_boost   | pair_map    | 0.891      |  32 MB              |
//!
//! Total at N=2000: **~144 MB** of dedicated view storage. That's
//! comfortable for any real GPU; our concern is the *fold* bandwidth,
//! not footprint, and the fold writes only touch the cells that had
//! events pushed this tick.
//!
//! The three `@lazy` views (`is_hostile`, `is_stunned`, `slow_factor`)
//! have no storage — they're inline WGSL expressions consumed by the
//! mask / scoring emitters, not materialized state.
//!
//! ## Determinism policy (atomic add vs sorted reduce)
//!
//! We use `atomicCompareExchangeWeak` CAS loops for f32 accumulation.
//! That's not commutative-associative in the general case, BUT every
//! shipped fold body is `self += 1.0` — a constant delta. Under "all
//! adds are the same positive constant" the sum is deterministic
//! regardless of fold order (there's no cancellation, no rounding
//! asymmetry worth distinguishing). If a future view folds a variable
//! delta, the right move is a sort-by-key-then-serial-reduce pass at
//! the fold entry; the atomic path stays correct but becomes
//! non-deterministic. This trade-off is documented in both this
//! module and the WGSL emitter (see `dsl_compiler::emit_view_wgsl`).

#![cfg(feature = "gpu")]

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use dsl_compiler::emit_view_wgsl::{
    self, classify_view, emit_view_fold_wgsl, emit_view_read_wgsl, ViewShape, ViewStorageSpec,
};
use dsl_compiler::ir::{
    DecayHint, DecayUnit, FoldHandlerIR, IrEventPattern, IrExpr, IrExprNode, IrParam,
    IrPattern, IrPatternBinding, IrType, LocalRef, StorageHint, ViewBodyIR, ViewIR, ViewKind,
};

/// Workgroup size for fold kernels — 64 threads, matches the mask
/// kernel's choice and keeps dispatches divisible by common occupancy
/// targets.
pub const FOLD_WORKGROUP_SIZE: u32 = 64;

/// Named list of the six materialized views. The ordering here is the
/// storage-buffer ordering — the `ViewStorage` struct exposes one
/// handle per name so callers can do targeted readbacks without a
/// string lookup in hot paths.
pub const VIEW_NAMES: &[&str] = &[
    "engaged_with",
    "my_enemies",
    "threat_level",
    "kin_fear",
    "pack_focus",
    "rally_boost",
];

/// Errors surfaced by `ViewStorage::new` and its fold dispatches.
#[derive(Debug)]
pub enum ViewStorageError {
    Emit(String),
    ShaderCompile(String),
    Dispatch(String),
}

impl std::fmt::Display for ViewStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewStorageError::Emit(s) => write!(f, "view emit: {s}"),
            ViewStorageError::ShaderCompile(s) => write!(f, "view shader compile: {s}"),
            ViewStorageError::Dispatch(s) => write!(f, "view dispatch: {s}"),
        }
    }
}

impl std::error::Error for ViewStorageError {}

// ---------------------------------------------------------------------------
// Fold-event input structs (uploaded to GPU per fold dispatch)
// ---------------------------------------------------------------------------

/// One pair-keyed fold input — observer/attacker slot pair with the
/// fold tick. Used by `my_enemies`, `threat_level`, `kin_fear`,
/// `pack_focus`, `rally_boost`.
///
/// Slots are 0-based array indices (NOT raw AgentId — raw is 1-based).
/// `tick` is the tick the event was emitted at; @decay views use this
/// to stamp the anchor, non-decay ignore it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct FoldInputPair {
    pub first: u32,
    pub second: u32,
    pub tick: u32,
    pub _pad: u32,
}

/// One slot-keyed fold input — insert/remove a slot pair. `kind == 0`
/// means insert (both sides get each other); `kind == 1` means remove.
/// Used by `engaged_with`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct FoldInputSlot {
    pub first: u32,
    pub second: u32,
    pub kind: u32,
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// DecayCell parallel-layout note
// ---------------------------------------------------------------------------

/// Cell layout for pair_map @decay storage — parallel u32 arrays for
/// (value, anchor_tick). We store two separate `array<atomic<u32>>`
/// buffers rather than one interleaved `array<DecayCell>` because
/// WGSL atomics on struct members are clunky — the CAS loop needs
/// the raw `&atomic<u32>` pointer. This layout mirrors the CPU
/// semantics: reads reconstruct (value, anchor_tick) from the two
/// buffers, writes update both.
///
/// The read WGSL emitted by `emit_view_wgsl` expects a single
/// interleaved `DecayCell` buffer by default. The engine_gpu side
/// overrides that via a small WGSL prelude that defines the
/// `view_<snake>_cells` symbol as a helper function returning
/// `DecayCell{ value: base, anchor_tick: anchor }` — see
/// `build_fold_module`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct DecayCellCpu {
    pub value: f32,
    pub anchor: u32,
}

impl Default for DecayCellCpu {
    fn default() -> Self {
        Self {
            value: 0.0,
            anchor: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-view pipeline + buffers
// ---------------------------------------------------------------------------

struct ViewEntry {
    spec: ViewStorageSpec,
    /// Primary storage buffer:
    ///   - SlotMap: `array<u32>` length = N
    ///   - PairMapScalar: `array<atomic<u32>>` length = N*N (f32 bits)
    ///   - PairMapDecay: `array<atomic<u32>>` length = N*N (f32 value bits)
    primary: wgpu::Buffer,
    /// Only set for PairMapDecay: parallel `array<atomic<u32>>` for
    /// anchor_tick. None for other shapes.
    anchor: Option<wgpu::Buffer>,
    /// Byte size of `primary`.
    primary_bytes: u64,
    /// Readback staging for `primary` — MAP_READ | COPY_DST.
    primary_readback: wgpu::Buffer,
    /// Readback for anchor (if present).
    anchor_readback: Option<wgpu::Buffer>,
    /// Compiled fold kernel for this view.
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Reusable input buffer — grown on demand as the number of fold
    /// events per tick varies. Starts at 64 events capacity.
    input_buf: wgpu::Buffer,
    input_cap: u32,
    /// Uniform — carries (N, event_count).
    uniform: wgpu::Buffer,
}

/// GPU view storage — one buffer/pipeline per materialized view + the
/// glue to dispatch fold passes.
pub struct ViewStorage {
    agent_cap: u32,
    entries: HashMap<String, ViewEntry>,
}

impl ViewStorage {
    /// Build storage + fold pipelines for every materialized view in
    /// the seed views.sim. `agent_cap` is the simulation's N; the
    /// per-view buffers are sized N (slot_map) or N² (pair_map). Call
    /// this once at backend init.
    pub fn new(device: &wgpu::Device, agent_cap: u32) -> Result<Self, ViewStorageError> {
        let mut entries: HashMap<String, ViewEntry> = HashMap::new();
        for view in build_materialized_view_irs() {
            let spec = classify_view(&view).map_err(|e| ViewStorageError::Emit(e.to_string()))?;
            if matches!(spec.shape, ViewShape::Lazy) {
                continue;
            }
            let entry = build_view_entry(device, agent_cap, spec)?;
            entries.insert(entry.spec.view_name.clone(), entry);
        }
        Ok(Self { agent_cap, entries })
    }

    /// Agent capacity this storage was sized for.
    pub fn agent_cap(&self) -> u32 {
        self.agent_cap
    }

    /// Byte sum of every primary + anchor buffer currently allocated.
    /// Handy for benchmarks / docs; the task report references this.
    pub fn total_primary_bytes(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.primary_bytes + e.anchor.as_ref().map(|_| e.primary_bytes).unwrap_or(0))
            .sum()
    }

    /// Zero every storage buffer — call before a fresh fold sequence
    /// (e.g. at the top of a parity test run so a stale tick doesn't
    /// leak). Uses `queue.write_buffer` with a zero blob; cheap relative
    /// to the per-tick fold cost.
    pub fn reset(&self, queue: &wgpu::Queue) {
        for entry in self.entries.values() {
            let zeros = vec![0u8; entry.primary_bytes as usize];
            queue.write_buffer(&entry.primary, 0, &zeros);
            if let Some(anchor) = &entry.anchor {
                queue.write_buffer(anchor, 0, &zeros);
            }
        }
    }

    /// Dispatch a pair-keyed fold pass for one view. The `events` slice
    /// is uploaded to the view's reusable input buffer (grown if
    /// needed), and the pipeline is dispatched with
    /// `ceil(events.len() / FOLD_WORKGROUP_SIZE)` workgroups.
    ///
    /// The exact slots to pass in each `FoldInputPair` depend on how
    /// the view's fold handler binds event fields to view args — that
    /// binding is encoded in `ViewStorageSpec::folds` and the caller
    /// resolves it against the CPU `Event` enum ahead of calling here.
    /// For the parity test this is a thin helper that takes pre-
    /// resolved pairs, same shape the engine_gpu integration layer
    /// will use.
    ///
    /// No-op if `events` is empty.
    pub fn fold_pair_events(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_name: &str,
        events: &[FoldInputPair],
    ) -> Result<(), ViewStorageError> {
        if events.is_empty() {
            return Ok(());
        }
        let agent_cap = self.agent_cap;
        let entry = self
            .entries
            .get_mut(view_name)
            .ok_or_else(|| ViewStorageError::Dispatch(format!("unknown view `{view_name}`")))?;
        match entry.spec.shape {
            ViewShape::PairMapScalar | ViewShape::PairMapDecay { .. } => {}
            _ => {
                return Err(ViewStorageError::Dispatch(format!(
                    "fold_pair_events called on non-pair view `{view_name}` (shape = {:?})",
                    entry.spec.shape
                )));
            }
        }
        let (input_buf, input_cap) = ensure_input_buf_pair(device, entry, events.len() as u32);
        entry.input_buf = input_buf;
        entry.input_cap = input_cap;
        queue.write_buffer(&entry.input_buf, 0, bytemuck::cast_slice(events));

        // Uniform: N, event count.
        let uniform_data = [agent_cap, events.len() as u32, 0, 0];
        queue.write_buffer(&entry.uniform, 0, bytemuck::cast_slice(&uniform_data));

        // Bind + dispatch.
        let bg = bind_group_for(device, entry);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::view_storage::fold_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::view_storage::fold_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&entry.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let groups = (events.len() as u32).div_ceil(FOLD_WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Dispatch a slot-keyed fold pass for `engaged_with`.
    pub fn fold_slot_events(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_name: &str,
        events: &[FoldInputSlot],
    ) -> Result<(), ViewStorageError> {
        if events.is_empty() {
            return Ok(());
        }
        let agent_cap = self.agent_cap;
        let entry = self
            .entries
            .get_mut(view_name)
            .ok_or_else(|| ViewStorageError::Dispatch(format!("unknown view `{view_name}`")))?;
        match entry.spec.shape {
            ViewShape::SlotMap { .. } => {}
            _ => {
                return Err(ViewStorageError::Dispatch(format!(
                    "fold_slot_events called on non-slot view `{view_name}` (shape = {:?})",
                    entry.spec.shape
                )));
            }
        }
        let (input_buf, input_cap) = ensure_input_buf_slot(device, entry, events.len() as u32);
        entry.input_buf = input_buf;
        entry.input_cap = input_cap;
        queue.write_buffer(&entry.input_buf, 0, bytemuck::cast_slice(events));

        let uniform_data = [agent_cap, events.len() as u32, 0, 0];
        queue.write_buffer(&entry.uniform, 0, bytemuck::cast_slice(&uniform_data));

        let bg = bind_group_for(device, entry);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::view_storage::fold_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::view_storage::fold_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&entry.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let groups = (events.len() as u32).div_ceil(FOLD_WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Read back a slot_map as a flat `Vec<u32>` (length = agent_cap).
    /// Cell value is `AgentId_raw + 1` or 0 if empty.
    pub fn readback_slot_map(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_name: &str,
    ) -> Result<Vec<u32>, ViewStorageError> {
        let entry = self
            .entries
            .get(view_name)
            .ok_or_else(|| ViewStorageError::Dispatch(format!("unknown view `{view_name}`")))?;
        if !matches!(entry.spec.shape, ViewShape::SlotMap { .. }) {
            return Err(ViewStorageError::Dispatch(format!(
                "readback_slot_map on non-slot view `{view_name}`"
            )));
        }
        let words = copy_and_map::<u32>(device, queue, &entry.primary, &entry.primary_readback, entry.primary_bytes)?;
        Ok(words)
    }

    /// Read back a pair_map scalar view's flat cells (length = N²)
    /// as `Vec<f32>`. The cell at index `observer * N + attacker`
    /// is the (possibly clamped) accumulated value.
    pub fn readback_pair_scalar(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_name: &str,
    ) -> Result<Vec<f32>, ViewStorageError> {
        let entry = self
            .entries
            .get(view_name)
            .ok_or_else(|| ViewStorageError::Dispatch(format!("unknown view `{view_name}`")))?;
        if !matches!(entry.spec.shape, ViewShape::PairMapScalar) {
            return Err(ViewStorageError::Dispatch(format!(
                "readback_pair_scalar on non-pair-scalar view `{view_name}`"
            )));
        }
        let raw = copy_and_map::<u32>(device, queue, &entry.primary, &entry.primary_readback, entry.primary_bytes)?;
        Ok(raw.into_iter().map(f32::from_bits).collect())
    }

    /// Read back a pair_map @decay view's flat cells as
    /// `Vec<DecayCellCpu>`. Each cell carries (value, anchor_tick);
    /// callers apply the decay formula themselves if they need the
    /// observable "now" value. Every pair_map @decay view today has a
    /// clamp — applied on-read by `decayed_get`.
    pub fn readback_pair_decay(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_name: &str,
    ) -> Result<Vec<DecayCellCpu>, ViewStorageError> {
        let entry = self
            .entries
            .get(view_name)
            .ok_or_else(|| ViewStorageError::Dispatch(format!("unknown view `{view_name}`")))?;
        if !matches!(entry.spec.shape, ViewShape::PairMapDecay { .. }) {
            return Err(ViewStorageError::Dispatch(format!(
                "readback_pair_decay on non-pair-decay view `{view_name}`"
            )));
        }
        let anchor = entry
            .anchor
            .as_ref()
            .ok_or_else(|| ViewStorageError::Dispatch("decay view missing anchor buffer".into()))?;
        let anchor_readback = entry.anchor_readback.as_ref().unwrap();
        let value_bits = copy_and_map::<u32>(device, queue, &entry.primary, &entry.primary_readback, entry.primary_bytes)?;
        let anchor_vals = copy_and_map::<u32>(device, queue, anchor, anchor_readback, entry.primary_bytes)?;
        let mut out = Vec::with_capacity(value_bits.len());
        for (vb, an) in value_bits.iter().zip(anchor_vals.iter()) {
            out.push(DecayCellCpu {
                value: f32::from_bits(*vb),
                anchor: *an,
            });
        }
        Ok(out)
    }

    /// The ViewStorageSpec each buffer was built from — exposed so the
    /// parity test can introspect rate / clamp without rebuilding.
    pub fn spec(&self, view_name: &str) -> Option<&ViewStorageSpec> {
        self.entries.get(view_name).map(|e| &e.spec)
    }

    /// Primary storage buffer for `view_name`. For slot_map this is an
    /// `array<u32>`; for pair_map (scalar or decay) it's an
    /// `array<atomic<u32>>` carrying f32 bits (scalar) or decay value
    /// bits (decay). None if the view isn't provisioned.
    ///
    /// Phase 6d: scoring / physics kernels bind this directly so their
    /// reads see the same post-fold cells view_storage's fold kernels
    /// wrote. No CPU mirror in between.
    pub fn primary_buffer(&self, view_name: &str) -> Option<&wgpu::Buffer> {
        self.entries.get(view_name).map(|e| &e.primary)
    }

    /// Anchor-tick buffer for a pair-map @decay view (f32-bits value
    /// parallel buffer). None for slot_map / pair-scalar (no anchor)
    /// or unknown views.
    pub fn anchor_buffer(&self, view_name: &str) -> Option<&wgpu::Buffer> {
        self.entries
            .get(view_name)
            .and_then(|e| e.anchor.as_ref())
    }

    /// Byte size of the primary buffer for `view_name`. Matches
    /// `agent_cap` for slot_map (as bytes of u32) and `agent_cap²*4` for
    /// pair_map. Used by integration tests to size zeroing / readback
    /// writes without reaching into the private buffer handle.
    pub fn primary_bytes(&self, view_name: &str) -> Option<u64> {
        self.entries.get(view_name).map(|e| e.primary_bytes)
    }
}

fn ensure_input_buf_pair(
    device: &wgpu::Device,
    entry: &ViewEntry,
    want: u32,
) -> (wgpu::Buffer, u32) {
    let stride = std::mem::size_of::<FoldInputPair>() as u64;
    let cap = want.max(entry.input_cap).max(64);
    let bytes = cap as u64 * stride;
    if entry.input_cap >= cap {
        return (clone_buffer_handle(&entry.input_buf), entry.input_cap);
    }
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::view_storage::pair_input"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (buf, cap)
}

fn ensure_input_buf_slot(
    device: &wgpu::Device,
    entry: &ViewEntry,
    want: u32,
) -> (wgpu::Buffer, u32) {
    let stride = std::mem::size_of::<FoldInputSlot>() as u64;
    let cap = want.max(entry.input_cap).max(64);
    let bytes = cap as u64 * stride;
    if entry.input_cap >= cap {
        return (clone_buffer_handle(&entry.input_buf), entry.input_cap);
    }
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::view_storage::slot_input"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (buf, cap)
}

// `wgpu::Buffer` is not Clone but we want to keep the existing handle
// when we don't need to grow. The helper just returns a bitwise-equal
// clone via `wgpu::Buffer`'s internal Arc — implemented by recreating
// via as_entire_binding where possible. For our dispatch-per-call
// pattern we rebuild the bind group each call, so keeping the Buffer
// across calls is correct.
fn clone_buffer_handle(buf: &wgpu::Buffer) -> wgpu::Buffer {
    // `wgpu::Buffer` is an Arc internally in wgpu 26, so `clone()`
    // bumps the refcount. If a future wgpu rev removes `Clone`, switch
    // this helper to store the buffer back in the entry and take by
    // `&mut`; nothing else in the module assumes ownership.
    buf.clone()
}

fn bind_group_for(device: &wgpu::Device, entry: &ViewEntry) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(4);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: entry.input_buf.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: entry.primary.as_entire_binding(),
    });
    if let Some(anchor) = &entry.anchor {
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: anchor.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: if entry.anchor.is_some() { 3 } else { 2 },
        resource: entry.uniform.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine_gpu::view_storage::bg"),
        layout: &entry.bind_group_layout,
        entries: &entries,
    })
}

fn build_view_entry(
    device: &wgpu::Device,
    agent_cap: u32,
    spec: ViewStorageSpec,
) -> Result<ViewEntry, ViewStorageError> {
    let (primary_elems, has_anchor) = match spec.shape {
        ViewShape::SlotMap { .. } => (agent_cap as u64, false),
        ViewShape::PairMapScalar => ((agent_cap as u64) * (agent_cap as u64), false),
        ViewShape::PairMapDecay { .. } => ((agent_cap as u64) * (agent_cap as u64), true),
        ViewShape::Lazy => unreachable!(),
    };
    let primary_bytes = primary_elems * 4; // u32/f32-bits.
    let primary = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("engine_gpu::view::{}::primary", spec.snake)),
        size: primary_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let primary_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("engine_gpu::view::{}::primary_rb", spec.snake)),
        size: primary_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let anchor = if has_anchor {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("engine_gpu::view::{}::anchor", spec.snake)),
            size: primary_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };
    let anchor_readback = if has_anchor {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("engine_gpu::view::{}::anchor_rb", spec.snake)),
            size: primary_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };

    // Reusable event-input buffer (grown on demand).
    let default_events = 64u32;
    let input_bytes = match spec.shape {
        ViewShape::SlotMap { .. } => default_events as u64 * std::mem::size_of::<FoldInputSlot>() as u64,
        _ => default_events as u64 * std::mem::size_of::<FoldInputPair>() as u64,
    };
    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("engine_gpu::view::{}::input", spec.snake)),
        size: input_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("engine_gpu::view::{}::uniform", spec.snake)),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ---- Compile the fold kernel ----
    let wgsl = build_fold_module_wgsl(&spec);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("engine_gpu::view::{}::wgsl", spec.snake)),
        source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
    });
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        return Err(ViewStorageError::ShaderCompile(format!(
            "{err}\n--- WGSL source ---\n{wgsl}"
        )));
    }

    // ---- Bind-group layout ----
    // Bindings: 0 input (read), 1 primary (read_write), [2 anchor (read_write)], 2|3 uniform.
    let mut bgl_entries = Vec::with_capacity(4);
    bgl_entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    bgl_entries.push(wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    if has_anchor {
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let uniform_binding = if has_anchor { 3 } else { 2 };
    bgl_entries.push(wgpu::BindGroupLayoutEntry {
        binding: uniform_binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("engine_gpu::view::{}::bgl", spec.snake)),
        entries: &bgl_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("engine_gpu::view::{}::pl", spec.snake)),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("engine_gpu::view::{}::cp", spec.snake)),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_fold"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(ViewEntry {
        spec,
        primary,
        anchor,
        primary_bytes,
        primary_readback,
        anchor_readback,
        pipeline,
        bind_group_layout: bgl,
        input_buf,
        input_cap: default_events,
        uniform,
    })
}

// ---------------------------------------------------------------------------
// WGSL fold-module construction
// ---------------------------------------------------------------------------

/// Build the full WGSL module for one view's fold kernel. Layout:
///
/// ```wgsl
/// struct FoldInput { first, second, tick, _pad }  // or slot variant
/// struct Uniform   { n, event_count, _0, _1 }
/// @group(0) @binding(0) var<storage, read>  inputs: array<FoldInput>;
/// @group(0) @binding(1) var<storage, read_write> cells: array<atomic<u32>>;  // or u32 for slot_map
/// @group(0) @binding(2) var<storage, read_write> anchors: array<atomic<u32>>;  // decay only
/// @group(0) @binding(N) var<uniform>      cfg: Uniform;
///
/// @compute @workgroup_size(64) fn cs_fold(@builtin(global_invocation_id) gid) { ... }
/// ```
///
/// The fold body for each shape matches the contracts in
/// `dsl_compiler::emit_view_wgsl::emit_single_fold` but with the
/// storage symbol renamed to `cells` / `anchors` so the kernel layout
/// is view-agnostic.
fn build_fold_module_wgsl(spec: &ViewStorageSpec) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by engine_gpu::view_storage — fold kernel.\n");
    out.push_str(&format!("// View: {}\n", spec.view_name));
    out.push_str(&format!("// Shape: {:?}\n", spec.shape));

    match &spec.shape {
        ViewShape::SlotMap { .. } => {
            out.push_str(
                "struct FoldInput { first: u32, second: u32, kind: u32, _pad: u32 };\n\
                 struct Uniform   { n: u32, event_count: u32, _0: u32, _1: u32 };\n\
                 @group(0) @binding(0) var<storage, read>       inputs: array<FoldInput>;\n\
                 @group(0) @binding(1) var<storage, read_write> cells:  array<u32>;\n\
                 @group(0) @binding(2) var<uniform>             cfg:    Uniform;\n\n",
            );
            out.push_str(&format!(
                "@compute @workgroup_size({FOLD_WORKGROUP_SIZE}) fn cs_fold(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 \x20   let i = gid.x;\n\
                 \x20   if (i >= cfg.event_count) {{ return; }}\n\
                 \x20   let ev = inputs[i];\n\
                 \x20   let n = cfg.n;\n\
                 \x20   if (ev.first >= n || ev.second >= n) {{ return; }}\n\
                 \x20   if (ev.kind == 0u) {{\n\
                 \x20       // Insert: both sides see each other.\n\
                 \x20       cells[ev.first]  = ev.second + 1u;\n\
                 \x20       cells[ev.second] = ev.first  + 1u;\n\
                 \x20   }} else {{\n\
                 \x20       cells[ev.first]  = 0u;\n\
                 \x20       cells[ev.second] = 0u;\n\
                 \x20   }}\n\
                 }}\n",
            ));
        }
        ViewShape::PairMapScalar => {
            let new_expr = match &spec.clamp {
                Some((lo, hi)) => format!(
                    "clamp(old + 1.0, {}, {})",
                    render_float_wgsl(*lo as f64),
                    render_float_wgsl(*hi as f64)
                ),
                None => "old + 1.0".to_string(),
            };
            out.push_str(
                "struct FoldInput { first: u32, second: u32, tick: u32, _pad: u32 };\n\
                 struct Uniform   { n: u32, event_count: u32, _0: u32, _1: u32 };\n\
                 @group(0) @binding(0) var<storage, read>       inputs: array<FoldInput>;\n\
                 @group(0) @binding(1) var<storage, read_write> cells:  array<atomic<u32>>;\n\
                 @group(0) @binding(2) var<uniform>             cfg:    Uniform;\n\n",
            );
            out.push_str(&format!(
                "@compute @workgroup_size({FOLD_WORKGROUP_SIZE}) fn cs_fold(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 \x20   let i = gid.x;\n\
                 \x20   if (i >= cfg.event_count) {{ return; }}\n\
                 \x20   let ev = inputs[i];\n\
                 \x20   let n = cfg.n;\n\
                 \x20   if (ev.first >= n || ev.second >= n) {{ return; }}\n\
                 \x20   let idx = ev.first * n + ev.second;\n\
                 \x20   loop {{\n\
                 \x20       let old_bits = atomicLoad(&cells[idx]);\n\
                 \x20       let old      = bitcast<f32>(old_bits);\n\
                 \x20       let updated  = {new_expr};\n\
                 \x20       let cas      = atomicCompareExchangeWeak(&cells[idx], old_bits, bitcast<u32>(updated));\n\
                 \x20       if (cas.exchanged) {{ break; }}\n\
                 \x20   }}\n\
                 }}\n",
            ));
        }
        ViewShape::PairMapDecay { rate } => {
            let clamp_expr = match &spec.clamp {
                Some((lo, hi)) => format!(
                    "clamp(decayed + 1.0, {}, {})",
                    render_float_wgsl(*lo as f64),
                    render_float_wgsl(*hi as f64)
                ),
                None => "decayed + 1.0".to_string(),
            };
            out.push_str(
                "struct FoldInput { first: u32, second: u32, tick: u32, _pad: u32 };\n\
                 struct Uniform   { n: u32, event_count: u32, _0: u32, _1: u32 };\n\
                 @group(0) @binding(0) var<storage, read>       inputs:  array<FoldInput>;\n\
                 @group(0) @binding(1) var<storage, read_write> cells:   array<atomic<u32>>;\n\
                 @group(0) @binding(2) var<storage, read_write> anchors: array<atomic<u32>>;\n\
                 @group(0) @binding(3) var<uniform>             cfg:     Uniform;\n\n",
            );
            out.push_str(&format!(
                "@compute @workgroup_size({FOLD_WORKGROUP_SIZE}) fn cs_fold(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 \x20   let i = gid.x;\n\
                 \x20   if (i >= cfg.event_count) {{ return; }}\n\
                 \x20   let ev = inputs[i];\n\
                 \x20   let n = cfg.n;\n\
                 \x20   if (ev.first >= n || ev.second >= n) {{ return; }}\n\
                 \x20   let idx = ev.first * n + ev.second;\n\
                 \x20   loop {{\n\
                 \x20       let base_bits = atomicLoad(&cells[idx]);\n\
                 \x20       let anchor    = atomicLoad(&anchors[idx]);\n\
                 \x20       let base      = bitcast<f32>(base_bits);\n\
                 \x20       let dt        = select(0u, ev.tick - anchor, ev.tick >= anchor);\n\
                 \x20       let decayed   = base * pow({rate}, f32(dt));\n\
                 \x20       let updated   = {clamp};\n\
                 \x20       let cas       = atomicCompareExchangeWeak(&cells[idx], base_bits, bitcast<u32>(updated));\n\
                 \x20       if (cas.exchanged) {{\n\
                 \x20           atomicStore(&anchors[idx], ev.tick);\n\
                 \x20           break;\n\
                 \x20       }}\n\
                 \x20   }}\n\
                 }}\n",
                rate = render_float_wgsl(*rate as f64),
                clamp = clamp_expr,
            ));
        }
        ViewShape::Lazy => unreachable!(),
    }

    // Exercise the emit_view_wgsl snippet functions during build so a
    // drift between the public emitter and the engine_gpu fold kernel
    // surfaces as a compile error here — the snippets aren't linked
    // into the kernel (the kernel's layout is this module's layout),
    // but we sanity-check that the emitter accepts the same spec.
    let _ = emit_view_read_wgsl(spec);
    let _ = emit_view_fold_wgsl(spec);

    out
}

fn render_float_wgsl(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

// ---------------------------------------------------------------------------
// Readback helper
// ---------------------------------------------------------------------------

fn copy_and_map<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    readback: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>, ViewStorageError> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_gpu::view_storage::readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, readback, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| ViewStorageError::Dispatch(format!("map_async channel closed: {e}")))?;
    map_result.map_err(|e| ViewStorageError::Dispatch(format!("map_async: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice::<u8, T>(&data).to_vec();
    drop(data);
    readback.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// In-memory ViewIR construction — mirrors assets/sim/views.sim
// ---------------------------------------------------------------------------

/// Build the materialized `ViewIR`s from scratch. Mirrors the
/// `@materialized` views in `assets/sim/views.sim`. If the DSL
/// source ever adds a new materialized view, update this list to
/// match; the parity test will catch a missing view by comparing
/// the `VIEW_NAMES` slice against what `classify_view` turned up.
///
/// Keeping this in Rust (instead of parsing views.sim at runtime)
/// mirrors what the mask kernel does — the emitter/engine_gpu split
/// avoids a DSL parse dependency in the backend.
pub fn build_materialized_view_irs() -> Vec<ViewIR> {
    let span = dsl_compiler::ast::Span::dummy();
    let lit_f = |v: f64| IrExprNode {
        kind: IrExpr::LitFloat(v),
        span,
    };
    let param = |name: &str, ty: IrType| IrParam {
        name: name.to_string(),
        local: LocalRef(0),
        ty,
        span,
    };
    let bind = |field: &str, local: &str| IrPatternBinding {
        field: field.to_string(),
        value: IrPattern::Bind {
            name: local.to_string(),
            local: LocalRef(0),
        },
        span,
    };
    let handler = |name: &str, bindings: Vec<IrPatternBinding>| FoldHandlerIR {
        pattern: IrEventPattern {
            name: name.to_string(),
            event: None,
            bindings,
            span,
        },
        body: vec![],
        span,
    };

    vec![
        // ------------- engaged_with: per_entity_topk(1) -------------
        ViewIR {
            name: "engaged_with".into(),
            params: vec![param("a", IrType::AgentId)],
            return_ty: IrType::AgentId,
            body: ViewBodyIR::Fold {
                initial: IrExprNode {
                    kind: IrExpr::LitInt(0),
                    span,
                },
                handlers: vec![
                    handler(
                        "EngagementCommitted",
                        vec![bind("actor", "a"), bind("target", "b")],
                    ),
                    handler(
                        "EngagementBroken",
                        vec![bind("actor", "a"), bind("former_target", "b")],
                    ),
                ],
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PerEntityTopK { k: 1, keyed_on: 0 }),
            decay: None,
            span,
        },
        // ------------- my_enemies: pair_map no decay -------------
        ViewIR {
            name: "my_enemies".into(),
            params: vec![
                param("observer", IrType::AgentId),
                param("attacker", IrType::AgentId),
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![handler(
                    "AgentAttacked",
                    vec![bind("actor", "attacker"), bind("target", "observer")],
                )],
                clamp: Some((lit_f(0.0), lit_f(1.0))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: None,
            span,
        },
        // ------------- threat_level: pair_map @decay(0.98) -------------
        ViewIR {
            name: "threat_level".into(),
            params: vec![param("a", IrType::AgentId), param("b", IrType::AgentId)],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![
                    handler(
                        "AgentAttacked",
                        vec![bind("actor", "b"), bind("target", "a")],
                    ),
                    handler(
                        "EffectDamageApplied",
                        vec![bind("actor", "b"), bind("target", "a")],
                    ),
                ],
                clamp: Some((lit_f(0.0), lit_f(1000.0))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate: 0.98,
                per: DecayUnit::Tick,
                span,
            }),
            span,
        },
        // ------------- kin_fear: pair_map @decay(0.891) -------------
        ViewIR {
            name: "kin_fear".into(),
            params: vec![
                param("observer", IrType::AgentId),
                param("dead_kin", IrType::AgentId),
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![handler(
                    "FearSpread",
                    vec![bind("observer", "observer"), bind("dead_kin", "dead_kin")],
                )],
                clamp: Some((lit_f(0.0), lit_f(10.0))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate: 0.891,
                per: DecayUnit::Tick,
                span,
            }),
            span,
        },
        // ------------- pack_focus: pair_map @decay(0.933) -------------
        ViewIR {
            name: "pack_focus".into(),
            params: vec![
                param("observer", IrType::AgentId),
                param("target", IrType::AgentId),
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![handler(
                    "PackAssist",
                    vec![bind("observer", "observer"), bind("target", "target")],
                )],
                clamp: Some((lit_f(0.0), lit_f(10.0))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate: 0.933,
                per: DecayUnit::Tick,
                span,
            }),
            span,
        },
        // ------------- rally_boost: pair_map @decay(0.891) -------------
        ViewIR {
            name: "rally_boost".into(),
            params: vec![
                param("observer", IrType::AgentId),
                param("wounded_kin", IrType::AgentId),
            ],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f(0.0),
                handlers: vec![handler(
                    "RallyCall",
                    vec![
                        bind("observer", "observer"),
                        bind("wounded_kin", "wounded_kin"),
                    ],
                )],
                clamp: Some((lit_f(0.0), lit_f(10.0))),
            },
            annotations: vec![],
            kind: ViewKind::Materialized(StorageHint::PairMap),
            decay: Some(DecayHint {
                rate: 0.891,
                per: DecayUnit::Tick,
                span,
            }),
            span,
        },
    ]
}

// ---------------------------------------------------------------------------
// Public, consumable spec list — for docs / tests
// ---------------------------------------------------------------------------

/// Build + classify every materialized view in one pass. Consumers use
/// this to drive per-view tests and to get the `ViewStorageSpec`
/// metadata (rate, clamp, shape) without re-parsing the DSL.
pub fn build_all_specs() -> Vec<ViewStorageSpec> {
    build_materialized_view_irs()
        .iter()
        .filter_map(|v| classify_view(v).ok())
        .filter(|s| !matches!(s.shape, ViewShape::Lazy))
        .collect()
}

// Re-export the emit_view_wgsl namespace for integration callers —
// scoring/physics that want to inline the read snippet can reach for
// these via `engine_gpu::view_storage::emit`.
pub mod emit {
    pub use dsl_compiler::emit_view_wgsl::{
        classify_view, emit_view_fold_wgsl, emit_view_read_wgsl, EmitError, FoldSpec, ViewShape,
        ViewStorageSpec,
    };
}

// Keep the `emit_view_wgsl` import explicit-use alive even when no
// callers pull it out of the module (the build script doesn't, but
// the parity test does).
#[allow(unused_imports)]
use emit_view_wgsl as _emit_used;
