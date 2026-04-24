//! Phase 2 fused-mask kernel — all supported masks in one dispatch.
//!
//! Phase 1 shipped a single-kernel Attack-only path
//! (`engine_gpu::attack_mask`). Phase 2 generalises: one WGSL module
//! with one `cs_fused_masks` entry point writes N bitmap outputs per
//! tick, where N is the count of masks whose IR the fused emitter
//! accepts.
//!
//! ## What made it into the fused kernel
//!
//! The engine ships 8 masks (see `assets/sim/masks.sim`). Seven fit the
//! emitter's current subset and run on GPU:
//!
//!   | Mask        | Shape              | Status                       |
//!   |-------------|--------------------|------------------------------|
//!   | Attack      | target-bound       | implemented + byte-parity    |
//!   | MoveToward  | target-bound       | implemented + byte-parity    |
//!   | Hold        | self-only          | implemented + byte-parity    |
//!   | Flee        | self-only          | implemented + byte-parity    |
//!   | Eat         | self-only          | implemented + byte-parity    |
//!   | Drink       | self-only          | implemented + byte-parity    |
//!   | Rest        | self-only          | implemented + byte-parity    |
//!   | **Cast**    | parametric         | **skipped — Phase 4+ blocker** |
//!
//! ### Why Cast is skipped
//!
//! The DSL-level `mask Cast(ability: AbilityId)` takes a non-Agent
//! parameter and its predicate reads three views + one cooldown field
//! the GPU backend has no storage for:
//!
//!   * `view::is_stunned(self)` — reads `agents.stun_expires_at_tick`
//!   * `abilities.known(self, ability)` — reads the ability registry
//!   * `abilities.cooldown_ready(self, ability)` — reads per-agent
//!     cooldown state
//!   * `agents.engaged_with(self) == None` — reads the
//!     `engaged_with` @materialized view
//!
//! Phase 4 of the plan ports views + cooldown storage; Cast re-joins
//! the fused kernel once those buffers exist. Until then the emitter
//! surfaces `Unsupported("... Skip on GPU ...")` when fed Cast's
//! `IrActionHeadShape`, and this module filters it out at registration
//! time so the rest of the masks dispatch cleanly.
//!
//! ## Buffer layout
//!
//! The fused module's bind group shape, from the DSL compiler side
//! (`emit_mask_wgsl::FUSED_BITMAP_BINDING_BASE`):
//!
//!   * `@binding(0)` `agent_pos: array<Vec3f32>`
//!   * `@binding(1)` `agent_alive: array<u32>`
//!   * `@binding(2)` `agent_creature_type: array<u32>`
//!   * `@binding(3..=9)` `bitmap_<mask>: array<atomic<u32>>` (7 masks)
//!   * `@binding(10)` `cfg: ConfigUniform`
//!
//! One `write_buffer` call per SoA input per tick; one dispatch; seven
//! `copy_buffer_to_buffer` + `map_async` readbacks (done in parallel).
//! The upload/readback cost dominates at small agent counts; Phase 5's
//! spatial hash is the unblocker for larger N.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use dsl_compiler::{
    ast::{BinOp, Span},
    emit_mask_wgsl::{
        emit_masks_wgsl_fused, FusedMaskBinding, FusedMaskModule, FUSED_BITMAP_BINDING_BASE,
    },
    ir::{
        Builtin, IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef,
        MaskIR, NamespaceId,
    },
};
use engine::ids::AgentId;
use engine::state::SimState;
use wgpu::util::DeviceExt;

/// Workgroup size — must agree with `emit_mask_wgsl::WORKGROUP_SIZE`.
pub const WORKGROUP_SIZE: u32 = dsl_compiler::emit_mask_wgsl::WORKGROUP_SIZE;

/// GPU-per-slot agent position — matches the WGSL `struct Vec3f32`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPos {
    x: f32,
    y: f32,
    z: f32,
}

/// Config uniform buffer — matches `struct ConfigUniform` in the WGSL.
/// Carries subsystem-local scalars only; world-scalars (currently
/// `attack_range`) live in `SimCfg` and are bound separately as of
/// Task 2.4 of the GPU sim-state refactor.
///
/// One live scalar (`movement_max_move_radius`) plus 12 bytes of
/// padding to align the block to 16 bytes (WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuConfig {
    movement_max_move_radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Error surface for fused-kernel init + dispatch.
#[derive(Debug)]
pub enum KernelError {
    EmitWgsl(String),
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::EmitWgsl(s) => write!(f, "emit WGSL: {s}"),
            KernelError::ShaderCompile(s) => write!(f, "shader compile: {s}"),
            KernelError::Dispatch(s) => write!(f, "dispatch: {s}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Names of the masks the fused emitter accepts, in a stable order the
/// kernel builder, the backend, and the parity test all agree on.
/// Cast is conspicuously absent — see the module docstring for why.
///
/// Order matters: the `FusedMaskBinding::index` for each mask matches
/// the position of its name in this slice, which is what the
/// `FusedMaskBitmaps` accessor uses to resolve `bitmap_for(name)`.
pub const FUSED_MASK_NAMES: &[&str] = &[
    "Attack",
    "MoveToward",
    "Hold",
    "Flee",
    "Eat",
    "Drink",
    "Rest",
];

/// Build the in-memory MaskIRs the fused emitter consumes. Mirrors
/// `assets/sim/masks.sim` — if the DSL source drifts (new clause,
/// renamed knob), update this file to match so the GPU path stays
/// aligned with the CPU mask predicates.
///
/// The per-mask structure echoes
/// `dsl_compiler::emit_mask::tests::attack_mask_ir()` and its siblings,
/// so keep them in sync when adding a new mask.
pub fn build_fused_mask_irs() -> Vec<MaskIR> {
    fn span() -> Span {
        Span::dummy()
    }
    fn local(name: &str, id: u16) -> IrExprNode {
        IrExprNode { kind: IrExpr::Local(LocalRef(id), name.to_string()), span: span() }
    }
    fn ns_call(ns: NamespaceId, method: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns,
                method: method.to_string(),
                args: args
                    .into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            },
            span: span(),
        }
    }
    fn unresolved(name: &str, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::UnresolvedCall(
                name.to_string(),
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }
    fn builtin(b: Builtin, args: Vec<IrExprNode>) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::BuiltinCall(
                b,
                args.into_iter()
                    .map(|a| IrCallArg { name: None, value: a, span: span() })
                    .collect(),
            ),
            span: span(),
        }
    }
    fn binop(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode { kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)), span: span() }
    }
    fn lit_float(v: f64) -> IrExprNode {
        IrExprNode { kind: IrExpr::LitFloat(v), span: span() }
    }

    // -- Attack(target) --
    // `agents.alive(target) && is_hostile(self, target)
    //   && distance(agents.pos(self), agents.pos(target)) < 2.0`
    let attack_ir = {
        let self_local = local("self", 0);
        let target_local = local("target", 1);
        let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
        let hostile = unresolved("is_hostile", vec![self_local.clone(), target_local.clone()]);
        let self_pos = ns_call(NamespaceId::Agents, "pos", vec![self_local]);
        let target_pos = ns_call(NamespaceId::Agents, "pos", vec![target_local]);
        let dist = builtin(Builtin::Distance, vec![self_pos, target_pos]);
        let lt = binop(BinOp::Lt, dist, lit_float(2.0));
        let and1 = binop(BinOp::And, alive, hostile);
        let predicate = binop(BinOp::And, and1, lt);
        MaskIR {
            head: IrActionHead {
                name: "Attack".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        }
    };

    // -- MoveToward(target) --
    // `agents.alive(target) && target != self`
    let move_toward_ir = {
        let self_local = local("self", 0);
        let target_local = local("target", 1);
        let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
        let ne = binop(BinOp::NotEq, target_local, self_local);
        let predicate = binop(BinOp::And, alive, ne);
        MaskIR {
            head: IrActionHead {
                name: "MoveToward".into(),
                shape: IrActionHeadShape::Positional(vec![(
                    "target".into(),
                    LocalRef(1),
                    IrType::AgentId,
                )]),
                span: span(),
            },
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(),
        }
    };

    // -- Self-only masks (Hold / Flee / Eat / Drink / Rest) --
    // All five predicates collapse to `agents.alive(self)`.
    fn self_only_mask(name: &str) -> MaskIR {
        let self_local = IrExprNode {
            kind: IrExpr::Local(LocalRef(0), "self".into()),
            span: Span::dummy(),
        };
        let alive = IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns: NamespaceId::Agents,
                method: "alive".into(),
                args: vec![IrCallArg {
                    name: None,
                    value: self_local,
                    span: Span::dummy(),
                }],
            },
            span: Span::dummy(),
        };
        MaskIR {
            head: IrActionHead {
                name: name.into(),
                shape: IrActionHeadShape::None,
                span: Span::dummy(),
            },
            candidate_source: None,
            predicate: alive,
            annotations: vec![],
            span: Span::dummy(),
        }
    }

    vec![
        attack_ir,
        move_toward_ir,
        self_only_mask("Hold"),
        self_only_mask("Flee"),
        self_only_mask("Eat"),
        self_only_mask("Drink"),
        self_only_mask("Rest"),
    ]
}

/// Compiled fused-mask pipeline + per-tick buffer pool.
///
/// Owns one `wgpu::ComputePipeline`, one bind-group layout, and a pool
/// of buffers sized for the state's `agent_cap`. The pool re-creates
/// itself if the cap changes across ticks — rare in practice (Phase 2
/// holds a single `SimState` capacity), cheap when it happens.
pub struct FusedMaskKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bindings: Vec<FusedMaskBinding>,
    /// Binding number of the shared `SimCfg` storage buffer in this
    /// kernel's BGL. The resident path binds the caller-supplied
    /// `sim_cfg_buf` here; the sync path binds the pool-owned fallback.
    sim_cfg_binding: u32,
    pool: Option<BufferPool>,
}

struct BufferPool {
    agent_cap: u32,
    mask_words: u32,
    pos_buf: wgpu::Buffer,
    alive_buf: wgpu::Buffer,
    creature_type_buf: wgpu::Buffer,
    /// Per-mask storage buffers, in `bindings` order.
    bitmap_bufs: Vec<wgpu::Buffer>,
    /// Matching readback (MAP_READ) buffers, in `bindings` order.
    bitmap_readback_bufs: Vec<wgpu::Buffer>,
    /// Concat of all per-mask bitmaps (`N * mask_words` u32 words,
    /// `bindings` order). Populated by the resident path via
    /// `copy_buffer_to_buffer` after the dispatch; exposed to
    /// downstream kernels through [`FusedMaskKernel::mask_bitmaps_buf`].
    /// Not used by the sync `run_and_readback` path.
    bitmaps_concat_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    /// Pool-owned `SimCfg` snapshot buffer — used by the sync path
    /// (`run_and_readback`) so it can keep operating without a
    /// caller-supplied resident `sim_cfg_buf`. The resident path
    /// (`run_resident`) binds the caller's buffer instead, via
    /// `resident_bg`.
    sync_sim_cfg_buf: wgpu::Buffer,
    /// Pool-owned alive bitmap for the sync path. Packed host-side
    /// from `state.agent_alive()` on each `run_and_readback` call and
    /// bound at slot 22. The resident path binds the caller-supplied
    /// bitmap instead.
    sync_alive_bitmap_buf: wgpu::Buffer,
    /// Bind group wired with the sync-path `sync_sim_cfg_buf`.
    /// Used by `run_and_readback` only.
    sync_bind_group: wgpu::BindGroup,
    /// Cached bind group for the resident path, keyed by the caller-
    /// supplied `sim_cfg_buf` identity. Stable across a batch (same
    /// buffer every tick), so this amortises to a single BG build.
    resident_bind_group: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl FusedMaskKernel {
    /// Build the fused pipeline on `device`. Emits WGSL via
    /// `emit_masks_wgsl_fused`, parses through wgpu's runtime shader
    /// frontend (naga), and creates the bind-group layout matching
    /// the emitter's binding numbers.
    pub fn new(device: &wgpu::Device) -> Result<Self, KernelError> {
        let irs = build_fused_mask_irs();
        let module = emit_masks_wgsl_fused(&irs).map_err(KernelError::EmitWgsl)?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::fused_mask::wgsl"),
            source: wgpu::ShaderSource::Wgsl(module.wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(KernelError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{}",
                module.wgsl
            )));
        }

        // One bind-group layout covering:
        //   0..=2      SoA reads
        //   3..=2+N    per-mask bitmaps (storage, read_write)
        //   2+N+1      cfg uniform
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        for i in 0..module.bindings.len() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: FUSED_BITMAP_BINDING_BASE + i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: module.config_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        // SimCfg binding (Task 2.4) — shared world-scalars, read-only
        // storage. The kernel only reads `sim_cfg.attack_range` today;
        // future kernel migrations will widen the read surface.
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: module.sim_cfg_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(
                    std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
                ),
            },
            count: None,
        });
        // Per-tick alive bitmap at slot 22 — shared across physics +
        // scoring + mask kernels. The fused mask WGSL declares the
        // binding in its shader prefix; the DSL-lowered `alive_bit(x)`
        // helper reads a single bit instead of a 4 B SoA load.
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: crate::alive_bitmap::ALIVE_BITMAP_BINDING,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::fused_mask::bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::fused_mask::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::fused_mask::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_fused_masks"),
            compilation_options: Default::default(),
            cache: None,
        });

        let FusedMaskModule {
            wgsl: _,
            bindings,
            config_binding: _,
            sim_cfg_binding,
        } = module;

        Ok(Self {
            pipeline,
            bind_group_layout,
            bindings,
            sim_cfg_binding,
            pool: None,
        })
    }

    /// Per-mask binding metadata — name, index, shape — in emission
    /// order. The parity test uses this to partition "GPU says set for
    /// this slot?" assertions per mask and to map the bitmap-vec index
    /// back to the mask name for diagnostics.
    pub fn bindings(&self) -> &[FusedMaskBinding] {
        &self.bindings
    }

    pub(crate) fn ensure_pool(&mut self, device: &wgpu::Device, agent_cap: u32) {
        if let Some(p) = &self.pool {
            if p.agent_cap == agent_cap {
                return;
            }
        }
        let mask_words = agent_cap.div_ceil(32).max(1);

        let pos_bytes = (agent_cap as usize) * std::mem::size_of::<GpuPos>();
        let alive_bytes = (agent_cap as usize) * std::mem::size_of::<u32>();
        let ct_bytes = (agent_cap as usize) * std::mem::size_of::<u32>();
        let mask_bytes = (mask_words as usize) * std::mem::size_of::<u32>();

        let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::fused_mask::agent_pos"),
            size: pos_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::fused_mask::agent_alive"),
            size: alive_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let creature_type_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::fused_mask::agent_creature_type"),
            size: ct_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut bitmap_bufs = Vec::with_capacity(self.bindings.len());
        let mut bitmap_readback_bufs = Vec::with_capacity(self.bindings.len());
        for b in &self.bindings {
            let snake = snake_case_lower(&b.mask_name);
            bitmap_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("engine_gpu::fused_mask::bitmap_{snake}")),
                size: mask_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            bitmap_readback_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("engine_gpu::fused_mask::readback_{snake}")),
                size: mask_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::fused_mask::cfg"),
            contents: bytemuck::cast_slice(&[GpuConfig {
                movement_max_move_radius: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Sync-path fallback `SimCfg` buffer. `run_and_readback` keeps
        // its historical behaviour (upload full cfg uniform on each
        // tick) by uploading a fresh snapshot here too. The resident
        // path ignores this and binds the caller-supplied `sim_cfg_buf`
        // in `resident_bind_group`.
        let sync_sim_cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::fused_mask::sync_sim_cfg"),
            size: std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sync-path alive bitmap for the mask kernel. Sized for
        // `agent_cap` — packed host-side from `state.agent_alive()`
        // on every `run_and_readback` call. The resident path binds
        // the caller-supplied bitmap (populated by `alive_pack_kernel`)
        // instead.
        let sync_alive_bitmap_buf =
            crate::alive_bitmap::create_alive_bitmap_buffer(device, agent_cap);

        // Resident-path output: single concat of all per-mask bitmaps.
        // `bindings.len() * mask_words` u32 words, in `bindings` order
        // (same order the scoring kernel's `mask_bitmaps` binding
        // expects after `pack_mask_bitmaps` on the sync path).
        let concat_words = (self.bindings.len() as u64) * (mask_words as u64);
        let bitmaps_concat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::fused_mask::bitmaps_concat"),
            size: concat_words * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sync-path bind group — SoA (0..=2), N bitmaps, cfg uniform,
        // sync-path `SimCfg` storage fallback. Used by
        // `run_and_readback` only; the resident path uses
        // `resident_bind_group` with the caller's `sim_cfg_buf`.
        let mut bg_entries: Vec<wgpu::BindGroupEntry<'_>> =
            Vec::with_capacity(3 + bitmap_bufs.len() + 2);
        bg_entries.push(wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() });
        bg_entries.push(wgpu::BindGroupEntry { binding: 1, resource: alive_buf.as_entire_binding() });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: creature_type_buf.as_entire_binding(),
        });
        for (i, buf) in bitmap_bufs.iter().enumerate() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: FUSED_BITMAP_BINDING_BASE + i as u32,
                resource: buf.as_entire_binding(),
            });
        }
        let cfg_binding = FUSED_BITMAP_BINDING_BASE + bitmap_bufs.len() as u32;
        bg_entries.push(wgpu::BindGroupEntry {
            binding: cfg_binding,
            resource: cfg_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: self.sim_cfg_binding,
            resource: sync_sim_cfg_buf.as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: crate::alive_bitmap::ALIVE_BITMAP_BINDING,
            resource: sync_alive_bitmap_buf.as_entire_binding(),
        });

        let sync_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::fused_mask::sync_bg"),
            layout: &self.bind_group_layout,
            entries: &bg_entries,
        });

        self.pool = Some(BufferPool {
            agent_cap,
            mask_words,
            pos_buf,
            alive_buf,
            creature_type_buf,
            bitmap_bufs,
            bitmap_readback_bufs,
            bitmaps_concat_buf,
            cfg_buf,
            sync_sim_cfg_buf,
            sync_alive_bitmap_buf,
            sync_bind_group,
            resident_bind_group: None,
        });
    }

    // -----------------------------------------------------------------
    // Resident path (Phase B of the GPU-resident cascade refactor)
    //
    // The resident entry point encodes the mask dispatch into a
    // caller-owned command encoder and writes output into a stable
    // concat buffer (see [`Self::mask_bitmaps_buf`]). The sync path
    // [`Self::run_and_readback`] is untouched and continues to own its
    // own submit + readback chain.
    //
    // ### Note on `agents_buf`
    //
    // The plan's `run_resident` signature (see
    // `docs/superpowers/plans/2026-04-22-gpu-resident-cascade.md` §B3)
    // takes a caller-supplied `agents_buf: &wgpu::Buffer` alongside
    // `agent_cap`. That signature was written assuming the mask kernel
    // reads a single packed `GpuAgentSlot` AoS buffer (like the
    // physics / apply_actions / movement kernels do).
    //
    // The current fused-mask WGSL, however, binds three separate SoA
    // buffers at @binding(0..=2) — `agent_pos`, `agent_alive`,
    // `agent_creature_type` — sized to `agent_cap` elements each. It
    // does NOT read a packed AgentSlot. Until the WGSL is rewritten to
    // consume a packed layout (tracked as a follow-up — out of scope
    // for this infra-only task), the resident path keeps uploading
    // SoA into the pool's internal buffers through
    // [`Self::upload_soa_from_state`], which callers must invoke once
    // per tick before [`Self::run_resident`].
    //
    // `run_resident` therefore accepts `agents_buf` to lock in the
    // planned signature (so Task D4's cascade wiring compiles and
    // stays stable across the follow-up WGSL rewrite) but does not
    // bind it. When the WGSL is rewritten the body will swap out the
    // three SoA bindings for `agents_buf` without a signature change.
    // -----------------------------------------------------------------

    /// Upload the SoA fields + config uniform + clear bitmaps for the
    /// resident path, from `state`. Must be called once per tick,
    /// before [`Self::run_resident`], so the dispatch sees current
    /// agent state.
    ///
    /// Separated from `run_resident` so the resident path's signature
    /// matches the plan (no `&SimState` argument) — see the module-
    /// level note above `run_resident` on why the SoA upload remains.
    pub fn upload_soa_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
    ) {
        let agent_cap = state.agent_cap();
        self.ensure_pool(device, agent_cap);
        let pool = self.pool.as_ref().expect("pool ensured");

        let pos_src: Vec<GpuPos> = state
            .hot_pos()
            .iter()
            .map(|v| GpuPos { x: v.x, y: v.y, z: v.z })
            .collect();
        queue.write_buffer(&pool.pos_buf, 0, bytemuck::cast_slice(&pos_src));

        let alive_src: Vec<u32> = state
            .hot_alive()
            .iter()
            .map(|&b| if b { 1u32 } else { 0u32 })
            .collect();
        queue.write_buffer(&pool.alive_buf, 0, bytemuck::cast_slice(&alive_src));

        // Pack + upload the sync-path alive bitmap. The fused mask
        // shader's DSL-lowered `alive_bit(x)` calls read from slot 22.
        upload_sync_alive_bitmap(queue, &pool.sync_alive_bitmap_buf, &alive_src);

        let ct_src: Vec<u32> = (0..agent_cap)
            .map(|slot| {
                let id = AgentId::new(slot + 1).unwrap();
                match state.agent_creature_type(id) {
                    Some(ct) => ct as u8 as u32,
                    None => u32::MAX,
                }
            })
            .collect();
        queue.write_buffer(&pool.creature_type_buf, 0, bytemuck::cast_slice(&ct_src));

        queue.write_buffer(
            &pool.cfg_buf,
            0,
            bytemuck::cast_slice(&[GpuConfig {
                movement_max_move_radius: state.config.movement.max_move_radius,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
        );

        // Keep the sync-path fallback `SimCfg` snapshot in lockstep.
        // Callers of the resident path supply their own `sim_cfg_buf`
        // and never read from this one, but we still refresh it so
        // `run_and_readback` (which uses this buffer via
        // `sync_bind_group`) sees current values.
        let sim_cfg = crate::sim_cfg::SimCfg::from_state(state);
        crate::sim_cfg::upload_sim_cfg(queue, &pool.sync_sim_cfg_buf, &sim_cfg);

        // Zero every per-mask bitmap — atomicOr accumulates, so
        // leftover bits from a previous tick would poison the next
        // read.
        let zeros = vec![0u32; pool.mask_words as usize];
        for buf in &pool.bitmap_bufs {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&zeros));
        }
    }

    /// Resident-path sibling to [`Self::run_and_readback`].
    ///
    /// Records the mask dispatch into `encoder` and, within the same
    /// command list, copies each per-mask bitmap into the concat
    /// output buffer exposed by [`Self::mask_bitmaps_buf`]. Does NOT
    /// submit, does NOT copy bitmaps to CPU — the caller binds the
    /// concat buffer directly into the next kernel (typically
    /// scoring).
    ///
    /// ### Preconditions
    ///
    /// * [`Self::upload_soa_from_state`] must have been called on
    ///   this tick (same `queue`) with a `state` whose `agent_cap()`
    ///   equals the `agent_cap` argument passed here. The SoA
    ///   buffers, the cfg uniform, and the per-mask bitmap clears
    ///   all happen there.
    ///
    /// ### `agents_buf`
    ///
    /// Accepted but currently unused — see the module-level note
    /// immediately above this method for the rationale (mask WGSL
    /// reads three SoA buffers, not a packed `GpuAgentSlot`). The
    /// parameter is present to lock in the planned signature so Task
    /// D4's caller wiring stays stable across a future WGSL rewrite.
    pub fn run_resident(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        sim_cfg_buf: &wgpu::Buffer,
        alive_bitmap_buf: &wgpu::Buffer,
        agent_cap: u32,
    ) -> Result<(), KernelError> {
        // Silence unused-param lint without sacrificing the stable
        // API surface — `agents_buf` will be bound once the WGSL is
        // rewritten to read a packed AgentSlot layout.
        let _ = agents_buf;

        self.ensure_pool(device, agent_cap);

        // Build (or reuse the cached) resident bind group that binds
        // the caller-supplied `sim_cfg_buf` at `self.sim_cfg_binding`.
        // Cache hits once per batch (the backend holds a stable
        // resident `sim_cfg_buf` across all ticks of a batch).
        let sim_cfg_binding = self.sim_cfg_binding;
        let bind_group_layout = &self.bind_group_layout;
        let pool = self.pool.as_mut().expect("pool ensured");
        debug_assert_eq!(
            pool.agent_cap, agent_cap,
            "ensure_pool must size the pool to the requested agent_cap",
        );
        let needs_rebuild = match &pool.resident_bind_group {
            Some((cached_buf, _)) => cached_buf != sim_cfg_buf,
            None => true,
        };
        if needs_rebuild {
            let mut bg_entries: Vec<wgpu::BindGroupEntry<'_>> =
                Vec::with_capacity(3 + pool.bitmap_bufs.len() + 3);
            bg_entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: pool.pos_buf.as_entire_binding(),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: pool.alive_buf.as_entire_binding(),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: pool.creature_type_buf.as_entire_binding(),
            });
            for (i, buf) in pool.bitmap_bufs.iter().enumerate() {
                bg_entries.push(wgpu::BindGroupEntry {
                    binding: FUSED_BITMAP_BINDING_BASE + i as u32,
                    resource: buf.as_entire_binding(),
                });
            }
            let cfg_binding = FUSED_BITMAP_BINDING_BASE + pool.bitmap_bufs.len() as u32;
            bg_entries.push(wgpu::BindGroupEntry {
                binding: cfg_binding,
                resource: pool.cfg_buf.as_entire_binding(),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: sim_cfg_binding,
                resource: sim_cfg_buf.as_entire_binding(),
            });
            // Per-tick alive bitmap at slot 22 — caller-supplied on
            // the resident path (populated by `alive_pack_kernel`).
            bg_entries.push(wgpu::BindGroupEntry {
                binding: crate::alive_bitmap::ALIVE_BITMAP_BINDING,
                resource: alive_bitmap_buf.as_entire_binding(),
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::fused_mask::resident_bg"),
                layout: bind_group_layout,
                entries: &bg_entries,
            });
            pool.resident_bind_group = Some((sim_cfg_buf.clone(), bg));
        }
        let resident_bg = &pool
            .resident_bind_group
            .as_ref()
            .expect("resident_bind_group just built")
            .1;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::fused_mask::cpass_resident"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, resident_bg, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        // Pack per-mask bitmaps into the single concat buffer the
        // resident path exposes. Downstream kernels (scoring) expect
        // a single flat layout `[mask_0_words, mask_1_words, ...]`
        // matching `pack_mask_bitmaps`'s contract.
        let mask_bytes = (pool.mask_words as u64) * 4;
        for (i, storage) in pool.bitmap_bufs.iter().enumerate() {
            let dst_offset = (i as u64) * mask_bytes;
            encoder.copy_buffer_to_buffer(
                storage,
                0,
                &pool.bitmaps_concat_buf,
                dst_offset,
                mask_bytes,
            );
        }

        Ok(())
    }

    /// Buffer handle for the concat mask-bitmap output on the
    /// resident path. Stable across ticks within a given `agent_cap`
    /// — the pool rebuilds (and this handle is invalidated) if
    /// `agent_cap` changes between calls.
    ///
    /// Layout: `bindings().len() * mask_words` `u32` words, laid out
    /// as `[mask_0 bitmap, mask_1 bitmap, ...]` in [`Self::bindings`]
    /// order. Matches what [`crate::scoring::pack_mask_bitmaps`]
    /// produces on the sync path.
    ///
    /// ### Panics
    ///
    /// Panics if [`Self::run_resident`] (or another pool-sizing
    /// method such as [`Self::upload_soa_from_state`]) has not been
    /// called yet — the pool is lazily initialised on first use.
    pub fn mask_bitmaps_buf(&self) -> &wgpu::Buffer {
        &self
            .pool
            .as_ref()
            .expect("mask_bitmaps_buf: pool not initialised; call run_resident or upload_soa_from_state first")
            .bitmaps_concat_buf
    }

    /// Upload state into the GPU buffers, dispatch the fused kernel,
    /// and read back one bitmap per mask. Output is indexed by
    /// `bindings()[i].index` — i.e. the same order the kernel module
    /// wrote.
    ///
    /// Each bitmap is a `Vec<u32>`; bit `k` of word `k/32` is set iff
    /// the k-th slot's agent passed that mask's predicate against at
    /// least one candidate (target-bound) or passed the self-only
    /// predicate (self-only).
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
    ) -> Result<Vec<Vec<u32>>, KernelError> {
        let agent_cap = state.agent_cap();
        self.ensure_pool(device, agent_cap);
        let pool = self.pool.as_ref().expect("pool ensured");

        // --- Upload SoA fields ---
        let pos_src: Vec<GpuPos> = state
            .hot_pos()
            .iter()
            .map(|v| GpuPos { x: v.x, y: v.y, z: v.z })
            .collect();
        queue.write_buffer(&pool.pos_buf, 0, bytemuck::cast_slice(&pos_src));

        let alive_src: Vec<u32> = state
            .hot_alive()
            .iter()
            .map(|&b| if b { 1u32 } else { 0u32 })
            .collect();
        queue.write_buffer(&pool.alive_buf, 0, bytemuck::cast_slice(&alive_src));

        // Pack + upload the sync-path alive bitmap at slot 22.
        upload_sync_alive_bitmap(queue, &pool.sync_alive_bitmap_buf, &alive_src);

        let ct_src: Vec<u32> = (0..agent_cap)
            .map(|slot| {
                let id = AgentId::new(slot + 1).unwrap();
                match state.agent_creature_type(id) {
                    Some(ct) => ct as u8 as u32,
                    None => u32::MAX,
                }
            })
            .collect();
        queue.write_buffer(&pool.creature_type_buf, 0, bytemuck::cast_slice(&ct_src));

        // Config uniform — subsystem-local knobs only. World-scalars
        // live in the pool's `sync_sim_cfg_buf`, uploaded just below.
        queue.write_buffer(
            &pool.cfg_buf,
            0,
            bytemuck::cast_slice(&[GpuConfig {
                movement_max_move_radius: state.config.movement.max_move_radius,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
        );

        // Upload sync-path `SimCfg` snapshot — the sync bind group
        // wires `pool.sync_sim_cfg_buf` at `self.sim_cfg_binding`.
        let sim_cfg = crate::sim_cfg::SimCfg::from_state(state);
        crate::sim_cfg::upload_sim_cfg(queue, &pool.sync_sim_cfg_buf, &sim_cfg);

        // Zero every bitmap — atomicOr accumulates, so leftover bits
        // from a previous tick would poison the next read.
        let zeros = vec![0u32; pool.mask_words as usize];
        for buf in &pool.bitmap_bufs {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&zeros));
        }

        // --- Dispatch ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::fused_mask::encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::fused_mask::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &pool.sync_bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        // Copy each output bitmap to its MAP_READ staging buffer.
        for (storage, readback) in pool.bitmap_bufs.iter().zip(pool.bitmap_readback_bufs.iter()) {
            encoder.copy_buffer_to_buffer(storage, 0, readback, 0, (pool.mask_words as u64) * 4);
        }
        queue.submit(Some(encoder.finish()));

        // --- Readback (batched map_async + single poll) ---
        //
        // Prior shape did one map_async + poll PER mask, issuing 7 GPU
        // synchronisation fences per dispatch. Task 197 fires every
        // map_async up front then drains with one `device.poll(Wait)`.
        // On backends where map_async commits are lazy (Vulkan), this
        // collapses the sync overhead onto the single poll rather than
        // paying one fence per mask.
        let n = self.bindings.len();
        let mut rxs: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(n);
        for readback in &pool.bitmap_readback_bufs {
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rxs.push(rx);
        }
        let _ = device.poll(wgpu::PollType::Wait);

        let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
        for (readback, rx) in pool.bitmap_readback_bufs.iter().zip(rxs.into_iter()) {
            let map_result = rx
                .recv()
                .map_err(|e| KernelError::Dispatch(format!("map_async channel closed: {e}")))?;
            map_result.map_err(|e| KernelError::Dispatch(format!("map_async: {e:?}")))?;
            let slice = readback.slice(..);
            let data = slice.get_mapped_range();
            let words: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback.unmap();
            out.push(words);
        }

        Ok(out)
    }
}

/// Pack the per-slot `alive_src` u32 array into a bitmap and upload
/// it to `buf`. Called by both sync-path entry points
/// (`run_and_readback` + the fused unpack path) after the SoA alive
/// array is packed — keeps the two representations in lockstep so
/// the fused mask shader sees the same alive state via both the
/// traditional `agent_alive[...]` SoA read AND the new `alive_bit(x)`
/// bitmap read.
fn upload_sync_alive_bitmap(queue: &wgpu::Queue, buf: &wgpu::Buffer, alive_src: &[u32]) {
    let agent_cap = alive_src.len() as u32;
    let words = crate::alive_bitmap::alive_bitmap_words(agent_cap) as usize;
    let mut packed = vec![0u32; words.max(1)];
    for (slot_idx, &val) in alive_src.iter().enumerate() {
        if val != 0 {
            packed[slot_idx >> 5] |= 1u32 << (slot_idx & 31);
        }
    }
    queue.write_buffer(buf, 0, bytemuck::cast_slice(&packed));
}

// ---------------------------------------------------------------------------
// Mask unpack kernel — GPU-side SoA derivation from GpuAgentSlot AoS.
// ---------------------------------------------------------------------------
//
// The fused mask WGSL binds three SoA buffers (agent_pos / agent_alive
// / agent_creature_type), sized to `agent_cap`. The sync path packs
// these host-side via [`FusedMaskKernel::upload_soa_from_state`]; the
// resident batch path would do the same 100× per 100-tick batch which
// (a) costs ~450 µs/tick at N=2048 in pack loops + write_buffer and
// (b) reads STALE `SimState` positions because `step_batch` mutates
// `resident_agents_buf` on GPU without syncing back to `SimState`
// mid-batch.
//
// [`MaskUnpackKernel`] fixes both: a one-pass compute kernel reads
// the caller-supplied `resident_agents_buf` (packed `GpuAgentSlot`)
// and writes the three SoA buffers inside the same encoder the batch
// loop is building, so targeting decisions see current GPU state at
// every tick.

/// Unpack-kernel config uniform. One u32 (agent_cap) + 12 bytes of
/// padding for 16B alignment.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UnpackCfg {
    agent_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// WGSL source for the mask-SoA unpack kernel. Mirrors the physics
/// `AgentSlot` layout (64 bytes) 1:1 — if `GpuAgentSlot` changes in
/// `physics.rs`, this struct must move in lockstep.
const MASK_UNPACK_WGSL: &str = r#"
struct Vec3f32 { x: f32, y: f32, z: f32 };

struct AgentSlot {
    hp: f32,
    max_hp: f32,
    shield_hp: f32,
    attack_damage: f32,
    alive: u32,
    creature_type: u32,
    engaged_with: u32,
    stun_expires_at: u32,
    slow_expires_at: u32,
    slow_factor_q8: u32,
    cooldown_next_ready: u32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _pad0: u32,
    _pad1: u32,
};

struct UnpackCfg {
    agent_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       agents:            array<AgentSlot>;
@group(0) @binding(1) var<storage, read_write> pos_out:           array<Vec3f32>;
@group(0) @binding(2) var<storage, read_write> alive_out:         array<u32>;
@group(0) @binding(3) var<storage, read_write> creature_type_out: array<u32>;
@group(0) @binding(4) var<uniform>             cfg:               UnpackCfg;

@compute @workgroup_size(64)
fn cs_mask_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.agent_cap) { return; }
    let a = agents[i];
    pos_out[i] = Vec3f32(a.pos_x, a.pos_y, a.pos_z);
    alive_out[i] = a.alive;
    if (a.alive == 0u) {
        creature_type_out[i] = 0xFFFFFFFFu;
    } else {
        creature_type_out[i] = a.creature_type;
    }
}
"#;

/// GPU-side unpack kernel: derives mask's SoA buffers (pos / alive /
/// creature_type) from the resident `GpuAgentSlot` AoS each tick.
///
/// Lives alongside [`FusedMaskKernel`] — shares its pool's SoA buffers
/// so the mask dispatch reads fresh data without any host-side pack.
pub struct MaskUnpackKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Cfg uniform buffer — 16 bytes, lazy-init on first use. The
    /// bind group is rebuilt every call since `agents_buf` + mask's
    /// SoA buffer handles can change if the backend grows
    /// `agent_cap`; bind-group construction is cheap relative to the
    /// ~20–1700 µs/tick work this kernel replaces.
    cfg_buf: Option<wgpu::Buffer>,
    /// Last agent_cap written to `cfg_buf`. Skip redundant writes —
    /// agent_cap is stable across a batch (only changes on
    /// `ensure_pool` grow, which also flips the SoA buffer handles so
    /// the bind group rebuilds anyway).
    last_agent_cap: u32,
    /// Cached bind group keyed by agent_cap. The (agents_buf,
    /// mask_pool_SoA, cfg_buf) tuple is stable across a batch; we
    /// invalidate on agent_cap change (which swaps agents_buf +
    /// mask's SoA handles, invalidating this bind group). Saves a
    /// bind-group build per batch tick.
    cached_bg: Option<(u32, wgpu::BindGroup)>,
}

impl MaskUnpackKernel {
    /// Build the unpack pipeline on `device`. WGSL is statically
    /// embedded — no emitter round-trip.
    pub fn new(device: &wgpu::Device) -> Result<Self, KernelError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::mask_unpack::wgsl"),
            source: wgpu::ShaderSource::Wgsl(MASK_UNPACK_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(KernelError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{MASK_UNPACK_WGSL}"
            )));
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::mask_unpack::bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::mask_unpack::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::mask_unpack::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_mask_unpack"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            cfg_buf: None,
            last_agent_cap: 0,
            cached_bg: None,
        })
    }

    /// Record the unpack dispatch into `encoder`. Reads `agents_buf`
    /// (packed `GpuAgentSlot` AoS, size ≥ agent_cap slots), writes
    /// mask's internal SoA buffers.
    ///
    /// Must be called AFTER [`FusedMaskKernel::ensure_pool`] (which
    /// allocates the SoA buffers this kernel writes into) — the
    /// caller-supplied `mask` argument hands the pool to us. Any
    /// subsequent mask dispatch on `mask` in the same encoder reads
    /// the fresh SoA.
    pub fn encode_unpack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask: &mut FusedMaskKernel,
        agents_buf: &wgpu::Buffer,
        agent_cap: u32,
    ) {
        // Make sure mask's pool exists at `agent_cap` — that's where
        // our SoA writes land, and the subsequent mask dispatch reads
        // from. Cheap no-op if already sized.
        mask.ensure_pool(device, agent_cap);
        let pool = mask.pool.as_ref().expect("mask pool ensured");

        // (Re)write cfg uniform only if agent_cap changed. Stable
        // across a batch — agent_cap only changes on `ensure_pool`
        // grow, which forces a SoA-buffer rebuild anyway. Saves a
        // 16-byte `queue.write_buffer` per batch tick.
        let cfg_buf = match &self.cfg_buf {
            Some(b) if self.last_agent_cap == agent_cap => b,
            Some(b) => {
                let cfg = UnpackCfg { agent_cap, _pad0: 0, _pad1: 0, _pad2: 0 };
                queue.write_buffer(b, 0, bytemuck::bytes_of(&cfg));
                self.last_agent_cap = agent_cap;
                b
            }
            None => {
                let cfg = UnpackCfg { agent_cap, _pad0: 0, _pad1: 0, _pad2: 0 };
                let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("engine_gpu::mask_unpack::cfg"),
                    contents: bytemuck::bytes_of(&cfg),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                self.cfg_buf = Some(b);
                self.last_agent_cap = agent_cap;
                self.cfg_buf.as_ref().unwrap()
            }
        };

        // Reuse cached bind group when agent_cap is unchanged. The
        // caller-supplied `agents_buf` is assumed stable across a
        // batch (enforced by `GpuBackend::ensure_resident_init` —
        // agent_cap grow rebuilds `resident_agents_buf` AND invokes
        // `ensure_pool(..., agent_cap)` on the mask pool, both of
        // which trip the cap mismatch below).
        let need_rebuild = match &self.cached_bg {
            Some((cap, _)) => *cap != agent_cap,
            None => true,
        };
        if need_rebuild {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::mask_unpack::bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: agents_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pool.pos_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pool.alive_buf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pool.creature_type_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: cfg_buf.as_entire_binding() },
                ],
            });
            self.cached_bg = Some((agent_cap, bg));
        }
        let bind_group = &self
            .cached_bg
            .as_ref()
            .expect("cached_bg populated above")
            .1;

        // Zero every per-mask bitmap — the mask kernel uses atomicOr
        // to set bits, so leftover bits from a previous tick would
        // poison the next read. The sync path does this inside
        // `upload_soa_from_state` via `write_buffer`; here we record
        // a GPU clear so the whole tick stays on the GPU.
        let mask_bytes = (pool.mask_words as u64) * 4;
        for buf in &pool.bitmap_bufs {
            encoder.clear_buffer(buf, 0, Some(mask_bytes));
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::mask_unpack::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused agent-SoA unpack kernel (merges mask_unpack + scoring_unpack)
// ---------------------------------------------------------------------------

/// WGSL source for the fused unpack kernel. Combines the two
/// single-pass unpack kernels (`MASK_UNPACK_WGSL` +
/// `SCORING_UNPACK_WGSL`) into ONE kernel that writes all four output
/// buffers in a single dispatch. Saves one compute pass begin/end +
/// one pipeline set per tick in `step_batch`. Bindings:
///
///   0 (ro) agents:          array<AgentSlot>
///   1 (rw) mask_pos_out:    array<Vec3f32>      (mask SoA)
///   2 (rw) mask_alive_out:  array<u32>          (mask SoA)
///   3 (rw) mask_ct_out:     array<u32>          (mask SoA)
///   4 (rw) scoring_data:    array<AgentData>    (scoring AoS)
///   5 (uniform) cfg:        UnpackCfg
const FUSED_AGENT_UNPACK_WGSL: &str = r#"
struct Vec3f32 { x: f32, y: f32, z: f32 };

struct AgentSlot {
    hp: f32,
    max_hp: f32,
    shield_hp: f32,
    attack_damage: f32,
    alive: u32,
    creature_type: u32,
    engaged_with: u32,
    stun_expires_at: u32,
    slow_expires_at: u32,
    slow_factor_q8: u32,
    cooldown_next_ready: u32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _pad0: u32,
    _pad1: u32,
};

// Must match `GpuAgentData` 1:1 — fields this kernel writes are pos,
// hp, max_hp, shield_hp, alive, creature_type, hp_pct. Static fields
// (attack_range, hunger, thirst, fatigue) are left untouched.
struct AgentData {
    pos: Vec3f32,
    hp: f32,
    max_hp: f32,
    shield_hp: f32,
    attack_range: f32,
    hunger: f32,
    thirst: f32,
    fatigue: f32,
    alive: u32,
    creature_type: u32,
    hp_pct: f32,
    target_hp_pct_unused: f32,
    _pad2: u32,
    _pad3: u32,
};

struct UnpackCfg {
    agent_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       agents:            array<AgentSlot>;
@group(0) @binding(1) var<storage, read_write> mask_pos_out:      array<Vec3f32>;
@group(0) @binding(2) var<storage, read_write> mask_alive_out:    array<u32>;
@group(0) @binding(3) var<storage, read_write> mask_ct_out:       array<u32>;
@group(0) @binding(4) var<storage, read_write> scoring_data:      array<AgentData>;
@group(0) @binding(5) var<uniform>             cfg:               UnpackCfg;

@compute @workgroup_size(64)
fn cs_fused_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.agent_cap) { return; }
    let a = agents[i];
    // Mask SoA outputs.
    mask_pos_out[i] = Vec3f32(a.pos_x, a.pos_y, a.pos_z);
    mask_alive_out[i] = a.alive;
    if (a.alive == 0u) {
        mask_ct_out[i] = 0xFFFFFFFFu;
    } else {
        mask_ct_out[i] = a.creature_type;
    }
    // Scoring AoS outputs (mutable subset only — leaves static fields
    // alone).
    scoring_data[i].pos = Vec3f32(a.pos_x, a.pos_y, a.pos_z);
    scoring_data[i].hp = a.hp;
    scoring_data[i].max_hp = a.max_hp;
    scoring_data[i].shield_hp = a.shield_hp;
    scoring_data[i].alive = a.alive;
    if (a.alive == 0u) {
        scoring_data[i].creature_type = 0xFFFFFFFFu;
        scoring_data[i].hp_pct = 0.0;
    } else {
        scoring_data[i].creature_type = a.creature_type;
        if (a.max_hp > 0.0) {
            scoring_data[i].hp_pct = a.hp / a.max_hp;
        } else {
            scoring_data[i].hp_pct = 0.0;
        }
    }
}
"#;

/// GPU-side fused unpack kernel: replaces the back-to-back
/// [`MaskUnpackKernel`] and `ScoringUnpackKernel` dispatches with a
/// single dispatch that writes to all four output buffers at once.
/// Saves one compute-pass begin/end + one pipeline set per tick.
pub struct FusedAgentUnpackKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    cfg_buf: Option<wgpu::Buffer>,
    last_agent_cap: u32,
    cached_bg: Option<(u32, wgpu::BindGroup)>,
}

impl FusedAgentUnpackKernel {
    /// Build the fused unpack pipeline on `device`.
    pub fn new(device: &wgpu::Device) -> Result<Self, KernelError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::fused_unpack::wgsl"),
            source: wgpu::ShaderSource::Wgsl(FUSED_AGENT_UNPACK_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(KernelError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{FUSED_AGENT_UNPACK_WGSL}"
            )));
        }

        let storage_ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::fused_unpack::bgl"),
            entries: &[
                storage_ro(0),
                storage_rw(1),
                storage_rw(2),
                storage_rw(3),
                storage_rw(4),
                uniform(5),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::fused_unpack::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::fused_unpack::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_fused_unpack"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            cfg_buf: None,
            last_agent_cap: 0,
            cached_bg: None,
        })
    }

    /// Record the fused unpack dispatch into `encoder`. Reads
    /// `agents_buf` (packed `GpuAgentSlot` AoS) and writes:
    ///   * mask's SoA (pos / alive / creature_type) via the
    ///     `mask.pool` buffers,
    ///   * scoring's `agent_data_buf` (mutable subset).
    ///
    /// Also emits the per-tick mask-bitmap clears (outside the pass)
    /// since the mask kernel uses atomicOr and requires zeroed bitmaps.
    pub fn encode_unpack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mask: &mut FusedMaskKernel,
        scoring: &mut crate::scoring::ScoringKernel,
        agents_buf: &wgpu::Buffer,
        agent_cap: u32,
    ) -> Result<(), KernelError> {
        let num_mask_words = agent_cap.div_ceil(32).max(1);
        mask.ensure_pool(device, agent_cap);
        scoring
            .ensure_pool_for_fused_unpack(device, agent_cap, num_mask_words);

        let mask_pool = mask.pool.as_ref().expect("mask pool ensured");
        let scoring_pool = scoring
            .pool_buffers_for_fused_unpack()
            .expect("scoring pool ensured");

        // (Re)write cfg uniform only if agent_cap changed.
        let cfg_buf = match &self.cfg_buf {
            Some(b) if self.last_agent_cap == agent_cap => b,
            Some(b) => {
                let cfg = UnpackCfg { agent_cap, _pad0: 0, _pad1: 0, _pad2: 0 };
                queue.write_buffer(b, 0, bytemuck::bytes_of(&cfg));
                self.last_agent_cap = agent_cap;
                b
            }
            None => {
                let cfg = UnpackCfg { agent_cap, _pad0: 0, _pad1: 0, _pad2: 0 };
                let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("engine_gpu::fused_unpack::cfg"),
                    contents: bytemuck::bytes_of(&cfg),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                self.cfg_buf = Some(b);
                self.last_agent_cap = agent_cap;
                self.cfg_buf.as_ref().unwrap()
            }
        };

        let need_rebuild = match &self.cached_bg {
            Some((cap, _)) => *cap != agent_cap,
            None => true,
        };
        if need_rebuild {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::fused_unpack::bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: agents_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mask_pool.pos_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: mask_pool.alive_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: mask_pool.creature_type_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scoring_pool.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: cfg_buf.as_entire_binding() },
                ],
            });
            self.cached_bg = Some((agent_cap, bg));
        }
        let bind_group = &self.cached_bg.as_ref().expect("cached_bg populated").1;

        // Bitmap clears — required before the mask dispatch (which
        // atomicOr's bits into these).
        let mask_bytes = (mask_pool.mask_words as u64) * 4;
        for buf in &mask_pool.bitmap_bufs {
            encoder.clear_buffer(buf, 0, Some(mask_bytes));
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::fused_unpack::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        Ok(())
    }
}

/// Local snake-case helper — identical shape to the emitter's, kept in
/// this module so the buffer-label strings don't depend on the
/// emitter's private naming contract. If the emitter's `snake_case`
/// ever disagrees with this, the binding-number → mask-name mapping
/// still flows through `FusedMaskBinding::mask_name` / `::index`;
/// labels are debug-only.
fn snake_case_lower(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU reference bitmaps
// ---------------------------------------------------------------------------

/// Compute the CPU-side bitmap for a given mask name — the function
/// the GPU kernel's output is asserted against tick-by-tick. Returns
/// `None` if the mask isn't one the Phase 2 backend knows how to
/// compute; callers fall back to skipping that mask in the parity
/// check.
///
/// Mirrors the GPU kernel's semantics exactly:
///
///   * Self-only masks: bit set iff `agents.alive(self)`. Since the
///     DSL predicates for Hold/Flee/Eat/Drink/Rest all collapse to
///     exactly that clause, the CPU reference is "the alive bitmap".
///   * Attack: bit set iff there's at least one alive hostile target
///     within `combat.attack_range` of self, and
///     `distance(self, target) < 2.0` (the inner clause from the DSL
///     predicate — identical to the Phase 1 path).
///   * MoveToward: bit set iff there's at least one alive neighbour
///     within `movement.max_move_radius` of self, excluding self.
pub fn cpu_mask_bitmap(state: &SimState, mask_name: &str) -> Option<Vec<u32>> {
    let agent_cap = state.agent_cap();
    let mask_words = agent_cap.div_ceil(32).max(1);
    let mut out = vec![0u32; mask_words as usize];

    match mask_name {
        "Attack" => {
            use engine::generated::mask::mask_attack;
            for self_slot in 0..agent_cap {
                let self_id = match AgentId::new(self_slot + 1) {
                    Some(id) => id,
                    None => continue,
                };
                if !state.agent_alive(self_id) {
                    continue;
                }
                let self_pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);
                let mut found = false;
                for target_slot in 0..agent_cap {
                    if target_slot == self_slot {
                        continue;
                    }
                    let target_id = match AgentId::new(target_slot + 1) {
                        Some(id) => id,
                        None => continue,
                    };
                    if !state.agent_alive(target_id) {
                        continue;
                    }
                    let target_pos = state.agent_pos(target_id).unwrap_or(glam::Vec3::ZERO);
                    if self_pos.distance(target_pos) > state.config.combat.attack_range {
                        continue;
                    }
                    if mask_attack(state, self_id, target_id) {
                        found = true;
                        break;
                    }
                }
                if found {
                    set_bit(&mut out, self_slot);
                }
            }
        }
        "MoveToward" => {
            use engine::generated::mask::mask_move_toward;
            for self_slot in 0..agent_cap {
                let self_id = match AgentId::new(self_slot + 1) {
                    Some(id) => id,
                    None => continue,
                };
                if !state.agent_alive(self_id) {
                    continue;
                }
                let self_pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);
                let mut found = false;
                for target_slot in 0..agent_cap {
                    if target_slot == self_slot {
                        continue;
                    }
                    let target_id = match AgentId::new(target_slot + 1) {
                        Some(id) => id,
                        None => continue,
                    };
                    if !state.agent_alive(target_id) {
                        continue;
                    }
                    let target_pos = state.agent_pos(target_id).unwrap_or(glam::Vec3::ZERO);
                    if self_pos.distance(target_pos) > state.config.movement.max_move_radius {
                        continue;
                    }
                    if mask_move_toward(state, self_id, target_id) {
                        found = true;
                        break;
                    }
                }
                if found {
                    set_bit(&mut out, self_slot);
                }
            }
        }
        // Self-only masks: bit set iff alive. All five share this shape
        // in the DSL source so one branch covers them all.
        "Hold" | "Flee" | "Eat" | "Drink" | "Rest" => {
            for self_slot in 0..agent_cap {
                let self_id = match AgentId::new(self_slot + 1) {
                    Some(id) => id,
                    None => continue,
                };
                if state.agent_alive(self_id) {
                    set_bit(&mut out, self_slot);
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

#[inline]
fn set_bit(words: &mut [u32], slot: u32) {
    let word = (slot / 32) as usize;
    let bit = slot % 32;
    words[word] |= 1u32 << bit;
}
