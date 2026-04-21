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
/// Two scalars (attack_range, max_move_radius) plus 8 bytes of padding
/// to align the block to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuConfig {
    combat_attack_range: f32,
    movement_max_move_radius: f32,
    _pad0: f32,
    _pad1: f32,
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
    cfg_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
        } = module;

        Ok(Self { pipeline, bind_group_layout, bindings, pool: None })
    }

    /// Per-mask binding metadata — name, index, shape — in emission
    /// order. The parity test uses this to partition "GPU says set for
    /// this slot?" assertions per mask and to map the bitmap-vec index
    /// back to the mask name for diagnostics.
    pub fn bindings(&self) -> &[FusedMaskBinding] {
        &self.bindings
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, agent_cap: u32) {
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
                combat_attack_range: 0.0,
                movement_max_move_radius: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group — SoA (0..=2), N bitmaps, cfg.
        let mut bg_entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(3 + bitmap_bufs.len() + 1);
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::fused_mask::bg"),
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
            cfg_buf,
            bind_group,
        });
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

        // Config uniform — pack the knobs the fused kernel reads.
        queue.write_buffer(
            &pool.cfg_buf,
            0,
            bytemuck::cast_slice(&[GpuConfig {
                combat_attack_range: state.config.combat.attack_range,
                movement_max_move_radius: state.config.movement.max_move_radius,
                _pad0: 0.0,
                _pad1: 0.0,
            }]),
        );

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
            cpass.set_bind_group(0, &pool.bind_group, &[]);
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
