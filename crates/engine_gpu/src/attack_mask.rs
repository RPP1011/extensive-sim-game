//! Phase 1 GPU Attack-mask kernel.
//!
//! One compute pipeline, four storage buffers + one uniform, a readback
//! buffer, and a small helper that assembles the Attack `MaskIR` in
//! memory (mirroring the test fixture in `dsl_compiler::emit_mask`) so
//! the WGSL can be emitted at backend init time without dragging the
//! full .sim resolve pipeline into the GPU path. Phase 2 will widen the
//! scope by compiling the real `.sim` files at startup; Phase 1 prefers
//! the narrower inline `MaskIR` so a regression in the general resolver
//! can't break the GPU path.
//!
//! Alternative approach (preferred at Phase 1, per the plan): the
//! kernel here runs alongside the CPU step and its output is compared
//! against the CPU-computed attack-mask bitmap inside the parity test.
//! That proves WGSL emit → naga parse → dispatch → readback works
//! without touching the engine's step kernel. Swapping the GPU result
//! into `SimScratch.mask` is the Phase 2 drop-in step.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use dsl_compiler::{
    ast::{BinOp, Span},
    emit_mask_wgsl::emit_mask_wgsl,
    ir::{
        Builtin, IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef,
        MaskIR, NamespaceId,
    },
};
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::SimState;
use wgpu::util::DeviceExt;

/// Workgroup size — must agree with `emit_mask_wgsl::WORKGROUP_SIZE`.
/// Duplicated here deliberately: if the emitter's value drifts without
/// updating the dispatch count, the parity test's per-slot agreement
/// check fails loud — the constant mismatch surfaces as a missing bit
/// at a predictable slot, not a silent kernel hang.
pub const WORKGROUP_SIZE: u32 = dsl_compiler::emit_mask_wgsl::WORKGROUP_SIZE;

/// Build the in-memory Attack `MaskIR` that `assets/sim/masks.sim`
/// lowers to. Keeps the Phase 1 GPU path decoupled from disk — the
/// WGSL emitter produces identical output whether fed this fixture or
/// the real compiled IR, so the kernel is correct as long as the
/// fixture matches the .sim source.
///
/// The shape mirrors `dsl_compiler::emit_mask::tests::attack_mask_ir()`:
///   `agents.alive(target) && is_hostile(self, target)
///    && distance(agents.pos(self), agents.pos(target)) < 2.0`.
fn attack_mask_ir() -> MaskIR {
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

    let self_local = local("self", 0);
    let target_local = local("target", 1);

    let alive = ns_call(NamespaceId::Agents, "alive", vec![target_local.clone()]);
    let hostile = unresolved("is_hostile", vec![self_local.clone(), target_local.clone()]);
    let self_pos = ns_call(NamespaceId::Agents, "pos", vec![self_local.clone()]);
    let target_pos = ns_call(NamespaceId::Agents, "pos", vec![target_local.clone()]);
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
            span: Span::dummy(),
        },
        candidate_source: None,
        predicate,
        annotations: vec![],
        span: Span::dummy(),
    }
}

/// Per-tick upload payload — one `Vec3f32` per slot. Matches the WGSL
/// struct in `emit_mask_wgsl::emit_bindings`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPos {
    x: f32,
    y: f32,
    z: f32,
}

/// Config uniform buffer — matches `struct ConfigUniform` in the WGSL.
/// Padding aligns to 16 bytes so WGSL/std140-ish doesn't complain.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuConfig {
    combat_attack_range: f32,
    _pad: [f32; 3],
}

/// Compiled Attack-mask kernel + persistent device-side storage.
///
/// `agent_cap_words` is the number of `u32` words the output bitmap
/// needs — `ceil(agent_cap / 32)`. We size the buffer pool once at
/// construction to the state's `agent_cap`; re-size would require
/// re-creating the bind group, which Phase 1 avoids by pinning the
/// backend to a single SimState size.
pub struct AttackMaskKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // Current buffer pool. Recreated when `agent_cap` changes.
    pool: Option<BufferPool>,
}

struct BufferPool {
    agent_cap: u32,
    pos_buf: wgpu::Buffer,
    alive_buf: wgpu::Buffer,
    creature_type_buf: wgpu::Buffer,
    mask_out_buf: wgpu::Buffer,
    mask_readback_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    mask_words: u32,
}

/// Error surface for kernel init + dispatch. Converts to the parent
/// `engine_gpu::Error` via `From`.
#[derive(Debug)]
pub enum KernelError {
    EmitWgsl(String),
    ShaderCompile(String),
    NoDevice,
    Dispatch(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::EmitWgsl(s) => write!(f, "emit WGSL: {s}"),
            KernelError::ShaderCompile(s) => write!(f, "shader compile: {s}"),
            KernelError::NoDevice => write!(f, "no compatible GPU adapter found"),
            KernelError::Dispatch(s) => write!(f, "dispatch: {s}"),
        }
    }
}

impl std::error::Error for KernelError {}

impl AttackMaskKernel {
    /// Build the Attack-mask compute pipeline on `device`. Emits the
    /// WGSL source via `emit_mask_wgsl`, parses it through wgpu's
    /// runtime shader frontend (which hands off to `naga`), and
    /// creates the bind-group layout + compute pipeline.
    pub fn new(device: &wgpu::Device) -> Result<Self, KernelError> {
        let ir = attack_mask_ir();
        let wgsl = emit_mask_wgsl(&ir).map_err(KernelError::EmitWgsl)?;

        // Runtime WGSL parse — wgpu's shader-module constructor routes
        // through `naga::front::wgsl::parse_str`. Errors bubble out as
        // validation panics by default; we'd rather see them as
        // `KernelError`s, so push a scope and grab any failure.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::attack_mask::wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        // Drain the scope; any validation error is a shader-compile
        // error we want to surface as KernelError::ShaderCompile.
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(KernelError::ShaderCompile(format!("{err}")));
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::attack_mask::bgl"),
            entries: &[
                // agent_pos: read storage buffer
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
                // agent_alive: read storage buffer
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
                // agent_creature_type: read storage buffer
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
                // mask_out: read-write storage buffer (atomic<u32>)
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
                // cfg: uniform buffer
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
            label: Some("engine_gpu::attack_mask::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::attack_mask::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_attack"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { pipeline, bind_group_layout, pool: None })
    }

    /// Recreate (or create) the buffer pool for a given `agent_cap`.
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
            label: Some("engine_gpu::attack_mask::agent_pos"),
            size: pos_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::attack_mask::agent_alive"),
            size: alive_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let creature_type_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::attack_mask::agent_creature_type"),
            size: ct_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::attack_mask::mask_out"),
            size: mask_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::attack_mask::mask_readback"),
            size: mask_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::attack_mask::cfg"),
            contents: bytemuck::cast_slice(&[GpuConfig {
                combat_attack_range: 0.0,
                _pad: [0.0; 3],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::attack_mask::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: alive_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: creature_type_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mask_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.pool = Some(BufferPool {
            agent_cap,
            pos_buf,
            alive_buf,
            creature_type_buf,
            mask_out_buf,
            mask_readback_buf,
            cfg_buf,
            bind_group,
            mask_words,
        });
    }

    /// Upload `state`'s per-slot SoA into GPU buffers, dispatch the
    /// Attack-mask kernel, and read back the packed u32 bitmap. Returns
    /// the raw bitmap in word order — bit `i` of word `i/32` is set iff
    /// agent at slot `i` passes the Attack predicate against at least
    /// one target. Slot `i` (0-based) corresponds to `AgentId(i+1)`.
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
    ) -> Result<Vec<u32>, KernelError> {
        let agent_cap = state.agent_cap();
        self.ensure_pool(device, agent_cap);

        // Pool is Some after ensure_pool; unwrap is safe.
        let pool = self.pool.as_ref().expect("pool ensured");

        // --- Upload SoA fields ---
        // agent_pos: reinterpret `&[Vec3]` as `&[GpuPos]` — both are
        // `#[repr(C)]` with three `f32`s, identical memory layout.
        let pos_src: Vec<GpuPos> = state
            .hot_pos()
            .iter()
            .map(|v| GpuPos { x: v.x, y: v.y, z: v.z })
            .collect();
        queue.write_buffer(&pool.pos_buf, 0, bytemuck::cast_slice(&pos_src));

        // agent_alive: convert `bool` -> `u32 { 0, 1 }` — WGSL can't
        // address <32-bit storage elements.
        let alive_src: Vec<u32> = state
            .hot_alive()
            .iter()
            .map(|&b| if b { 1u32 } else { 0u32 })
            .collect();
        queue.write_buffer(&pool.alive_buf, 0, bytemuck::cast_slice(&alive_src));

        // agent_creature_type: Option<CreatureType> -> u32. Dead / unspawned
        // slots map to u32::MAX, which never compares hostile to any
        // variant so the kernel naturally skips them.
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

        // Config uniform — Attack mask pulls `combat.attack_range`.
        queue.write_buffer(
            &pool.cfg_buf,
            0,
            bytemuck::cast_slice(&[GpuConfig {
                combat_attack_range: state.config.combat.attack_range,
                _pad: [0.0; 3],
            }]),
        );

        // Zero the output bitmap — atomicOr accumulates, so leftover
        // bits from a previous tick would poison the result.
        let zeros = vec![0u32; pool.mask_words as usize];
        queue.write_buffer(&pool.mask_out_buf, 0, bytemuck::cast_slice(&zeros));

        // --- Dispatch ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::attack_mask::encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::attack_mask::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &pool.bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        // Copy output to a MAP_READ staging buffer.
        encoder.copy_buffer_to_buffer(
            &pool.mask_out_buf,
            0,
            &pool.mask_readback_buf,
            0,
            (pool.mask_words as u64) * 4,
        );
        queue.submit(Some(encoder.finish()));

        // --- Readback ---
        let slice = pool.mask_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Drive the device until the buffer is mapped.
        let _ = device.poll(wgpu::PollType::Wait);
        let map_result = rx
            .recv()
            .map_err(|e| KernelError::Dispatch(format!("map_async channel closed: {e}")))?;
        map_result.map_err(|e| KernelError::Dispatch(format!("map_async: {e:?}")))?;

        let data = slice.get_mapped_range();
        let words: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.mask_readback_buf.unmap();

        Ok(words)
    }
}

/// Compute the CPU-side Attack mask bitmap for `state`. Bit `i` set iff
/// `mask_attack_candidates(AgentId(i+1))` would push at least one
/// candidate — i.e. the agent is allowed to Attack this tick. Kept in
/// the kernel module so the parity harness has a direct CPU reference
/// identical in semantics to what the GPU kernel computes.
pub fn cpu_attack_mask_bitmap(state: &SimState) -> Vec<u32> {
    let agent_cap = state.agent_cap();
    let mask_words = agent_cap.div_ceil(32).max(1);
    let mut out = vec![0u32; mask_words as usize];

    use engine::generated::mask::mask_attack;

    for self_slot in 0..agent_cap {
        let self_id = match AgentId::new(self_slot + 1) {
            Some(id) => id,
            None => continue,
        };
        if !state.agent_alive(self_id) {
            continue;
        }
        // Brute-force enumerate: mirror the GPU kernel's behavior
        // (no spatial hash — walk every slot, filter by predicate).
        // The `attack_range` prefilter + `distance < 2.0` inner are
        // both enforced inside `mask_attack` itself, so a simple
        // "any target passes" test suffices.
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
            let self_pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);
            let target_pos = state.agent_pos(target_id).unwrap_or(glam::Vec3::ZERO);
            if self_pos.distance(target_pos) > state.config.combat.attack_range {
                // `from` clause radius prefilter — matches the GPU kernel.
                continue;
            }
            if mask_attack(state, self_id, target_id) {
                found = true;
                break;
            }
        }
        if found {
            let word = (self_slot / 32) as usize;
            let bit = self_slot % 32;
            out[word] |= 1u32 << bit;
        }
    }

    // Reference the CreatureType re-export so the import can't be
    // pruned when it isn't used elsewhere in this module — some
    // code paths conditionally consume it.
    let _ = std::marker::PhantomData::<CreatureType>;
    out
}
