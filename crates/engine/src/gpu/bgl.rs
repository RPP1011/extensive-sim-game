//! BGL-entry helpers — `bgl_storage` and `bgl_uniform` — every
//! per-fixture compiler-emitted kernel module calls these to
//! describe its bindings.
//!
//! These mirror exactly the form the DSL compiler used to emit into
//! the per-fixture runtime crate's `lib.rs` (when the runtime crate
//! was `engine_gpu_rules`). Hand-writing them in `engine::gpu`
//! removes the per-fixture compiler-output duplication: one
//! definition, every fixture uses it via `engine::gpu::bgl_storage`.

/// Construct a `BindGroupLayoutEntry` for a storage buffer at
/// binding slot `b`. `read_only = true` for buffers the kernel
/// only reads (`var<storage, read>`); `false` for buffers the
/// kernel writes (`var<storage, read_write>`).
///
/// `visibility` is hardcoded to `COMPUTE` because every
/// compiler-emitted kernel today is a compute kernel; if/when
/// vertex / fragment kernels arrive the helper grows a visibility
/// argument.
pub fn bgl_storage(b: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Construct a `BindGroupLayoutEntry` for a uniform buffer at
/// binding slot `b`. Used for per-kernel `Cfg` bindings (small,
/// per-tick-uploaded payloads — agent_cap, view radius, etc.).
pub fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
