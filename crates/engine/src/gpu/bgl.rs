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

/// Reuse a kernel's bind group across ticks. `cache` holds the
/// `(buffer, offset, size)` triple of every entry the group was built
/// from beside the group; the group is rebuilt only when one differs
/// (a ping-pong buffer, a re-created resource, a different cfg slot).
/// Buffer equality in wgpu is identity (the same underlying resource),
/// so a hit is exactly "the same descriptor as last time" — the same
/// GPU work, bit for bit, minus a `create_bind_group` per dispatch per
/// tick.
///
/// Only buffer bindings are cacheable; a group with any other resource
/// kind is rebuilt on every call (never cached, never wrong).
pub fn cached_bind_group<'c>(
    cache: &'c mut Option<(Vec<(wgpu::Buffer, u64, u64)>, wgpu::BindGroup)>,
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> &'c wgpu::BindGroup {
    let mut key: Vec<(wgpu::Buffer, u64, u64)> = Vec::with_capacity(entries.len());
    let mut cacheable = true;
    for e in entries {
        match &e.resource {
            wgpu::BindingResource::Buffer(bb) => {
                key.push((bb.buffer.clone(), bb.offset, bb.size.map(|n| n.get()).unwrap_or(0)));
            }
            _ => {
                cacheable = false;
                break;
            }
        }
    }
    let hit = cacheable && cache.as_ref().is_some_and(|(k, _)| *k == key);
    if !hit {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        });
        // A non-cacheable group is stored with an empty key so it can never
        // match (an empty entry list is impossible for a real kernel).
        *cache = Some((if cacheable { key } else { Vec::new() }, bg));
    }
    &cache.as_ref().unwrap().1
}
