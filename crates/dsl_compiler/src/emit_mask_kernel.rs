//! Emits `engine_gpu_rules/src/fused_mask.rs` + `fused_mask.wgsl` and
//! `mask_unpack.rs` + `mask_unpack.wgsl`.
//!
//! `fused_mask` is the canonical home for the BGL-helper functions
//! `bgl_storage` / `bgl_uniform`. Every other kernel module imports
//! them via `crate::fused_mask::bgl_*`. The fused-mask Rust module
//! itself is emitted via the spec-driven `emit_kernel_module_rs` —
//! see `fused_mask_spec` below — and the helper functions are
//! appended to the same file because they have no other natural home.

use std::fmt::Write;

use crate::emit_kernel_module::emit_kernel_module_rs;
use crate::kernel_binding_ir::{AccessMode, BgSource, KernelBinding, KernelSpec};

/// Build the `KernelSpec` for the fused-mask kernel. Four bindings:
/// agents (read), mask_bitmaps (atomic), sim_cfg (read), cfg (uniform).
/// Matches the WGSL emitted alongside.
fn fused_mask_spec() -> KernelSpec {
    KernelSpec {
        name: "fused_mask".into(),
        pascal: "FusedMask".into(),
        entry_point: "cs_fused_masks".into(),
        cfg_struct: "FusedMaskCfg".into(),
        cfg_build_expr: "FusedMaskCfg { agent_cap: state.agent_cap(), num_mask_words: (state.agent_cap() + 31) / 32, _pad0: 0, _pad1: 0 }".into(),
        cfg_struct_decl:
            "#[repr(C)]\n\
             #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
             pub struct FusedMaskCfg {\n\
             \x20   pub agent_cap: u32,\n\
             \x20   pub num_mask_words: u32,\n\
             \x20   pub _pad0: u32,\n\
             \x20   pub _pad1: u32,\n\
             }".into(),
        bindings: vec![
            KernelBinding {
                slot: 0,
                name: "agents".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("agents".into()),
            },
            KernelBinding {
                slot: 1,
                name: "mask_bitmaps".into(),
                access: AccessMode::AtomicStorage,
                wgsl_ty: "u32".into(),
                bg_source: BgSource::Transient("mask_bitmaps".into()),
            },
            KernelBinding {
                slot: 2,
                name: "sim_cfg".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("sim_cfg".into()),
            },
            KernelBinding {
                slot: 3,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: "FusedMaskCfg".into(),
                bg_source: BgSource::Cfg,
            },
        ],
    }
}

/// Public accessor — used by `emit_mask_wgsl` to derive the WGSL
/// `@group(0)` declarations from the same spec the Rust BGL is built
/// from.
pub fn fused_mask_kernel_spec() -> KernelSpec {
    fused_mask_spec()
}

/// Emit `engine_gpu_rules/src/fused_mask.rs` — spec-driven module +
/// the BGL helpers (`bgl_storage`, `bgl_uniform`) appended.
pub fn emit_fused_mask_rs() -> String {
    let mut out = emit_kernel_module_rs(&fused_mask_spec());

    // Append the BGL helpers — every other kernel imports these via
    // `crate::fused_mask::bgl_*`. They live here because fused_mask
    // is the canonical "first" kernel emitted alphabetically.
    writeln!(out).unwrap();
    writeln!(out, "pub(crate) fn bgl_storage(b: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Storage {{ read_only }},").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub(crate) fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Uniform,").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Legacy entry point — preserved for any straggler caller. The body
/// now delegates to the spec-driven path.
#[deprecated(note = "use emit_fused_mask_rs() — kept for transitional callers")]
#[allow(dead_code)]
pub fn emit_fused_mask_rs_legacy() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_mask_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct FusedMaskKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct FusedMaskBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct FusedMaskCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub num_mask_words: u32,").unwrap();
    writeln!(out, "    pub _pad0: u32,").unwrap();
    writeln!(out, "    pub _pad1: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"fused_mask.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for FusedMaskKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = FusedMaskBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = FusedMaskCfg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                bgl_storage(0, true),  // agents (read)").unwrap();
    writeln!(out, "                bgl_storage(1, false), // mask_bitmaps (atomicOr)").unwrap();
    writeln!(out, "                bgl_storage(2, true),  // sim_cfg (read)").unwrap();
    writeln!(out, "                bgl_uniform(3),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_fused_masks\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> FusedMaskCfg {{").unwrap();
    writeln!(out, "        let agent_cap = state.agent_cap();").unwrap();
    writeln!(out, "        // Mask layout: ceil(agent_cap/32) words per mask × N masks.").unwrap();
    writeln!(out, "        let words_per_mask = (agent_cap + 31) / 32;").unwrap();
    writeln!(out, "        FusedMaskCfg {{").unwrap();
    writeln!(out, "            agent_cap,").unwrap();
    writeln!(out, "            num_mask_words: words_per_mask,").unwrap();
    writeln!(out, "            _pad0: 0, _pad1: 0,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> FusedMaskBindings<'a> {{").unwrap();
    writeln!(out, "        FusedMaskBindings {{").unwrap();
    writeln!(out, "            agents:       sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps: sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            sim_cfg:      sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &FusedMaskBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fused_mask::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        let wg = (agent_cap + 63) / 64;").unwrap();
    writeln!(out, "        pass.dispatch_workgroups(wg, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub(crate) fn bgl_storage(b: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Storage {{ read_only }},").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub(crate) fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "    wgpu::BindGroupLayoutEntry {{").unwrap();
    writeln!(out, "        binding: b,").unwrap();
    writeln!(out, "        visibility: wgpu::ShaderStages::COMPUTE,").unwrap();
    writeln!(out, "        ty: wgpu::BindingType::Buffer {{").unwrap();
    writeln!(out, "            ty: wgpu::BufferBindingType::Uniform,").unwrap();
    writeln!(out, "            has_dynamic_offset: false,").unwrap();
    writeln!(out, "            min_binding_size: None,").unwrap();
    writeln!(out, "        }},").unwrap();
    writeln!(out, "        count: None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Emit `engine_gpu_rules/src/mask_unpack.rs` — agents SoA → bitmap pack.
pub fn emit_mask_unpack_rs() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_mask_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MaskUnpackKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct MaskUnpackBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents_soa: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub agents_input: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct MaskUnpackCfg {{ pub agent_cap: u32, pub _pad: [u32; 3] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"mask_unpack.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for MaskUnpackKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = MaskUnpackBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = MaskUnpackCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::bgl\"),").unwrap();
    writeln!(out, "            entries: &[crate::fused_mask::bgl_storage(0, false), crate::fused_mask::bgl_storage(1, true), crate::fused_mask::bgl_uniform(2)],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_mask_unpack\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> MaskUnpackCfg {{").unwrap();
    writeln!(out, "        MaskUnpackCfg {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> MaskUnpackBindings<'a> {{").unwrap();
    writeln!(out, "        MaskUnpackBindings {{").unwrap();
    writeln!(out, "            agents_soa:   sources.external.agents,").unwrap();
    writeln!(out, "            agents_input: sources.transient.mask_unpack_agents_input,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &MaskUnpackBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents_soa.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.agents_input.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::mask_unpack::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
