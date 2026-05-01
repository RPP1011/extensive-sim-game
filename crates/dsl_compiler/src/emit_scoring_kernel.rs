//! Emits `engine_gpu_rules/src/scoring.rs` + `scoring_unpack.rs`.
//!
//! Today's hand-written `engine_gpu::scoring::ScoringKernel` builds its
//! BGL by walking `scoring_view_binding_order(specs)` and adding per-
//! view bindings. This emitter does the same walk but produces the
//! BGL-entry list as runtime-built code in the emitted Rust file —
//! mirroring the hand-written `BindGroupLayoutEntry` Vec construction
//! one-for-one.
//!
//! ## Binding-count delta with the hand-written wrapper
//!
//! The hand-written `engine_gpu::scoring::ScoringKernel::new` (see
//! `crates/engine_gpu/src/scoring.rs:519`) constructs the BGL with:
//!
//!   1. 5 core bindings (agent_data, mask_bitmaps, scoring_table,
//!      scoring_out, cfg).
//!   2. Per-view bindings (1, 2, or 3 per view depending on shape +
//!      topk).
//!   3. `sim_cfg` — read-only storage at
//!      `scoring_sim_cfg_binding(specs, atomic=true)`.
//!   4. `alive_bitmap` — read-only storage at slot 22
//!      (`engine_gpu::alive_bitmap::ALIVE_BITMAP_BINDING`).
//!   5. `spatial_within` candidate buffer — read-only storage at slot
//!      25 (`engine_gpu::scoring::SCORING_SPATIAL_WITHIN_BINDING`).
//!
//! Steps (4) + (5) are NOT yet emitted by this module — they correspond
//! to engine_gpu-resident inputs (alive bitmap is a per-tick scratch
//! produced by the cascade pre-step; spatial_within is produced by
//! `cascade_resident::run_spatial_resident_pre_scoring`). The wire-up
//! task (T7) adds these bindings as transient/resident fields on the
//! engine_gpu_rules side; this T6 emit produces the 5-core + per-view
//! + sim_cfg subset matching `emit_scoring_wgsl_atomic_views`'s actual
//! shader bindings under the post-T6/pre-T7 partial wire-up.

use std::fmt::Write;

/// Minimal `ViewStorageSpec` snapshot the emitter consumes. Real type
/// lives in `dsl_compiler::emit_view_wgsl::ViewStorageSpec`; this
/// version is the subset the xtask passes from a `Compilation`'s view
/// list. Shape strings are one of `"SlotMap"`, `"PairMapScalar"`, or
/// `"PairMapDecay"` (matches the hand-written wrapper's
/// `ViewShape` discriminants).
#[derive(Debug, Clone)]
pub struct ViewSpecForEmit {
    pub name: String,
    pub shape: String,
    pub topk: bool,
}

/// Build the 5-slot scoring kernel spec — agent_data, mask_bitmaps,
/// scoring_out, sim_cfg, cfg. View bindings deferred (the WGSL body
/// in `emit_scoring_wgsl::emit_scoring_wgsl_v2` doesn't read views).
pub fn scoring_kernel_spec() -> crate::kernel_binding_ir::KernelSpec {
    use crate::kernel_binding_ir::{AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec};
    KernelSpec {
        name: "scoring".into(),
        pascal: "Scoring".into(),
        entry_point: "cs_scoring".into(),
        cfg_struct: "ScoringCfg".into(),
        cfg_build_expr: "ScoringCfg { agent_cap: state.agent_cap(), num_actions: 16, tick: state.tick, _pad: 0 }".into(),
        cfg_struct_decl:
            "#[repr(C)]\n\
             #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
             pub struct ScoringCfg { pub agent_cap: u32, pub num_actions: u32, pub tick: u32, pub _pad: u32 }".into(),
        bindings: vec![
            KernelBinding {
                slot: 0,
                name: "agent_data".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("agents".into()),
            },
            KernelBinding {
                slot: 1,
                name: "mask_bitmaps".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Transient("mask_bitmaps".into()),
            },
            KernelBinding {
                slot: 2,
                name: "scoring_out".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Transient("action_buf".into()),
            },
            KernelBinding {
                slot: 3,
                name: "sim_cfg".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("sim_cfg".into()),
            },
            KernelBinding {
                slot: 4,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: "ScoringCfg".into(),
                bg_source: BgSource::Cfg,
            },
        ],
        kind: KernelKind::Generic,
    }
}

/// V2 emitter: produces a 5-slot stub scoring kernel with NO view
/// bindings. Matches a 5-slot stub WGSL (via `emit_scoring_wgsl_v2`).
/// The legacy `emit_scoring_rs` produces a dynamic-per-view BGL whose
/// bind() construction passes 1 buffer per view but BGL declares
/// 3 — drift that wgpu validation panics on.
///
/// Stub-quality scoring: writes a deterministic action_buf entry
/// (hold) per agent. Full scoring (consult views, atomic argmax)
/// is a follow-up; this v2 unblocks the integration test.
pub fn emit_scoring_rs_v2() -> String {
    crate::emit_kernel_module::emit_kernel_module_rs(&scoring_kernel_spec())
}

/// Legacy stringly-typed emitter — kept for transitional callers but
/// the spec-driven path is now canonical. Body retained verbatim
/// from before refactor.
#[deprecated(note = "use emit_scoring_rs_v2() — spec-driven via KernelSpec")]
#[allow(dead_code)]
fn emit_scoring_rs_v2_legacy() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel::emit_scoring_rs_v2.").unwrap();
    writeln!(out, "// 5-slot stub kernel matching scoring.wgsl v2 stub. View reads are TODO.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringBindings<'a> {{").unwrap();
    writeln!(out, "    pub agent_data: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ScoringCfg {{ pub agent_cap: u32, pub num_actions: u32, pub tick: u32, pub _pad: u32 }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"scoring.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ScoringKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ScoringBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ScoringCfg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, true),  // agent_data").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // mask_bitmaps").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // scoring_out").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(4),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_scoring\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ScoringCfg {{").unwrap();
    writeln!(out, "        ScoringCfg {{ agent_cap: state.agent_cap(), num_actions: 16, tick: state.tick, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ScoringBindings<'a> {{").unwrap();
    writeln!(out, "        ScoringBindings {{").unwrap();
    writeln!(out, "            agent_data:    sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps:  sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            scoring_out:   sources.transient.action_buf,").unwrap();
    writeln!(out, "            sim_cfg:       sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ScoringBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agent_data.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

#[allow(dead_code)]
pub fn emit_scoring_rs(specs: &[ViewSpecForEmit]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringBindings<'a> {{").unwrap();
    writeln!(out, "    pub agent_data: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub mask_bitmaps: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_table: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_buffers: &'a [&'a wgpu::Buffer],").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ScoringCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub num_actions: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "    pub _pad: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"scoring.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ScoringKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ScoringBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ScoringCfg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(0, true),  // agent_data").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(1, true),  // mask_bitmaps").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(2, true),  // scoring_table").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_storage(3, false), // scoring_out").unwrap();
    writeln!(out, "            crate::fused_mask::bgl_uniform(4),        // cfg").unwrap();
    writeln!(out, "        ];").unwrap();
    writeln!(out, "        let mut binding: u32 = 5;").unwrap();
    for spec in specs {
        match spec.shape.as_str() {
            "SlotMap" => {
                writeln!(out, "        // view '{}' shape=SlotMap (1 binding)", spec.name).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, true)); binding += 1;").unwrap();
            }
            "PairMapScalar" => {
                writeln!(out, "        // view '{}' shape=PairMapScalar (1 binding{})", spec.name, if spec.topk { ", topk: +2 anchors+ids" } else { "" }).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                if spec.topk {
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                }
            }
            "PairMapDecay" => {
                writeln!(out, "        // view '{}' shape=PairMapDecay (2 bindings{})", spec.name, if spec.topk { ", topk: +1 ids" } else { "" }).unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                if spec.topk {
                    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, false)); binding += 1;").unwrap();
                }
            }
            _ => {}
        }
    }
    writeln!(out, "        // sim_cfg sits past every view binding (matches emit_scoring_wgsl::scoring_sim_cfg_binding).").unwrap();
    writeln!(out, "        entries.push(crate::fused_mask::bgl_storage(binding, true));").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bgl\"),").unwrap();
    writeln!(out, "            entries: &entries,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_scoring\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ScoringCfg {{").unwrap();
    writeln!(out, "        ScoringCfg {{ agent_cap: state.agent_cap(), num_actions: 16, tick: state.tick, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ScoringBindings<'a> {{").unwrap();
    writeln!(out, "        ScoringBindings {{").unwrap();
    writeln!(out, "            agent_data:    sources.external.agents,").unwrap();
    writeln!(out, "            mask_bitmaps:  sources.transient.mask_bitmaps,").unwrap();
    writeln!(out, "            scoring_table: &sources.resident.scoring_table,").unwrap();
    writeln!(out, "            scoring_out:   sources.transient.action_buf,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "            sim_cfg:       sources.external.sim_cfg,").unwrap();
    writeln!(out, "            // view_buffers is the per-view list pulled from resident; the").unwrap();
    writeln!(out, "            // emitter walks `specs` to know which fields to slice. Per-view").unwrap();
    writeln!(out, "            // accessor `sources.resident.view_storage_<name>` gives the").unwrap();
    writeln!(out, "            // primary buffer; topk views require additional fields. The").unwrap();
    writeln!(out, "            // helper `sources.resident.scoring_view_buffers_slice()` (emitted").unwrap();
    writeln!(out, "            // alongside the resident context) returns the expected slice.").unwrap();
    writeln!(out, "            view_buffers: sources.resident.scoring_view_buffers_slice(),").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ScoringBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let mut bg_entries: Vec<wgpu::BindGroupEntry> = vec![").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agent_data.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 2, resource: bindings.scoring_table.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 3, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "            wgpu::BindGroupEntry {{ binding: 4, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "        ];").unwrap();
    writeln!(out, "        let mut next_b: u32 = 5;").unwrap();
    writeln!(out, "        for buf in bindings.view_buffers.iter() {{").unwrap();
    writeln!(out, "            bg_entries.push(wgpu::BindGroupEntry {{ binding: next_b, resource: buf.as_entire_binding() }});").unwrap();
    writeln!(out, "            next_b += 1;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        bg_entries.push(wgpu::BindGroupEntry {{ binding: next_b, resource: bindings.sim_cfg.as_entire_binding() }});").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &bg_entries,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Emit `engine_gpu_rules/src/apply_actions.rs` — the per-tick "apply
/// chosen actions" kernel that reads the scoring output, mutates agent
/// state (HP / alive / shield), and emits damage/heal events into the
/// cascade A-ring.
///
/// ## Binding shape (IR-walk)
///
/// The apply-row in `assets/sim/` reads `scoring_out` + `agent_data`,
/// writes back into `agent_data`, and appends events into the
/// cascade-physics A-ring. From an IR perspective the kernel needs:
///
///   1. `agents` (rw) — agent SoA (read for self/target lookup, write
///      for HP / alive deltas).
///   2. `scoring_out` (ro) — chosen-action-per-agent buffer produced by
///      `cs_scoring`.
///   3. `event_ring_records` (rw) — A-ring record array.
///   4. `event_ring_tail` (rw atomic) — A-ring monotonic counter.
///   5. `sim_cfg` (ro storage) — shared world scalars (tick + attack
///      range).
///   6. `cfg` (uniform) — per-dispatch agent_cap + ring capacity.
///
/// = 6 bindings total, well under the 16-per-group cap.
///
/// ## Divergence from `engine_gpu::apply_actions::ApplyActionsKernel`
///
/// The hand-written wrapper also uses 6 bindings, but with the cfg
/// uniform at slot 2 (between scoring and event_ring) and sim_cfg at
/// slot 5. The emitted layout puts cfg last (slot 5) and sim_cfg before
/// it (slot 4) — matching the convention emit_scoring_kernel /
/// emit_mask_kernel use ("scratch first, world-scalars second, cfg
/// last"). Because the WGSL produced by xtask alongside this Rust
/// module matches the emitted Rust BGL slot-for-slot, the pair is
/// internally consistent; the engine_gpu wire-up is feature-gated
/// behind `engine_gpu_emitted_apply_actions_dispatch` (off by default)
/// so the hand-written kernel keeps running until T16 retires it.
pub fn emit_apply_actions_rs() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ApplyActionsKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ApplyActionsBindings<'a> {{").unwrap();
    writeln!(out, "    pub agents: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub scoring_out: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_records: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_ring_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ApplyActionsCfg {{").unwrap();
    writeln!(out, "    pub agent_cap: u32,").unwrap();
    writeln!(out, "    pub tick: u32,").unwrap();
    writeln!(out, "    pub event_ring_capacity: u32,").unwrap();
    writeln!(out, "    pub _pad: u32,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"apply_actions.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ApplyActionsKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ApplyActionsBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ApplyActionsCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(0, false), // agents (rw)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(1, true),  // scoring_out").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(2, false), // event_ring_records").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(3, false), // event_ring_tail (atomic)").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_storage(4, true),  // sim_cfg").unwrap();
    writeln!(out, "                crate::fused_mask::bgl_uniform(5),        // cfg").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_apply_actions\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ApplyActionsCfg {{").unwrap();
    writeln!(out, "        ApplyActionsCfg {{ agent_cap: state.agent_cap(), tick: state.tick, event_ring_capacity: 4096, _pad: 0 }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ApplyActionsBindings<'a> {{").unwrap();
    writeln!(out, "        ApplyActionsBindings {{").unwrap();
    writeln!(out, "            agents:             sources.external.agents,").unwrap();
    writeln!(out, "            scoring_out:        sources.transient.action_buf,").unwrap();
    writeln!(out, "            event_ring_records: &sources.pingpong.events_a_records,").unwrap();
    writeln!(out, "            event_ring_tail:    &sources.pingpong.events_a_tail,").unwrap();
    writeln!(out, "            sim_cfg:            sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ApplyActionsBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agents.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.scoring_out.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.event_ring_records.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 3, resource: bindings.event_ring_tail.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 4, resource: bindings.sim_cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 5, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::apply_actions::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn emit_scoring_unpack_rs() -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_scoring_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringUnpackKernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub struct ScoringUnpackBindings<'a> {{").unwrap();
    writeln!(out, "    pub agent_data: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub agents_input: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[repr(C)]").unwrap();
    writeln!(out, "#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]").unwrap();
    writeln!(out, "pub struct ScoringUnpackCfg {{ pub agent_cap: u32, pub _pad: [u32; 3] }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "const SHADER_SRC: &str = include_str!(\"scoring_unpack.wgsl\");").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl crate::Kernel for ScoringUnpackKernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = ScoringUnpackBindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = ScoringUnpackCfg;").unwrap();
    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::bgl\"),").unwrap();
    writeln!(out, "            entries: &[crate::fused_mask::bgl_storage(0, false), crate::fused_mask::bgl_storage(1, true), crate::fused_mask::bgl_uniform(2)],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl],").unwrap();
    writeln!(out, "            push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl),").unwrap();
    writeln!(out, "            module: &shader,").unwrap();
    writeln!(out, "            entry_point: Some(\"cs_scoring_unpack\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(),").unwrap();
    writeln!(out, "            cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> ScoringUnpackCfg {{").unwrap();
    writeln!(out, "        ScoringUnpackCfg {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> ScoringUnpackBindings<'a> {{").unwrap();
    writeln!(out, "        ScoringUnpackBindings {{").unwrap();
    writeln!(out, "            agent_data:   sources.external.agents,").unwrap();
    writeln!(out, "            agents_input: sources.transient.scoring_unpack_agents_input,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &ScoringUnpackBindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 0, resource: bindings.agent_data.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 1, resource: bindings.agents_input.as_entire_binding() }},").unwrap();
    writeln!(out, "                wgpu::BindGroupEntry {{ binding: 2, resource: bindings.cfg.as_entire_binding() }},").unwrap();
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::scoring_unpack::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
