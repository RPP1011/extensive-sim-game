//! Emits per-view `Fold<Pascal>Kernel` Rust modules from a
//! `KernelSpec` walked once. The four emit-time outputs (BGL entries,
//! BindGroupEntry construction, Bindings-struct fields, WGSL bindings)
//! all derive from the same spec — drift is structurally impossible.
//!
//! Fold kernels deviate from the generic spec-driven module emitter
//! (`emit_kernel_module::emit_kernel_module_rs`) in one place: the
//! `Bindings<'a>` struct fields are typed `Option<&'a wgpu::Buffer>`
//! for the anchor/ids slots so views without explicit anchor/ids
//! buffers can pass `None` and have the `record()` body fall back to
//! the primary buffer. The BGL slot still has to exist (wgpu pipeline
//! creation enforces it), so the fallback rebinds primary to keep the
//! contract live without allocating extra zero-byte placeholders.
//!
//! ## Per-storage-shape binding spec
//!
//! - **`SymmetricPairTopK { k }`** — 2 storage layers (slots, counts).
//!   The WGSL emitter declares: slot 0 = `view_<name>_slots`
//!   (read_write, `array<<Edge>>`); slot 1 = `view_<name>_counts`
//!   (atomic). Followed by per-handler `events_<ev>` (read) +
//!   `cfg_<ev>` (uniform) pairs.
//! - **`PerEntityRing { k }`** — 2 storage layers (ring, cursors).
//!   slot 0 = `view_<name>_rings`; slot 1 = `view_<name>_cursors`
//!   (atomic). Same per-handler pair as above.
//! - **Other shapes (PairMap, SlotMap, PerEntityTopK)** — fall back
//!   to the 7-binding generic stub: event_ring + event_tail +
//!   view_storage_primary + anchor + ids + sim_cfg + cfg.
//!
//! BGL slot allocation: storage layers first (slot 0..N), then
//! per-handler (events + cfg) pairs.

use std::fmt::Write;

use crate::ir::{StorageHint, ViewBodyIR, ViewIR, ViewKind};
use crate::kernel_binding_ir::{snake_to_pascal, AccessMode, BgSource, KernelBinding, KernelSpec};
use crate::kernel_lowerings::lower_rust_bgl_entries;

/// Event-pattern name → snake_case (matches the WGSL emitter's
/// `event_snake` derivation). The DSL pattern names are already snake
/// for our event types; this is mostly a guard against PascalCase
/// inputs.
fn event_snake(name: &str) -> String {
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

/// Returns the event handler patterns this view folds. Empty for non-
/// `Fold`-bodied views (lazy `Expr` views — they don't get a fold
/// kernel anyway).
fn fold_event_names(view: &ViewIR) -> Vec<String> {
    match &view.body {
        ViewBodyIR::Fold { handlers, .. } => {
            handlers.iter().map(|h| h.pattern.name.clone()).collect()
        }
        ViewBodyIR::Expr(_) => Vec::new(),
    }
}

/// Build the `KernelSpec` for a view's fold kernel. Picks the
/// per-shape binding layout from `view.kind`. The spec models the
/// "anchor/ids fall back to primary" pattern via `BgSource::AliasOf`
/// for views whose resident accessor returns `None` for those slots.
///
/// The `Bindings<'a>` struct's `Option<&wgpu::Buffer>` fields are
/// emitted in `emit_view_fold_rs` directly because the generic
/// kernel-module emitter assumes plain `&wgpu::Buffer` fields. The
/// BGL entries and BindGroupEntry construction reuse the lowerings.
pub fn fold_kernel_spec(view: &ViewIR) -> KernelSpec {
    let view_snake = view.name.clone();
    let pascal = snake_to_pascal(&view_snake);
    let mut bindings: Vec<KernelBinding> = Vec::new();
    let mut slot: u32 = 0;

    let mut shape_aware = true; // true for sym_pair_topk + per_entity_ring
    match view.kind {
        ViewKind::Materialized(StorageHint::SymmetricPairTopK { .. }) => {
            // slot 0: view_<name>_slots (rw)
            bindings.push(KernelBinding {
                slot,
                name: format!("view_{view_snake}_slots"),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(), // overridden by per-shape WGSL body
                bg_source: BgSource::Resident("__view_handle_primary".into()),
            });
            slot += 1;
            // slot 1: view_<name>_counts (atomic). Per-shape WGSL emits
            // `array<atomic<u32>>` for this slot. Marking it AtomicStorage
            // so the WGSL lowerer wraps the type correctly when we use
            // it (we don't here — the body is per-shape WGSL).
            bindings.push(KernelBinding {
                slot,
                name: format!("view_{view_snake}_counts"),
                access: AccessMode::AtomicStorage,
                wgsl_ty: "u32".into(),
                bg_source: BgSource::Resident("__view_handle_anchor_or_primary".into()),
            });
            slot += 1;
        }
        ViewKind::Materialized(StorageHint::PerEntityRing { .. }) => {
            // slot 0: view_<name>_rings (rw)
            bindings.push(KernelBinding {
                slot,
                name: format!("view_{view_snake}_rings"),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("__view_handle_primary".into()),
            });
            slot += 1;
            // slot 1: view_<name>_cursors (atomic).
            bindings.push(KernelBinding {
                slot,
                name: format!("view_{view_snake}_cursors"),
                access: AccessMode::AtomicStorage,
                wgsl_ty: "u32".into(),
                bg_source: BgSource::Resident("__view_handle_anchor_or_primary".into()),
            });
            slot += 1;
        }
        _ => {
            // 7-binding generic stub for PairMap / SlotMap / PerEntityTopK.
            shape_aware = false;
            bindings.push(KernelBinding {
                slot: 0,
                name: "event_ring".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("batch_events_ring".into()),
            });
            bindings.push(KernelBinding {
                slot: 1,
                name: "event_tail".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("batch_events_tail".into()),
            });
            bindings.push(KernelBinding {
                slot: 2,
                name: "view_storage_primary".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("__view_handle_primary".into()),
            });
            bindings.push(KernelBinding {
                slot: 3,
                name: "view_storage_anchor".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("__view_handle_anchor_or_primary".into()),
            });
            bindings.push(KernelBinding {
                slot: 4,
                name: "view_storage_ids".into(),
                access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("__view_handle_ids_or_primary".into()),
            });
            bindings.push(KernelBinding {
                slot: 5,
                name: "sim_cfg".into(),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::External("sim_cfg".into()),
            });
            bindings.push(KernelBinding {
                slot: 6,
                name: "cfg".into(),
                access: AccessMode::Uniform,
                wgsl_ty: format!("Fold{pascal}Cfg"),
                bg_source: BgSource::Cfg,
            });
        }
    }

    // For shape-aware shapes (sym_pair_topk + per_entity_ring): the
    // WGSL emits one (events, cfg) pair per fold handler.
    if shape_aware {
        for ev in fold_event_names(view) {
            let ev_snake = event_snake(&ev);
            bindings.push(KernelBinding {
                slot,
                name: format!("events_{ev_snake}"),
                access: AccessMode::ReadStorage,
                wgsl_ty: "array<u32>".into(),
                bg_source: BgSource::Resident("batch_events_ring".into()),
            });
            slot += 1;
            bindings.push(KernelBinding {
                slot,
                name: format!("cfg_{ev_snake}"),
                access: AccessMode::Uniform,
                wgsl_ty: "FoldCfg".into(),
                bg_source: BgSource::Cfg,
            });
            slot += 1;
        }
    }

    KernelSpec {
        name: format!("fold_{view_snake}"),
        pascal: format!("Fold{pascal}"),
        entry_point: format!("cs_fold_{view_snake}"),
        cfg_struct: format!("Fold{pascal}Cfg"),
        cfg_build_expr: format!("Fold{pascal}Cfg {{ event_count: 0, tick: state.tick, _pad: [0; 2] }}"),
        cfg_struct_decl: format!(
            "#[repr(C)]\n\
             #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
             pub struct Fold{pascal}Cfg {{ pub event_count: u32, pub tick: u32, pub _pad: [u32; 2] }}"
        ),
        bindings,
    }
}

/// Emit `engine_gpu_rules/src/fold_<view>.rs`. Fold kernels deviate
/// from `emit_kernel_module_rs` only in their `Bindings<'a>` struct,
/// which uses `Option<&'a wgpu::Buffer>` for anchor/ids so resident
/// views without explicit secondary buffers can pass `None` and have
/// `record()` rebind primary into those slots.
pub fn emit_view_fold_rs(view: &ViewIR) -> String {
    let view_name = &view.name;
    let pascal = snake_to_pascal(view_name);
    let spec = fold_kernel_spec(view);

    // Pre-compute the lowered chunks. BGL + BindGroup entries are pure
    // string outputs; the Bindings struct fields and bind() body are
    // hand-rolled here because of the Option-typed fallback fields.
    let bgl_entries = lower_rust_bgl_entries(&spec);
    let bg_entries = render_fold_bg_entries(&spec);

    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler::emit_view_fold_kernel. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::binding_sources::BindingSources;").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "pub struct Fold{pascal}Kernel {{").unwrap();
    writeln!(out, "    pipeline: wgpu::ComputePipeline,").unwrap();
    writeln!(out, "    bgl: wgpu::BindGroupLayout,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Bindings struct — three semantic fields the bind body sets up,
    // regardless of the underlying spec:
    //   - event_ring / event_tail / sim_cfg / cfg (always live)
    //   - view_storage_primary (always Some)
    //   - view_storage_anchor / view_storage_ids (Optional; record()
    //     falls back to primary when None).
    //
    // The same three semantic fields cover both the 7-slot generic
    // stub and the shape-aware sym_pair_topk / per_entity_ring layouts;
    // the BGL itself comes from the spec, and the bind-group entries
    // route slots to fields based on each slot's BgSource.
    writeln!(out, "pub struct Fold{pascal}Bindings<'a> {{").unwrap();
    writeln!(out, "    pub event_ring: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub event_tail: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_storage_primary: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub view_storage_anchor: Option<&'a wgpu::Buffer>,").unwrap();
    writeln!(out, "    pub view_storage_ids: Option<&'a wgpu::Buffer>,").unwrap();
    writeln!(out, "    pub sim_cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "    pub cfg: &'a wgpu::Buffer,").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    out.push_str(&spec.cfg_struct_decl);
    writeln!(out).unwrap();
    writeln!(out).unwrap();

    writeln!(out, "const SHADER_SRC: &str = include_str!(\"fold_{view_name}.wgsl\");").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "impl crate::Kernel for Fold{pascal}Kernel {{").unwrap();
    writeln!(out, "    type Bindings<'a> = Fold{pascal}Bindings<'a>;").unwrap();
    writeln!(out, "    type Cfg = Fold{pascal}Cfg;").unwrap();

    writeln!(out, "    fn new(device: &wgpu::Device) -> Self {{").unwrap();
    writeln!(out, "        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::wgsl\"),").unwrap();
    writeln!(out, "            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::bgl\"),").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    out.push_str(&bgl_entries);
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pl\"),").unwrap();
    writeln!(out, "            bind_group_layouts: &[&bgl], push_constant_ranges: &[],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pipeline\"),").unwrap();
    writeln!(out, "            layout: Some(&pl), module: &shader, entry_point: Some(\"cs_fold_{view_name}\"),").unwrap();
    writeln!(out, "            compilation_options: Default::default(), cache: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        Self {{ pipeline, bgl }}").unwrap();
    writeln!(out, "    }}").unwrap();

    writeln!(out, "    fn build_cfg(&self, state: &engine::state::SimState) -> Fold{pascal}Cfg {{").unwrap();
    writeln!(out, "        Fold{pascal}Cfg {{ event_count: 0, tick: state.tick, _pad: [0; 2] }}").unwrap();
    writeln!(out, "    }}").unwrap();

    writeln!(out, "    fn bind<'a>(&'a self, sources: &'a BindingSources<'a>, cfg: &'a wgpu::Buffer) -> Fold{pascal}Bindings<'a> {{").unwrap();
    writeln!(out, "        let (primary, anchor, ids) = sources.resident.fold_view_{view_name}_handles();").unwrap();
    writeln!(out, "        Fold{pascal}Bindings {{").unwrap();
    writeln!(out, "            event_ring:           &sources.resident.batch_events_ring,").unwrap();
    writeln!(out, "            event_tail:           &sources.resident.batch_events_tail,").unwrap();
    writeln!(out, "            view_storage_primary: primary,").unwrap();
    writeln!(out, "            view_storage_anchor:  anchor,").unwrap();
    writeln!(out, "            view_storage_ids:     ids,").unwrap();
    writeln!(out, "            sim_cfg:              sources.external.sim_cfg,").unwrap();
    writeln!(out, "            cfg,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();

    writeln!(out, "    fn record(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, bindings: &Fold{pascal}Bindings<'_>, agent_cap: u32) {{").unwrap();
    writeln!(out, "        // Resolve view-handle slots: anchor/ids fall back to primary").unwrap();
    writeln!(out, "        // when the view doesn't expose dedicated buffers (the BGL slot").unwrap();
    writeln!(out, "        // still has to be live; rebinding primary is a no-op safe choice).").unwrap();
    writeln!(out, "        let primary_buf = bindings.view_storage_primary;").unwrap();
    writeln!(out, "        let anchor_buf = bindings.view_storage_anchor.unwrap_or(primary_buf);").unwrap();
    writeln!(out, "        let ids_buf = bindings.view_storage_ids.unwrap_or(primary_buf);").unwrap();
    writeln!(out, "        let _ = (anchor_buf, ids_buf);  // silence unused on shapes that don't bind anchor/ids").unwrap();
    writeln!(out, "        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::bg\"),").unwrap();
    writeln!(out, "            layout: &self.bgl,").unwrap();
    writeln!(out, "            entries: &[").unwrap();
    out.push_str(&bg_entries);
    writeln!(out, "            ],").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{").unwrap();
    writeln!(out, "            label: Some(\"engine_gpu_rules::fold_{view_name}::pass\"),").unwrap();
    writeln!(out, "            timestamp_writes: None,").unwrap();
    writeln!(out, "        }});").unwrap();
    writeln!(out, "        pass.set_pipeline(&self.pipeline);").unwrap();
    writeln!(out, "        pass.set_bind_group(0, &bg, &[]);").unwrap();
    writeln!(out, "        pass.dispatch_workgroups((agent_cap + 63) / 64, 1, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Like `lower_rust_bg_entries` but with fold-specific BgSource
/// resolution for the synthetic `__view_handle_*` markers and the
/// `batch_events_ring/tail` resident handles that go through
/// `&sources.resident.<field>` (note: those are field accesses, not
/// the standard `sources.resident.<field>` direct access).
fn render_fold_bg_entries(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        let buf_expr = match &b.bg_source {
            BgSource::Resident(field) => match field.as_str() {
                "__view_handle_primary" => "primary_buf".to_string(),
                "__view_handle_anchor_or_primary" => "anchor_buf".to_string(),
                "__view_handle_ids_or_primary" => "ids_buf".to_string(),
                "batch_events_ring" => "bindings.event_ring".to_string(),
                "batch_events_tail" => "bindings.event_tail".to_string(),
                other => panic!("fold spec used unexpected resident field `{other}`"),
            },
            BgSource::External(field) => match field.as_str() {
                "sim_cfg" => "bindings.sim_cfg".to_string(),
                other => panic!("fold spec used unexpected external field `{other}`"),
            },
            BgSource::Cfg => "bindings.cfg".to_string(),
            BgSource::Transient(_) | BgSource::Pool(_) | BgSource::ViewHandle { .. }
            | BgSource::AliasOf(_) => {
                panic!("fold spec used unsupported bg_source: {:?}", b.bg_source)
            }
        };
        writeln!(
            out,
            "                wgpu::BindGroupEntry {{ binding: {}, resource: {}.as_entire_binding() }},",
            b.slot, buf_expr
        )
        .unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrType, ViewBodyIR, ViewIR, ViewKind};
    use dsl_ast::ast::Span;

    fn dummy_view(name: &str, kind: ViewKind) -> ViewIR {
        use crate::ir::{IrExpr, IrExprNode};
        let zero_lit = IrExprNode {
            kind: IrExpr::LitInt(0),
            span: Span::dummy(),
        };
        ViewIR {
            name: name.into(),
            kind,
            params: vec![],
            return_ty: IrType::I32,
            decay: None,
            body: ViewBodyIR::Fold {
                initial: zero_lit,
                handlers: vec![],
                clamp: None,
            },
            annotations: vec![],
            span: Span::dummy(),
        }
    }

    fn dummy_view_pair_map(name: &str) -> ViewIR {
        dummy_view(name, ViewKind::Materialized(StorageHint::PairMap))
    }

    fn dummy_view_sym_pair_topk(name: &str) -> ViewIR {
        dummy_view(
            name,
            ViewKind::Materialized(StorageHint::SymmetricPairTopK { k: 8 }),
        )
    }

    fn dummy_view_per_entity_ring(name: &str) -> ViewIR {
        dummy_view(
            name,
            ViewKind::Materialized(StorageHint::PerEntityRing { k: 64 }),
        )
    }

    #[test]
    fn pair_map_view_yields_7_binding_stub_spec() {
        let spec = fold_kernel_spec(&dummy_view_pair_map("engaged_with"));
        assert_eq!(spec.bindings.len(), 7);
        assert_eq!(spec.bindings[6].name, "cfg");
        assert_eq!(spec.entry_point, "cs_fold_engaged_with");
        assert_eq!(spec.pascal, "FoldEngagedWith");
    }

    #[test]
    fn sym_pair_topk_view_yields_2_storage_layers() {
        let spec = fold_kernel_spec(&dummy_view_sym_pair_topk("standing"));
        // No fold handlers → just slots + counts.
        assert_eq!(spec.bindings.len(), 2);
        assert_eq!(spec.bindings[0].name, "view_standing_slots");
        assert_eq!(spec.bindings[1].name, "view_standing_counts");
        assert!(matches!(spec.bindings[1].access, AccessMode::AtomicStorage));
    }

    #[test]
    fn per_entity_ring_view_yields_2_storage_layers() {
        let spec = fold_kernel_spec(&dummy_view_per_entity_ring("memory"));
        assert_eq!(spec.bindings.len(), 2);
        assert_eq!(spec.bindings[0].name, "view_memory_rings");
        assert_eq!(spec.bindings[1].name, "view_memory_cursors");
        assert!(matches!(spec.bindings[1].access, AccessMode::AtomicStorage));
    }

    #[test]
    fn snake_to_pascal_basic() {
        assert_eq!(snake_to_pascal("engaged_with"), "EngagedWith");
        assert_eq!(snake_to_pascal("threat_level"), "ThreatLevel");
        assert_eq!(snake_to_pascal("standing"), "Standing");
        assert_eq!(snake_to_pascal("kin_fear"), "KinFear");
    }
}
