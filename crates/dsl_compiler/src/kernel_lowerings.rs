//! Pure-function lowerings from `KernelSpec` to the four kernel
//! emit-time outputs:
//!
//!   1. `lower_wgsl_bindings` — `@group(0) @binding(N)` declarations
//!      for the WGSL shader.
//!   2. `lower_rust_bgl_entries` — Rust `BindGroupLayoutEntry`
//!      constructors (one per slot).
//!   3. `lower_rust_bg_entries` — Rust `BindGroupEntry` constructors
//!      that resolve each slot's `BgSource` into a buffer expression.
//!   4. `lower_rust_bindings_struct_fields` — Rust struct fields for
//!      the emitted `<Pascal>Bindings<'a>` struct (one `&wgpu::Buffer`
//!      per non-Cfg, non-AliasOf binding).
//!
//! No shared state, no globals — every function takes a `&KernelSpec`
//! and returns a `String`. Each is tested in isolation; the kernel
//! emitter that walks all four together (`emit_kernel_module.rs`) can
//! also assume "if these four agree on the same KernelSpec, they
//! can't drift." Tests live alongside each lowering at the bottom of
//! this file.

use std::fmt::Write;

use crate::kernel_binding_ir::{AccessMode, BgSource, KernelBinding, KernelSpec};

/// Lower the binding spec to a WGSL declaration block. One
/// `@group(0) @binding(N)` line per binding, in slot order. The cfg
/// struct decl is NOT emitted here — kernel-module emission handles
/// that (it's a Rust-side type that mirrors a WGSL struct).
///
/// For atomic storage, the type is wrapped in `array<atomic<...>>` —
/// e.g. `wgsl_ty = "u32"` becomes `array<atomic<u32>>`. For non-atomic
/// storage the type is written verbatim.
pub fn lower_wgsl_bindings(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        let qualifier = match b.access {
            AccessMode::ReadStorage => "var<storage, read>",
            AccessMode::ReadWriteStorage => "var<storage, read_write>",
            AccessMode::AtomicStorage => "var<storage, read_write>",
            AccessMode::Uniform => "var<uniform>",
        };
        let ty_str = match b.access {
            AccessMode::AtomicStorage => format!("array<atomic<{}>>", b.wgsl_ty),
            _ => b.wgsl_ty.clone(),
        };
        writeln!(
            out,
            "@group(0) @binding({}) {} {}: {};",
            b.slot, qualifier, b.name, ty_str
        )
        .unwrap();
    }
    out
}

/// Lower the binding spec to Rust BGL entries — the body of the
/// `entries: &[ ... ]` slice in `device.create_bind_group_layout`.
/// Helper functions `bgl_storage(N, read_only)` and `bgl_uniform(N)`
/// live in `crate::fused_mask` (the canonical home is fused_mask
/// because it's the first kernel emitted alphabetically; every other
/// kernel module imports them via `crate::fused_mask::bgl_*`).
pub fn lower_rust_bgl_entries(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        let descriptor = match b.access {
            AccessMode::ReadStorage => format!("crate::fused_mask::bgl_storage({}, true)", b.slot),
            AccessMode::ReadWriteStorage | AccessMode::AtomicStorage => {
                format!("crate::fused_mask::bgl_storage({}, false)", b.slot)
            }
            AccessMode::Uniform => format!("crate::fused_mask::bgl_uniform({})", b.slot),
        };
        writeln!(out, "                {}, // {}", descriptor, b.name).unwrap();
    }
    out
}

/// Resolve a `BgSource` to the Rust expression that yields the
/// `&wgpu::Buffer` for that binding. Used by `lower_rust_bg_entries`
/// (to fill in `BindGroupEntry::resource`) and by the kernel-module
/// emitter (which builds the `bind()` body that populates the emitted
/// `<Pascal>Bindings<'a>` struct from `BindingSources`).
fn bg_source_to_buf_expr(src: &BgSource, _all: &[KernelBinding]) -> String {
    match src {
        BgSource::Resident(field) => format!("sources.resident.{}", field),
        BgSource::Transient(field) => format!("sources.transient.{}", field),
        BgSource::External(field) => format!("sources.external.{}", field),
        BgSource::Pool(field) => format!("sources.pool.{}", field),
        BgSource::Cfg => "cfg".to_string(),
        BgSource::ViewHandle { .. } => {
            // ViewHandle is intentionally NOT inlined here — the
            // kernel-module emitter for fold kernels handles the
            // `(primary, anchor, ids)` tuple destructuring once and
            // then references local `primary_buf` / `anchor_buf` /
            // `ids_buf` names. lower_rust_bg_entries treats
            // ViewHandle by referencing the appropriate local
            // (`view_storage_<role>`).
            "<view_handle>".to_string()
        }
        BgSource::AliasOf(target) => format!("<alias of {}>", target),
    }
}

/// Lower the binding spec to Rust `BindGroupEntry` constructors — the
/// body of the `entries: &[ ... ]` slice in
/// `device.create_bind_group(...)`. Inside `record()`, the buffers
/// have already been pulled out of `Bindings<'a>` (a struct field per
/// non-Cfg / non-AliasOf binding). This lowering writes one
/// `wgpu::BindGroupEntry { binding: N, resource: bindings.<field>.as_entire_binding() }`
/// per slot, resolving aliases and Cfg specially.
pub fn lower_rust_bg_entries(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        let buf_expr = match &b.bg_source {
            BgSource::Cfg => format!("bindings.{}", b.name),
            BgSource::AliasOf(target) => format!("bindings.{}", target),
            _ => format!("bindings.{}", b.name),
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

/// Lower the binding spec to the `<Pascal>Bindings<'a>` struct's field
/// list. One `pub <field>: &'a wgpu::Buffer,` per non-AliasOf binding.
/// The Cfg binding becomes `pub cfg: &'a wgpu::Buffer,` (passed in as
/// a separate arg to `bind(sources, cfg)` and then placed into the
/// struct's `cfg` field).
///
/// AliasOf bindings reuse another binding's buffer — they don't get
/// their own struct field; the `lower_rust_bg_entries` lowering routes
/// the alias to the target's field at BindGroupEntry-construction time.
pub fn lower_rust_bindings_struct_fields(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        if matches!(b.bg_source, BgSource::AliasOf(_)) {
            continue;
        }
        writeln!(out, "    pub {}: &'a wgpu::Buffer,", b.name).unwrap();
    }
    out
}

/// Lower the binding spec to the body of the `bind()` method's
/// `<Pascal>Bindings { ... }` initializer. One field per non-AliasOf
/// binding; Cfg fields receive the `cfg` argument; other sources
/// reference `sources.<bucket>.<field>` directly.
///
/// Note: this DOES NOT handle `ViewHandle` sources — those are
/// emitted at the kernel-module level so the `(primary, anchor, ids)`
/// tuple destructuring happens at the right scope. For now no kernel
/// in this refactor uses `BgSource::ViewHandle`; fold kernels use
/// custom prologue lines, see `emit_kernel_module::emit_fold_bind_body`.
pub fn lower_rust_bind_body(spec: &KernelSpec) -> String {
    let mut out = String::new();
    for b in &spec.bindings {
        if matches!(b.bg_source, BgSource::AliasOf(_)) {
            continue;
        }
        let value_expr = match &b.bg_source {
            BgSource::Cfg => "cfg".to_string(),
            BgSource::AliasOf(_) => unreachable!(),
            BgSource::ViewHandle { .. } => {
                // Caller-side handles ViewHandle. Emit a local var
                // reference; kernel-module emitter pre-defines the
                // local.
                format!("/* view_handle: {} */ todo!()", b.name)
            }
            other => bg_source_to_buf_expr(other, &spec.bindings),
        };
        writeln!(out, "            {}: {},", b.name, value_expr).unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_binding_ir::{AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec};

    fn make_demo_spec() -> KernelSpec {
        KernelSpec {
            name: "demo".into(),
            pascal: "Demo".into(),
            entry_point: "cs_demo".into(),
            cfg_struct: "DemoCfg".into(),
            cfg_build_expr: "DemoCfg::default()".into(),
            cfg_struct_decl: "pub struct DemoCfg;".into(),
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
                    name: "cfg".into(),
                    access: AccessMode::Uniform,
                    wgsl_ty: "DemoCfg".into(),
                    bg_source: BgSource::Cfg,
                },
            ],
            kind: KernelKind::Generic,
        }
    }

    #[test]
    fn lower_wgsl_bindings_emits_per_slot_decl() {
        let spec = make_demo_spec();
        let wgsl = lower_wgsl_bindings(&spec);
        assert!(wgsl.contains("@group(0) @binding(0) var<storage, read> agents: array<u32>;"));
        assert!(wgsl.contains(
            "@group(0) @binding(1) var<storage, read_write> mask_bitmaps: array<atomic<u32>>;"
        ));
        assert!(wgsl.contains("@group(0) @binding(2) var<uniform> cfg: DemoCfg;"));
    }

    #[test]
    fn lower_rust_bgl_entries_uses_helpers() {
        let spec = make_demo_spec();
        let rs = lower_rust_bgl_entries(&spec);
        assert!(rs.contains("crate::fused_mask::bgl_storage(0, true)"));
        assert!(rs.contains("crate::fused_mask::bgl_storage(1, false)"));
        assert!(rs.contains("crate::fused_mask::bgl_uniform(2)"));
    }

    #[test]
    fn lower_rust_bg_entries_resolves_to_bindings_fields() {
        let spec = make_demo_spec();
        let rs = lower_rust_bg_entries(&spec);
        assert!(rs.contains(
            "wgpu::BindGroupEntry { binding: 0, resource: bindings.agents.as_entire_binding() }"
        ));
        assert!(rs.contains(
            "wgpu::BindGroupEntry { binding: 1, resource: bindings.mask_bitmaps.as_entire_binding() }"
        ));
        assert!(rs.contains(
            "wgpu::BindGroupEntry { binding: 2, resource: bindings.cfg.as_entire_binding() }"
        ));
    }

    #[test]
    fn lower_rust_bindings_struct_fields_one_per_binding() {
        let spec = make_demo_spec();
        let rs = lower_rust_bindings_struct_fields(&spec);
        assert!(rs.contains("pub agents: &'a wgpu::Buffer,"));
        assert!(rs.contains("pub mask_bitmaps: &'a wgpu::Buffer,"));
        assert!(rs.contains("pub cfg: &'a wgpu::Buffer,"));
    }

    #[test]
    fn lower_rust_bind_body_resolves_sources() {
        let spec = make_demo_spec();
        let rs = lower_rust_bind_body(&spec);
        assert!(rs.contains("agents: sources.external.agents,"));
        assert!(rs.contains("mask_bitmaps: sources.transient.mask_bitmaps,"));
        assert!(rs.contains("cfg: cfg,"));
    }

    #[test]
    fn alias_of_skips_struct_field_but_routes_in_bg_entries() {
        // Build a spec where binding 1 aliases binding 0 — anchor
        // collapses to primary. The struct field for the alias should
        // be omitted; the BindGroupEntry for slot 1 should reuse the
        // primary buffer.
        let spec = KernelSpec {
            name: "fold_demo".into(),
            pascal: "FoldDemo".into(),
            entry_point: "cs_fold_demo".into(),
            cfg_struct: "FoldDemoCfg".into(),
            cfg_build_expr: "FoldDemoCfg::default()".into(),
            cfg_struct_decl: "pub struct FoldDemoCfg;".into(),
            bindings: vec![
                KernelBinding {
                    slot: 0,
                    name: "primary".into(),
                    access: AccessMode::ReadWriteStorage,
                    wgsl_ty: "array<u32>".into(),
                    bg_source: BgSource::Resident("standing_primary".into()),
                },
                KernelBinding {
                    slot: 1,
                    name: "anchor".into(),
                    access: AccessMode::ReadWriteStorage,
                    wgsl_ty: "array<u32>".into(),
                    bg_source: BgSource::AliasOf("primary".into()),
                },
            ],
            kind: KernelKind::Generic,
        };
        let fields = lower_rust_bindings_struct_fields(&spec);
        assert!(fields.contains("pub primary: &'a wgpu::Buffer,"));
        assert!(!fields.contains("pub anchor"));

        let entries = lower_rust_bg_entries(&spec);
        assert!(entries.contains(
            "wgpu::BindGroupEntry { binding: 0, resource: bindings.primary.as_entire_binding() }"
        ));
        assert!(entries.contains(
            "wgpu::BindGroupEntry { binding: 1, resource: bindings.primary.as_entire_binding() }"
        ));
    }
}
