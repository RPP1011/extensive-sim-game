//! Structured IR for kernel binding layouts. Every kernel's four
//! emit-time outputs (Rust BGL entries, WGSL binding decls, Rust
//! BindGroupEntry construction, Rust Bindings struct fields) are
//! lowered from a single `KernelSpec` walked once. Drift is
//! structurally impossible because all four outputs derive from the
//! same source.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessMode {
    /// `var<storage, read>` / `bgl_storage(N, true)`
    ReadStorage,
    /// `var<storage, read_write>` / `bgl_storage(N, false)`
    ReadWriteStorage,
    /// `var<storage, read_write>` with `array<atomic<u32>>` type — same
    /// BGL entry as ReadWriteStorage but the WGSL type is wrapped in
    /// `atomic<...>`. Distinct so the lowering can produce the right
    /// WGSL declaration.
    AtomicStorage,
    /// `var<uniform>` / `bgl_uniform(N)`
    Uniform,
}

/// Where the bind() construction reads its `&wgpu::Buffer` from.
/// Mirrors the BindingSources struct's resident/transient/external/pool
/// dispatch.
#[derive(Debug, Clone)]
pub enum BgSource {
    /// `sources.resident.<field>` — for resident-path buffers (agents,
    /// view storage primaries, scoring_table, etc.)
    Resident(String),
    /// `sources.transient.<field>` — per-tick scratch (mask_bitmaps,
    /// action_buf, cascade ring records, etc.)
    Transient(String),
    /// `sources.external.<field>` — host-supplied (sim_cfg, agents
    /// when loaded externally, etc.)
    External(String),
    /// `sources.pool.<field>` — pool-allocated transient (spatial
    /// grid cells, query results)
    #[allow(dead_code)]
    Pool(String),
    /// The per-dispatch cfg uniform passed as a separate argument to
    /// `bind(sources, cfg)`.
    Cfg,
    /// Optional view-storage handle resolved via a per-view accessor.
    /// `accessor` is the method name like `fold_view_standing_handles`;
    /// `tuple_idx` is which of the (primary, anchor, ids) elements to
    /// pick. Falls back to `primary` when None.
    #[allow(dead_code)]
    ViewHandle { accessor: String, tuple_idx: u8 },
    /// Bound to the same buffer as another binding (used when a kernel
    /// has optional anchor/ids slots that collapse to primary).
    AliasOf(String),
}

#[derive(Debug, Clone)]
pub struct KernelBinding {
    /// Binding slot number — 0-based, contiguous.
    pub slot: u32,
    /// WGSL identifier in the binding declaration (also the field name
    /// in the emitted Bindings struct).
    pub name: String,
    pub access: AccessMode,
    /// WGSL type string, e.g. "array<u32>", "array<atomic<u32>>",
    /// "FoldCfg". For atomic storage this should be the inner type
    /// (the `atomic<>` wrapper is added by the lowering).
    pub wgsl_ty: String,
    pub bg_source: BgSource,
}

#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// snake_case kernel name, e.g. "fold_standing".
    pub name: String,
    /// PascalCase, derived from name.
    pub pascal: String,
    /// WGSL entry-point function name, e.g. "cs_fold_standing".
    pub entry_point: String,
    /// Rust struct name for the cfg uniform, e.g. "FoldStandingCfg".
    pub cfg_struct: String,
    /// Rust expression building the cfg, e.g.
    /// "FoldStandingCfg { event_count: 0, tick: state.tick, _pad: [0; 2] }".
    /// Substituted into the emitted `build_cfg` body.
    pub cfg_build_expr: String,
    /// Rust struct definition for the cfg, including #[repr(C)] +
    /// #[derive(Pod, Zeroable)]. Goes into the kernel module verbatim.
    pub cfg_struct_decl: String,
    pub bindings: Vec<KernelBinding>,
}

impl KernelSpec {
    /// Validate the spec — slots must be contiguous from 0, cfg must
    /// be present exactly once when one of the bindings has access ==
    /// Uniform with `BgSource::Cfg`, AliasOf must point to an existing
    /// binding name. Returns an error string on any structural issue.
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), String> {
        // Slots contiguous and 0-based.
        for (i, b) in self.bindings.iter().enumerate() {
            if b.slot != i as u32 {
                return Err(format!(
                    "kernel {}: binding[{}] has slot={} but expected {}",
                    self.name, i, b.slot, i
                ));
            }
        }
        // AliasOf targets exist.
        let names: std::collections::HashSet<&str> =
            self.bindings.iter().map(|b| b.name.as_str()).collect();
        for b in &self.bindings {
            if let BgSource::AliasOf(target) = &b.bg_source {
                if !names.contains(target.as_str()) {
                    return Err(format!(
                        "kernel {}: binding `{}` aliases unknown name `{}`",
                        self.name, b.name, target
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Pascal-case helper used by every spec builder.
pub fn snake_to_pascal(s: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in s.chars() {
        if c == '_' {
            up = true;
            continue;
        }
        if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(slot: u32, name: &str, access: AccessMode, wgsl_ty: &str, src: BgSource) -> KernelBinding {
        KernelBinding {
            slot,
            name: name.into(),
            access,
            wgsl_ty: wgsl_ty.into(),
            bg_source: src,
        }
    }

    #[test]
    fn validate_accepts_clean_spec() {
        let spec = KernelSpec {
            name: "demo".into(),
            pascal: "Demo".into(),
            entry_point: "cs_demo".into(),
            cfg_struct: "DemoCfg".into(),
            cfg_build_expr: "DemoCfg::default()".into(),
            cfg_struct_decl: "pub struct DemoCfg;".into(),
            bindings: vec![
                b(0, "agents", AccessMode::ReadStorage, "array<u32>",
                  BgSource::External("agents".into())),
                b(1, "cfg", AccessMode::Uniform, "DemoCfg", BgSource::Cfg),
            ],
        };
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_contiguous_slots() {
        let spec = KernelSpec {
            name: "demo".into(),
            pascal: "Demo".into(),
            entry_point: "cs_demo".into(),
            cfg_struct: "DemoCfg".into(),
            cfg_build_expr: "DemoCfg::default()".into(),
            cfg_struct_decl: "pub struct DemoCfg;".into(),
            bindings: vec![
                b(0, "a", AccessMode::ReadStorage, "array<u32>",
                  BgSource::External("agents".into())),
                b(2, "cfg", AccessMode::Uniform, "DemoCfg", BgSource::Cfg),
            ],
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn validate_rejects_dangling_alias() {
        let spec = KernelSpec {
            name: "demo".into(),
            pascal: "Demo".into(),
            entry_point: "cs_demo".into(),
            cfg_struct: "DemoCfg".into(),
            cfg_build_expr: "DemoCfg::default()".into(),
            cfg_struct_decl: "pub struct DemoCfg;".into(),
            bindings: vec![
                b(0, "primary", AccessMode::ReadWriteStorage, "array<u32>",
                  BgSource::Resident("standing_primary".into())),
                b(1, "anchor", AccessMode::ReadWriteStorage, "array<u32>",
                  BgSource::AliasOf("not_a_field".into())),
            ],
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn snake_to_pascal_basic() {
        assert_eq!(snake_to_pascal("engaged_with"), "EngagedWith");
        assert_eq!(snake_to_pascal("standing"), "Standing");
        assert_eq!(snake_to_pascal("kin_fear"), "KinFear");
        assert_eq!(snake_to_pascal("fused_mask"), "FusedMask");
    }
}
