//! Regression: fold kernels' `cfg.event_count` is snapshotted from
//! the live GPU `event_tail` BEFORE clear_tail_in zeroes it for the
//! current tick.
//!
//! Pre-fix, fold kernels' per-tick cfg uniform
//! was `[agent_count, tick, 1, agent_count]` with `event_count =
//! agent_count` (1024 typical). Folds then walked the FULL
//! `agent_count` slots of the unified `event_ring`, including stale
//! records from prior ticks. Producer atomicAdd reordering varies
//! which fresh events overwrite stale ones, so per-run results drift
//! (forest_fire_pin observed `max |Δ| ≈ 470 / 1024` slots).
//!
//! The fix: at the TOP of step(), AFTER encoder creation but BEFORE
//! `event_ring.clear_tail_in(&mut encoder)`, copy the live GPU
//! `event_tail` (which still holds the prior-tick producer count
//! since clear hasn't run yet) into each fold consumer's
//! `cfg.event_count` slot. Folds then walk exactly `0..prior_tail`,
//! seeing only the records the prior tick's producers actually wrote.
//!
//! Indirect chronicle consumers (`physics_Apply*`-shape) take a
//! separate cfg-copy at PER-DISPATCH time from the LIVE event_tail
//! — that's correct for in-tick chronicle semantics where producers
//! and consumers run in the same tick. Folds need prior-tick
//! semantics; the two copies coexist.
//!
//! This test pins the codegen by constructing a synthetic fold-shape
//! `KernelSpec` (binds `event_ring`, `event_tail`, `cfg`, and a
//! `view_storage_primary` write target — the shape every emitted
//! `fold_<view>` kernel takes) and asserting the synthesized
//! `runtime_core.rs` contains the snapshot block in step() before
//! clear_tail_in.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::kernel_binding_ir::{
    AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

fn fold_shape_kernel_spec(name: &str) -> KernelSpec {
    // Mirrors the binding shape every `fold_<view>` kernel takes after
    // ViewFold synthesis: event_ring (atomic for the producer side of a
    // re-emit) + event_tail (read for bounds + atomic for in-step
    // emits) + sim_cfg + cfg + view_storage_primary write target.
    // Slot ordering matches `cg::emit::program::compose_view_fold_*`.
    let bindings = vec![
        KernelBinding {
            slot: 0,
            name: "event_ring".into(),
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
            bg_source: BgSource::Resident("event_ring_ring".into()),
        },
        KernelBinding {
            slot: 1,
            name: "event_tail".into(),
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
            bg_source: BgSource::Resident("event_ring_tail".into()),
        },
        KernelBinding {
            slot: 2,
            name: "sim_cfg".into(),
            access: AccessMode::Uniform,
            wgsl_ty: "SimCfg".into(),
            bg_source: BgSource::External("sim_cfg".into()),
        },
        KernelBinding {
            slot: 3,
            name: "cfg".into(),
            access: AccessMode::Uniform,
            wgsl_ty: "FoldCfg".into(),
            bg_source: BgSource::Cfg,
        },
        KernelBinding {
            slot: 4,
            name: "view_storage_primary".into(),
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
            bg_source: BgSource::Resident("view_storage_primary".into()),
        },
    ];
    KernelSpec {
        name: name.to_string(),
        pascal: snake_to_pascal(name),
        entry_point: format!("cs_{name}"),
        cfg_struct: format!("{}Cfg", snake_to_pascal(name)),
        cfg_build_expr: format!(
            "{}Cfg {{ event_count: 0, tick: 0, _pad: [0; 2] }}",
            snake_to_pascal(name)
        ),
        cfg_struct_decl: format!(
            "#[repr(C)]\n#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\npub struct {}Cfg {{ pub event_count: u32, pub tick: u32, pub _pad: [u32; 2] }}\n",
            snake_to_pascal(name)
        ),
        bindings,
        kind: KernelKind::ViewFold,
        y_dim_override: None,
        runtime_cfg_fields: Vec::new(),
    }
}

fn snake_to_pascal(s: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in s.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn fold_step_snapshot_copies_event_tail_into_cfg_event_count() {
    let mut artifacts = EmittedArtifacts::default();
    let spec = fold_shape_kernel_spec("fold_my_view");
    artifacts.kernel_index.push(spec.name.clone());
    // Empty rust_files entry → the synthesizer takes the
    // Bindings-direct path (no `from_context_with_extras`), which is
    // the production fold-dispatch path.
    artifacts
        .rust_files
        .insert(format!("{}.rs", spec.name), String::new());
    artifacts.kernel_specs.push(spec);

    let out = dsl_compiler::build_helper::synthesize_runtime_core_a2(
        "fold_cfg_copy_smoke",
        &artifacts,
        &[],
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &[],
        None,
        &[],
        false,
        false, // binds_navgrid
        // CRITICAL: empty indirect_consumer_kernel_names — the fold
        // kernel goes through the `Kernel(...)` arm, NOT the indirect
        // arm. Pre-fix this guaranteed no cfg copy. Post-fix the
        // event-ring binding triggers (a) the side-buffer alloc, (b)
        // a snapshot encoder + submit at the top of step(), and (c) a
        // copy from the side buffer into each fold's cfg.event_count
        // slot inside the main encoder.
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
        dsl_compiler::cg::lower::DebugDepth::Off,
    );

    // (a) Side buffer is allocated + threaded through the struct.
    assert!(
        out.contains("pub prev_event_tail_buf: wgpu::Buffer,"),
        "missing prev_event_tail_buf field on GeneratedRuntime.\n\
         --- generated runtime_core.rs ---\n{out}",
    );
    assert!(
        out.contains("let prev_event_tail_buf = gpu.device.create_buffer"),
        "missing prev_event_tail_buf alloc in try_new.\n\
         --- generated runtime_core.rs ---\n{out}",
    );

    // (b) Snapshot encoder + submit at top of step(). The encoder
    // copies event_tail into prev_event_tail_buf and submits BEFORE
    // any host queue.write_buffer overwrites the GPU tail. wgpu
    // guarantees queue writes land before encoder commands within
    // the SAME submit, so the separate-submit pattern is the only
    // way to capture the prior-tick GPU producer count.
    let snapshot_block = "snap_encoder.copy_buffer_to_buffer(\n                \
                          self.event_ring.tail(),\n                \
                          0,\n                \
                          &self.prev_event_tail_buf,\n                \
                          0,\n                \
                          4,\n            );";
    assert!(
        out.contains(snapshot_block),
        "missing prev-tick tail snapshot block.\n\
         Expected substring:\n{snapshot_block}\n\n\
         --- generated runtime_core.rs ---\n{out}",
    );
    assert!(
        out.contains("self.gpu.queue.submit(Some(snap_encoder.finish()));"),
        "missing snap_encoder submit.\n\
         --- generated runtime_core.rs ---\n{out}",
    );

    // (c) Main encoder copies prev_event_tail_buf → cfg_<fold>_buf
    // slot 0. This MUST happen AFTER the per-tick cfg_bytes
    // queue.write_buffer (which sets all four slots to
    // [agent_count, tick, 1, agent_count]) so it overwrites slot 0
    // with the snapshot value. Encoder commands run after queue
    // writes within a submit, so source order doesn't matter — but
    // the copy MUST be in the main encoder, not the snap encoder.
    let cfg_copy = "encoder.copy_buffer_to_buffer(\n            \
                    &self.prev_event_tail_buf,\n            \
                    0,\n            \
                    &self.cfg_fold_my_view_buf,\n            \
                    0,\n            \
                    4,\n        );";
    assert!(
        out.contains(cfg_copy),
        "missing main-encoder copy from prev_event_tail_buf into fold cfg.\n\
         Expected substring:\n{cfg_copy}\n\n\
         --- generated runtime_core.rs ---\n{out}",
    );

    // Order pin: the snapshot submit MUST land before clear_tail_in.
    // (clear_tail_in is an encoder command that zeroes the GPU tail
    // for the current tick's producers; if the snapshot ran after,
    // it would capture 0 instead of the prior-tick producer count.)
    let snap_submit_pos = out
        .find("self.gpu.queue.submit(Some(snap_encoder.finish()));")
        .expect("snap_encoder submit missing");
    let clear_pos = out
        .find("self.event_ring.clear_tail_in(&mut encoder);")
        .expect("clear_tail_in missing in fold-only fixture");
    assert!(
        snap_submit_pos < clear_pos,
        "snapshot submit must precede clear_tail_in; got snap_submit_pos={snap_submit_pos}, clear_pos={clear_pos}",
    );
}

#[test]
fn non_event_ring_kernel_does_not_get_snapshot_copy() {
    // Negative pin: a kernel that does NOT bind event_ring/event_tail
    // (e.g. a pure per-agent physics kernel) must NOT get a snapshot
    // copy — that would wrongly overwrite a per-agent kernel's
    // cfg.agent_cap slot with a runtime tail value.
    let mut artifacts = EmittedArtifacts::default();
    let bindings = vec![
        KernelBinding {
            slot: 0,
            name: "agent_hp".into(),
            access: AccessMode::ReadWriteStorage,
            wgsl_ty: "f32".into(),
            bg_source: BgSource::Resident("agent_hp".into()),
        },
        KernelBinding {
            slot: 1,
            name: "cfg".into(),
            access: AccessMode::Uniform,
            wgsl_ty: "PerAgentCfg".into(),
            bg_source: BgSource::Cfg,
        },
    ];
    let spec = KernelSpec {
        name: "physics_per_agent".into(),
        pascal: "PhysicsPerAgent".into(),
        entry_point: "cs_physics_per_agent".into(),
        cfg_struct: "PhysicsPerAgentCfg".into(),
        cfg_build_expr: "PhysicsPerAgentCfg { agent_cap: 0, tick: 0, seed: 0, _pad: 0 }".into(),
        cfg_struct_decl: "#[repr(C)]\n#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\npub struct PhysicsPerAgentCfg { pub agent_cap: u32, pub tick: u32, pub seed: u32, pub _pad: u32 }\n".into(),
        bindings,
        kind: KernelKind::Generic,
        y_dim_override: None,
        runtime_cfg_fields: Vec::new(),
    };
    artifacts.kernel_index.push(spec.name.clone());
    artifacts
        .rust_files
        .insert(format!("{}.rs", spec.name), String::new());
    artifacts.kernel_specs.push(spec);

    let out = dsl_compiler::build_helper::synthesize_runtime_core_a2(
        "no_event_ring_smoke",
        &artifacts,
        &[],
        &[],
        &std::collections::BTreeMap::new(),
        &[],
        &[],
        None,
        &[],
        false,
        false, // binds_navgrid
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
        dsl_compiler::cg::lower::DebugDepth::Off,
    );

    // No fold consumers → no prev_event_tail_buf field, no
    // snap_encoder, no cfg copy from a side buffer.
    assert!(
        !out.contains("prev_event_tail_buf"),
        "non-event-ring kernel must NOT trigger prev_event_tail_buf.\n\
         --- generated runtime_core.rs ---\n{out}",
    );
    assert!(
        !out.contains("snap_encoder"),
        "non-event-ring kernel must NOT emit snapshot encoder.\n\
         --- generated runtime_core.rs ---\n{out}",
    );
}
