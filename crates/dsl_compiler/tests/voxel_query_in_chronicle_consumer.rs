//! Regression for Gap #2 of `among_us` (docs/architecture/gaps_among_us.md).
//!
//! `terrain.line_of_sight(a, b)` must be callable from a `@phase(post)`
//! chronicle-consumer kernel — the consumer's body lowers `terrain_line_of_sight(`
//! into the WGSL, the kernel composer (`cg::emit::kernel`) substring-scans
//! for that call, and synthesizes a `voxel_grid` storage binding so the
//! WGSL emit + Rust `Bindings` + host-side dispatch all surface the slot
//! the runtime fills via `KernelBindingsContext::voxel_grid`.
//!
//! Without this gate, hill_raid's defenders would silently never gate on
//! LoS in the consumer, and among_us's ApplyWitness can't filter
//! witness writes through cover. The fix wired the same binding
//! synthesis that the per-agent path already had (`hill_raid::DefenderFire`)
//! through the chronicle-consumer (PerEventEmit) path too, by making
//! the substring scan kernel-kind agnostic.
//!
//! What this test pins:
//!
//!   1. A `.sim` whose `@phase(post) physics ApplyWitness { on Damaged ... }`
//!      body calls `terrain.line_of_sight(self.pos, target.pos)` lowers
//!      cleanly — no parser / resolver / lowering / emit error.
//!   2. The resulting consumer kernel's `KernelSpec.bindings` contains
//!      a `voxel_grid` binding (the new wiring).
//!   3. The emitted Rust `from_context()` for that kernel routes
//!      `voxel_grid` through `BindingSource::VoxelGrid`
//!      (i.e. `ctx.voxel_grid.expect(...)`).
//!   4. Naga still parses every emitted WGSL file (no dangling refs).

use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};
use dsl_compiler::kernel_binding_ir::KernelKind;

const FIXTURE: &str = r#"
event Tick { }

@replayable @gpu_amenable
event Damaged {
  source: AgentId,
  target: AgentId,
  amount: f32,
}

entity Witness : Agent {
  pos: vec3,
  hp:  f32,
  mana: f32,
}

// Per-agent producer — drops a Damaged event each tick targeting self
// so the consumer below has something to react to. Body is intentionally
// trivial; the test focuses on the consumer's voxel binding.
@phase(per_agent)
physics StrikeSelf {
  on Tick {} {
    emit Damaged { source: self, target: self, amount: 1.0 }
  }
}

// Chronicle consumer with a `terrain.line_of_sight` gate inside a
// `for_each_agent` body. This is the among_us ApplyWitness shape: the
// consumer reacts to each Damaged event AND walks every agent slot to
// gate witness writes through cover. PRE-FIX, the consumer kernel's
// BGL omitted `voxel_grid`.
@phase(post)
physics ApplyWitnessLos {
  on Damaged { source: src, target: tgt, amount: a } {
    if (a > 0.0) {
      let src_pos = agents.pos(src);
      for_each_agent slot {
        let slot_pos = agents.pos(slot);
        if (terrain.line_of_sight(src_pos, slot_pos)) {
          agents.set_mana(slot, agents.mana(slot) + 1.0);
        }
      }
    }
  }
}
"#;

#[test]
fn chronicle_consumer_with_terrain_los_emits_voxel_grid_binding() {
    let program = dsl_compiler::parse(FIXTURE).expect("parse fixture");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve fixture");
    let cg = lower_compilation_to_cg(&comp).expect("lower to CG");
    let schedule_result = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let artifacts = emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit_cg_program succeeds for chronicle consumer with terrain LoS");

    // Find the consumer kernel — its name contains `ApplyWitnessLos`
    // (semantic_kernel_name preserves the physics-rule name in snake
    // case, so it'll show up as `physics_apply_witness_los` or similar
    // — match on the substring to be drift-tolerant).
    let consumer = artifacts
        .kernel_specs
        .iter()
        .find(|spec| spec.name.contains("ApplyWitnessLos"))
        .unwrap_or_else(|| {
            panic!(
                "no kernel matched substring `ApplyWitnessLos`; got: {:?}",
                artifacts
                    .kernel_specs
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
            )
        });

    // PerEventEmit is the kernel kind chronicle-consumer kernels are
    // stamped with (cg/emit/kernel.rs ~line 583). Pin this so a future
    // refactor that re-routes consumers through `Generic` doesn't
    // silently bypass the per-event-cfg shape.
    assert_eq!(
        consumer.kind,
        KernelKind::PerEventEmit,
        "chronicle-consumer kernel must be stamped PerEventEmit; got {:?}",
        consumer.kind
    );

    // The fix: the consumer's BGL now includes `voxel_grid`.
    let has_voxel_grid = consumer.bindings.iter().any(|b| b.name == "voxel_grid");
    assert!(
        has_voxel_grid,
        "consumer kernel `{}` must bind `voxel_grid` because its body \
         calls `terrain.line_of_sight(...)`. Got bindings: {:?}",
        consumer.name,
        consumer.bindings.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
    );

    // The Rust kernel module's `from_context()` must route `voxel_grid`
    // through `ctx.voxel_grid.expect(...)`. Mirrors the
    // `from_context_expr_for_voxel_grid_unwraps_optional` unit test in
    // cg::emit::program — but here the routing has to survive a full
    // emit pass, not just the helper's classify-binding logic.
    let rust = artifacts
        .rust_files
        .get(&format!("{}.rs", consumer.name))
        .unwrap_or_else(|| {
            panic!(
                "no rust file for consumer kernel `{}`; got: {:?}",
                consumer.name,
                artifacts.rust_files.keys().collect::<Vec<_>>()
            )
        });
    assert!(
        rust.contains("ctx.voxel_grid.expect("),
        "consumer kernel `{}` rust module must `.expect(...)` the optional \
         `ctx.voxel_grid`; got rust source:\n{rust}",
        consumer.name,
    );

    // Naga validation — every emitted WGSL file must parse. A consumer
    // body that references `terrain_line_of_sight(` without a matching
    // `voxel_grid` binding would fail naga validation with an unbound
    // identifier; this assertion catches that regression even if
    // someone forgets to update the substring scan in `kernel.rs`.
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, body) in &artifacts.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            failures.push((name.clone(), e.emit_to_string(body)));
        }
    }
    if !failures.is_empty() {
        for (name, msg) in &failures {
            eprintln!("\n--- naga validation failure: {name} ---\n{msg}");
        }
        let consumer_wgsl = artifacts
            .wgsl_files
            .get(&format!("{}.wgsl", consumer.name))
            .cloned()
            .unwrap_or_default();
        eprintln!(
            "\n--- consumer WGSL ({}) for context ---\n{consumer_wgsl}",
            consumer.name
        );
        panic!(
            "{} of {} kernels failed naga validation",
            failures.len(),
            artifacts.wgsl_files.len()
        );
    }
}

/// Negative cousin: a chronicle consumer that does NOT call any
/// terrain method must NOT bind `voxel_grid`. Pins that the substring
/// scan stays gated — we don't want every chronicle consumer to pull
/// the voxel mirror across the dispatch boundary unconditionally.
#[test]
fn chronicle_consumer_without_terrain_does_not_bind_voxel_grid() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Damaged {
  source: AgentId,
  target: AgentId,
  amount: f32,
}

entity Witness : Agent {
  pos: vec3,
  hp:  f32,
  mana: f32,
}

@phase(per_agent)
physics StrikeSelf {
  on Tick {} {
    emit Damaged { source: self, target: self, amount: 1.0 }
  }
}

@phase(post)
physics ApplyMana {
  on Damaged { source: _, target: tgt, amount: a } {
    if (a > 0.0) {
      agents.set_mana(tgt, agents.mana(tgt) + 1.0);
    }
  }
}
"#;
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = lower_compilation_to_cg(&comp).expect("lower");
    let schedule_result = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let artifacts = emit_cg_program(&schedule_result.schedule, &cg).expect("emit");

    let consumer = artifacts
        .kernel_specs
        .iter()
        .find(|spec| spec.name.contains("ApplyMana"))
        .expect("expected an `ApplyMana` consumer kernel");
    assert!(
        consumer.bindings.iter().all(|b| b.name != "voxel_grid"),
        "consumer kernel `{}` must NOT bind `voxel_grid` (no terrain call \
         in body); got bindings: {:?}",
        consumer.name,
        consumer.bindings.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
    );
}
