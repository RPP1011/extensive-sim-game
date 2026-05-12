//! Pin: every `for x in spatial.nearby_*(self) { … }` body-form walk
//! AND every `sum(x in spatial.nearby_*(self) where …)` fold-form
//! walk has an auto-injected per-candidate squared-distance gate that
//! drops boundary-cell-clamp pool agents.
//!
//! Closes the regression-test half of **Gap dungeon_layout #1** in
//! `docs/architecture/gaps_dungeon_layout.md` — the spatial grid's
//! `cell_index(cx, cy, cz)` helper clamps OOB coordinates to
//! `GRID_DIM - 1`, which silently pools every distant agent into the
//! boundary cell. A walk on a boundary-side cell then sees those OOB
//! agents as "neighbours" by cell-coord arithmetic even though they
//! are FAR in world coordinates. The gap doc captured this with
//! Goblin slot 25 at world `(58.5, 5.5)` (cell `(20, 11, 10)`) hitting
//! heroes at world `(6, 30)` (cell `(11, 15, 10)`) — cells 9 columns
//! apart on the x axis, well outside the spatial walk's 3-cell window.
//!
//! Pre-fix, every .sim author had to re-state the per-pair
//! `length(candidate.pos - self.pos) < <range>` gate inside every
//! verb body's `if (…)` clause to mask the OOB pool. That manual
//! gating shape spread across `hill_raid`, `palace_coup`,
//! `dungeon_layout`, etc. Adding the auto-gate as part of the spatial
//! walk emit keeps the cell-walk shape but drops the OOB pool at
//! source.
//!
//! The auto-gate's bound is the 3D-diagonal of the cell-walk
//! neighbourhood: `(r+1) * cell_size * sqrt(3)`, squared. Squaring
//! avoids the sqrt at runtime; the `(r+1)` margin accounts for
//! self's position within its own cell (worst case: self at one
//! corner, candidate at the opposite-most corner of the most-
//! diagonal visited cell). For the today-hard-coded `radius_cells =
//! 1` this is `3 * 4 * SPATIAL_CELL_SIZE²` = `12 * SPATIAL_CELL_SIZE²`
//! — covers any in-window candidate without false rejection.
//!
//! What the test pins:
//!
//!   1. The body-form spatial walk (`for x in spatial.nearby_X(self) {
//!      … }`) emits the `_gate_dxyz = agent_pos[...] - agent_pos[...]`
//!      + `_gate_dist_sq = dot(...)` + `if (_gate_dist_sq >
//!      _gate_radius_sq) { continue; }` triple BEFORE the user's body
//!      executes.
//!   2. The fold-form spatial walk (`sum(x in spatial.nearby_X(self)
//!      where …)`) emits the same triple BEFORE the per-candidate
//!      accumulator update.
//!   3. The gate radius computation references `SPATIAL_CELL_SIZE`
//!      (the WGSL constant) — so a future bump to `CELL_SIZE` flows
//!      through automatically without an emit-side edit.
//!   4. Every emitted kernel parses cleanly with naga (sanity gate
//!      that the gate's WGSL doesn't break the surrounding loop
//!      structure).
//!
//! A future regression that strips the gate (e.g., a "minor cleanup"
//! refactor that drops the `continue` site, or replaces the squared-
//! distance check with a no-op) trips assertions 1, 2, or 3.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;

fn compile_inline(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            for diag in &outcome.diagnostics {
                eprintln!("[lower diagnostic] {diag}");
            }
            panic!(
                "lower_compilation_to_cg returned {} diagnostic(s) — see stderr above",
                outcome.diagnostics.len()
            );
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn kernel_body_containing<'a>(
    art: &'a EmittedArtifacts,
    needle: &str,
) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
}

/// Body-form walk: `for candidate in spatial.nearby_X(self) { … }`.
/// The auto-gate sits immediately AFTER `let per_pair_candidate =
/// spatial_grid_cells[_i];` and BEFORE the user's body — every
/// candidate gets the squared-distance check + `continue` site
/// regardless of what the user's body does.
#[test]
fn body_form_walk_has_auto_distance_gate() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Pinged {
  observer: AgentId,
  target:   AgentId,
}

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

spatial_query nearby_others(self: AgentId, candidate: AgentId) =
  candidate != self

@phase(per_agent)
physics PingNeighbors {
  on Tick {} where (self.alive) {
    for other in spatial.nearby_others(self) {
      emit Pinged {
        observer: self,
        target:   other,
      }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_body_containing(&art, "PingNeighbors").unwrap_or_else(|| {
        let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
        panic!("expected a PingNeighbors kernel; got: {names:?}");
    });

    // 1. The squared-distance compute is present (the gate's
    //    centerpiece). `_gate_dxyz = agent_pos[per_pair_candidate] -
    //    agent_pos[agent_id]` followed by `_gate_dist_sq = dot(...)`.
    assert!(
        body.contains("_gate_dxyz = agent_pos[per_pair_candidate] - agent_pos[agent_id]"),
        "expected `_gate_dxyz` delta compute inside the per-candidate \
         spatial walk body; body:\n{body}",
    );
    assert!(
        body.contains("_gate_dist_sq = dot(_gate_dxyz, _gate_dxyz)"),
        "expected `_gate_dist_sq = dot(_gate_dxyz, _gate_dxyz)` squared \
         distance compute; body:\n{body}",
    );

    // 2. The `continue` site is reached when the squared distance
    //    exceeds the gate. Without `continue` the gate is a no-op.
    assert!(
        body.contains("if (_gate_dist_sq > _gate_radius_sq) { continue; }"),
        "expected `if (_gate_dist_sq > _gate_radius_sq) {{ continue; }}` \
         gate site; body:\n{body}",
    );

    // 3. The gate radius is computed from `SPATIAL_CELL_SIZE` — a
    //    future cell-size bump must flow through automatically. The
    //    `3.0 * f32(N) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE`
    //    shape encodes the 3D-diagonal squared bound `3 * (r+1)² *
    //    cell_size²`.
    assert!(
        body.contains("_gate_radius_sq = 3.0 * f32(")
            && body.contains("u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE"),
        "expected `_gate_radius_sq = 3.0 * f32(N) * SPATIAL_CELL_SIZE \
         * SPATIAL_CELL_SIZE` formula referencing the WGSL constant; \
         body:\n{body}",
    );

    // 4. Naga parses every emitted kernel cleanly — the gate doesn't
    //    desync the surrounding loop scaffolding.
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, file_body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(file_body) {
            failures.push((name.clone(), e.emit_to_string(file_body)));
        }
    }
    assert!(
        failures.is_empty(),
        "naga validation failures after auto-gate injection:\n{:#?}",
        failures,
    );
}

/// Fold-form walk: `sum(other in spatial.nearby_X(self) where …)`.
/// The auto-gate sits immediately AFTER `let per_pair_candidate =
/// spatial_grid_cells[_i];` and BEFORE the per-candidate accumulator
/// update — every projection skips OOB-pool candidates.
#[test]
fn fold_form_walk_has_auto_distance_gate() {
    // A view that folds over `spatial.nearby_X(self)` — projecting a
    // constant `1u` gets us a "count neighbours" fold without
    // entangling per-field reads. The fold lowers to `CgStmt::
    // ForEachNeighbor` which emits via `emit_fused_for_each_neighbor`
    // (the path Gap dungeon_layout #1's fix targets).
    let src = r#"
event Tick { }

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

spatial_query nearby_others(self: AgentId, candidate: AgentId) =
  candidate != self

@phase(per_agent)
physics CountNeighbors {
  on Tick {} where (self.alive) {
    let n = sum(other in spatial.nearby_others(self) where 1.0);
    agents.set_vel(self, vec3(n, 0.0, 0.0));
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_body_containing(&art, "CountNeighbors").unwrap_or_else(|| {
        let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
        panic!("expected a CountNeighbors kernel; got: {names:?}");
    });

    // Same three structural assertions as the body-form path, on the
    // emit-fused fold walk.
    assert!(
        body.contains("_gate_dxyz = agent_pos[per_pair_candidate] - agent_pos[agent_id]"),
        "expected `_gate_dxyz` delta compute in fused fold walk; body:\n{body}",
    );
    assert!(
        body.contains("_gate_dist_sq = dot(_gate_dxyz, _gate_dxyz)"),
        "expected `_gate_dist_sq` squared distance in fused fold walk; \
         body:\n{body}",
    );
    assert!(
        body.contains("if (_gate_dist_sq > _gate_radius_sq) { continue; }"),
        "expected `continue` site in fused fold walk; body:\n{body}",
    );
    assert!(
        body.contains("_gate_radius_sq = 3.0 * f32(")
            && body.contains("u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE"),
        "expected gate radius formula referencing SPATIAL_CELL_SIZE \
         in fused fold walk; body:\n{body}",
    );

    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, file_body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(file_body) {
            failures.push((name.clone(), e.emit_to_string(file_body)));
        }
    }
    assert!(
        failures.is_empty(),
        "naga validation failures after auto-gate injection:\n{:#?}",
        failures,
    );
}
