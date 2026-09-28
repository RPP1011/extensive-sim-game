//! Emit-level pins for the spatial-grid walk contract documented in
//! `crates/dsl_compiler/src/cg/emit/spatial.rs` ("Walk semantics"):
//!
//! 1. A query carrying `@spatial(radius = R)` — literal or
//!    `config.<block>.<field>` — lowers to the declared-radius walk: the
//!    cell box comes from `pos_to_cell_xyz(origin -/+ R)` and candidates
//!    are gated on `R²`, in both the fold and the body form; such folds
//!    are never retagged onto the one-ring tiled PerCell kernel.
//! 2. Un-annotated one-ring walks skip out-of-grid neighbour cells
//!    (`cell_in_grid`) instead of clamping them onto the edge cell.
//! 3. The tiled PerCell kernel keeps each neighbour cell's full count
//!    and reads candidates past the MAX_PER_CELL tile from global memory.
//! 4. The counting-sort scatter leaves `spatial_grid_offsets` zeroed, so
//!    the next tick's count does not start from stale counts.
//! 5. An unsupported radius form is a lowering diagnostic, not a
//!    silently ignored radius.
//!
//! Every emitted kernel is also parsed and validated with naga.
//!
//! The runtime behaviour (GPU results against a CPU brute force) is
//! pinned separately by `crates/sims/tests/spatial_grid_probe_pin.rs`.

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
            panic!("lower_compilation_to_cg returned {} diagnostic(s)", outcome.diagnostics.len());
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn kernel<'a>(art: &'a EmittedArtifacts, needle: &str) -> &'a str {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no kernel matching `{needle}`; got: {names:?}")
        })
}

fn assert_all_kernels_validate(art: &EmittedArtifacts) {
    let mut failures = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: {}", e.emit_to_string(body)));
                continue;
            }
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            failures.push(format!("{name}: {}", e.emit_to_string(body)));
        }
    }
    assert!(failures.is_empty(), "naga failures:\n{}", failures.join("\n"));
}

const SRC: &str = r#"
event Tick { }

@replayable @gpu_amenable
event Seen {
  observer: AgentId,
  target:   AgentId,
}

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

config probe {
  sight: f32 = 17.5,
}

@spatial(radius = 2.5, kind = [Agent])
spatial_query nearby_close(self: AgentId, candidate: AgentId) =
  candidate != self

@spatial(radius = 30, kind = [Agent])
spatial_query nearby_far(self: AgentId, candidate: AgentId) =
  candidate != self

@spatial(radius = config.probe.sight, kind = [Agent])
spatial_query nearby_tuned(self: AgentId, candidate: AgentId) =
  candidate != self

spatial_query nearby_ring(self: AgentId, candidate: AgentId) =
  candidate != self

// Pure folds + sets: tile-eligible shape. Must stay PerAgent because
// the folds carry a declared radius.
physics CountRadius @phase(per_agent) {
  on Tick {} {
    let a = sum(o in spatial.nearby_close(self) where 1.0);
    let b = sum(o in spatial.nearby_far(self) where 1.0);
    let c = sum(o in spatial.nearby_tuned(self) where 1.0);
    agents.set_hunger(self, a);
    agents.set_hp(self, b);
    agents.set_mana(self, c);
  }
}

// Body form over a declared radius.
physics PingFar @phase(per_agent) {
  on Tick {} where (self.alive) {
    for o in spatial.nearby_far(self) {
      emit Seen { observer: self, target: o }
    }
  }
}

// Un-annotated pure-fold body: the tiled PerCell (boids) shape.
physics Flock @phase(per_agent) {
  on Tick {} {
    let v = sum(o in spatial.nearby_ring(self) where o.vel);
    let p = sum(o in spatial.nearby_ring(self) where o.pos);
    agents.set_vel(self, v + p);
  }
}

// Un-annotated body form: per-agent one-ring walk.
physics PingRing @phase(per_agent) {
  on Tick {} where (self.alive) {
    for o in spatial.nearby_ring(self) {
      emit Seen { observer: self, target: o }
    }
  }
}
"#;

#[test]
fn declared_radius_drives_the_fold_walk() {
    let art = compile_inline(SRC);
    let body = kernel(&art, "CountRadius");
    // Literal, integer-literal and config radii each get their own walk.
    for radius in ["f32(2.5)", "f32(30.0)"] {
        assert!(
            body.contains(&format!("let _q_radius: f32 = {radius};")),
            "expected a declared-radius walk with `{radius}`; body:\n{body}"
        );
    }
    assert!(
        body.contains("let _q_radius: f32 = f32(config_"),
        "config radius must lower to a config-const read; body:\n{body}"
    );
    assert!(body.contains("_q_lo = pos_to_cell_xyz(_gate_origin_pos - _q_reach)"), "{body}");
    assert!(body.contains("_q_hi = pos_to_cell_xyz(_gate_origin_pos + _q_reach)"), "{body}");
    assert!(body.contains("let _gate_radius_sq = _q_radius * _q_radius;"), "{body}");
    assert!(body.contains("if (_gate_dist_sq > _gate_radius_sq) { continue; }"), "{body}");
    // Not the one-ring walk, and not the tiled kernel.
    assert!(!body.contains("for (var dz: i32 = -1;"), "radius folds must not walk the fixed ring:\n{body}");
    assert!(!body.contains("tile_pos"), "radius folds must not be retagged to the tiled kernel:\n{body}");
    assert_all_kernels_validate(&art);
}

#[test]
fn declared_radius_drives_the_body_walk() {
    let art = compile_inline(SRC);
    let body = kernel(&art, "PingFar");
    assert!(body.contains("let _q_radius: f32 = f32(30.0);"), "{body}");
    assert!(body.contains("for (var _qx: i32 = _q_lo.x; _qx <= _q_hi.x; _qx = _qx + 1)"), "{body}");
    assert!(!body.contains("for (var dz: i32 = -1;"), "{body}");
}

#[test]
fn one_ring_walks_skip_out_of_grid_cells() {
    let art = compile_inline(SRC);
    let body = kernel(&art, "PingRing");
    let guard = "if (!cell_in_grid(_self_cx + dx, _self_cy + dy, _self_cz + dz)) { continue; }";
    let walk = "let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);";
    let g = body.find(guard).unwrap_or_else(|| panic!("missing in-grid guard; body:\n{body}"));
    let w = body.find(walk).expect("one-ring cell_index");
    assert!(g < w, "the in-grid guard must precede the cell lookup");
    assert!(body.contains("fn cell_in_grid(cx: i32, cy: i32, cz: i32) -> bool"), "prelude helper");
}

#[test]
fn tiled_kernel_keeps_full_counts_and_spills_past_the_tile() {
    let art = compile_inline(SRC);
    let body = kernel(&art, "Flock");
    assert!(body.contains("tile_pos"), "Flock should be the tiled PerCell kernel; body:\n{body}");
    // Tile load: out-of-grid lanes get count 0; the full count is stored.
    assert!(body.contains("if (cell_in_grid(i32(home_cx) + dx, i32(home_cy) + dy, i32(home_cz) + dz))"), "{body}");
    assert!(body.contains("_nbr_count = _nbr_end - _nbr_start;"), "{body}");
    assert!(body.contains("let _tile_n = min(_nbr_count, SPATIAL_MAX_PER_CELL);"), "{body}");
    assert!(!body.contains("min(_nbr_end - _nbr_start, SPATIAL_MAX_PER_CELL)"), "count must not be capped");
    // Walk: tile for the first MAX_PER_CELL, global memory past it.
    assert!(body.contains("if (_i < SPATIAL_MAX_PER_CELL) {"), "{body}");
    let spill = body.split("} else {").nth(1).unwrap_or("");
    assert!(
        spill.contains("agent_vel[per_pair_candidate]") || spill.contains("agent_vel_prev[per_pair_candidate]"),
        "spill branch must read candidate vel from global memory; body:\n{body}"
    );
    assert_all_kernels_validate(&art);
}

#[test]
fn scatter_leaves_offsets_zeroed_for_the_next_tick() {
    let art = compile_inline(SRC);
    let body = kernel(&art, "spatial_build_hash_scatter");
    let reset = "atomicStore(&spatial_grid_offsets[_rc], 0u);";
    let r = body.find(reset).unwrap_or_else(|| panic!("scatter must zero offsets; body:\n{body}"));
    let last_bump = body.rfind("atomicAdd(&spatial_grid_offsets[_c], 1u)").expect("cursor bump");
    assert!(r > last_bump, "the reset must follow the last cursor bump");
    assert!(body.contains("storageBarrier();"), "{body}");
}

#[test]
fn unsupported_radius_form_is_a_diagnostic() {
    let src = r#"
event Tick { }
entity Walker : Agent { pos: vec3, vel: vec3 }
@spatial(radius = far_away, kind = [Agent])
spatial_query nearby_bad(self: AgentId, candidate: AgentId) =
  candidate != self
physics Ping @phase(per_agent) {
  on Tick {} {
    let n = sum(o in spatial.nearby_bad(self) where 1.0);
    agents.set_vel(self, vec3(n, 0.0, 0.0));
  }
}
"#;
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let outcome = lower_compilation_to_cg(&comp).expect_err("bad radius must not lower silently");
    assert!(
        outcome.diagnostics.iter().any(|d| d.to_string().contains("@spatial(radius")),
        "expected an @spatial(radius) diagnostic; got {:?}",
        outcome.diagnostics.iter().map(|d| d.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn declared_filter_is_applied_in_fold_and_body_walks() {
    let art = compile_inline(SRC);
    // Fold form: the projection is wrapped in the query's filter
    // (`candidate != self`, with candidate -> per_pair_candidate and
    // self -> the walk origin, here agent_id).
    let fold = kernel(&art, "CountRadius");
    assert!(fold.contains("select(0.0, 1.0, (per_pair_candidate != agent_id))"), "{fold}");
    // Body form: the body runs under `if (<filter>)`.
    let body = kernel(&art, "PingFar");
    assert!(body.contains("if ((per_pair_candidate != agent_id)) {"), "{body}");
    // Tiled fold: same filter inside the tile walk.
    let tiled = kernel(&art, "Flock");
    assert!(tiled.contains("(per_pair_candidate != agent_id)"), "{tiled}");
}

#[test]
fn declared_filter_binds_self_to_a_non_self_origin() {
    // `self` in the declaration means the walk origin — here an agent
    // bound from an event, not the dispatching thread's agent.
    let src = r#"
event Tick { }
@replayable @gpu_amenable
event Poke { who: AgentId }
entity Walker : Agent { pos: vec3, vel: vec3 }
@spatial(radius = 4.0, kind = [Agent])
spatial_query richer(self: AgentId, candidate: AgentId) =
  candidate != self && candidate.hp > self.hp
@phase(post)
physics Around {
  on Poke { who: w } {
    let n = sum(o in spatial.richer(w) where 1.0);
    agents.set_mana(w, n);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel(&art, "Around");
    assert!(!body.contains("!= agent_id"), "self must not bind to agent_id here:\n{body}");
    // The origin is the event-bound `w` (a `local_N`); the filter's
    // `self` reads resolve to it.
    assert!(body.contains("(per_pair_candidate != local_"), "{body}");
    assert!(body.contains("agent_hp[per_pair_candidate] > agent_hp[target_expr_"), "{body}");
    assert_all_kernels_validate(&art);
}

#[test]
fn tiled_walk_gates_like_the_non_tiled_walk() {
    let art = compile_inline(SRC);
    let tiled = kernel(&art, "Flock");
    assert!(tiled.contains("let _gate_radius_sq = 3.0 * f32(4u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE;"), "{tiled}");
    assert!(tiled.contains("let _gate_dxyz = tile_pos[_tile_idx] - _gate_origin_pos;"), "{tiled}");
    let spill = tiled.split("} else {").nth(1).unwrap_or("");
    assert!(spill.contains("_gate_dxyz = agent_pos"), "spill branch must gate too:\n{tiled}");
    let ring = kernel(&art, "PingRing");
    assert!(ring.contains("let _gate_radius_sq = 3.0 * f32(4u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE;"), "{ring}");
}

#[test]
fn tiled_kernel_without_vel_reads_skips_the_vel_tile() {
    let src = r#"
event Tick { }
entity Walker : Agent { pos: vec3, vel: vec3 }
spatial_query ring(self: AgentId, candidate: AgentId) =
  candidate != self
physics CountRing @phase(per_agent) {
  on Tick {} {
    let n = sum(o in spatial.ring(self) where 1.0);
    agents.set_mana(self, n);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel(&art, "CountRing");
    assert!(body.contains("tile_pos[_dst_base + i]"), "should be the tiled kernel:\n{body}");
    assert!(!body.contains("tile_vel[_dst_base + i]"), "vel must not be loaded:\n{body}");
    assert_all_kernels_validate(&art);
    // ...and a kernel that does read vel still loads it.
    let flock = compile_inline(SRC);
    assert!(kernel(&flock, "Flock").contains("tile_vel[_dst_base + i]"));
}
