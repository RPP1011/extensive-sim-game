//! Emission of `engine_rules/src/mask_fill.rs` — the `fill_all` function that
//! populates the `MaskBuffer` and `TargetMask` for every alive agent.
//!
//! The emitted function is derived from the `mask` declarations in the IR:
//!   - Self-only masks (no positional args) produce a `buf.set(slot, Kind, true)` call.
//!   - Target-bound masks (with a `from <source>` clause) produce a candidate-
//!     enumerator call (`mask_<name>_candidates`) followed by a check on whether
//!     any candidates were pushed.
//!
//! The Cast mask is special — it takes an `ability: AbilityId` argument. The
//! emitter generates a per-ability loop over the registry that sets the Cast bit
//! if any ability passes the gate.

use std::fmt::Write;

use crate::ir::{IrActionHeadShape, MaskIR};

fn snake_case(name: &str) -> String {
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

/// Emit `engine_rules/src/mask_fill.rs` from the mask IR.
pub fn emit_mask_fill(masks: &[MaskIR], source_file: Option<&str>) -> String {
    let mut out = String::new();
    emit_header(&mut out, source_file);

    writeln!(out, "use engine::ability::AbilityId;").unwrap();
    writeln!(out, "use engine::backend::ComputeBackend;").unwrap();
    writeln!(out, "use engine::mask::{{MaskBuffer, MicroKind, TargetMask}};").unwrap();
    writeln!(out, "use engine::state::SimState;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Fill every mask bit and target-mask candidate list for the current tick.").unwrap();
    writeln!(out, "/// Resets `buf` and `targets` before populating. Called at the top of").unwrap();
    writeln!(out, "/// `engine_rules::step::step` after the `tick` increment.").unwrap();
    writeln!(out, "pub fn fill_all<B: engine::backend::ComputeBackend>(backend: &mut B, buf: &mut MaskBuffer, targets: &mut TargetMask, state: &SimState) {{").unwrap();
    writeln!(out, "    backend.reset_mask(buf);").unwrap();
    writeln!(out, "    targets.reset();").unwrap();
    writeln!(out, "    for id in state.agents_alive() {{").unwrap();
    writeln!(out, "        let slot = (id.raw() - 1) as usize;").unwrap();

    // Collect self-only and target-bound masks separately so we can emit
    // them in the committed order: self-only block (with group comment),
    // then cast, then target-bound enumerators + checks together.
    let mut self_only: Vec<(&MaskIR, String)> = Vec::new();
    let mut cast_present = false;
    let mut target_bound: Vec<(&MaskIR, String)> = Vec::new();

    for mask in masks {
        let stem = snake_case(&mask.head.name);
        match &mask.head.shape {
            IrActionHeadShape::None => {
                self_only.push((mask, stem));
            }
            IrActionHeadShape::Positional(params) => {
                if params.is_empty() {
                    self_only.push((mask, stem));
                } else {
                    let is_cast = params.iter().any(|(n, _, _)| n == "ability");
                    if is_cast {
                        cast_present = true;
                    } else if mask.candidate_source.is_some() {
                        target_bound.push((mask, stem));
                    } else {
                        self_only.push((mask, stem));
                    }
                }
            }
            IrActionHeadShape::Named(_) => {
                self_only.push((mask, stem));
            }
        }
    }

    // Self-only block.
    if !self_only.is_empty() {
        writeln!(out, "        // Self-only masks — each call returns a bool gating the action head.").unwrap();
        for (mask, stem) in &self_only {
            emit_self_only_mask(&mut out, &mask.head.name, stem);
        }
    }

    // Cast mask (per-ability loop).
    if cast_present {
        emit_cast_mask(&mut out);
    }

    // Target-bound masks — enumerators first (alphabetical order to match
    // committed mask_fill.rs), then candidate checks in the same order.
    if !target_bound.is_empty() {
        // Sort alphabetically by mask name so emitter output is stable and
        // matches the committed file's Attack-before-MoveToward order.
        let mut sorted_target: Vec<(&MaskIR, String)> = target_bound;
        sorted_target.sort_by(|(a, _), (b, _)| a.head.name.cmp(&b.head.name));
        writeln!(out, "        // Target-bound masks — run the candidate enumerator.").unwrap();
        writeln!(out, "        // The enumerators push into `targets` directly.").unwrap();
        for (mask, stem) in &sorted_target {
            if micro_kind_from_name(&mask.head.name).is_some() {
                writeln!(out, "        crate::mask::mask_{stem}_candidates(state, id, targets);").unwrap();
            }
        }
        writeln!(out, "        // Mark Attack / MoveToward allowed if any candidates were pushed.").unwrap();
        for (mask, _stem) in &sorted_target {
            if let Some(kind) = micro_kind_from_name(&mask.head.name) {
                writeln!(out, "        if !targets.candidates_for(id, {kind}).is_empty() {{").unwrap();
                writeln!(out, "            backend.set_mask_bit(buf, slot, {kind});").unwrap();
                writeln!(out, "        }}").unwrap();
            }
        }
    }

    writeln!(out, "    }}").unwrap();
    writeln!(out, "    backend.commit_mask(buf);").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

fn emit_header(out: &mut String, _source_file: Option<&str>) {
    // mask_fill.rs committed header has no "// Source:" line.
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.").unwrap();
    writeln!(out, "// Regenerate with `cargo run --bin xtask -- compile-dsl`.").unwrap();
    writeln!(out).unwrap();
}

fn micro_kind_from_name(name: &str) -> Option<&'static str> {
    match name {
        "Hold"       => Some("MicroKind::Hold"),
        "MoveToward" => Some("MicroKind::MoveToward"),
        "Flee"       => Some("MicroKind::Flee"),
        "Attack"     => Some("MicroKind::Attack"),
        "Cast"       => Some("MicroKind::Cast"),
        "UseItem"    => Some("MicroKind::UseItem"),
        "Eat"        => Some("MicroKind::Eat"),
        "Drink"      => Some("MicroKind::Drink"),
        "Rest"       => Some("MicroKind::Rest"),
        _            => None,
    }
}

fn emit_self_only_mask(out: &mut String, name: &str, stem: &str) {
    if let Some(kind) = micro_kind_from_name(name) {
        writeln!(out, "        if crate::mask::mask_{stem}(state, id) {{").unwrap();
        writeln!(out, "            backend.set_mask_bit(buf, slot, {kind});").unwrap();
        writeln!(out, "        }}").unwrap();
    }
}

fn emit_cast_mask(out: &mut String) {
    writeln!(out, "        // Cast mask — `mask_cast` is per-ability; we set the global Cast bit").unwrap();
    writeln!(out, "        // if ANY ability in the registry passes the gate. When the registry").unwrap();
    writeln!(out, "        // is empty the bit is set permissively (legacy fallback).").unwrap();
    writeln!(out, "        let n_abilities = state.ability_registry.len();").unwrap();
    writeln!(out, "        if n_abilities == 0 {{").unwrap();
    writeln!(out, "            backend.set_mask_bit(buf, slot, MicroKind::Cast);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            'outer: for raw in 1..=(n_abilities as u32) {{").unwrap();
    writeln!(out, "                if let Some(ability_id) = AbilityId::new(raw) {{").unwrap();
    writeln!(out, "                    if crate::mask::mask_cast(state, id, ability_id) {{").unwrap();
    writeln!(out, "                        backend.set_mask_bit(buf, slot, MicroKind::Cast);").unwrap();
    writeln!(out, "                        break 'outer;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
}

#[allow(dead_code)]
fn emit_target_bound_mask(out: &mut String, name: &str, stem: &str) {
    if let Some(kind) = micro_kind_from_name(name) {
        writeln!(out, "        crate::mask::mask_{stem}_candidates(state, id, targets);").unwrap();
        writeln!(out, "        if !targets.candidates_for(id, {kind}).is_empty() {{").unwrap();
        writeln!(out, "            backend.set_mask_bit(buf, slot, {kind});").unwrap();
        writeln!(out, "        }}").unwrap();
    }
}
