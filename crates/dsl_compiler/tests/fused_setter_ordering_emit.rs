//! Pin: WGSL emit must produce a topologically-valid `let` ordering
//! when a single `@phase(per_agent)` rule body writes BOTH a non-atomic
//! SoA column (e.g. `set_pos`) AND an f32 column upgraded to atomic
//! (e.g. `set_mana`), where the first write's RHS shares value-chain
//! Lets with the second write's RHS.
//!
//! Closes Gap #3 of `docs/architecture/gaps_among_us.md`.
//!
//! Background
//! ----------
//!
//! In `assets/sim/among_us.sim`, the original single-rule body looked
//! like:
//!
//! ```text
//! @phase(per_agent)
//! physics AgentTaskSteer {
//!   on Tick {} where self.alive {
//!     let task_idx = self.mana;          // chain root: reads upgraded `mana`
//!     ...
//!     let new_pos = ...;                  // depends transitively on task_idx
//!     let new_task = ...;                 // depends transitively on task_idx
//!     agents.set_pos(self, new_pos);      // <- non-atomic write
//!     agents.set_mana(self, new_task);    // <- atomic CAS write triggers RMW upgrade
//!   }
//! }
//! ```
//!
//! Symptom: emitted WGSL placed `agent_pos[agent_id] = local_N;`
//! BEFORE the `var local_N: vec3<f32>;` declaration that the f32 RMW
//! prologue inserts. This produced naga validation error:
//!
//! ```text
//! local_N use before declaration
//! ```
//!
//! The fixture-side workaround (still present in among_us.sim) was to
//! split the rule into `AgentTaskSteer` (set_pos only) +
//! `AgentTaskAdvance` (set_mana only). This pin verifies the EMITTER
//! handles the fused single-rule shape — once green, the fixture
//! workaround can be unwound (separate follow-up).
//!
//! Root cause
//! ----------
//!
//! `lower_cg_stmt_list_to_wgsl` (in
//! `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`) walks residual
//! stmts in source order. When an f32-RMW upgrade matches an Assign
//! (the `set_mana` write):
//!
//!   1. Forward chain analysis identifies prefix Lets whose values
//!      depend on the upgraded field's reads (transitively through
//!      `expr_depends_on_upgraded_field`).
//!   2. Those chain-Lets are var-promoted (`var local_N: T;`
//!      declared outside the loop, assigned inside) so each retry
//!      sees a fresh atomicLoad snapshot.
//!   3. The `var` declarations are inserted into `parts` at the index
//!      of the CAS loop (post-walk).
//!
//! Bug: a NON-CHAIN prefix stmt that READS one of the chain-locals
//! (e.g. `agents.set_pos(self, new_pos)` where `new_pos` was a
//! chain-let earlier in the same body) is emitted as the plain
//! `agent_pos[agent_id] = local_N;` line BEFORE the var declarations
//! are inserted. The reader-stmt is a plain non-chain stmt so it
//! goes into `parts` directly — but it references `local_N` which
//! only gets its `var` declaration AFTER it in the parts list.
//!
//! Fix
//! ---
//!
//! When inserting the `var local_N: T;` declarations, locate the
//! EARLIEST `parts` index that already references any of the
//! var-promoted locals — insert the declarations BEFORE that index
//! (rather than at `cas_loop_parts_idx`). This keeps the existing
//! "declarations precede the CAS loop" invariant while extending it
//! to cover non-chain consumers that source-precede the loop.

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

fn kernel_wgsl<'a>(art: &'a EmittedArtifacts, needle: &str) -> &'a str {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no kernel matched {needle:?}; emitted kernels: {names:?}")
        })
    }

/// Returns (line_number, line_text) of the first line that mentions
/// `needle` (substring match) in `wgsl`. Lines are 1-indexed.
fn find_first_line<'a>(wgsl: &'a str, needle: &str) -> Option<(usize, &'a str)> {
    wgsl.lines()
        .enumerate()
        .find(|(_, line)| line.contains(needle))
        .map(|(i, line)| (i + 1, line))
}

/// Repro: a single PerAgent rule that writes `set_pos` THEN `set_mana`
/// in the same body, with both RHS expressions sharing chain-Lets that
/// read `self.mana` (the upgraded f32 field).
///
/// The pin asserts every reference to `local_N` in the emitted body
/// is preceded (in source order) by either a `let local_N` or a
/// `var local_N` declaration.
#[test]
fn fused_set_pos_and_set_mana_emits_topologically_ordered_locals() {
    // Minimal repro mirroring among_us.sim's AgentTaskSteer body
    // shape. Both `new_pos` and `new_task` depend on `task_idx`
    // (a chain-let reading `self.mana`).
    let src = r#"
event Tick { }

@phase(per_agent)
physics FusedSteerAdvance {
  on Tick {} where (self.alive) {
    let task_idx = self.mana;
    let target = vec3(task_idx, task_idx, 0.0);
    let to_target = target - self.pos;
    let dist = length(to_target);
    let new_pos = self.pos + (to_target / dist) * 0.1;
    let advanced = task_idx + 1.0;
    let new_task = if (dist <= 0.5) { advanced } else { task_idx };
    agents.set_pos(self, new_pos);
    agents.set_mana(self, new_task);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_FusedSteerAdvance");

    // Sanity: the f32 RMW upgrade fired (mana is atomic).
    assert!(
        body.contains("var<storage, read_write> agent_mana: array<atomic<u32>>;"),
        "set_mana write must upgrade agent_mana to array<atomic<u32>>; got body:\n{body}",
    );
    assert!(
        body.contains("atomicCompareExchangeWeak(&agent_mana["),
        "expected atomicCompareExchangeWeak on agent_mana; got body:\n{body}",
    );

    // Topological-order check: scan every `local_N` reference in
    // body order; for each N, the FIRST occurrence must be a
    // declaration line (`let local_N` or `var local_N`), not a
    // use site.
    //
    // We tolerate references inside the CAS loop body that re-assign
    // var-promoted locals (`local_N = …;`) — those are valid uses
    // of an outer-scope `var` declaration that must appear EARLIER.
    use std::collections::HashSet;
    let mut declared: HashSet<u32> = HashSet::new();
    for (lineno, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        // Find every `local_<digits>` mention on this line.
        let mut search_from = 0usize;
        while let Some(pos) = trimmed[search_from..].find("local_") {
            let abs = search_from + pos;
            let after = &trimmed[abs + "local_".len()..];
            // Parse trailing digits.
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                search_from = abs + "local_".len();
                continue;
            }
            let n: u32 = digits.parse().unwrap();
            // Was the FIRST reference on a declaration line?
            // Decide via the immediate prefix of "local_N" on this
            // textual line: a declaration is `let local_N` or
            // `var local_N`.
            let local_tok = format!("local_{}", n);
            let local_idx = trimmed.find(&local_tok).unwrap();
            let prefix = &trimmed[..local_idx];
            let is_decl = prefix.ends_with("let ") || prefix.ends_with("var ");
            if is_decl {
                declared.insert(n);
            } else if !declared.contains(&n) {
                panic!(
                    "topological-order violation: `local_{n}` used at line {} \
                     before any declaration. Line: {trimmed:?}\n\nFull body:\n{body}",
                    lineno + 1
                );
            }
            search_from = abs + "local_".len() + digits.len();
        }
    }
}

/// Independent shape: even if the fix lands purely as a "hoist var
/// declarations earlier" tweak, this pin verifies we DO get a
/// `var local_N: vec3<f32>;` declaration for the position chain
/// (the field shared between `set_pos` consumer + `set_mana` chain).
///
/// Without the var-promotion firing, the fixed-emit could regress to
/// emitting two `let local_N` bindings (one per parts insertion)
/// which would fail WGSL's "no shadowing in same block" rule.
#[test]
fn fused_set_pos_and_set_mana_promotes_chain_locals_to_var() {
    let src = r#"
event Tick { }

@phase(per_agent)
physics FusedSteerAdvance2 {
  on Tick {} where (self.alive) {
    let task_idx = self.mana;
    let target = vec3(task_idx, task_idx, 0.0);
    let to_target = target - self.pos;
    let dist = length(to_target);
    let new_pos = self.pos + (to_target / dist) * 0.1;
    let advanced = task_idx + 1.0;
    let new_task = if (dist <= 0.5) { advanced } else { task_idx };
    agents.set_pos(self, new_pos);
    agents.set_mana(self, new_task);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_FusedSteerAdvance2");

    // The chain analysis must var-promote AT LEAST one vec3<f32>
    // local (`new_pos` derives from `task_idx = self.mana` so it's
    // tainted) AND the `agent_pos` writeback must consume one of
    // those var-promoted locals (the bug surface — set_pos consumes
    // a chain local but used to be emitted before its declaration).
    let var_vec3_locals: Vec<u32> = body
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("var local_") && l.contains(": vec3<f32>;"))
        .filter_map(|l| {
            l.strip_prefix("var local_")?
                .split(':')
                .next()?
                .parse::<u32>()
                .ok()
        })
        .collect();
    assert!(
        !var_vec3_locals.is_empty(),
        "expected at least one `var local_N: vec3<f32>;` declaration \
         (the var-promoted chain local for `new_pos`); got body:\n{body}",
    );
    // Find the agent_pos consumer line and parse its `local_N`.
    let pos_line = body
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("agent_pos[agent_id] = local_"))
        .unwrap_or_else(|| {
            panic!(
                "expected `agent_pos[agent_id] = local_<N>;` (set_pos \
                 consuming a var-promoted vec3 local); got body:\n{body}",
            )
        });
    let n: u32 = pos_line
        .strip_prefix("agent_pos[agent_id] = local_")
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .parse()
        .expect("parse pos consumer local index");
    assert!(
        var_vec3_locals.contains(&n),
        "`agent_pos[agent_id] = local_{n};` consumes a local that's NOT \
         in the var-promoted chain set {var_vec3_locals:?}; got body:\n{body}",
    );
    // Declaration MUST precede the consumer line (the bug surface).
    let (decl_line, _) = find_first_line(body, &format!("var local_{n}:")).unwrap();
    let (use_line, _) = find_first_line(body, pos_line).unwrap();
    assert!(
        decl_line < use_line,
        "`var local_{n}` declared at line {decl_line} but consumed at line \
         {use_line}; declaration must precede use; got body:\n{body}",
    );
}

/// Strongest-form pin: every kernel emitted by the fused-setter
/// fixture must parse cleanly through naga's WGSL parser. naga's
/// strict variable-scope checking is what surfaced the original
/// `local_N use before declaration` error in the among_us pin
/// build — pinning a clean parse here catches any future emit
/// regression that produces structurally invalid WGSL even if the
/// textual ordering looks plausible to the human-readable assertions
/// above.
#[test]
fn fused_set_pos_and_set_mana_emits_naga_clean_wgsl() {
    let src = r#"
event Tick { }

@phase(per_agent)
physics FusedSteerAdvance3 {
  on Tick {} where (self.alive) {
    let task_idx = self.mana;
    let target = vec3(task_idx, task_idx, 0.0);
    let to_target = target - self.pos;
    let dist = length(to_target);
    let new_pos = self.pos + (to_target / dist) * 0.1;
    let advanced = task_idx + 1.0;
    let new_task = if (dist <= 0.5) { advanced } else { task_idx };
    agents.set_pos(self, new_pos);
    agents.set_mana(self, new_task);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_FusedSteerAdvance3");

    if let Err(e) = naga::front::wgsl::parse_str(body) {
        let msg = e.emit_to_string(body);
        panic!(
            "naga rejected the fused set_pos+set_mana kernel — Gap among_us#3 \
             regression. Error:\n{msg}\n\nFull body:\n{body}",
        );
    }
}
