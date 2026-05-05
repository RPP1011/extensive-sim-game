//! Wave 1.7 — `AbilityRegistry` build + `cast <Name>` resolution.
//!
//! The Wave 1.6 lowering pass (`crate::ability_lower`) emits one
//! `engine::ability::AbilityProgram` per `.ability` decl, but every
//! `EffectOp::CastAbility` it produces carries a placeholder `AbilityId`
//! (slot 0) because lowering has no view of the cross-file name table.
//! This module is the next compiler step: it walks every parsed
//! `AbilityFile`, assigns stable 1-based `AbilityId`s in input order,
//! resolves each `cast <Name>` to its real id, detects dependency cycles,
//! and finally drives an `AbilityRegistryBuilder` to produce the frozen
//! `AbilityRegistry` the engine consumes at runtime.
//!
//! Scope of this slice (per `docs/spec/ability_dsl_unified.md` §7.9 +
//! the Wave 1.6 module docs in `ability_lower.rs`):
//!
//! * **Resolves:** the single `cast <ability_name>` verb. Selector stays
//!   `TargetSelector::Caster` per Wave 1.6 — the future `cast_on
//!   <selector>` modifier is Wave 2+.
//! * **Diagnostics:** duplicate ability names across the input set,
//!   unresolved cast targets, and cast-dependency cycles (self, two-,
//!   three-, N-cycles).
//! * **Out of scope:** Hero TOML binding (Wave 1.8), GPU buffer layout
//!   (Wave 1.9), the remaining 19 `EffectOp` variants (Waves 2-5).
//!
//! Constitution touch-points:
//! * P1 (compiler-first): this is a build-time compiler pass. It runs
//!   once at startup, not on the deterministic tick path.
//! * P2 (schema-hash): no engine type changes; pure consumer of the
//!   existing `AbilityRegistry` + `AbilityProgram` shapes.
//! * P10 (no panics on the deterministic path): every user-input failure
//!   surfaces via `Result<_, RegistryBuildError>`. The single `panic!` /
//!   `expect()` site below guards an internal invariant established by
//!   Pass 1 (the cast-verb-without-CastAbility check) — Wave 1.6
//!   guarantees that pairing, so its failure is a compiler bug, not a
//!   user-facing error.

use std::collections::{HashMap, HashSet};

use dsl_ast::ast::{AbilityDecl, AbilityFile, EffectArg, Span};
use engine::ability::program::{AbilityProgram, EffectOp};
use engine::ability::{AbilityId, AbilityRegistry, AbilityRegistryBuilder};

use crate::ability_lower::{lower_ability_decl, LowerError};

/// Output of `build_registry` — the frozen `AbilityRegistry` plus the
/// name -> id map callers need to bind hero templates against later
/// (Wave 1.8). `names` preserves the spelling of every ability decl
/// across every input file; collisions short-circuit at Pass 1.
pub struct BuiltRegistry {
    pub registry: AbilityRegistry,
    pub names:    HashMap<String, AbilityId>,
}

// Hand-rolled `Debug` so call sites can use `Result::expect_err` etc.
// `AbilityRegistry` itself isn't `Debug` (registry interior is opaque to
// the outside; see `engine::ability::registry`), so we surface only the
// outer shape — slot count + name table — which is all a failing-test
// renderer needs.
impl std::fmt::Debug for BuiltRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltRegistry")
            .field("registry_len", &self.registry.len())
            .field("names", &self.names)
            .finish()
    }
}

/// Errors surfaced by `build_registry`.
///
/// Spans + file labels are carried so a future caret-diagnostic renderer
/// can point at the exact source location. Wave 1.7 only renders a one-
/// liner via `Display`; the renderer itself is a separate task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryBuildError {
    /// Wave 1.6 lowering rejected one of the input decls. The wrapped
    /// error retains its own span; the registry pass adds no extra
    /// context here because lowering errors are always per-decl.
    Lower(LowerError),
    /// Two `ability` decls share a name. The first occurrence wins the
    /// slot id; the duplicate is rejected. Both file labels + spans are
    /// reported so the renderer can show both call sites.
    DuplicateAbilityName {
        name:        String,
        first_file:  String,
        first_span:  Span,
        dup_file:    String,
        dup_span:    Span,
    },
    /// A `cast <Name>` verb references an ability that no input file
    /// declares. Carries the name of the casting ability (so the renderer
    /// can label the source) plus the missing target spelling.
    UnresolvedCastTarget {
        from_ability: String,
        target_name:  String,
        file:         String,
        span:         Span,
    },
    /// A cycle of cast edges was detected. `path` is the list of ability
    /// names along the back-edge, with the closing edge made explicit by
    /// repeating the cycle root at the tail. Examples:
    ///   * self-cast: `["A", "A"]`
    ///   * two-cycle: `["A", "B", "A"]`
    ///   * N-cycle:   `["A", "B", ..., "A"]`
    /// Path order matches the discovery order of the depth-first walk
    /// rooted at the lowest-id Pass-1 entry (deterministic across runs).
    CastCycle { path: Vec<String> },
}

impl std::fmt::Display for RegistryBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryBuildError::Lower(e) => write!(f, "lowering failed: {e}"),
            RegistryBuildError::DuplicateAbilityName {
                name, first_file, dup_file, ..
            } => write!(
                f,
                "duplicate ability name '{name}' (first declared in '{first_file}', redeclared in '{dup_file}')",
            ),
            RegistryBuildError::UnresolvedCastTarget {
                from_ability, target_name, file, ..
            } => write!(
                f,
                "ability '{from_ability}' (in '{file}') casts unknown ability '{target_name}'",
            ),
            RegistryBuildError::CastCycle { path } => {
                write!(f, "cast cycle detected: ")?;
                for (i, name) in path.iter().enumerate() {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{name}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for RegistryBuildError {}

impl From<LowerError> for RegistryBuildError {
    fn from(e: LowerError) -> Self {
        RegistryBuildError::Lower(e)
    }
}

/// Build a frozen `AbilityRegistry` from the parsed input set.
///
/// `files` is `(display_path, AbilityFile)` so the caller picks the file
/// label used in diagnostics — typically the path string passed to
/// `parse_ability_file`. Decls are registered in the order they appear:
/// file-major (input slice order) then decl-major (source order inside
/// each file). The resulting `AbilityId`s are stable across runs given
/// the same input order.
///
/// The pass is a four-step pipeline:
///   1. Build the `name -> AbilityId` table; reject duplicates.
///   2. Lower each decl (`ability_lower::lower_ability_decl`) and patch
///      every placeholder `EffectOp::CastAbility { ability, .. }` with
///      the resolved id from step 1.
///   3. Walk the resolved cast-edge graph for cycle detection.
///   4. Drive `AbilityRegistryBuilder::register()` in step-1 order.
pub fn build_registry(
    files: &[(String, AbilityFile)],
) -> Result<BuiltRegistry, RegistryBuildError> {
    // ---- Pass 1: name table + per-decl bookkeeping. -----------------------
    //
    // We keep `entries` parallel to the future registry slot order so
    // Pass 2 can index into it without an extra lookup. `names` is the
    // public output; `name_to_idx` is the working lookup used during the
    // lowering walk (one entry per registered ability).
    let mut names: HashMap<String, AbilityId> = HashMap::new();
    let mut entries: Vec<DeclEntry<'_>> = Vec::new();

    for (file_label, file) in files {
        for decl in &file.abilities {
            // 1-based id; slot index is `entries.len()` because slots
            // are assigned in the same order we push here.
            let raw = (entries.len() as u32) + 1;
            let id = AbilityId::new(raw)
                .expect("AbilityId::new(raw) — raw is always >= 1 here");

            if let Some(existing_id) = names.get(&decl.name) {
                // Look up the original entry so we can report its file
                // + span. `existing_id.slot()` indexes `entries` because
                // both vectors grow in lockstep (no removals).
                let existing = &entries[existing_id.slot()];
                return Err(RegistryBuildError::DuplicateAbilityName {
                    name:       decl.name.clone(),
                    first_file: existing.file.clone(),
                    first_span: existing.decl.span,
                    dup_file:   file_label.clone(),
                    dup_span:   decl.span,
                });
            }

            names.insert(decl.name.clone(), id);
            entries.push(DeclEntry { file: file_label.clone(), decl, id });
        }
    }

    // ---- Pass 2: lower each decl + patch every cast id. -------------------
    //
    // `programs` is built in the same order as `entries`; `cast_edges`
    // collects the resolved edge set used by Pass 3. The per-decl
    // ordering of the `EffectOp::CastAbility` slots inside a program
    // matches the AST order, by Wave 1.6's lockstep guarantee.
    let mut programs: Vec<AbilityProgram> = Vec::with_capacity(entries.len());
    let mut cast_edges: HashMap<AbilityId, Vec<AbilityId>> = HashMap::new();

    for entry in &entries {
        let mut program = lower_ability_decl(entry.decl)?;

        // Walk AST + lowered ops in lockstep. Wave 1.6 emits exactly one
        // `EffectOp` per `EffectStmt` (no fan-in / fan-out), so a shared
        // index is sufficient. Indices align even when the lowered op is
        // not a `cast` — we only mutate on the cast verb.
        for (stmt, op) in entry.decl.effects.iter().zip(program.effects.iter_mut()) {
            if stmt.verb != "cast" {
                continue;
            }
            // Wave 1.6 lockstep invariant: `cast` always lowers to
            // `CastAbility`. If this fails, the bug is in `ability_lower`,
            // not user input.
            let (ability_slot, _selector) = match op {
                EffectOp::CastAbility { ability, selector } => (ability, selector),
                other => panic!(
                    "invariant: cast verb did not lower to CastAbility (got {other:?})"
                ),
            };

            // The first arg is the target ability name; lowering already
            // accepted only `Ident` / `String`, so any other arg shape
            // would have surfaced as a `LowerError` above.
            let target_name = match stmt.args.first() {
                Some(EffectArg::Ident(s)) => s.clone(),
                Some(EffectArg::String(s)) => s.clone(),
                _ => unreachable!(
                    "Wave 1.6 lowering accepts cast only with Ident/String first arg"
                ),
            };

            match names.get(&target_name) {
                Some(target_id) => {
                    *ability_slot = *target_id;
                    cast_edges.entry(entry.id).or_default().push(*target_id);
                }
                None => {
                    return Err(RegistryBuildError::UnresolvedCastTarget {
                        from_ability: entry.decl.name.clone(),
                        target_name,
                        file:         entry.file.clone(),
                        span:         stmt.span,
                    });
                }
            }
        }

        programs.push(program);
    }

    // ---- Pass 3: cycle detection. -----------------------------------------
    //
    // Iterative DFS over the resolved cast-edge graph. `visited` is a
    // global dedup set so each node is fully explored at most once
    // across all DFS roots; `on_stack` is per-root and powers the back-
    // edge detection. Roots are visited in slot order so the cycle path
    // we surface is deterministic across runs.
    let id_to_name: HashMap<AbilityId, String> = entries
        .iter()
        .map(|e| (e.id, e.decl.name.clone()))
        .collect();

    let mut visited: HashSet<AbilityId> = HashSet::new();
    for entry in &entries {
        if visited.contains(&entry.id) {
            continue;
        }
        if let Some(cycle_ids) = find_cycle_from(entry.id, &cast_edges, &mut visited) {
            // Translate the id path back to names. Every id in the cycle
            // came from `entries` (Pass 1 is the only id source), so the
            // reverse lookup is total.
            let path = cycle_ids
                .into_iter()
                .map(|id| id_to_name.get(&id).cloned()
                    .expect("cycle id originated in Pass 1; reverse lookup must hit"))
                .collect();
            return Err(RegistryBuildError::CastCycle { path });
        }
    }

    // ---- Pass 4: drive the builder + sanity-check id assignment. ----------
    //
    // We register in `entries` order — the same order Pass 1 used to
    // assign ids — so the builder hands back ids that match. The assert
    // catches future drift if the iteration order ever changes (e.g. a
    // refactor that sorts entries).
    let mut builder = AbilityRegistryBuilder::new();
    for (entry, program) in entries.iter().zip(programs.into_iter()) {
        let assigned = builder.register(program);
        debug_assert_eq!(
            assigned, entry.id,
            "Pass-4 builder slot drifted from Pass-1 id assignment",
        );
    }

    Ok(BuiltRegistry { registry: builder.build(), names })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Per-decl bookkeeping carried from Pass 1 through Pass 4.
///
/// The borrow of `decl` is what ties `BuiltRegistry`'s build pass to the
/// caller's parsed AST lifetime — the function returns the registry by
/// value, so this struct is purely local scratch.
struct DeclEntry<'a> {
    file: String,
    decl: &'a AbilityDecl,
    id:   AbilityId,
}

/// Iterative DFS rooted at `start`. Returns the cycle path as a vector
/// of ids whose first and last entries are equal (closing the loop) when
/// a back-edge is found; returns `None` when the subtree is acyclic.
///
/// `visited` is mutated to include every node fully explored from this
/// root, so the caller's outer loop never re-explores them. `on_stack`
/// is local to this call so distinct DFS roots don't poison each
/// other's back-edge detection.
fn find_cycle_from(
    start:     AbilityId,
    edges:     &HashMap<AbilityId, Vec<AbilityId>>,
    visited:   &mut HashSet<AbilityId>,
) -> Option<Vec<AbilityId>> {
    // Each frame records (node, next-edge-index-to-try). When a frame's
    // edge cursor advances past the last child we pop it (post-order).
    // Pushing a new frame corresponds to descending; finding a child
    // already on the stack is the cycle hit.
    let mut stack: Vec<(AbilityId, usize)> = Vec::new();
    let mut on_stack_set: HashSet<AbilityId> = HashSet::new();
    let mut path: Vec<AbilityId> = Vec::new();

    stack.push((start, 0));
    on_stack_set.insert(start);
    path.push(start);

    while let Some((node, cursor)) = stack.last().copied() {
        let children: &[AbilityId] = edges.get(&node).map(|v| v.as_slice()).unwrap_or(&[]);
        if cursor < children.len() {
            // Advance this frame's cursor before recursing so the next
            // pop resumes at the correct sibling.
            let last = stack.last_mut().expect("stack non-empty in this branch");
            last.1 = cursor + 1;
            let child = children[cursor];

            if on_stack_set.contains(&child) {
                // Back edge — slice the path from `child` onward and
                // close the loop with `child` at the tail. `find` is
                // O(path-len), bounded by the number of distinct
                // abilities, which is fine for build-time.
                let start_idx = path.iter().position(|&n| n == child)
                    .expect("child is on stack therefore in path");
                let mut cycle: Vec<AbilityId> = path[start_idx..].to_vec();
                cycle.push(child);
                return Some(cycle);
            }
            if visited.contains(&child) {
                // Cross / forward edge into a fully-explored subtree —
                // not a cycle (cycles must touch the live stack).
                continue;
            }

            stack.push((child, 0));
            on_stack_set.insert(child);
            path.push(child);
        } else {
            // Finished `node` — mark visited globally + pop.
            visited.insert(node);
            on_stack_set.remove(&node);
            path.pop();
            stack.pop();
        }
    }

    None
}
