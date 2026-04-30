//! `CgStmt` — the compute-graph statement tree.
//!
//! Statements are the "body" form for compute ops that mutate state in
//! a structured way: physics rules and view-fold handlers. They are
//! sibling to [`crate::cg::CgExpr`]: expressions are pure and produce a
//! value, statements are effectful and produce a side effect on a
//! [`DataHandle`] or an event ring.
//!
//! # Variant set
//!
//! The CG layer ships three statement forms — assignment, event emit,
//! and conditional. The DSL surface AST has additional control-flow
//! and binding forms (`for`, `match`, `let`, `belief observe`); the
//! AST → CG lowering (Phase 2 of the plan) is responsible for
//! desugaring them — `for` unrolls or fuses into the dispatch shape,
//! `match` cascades into nested `If`s, `let` flattens via SSA expression
//! sharing, `BeliefObserve` decomposes into a sequence of `Assign`s
//! against the BeliefState SoA fields. Adding a new CG-level statement
//! variant is therefore a deliberate choice: it widens the set of
//! shapes every later layer (HIR/MIR/LIR + emit) must handle.
//!
//! # Arena vs node
//!
//! Like [`crate::cg::CgExpr`], statements form a tree but reference
//! their children by id (`CgStmtListId` for sub-bodies, `CgStmtId` for
//! sibling list entries). The [`StmtArena`] trait resolves statement
//! ids to nodes; [`crate::cg::expr::ExprArena`] resolves expression
//! ids. The actual `Vec<CgStmt>` arena lives in `CgProgram` (Task 1.5);
//! tests use `Vec<CgStmt>` directly via the `impl StmtArena for Vec`
//! provided here.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::{CgExprId, DataHandle};
use super::expr::{CgExpr, ExprArena};
use super::op::EventKindId;

// ---------------------------------------------------------------------------
// Newtype IDs
// ---------------------------------------------------------------------------

/// Stable id for a statement node in a [`StmtArena`]. Sibling to
/// [`CgExprId`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CgStmtId(pub u32);

/// Stable id for a [`CgStmtList`] in the program. The list arena is
/// separate from the statement arena: `CgStmtListId` indexes into a
/// `Vec<CgStmtList>`, each list holds its `Vec<CgStmtId>` of statements
/// in execution order.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CgStmtListId(pub u32);

/// Stable id for a field within an event variant. `event` names which
/// event variant; `index` is the 0-based position of the field in the
/// variant's declared field list. The pair maps to a concrete struct
/// field at emit time (the lowering resolves it against the program's
/// event table).
///
/// Encoding the (event, index) pair structurally — rather than a flat
/// `EventFieldId(u32)` — keeps the IR honest about the fact that field
/// indices only make sense relative to a particular event variant.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventField {
    pub event: EventKindId,
    pub index: u8,
}

impl fmt::Display for EventField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "event[#{}].field#{}", self.event.0, self.index)
    }
}

// ---------------------------------------------------------------------------
// CgStmt
// ---------------------------------------------------------------------------

/// Statement node in the compute-graph statement tree.
///
/// Every variant produces a side effect:
/// - `Assign` writes a value into a [`DataHandle`].
/// - `Emit` appends an event record into an event ring (the ring is
///   identified by the event variant's [`EventKindId`]; the lowering
///   resolves which ring at emit time).
/// - `If` branches between two sub-bodies. Both arms are themselves
///   `CgStmtList`s; the `else` branch is optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CgStmt {
    /// Write `value` into the slot named by `target`. The slot type is
    /// determined by the data handle (see [`crate::cg::data_handle_ty`]);
    /// the type checker (Task 1.6 well-formed pass) ensures `value`'s
    /// CgTy matches.
    Assign { target: DataHandle, value: CgExprId },

    /// Emit an event record. `event` selects the event variant;
    /// `fields` lists each declared field of that variant in order
    /// with the expression that computes its payload. Field indices
    /// not present in `fields` carry the variant's default value at
    /// emit time (the well-formed pass enforces full coverage).
    Emit {
        event: EventKindId,
        fields: Vec<(EventField, CgExprId)>,
    },

    /// Conditional execution. `cond` is a Bool-typed expression;
    /// `then` runs when it is true, `else_` (if present) runs when
    /// false.
    If {
        cond: CgExprId,
        then: CgStmtListId,
        else_: Option<CgStmtListId>,
    },
}

impl fmt::Display for CgStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgStmt::Assign { target, value } => {
                write!(f, "assign({} <- expr#{})", target, value.0)
            }
            CgStmt::Emit { event, fields } => {
                write!(f, "emit(event[#{}]", event.0)?;
                for (fld, expr) in fields {
                    write!(f, ", {}=expr#{}", fld, expr.0)?;
                }
                f.write_str(")")
            }
            CgStmt::If { cond, then, else_ } => match else_ {
                Some(else_id) => write!(
                    f,
                    "if(cond=expr#{}, then=stmts#{}, else=stmts#{})",
                    cond.0, then.0, else_id.0
                ),
                None => write!(f, "if(cond=expr#{}, then=stmts#{})", cond.0, then.0),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// CgStmtList
// ---------------------------------------------------------------------------

/// Ordered list of statement ids — one body in the program. The list
/// arena (in `CgProgram`, Task 1.5) holds many of these, indexed by
/// [`CgStmtListId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CgStmtList {
    pub stmts: Vec<CgStmtId>,
}

impl CgStmtList {
    pub fn new(stmts: Vec<CgStmtId>) -> Self {
        Self { stmts }
    }

    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.stmts.len()
    }
}

impl fmt::Display for CgStmtList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, id) in self.stmts.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "stmt#{}", id.0)?;
        }
        f.write_str("]")
    }
}

// ---------------------------------------------------------------------------
// StmtArena + ListArena
// ---------------------------------------------------------------------------

/// Resolves a [`CgStmtId`] to its underlying [`CgStmt`]. Sibling to
/// [`crate::cg::expr::ExprArena`]. Implementations:
/// - `CgProgram` (Task 1.5) implements it against its `Vec<CgStmt>`
///   arena.
/// - `[CgStmt]` and `Vec<CgStmt>` impls below let tests carry a tiny
///   arena directly.
pub trait StmtArena {
    fn get(&self, id: CgStmtId) -> &CgStmt;
}

impl StmtArena for [CgStmt] {
    fn get(&self, id: CgStmtId) -> &CgStmt {
        &self[id.0 as usize]
    }
}

impl StmtArena for Vec<CgStmt> {
    fn get(&self, id: CgStmtId) -> &CgStmt {
        &self[id.0 as usize]
    }
}

/// Resolves a [`CgStmtListId`] to its underlying [`CgStmtList`]. Same
/// shape as [`StmtArena`] but for list ids — kept separate so the
/// program can store statement nodes and statement lists in different
/// arenas (which `CgProgram` does — see Task 1.5).
pub trait StmtListArena {
    fn get(&self, id: CgStmtListId) -> &CgStmtList;
}

impl StmtListArena for [CgStmtList] {
    fn get(&self, id: CgStmtListId) -> &CgStmtList {
        &self[id.0 as usize]
    }
}

impl StmtListArena for Vec<CgStmtList> {
    fn get(&self, id: CgStmtListId) -> &CgStmtList {
        &self[id.0 as usize]
    }
}

// ---------------------------------------------------------------------------
// Dependency walks
// ---------------------------------------------------------------------------

/// Walk an expression tree rooted at `id` and append every
/// `CgExpr::Read(handle)` it contains into `out`. Order is depth-first
/// in operand order (lhs before rhs, args left-to-right, condition
/// before then before else), which makes the walk deterministic — the
/// same input arena always produces the same output ordering, including
/// duplicates.
pub fn collect_expr_reads(id: CgExprId, exprs: &dyn ExprArena, out: &mut Vec<DataHandle>) {
    let node = exprs.get(id);
    match node {
        CgExpr::Read(h) => out.push(h.clone()),
        CgExpr::Lit(_) => {}
        CgExpr::Binary { lhs, rhs, .. } => {
            collect_expr_reads(*lhs, exprs, out);
            collect_expr_reads(*rhs, exprs, out);
        }
        CgExpr::Unary { arg, .. } => {
            collect_expr_reads(*arg, exprs, out);
        }
        CgExpr::Builtin { args, .. } => {
            for a in args {
                collect_expr_reads(*a, exprs, out);
            }
        }
        CgExpr::Rng { purpose, .. } => {
            // RNG draws are reads of the deterministic per-agent stream;
            // record the handle so dependency analysis knows the op
            // touches the seed.
            out.push(DataHandle::Rng { purpose: *purpose });
        }
        CgExpr::Select {
            cond, then, else_, ..
        } => {
            collect_expr_reads(*cond, exprs, out);
            collect_expr_reads(*then, exprs, out);
            collect_expr_reads(*else_, exprs, out);
        }
    }
}

/// Walk a single statement and append its (reads, writes) into the
/// supplied buffers. For nested `If` bodies the walk recurses through
/// `lists` to fetch the sub-`CgStmtList`s.
///
/// Order: depth-first in source-code order. `Assign` records the RHS
/// expression's reads first, then the assignment's target as a write.
/// `Emit` records each field expression's reads in field order; the
/// destination ring write is **not** synthesized here — the
/// [`EventKindId`] alone doesn't determine which ring stores the
/// event (rings are an emit-time concept resolved by lowering against
/// the event registry). The AST → HIR lowering pass adds the ring
/// write to the enclosing op's `writes` via
/// [`crate::cg::op::ComputeOp::record_write`]. `If` records the
/// condition's reads, then walks the then-arm (in order), then the
/// else-arm (if any).
pub fn collect_stmt_dependencies(
    id: CgStmtId,
    exprs: &dyn ExprArena,
    stmts: &dyn StmtArena,
    lists: &dyn StmtListArena,
    reads: &mut Vec<DataHandle>,
    writes: &mut Vec<DataHandle>,
) {
    let node = stmts.get(id);
    match node {
        CgStmt::Assign { target, value } => {
            collect_expr_reads(*value, exprs, reads);
            writes.push(target.clone());
        }
        CgStmt::Emit { event: _, fields } => {
            for (_, expr_id) in fields {
                collect_expr_reads(*expr_id, exprs, reads);
            }
            // No synthesized ring write here — see the doc comment
            // above for the rationale.
        }
        CgStmt::If { cond, then, else_ } => {
            collect_expr_reads(*cond, exprs, reads);
            collect_list_dependencies(*then, exprs, stmts, lists, reads, writes);
            if let Some(else_id) = else_ {
                collect_list_dependencies(*else_id, exprs, stmts, lists, reads, writes);
            }
        }
    }
}

/// Walk every statement in a [`CgStmtList`] in order and append the
/// union of their (reads, writes).
pub fn collect_list_dependencies(
    list_id: CgStmtListId,
    exprs: &dyn ExprArena,
    stmts: &dyn StmtArena,
    lists: &dyn StmtListArena,
    reads: &mut Vec<DataHandle>,
    writes: &mut Vec<DataHandle>,
) {
    let list = lists.get(list_id);
    for stmt_id in &list.stmts {
        collect_stmt_dependencies(*stmt_id, exprs, stmts, lists, reads, writes);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{AgentFieldId, AgentRef, CgExprId, RngPurpose};
    use crate::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
    use crate::cg::op::EventKindId;

    fn assert_roundtrip<T>(v: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let json = serde_json::to_string(v).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, v, "round-trip changed value (json was {json})");
    }

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn read_self_pos() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Pos,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    fn lit_bool(b: bool) -> CgExpr {
        CgExpr::Lit(LitValue::Bool(b))
    }

    // ---- newtype roundtrips ----

    #[test]
    fn cg_stmt_id_roundtrip() {
        assert_roundtrip(&CgStmtId(0));
        assert_roundtrip(&CgStmtId(42));
    }

    #[test]
    fn cg_stmt_list_id_roundtrip() {
        assert_roundtrip(&CgStmtListId(0));
        assert_roundtrip(&CgStmtListId(7));
    }

    #[test]
    fn event_field_display_and_roundtrip() {
        let ef = EventField {
            event: EventKindId(2),
            index: 3,
        };
        assert_eq!(format!("{}", ef), "event[#2].field#3");
        assert_roundtrip(&ef);
    }

    // ---- CgStmt display + roundtrip ----

    #[test]
    fn assign_display_and_roundtrip() {
        let s = CgStmt::Assign {
            target: DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            value: CgExprId(4),
        };
        assert_eq!(format!("{}", s), "assign(agent.self.hp <- expr#4)");
        assert_roundtrip(&s);
    }

    #[test]
    fn emit_display_and_roundtrip() {
        let s = CgStmt::Emit {
            event: EventKindId(1),
            fields: vec![
                (
                    EventField {
                        event: EventKindId(1),
                        index: 0,
                    },
                    CgExprId(2),
                ),
                (
                    EventField {
                        event: EventKindId(1),
                        index: 1,
                    },
                    CgExprId(3),
                ),
            ],
        };
        assert_eq!(
            format!("{}", s),
            "emit(event[#1], event[#1].field#0=expr#2, event[#1].field#1=expr#3)"
        );
        assert_roundtrip(&s);
    }

    #[test]
    fn if_with_else_display_and_roundtrip() {
        let s = CgStmt::If {
            cond: CgExprId(1),
            then: CgStmtListId(2),
            else_: Some(CgStmtListId(3)),
        };
        assert_eq!(
            format!("{}", s),
            "if(cond=expr#1, then=stmts#2, else=stmts#3)"
        );
        assert_roundtrip(&s);
    }

    #[test]
    fn if_without_else_display_and_roundtrip() {
        let s = CgStmt::If {
            cond: CgExprId(0),
            then: CgStmtListId(1),
            else_: None,
        };
        assert_eq!(format!("{}", s), "if(cond=expr#0, then=stmts#1)");
        assert_roundtrip(&s);
    }

    // ---- CgStmtList display + roundtrip ----

    #[test]
    fn stmt_list_empty_display() {
        let l = CgStmtList::new(vec![]);
        assert_eq!(format!("{}", l), "[]");
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        assert_roundtrip(&l);
    }

    #[test]
    fn stmt_list_two_entries_display() {
        let l = CgStmtList::new(vec![CgStmtId(0), CgStmtId(1)]);
        assert_eq!(format!("{}", l), "[stmt#0, stmt#1]");
        assert_eq!(l.len(), 2);
        assert!(!l.is_empty());
        assert_roundtrip(&l);
    }

    // ---- Arena impls ----

    #[test]
    fn stmt_arena_resolves_by_id() {
        let arena: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            },
            CgStmt::If {
                cond: CgExprId(1),
                then: CgStmtListId(0),
                else_: None,
            },
        ];
        let s0 = StmtArena::get(&arena, CgStmtId(0));
        match s0 {
            CgStmt::Assign { value, .. } => assert_eq!(*value, CgExprId(0)),
            _ => panic!("expected Assign at #0"),
        }
        let s1 = StmtArena::get(&arena, CgStmtId(1));
        match s1 {
            CgStmt::If { cond, .. } => assert_eq!(*cond, CgExprId(1)),
            _ => panic!("expected If at #1"),
        }
    }

    #[test]
    fn stmt_list_arena_resolves_by_id() {
        let arena: Vec<CgStmtList> = vec![
            CgStmtList::new(vec![]),
            CgStmtList::new(vec![CgStmtId(7)]),
        ];
        let l0 = StmtListArena::get(&arena, CgStmtListId(0));
        assert!(l0.is_empty());
        let l1 = StmtListArena::get(&arena, CgStmtListId(1));
        assert_eq!(l1.stmts, vec![CgStmtId(7)]);
    }

    // ---- Dependency walks ----

    #[test]
    fn collect_expr_reads_lit_is_empty() {
        let exprs: Vec<CgExpr> = vec![lit_f32(1.0)];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(0), &exprs, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_expr_reads_binary_lhs_then_rhs() {
        // (hp + 1.0) — reads = [agent.self.hp]
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(1.0),   // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2
        ];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(2), &exprs, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
    }

    #[test]
    fn collect_expr_reads_select_walks_all_three_arms() {
        // select(true, hp, pos) -- pos is wrong type for then/else, but
        // the read collector doesn't care about types.
        let exprs: Vec<CgExpr> = vec![
            lit_bool(true), // 0
            read_self_hp(), // 1
            read_self_pos(),// 2
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::F32,
            }, // 3
        ];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(3), &exprs, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            out[1],
            DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }
        );
    }

    #[test]
    fn collect_expr_reads_rng_records_handle() {
        let exprs: Vec<CgExpr> = vec![CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::U32,
        }];
        let mut out = Vec::new();
        collect_expr_reads(CgExprId(0), &exprs, &mut out);
        assert_eq!(
            out,
            vec![DataHandle::Rng {
                purpose: RngPurpose::Action,
            }]
        );
    }

    #[test]
    fn collect_stmt_dependencies_assign_reads_value_writes_target() {
        // assign(agent.self.hp <- (agent.self.hp + 1.0))
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(1.0),   // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![CgStmt::Assign {
            target: DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            value: CgExprId(2),
        }];
        let lists: Vec<CgStmtList> = vec![];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        assert_eq!(
            reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(
            writes,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
    }

    #[test]
    fn collect_stmt_dependencies_emit_walks_field_value_reads_only() {
        // The walker descends into each field-value expression for
        // reads, but does NOT synthesize a destination event-ring
        // write — ring binding is a lowering concern (see the doc
        // comment on `collect_stmt_dependencies`).
        let exprs: Vec<CgExpr> = vec![read_self_hp(), lit_f32(0.0)];
        let stmts: Vec<CgStmt> = vec![CgStmt::Emit {
            event: EventKindId(7),
            fields: vec![
                (
                    EventField {
                        event: EventKindId(7),
                        index: 0,
                    },
                    CgExprId(0),
                ),
                (
                    EventField {
                        event: EventKindId(7),
                        index: 1,
                    },
                    CgExprId(1),
                ),
            ],
        }];
        let lists: Vec<CgStmtList> = vec![];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Reads: hp from the first field's value expr.
        assert_eq!(reads.len(), 1);
        assert!(reads.contains(&DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }));
        // Writes: none — the `Emit`'s ring binding is added by lowering.
        assert!(writes.is_empty(), "expected no writes, got {writes:?}");
    }

    #[test]
    fn collect_stmt_dependencies_if_walks_both_arms() {
        // if (true) { assign(hp <- 1.0) } else { assign(hp <- 0.0) }
        let exprs: Vec<CgExpr> = vec![
            lit_bool(true), // 0
            lit_f32(1.0),   // 1
            lit_f32(0.0),   // 2
        ];
        let stmts: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(1),
            }, // 0
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(2),
            }, // 1
            CgStmt::If {
                cond: CgExprId(0),
                then: CgStmtListId(0),
                else_: Some(CgStmtListId(1)),
            }, // 2
        ];
        let lists: Vec<CgStmtList> = vec![
            CgStmtList::new(vec![CgStmtId(0)]),
            CgStmtList::new(vec![CgStmtId(1)]),
        ];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_stmt_dependencies(CgStmtId(2), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Both arms write hp; cond is a literal so no reads from there.
        assert!(reads.is_empty());
        assert_eq!(writes.len(), 2);
    }

    #[test]
    fn collect_list_dependencies_walks_in_order() {
        // [assign(hp <- 1.0), emit(event#3)]
        let exprs: Vec<CgExpr> = vec![
            lit_f32(1.0),   // 0
            read_self_hp(), // 1
        ];
        let stmts: Vec<CgStmt> = vec![
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            }, // 0
            CgStmt::Emit {
                event: EventKindId(3),
                fields: vec![(
                    EventField {
                        event: EventKindId(3),
                        index: 0,
                    },
                    CgExprId(1),
                )],
            }, // 1
        ];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![CgStmtId(0), CgStmtId(1)])];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        collect_list_dependencies(CgStmtListId(0), &exprs, &stmts, &lists, &mut reads, &mut writes);
        // Reads: emit's field expr reads hp.
        assert_eq!(reads.len(), 1);
        // Writes: just assign's hp — the emit's destination ring is
        // bound by lowering, not by the auto-walker.
        assert_eq!(writes.len(), 1);
        match &writes[0] {
            DataHandle::AgentField { field, .. } => assert_eq!(*field, AgentFieldId::Hp),
            _ => panic!("expected hp write"),
        }
    }
}
