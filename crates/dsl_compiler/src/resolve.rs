//! Two-pass name resolution: AST → IR.
//!
//! Pass 1: collect all top-level decl names into a `SymbolTable`, assign IR
//! indices. Duplicate names (same kind) are errors.
//! Pass 2: walk each decl's bodies, resolving identifiers against a local
//! scope stack, the stdlib symbol table, and the top-level decls.
//!
//! Unresolvable call callees become `UnresolvedCall` (flagged for a later
//! milestone, 1b). Bare unresolved identifiers are errors.

use std::collections::HashMap;

use crate::ast::{self, ActionHeadShape, AssertExpr, Decl, Expr, ExprKind, Program, Span, Stmt};
use crate::ir::*;
use crate::resolve_error::ResolveError;

// ---------------------------------------------------------------------------
// Stdlib symbol table
// ---------------------------------------------------------------------------

mod stdlib {
    use super::*;

    pub fn seed(symbols: &mut SymbolTable) {
        let prims = [
            ("bool", IrType::Bool),
            ("i32", IrType::I32),
            ("u32", IrType::U32),
            ("i64", IrType::I64),
            ("u64", IrType::U64),
            ("f32", IrType::F32),
            ("f64", IrType::F64),
            ("vec3", IrType::Vec3),
            ("string", IrType::String),
            ("String", IrType::String),
            ("AgentId", IrType::AgentId),
            ("ItemId", IrType::ItemId),
            ("GroupId", IrType::GroupId),
            ("QuestId", IrType::QuestId),
            ("AuctionId", IrType::AuctionId),
            ("EventId", IrType::EventId),
            ("AbilityId", IrType::AbilityId),
            // Tick / other pseudo-primitives commonly seen in fixtures.
            ("Tick", IrType::U64),
        ];
        for (n, t) in prims {
            symbols.stdlib_types.insert(n.to_string(), t);
        }
        symbols.builtins.insert("count".into(), Builtin::Count);
        symbols.builtins.insert("sum".into(), Builtin::Sum);
        symbols.builtins.insert("min".into(), Builtin::Min);
        symbols.builtins.insert("max".into(), Builtin::Max);
        symbols.builtins.insert("distance".into(), Builtin::Distance);
        symbols.builtins.insert("planar_distance".into(), Builtin::PlanarDistance);
        symbols.builtins.insert("z_separation".into(), Builtin::ZSeparation);
        symbols.builtins.insert("entity".into(), Builtin::Entity);
        symbols.builtins.insert("forall".into(), Builtin::Forall);
        symbols.builtins.insert("exists".into(), Builtin::Exists);

        for ns in [
            "cascade", "event", "agents", "items", "groups", "quests",
            "auctions", "mask", "action", "world", "tick", "rng", "query",
        ] {
            symbols.stdlib_namespaces.insert(ns.to_string());
        }
    }
}

// ---------------------------------------------------------------------------
// Symbol table
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SymbolTable {
    pub events: HashMap<String, EventRef>,
    pub entities: HashMap<String, EntityRef>,
    pub physics: HashMap<String, PhysicsRef>,
    pub masks: HashMap<String, MaskRef>,
    pub scoring: HashMap<String, ScoringRef>,
    pub views: HashMap<String, ViewRef>,
    pub verbs: HashMap<String, VerbRef>,
    pub invariants: HashMap<String, InvariantRef>,
    pub probes: HashMap<String, ProbeRef>,
    pub metrics: HashMap<String, MetricRef>,
    pub builtins: HashMap<String, Builtin>,
    pub stdlib_types: HashMap<String, IrType>,
    /// Sim-wide accessor namespaces: `cascade`, `event`, `agents`, `mask`,
    /// `action`. They behave as opaque identifiers for 1a; 1b will type-check
    /// the fields / calls hanging off them.
    pub stdlib_namespaces: std::collections::HashSet<String>,
    // Span of first declaration — for duplicate-decl diagnostics.
    pub first_span: HashMap<(&'static str, String), Span>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut s = Self::default();
        stdlib::seed(&mut s);
        s
    }

    fn record_first(&mut self, kind: &'static str, name: &str, span: Span) {
        self.first_span.insert((kind, name.to_string()), span);
    }

    fn first_of(&self, kind: &'static str, name: &str) -> Option<Span> {
        self.first_span.get(&(kind, name.to_string())).copied()
    }
}

// ---------------------------------------------------------------------------
// Local scope (stacked)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LocalBinding {
    name: String,
    local: LocalRef,
    #[allow(dead_code)]
    ty: IrType,
}

#[derive(Debug, Default)]
struct LocalScope {
    stack: Vec<Vec<LocalBinding>>,
    next_id: u16,
    // Tracks whether `self` has been bound in the current decl.
    self_bound: bool,
}

impl LocalScope {
    fn new() -> Self {
        LocalScope { stack: vec![Vec::new()], next_id: 0, self_bound: false }
    }

    fn push(&mut self) {
        self.stack.push(Vec::new());
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    fn fresh(&mut self) -> LocalRef {
        let r = LocalRef(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        r
    }

    fn bind(&mut self, name: &str, ty: IrType) -> LocalRef {
        let local = self.fresh();
        if name == "self" {
            self.self_bound = true;
        }
        self.stack
            .last_mut()
            .unwrap()
            .push(LocalBinding { name: name.to_string(), local, ty });
        local
    }

    fn lookup(&self, name: &str) -> Option<&LocalBinding> {
        for frame in self.stack.iter().rev() {
            for b in frame.iter().rev() {
                if b.name == name {
                    return Some(b);
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn resolve(program: Program) -> Result<Compilation, ResolveError> {
    let mut symbols = SymbolTable::new();
    let mut comp = Compilation::default();

    // Pass 1: collect top-level names + reserve IR slots (still empty).
    collect(&program, &mut symbols, &mut comp)?;

    // Pass 2: resolve bodies into the reserved slots.
    resolve_bodies(&program, &symbols, &mut comp)?;

    Ok(comp)
}

// ---------------------------------------------------------------------------
// Pass 1: collect
// ---------------------------------------------------------------------------

fn collect(
    program: &Program,
    symbols: &mut SymbolTable,
    comp: &mut Compilation,
) -> Result<(), ResolveError> {
    // We pre-allocate empty IR shells with the right names/spans so indices
    // are stable. Pass 2 will overwrite the bodies.
    for decl in &program.decls {
        match decl {
            Decl::Event(d) => {
                check_dup(symbols, "event", &d.name, d.span, |s| s.events.contains_key(&d.name))?;
                let idx = push_idx(comp.events.len(), "event")?;
                symbols.events.insert(d.name.clone(), EventRef(idx));
                symbols.record_first("event", &d.name, d.span);
                comp.events.push(EventIR {
                    name: d.name.clone(),
                    fields: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Entity(d) => {
                check_dup(symbols, "entity", &d.name, d.span, |s| s.entities.contains_key(&d.name))?;
                let idx = push_idx(comp.entities.len(), "entity")?;
                symbols.entities.insert(d.name.clone(), EntityRef(idx));
                symbols.record_first("entity", &d.name, d.span);
                comp.entities.push(EntityIR {
                    name: d.name.clone(),
                    root: d.root,
                    fields: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Physics(d) => {
                check_dup(symbols, "physics", &d.name, d.span, |s| s.physics.contains_key(&d.name))?;
                let idx = push_idx(comp.physics.len(), "physics")?;
                symbols.physics.insert(d.name.clone(), PhysicsRef(idx));
                symbols.record_first("physics", &d.name, d.span);
                comp.physics.push(PhysicsIR {
                    name: d.name.clone(),
                    handlers: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Mask(d) => {
                let key = d.head.name.clone();
                check_dup(symbols, "mask", &key, d.span, |s| s.masks.contains_key(&key))?;
                let idx = push_idx(comp.masks.len(), "mask")?;
                symbols.masks.insert(key.clone(), MaskRef(idx));
                symbols.record_first("mask", &key, d.span);
                comp.masks.push(MaskIR {
                    head: IrActionHead {
                        name: d.head.name.clone(),
                        shape: IrActionHeadShape::None,
                        span: d.head.span,
                    },
                    predicate: IrExprNode { kind: IrExpr::LitBool(true), span: d.span },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Scoring(d) => {
                // Scoring blocks are unnamed; use synthetic name keyed by
                // index + span. Duplicates are tolerated (multiple blocks are
                // allowed per spec).
                let synthetic = format!("__scoring_{}", comp.scoring.len());
                let idx = push_idx(comp.scoring.len(), "scoring")?;
                symbols.scoring.insert(synthetic, ScoringRef(idx));
                comp.scoring.push(ScoringIR {
                    entries: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::View(d) => {
                check_dup(symbols, "view", &d.name, d.span, |s| s.views.contains_key(&d.name))?;
                let idx = push_idx(comp.views.len(), "view")?;
                symbols.views.insert(d.name.clone(), ViewRef(idx));
                symbols.record_first("view", &d.name, d.span);
                comp.views.push(ViewIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    return_ty: IrType::Unknown,
                    body: ViewBodyIR::Expr(IrExprNode { kind: IrExpr::LitBool(true), span: d.span }),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Verb(d) => {
                check_dup(symbols, "verb", &d.name, d.span, |s| s.verbs.contains_key(&d.name))?;
                let idx = push_idx(comp.verbs.len(), "verb")?;
                symbols.verbs.insert(d.name.clone(), VerbRef(idx));
                symbols.record_first("verb", &d.name, d.span);
                comp.verbs.push(VerbIR {
                    name: d.name.clone(),
                    params: Vec::new(),
                    action: VerbActionIR {
                        name: d.action.name.clone(),
                        args: Vec::new(),
                        span: d.action.span,
                    },
                    when: None,
                    emits: Vec::new(),
                    scoring: None,
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Invariant(d) => {
                check_dup(symbols, "invariant", &d.name, d.span, |s| {
                    s.invariants.contains_key(&d.name)
                })?;
                let idx = push_idx(comp.invariants.len(), "invariant")?;
                symbols.invariants.insert(d.name.clone(), InvariantRef(idx));
                symbols.record_first("invariant", &d.name, d.span);
                comp.invariants.push(InvariantIR {
                    name: d.name.clone(),
                    scope: Vec::new(),
                    mode: d.mode,
                    predicate: IrExprNode { kind: IrExpr::LitBool(true), span: d.span },
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Probe(d) => {
                check_dup(symbols, "probe", &d.name, d.span, |s| s.probes.contains_key(&d.name))?;
                let idx = push_idx(comp.probes.len(), "probe")?;
                symbols.probes.insert(d.name.clone(), ProbeRef(idx));
                symbols.record_first("probe", &d.name, d.span);
                comp.probes.push(ProbeIR {
                    name: d.name.clone(),
                    scenario: d.scenario.clone(),
                    seed: d.seed,
                    seeds: d.seeds.clone(),
                    ticks: d.ticks,
                    tolerance: d.tolerance,
                    asserts: Vec::new(),
                    annotations: d.annotations.clone(),
                    span: d.span,
                });
            }
            Decl::Metric(block) => {
                for m in &block.metrics {
                    check_dup(symbols, "metric", &m.name, m.span, |s| {
                        s.metrics.contains_key(&m.name)
                    })?;
                    let idx = push_idx(comp.metrics.len(), "metric")?;
                    symbols.metrics.insert(m.name.clone(), MetricRef(idx));
                    symbols.record_first("metric", &m.name, m.span);
                    comp.metrics.push(MetricIR {
                        name: m.name.clone(),
                        value: IrExprNode { kind: IrExpr::LitBool(true), span: m.span },
                        window: m.window,
                        emit_every: m.emit_every,
                        conditioned_on: None,
                        alert_when: None,
                        annotations: block.annotations.clone(),
                        span: m.span,
                    });
                }
            }
            Decl::Query(_) => {
                // Queries are not a milestone-1a surface yet. Skip silently;
                // 1b will handle.
            }
        }
    }
    Ok(())
}

fn push_idx(len: usize, kind: &'static str) -> Result<u16, ResolveError> {
    u16::try_from(len).map_err(|_| ResolveError::TooManyDecls { kind })
}

fn check_dup(
    symbols: &SymbolTable,
    kind: &'static str,
    name: &str,
    second: Span,
    contains: impl FnOnce(&SymbolTable) -> bool,
) -> Result<(), ResolveError> {
    if contains(symbols) {
        let first = symbols.first_of(kind, name).unwrap_or(Span::dummy());
        return Err(ResolveError::DuplicateDecl {
            kind,
            name: name.to_string(),
            first,
            second,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pass 2: resolve bodies
// ---------------------------------------------------------------------------

fn resolve_bodies(
    program: &Program,
    symbols: &SymbolTable,
    comp: &mut Compilation,
) -> Result<(), ResolveError> {
    let mut event_idx = 0;
    let mut entity_idx = 0;
    let mut physics_idx = 0;
    let mut mask_idx = 0;
    let mut scoring_idx = 0;
    let mut view_idx = 0;
    let mut verb_idx = 0;
    let mut invariant_idx = 0;
    let mut probe_idx = 0;
    let mut metric_start_idx = 0usize;

    for decl in &program.decls {
        match decl {
            Decl::Event(d) => {
                let fields = d
                    .fields
                    .iter()
                    .map(|f| EventField {
                        name: f.name.clone(),
                        ty: resolve_type(&f.ty, symbols),
                        span: f.span,
                    })
                    .collect();
                comp.events[event_idx].fields = fields;
                event_idx += 1;
            }
            Decl::Entity(d) => {
                let fields = d
                    .fields
                    .iter()
                    .map(|f| resolve_entity_field(f, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                comp.entities[entity_idx].fields = fields;
                entity_idx += 1;
            }
            Decl::Physics(d) => {
                let handlers = d
                    .handlers
                    .iter()
                    .map(|h| {
                        let mut scope = LocalScope::new();
                        // self is implicit in physics handlers (the entity
                        // whose action/event is being handled).
                        scope.bind("self", IrType::Unknown);
                        let pattern = resolve_event_pattern(&h.pattern, &mut scope, symbols);
                        let where_clause = h
                            .where_clause
                            .as_ref()
                            .map(|w| resolve_expr(w, &mut scope, symbols))
                            .transpose()?;
                        let body = resolve_stmts(&h.body, &mut scope, symbols)?;
                        Ok::<_, ResolveError>(PhysicsHandlerIR {
                            pattern,
                            where_clause,
                            body,
                            span: h.span,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.physics[physics_idx].handlers = handlers;
                physics_idx += 1;
            }
            Decl::Mask(d) => {
                let mut scope = LocalScope::new();
                scope.bind("self", IrType::Unknown);
                let head = resolve_action_head(&d.head, &mut scope, symbols);
                let predicate = resolve_expr(&d.predicate, &mut scope, symbols)?;
                comp.masks[mask_idx].head = head;
                comp.masks[mask_idx].predicate = predicate;
                mask_idx += 1;
            }
            Decl::Scoring(d) => {
                let entries = d
                    .entries
                    .iter()
                    .map(|e| {
                        let mut scope = LocalScope::new();
                        scope.bind("self", IrType::Unknown);
                        let head = resolve_action_head(&e.head, &mut scope, symbols);
                        let expr = resolve_expr(&e.expr, &mut scope, symbols)?;
                        Ok::<_, ResolveError>(ScoringEntryIR { head, expr, span: e.span })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.scoring[scoring_idx].entries = entries;
                scoring_idx += 1;
            }
            Decl::View(d) => {
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let return_ty = resolve_type(&d.return_ty, symbols);
                let body = match &d.body {
                    ast::ViewBody::Expr(e) => ViewBodyIR::Expr(resolve_expr(e, &mut scope, symbols)?),
                    ast::ViewBody::Fold { initial, handlers, clamp } => {
                        let initial = resolve_expr(initial, &mut scope, symbols)?;
                        let handlers_ir = handlers
                            .iter()
                            .map(|h| {
                                let mut inner = LocalScope::new();
                                // Copy outer scope bindings into the inner
                                // scope so fold handlers see the view
                                // parameters.
                                for binding in scope.stack.iter().flatten() {
                                    inner.stack[0].push(binding.clone());
                                }
                                inner.next_id = scope.next_id;
                                let pattern =
                                    resolve_event_pattern(&h.pattern, &mut inner, symbols);
                                let body = resolve_stmts(&h.body, &mut inner, symbols)?;
                                Ok::<_, ResolveError>(FoldHandlerIR {
                                    pattern,
                                    body,
                                    span: h.span,
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let clamp = match clamp {
                            Some((lo, hi)) => Some((
                                resolve_expr(lo, &mut scope, symbols)?,
                                resolve_expr(hi, &mut scope, symbols)?,
                            )),
                            None => None,
                        };
                        ViewBodyIR::Fold { initial, handlers: handlers_ir, clamp }
                    }
                };
                comp.views[view_idx].params = params;
                comp.views[view_idx].return_ty = return_ty;
                comp.views[view_idx].body = body;
                view_idx += 1;
            }
            Decl::Verb(d) => {
                let mut scope = LocalScope::new();
                let params = resolve_params(&d.params, &mut scope, symbols);
                let action_args = d
                    .action
                    .args
                    .iter()
                    .map(|a| resolve_call_arg(a, &mut scope, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                let action = VerbActionIR {
                    name: d.action.name.clone(),
                    args: action_args,
                    span: d.action.span,
                };
                let when = d
                    .when
                    .as_ref()
                    .map(|e| resolve_expr(e, &mut scope, symbols))
                    .transpose()?;
                let emits = d
                    .emits
                    .iter()
                    .map(|e| resolve_emit(e, &mut scope, symbols))
                    .collect::<Result<Vec<_>, _>>()?;
                let scoring = d
                    .scoring
                    .as_ref()
                    .map(|e| resolve_expr(e, &mut scope, symbols))
                    .transpose()?;
                comp.verbs[verb_idx].params = params;
                comp.verbs[verb_idx].action = action;
                comp.verbs[verb_idx].when = when;
                comp.verbs[verb_idx].emits = emits;
                comp.verbs[verb_idx].scoring = scoring;
                verb_idx += 1;
            }
            Decl::Invariant(d) => {
                let mut scope = LocalScope::new();
                // Invariants don't have an implicit self — only their scope
                // params. A metric / probe / invariant that mentions `self`
                // without a param is an error (SelfInTopLevel).
                let scope_params = resolve_params(&d.scope, &mut scope, symbols);
                let predicate = resolve_expr(&d.predicate, &mut scope, symbols)?;
                comp.invariants[invariant_idx].scope = scope_params;
                comp.invariants[invariant_idx].predicate = predicate;
                invariant_idx += 1;
            }
            Decl::Probe(d) => {
                let asserts = d
                    .asserts
                    .iter()
                    .map(|a| {
                        let mut scope = LocalScope::new();
                        scope.bind("self", IrType::Unknown);
                        scope.bind("action", IrType::Unknown);
                        resolve_assert(a, &mut scope, symbols)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                comp.probes[probe_idx].asserts = asserts;
                probe_idx += 1;
            }
            Decl::Metric(block) => {
                for m in &block.metrics {
                    let mut scope = LocalScope::new();
                    let value = resolve_expr(&m.value, &mut scope, symbols)?;
                    let cond = m
                        .conditioned_on
                        .as_ref()
                        .map(|e| resolve_expr(e, &mut scope, symbols))
                        .transpose()?;
                    // `alert when` clauses see implicit bindings: `value`
                    // (scalar metrics), `max_bin` (histograms). Bind both as
                    // Unknown so 1b can specialize.
                    let mut alert_scope = LocalScope::new();
                    alert_scope.bind("value", IrType::Unknown);
                    alert_scope.bind("max_bin", IrType::Unknown);
                    let alert = m
                        .alert_when
                        .as_ref()
                        .map(|e| resolve_expr(e, &mut alert_scope, symbols))
                        .transpose()?;
                    let slot = metric_start_idx;
                    comp.metrics[slot].value = value;
                    comp.metrics[slot].conditioned_on = cond;
                    comp.metrics[slot].alert_when = alert;
                    metric_start_idx += 1;
                }
            }
            Decl::Query(_) => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn resolve_type(ty: &ast::TypeRef, symbols: &SymbolTable) -> IrType {
    match &ty.kind {
        ast::TypeKind::Named(n) => {
            if let Some(t) = symbols.stdlib_types.get(n) {
                return t.clone();
            }
            if let Some(r) = symbols.entities.get(n) {
                return IrType::EntityRef(*r);
            }
            if let Some(r) = symbols.events.get(n) {
                return IrType::EventRef(*r);
            }
            // Probably a user-defined enum / struct we don't have a decl kind
            // for in 1a — keep as Named.
            IrType::Named(n.clone())
        }
        ast::TypeKind::Generic { name, args } => match name.as_str() {
            "SortedVec" | "RingBuffer" | "SmallVec" | "Array" => {
                let (elem, cap) = extract_elem_cap(args, symbols);
                match name.as_str() {
                    "SortedVec" => IrType::SortedVec(Box::new(elem), cap),
                    "RingBuffer" => IrType::RingBuffer(Box::new(elem), cap),
                    "SmallVec" => IrType::SmallVec(Box::new(elem), cap),
                    "Array" => IrType::Array(Box::new(elem), cap),
                    _ => unreachable!(),
                }
            }
            "Option" => {
                if let Some(ast::TypeArg::Type(inner)) = args.first() {
                    IrType::Optional(Box::new(resolve_type(inner, symbols)))
                } else {
                    IrType::Named(name.clone())
                }
            }
            _ => IrType::Named(name.clone()),
        },
        ast::TypeKind::List(inner) => IrType::List(Box::new(resolve_type(inner, symbols))),
        ast::TypeKind::Tuple(inners) => {
            IrType::Tuple(inners.iter().map(|t| resolve_type(t, symbols)).collect())
        }
        ast::TypeKind::Option(inner) => IrType::Optional(Box::new(resolve_type(inner, symbols))),
    }
}

fn extract_elem_cap(args: &[ast::TypeArg], symbols: &SymbolTable) -> (IrType, u16) {
    let mut elem = IrType::Unknown;
    let mut cap: u16 = 0;
    for a in args {
        match a {
            ast::TypeArg::Type(t) => elem = resolve_type(t, symbols),
            ast::TypeArg::Const(n) => cap = u16::try_from(*n).unwrap_or(0),
        }
    }
    (elem, cap)
}

// ---------------------------------------------------------------------------
// Entity fields
// ---------------------------------------------------------------------------

fn resolve_entity_field(
    f: &ast::EntityField,
    symbols: &SymbolTable,
) -> Result<EntityFieldIR, ResolveError> {
    let value = match &f.value {
        ast::EntityFieldValue::Type(t) => EntityFieldValueIR::Type(resolve_type(t, symbols)),
        ast::EntityFieldValue::StructLiteral { ty, fields } => {
            let ty = resolve_type(ty, symbols);
            let fields = fields
                .iter()
                .map(|g| resolve_entity_field(g, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            EntityFieldValueIR::StructLiteral { ty, fields }
        }
        ast::EntityFieldValue::List(exprs) => {
            let mut scope = LocalScope::new();
            let exprs = exprs
                .iter()
                .map(|e| resolve_expr(e, &mut scope, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            EntityFieldValueIR::List(exprs)
        }
        ast::EntityFieldValue::Expr(e) => {
            let mut scope = LocalScope::new();
            EntityFieldValueIR::Expr(resolve_expr(e, &mut scope, symbols)?)
        }
    };
    Ok(EntityFieldIR {
        name: f.name.clone(),
        value,
        annotations: f.annotations.clone(),
        span: f.span,
    })
}

// ---------------------------------------------------------------------------
// Params / action heads / event patterns
// ---------------------------------------------------------------------------

fn resolve_params(
    params: &[ast::Param],
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Vec<IrParam> {
    params
        .iter()
        .map(|p| {
            let ty = resolve_type(&p.ty, symbols);
            let local = scope.bind(&p.name, ty.clone());
            IrParam { name: p.name.clone(), local, ty, span: p.span }
        })
        .collect()
}

fn resolve_action_head(
    head: &ast::ActionHead,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrActionHead {
    let shape = match &head.shape {
        ActionHeadShape::None => IrActionHeadShape::None,
        ActionHeadShape::Positional(names) => {
            let bound = names
                .iter()
                .map(|n| (n.clone(), scope.bind(n, IrType::Unknown)))
                .collect();
            IrActionHeadShape::Positional(bound)
        }
        ActionHeadShape::Named(bindings) => {
            let bs = bindings
                .iter()
                .map(|b| resolve_pattern_binding(b, scope, symbols))
                .collect();
            IrActionHeadShape::Named(bs)
        }
    };
    IrActionHead { name: head.name.clone(), shape, span: head.span }
}

fn resolve_event_pattern(
    p: &ast::EventPattern,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrEventPattern {
    let event = symbols.events.get(&p.name).copied();
    let bindings = p
        .bindings
        .iter()
        .map(|b| resolve_pattern_binding(b, scope, symbols))
        .collect();
    IrEventPattern { name: p.name.clone(), event, bindings, span: p.span }
}

fn resolve_pattern_binding(
    b: &ast::PatternBinding,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrPatternBinding {
    let value = resolve_pattern_value(&b.value, scope, symbols);
    IrPatternBinding { field: b.field.clone(), value, span: b.span }
}

fn resolve_pattern_value(
    v: &ast::PatternValue,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> IrPattern {
    match v {
        ast::PatternValue::Bind(n) => {
            let local = scope.bind(n, IrType::Unknown);
            IrPattern::Bind { name: n.clone(), local }
        }
        ast::PatternValue::Ctor { name, inner } => {
            let ctor = ctor_ref(name, symbols);
            let inner = inner
                .iter()
                .map(|p| resolve_pattern_value(p, scope, symbols))
                .collect();
            IrPattern::Ctor { name: name.clone(), ctor, inner }
        }
        ast::PatternValue::Expr(e) => {
            // Best-effort: try to resolve the expression against the current
            // scope. If it fails (unknown ident), we keep Raw.
            let mut throwaway = scope_clone(scope);
            match resolve_expr(e, &mut throwaway, symbols) {
                Ok(ir) => IrPattern::Expr(ir),
                Err(_) => IrPattern::Expr(IrExprNode {
                    kind: IrExpr::Raw(Box::new(e.clone())),
                    span: e.span,
                }),
            }
        }
        ast::PatternValue::Wildcard => IrPattern::Wildcard,
    }
}

fn scope_clone(s: &LocalScope) -> LocalScope {
    LocalScope {
        stack: s.stack.clone(),
        next_id: s.next_id,
        self_bound: s.self_bound,
    }
}

fn ctor_ref(name: &str, symbols: &SymbolTable) -> Option<CtorRef> {
    if let Some(r) = symbols.events.get(name) {
        return Some(CtorRef::Event(*r));
    }
    if let Some(r) = symbols.entities.get(name) {
        return Some(CtorRef::Entity(*r));
    }
    None
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn resolve_stmts(
    stmts: &[Stmt],
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<Vec<IrStmt>, ResolveError> {
    stmts.iter().map(|s| resolve_stmt(s, scope, symbols)).collect()
}

fn resolve_stmt(
    stmt: &Stmt,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrStmt, ResolveError> {
    match stmt {
        Stmt::Let { name, value, span } => {
            let v = resolve_expr(value, scope, symbols)?;
            let local = scope.bind(name, IrType::Unknown);
            Ok(IrStmt::Let { name: name.clone(), local, value: v, span: *span })
        }
        Stmt::Emit(e) => Ok(IrStmt::Emit(resolve_emit(e, scope, symbols)?)),
        Stmt::For { binder, iter, filter, body, span } => {
            let iter_ir = resolve_expr(iter, scope, symbols)?;
            scope.push();
            let local = scope.bind(binder, IrType::Unknown);
            let filter_ir = filter
                .as_ref()
                .map(|f| resolve_expr(f, scope, symbols))
                .transpose()?;
            let body_ir = resolve_stmts(body, scope, symbols)?;
            scope.pop();
            Ok(IrStmt::For {
                binder: local,
                binder_name: binder.clone(),
                iter: iter_ir,
                filter: filter_ir,
                body: body_ir,
                span: *span,
            })
        }
        Stmt::If { cond, then_body, else_body, span } => {
            let cond = resolve_expr(cond, scope, symbols)?;
            scope.push();
            let then_body = resolve_stmts(then_body, scope, symbols)?;
            scope.pop();
            let else_body = match else_body {
                Some(b) => {
                    scope.push();
                    let r = resolve_stmts(b, scope, symbols)?;
                    scope.pop();
                    Some(r)
                }
                None => None,
            };
            Ok(IrStmt::If { cond, then_body, else_body, span: *span })
        }
        Stmt::Match { scrutinee, arms, span } => {
            let scrutinee = resolve_expr(scrutinee, scope, symbols)?;
            let arms = arms
                .iter()
                .map(|a| {
                    scope.push();
                    let pattern = resolve_pattern_value(&a.pattern, scope, symbols);
                    let body = resolve_stmts(&a.body, scope, symbols)?;
                    scope.pop();
                    Ok::<_, ResolveError>(IrStmtMatchArm { pattern, body, span: a.span })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrStmt::Match { scrutinee, arms, span: *span })
        }
        Stmt::SelfUpdate { op, value, span } => {
            let value = resolve_expr(value, scope, symbols)?;
            Ok(IrStmt::SelfUpdate { op: op.clone(), value, span: *span })
        }
        Stmt::Expr(e) => Ok(IrStmt::Expr(resolve_expr(e, scope, symbols)?)),
    }
}

fn resolve_emit(
    e: &ast::EmitStmt,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrEmit, ResolveError> {
    let event = symbols.events.get(&e.event_name).copied();
    let fields = e
        .fields
        .iter()
        .map(|f| {
            Ok::<_, ResolveError>(IrFieldInit {
                name: f.name.clone(),
                value: resolve_expr(&f.value, scope, symbols)?,
                span: f.span,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IrEmit { event_name: e.event_name.clone(), event, fields, span: e.span })
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn resolve_expr(
    e: &Expr,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExprNode, ResolveError> {
    let span = e.span;
    let kind = match &e.kind {
        ExprKind::Int(v) => IrExpr::LitInt(*v),
        ExprKind::Float(v) => IrExpr::LitFloat(*v),
        ExprKind::Bool(v) => IrExpr::LitBool(*v),
        ExprKind::String(v) => IrExpr::LitString(v.clone()),
        ExprKind::Ident(name) => resolve_ident(name, span, scope, symbols)?,
        ExprKind::Field(base, name) => {
            let base_ir = resolve_expr(base, scope, symbols)?;
            IrExpr::Field {
                base: Box::new(base_ir),
                field_name: name.clone(),
                field: None,
            }
        }
        ExprKind::Index(base, idx) => IrExpr::Index(
            Box::new(resolve_expr(base, scope, symbols)?),
            Box::new(resolve_expr(idx, scope, symbols)?),
        ),
        ExprKind::Call(callee, args) => resolve_call(callee, args, span, scope, symbols)?,
        ExprKind::Binary { op, lhs, rhs } => IrExpr::Binary(
            *op,
            Box::new(resolve_expr(lhs, scope, symbols)?),
            Box::new(resolve_expr(rhs, scope, symbols)?),
        ),
        ExprKind::Unary { op, rhs } => {
            IrExpr::Unary(*op, Box::new(resolve_expr(rhs, scope, symbols)?))
        }
        ExprKind::In { item, set } => IrExpr::In(
            Box::new(resolve_expr(item, scope, symbols)?),
            Box::new(resolve_expr(set, scope, symbols)?),
        ),
        ExprKind::Contains { set, item } => IrExpr::Contains(
            Box::new(resolve_expr(set, scope, symbols)?),
            Box::new(resolve_expr(item, scope, symbols)?),
        ),
        ExprKind::Quantifier { kind, binder, iter, body } => {
            let iter_ir = resolve_expr(iter, scope, symbols)?;
            scope.push();
            let local = scope.bind(binder, IrType::Unknown);
            let body_ir = resolve_expr(body, scope, symbols)?;
            scope.pop();
            IrExpr::Quantifier {
                kind: *kind,
                binder: local,
                binder_name: binder.clone(),
                iter: Box::new(iter_ir),
                body: Box::new(body_ir),
            }
        }
        ExprKind::Fold { kind, binder, iter, body } => {
            let iter_ir = iter
                .as_ref()
                .map(|i| resolve_expr(i, scope, symbols))
                .transpose()?;
            scope.push();
            let local = binder.as_ref().map(|b| scope.bind(b, IrType::Unknown));
            let body_ir = resolve_expr(body, scope, symbols)?;
            scope.pop();
            IrExpr::Fold {
                kind: *kind,
                binder: local,
                binder_name: binder.clone(),
                iter: iter_ir.map(Box::new),
                body: Box::new(body_ir),
            }
        }
        ExprKind::List(items) => IrExpr::List(
            items
                .iter()
                .map(|i| resolve_expr(i, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ExprKind::Tuple(items) => IrExpr::Tuple(
            items
                .iter()
                .map(|i| resolve_expr(i, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ExprKind::Struct { name, fields } => {
            let ctor = ctor_ref(name, symbols);
            let fields = fields
                .iter()
                .map(|f| {
                    Ok::<_, ResolveError>(IrFieldInit {
                        name: f.name.clone(),
                        value: resolve_expr(&f.value, scope, symbols)?,
                        span: f.span,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::StructLit { name: name.clone(), ctor, fields }
        }
        ExprKind::Ctor { name, args } => {
            let ctor = ctor_ref(name, symbols);
            let args = args
                .iter()
                .map(|a| resolve_expr(a, scope, symbols))
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::Ctor { name: name.clone(), ctor, args }
        }
        ExprKind::Match { scrutinee, arms } => {
            let scrutinee = resolve_expr(scrutinee, scope, symbols)?;
            let arms = arms
                .iter()
                .map(|a| {
                    scope.push();
                    let pattern = resolve_pattern_value(&a.pattern, scope, symbols);
                    let body = resolve_expr(&a.body, scope, symbols)?;
                    scope.pop();
                    Ok::<_, ResolveError>(IrMatchArm { pattern, body, span: a.span })
                })
                .collect::<Result<Vec<_>, _>>()?;
            IrExpr::Match { scrutinee: Box::new(scrutinee), arms }
        }
        ExprKind::If { cond, then_expr, else_expr } => IrExpr::If {
            cond: Box::new(resolve_expr(cond, scope, symbols)?),
            then_expr: Box::new(resolve_expr(then_expr, scope, symbols)?),
            else_expr: match else_expr {
                Some(x) => Some(Box::new(resolve_expr(x, scope, symbols)?)),
                None => None,
            },
        },
    };
    Ok(IrExprNode { kind, span })
}

fn resolve_ident(
    name: &str,
    span: Span,
    scope: &LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExpr, ResolveError> {
    // Bare `true` / `false` are literals. The parser emits them as Ident.
    if name == "true" {
        return Ok(IrExpr::LitBool(true));
    }
    if name == "false" {
        return Ok(IrExpr::LitBool(false));
    }
    if let Some(b) = scope.lookup(name) {
        return Ok(IrExpr::Local(b.local, b.name.clone()));
    }
    if name == "self" {
        // `self` is only valid inside a decl that binds it — if the scope
        // didn't see it, this is SelfInTopLevel.
        return Err(ResolveError::SelfInTopLevel { span });
    }
    if let Some(r) = symbols.entities.get(name) {
        return Ok(IrExpr::Entity(*r));
    }
    if let Some(r) = symbols.events.get(name) {
        return Ok(IrExpr::Event(*r));
    }
    if let Some(r) = symbols.views.get(name) {
        return Ok(IrExpr::View(*r));
    }
    if let Some(r) = symbols.verbs.get(name) {
        return Ok(IrExpr::Verb(*r));
    }
    if symbols.stdlib_namespaces.contains(name) {
        return Ok(IrExpr::Namespace(name.to_string()));
    }
    if let Some(t) = symbols.stdlib_types.get(name) {
        // The identifier referred to a type name used as a value. In 1a we
        // don't have a dedicated "type-as-value" node; fall through and keep
        // it as an unresolved enum variant marker (closest analogue: "ALL_CAPS
        // CONSTANT", "Stone", etc — also handled below).
        let _ = t;
    }
    // Identifiers that start uppercase are likely enum variants or constants
    // (Conquest, Family, Religion, Stone, FleeSet, AGGRO_RANGE, ...). We
    // don't know yet — keep the name on an EnumVariant with an empty type.
    // 1b will either promote to a real enum ref or error.
    if starts_upper(name) {
        return Ok(IrExpr::EnumVariant { ty: String::new(), variant: name.to_string() });
    }
    // Otherwise: bare lowercase ident with no match. This is an unknown
    // identifier — error out with suggestions.
    let suggestions = suggest_idents(name, scope, symbols);
    Err(ResolveError::UnknownIdent { name: name.to_string(), span, suggestions })
}

fn resolve_call(
    callee: &Expr,
    args: &[ast::CallArg],
    span: Span,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrExpr, ResolveError> {
    // Only resolve callees that are a bare Ident. Anything else (method
    // chain, field-call) falls through as UnresolvedCall or Raw.
    if let ExprKind::Ident(name) = &callee.kind {
        let ir_args = args
            .iter()
            .map(|a| resolve_call_arg(a, scope, symbols))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(b) = symbols.builtins.get(name) {
            return Ok(IrExpr::BuiltinCall(*b, ir_args));
        }
        if let Some(r) = symbols.views.get(name) {
            return Ok(IrExpr::ViewCall(*r, ir_args));
        }
        if let Some(r) = symbols.verbs.get(name) {
            return Ok(IrExpr::VerbCall(*r, ir_args));
        }
        // Local or unresolved.
        if scope.lookup(name).is_some() {
            // Calling a local (unusual; treat as unresolved for 1b).
            return Ok(IrExpr::UnresolvedCall(name.clone(), ir_args));
        }
        return Ok(IrExpr::UnresolvedCall(name.clone(), ir_args));
    }
    // Complex callee: keep raw for 1b.
    let _ = span;
    Ok(IrExpr::Raw(Box::new(Expr {
        kind: ExprKind::Call(
            Box::new(callee.clone()),
            args.to_vec(),
        ),
        span,
    })))
}

fn resolve_call_arg(
    a: &ast::CallArg,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrCallArg, ResolveError> {
    Ok(IrCallArg {
        name: a.name.clone(),
        value: resolve_expr(&a.value, scope, symbols)?,
        span: a.span,
    })
}

fn resolve_assert(
    a: &AssertExpr,
    scope: &mut LocalScope,
    symbols: &SymbolTable,
) -> Result<IrAssertExpr, ResolveError> {
    match a {
        AssertExpr::Count { filter, op, value, span } => Ok(IrAssertExpr::Count {
            filter: resolve_expr(filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
        AssertExpr::Pr { action_filter, obs_filter, op, value, span } => Ok(IrAssertExpr::Pr {
            action_filter: resolve_expr(action_filter, scope, symbols)?,
            obs_filter: resolve_expr(obs_filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
        AssertExpr::Mean { scalar, filter, op, value, span } => Ok(IrAssertExpr::Mean {
            scalar: resolve_expr(scalar, scope, symbols)?,
            filter: resolve_expr(filter, scope, symbols)?,
            op: op.clone(),
            value: resolve_expr(value, scope, symbols)?,
            span: *span,
        }),
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn starts_upper(s: &str) -> bool {
    s.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
}

fn suggest_idents(name: &str, scope: &LocalScope, symbols: &SymbolTable) -> Vec<String> {
    let mut pool: Vec<&str> = Vec::new();
    for frame in &scope.stack {
        for b in frame {
            pool.push(&b.name);
        }
    }
    for k in symbols.events.keys() {
        pool.push(k);
    }
    for k in symbols.entities.keys() {
        pool.push(k);
    }
    for k in symbols.views.keys() {
        pool.push(k);
    }
    for k in symbols.verbs.keys() {
        pool.push(k);
    }
    for k in symbols.builtins.keys() {
        pool.push(k);
    }
    let mut ranked: Vec<(usize, String)> = pool
        .iter()
        .map(|s| (edit_distance(name, s), s.to_string()))
        .collect();
    ranked.sort_by_key(|p| p.0);
    ranked.retain(|p| p.0 <= 3);
    ranked.into_iter().map(|p| p.1).take(3).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}
