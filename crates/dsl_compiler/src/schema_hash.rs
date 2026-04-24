//! Schema hashes covering the compiler-emitted DSL surface.
//!
//! `docs/compiler/spec.md` §2 specifies four sub-hashes plus one combined
//! hash:
//!
//! - `event_hash` — event taxonomy (milestone 2; canonical).
//! - `rules_hash` — physics cascades + masks + verbs (milestone 3 partial:
//!   physics only; masks + verbs land at milestones 4 / 7).
//! - `state_hash` — entity field layouts (milestone 6; canonical).
//! - `scoring_hash` — scoring tables (milestone 4 — placeholder zeros).
//! - `combined_hash` = `sha256(state || event || rules || scoring)`.
//!
//! Every sub-hash is reorder-stable: declarations are sorted by name before
//! folding. Field / handler order WITHIN a declaration is preserved (the
//! emitted Rust depends on it).

use sha2::{Digest, Sha256};

use crate::emit_entity;
use crate::ir::{
    ConfigIR, DecayUnit, EntityIR, EnumIR, EventIR, EventTagIR, IrExpr, IrExprNode, IrPattern,
    IrPatternBinding, IrPhysicsPattern, IrStmt, IrType, PhysicsHandlerIR, PhysicsIR, ScoringIR,
    StorageHint, ViewBodyIR, ViewIR, ViewKind,
};

pub fn event_hash(events: &[EventIR]) -> [u8; 32] {
    let mut sorted: Vec<&EventIR> = events.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut h = Sha256::new();
    for e in sorted {
        h.update(e.name.as_bytes());
        h.update([0u8]);
        for field in &e.fields {
            h.update(field.name.as_bytes());
            h.update([0u8]);
            let bytes = type_canonical_bytes(&field.ty);
            h.update(&bytes);
            h.update([0u8]);
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

/// Hash every declared enum. Variant order within an enum is load-bearing
/// (the `#[repr(u8)]` ordinal is the variant's source-order index), so
/// variants keep their declaration order; blocks themselves are sorted by
/// name for reorder stability.
pub fn enums_hash(enums: &[EnumIR]) -> [u8; 32] {
    let mut sorted: Vec<&EnumIR> = enums.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut h = Sha256::new();
    for e in sorted {
        h.update(e.name.as_bytes());
        h.update([0u8]);
        h.update(&(e.variants.len() as u32).to_le_bytes());
        for v in &e.variants {
            h.update(v.as_bytes());
            h.update([0u8]);
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

/// Hash the physics-rule subset of the rules taxonomy. Stable under rule
/// reordering: rules are sorted by name. Handlers within a rule keep their
/// source order (handler N's identity matters; reordering changes runtime
/// dispatch).
///
/// The bytes folded in cover the rule name, every handler's trigger event
/// + binding shape + body statement structure. We deliberately don't fold
/// in source spans, identifier byte offsets, or comments — those are
/// formatting noise the hash should be immune to.
pub fn rules_hash(
    physics: &[PhysicsIR],
    event_tags: &[EventTagIR],
    events: &[EventIR],
) -> [u8; 32] {
    let mut sorted: Vec<&PhysicsIR> = physics.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut h = Sha256::new();
    // Physics rules.
    for p in sorted {
        h.update(p.name.as_bytes());
        h.update([0u8]);
        h.update(&(p.handlers.len() as u32).to_le_bytes());
        for handler in &p.handlers {
            hash_handler(&mut h, handler);
        }
        h.update([0xFFu8]);
    }
    h.update([0xAAu8]);
    // Event tag declarations.
    let mut sorted_tags: Vec<&EventTagIR> = event_tags.iter().collect();
    sorted_tags.sort_by(|a, b| a.name.cmp(&b.name));
    for t in sorted_tags {
        h.update(t.name.as_bytes());
        h.update([0u8]);
        h.update(&(t.fields.len() as u32).to_le_bytes());
        for f in &t.fields {
            h.update(f.name.as_bytes());
            h.update([0u8]);
            h.update(&type_canonical_bytes(&f.ty));
            h.update([0u8]);
        }
        h.update([0xFFu8]);
    }
    h.update([0xBBu8]);
    // Per-event tag membership (event name → sorted tag ref ordinals).
    let mut sorted_events: Vec<&EventIR> = events.iter().collect();
    sorted_events.sort_by(|a, b| a.name.cmp(&b.name));
    for e in sorted_events {
        h.update(e.name.as_bytes());
        h.update([0u8]);
        let mut refs: Vec<u16> = e.tags.iter().map(|r| r.0).collect();
        refs.sort();
        h.update(&(refs.len() as u32).to_le_bytes());
        for r in refs {
            h.update(&r.to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Hash every `scoring` block's entries. Block order is preserved
/// (blocks in the DSL are positional), but within a block entries are
/// sorted by action-head name so a cosmetic reorder doesn't perturb the
/// hash. The expression body folds through `hash_expr`, which already
/// covers `Field`, `Binary`, `If`, and the literal cases the scoring
/// emitter needs.
/// Hash the entity state-layout surface. Entities are sorted by name for
/// reorder stability; within an entity, fields keep their source order
/// (field order drives struct layout). The tag-projection is delegated to
/// `emit_entity::schema_hash_input` so the hasher stays agnostic of the
/// exact IR shape — any future entity-IR extension only touches that
/// projection.
pub fn state_hash(entities: &[EntityIR]) -> [u8; 32] {
    let rows = emit_entity::schema_hash_input(entities);
    // Sort rows by entity name. Rows for the same entity stay contiguous in
    // their emission (source-preserved) order — field order within an entity
    // drives the emitted struct layout.
    let mut sorted = rows;
    sorted.sort_by(|a, b| a.entity.cmp(&b.entity));

    let mut h = Sha256::new();
    for r in &sorted {
        h.update(r.entity.as_bytes());
        h.update([0u8]);
        h.update(r.root.as_bytes());
        h.update([0u8]);
        h.update(r.field.as_bytes());
        h.update([0u8]);
        h.update(r.tag.as_bytes());
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

/// Hash every `config` block's field schema. Block order is reorder-stable
/// (blocks sorted by name); within a block, fields keep their source order
/// so a reorder changes the hash (field order drives the emitted struct
/// layout). Only the block name, field names, and field types are hashed —
/// default *values* are intentionally excluded so balance tuning via the
/// emitted TOML doesn't perturb the schema fingerprint.
pub fn config_hash(blocks: &[ConfigIR]) -> [u8; 32] {
    let mut sorted: Vec<&ConfigIR> = blocks.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut h = Sha256::new();
    for c in sorted {
        h.update(c.name.as_bytes());
        h.update([0u8]);
        h.update(&(c.fields.len() as u32).to_le_bytes());
        for f in &c.fields {
            h.update(f.name.as_bytes());
            h.update([0u8]);
            let bytes = type_canonical_bytes(&f.ty);
            h.update(&bytes);
            h.update([0u8]);
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

/// Hash every `view` declaration's schema-relevant surface: name, param
/// names + types (in declaration order — pair-map key ordering depends
/// on it), return type, view kind + storage hint, decay params (rate +
/// per-unit), and the structural form of the fold body (handler event
/// names, binding shape, statement structure). `@lazy` expression
/// bodies hash through `hash_expr` for the same structural coverage the
/// scoring hash uses.
///
/// Views are sorted by name for reorder stability; params + handlers
/// keep their source order since both drive the emitted Rust's shape.
pub fn views_hash(views: &[ViewIR]) -> [u8; 32] {
    let mut sorted: Vec<&ViewIR> = views.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut h = Sha256::new();
    for v in sorted {
        h.update(v.name.as_bytes());
        h.update([0u8]);
        // Params.
        h.update(&(v.params.len() as u32).to_le_bytes());
        for p in &v.params {
            h.update(p.name.as_bytes());
            h.update([0u8]);
            h.update(&type_canonical_bytes(&p.ty));
            h.update([0u8]);
        }
        // Return type.
        h.update(&type_canonical_bytes(&v.return_ty));
        h.update([0u8]);
        // Kind + storage hint.
        match v.kind {
            ViewKind::Lazy => h.update([0x01u8]),
            ViewKind::Materialized(hint) => {
                h.update([0x02u8]);
                match hint {
                    StorageHint::PairMap => h.update([0x10u8]),
                    StorageHint::PerEntityTopK { k, keyed_on } => {
                        h.update([0x11u8]);
                        h.update(&k.to_le_bytes());
                        h.update([keyed_on]);
                    }
                    StorageHint::LazyCached => h.update([0x12u8]),
                    StorageHint::SymmetricPairTopK { k } => {
                        h.update([0x13u8]);
                        h.update(&k.to_le_bytes());
                    }
                    StorageHint::PerEntityRing { k } => {
                        h.update([0x14u8]);
                        h.update(&k.to_le_bytes());
                    }
                }
            }
        }
        // Decay hint.
        match v.decay {
            None => h.update([0x00u8]),
            Some(d) => {
                h.update([0x01u8]);
                h.update(&d.rate.to_le_bytes());
                match d.per {
                    DecayUnit::Tick => h.update([0x01u8]),
                }
            }
        }
        // Body.
        match &v.body {
            ViewBodyIR::Expr(e) => {
                h.update([0xAAu8]);
                hash_expr(&mut h, e);
            }
            ViewBodyIR::Fold { initial, handlers, clamp } => {
                h.update([0xBBu8]);
                hash_expr(&mut h, initial);
                h.update(&(handlers.len() as u32).to_le_bytes());
                for fh in handlers {
                    h.update(fh.pattern.name.as_bytes());
                    h.update([0u8]);
                    h.update(&(fh.pattern.bindings.len() as u32).to_le_bytes());
                    for b in &fh.pattern.bindings {
                        hash_pattern_binding(&mut h, b);
                    }
                    h.update(&(fh.body.len() as u32).to_le_bytes());
                    for s in &fh.body {
                        hash_stmt(&mut h, s);
                    }
                }
                if let Some((lo, hi)) = clamp {
                    h.update([0x01u8]);
                    hash_expr(&mut h, lo);
                    hash_expr(&mut h, hi);
                } else {
                    h.update([0x00u8]);
                }
            }
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

pub fn scoring_hash(blocks: &[ScoringIR]) -> [u8; 32] {
    let mut h = Sha256::new();
    for block in blocks {
        let mut sorted: Vec<&crate::ir::ScoringEntryIR> = block.entries.iter().collect();
        sorted.sort_by(|a, b| a.head.name.cmp(&b.head.name));
        h.update(&(sorted.len() as u32).to_le_bytes());
        for e in sorted {
            h.update(e.head.name.as_bytes());
            h.update([0u8]);
            hash_expr(&mut h, &e.expr);
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

fn hash_handler(h: &mut Sha256, handler: &PhysicsHandlerIR) {
    match &handler.pattern {
        IrPhysicsPattern::Kind(p) => {
            h.update([0x01u8]);
            h.update(p.name.as_bytes());
            h.update([0u8]);
            h.update(&(p.bindings.len() as u32).to_le_bytes());
            for b in &p.bindings {
                hash_pattern_binding(h, b);
            }
        }
        IrPhysicsPattern::Tag { name, bindings, .. } => {
            h.update([0x02u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(bindings.len() as u32).to_le_bytes());
            for b in bindings {
                hash_pattern_binding(h, b);
            }
        }
    }
    h.update(if handler.where_clause.is_some() { [0x01u8] } else { [0x00u8] });
    if let Some(w) = &handler.where_clause {
        hash_expr(h, w);
    }
    h.update(&(handler.body.len() as u32).to_le_bytes());
    for s in &handler.body {
        hash_stmt(h, s);
    }
}

fn hash_pattern_binding(h: &mut Sha256, b: &IrPatternBinding) {
    h.update(b.field.as_bytes());
    h.update([0u8]);
    hash_pattern(h, &b.value);
}

fn hash_pattern(h: &mut Sha256, p: &IrPattern) {
    match p {
        IrPattern::Bind { name, .. } => {
            h.update([0x01u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
        }
        IrPattern::Wildcard => {
            h.update([0x02u8]);
        }
        IrPattern::Ctor { name, inner, .. } => {
            h.update([0x03u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(inner.len() as u32).to_le_bytes());
            for i in inner {
                hash_pattern(h, i);
            }
        }
        IrPattern::Struct { name, bindings, .. } => {
            h.update([0x05u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(bindings.len() as u32).to_le_bytes());
            for b in bindings {
                hash_pattern_binding(h, b);
            }
        }
        IrPattern::Expr(e) => {
            h.update([0x04u8]);
            hash_expr(h, e);
        }
    }
}

fn hash_stmt(h: &mut Sha256, s: &IrStmt) {
    match s {
        IrStmt::Let { name, value, .. } => {
            h.update([0x10u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            hash_expr(h, value);
        }
        IrStmt::Emit(e) => {
            h.update([0x11u8]);
            h.update(e.event_name.as_bytes());
            h.update([0u8]);
            h.update(&(e.fields.len() as u32).to_le_bytes());
            for f in &e.fields {
                h.update(f.name.as_bytes());
                h.update([0u8]);
                hash_expr(h, &f.value);
            }
        }
        IrStmt::If { cond, then_body, else_body, .. } => {
            h.update([0x12u8]);
            hash_expr(h, cond);
            h.update(&(then_body.len() as u32).to_le_bytes());
            for s in then_body {
                hash_stmt(h, s);
            }
            h.update(if else_body.is_some() { [0x01u8] } else { [0x00u8] });
            if let Some(b) = else_body {
                h.update(&(b.len() as u32).to_le_bytes());
                for s in b {
                    hash_stmt(h, s);
                }
            }
        }
        IrStmt::For { binder_name, iter, filter, body, .. } => {
            h.update([0x13u8]);
            h.update(binder_name.as_bytes());
            h.update([0u8]);
            hash_expr(h, iter);
            h.update(if filter.is_some() { [0x01u8] } else { [0x00u8] });
            if let Some(f) = filter {
                hash_expr(h, f);
            }
            h.update(&(body.len() as u32).to_le_bytes());
            for s in body {
                hash_stmt(h, s);
            }
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            h.update([0x14u8]);
            hash_expr(h, scrutinee);
            h.update(&(arms.len() as u32).to_le_bytes());
            for a in arms {
                hash_pattern(h, &a.pattern);
                h.update(&(a.body.len() as u32).to_le_bytes());
                for s in &a.body {
                    hash_stmt(h, s);
                }
            }
        }
        IrStmt::SelfUpdate { op, value, .. } => {
            h.update([0x15u8]);
            h.update(op.as_bytes());
            h.update([0u8]);
            hash_expr(h, value);
        }
        IrStmt::Expr(e) => {
            h.update([0x16u8]);
            hash_expr(h, e);
        }
    }
}

fn hash_expr(h: &mut Sha256, e: &IrExprNode) {
    hash_expr_kind(h, &e.kind)
}

fn hash_expr_kind(h: &mut Sha256, kind: &IrExpr) {
    match kind {
        IrExpr::LitBool(b) => {
            h.update([0x20u8]);
            h.update([*b as u8]);
        }
        IrExpr::LitInt(v) => {
            h.update([0x21u8]);
            h.update(&v.to_le_bytes());
        }
        IrExpr::LitFloat(v) => {
            h.update([0x22u8]);
            h.update(&v.to_le_bytes());
        }
        IrExpr::LitString(s) => {
            h.update([0x23u8]);
            h.update(&(s.len() as u32).to_le_bytes());
            h.update(s.as_bytes());
        }
        IrExpr::Local(_, name) => {
            h.update([0x24u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
        }
        IrExpr::Event(r) => {
            h.update([0x25u8]);
            h.update(&r.0.to_le_bytes());
        }
        IrExpr::Entity(r) => {
            h.update([0x26u8]);
            h.update(&r.0.to_le_bytes());
        }
        IrExpr::View(r) => {
            h.update([0x27u8]);
            h.update(&r.0.to_le_bytes());
        }
        IrExpr::Verb(r) => {
            h.update([0x28u8]);
            h.update(&r.0.to_le_bytes());
        }
        IrExpr::Namespace(ns) => {
            h.update([0x29u8]);
            h.update(ns.name().as_bytes());
            h.update([0u8]);
        }
        IrExpr::NamespaceField { ns, field, .. } => {
            h.update([0x2au8]);
            h.update(ns.name().as_bytes());
            h.update([0u8]);
            h.update(field.as_bytes());
            h.update([0u8]);
        }
        IrExpr::NamespaceCall { ns, method, args } => {
            h.update([0x2bu8]);
            h.update(ns.name().as_bytes());
            h.update([0u8]);
            h.update(method.as_bytes());
            h.update([0u8]);
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, &a.value);
            }
        }
        IrExpr::EnumVariant { ty, variant } => {
            h.update([0x2cu8]);
            h.update(ty.as_bytes());
            h.update([0u8]);
            h.update(variant.as_bytes());
            h.update([0u8]);
        }
        IrExpr::Field { base, field_name, .. } => {
            h.update([0x2du8]);
            hash_expr(h, base);
            h.update(field_name.as_bytes());
            h.update([0u8]);
        }
        IrExpr::Index(base, idx) => {
            h.update([0x2eu8]);
            hash_expr(h, base);
            hash_expr(h, idx);
        }
        IrExpr::ViewCall(r, args) => {
            h.update([0x2fu8]);
            h.update(&r.0.to_le_bytes());
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, &a.value);
            }
        }
        IrExpr::VerbCall(r, args) => {
            h.update([0x30u8]);
            h.update(&r.0.to_le_bytes());
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, &a.value);
            }
        }
        IrExpr::BuiltinCall(b, args) => {
            h.update([0x31u8]);
            h.update(b.name().as_bytes());
            h.update([0u8]);
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, &a.value);
            }
        }
        IrExpr::UnresolvedCall(name, args) => {
            h.update([0x32u8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, &a.value);
            }
        }
        IrExpr::Binary(op, l, r) => {
            h.update([0x33u8]);
            h.update(&[*op as u8]);
            hash_expr(h, l);
            hash_expr(h, r);
        }
        IrExpr::Unary(op, r) => {
            h.update([0x34u8]);
            h.update(&[*op as u8]);
            hash_expr(h, r);
        }
        IrExpr::In(a, b) => {
            h.update([0x35u8]);
            hash_expr(h, a);
            hash_expr(h, b);
        }
        IrExpr::Contains(a, b) => {
            h.update([0x36u8]);
            hash_expr(h, a);
            hash_expr(h, b);
        }
        IrExpr::Quantifier { kind, binder_name, iter, body, .. } => {
            h.update([0x37u8]);
            h.update(&[*kind as u8]);
            h.update(binder_name.as_bytes());
            h.update([0u8]);
            hash_expr(h, iter);
            hash_expr(h, body);
        }
        IrExpr::Fold { kind, binder_name, iter, body, .. } => {
            h.update([0x38u8]);
            h.update(&[*kind as u8]);
            if let Some(n) = binder_name {
                h.update([0x01u8]);
                h.update(n.as_bytes());
                h.update([0u8]);
            } else {
                h.update([0x00u8]);
            }
            if let Some(i) = iter {
                h.update([0x01u8]);
                hash_expr(h, i);
            } else {
                h.update([0x00u8]);
            }
            hash_expr(h, body);
        }
        IrExpr::List(items) | IrExpr::Tuple(items) => {
            h.update([0x39u8]);
            h.update(&(items.len() as u32).to_le_bytes());
            for it in items {
                hash_expr(h, it);
            }
        }
        IrExpr::StructLit { name, fields, .. } => {
            h.update([0x3au8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(fields.len() as u32).to_le_bytes());
            for f in fields {
                h.update(f.name.as_bytes());
                h.update([0u8]);
                hash_expr(h, &f.value);
            }
        }
        IrExpr::Ctor { name, args, .. } => {
            h.update([0x3bu8]);
            h.update(name.as_bytes());
            h.update([0u8]);
            h.update(&(args.len() as u32).to_le_bytes());
            for a in args {
                hash_expr(h, a);
            }
        }
        IrExpr::Match { scrutinee, arms } => {
            h.update([0x3cu8]);
            hash_expr(h, scrutinee);
            h.update(&(arms.len() as u32).to_le_bytes());
            for a in arms {
                hash_pattern(h, &a.pattern);
                hash_expr(h, &a.body);
            }
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            h.update([0x3du8]);
            hash_expr(h, cond);
            hash_expr(h, then_expr);
            h.update(if else_expr.is_some() { [0x01u8] } else { [0x00u8] });
            if let Some(e) = else_expr {
                hash_expr(h, e);
            }
        }
        IrExpr::PerUnit { expr, delta } => {
            h.update([0x3fu8]);
            hash_expr(h, expr);
            hash_expr(h, delta);
        }
        // GPU ability evaluation Phase 2 primitives. Pinned ordinal
        // bytes so renaming / reordering `AbilityTag` variants bumps
        // the schema hash (and trace-format guards catch drift).
        IrExpr::AbilityTag { tag } => {
            h.update([0x40u8]);
            h.update([*tag as u8]);
        }
        IrExpr::Raw(_) => {
            // Raw fallthrough (e.g. an expression the resolver couldn't lower)
            // — fold a stable tag so two `Raw` exprs with different contents
            // still produce different bytes via downstream regeneration.
            h.update([0x3eu8]);
        }
    }
}

/// Combine the seven sub-hashes into one, in a stable canonical order:
/// `state || event || rules || scoring || config || enums || views`.
/// Trace-format guards check this combined value.
pub fn combined_hash(
    state: &[u8; 32],
    event: &[u8; 32],
    rules: &[u8; 32],
    scoring: &[u8; 32],
    config: &[u8; 32],
    enums: &[u8; 32],
    views: &[u8; 32],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(state);
    h.update(event);
    h.update(rules);
    h.update(scoring);
    h.update(config);
    h.update(enums);
    h.update(views);
    h.finalize().into()
}

/// Emit the Rust source of `schema.rs` containing every sub-hash plus the
/// combined hash. All sub-hashes are canonical as of the config milestone;
/// a sub-hash is only all-zero when no declarations of that kind are in
/// scope.
pub fn emit_schema_rs(
    state: &[u8; 32],
    event: &[u8; 32],
    rules: &[u8; 32],
    scoring: &[u8; 32],
    config: &[u8; 32],
    enums: &[u8; 32],
    views: &[u8; 32],
) -> String {
    use std::fmt::Write;
    let combined = combined_hash(state, event, rules, scoring, config, enums, views);

    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.").unwrap();
    writeln!(
        out,
        "// Regenerate with `cargo run --bin xtask -- compile-dsl`."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "// `COMBINED_HASH` rolls the seven sub-hashes together per `docs/compiler/spec.md` \u{00a7}2."
    )
    .unwrap();
    writeln!(
        out,
        "// A sub-hash is only all-zero when no declarations of that kind are in scope."
    )
    .unwrap();
    writeln!(out).unwrap();

    write_hash_const(&mut out, "STATE_HASH", state);
    write_hash_const(&mut out, "EVENT_HASH", event);
    write_hash_const(&mut out, "RULES_HASH", rules);
    write_hash_const(&mut out, "SCORING_HASH", scoring);
    write_hash_const(&mut out, "CONFIG_HASH", config);
    write_hash_const(&mut out, "ENUMS_HASH", enums);
    write_hash_const(&mut out, "VIEW_HASH", views);
    write_hash_const(&mut out, "COMBINED_HASH", &combined);
    out
}

fn write_hash_const(out: &mut String, name: &str, hash: &[u8; 32]) {
    use std::fmt::Write;
    writeln!(out, "pub const {name}: [u8; 32] = [").unwrap();
    // 16 bytes per line matches rustfmt's default formatting for this shape,
    // so the emitter output survives `cargo fmt` unchanged.
    for chunk in hash.chunks(16) {
        let row: Vec<String> = chunk.iter().map(|b| format!("0x{:02x}", b)).collect();
        writeln!(out, "    {},", row.join(", ")).unwrap();
    }
    writeln!(out, "];").unwrap();
}

// ---------------------------------------------------------------------------
// Type canonicalization
// ---------------------------------------------------------------------------

/// Produce a unique, stable byte sequence for an `IrType`. The leading tag
/// byte discriminates variants; everything after is variant-specific payload.
/// Changing this function changes every downstream `event_hash` — treat any
/// tweak as a schema bump.
fn type_canonical_bytes(ty: &IrType) -> Vec<u8> {
    let mut out = Vec::new();
    write_type(ty, &mut out);
    out
}

fn write_type(ty: &IrType, out: &mut Vec<u8>) {
    match ty {
        IrType::Bool => out.push(0x01),
        IrType::I32 => out.push(0x02),
        IrType::U32 => out.push(0x03),
        IrType::I64 => out.push(0x04),
        IrType::U64 => out.push(0x05),
        IrType::F32 => out.push(0x06),
        IrType::F64 => out.push(0x07),
        IrType::Vec3 => out.push(0x08),
        IrType::String => out.push(0x09),
        IrType::I8 => out.push(0x0a),
        IrType::U8 => out.push(0x0b),
        IrType::I16 => out.push(0x0c),
        IrType::U16 => out.push(0x0d),
        IrType::AgentId => out.push(0x10),
        IrType::ItemId => out.push(0x11),
        IrType::GroupId => out.push(0x12),
        IrType::QuestId => out.push(0x13),
        IrType::AuctionId => out.push(0x14),
        IrType::EventId => out.push(0x15),
        IrType::AbilityId => out.push(0x16),
        IrType::SortedVec(inner, cap) => {
            out.push(0x20);
            out.extend_from_slice(&cap.to_le_bytes());
            write_type(inner, out);
        }
        IrType::RingBuffer(inner, cap) => {
            out.push(0x21);
            out.extend_from_slice(&cap.to_le_bytes());
            write_type(inner, out);
        }
        IrType::SmallVec(inner, cap) => {
            out.push(0x22);
            out.extend_from_slice(&cap.to_le_bytes());
            write_type(inner, out);
        }
        IrType::Array(inner, cap) => {
            out.push(0x23);
            out.extend_from_slice(&cap.to_le_bytes());
            write_type(inner, out);
        }
        IrType::Optional(inner) => {
            out.push(0x24);
            write_type(inner, out);
        }
        IrType::Tuple(items) => {
            out.push(0x25);
            out.extend_from_slice(&(items.len() as u16).to_le_bytes());
            for t in items {
                write_type(t, out);
            }
        }
        IrType::List(inner) => {
            out.push(0x26);
            write_type(inner, out);
        }
        IrType::EntityRef(r) => {
            out.push(0x30);
            out.extend_from_slice(&r.0.to_le_bytes());
        }
        IrType::EventRef(r) => {
            out.push(0x31);
            out.extend_from_slice(&r.0.to_le_bytes());
        }
        IrType::Enum { name, variants } => {
            out.push(0x40);
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(variants.len() as u16).to_le_bytes());
            for v in variants {
                out.extend_from_slice(&(v.len() as u16).to_le_bytes());
                out.extend_from_slice(v.as_bytes());
            }
        }
        IrType::Unknown => out.push(0xFE),
        IrType::Named(n) => {
            out.push(0xFD);
            out.extend_from_slice(&(n.len() as u16).to_le_bytes());
            out.extend_from_slice(n.as_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{EventField, EventIR};

    fn mk_event(name: &str, fields: Vec<(&str, IrType)>) -> EventIR {
        EventIR {
            name: name.into(),
            fields: fields
                .into_iter()
                .map(|(n, t)| EventField { name: n.into(), ty: t, span: Span::dummy() })
                .collect(),
            tags: vec![],
            annotations: vec![],
            span: Span::dummy(),
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let e = vec![
            mk_event("Damage", vec![("target", IrType::AgentId), ("amount", IrType::F32)]),
            mk_event("Heal", vec![("target", IrType::AgentId), ("amount", IrType::F32)]),
        ];
        let h1 = event_hash(&e);
        let h2 = event_hash(&e);
        assert_eq!(h1, h2);
    }

    #[test]
    fn order_independent() {
        let e_ab = vec![
            mk_event("Damage", vec![("target", IrType::AgentId)]),
            mk_event("Heal", vec![("target", IrType::AgentId)]),
        ];
        let e_ba = vec![
            mk_event("Heal", vec![("target", IrType::AgentId)]),
            mk_event("Damage", vec![("target", IrType::AgentId)]),
        ];
        assert_eq!(event_hash(&e_ab), event_hash(&e_ba));
    }

    #[test]
    fn field_rename_changes_hash() {
        let e1 = vec![mk_event("Damage", vec![("target", IrType::AgentId)])];
        let e2 = vec![mk_event("Damage", vec![("victim", IrType::AgentId)])];
        assert_ne!(event_hash(&e1), event_hash(&e2));
    }

    #[test]
    fn type_change_changes_hash() {
        let e1 = vec![mk_event("X", vec![("f", IrType::U32)])];
        let e2 = vec![mk_event("X", vec![("f", IrType::U64)])];
        assert_ne!(event_hash(&e1), event_hash(&e2));
    }

    #[test]
    fn field_order_matters() {
        let e1 = vec![mk_event(
            "X",
            vec![("a", IrType::U32), ("b", IrType::U64)],
        )];
        let e2 = vec![mk_event(
            "X",
            vec![("b", IrType::U64), ("a", IrType::U32)],
        )];
        assert_ne!(event_hash(&e1), event_hash(&e2));
    }

    #[test]
    fn emit_schema_rs_shape() {
        let h = [0u8; 32];
        let s = emit_schema_rs(&h, &h, &h, &h, &h, &h, &h);
        // All eight constants present.
        for name in [
            "STATE_HASH",
            "EVENT_HASH",
            "RULES_HASH",
            "SCORING_HASH",
            "CONFIG_HASH",
            "ENUMS_HASH",
            "VIEW_HASH",
            "COMBINED_HASH",
        ] {
            assert!(s.contains(&format!("pub const {name}: [u8; 32] = [")), "missing {name}");
        }
        // 16 bytes per line, byte rows still rustfmt-stable.
        let expected_row = "0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,";
        assert!(s.contains(expected_row), "schema.rs rows should be 16 bytes wide");
    }

    #[test]
    fn rules_hash_deterministic_and_order_independent() {
        use crate::ir::{
            EventRef, IrEventPattern, IrPattern, IrPatternBinding, IrPhysicsPattern, IrStmt,
            LocalRef, PhysicsHandlerIR, PhysicsIR,
        };
        let mk = |name: &str| PhysicsIR {
            name: name.into(),
            handlers: vec![PhysicsHandlerIR {
                pattern: IrPhysicsPattern::Kind(IrEventPattern {
                    name: "EffectDamageApplied".into(),
                    event: Some(EventRef(0)),
                    bindings: vec![IrPatternBinding {
                        field: "target".into(),
                        value: IrPattern::Bind { name: "t".into(), local: LocalRef(0) },
                        span: Span::dummy(),
                    }],
                    span: Span::dummy(),
                }),
                where_clause: None,
                body: vec![IrStmt::Expr(crate::ir::IrExprNode {
                    kind: crate::ir::IrExpr::LitBool(true),
                    span: Span::dummy(),
                })],
                span: Span::dummy(),
            }],
            annotations: vec![],
            cpu_only: false,
            span: Span::dummy(),
        };
        let h1 = rules_hash(&[mk("a"), mk("b")], &[], &[]);
        let h2 = rules_hash(&[mk("b"), mk("a")], &[], &[]);
        assert_eq!(h1, h2, "rules_hash must be sort-stable");
    }

    #[test]
    fn combined_hash_xors_in_each_subhash() {
        let zero = [0u8; 32];
        let mut event = [0u8; 32];
        event[0] = 1;
        let h_with_event = combined_hash(&zero, &event, &zero, &zero, &zero, &zero, &zero);
        let h_all_zero = combined_hash(&zero, &zero, &zero, &zero, &zero, &zero, &zero);
        assert_ne!(h_with_event, h_all_zero, "event change must alter combined");

        // Config sub-hash is part of the mix too.
        let mut cfg = [0u8; 32];
        cfg[0] = 1;
        let h_with_config = combined_hash(&zero, &zero, &zero, &zero, &cfg, &zero, &zero);
        assert_ne!(h_with_config, h_all_zero, "config change must alter combined");

        let mut enums = [0u8; 32];
        enums[0] = 1;
        let h_with_enums = combined_hash(&zero, &zero, &zero, &zero, &zero, &enums, &zero);
        assert_ne!(h_with_enums, h_all_zero, "enums change must alter combined");

        // Views sub-hash is also part of the mix.
        let mut views = [0u8; 32];
        views[0] = 1;
        let h_with_views = combined_hash(&zero, &zero, &zero, &zero, &zero, &zero, &views);
        assert_ne!(h_with_views, h_all_zero, "views change must alter combined");
    }

    #[test]
    fn config_hash_values_do_not_perturb() {
        use crate::ir::{ConfigFieldIR, ConfigIR};

        let mk = |name: &str, default: crate::ast::ConfigDefault| ConfigIR {
            name: name.into(),
            fields: vec![ConfigFieldIR {
                name: "x".into(),
                ty: IrType::F32,
                default,
                span: Span::dummy(),
            }],
            annotations: vec![],
            span: Span::dummy(),
        };
        let h1 = config_hash(&[mk("combat", crate::ast::ConfigDefault::Float(10.0))]);
        let h2 = config_hash(&[mk("combat", crate::ast::ConfigDefault::Float(20.0))]);
        assert_eq!(h1, h2, "default value changes must NOT alter config_hash");

        let h_renamed = config_hash(&[mk("movement", crate::ast::ConfigDefault::Float(10.0))]);
        assert_ne!(h1, h_renamed, "block rename must alter config_hash");
    }

    #[test]
    fn config_hash_is_reorder_stable() {
        use crate::ir::{ConfigFieldIR, ConfigIR};
        let a = ConfigIR {
            name: "alpha".into(),
            fields: vec![ConfigFieldIR {
                name: "x".into(),
                ty: IrType::F32,
                default: crate::ast::ConfigDefault::Float(1.0),
                span: Span::dummy(),
            }],
            annotations: vec![],
            span: Span::dummy(),
        };
        let b = ConfigIR {
            name: "beta".into(),
            fields: vec![ConfigFieldIR {
                name: "y".into(),
                ty: IrType::U32,
                default: crate::ast::ConfigDefault::Uint(2),
                span: Span::dummy(),
            }],
            annotations: vec![],
            span: Span::dummy(),
        };
        let h_ab = config_hash(&[a.clone(), b.clone()]);
        let h_ba = config_hash(&[b, a]);
        assert_eq!(h_ab, h_ba);
    }
}
