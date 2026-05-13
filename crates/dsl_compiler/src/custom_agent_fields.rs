//! Custom per-agent SoA field registry (Gap plague_city#P-A).
//!
//! Closes the closed-enum gap documented in
//! `docs/architecture/gaps_plague_city.md`: `AgentFieldId` was a
//! closed enum mirroring the engine's SoA, so .sim fixtures couldn't
//! declare new per-agent boolean / counter columns without modifying
//! the compiler. Pre-fix workaround was to overload an existing
//! semantically-loaded column (`hunger` repurposed as "infection
//! severity" in plague_city; "ant energy" in foraging_real).
//!
//! ## Surface
//!
//! `.sim` authors declare custom columns at the top level:
//!
//! ```text
//! field infected: u32
//! field infected_since_tick: u32
//! field cult_loyalty: f32
//! ```
//!
//! Each decl auto-allocates a per-agent SoA buffer
//! (`agent_<name>_buf`, sized `agent_count * elem_bytes`), gets
//! reader generation via `self.<name>` / `<binder>.<name>` reads,
//! and setter generation via `agents.set_<name>(target, value)`
//! writes. The new column behaves identically to a built-in
//! `AgentFieldId` variant for the duration of the compile.
//!
//! ## Implementation note
//!
//! `AgentFieldId` is `Copy + Eq + Hash + Ord` and is plumbed through
//! every layer (data_handle → lower → emit → build_helper) by value.
//! Threading a per-compilation registry through every site would
//! explode the change surface. Instead, the registry is **process-
//! local**: `intern_field("infected", U32)` leaks a `'static
//! CustomFieldDesc` and returns its address as a `CustomFieldId`.
//! Pointer identity gives us `Copy + Eq + Hash + Ord` for free; the
//! built-in `snake()` / `ty()` methods on `AgentFieldId::Custom`
//! dereference the leaked descriptor in O(1).
//!
//! The registry is **idempotent**: re-interning the same `(name,
//! ty)` returns the SAME pointer. This makes the registry safe to
//! re-populate across the multiple build_helper invocations a test
//! harness performs in a single process — the second call observes
//! the same set of leaked entries from the first call. (Build.rs
//! invocations are separate processes, so the leak boundary is the
//! test harness, not production.)
//!
//! Re-interning the same name with a CONFLICTING type is a hard
//! error (panic) — every .sim file MUST declare a custom field
//! consistently with itself, and tests cross-checking against shared
//! state would surface the conflict here rather than later in
//! lowering.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::cg::data_handle::AgentFieldTy;

/// Leaked descriptor referenced by `AgentFieldId::Custom`. The
/// `name` is a leaked `String` (so `&'static str` works in all the
/// `&'static str` contexts the rest of the compiler expects); the
/// `ty` is by-value `AgentFieldTy`.
#[derive(Debug)]
pub struct CustomFieldDesc {
    pub name: &'static str,
    pub ty: AgentFieldTy,
}

/// Pointer-identity wrapper around `&'static CustomFieldDesc`. The
/// pointer is interned via [`intern_field`]; two `CustomFieldId`s
/// compare equal iff they reference the same descriptor (which —
/// thanks to interning — happens iff they share a `name`).
///
/// `Eq` / `Hash` / `Ord` are defined by-pointer-identity (NOT by
/// dereffing into `CustomFieldDesc`). Interning guarantees that two
/// `CustomFieldId`s with the same name point to the same descriptor,
/// so pointer-identity is the right equivalence relation. Hand-rolled
/// to avoid the derive machinery recursing through the reference
/// into `CustomFieldDesc` (which is intentionally NOT `Eq + Hash +
/// Ord` — those are derived on the WRAPPER, not the payload).
#[derive(Debug, Copy, Clone)]
pub struct CustomFieldId(pub &'static CustomFieldDesc);

impl PartialEq for CustomFieldId {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}
impl Eq for CustomFieldId {}
impl std::hash::Hash for CustomFieldId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const CustomFieldDesc).hash(state);
    }
}
impl PartialOrd for CustomFieldId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CustomFieldId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by name (stable across runs of the same .sim, even
        // though pointer addresses are not). Sorting by name keeps
        // serde snapshots reproducible across processes.
        self.0.name.cmp(other.0.name)
    }
}

impl CustomFieldId {
    pub fn name(self) -> &'static str {
        self.0.name
    }
    pub fn ty(self) -> AgentFieldTy {
        self.0.ty
    }
}

// Process-local registry: name → leaked descriptor. The `RwLock`
// guards concurrent test runs; the leaked descriptors themselves
// are immutable after insertion so no locking is needed once a
// `CustomFieldId` is in hand.
static REGISTRY: RwLock<Option<HashMap<String, &'static CustomFieldDesc>>> =
    RwLock::new(None);

fn ensure_registry<R>(f: impl FnOnce(&mut HashMap<String, &'static CustomFieldDesc>) -> R) -> R {
    let mut guard = REGISTRY.write().expect("custom-field registry poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

/// Intern a custom field name + type. Re-interning with the same
/// `(name, ty)` returns the SAME pointer (idempotent). Re-interning
/// with a CONFLICTING type panics — the .sim author MUST be
/// internally consistent.
pub fn intern_field(name: &str, ty: AgentFieldTy) -> CustomFieldId {
    ensure_registry(|map| {
        if let Some(existing) = map.get(name) {
            if existing.ty != ty {
                panic!(
                    "custom agent field `{name}` re-declared with conflicting type: \
                     existing={:?}, new={:?}",
                    existing.ty, ty
                );
            }
            return CustomFieldId(*existing);
        }
        let leaked_name: &'static str = Box::leak(name.to_string().into_boxed_str());
        let desc = Box::leak(Box::new(CustomFieldDesc {
            name: leaked_name,
            ty,
        }));
        let id: &'static CustomFieldDesc = desc;
        map.insert(name.to_string(), id);
        CustomFieldId(id)
    })
}

/// Look up a previously-interned field by snake-case name. Returns
/// `None` if no `intern_field(name, _)` call has been observed in
/// this process. This is the lookup `expr.rs`'s `lower_field` /
/// `physics.rs`'s `agents_setter_field` consult after the built-in
/// `AgentFieldId::from_snake` returns `None`.
pub fn lookup_by_snake(name: &str) -> Option<CustomFieldId> {
    REGISTRY
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().and_then(|m| m.get(name).copied().map(CustomFieldId)))
}

/// Bytes per element for a custom field's primitive. Mirrors the
/// `elem_bytes_for_wgsl_ty` table in `build_helper.rs`; used by the
/// per-agent buffer sizing path when the binding name resolves to
/// a custom column.
pub fn elem_bytes(ty: AgentFieldTy) -> u64 {
    match ty {
        AgentFieldTy::F32 | AgentFieldTy::U32 | AgentFieldTy::Bool => 4,
        AgentFieldTy::I16 => 2,
        AgentFieldTy::Vec3 => 16, // std430 vec3 → padded to vec4
        AgentFieldTy::EnumU8 => 4, // widened to u32 on GPU
        AgentFieldTy::OptAgentId | AgentFieldTy::OptEnumU32 => 4,
    }
}

/// Parse the surface type name from `field <name>: <ty_name>`. Today
/// supports `u32`, `f32`, `bool`, and `vec3` — the q8-free arms of
/// `AgentFieldTy` that the registry's buffer-sizing + WGSL-emit
/// paths can already handle generically.
///
/// `vec3` resolves to `AgentFieldTy::Vec3` (the same primitive the
/// built-in `pos` / `vel` columns use). Storage shape: one
/// `vec3<f32>` per agent, std430-padded to 16 bytes per slot — see
/// [`elem_bytes`]. Reads (`agents.<field>(self)` /
/// `self.<field>`) return `vec3<f32>`; writes
/// (`agents.set_<field>(target, v)`) accept `vec3<f32>`. The
/// per-axis-scalar workaround used by `dungeon_stealth.sim`
/// (`patrol_origin_x` / `_y` etc.) is no longer required.
///
/// Returns `None` for unsupported surface names so the caller can
/// surface a typed error.
pub fn parse_field_ty(ty_name: &str) -> Option<AgentFieldTy> {
    Some(match ty_name {
        "u32" => AgentFieldTy::U32,
        "f32" => AgentFieldTy::F32,
        "bool" => AgentFieldTy::Bool,
        "vec3" => AgentFieldTy::Vec3,
        _ => return None,
    })
}

/// Populate the registry from a parsed Program's
/// `Decl::AgentField` entries. Returns the list of interned ids
/// (in source order) for callers that need to know the set —
/// today's caller is [`crate::build_helper`], which uses the names
/// to validate `init { ... }` against custom columns and to size
/// the auto-allocated buffers.
///
/// Panics on unrecognised surface type name; the .sim parser
/// already validated the field shape syntactically — type-name
/// validation lives here.
pub fn populate(program: &dsl_ast::ast::Program) -> Vec<CustomFieldId> {
    let mut ids = Vec::new();
    for decl in &program.decls {
        if let dsl_ast::ast::Decl::AgentField(d) = decl {
            let ty = parse_field_ty(&d.ty_name).unwrap_or_else(|| {
                panic!(
                    "unknown custom field type `{}` for `field {}` at {:?} \
                     (supported: u32, f32, bool, vec3)",
                    d.ty_name, d.name, d.span
                )
            });
            // Reject names that collide with a built-in AgentFieldId.
            if crate::cg::data_handle::AgentFieldId::from_snake_builtin(&d.name).is_some() {
                panic!(
                    "custom field `{}` collides with a built-in AgentFieldId \
                     of the same name; pick a different name",
                    d.name
                );
            }
            ids.push(intern_field(&d.name, ty));
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent() {
        let a = intern_field("test_idempotent_field", AgentFieldTy::U32);
        let b = intern_field("test_idempotent_field", AgentFieldTy::U32);
        assert_eq!(a, b);
        assert_eq!(a.name(), "test_idempotent_field");
        assert_eq!(a.ty(), AgentFieldTy::U32);
    }

    #[test]
    fn lookup_after_intern() {
        let a = intern_field("test_lookup_field", AgentFieldTy::F32);
        let found = lookup_by_snake("test_lookup_field").expect("must be present");
        assert_eq!(a, found);
        assert_eq!(found.ty(), AgentFieldTy::F32);
    }

    #[test]
    fn lookup_misses_unknown() {
        assert!(lookup_by_snake("never_interned_field_xyz").is_none());
    }

    #[test]
    fn parse_supported_types() {
        assert_eq!(parse_field_ty("u32"), Some(AgentFieldTy::U32));
        assert_eq!(parse_field_ty("f32"), Some(AgentFieldTy::F32));
        assert_eq!(parse_field_ty("bool"), Some(AgentFieldTy::Bool));
        assert_eq!(parse_field_ty("vec3"), Some(AgentFieldTy::Vec3));
        assert_eq!(parse_field_ty("blarg"), None);
    }

    #[test]
    fn vec3_field_uses_padded_elem_bytes() {
        // std430 vec3 → vec4-padded → 16 bytes per slot. Mirrors the
        // built-in `Pos` / `Vel` columns the per-fixture buffer
        // alloc loop already sizes correctly via the same table.
        assert_eq!(elem_bytes(AgentFieldTy::Vec3), 16);
    }

    #[test]
    fn intern_vec3_round_trips() {
        let id = intern_field("test_vec3_round_trip_field", AgentFieldTy::Vec3);
        assert_eq!(id.ty(), AgentFieldTy::Vec3);
        let found =
            lookup_by_snake("test_vec3_round_trip_field").expect("must be present");
        assert_eq!(id, found);
        assert_eq!(found.ty(), AgentFieldTy::Vec3);
    }
}
