//! `AbilityRegistry` — append-only, slot-stable table of compiled
//! `AbilityProgram`s built once at startup and read-only thereafter.
//!
//! The registry is built via `AbilityRegistryBuilder::register()`, which hands
//! back a freshly-minted `AbilityId` that remains valid for the lifetime of
//! the built registry (slots are never removed or reordered). Lookup is an
//! O(1) `Vec` index on the slot; unknown ids return `None` without panicking.
//!
//! Why append-only: cast-dispatch (`CastHandler`, Task 9) takes an
//! `Arc<AbilityRegistry>` and runs on the hot tick path. Mutation after
//! build would require an RwLock and would break determinism if a register
//! happened mid-cascade. Append-only at build time + frozen at runtime keeps
//! the contract simple.

use super::{AbilityId, AbilityProgram};

pub struct AbilityRegistry {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistry {
    /// Empty registry. Primarily useful for tests that want to construct a
    /// registry without the builder (e.g. to exercise `get(unknown_id)`).
    pub fn new() -> Self {
        Self { programs: Vec::new() }
    }

    /// Look up the program behind `id`. Returns `None` for out-of-range ids —
    /// callers that need a bool "is this id known" can use `.is_some()`.
    #[inline]
    pub fn get(&self, id: AbilityId) -> Option<&AbilityProgram> {
        self.programs.get(id.slot())
    }

    #[inline]
    pub fn len(&self) -> usize { self.programs.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.programs.is_empty() }
}

impl Default for AbilityRegistry {
    fn default() -> Self { Self::new() }
}

/// Single-use builder — `register()` N times, then `build()` to freeze.
pub struct AbilityRegistryBuilder {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistryBuilder {
    pub fn new() -> Self {
        Self { programs: Vec::new() }
    }

    /// Append `program` and hand back its stable `AbilityId`. IDs are
    /// 1-based (the NonZeroU32 niche means `AbilityId::new(0)` is reserved).
    pub fn register(&mut self, program: AbilityProgram) -> AbilityId {
        self.programs.push(program);
        AbilityId::new(self.programs.len() as u32)
            .expect("programs.len() > 0 immediately after push")
    }

    /// Freeze the builder into an immutable registry.
    pub fn build(self) -> AbilityRegistry {
        AbilityRegistry { programs: self.programs }
    }
}

impl Default for AbilityRegistryBuilder {
    fn default() -> Self { Self::new() }
}
