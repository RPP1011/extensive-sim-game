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

    /// Plan I-step-3 (hot-reload primitive). Produce a NEW registry
    /// with the program at `id` replaced. Slot ids stay stable
    /// (`out.get(id)` returns `Some(new_program)`; every other slot
    /// is cloned through unchanged), preserving the append-only
    /// contract for live registries that are still in use.
    ///
    /// This is the building block hot-reload runtimes call after
    /// re-parsing a changed `.ability` file: the runtime keeps an
    /// `Arc<AbilityRegistry>`, builds a fresh registry via this
    /// method (the old one stays immutable for any in-flight cast
    /// dispatch), then atomically swaps the Arc. Since the old Arc
    /// keeps the old registry alive until its references drain, no
    /// reader sees a torn write — the determinism contract from
    /// the docstring holds.
    ///
    /// Returns `None` if `id` is out of range. The caller should
    /// surface this as a hot-reload error rather than panicking;
    /// a stale id usually means the file was renamed or its
    /// `register()` order changed and the runtime needs to do a
    /// full rebuild instead of an in-place swap.
    pub fn with_program_replaced(
        &self,
        id: AbilityId,
        new_program: AbilityProgram,
    ) -> Option<AbilityRegistry> {
        let slot = id.slot();
        if slot >= self.programs.len() {
            return None;
        }
        let mut programs = self.programs.clone();
        programs[slot] = new_program;
        Some(AbilityRegistry { programs })
    }
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
