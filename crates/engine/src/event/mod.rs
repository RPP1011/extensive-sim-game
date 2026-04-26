//! Event vocabulary primitives — the event TYPE itself is emitted into
//! engine_data; engine declares only the trait that any event enum must
//! satisfy. See Spec B' D13.
//!
//! The `EventRing` buffer is an engine-side primitive (it's a ring of
//! events, not a vocabulary) so it stays here.
//!
//! TRANSITION (Task 2): `impl EventLike for engine_data::events::Event` is
//! now GENERATED — see `event_like_impl.rs` (emitted by dsl_compiler). It
//! lives here (in engine) rather than in engine_data to avoid the dep cycle
//! while engine retains its engine_data regular dep (chronicle.rs, Plan B2).
//! Once Plan B2 drops that dep, the impl can move to engine_data.

pub mod ring;
pub mod event_like_impl;
pub use ring::EventRing;
pub use crate::ids::EventId;

// Task 4 (Plan B1'): re-export of engine_data::events::Event dropped.
// Callers now import engine_data::events::Event directly.

/// Trait every concrete event enum implements. Engine's runtime primitives
/// (`EventRing`, `CascadeRegistry`, `CascadeHandler::handle`, view fold sites)
/// are generic over `E: EventLike`. The compiler emits `impl EventLike for
/// engine_data::Event { ... }` so the kind ordinal stays consistent across
/// regenerations.
pub trait EventLike: Sized + Clone + Send + Sync + 'static {
    fn kind(&self) -> crate::cascade::EventKindId;
    fn tick(&self) -> u32;
    fn is_replayable(&self) -> bool;
    /// Write the replayable bytes of this event into the given hasher.
    /// Called by `EventRing::replayable_sha256`. Only called when
    /// `is_replayable()` returns true. Sidecar fields (`id`, `cause`)
    /// must NOT be included.
    fn hash_replayable(&self, h: &mut sha2::Sha256);
}

// Note: impl EventLike for engine_data::events::Event is now GENERATED.
// See `crates/engine/src/event/event_like_impl.rs` (emitted by dsl_compiler).
// Regenerate with `cargo run --bin xtask -- compile-dsl`.
