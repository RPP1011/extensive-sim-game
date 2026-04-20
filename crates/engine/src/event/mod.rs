//! Event vocabulary — now DSL-emitted. See `assets/sim/events.sim`.
//!
//! The engine stopped owning this enum at milestone 2's integration step;
//! the compiler emits it into `engine_rules::events`. This module re-exports
//! the compiler output so existing `use crate::event::Event` call sites
//! keep working without modification.
//!
//! The `EventRing` buffer is an engine-side primitive (it's a ring of
//! events, not a vocabulary) so it stays here.

pub use engine_rules::events::Event;
pub use crate::ids::EventId;

pub mod ring;
pub use ring::EventRing;
