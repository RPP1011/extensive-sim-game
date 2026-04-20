//! Ability subsystem — programs, registry, cast dispatch, and the cast
//! gate predicate.
//!
//! Module layout (Combat Foundation Tasks 6–9):
//! - `id`       — `AbilityId` newtype (NonZeroU32)
//! - `program`  — `AbilityProgram` IR + `EffectOp` / `Area` / `Delivery` / `Gate`
//! - `registry` — `AbilityRegistry` + append-only builder
//! - `cast`     — `CastHandler` cascade dispatcher (one handler, branches on EffectOp)
//! - `gate`     — `evaluate_cast_gate` mask predicate
//!
//! Task 143 deleted the `expire` module — stun / slow are now stored as
//! absolute expiry ticks (`stun_expires_at_tick` / `slow_expires_at_tick`)
//! rather than per-tick-decremented counters, so the `tick_start_timers`
//! pass this module used to own is gone. The last per-tick reducer is
//! retired; every time-gated combat field is now a synthetic boundary
//! read off `state.tick`.
//!
//! Per-effect cascade handlers (`damage`, `heal`, `shield`, `stun`, `slow`,
//! `transfer_gold`, `modify_standing`, `opportunity_attack`, `record_memory`)
//! are compiler-emitted from `assets/sim/physics.sim`; the legacy hand-
//! written files that used to live here (one per effect, plus
//! `record_memory.rs`) were deleted as each effect migrated to DSL.
//! Consumers reach them at `crate::generated::physics::<name>::*`.

mod id;
pub use id::AbilityId;

pub mod cast;
pub mod gate;
pub mod program;
pub mod registry;

pub use cast::CastHandler;
pub use gate::evaluate_cast_gate;
pub use program::{Area, Delivery, EffectOp, Gate, TargetSelector, MAX_EFFECTS_PER_PROGRAM};
pub use program::AbilityProgram;
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
