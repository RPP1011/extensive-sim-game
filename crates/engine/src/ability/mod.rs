//! Ability subsystem — programs, registry, cast dispatch, and the unified
//! tick-start phase that decrements combat-timing fields and recomputes
//! engagement bindings.
//!
//! Module layout (Combat Foundation Tasks 3 + 6–9):
//! - `id`       — `AbilityId` newtype (NonZeroU32)
//! - `program`  — `AbilityProgram` IR + `EffectOp` / `Area` / `Delivery` / `Gate`
//! - `registry` — `AbilityRegistry` + append-only builder
//! - `cast`     — `CastHandler` cascade dispatcher (one handler, branches on EffectOp)
//! - `gate`     — `evaluate_cast_gate` mask predicate
//! - `expire`   — tick-start unified pass (decrement + expire + engagement)
//!
//! Per-effect cascade handlers (`damage`, `heal`, `shield`, `stun`, `slow`,
//! `transfer_gold`, `modify_standing`, `opportunity_attack`) are compiler-
//! emitted from `assets/sim/physics.sim`; the legacy hand-written files that
//! used to live here (one per effect) were deleted as each effect migrated
//! to DSL. Consumers reach them at `crate::generated::physics::<name>::*`.

mod id;
pub use id::AbilityId;

pub mod cast;
pub mod expire;
pub mod gate;
pub mod program;
pub mod record_memory;
pub mod registry;

pub use cast::CastHandler;
pub use gate::evaluate_cast_gate;
pub use record_memory::RecordMemoryHandler;
pub use program::{Area, Delivery, EffectOp, Gate, TargetSelector, MAX_EFFECTS_PER_PROGRAM};
pub use program::AbilityProgram;
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
