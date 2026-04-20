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

mod id;
pub use id::AbilityId;

pub mod cast;
// `damage` was deleted at milestone 3 — the DSL-emitted handler at
// `crate::generated::physics::damage::DamageHandler` now carries the rule.
pub mod expire;
pub mod gate;
pub mod gold;
pub mod heal;
pub mod program;
pub mod record_memory;
pub mod registry;
pub mod shield;
pub mod slow;
pub mod standing;
pub mod stun;

pub use cast::CastHandler;
// `DamageHandler` re-export removed at milestone 3 — consumers (engine
// builtin registration + tests) reach the DSL-emitted handler at
// `crate::generated::physics::damage::DamageHandler`.
pub use gate::evaluate_cast_gate;
pub use gold::TransferGoldHandler;
pub use heal::HealHandler;
pub use record_memory::RecordMemoryHandler;
pub use shield::ShieldHandler;
pub use slow::SlowHandler;
pub use standing::ModifyStandingHandler;
pub use stun::StunHandler;
pub use program::{Area, Delivery, EffectOp, Gate, TargetSelector, MAX_EFFECTS_PER_PROGRAM};
pub use program::AbilityProgram;
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
