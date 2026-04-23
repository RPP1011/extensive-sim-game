//! Ability subsystem — programs + registry.
//!
//! Module layout:
//! - `id`       — `AbilityId` newtype (NonZeroU32)
//! - `program`  — `AbilityProgram` IR + `EffectOp` / `Area` / `Delivery` / `Gate`
//! - `registry` — `AbilityRegistry` + append-only builder
//!
//! The `cast` cascade dispatch handler lived here until 2026-04-19, when
//! the DSL compiler grew `for ... in <collection>` loops and `match` over
//! sum-type variants (`EffectOp::*`). Cast dispatch is now a plain
//! DSL-emitted physics rule — see `assets/sim/physics.sim::cast` and the
//! generated `crates/engine/src/generated/physics/cast.rs`. `EffectOp` and
//! `TargetSelector` are treated as stdlib-known sum types by the
//! compiler (see `qualified_variant_name` in
//! `crates/dsl_compiler/src/emit_physics.rs`). The last hand-written
//! cascade handler with game logic is retired.
//!
//! Task 157 retired `gate::evaluate_cast_gate` — the caster-side
//! conjunction (alive + un-stunned + cooldown-ready + known +
//! not-engaged-elsewhere) is now `mask Cast(ability: AbilityId)` in
//! `assets/sim/masks.sim`, lowered to `crate::generated::mask::
//! mask_cast`. Target-side filters (target alive, in-range, hostility
//! matches) live on the engine mask-build path in
//! `crate::mask::inferred_cast_target`. This retired the last
//! hand-written game-logic predicate in the engine crate.
//!
//! Task 143 deleted the `expire` module — stun / slow are now stored as
//! absolute expiry ticks (`stun_expires_at_tick` / `slow_expires_at_tick`)
//! rather than per-tick-decremented counters, so the `tick_start_timers`
//! pass this module used to own is gone. The last per-tick reducer is
//! retired; every time-gated combat field is now a synthetic boundary
//! read off `state.tick`.
//!
//! Per-effect cascade handlers (`damage`, `heal`, `shield`, `stun`, `slow`,
//! `transfer_gold`, `modify_standing`, `opportunity_attack`, `record_memory`,
//! `cast`) are compiler-emitted from `assets/sim/physics.sim`; the legacy
//! hand-written files that used to live here (one per effect, plus
//! `record_memory.rs` and `cast.rs`) were deleted as each migrated to DSL.
//! Consumers reach them at `crate::generated::physics::<name>::*`.

mod id;
pub use id::AbilityId;

pub mod program;
pub mod registry;

pub use program::{
    Area, Delivery, EffectOp, Gate, TargetSelector, MAX_ABILITIES, MAX_EFFECTS_PER_PROGRAM,
};
pub use program::AbilityProgram;
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
