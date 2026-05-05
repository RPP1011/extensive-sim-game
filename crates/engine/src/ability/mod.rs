//! Ability subsystem — programs + registry.
//!
//! Module layout:
//! - `id`       — `AbilityId` newtype (NonZeroU32)
//! - `program`  — `AbilityProgram` IR + `EffectOp` / `Area` / `Delivery` / `Gate`
//! - `registry` — `AbilityRegistry` + append-only builder
//!
//! Today the engine ships only the host-side ability data model:
//! `AbilityProgram` (the lowered IR per ability), `AbilityRegistry`
//! (the slot-stable `Vec<AbilityProgram>`), and `PackedAbilityRegistry`
//! (the SoA layout for GPU consumption — Wave 1.9). Cast dispatch
//! itself is **per-fixture**, not engine-wide: each per-fixture `.sim`
//! authors its own `verb` declarations and `physics Apply*` chronicle
//! blocks (see `assets/sim/duel_1v1.sim` or `assets/sim/duel_abilities.sim`
//! for the canonical 1v1 shape). There is no global `physics.sim`
//! cast cascade today; the previous engine-side cascade-handler design
//! (`gate::evaluate_cast_gate`, `crate::generated::physics::cast`,
//! `assets/sim/masks.sim`) was deleted in the 2026-05-02 wolf-sim
//! wipe. A registry-driven kernel-emit path (so kernels read
//! `PackedAbilityRegistry` storage buffers and dispatch on `AbilityId`
//! rather than reading hand-mirrored .sim verb constants) is Wave 2+
//! work.
//!
//! Task 143 deleted the `expire` module — stun / slow are now stored as
//! absolute expiry ticks (`stun_expires_at_tick` / `slow_expires_at_tick`)
//! rather than per-tick-decremented counters, so the `tick_start_timers`
//! pass this module used to own is gone. The last per-tick reducer is
//! retired; every time-gated combat field is now a synthetic boundary
//! read off `state.tick`.
//!
//! Per-effect dispatch is per-fixture — each `.sim` file owns its own
//! `physics ApplyDamage` / `ApplyHeal` / `ApplyShield` / etc. blocks
//! (compiler-emitted into `OUT_DIR/generated.rs` of the per-fixture
//! runtime crate). The engine crate ships zero hand-written cascade
//! handlers. The `EffectOp` catalog (Damage / Heal / Shield / Stun /
//! Slow / TransferGold / ModifyStanding / CastAbility) is the
//! canonical lowering target for each `.ability` effect verb.

mod id;
pub use id::AbilityId;

pub mod packed;
pub mod program;
pub mod registry;

pub use program::{
    Area, Delivery, EffectOp, Gate, TargetSelector, MAX_ABILITIES, MAX_EFFECTS_PER_PROGRAM,
};
pub use program::AbilityProgram;
pub use program::{AbilityHint, AbilityTag, MAX_TAGS_PER_PROGRAM};
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
pub use packed::{
    PackedAbilityRegistry, EFFECT_KIND_EMPTY, HINT_NONE_SENTINEL, NUM_ABILITY_TAGS,
};
