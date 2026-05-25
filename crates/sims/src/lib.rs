//! `sims` — single-crate home for compiler-emitted fixtures. Each
//! `assets/sim/*.sim` becomes `sims::<fixture>::GeneratedRuntime`
//! with no per-fixture crate boilerplate. Replaces the
//! `crates/*_runtime` sprawl for new fixtures. See `build.rs` for the
//! auto-discovery + emit logic.

pub mod summon_alloc;
pub mod vampire_survivors_seed;

include!(concat!(env!("OUT_DIR"), "/sim_modules.rs"));
