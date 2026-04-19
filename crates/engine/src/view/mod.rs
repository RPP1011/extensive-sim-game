//! Derived views over simulation state. Three storage modes:
//! - `materialized`: full per-entity Vec, updated every tick via `fold()`.
//! - `lazy`: computed on demand, staleness-tracked.
//! - `topk`: fixed-size top-K per entity (Phase 2 task).

pub mod lazy;
pub mod materialized;
pub mod topk;

pub use lazy::{LazyView, NearestEnemyLazy};
pub use materialized::{DamageTaken, MaterializedView};
pub use topk::{MostHostileTopK, TopKView};
