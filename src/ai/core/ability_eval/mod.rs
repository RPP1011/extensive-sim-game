//! Ability priority evaluator — interrupt-driven ability decision system.
//!
//! Each ability category has a tiny model that fires when an ability is ready,
//! outputting an urgency score (0-1) and a target. The highest-urgency ability
//! above a threshold interrupts normal combat decisions.

mod categories;
mod features;
mod features_aoe;
mod oracle_scoring;
mod weights;
mod eval;
mod dataset;

#[allow(unused_imports)]
pub use categories::*;
#[allow(unused_imports)]
pub use features::*;
#[allow(unused_imports)]
pub use features_aoe::*;
#[allow(unused_imports)]
pub use oracle_scoring::*;
#[allow(unused_imports)]
pub use weights::*;
#[allow(unused_imports)]
pub use eval::*;
#[allow(unused_imports)]
pub use dataset::*;
