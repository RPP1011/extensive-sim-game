pub mod trait_;
pub mod registry;
pub mod builtins;

pub use trait_::{FailureMode, Invariant, Violation};
pub use registry::InvariantRegistry;
pub use builtins::{MaskValidityInvariant, PoolNonOverlapInvariant};
