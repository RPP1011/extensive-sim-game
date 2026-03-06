pub mod advanced;
pub mod control;
pub mod core;
pub mod pathing;
pub mod personality;
pub mod roles;
pub mod squad;
pub mod tooling;
pub mod utility;

pub mod spatial {
    pub use super::advanced::run_spatial_sample;
}

pub mod tactics {
    pub use super::advanced::run_tactical_sample;
}

pub mod coordination {
    pub use super::advanced::run_coordination_sample;
}
