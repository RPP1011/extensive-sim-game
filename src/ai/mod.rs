pub mod advanced;
pub mod control;
pub mod core;
pub mod effects;
pub mod goap;
pub mod pathing;
pub mod personality;
pub mod phase;
pub mod roles;
pub mod squad;
pub mod student;
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
