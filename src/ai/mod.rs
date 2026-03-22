// Thin re-export layer: all AI code lives in the `tactical_sim` crate.
pub use tactical_sim::sim as core;
pub use tactical_sim::effects;
pub use tactical_sim::pathing;
pub use tactical_sim::squad;
pub use tactical_sim::goap;
pub use tactical_sim::control;
pub use tactical_sim::personality;
pub use tactical_sim::roles;
pub use tactical_sim::utility;
pub use tactical_sim::phase;
pub use tactical_sim::advanced;
pub use tactical_sim::student;
pub use tactical_sim::tooling;

pub mod spatial {
    pub use tactical_sim::advanced::run_spatial_sample;
}

pub mod tactics {
    pub use tactical_sim::advanced::run_tactical_sample;
}

pub mod coordination {
    pub use tactical_sim::advanced::run_coordination_sample;
}
