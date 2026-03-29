// Thin re-export layer: all AI code lives in the `tactical_sim` crate.
#[allow(unused_imports)]
pub use tactical_sim::sim as core;
#[allow(unused_imports)]
pub use tactical_sim::effects;
#[allow(unused_imports)]
pub use tactical_sim::pathing;
#[allow(unused_imports)]
pub use tactical_sim::squad;
#[allow(unused_imports)]
pub use tactical_sim::goap;
#[allow(unused_imports)]
pub use tactical_sim::control;
#[allow(unused_imports)]
pub use tactical_sim::personality;
#[allow(unused_imports)]
pub use tactical_sim::roles;
#[allow(unused_imports)]
pub use tactical_sim::utility;
#[allow(unused_imports)]
pub use tactical_sim::phase;
#[allow(unused_imports)]
pub use tactical_sim::advanced;
#[allow(unused_imports)]
pub use tactical_sim::student;
#[allow(unused_imports)]
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
