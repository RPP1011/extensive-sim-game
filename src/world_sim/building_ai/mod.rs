//! NPC Building Intelligence — shared types and interface contracts.
//!
//! This module defines the types that four parallel workstreams code against:
//! - **Implementation gaps**: extends WorldState with new fields (memory, building detail)
//! - **Utility functions**: spatial feature computation, memory promotion
//! - **Scenario design**: seed generation, challenge injection, observation assembly
//! - **Testing harness**: action application, forward simulation, outcome scoring
//!
//! Scenarios are defined in TOML files under `building_scenarios/` and can be
//! modified without recompilation.

pub mod types;
pub mod scenario_config;
pub mod scenario_gen;
pub mod features;
pub mod oracle;
pub mod scoring;
pub mod mass_gen;
pub mod validation;
