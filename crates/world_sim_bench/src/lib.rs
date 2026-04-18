#![cfg_attr(nightly, feature(portable_simd))]

//! Criterion regression harness for world-sim hot loops.
//!
//! Load committed bincode fixtures (one per population scale) and run
//! criterion benches comparing Scalar vs Simd backends on each target
//! loop. The `nightly` cfg gates `std::simd`-using candidates; scalar
//! baselines always compile on stable.

pub mod fixtures;
