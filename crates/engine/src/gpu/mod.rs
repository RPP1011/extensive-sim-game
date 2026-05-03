//! `engine::gpu` — sim-agnostic GPU platform primitives.
//!
//! This module is the SDL-style platform layer: wgpu device + queue
//! setup, the [`Kernel`] trait per-fixture compiler-emitted dispatch
//! modules implement against, and the [`bgl_storage`] / [`bgl_uniform`]
//! BGL-entry helpers those modules call.
//!
//! Per-fixture runtime crates (`<fixture>_runtime`) depend on engine
//! and use this module to load the WGSL kernels their build.rs's
//! `dsl_compiler` invocation regenerated. Engine itself knows nothing
//! about specific entities, events, or fixtures — those concerns live
//! exclusively in the per-fixture runtime crates.
//!
//! ## Compile-time cost
//!
//! wgpu pulls in a substantial dependency tree (naga, wgpu-core,
//! wgpu-hal, etc.). engine is the deliberate home for that cost: it
//! rebuilds rarely (platform changes), and per-fixture runtime crates
//! pull in wgpu transitively without redeclaring it. The "engine
//! compile time vs game compile time" split — engine is SDL-equivalent
//! (slow, infrequent), per-fixture runtime + sim_app are the fast-
//! iteration layer.

pub mod bgl;
pub mod context;
pub mod kernel;

pub use bgl::{bgl_storage, bgl_uniform};
pub use context::{GpuContext, GpuContextError};
pub use kernel::Kernel;
