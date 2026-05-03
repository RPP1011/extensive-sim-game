//! [`Kernel`] — the contract every per-fixture compiler-emitted
//! kernel module implements.
//!
//! ## Why a trait at all
//!
//! Each per-kernel module the DSL compiler emits is an
//! `impl Kernel for <KernelName>` — same contract every fixture's
//! kernels satisfy. The trait gives:
//!   - A uniform `new(&device)` constructor that compiles + caches
//!     the WGSL shader and pipeline.
//!   - A uniform `record(...)` method for encoding a dispatch.
//!
//! ## Why minimal
//!
//! The trait deliberately does NOT expose `build_cfg` or `bind` —
//! those are per-kernel concerns (each kernel's `Cfg` carries
//! different fields; each kernel's `Bindings` carries different
//! buffers). Kernels expose them as inherent methods on the
//! `<KernelName>Kernel` struct; the trait's surface stays small so
//! the platform layer doesn't accumulate per-fixture concerns.

/// Contract every per-fixture compiler-emitted kernel module
/// implements. See module-level docs for the rationale on the
/// minimal surface.
///
/// Implementations are compiler-emitted; engine doesn't define any
/// inherent kernels.
pub trait Kernel {
    /// Per-kernel typed bindings struct. Holds `&'a wgpu::Buffer`
    /// references to the storage / uniform buffers the dispatch
    /// reads + writes. The lifetime parameter ties the bindings to
    /// the buffers' borrow.
    type Bindings<'a>
    where
        Self: 'a;

    /// Per-kernel uniform-buffer payload. Must be POD so it can be
    /// uploaded with `bytemuck::bytes_of`. Each kernel's `Cfg` is
    /// distinct (e.g., `PhysicsMoveBoidCfg { agent_cap, _pad }`).
    type Cfg: bytemuck::Pod + bytemuck::Zeroable + Copy;

    /// Compile the WGSL shader, build the BGL + pipeline-layout +
    /// compute-pipeline, store them in the kernel struct. Called
    /// once per kernel per [`crate::gpu::GpuContext`] — usually at
    /// per-fixture-runtime construction time.
    fn new(device: &wgpu::Device) -> Self
    where
        Self: Sized;

    /// Encode a single dispatch into `encoder`. The encoder is
    /// caller-owned — the per-fixture runtime constructs one
    /// encoder per tick and records every active kernel into it
    /// before submitting.
    ///
    /// `agent_cap` is the upper-bound on alive-agent slot indices
    /// the dispatch should cover; the kernel's WGSL preamble
    /// (`if (agent_id >= cfg.agent_cap) { return; }`) early-exits
    /// per-thread for slots beyond the live count.
    fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &Self::Bindings<'_>,
        agent_cap: u32,
    );
}
