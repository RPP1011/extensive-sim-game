//! [`GpuContext`] — the headless wgpu device + queue per-fixture
//! runtimes use to build kernels and dispatch compute.
//!
//! Sim-agnostic. No surface (compute-only); per-fixture runtimes
//! that eventually want a window create the surface against the same
//! adapter independently.

/// Aggregate of the four wgpu handles every per-fixture runtime
/// needs: instance, adapter, device, queue.
///
/// All four are stored together because they have a fixed lifetime
/// dependency (instance outlives adapter outlives device outlives
/// queue) and per-fixture runtimes generally hold one [`GpuContext`]
/// for the duration of the simulation.
///
/// Constructed via [`GpuContext::new`] (async) or
/// [`GpuContext::new_blocking`] (sync; uses `pollster::block_on`).
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Failure modes for [`GpuContext::new`].
///
/// Wgpu's adapter / device requests are fallible — no compatible
/// adapter, device-creation rejection. Surface as a typed error so
/// callers (per-fixture runtime constructors) can propagate without
/// `unwrap`.
#[derive(Debug)]
pub enum GpuContextError {
    /// `Instance::request_adapter` returned None — no compatible
    /// physical device. On headless hosts this typically means
    /// neither a discrete GPU nor a software-backed adapter (lavapipe,
    /// etc.) is available.
    NoAdapter,

    /// `Adapter::request_device` rejected the requested limits / features.
    DeviceRequest(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuContextError::NoAdapter => f.write_str(
                "no compatible wgpu adapter found (no discrete GPU + no software fallback)",
            ),
            GpuContextError::DeviceRequest(e) => write!(f, "wgpu device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuContextError::NoAdapter => None,
            GpuContextError::DeviceRequest(e) => Some(e),
        }
    }
}

impl GpuContext {
    /// Construct a headless GPU context. Async because wgpu's adapter
    /// + device requests are async; per-fixture runtime constructors
    /// (which today are sync) should call [`Self::new_blocking`]
    /// instead.
    pub async fn new() -> Result<Self, GpuContextError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuContextError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine::gpu::GpuContext::device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(GpuContextError::DeviceRequest)?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Sync constructor — convenience wrapper around
    /// [`Self::new`] using `pollster::block_on`. Lets per-fixture
    /// runtime constructors (`BoidsState::new` etc.) avoid threading
    /// async through their public API.
    pub fn new_blocking() -> Result<Self, GpuContextError> {
        pollster::block_on(Self::new())
    }
}
