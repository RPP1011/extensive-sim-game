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
        // Opportunistically request the timestamp-query feature
        // bundle so per-fixture runtime crates can attribute GPU time
        // per kernel via a `wgpu::QuerySet`. We ask for both:
        //
        // - `TIMESTAMP_QUERY` — base feature; lets us create a
        //   `QueryType::Timestamp` query set and resolve it.
        // - `TIMESTAMP_QUERY_INSIDE_ENCODERS` — additionally allows
        //   `encoder.write_timestamp()` *between* compute passes (vs.
        //   only at pass boundaries via `timestamp_writes`). The
        //   compiler-emitted dispatch helpers create their own
        //   compute passes internally, so wrapping each one would
        //   require threading the query set through the emit; using
        //   encoder-level timestamps keeps the boids_runtime
        //   instrumentation external to the generated code.
        //
        // Adapters that don't expose either feature (some software
        // backends, older drivers) get the empty intersection and
        // fall back to "instrumentation unsupported". The sim still
        // runs.
        let wanted = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let supported = adapter.features();
        let required_features = supported & wanted;
        // Use the adapter's reported limits as the floor (defaults
        // to 8 storage buffers per stage; a single multi-verb fixture
        // like tactical_horde_500 binds 11 storage buffers in its
        // scoring kernel — agent_hp + agent_level + 6 mask bitmaps
        // + scoring_output + event_ring + event_tail). Adapters
        // exposing higher limits will see them honoured; adapters
        // that don't are still fed the default downlevel set.
        let mut required_limits = adapter.limits();
        // Floor on the dimensions that fixtures depend on; without
        // these floors a downlevel adapter would force a smaller
        // limit than the kernel emits.
        required_limits.max_storage_buffers_per_shader_stage = required_limits
            .max_storage_buffers_per_shader_stage
            .max(12);
        required_limits.max_bind_groups = required_limits.max_bind_groups.max(4);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine::gpu::GpuContext::device"),
                required_features,
                required_limits,
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

    /// True iff the underlying device supports the encoder-level
    /// timestamp queries the per-fixture runtime uses to attribute
    /// GPU time per dispatch (`encoder.write_timestamp` between
    /// passes). Requires both `TIMESTAMP_QUERY` and
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS`. When false the runtime
    /// should skip every `write_timestamp` / `resolve_query_set` /
    /// readback step rather than panicking; the sim still produces
    /// correct output, just without per-kernel attribution.
    pub fn supports_timestamp_query(&self) -> bool {
        let f = self.device.features();
        f.contains(wgpu::Features::TIMESTAMP_QUERY)
            && f.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    }

    /// Sync constructor — convenience wrapper around
    /// [`Self::new`] using `pollster::block_on`. Lets per-fixture
    /// runtime constructors (`BoidsState::new` etc.) avoid threading
    /// async through their public API.
    pub fn new_blocking() -> Result<Self, GpuContextError> {
        pollster::block_on(Self::new())
    }
}
