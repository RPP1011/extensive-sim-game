//! Double-buffered snapshot staging. Feeds `GpuBackend::snapshot()`.
//!
//! Two staging pairs: {agents, events} × {front, back}. On each
//! snapshot call:
//!   - map_async on front (filled by the previous snapshot call)
//!   - copy_buffer_to_buffer from live GPU buffers into back
//!   - poll → decode front → GpuSnapshot
//!   - swap front / back
//! First call returns GpuSnapshot::empty().

#![cfg(feature = "gpu")]

use crate::event_ring::{EventRecord, GpuEventRing, RECORD_BYTES};
use crate::physics::GpuAgentSlot;

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub tick: u32,
    pub agents: Vec<GpuAgentSlot>,
    pub events_since_last: Vec<EventRecord>,
    // chronicle_since_last omitted for D1 — chronicle exposure is
    // handled separately if/when needed. Design doc flags it as
    // deferred.
}

impl GpuSnapshot {
    pub fn empty() -> Self {
        Self {
            tick: 0,
            agents: Vec::new(),
            events_since_last: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum SnapshotError {
    Ring(String),
    Map(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Ring(s) => write!(f, "snapshot ring error: {s}"),
            SnapshotError::Map(s) => write!(f, "snapshot map error: {s}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// One side of the double buffer. Holds staging buffers for agents +
/// events and a watermark tuple (event_ring_read, tick).
pub struct GpuStaging {
    agents_staging: wgpu::Buffer,
    events_staging: wgpu::Buffer,
    events_len_bytes: u64,
    tick: u32,
    /// True iff `kick_copy` has been called on this side since the
    /// last `take_snapshot`. `take_snapshot` returns empty if this
    /// is false (prevents double-take).
    filled: bool,
    /// Capacity the staging is sized for — used to skip-resize
    /// when agent_cap grows beyond this.
    agent_cap: u32,
    event_ring_cap: u32,
}

impl GpuStaging {
    /// Create a fresh staging sized for `agent_cap` + `event_ring_cap`.
    /// Both values can grow on a later call via `ensure_cap`.
    pub fn new(device: &wgpu::Device, agent_cap: u32, event_ring_cap: u32) -> Self {
        let agent_bytes = Self::agent_bytes_for(agent_cap);
        let event_bytes = Self::event_bytes_for(event_ring_cap);
        Self {
            agents_staging: Self::create_staging(device, "snapshot::agents_staging", agent_bytes),
            events_staging: Self::create_staging(device, "snapshot::events_staging", event_bytes),
            events_len_bytes: 0,
            tick: 0,
            filled: false,
            agent_cap,
            event_ring_cap,
        }
    }

    fn agent_bytes_for(agent_cap: u32) -> u64 {
        (agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64)
    }

    fn event_bytes_for(event_ring_cap: u32) -> u64 {
        (event_ring_cap as u64) * RECORD_BYTES
    }

    fn create_staging(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        // Allocate at least 4 bytes so we never create a zero-sized
        // buffer (wgpu rejects those).
        let size = size.max(4);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Grow staging if agent_cap or event_ring_cap has increased.
    /// No-op if both are already sufficient.
    pub fn ensure_cap(&mut self, device: &wgpu::Device, agent_cap: u32, event_ring_cap: u32) {
        if agent_cap > self.agent_cap {
            self.agents_staging = Self::create_staging(
                device,
                "snapshot::agents_staging",
                Self::agent_bytes_for(agent_cap),
            );
            self.agent_cap = agent_cap;
            // Growing invalidates any previously-kicked copy —
            // reset the filled flag so take_snapshot returns empty.
            self.filled = false;
        }
        if event_ring_cap > self.event_ring_cap {
            self.events_staging = Self::create_staging(
                device,
                "snapshot::events_staging",
                Self::event_bytes_for(event_ring_cap),
            );
            self.event_ring_cap = event_ring_cap;
            self.filled = false;
        }
    }

    /// Encode copy_buffer_to_buffer for agents + event slice into
    /// this staging. Does not submit. Caller must call queue.submit
    /// after this returns.
    ///
    /// `event_ring_start / event_ring_end` are record indices, not
    /// byte offsets. The slice copied is
    /// `records[event_ring_start..event_ring_end]` via
    /// `copy_buffer_to_buffer` on the records buffer.
    pub fn kick_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        agent_bytes: u64,
        event_ring: &GpuEventRing,
        event_ring_start: u64,
        event_ring_end: u64,
        tick: u32,
    ) {
        encoder.copy_buffer_to_buffer(agents_buf, 0, &self.agents_staging, 0, agent_bytes);
        let slice_len = event_ring_end.saturating_sub(event_ring_start);
        if slice_len > 0 {
            encoder.copy_buffer_to_buffer(
                event_ring.records_buffer(),
                event_ring_start * RECORD_BYTES,
                &self.events_staging,
                0,
                slice_len * RECORD_BYTES,
            );
        }
        self.events_len_bytes = slice_len * RECORD_BYTES;
        self.tick = tick;
        self.filled = true;
    }

    /// Map the staging for read, poll, decode into a snapshot,
    /// unmap. Assumes a previous `kick_copy` was followed by a
    /// `queue.submit`. If no copy has been kicked since construction,
    /// returns `GpuSnapshot::empty()`.
    pub fn take_snapshot(
        &mut self,
        device: &wgpu::Device,
        agent_count: usize,
    ) -> Result<GpuSnapshot, SnapshotError> {
        if !self.filled {
            return Ok(GpuSnapshot::empty());
        }
        let agent_bytes = (agent_count * std::mem::size_of::<GpuAgentSlot>()) as u64;
        if agent_bytes == 0 {
            self.filled = false;
            return Ok(GpuSnapshot {
                tick: self.tick,
                agents: Vec::new(),
                events_since_last: Vec::new(),
            });
        }

        let agents_slice = self.agents_staging.slice(..agent_bytes);
        let (atx, arx) = std::sync::mpsc::channel();
        agents_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = atx.send(r); });

        let (events_slice_opt, erx_opt) = if self.events_len_bytes > 0 {
            let slice = self.events_staging.slice(..self.events_len_bytes);
            let (etx, erx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = etx.send(r); });
            (Some(slice), Some(erx))
        } else {
            (None, None)
        };

        let _ = device.poll(wgpu::PollType::Wait);

        arx.recv()
            .map_err(|e| SnapshotError::Map(format!("agents rx closed: {e}")))?
            .map_err(|e| SnapshotError::Map(format!("agents map: {e:?}")))?;
        let agents: Vec<GpuAgentSlot> =
            bytemuck::cast_slice(&agents_slice.get_mapped_range()).to_vec();
        self.agents_staging.unmap();

        let events = if let (Some(slice), Some(erx)) = (events_slice_opt, erx_opt) {
            erx.recv()
                .map_err(|e| SnapshotError::Map(format!("events rx closed: {e}")))?
                .map_err(|e| SnapshotError::Map(format!("events map: {e:?}")))?;
            let data: Vec<EventRecord> =
                bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            self.events_staging.unmap();
            data
        } else {
            Vec::new()
        };

        self.filled = false;
        Ok(GpuSnapshot {
            tick: self.tick,
            agents,
            events_since_last: events,
        })
    }
}
