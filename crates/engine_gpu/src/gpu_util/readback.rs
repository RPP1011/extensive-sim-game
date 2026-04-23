//! Typed readback helpers. Collapses the `copy_buffer_to_buffer →
//! map_async → device.poll(Wait) → get_mapped_range → cast_slice →
//! unmap` pattern repeated across ~8 sites into one function.

use bytemuck::Pod;

/// Blocking readback. Allocates a throwaway staging buffer, copies
/// `byte_len` bytes from `src`, polls until the callback fires, casts
/// the mapped range to `Vec<T>`.
///
/// Not for the hot path — every call allocates a staging buffer. Use
/// the kernel's pooled staging if the call is per-tick. Fine for
/// tests, init, and the snapshot path (which owns its staging).
pub fn readback_typed<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<T>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_util::readback::staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_util::readback::enc"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);

    rx.recv()
        .map_err(|e| format!("readback channel closed: {e}"))?
        .map_err(|e| format!("readback map_async: {e:?}"))?;

    let data = slice.get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Non-blocking readback handle. Encodes the copy into the caller's
/// encoder (does NOT submit), returns a handle the caller polls later.
/// Used by `snapshot.rs` for the double-buffered staging path.
pub struct PendingReadback<T: Pod + Copy> {
    staging: wgpu::Buffer,
    // Retained for debugging / future range-limited reads; snapshot.rs
    // (Task D1) may inspect this to assert mapped-region length matches
    // the producing encoder's copy.
    #[allow(dead_code)]
    byte_len: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Pod + Copy> PendingReadback<T> {
    pub fn new(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        byte_len: u64,
    ) -> Self {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_util::readback::pending_staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        Self {
            staging,
            byte_len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Issue `map_async` on the staging buffer. Does NOT poll —
    /// caller polls once after all pending readbacks are kicked off.
    pub fn map_read(&self) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx
    }

    /// After `map_read` is awaited, decode mapped range into Vec<T>.
    pub fn take(self) -> Vec<T> {
        let slice = self.staging.slice(..);
        let data = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        out
    }
}
