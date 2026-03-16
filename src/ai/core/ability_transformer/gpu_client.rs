//! Shared-memory client for GPU inference server.
//!
//! Rayon threads submit requests via crossbeam channels. A batcher thread
//! collects them into batches, writes to a shared memory region, and polls
//! for the Python GPU server's response.
//!
//! Global header (512 bytes):
//!   [0x00] magic: u32 = 0x47505549
//!   [0x04] version: u32 = 1
//!   [0x08] cls_dim: u32
//!   [0x0C] max_batch_size: u32
//!   [0x10] sample_size: u32
//!   [0x14] response_sample_size: u32
//!   [0x18] h_dim: u32 (GRU hidden dimension, 0 = no GRU)
//!   [0x40] flag: u32 (0=idle, 1=request_ready, 2=response_ready)
//!   [0x44] batch_size: u32
//!   [0x80] reload_path: 256 bytes (null-terminated)
//!   [0x180] reload_request: u32
//!   [0x184] reload_ack: u32
//!
//!   [512..] request_data: max_batch × sample_size
//!   [512 + req_region..] response_data: max_batch × response_sample_size

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

const MAX_ENTITIES: usize = 20;
const MAX_THREATS: usize = 6;
const MAX_POSITIONS: usize = 8;
const MAX_ZONES: usize = 10;
const ENTITY_DIM: usize = 34;
const THREAT_DIM: usize = 10;
const POSITION_DIM: usize = 8;
const ZONE_DIM: usize = 12;
const MAX_ABILITIES: usize = 8;
const NUM_COMBAT_TYPES: usize = 10;

const SHM_MAGIC: u32 = 0x47505549;
const OFF_MAGIC: usize = 0;
const OFF_CLS_DIM: usize = 8;
const OFF_MAX_BATCH: usize = 12;
const OFF_SAMPLE_SIZE: usize = 16;
const OFF_RESPONSE_SAMPLE_SIZE: usize = 0x14;
const OFF_H_DIM: usize = 0x18;
const OFF_AGG_DIM: usize = 0x1C;
const OFF_FLAG: usize = 0x40;
const OFF_BATCH_SIZE: usize = 0x44;
const HEADER_SIZE: usize = 512;

#[derive(Clone)]
pub struct InferenceRequest {
    pub entities: Vec<Vec<f32>>,
    pub entity_types: Vec<u8>,
    pub threats: Vec<Vec<f32>>,
    pub positions: Vec<Vec<f32>>,
    pub zones: Vec<Vec<f32>>,
    pub combat_mask: Vec<bool>,
    pub ability_cls: Vec<Option<Vec<f32>>>,
    pub hidden_state: Vec<f32>,
    pub aggregate_features: Vec<f32>,
    /// Corner tokens for spatial cross-attention (V6). Up to 8 corners × 11 features.
    pub corner_tokens: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub move_dx: f32,
    pub move_dy: f32,
    pub combat_type: u8,
    pub target_idx: u16,
    pub lp_move: f32,
    pub lp_combat: f32,
    pub lp_pointer: f32,
    pub hidden_state_out: Vec<f32>,
}

struct BatchItem {
    request: InferenceRequest,
    response_tx: Sender<InferenceResult>,
}

/// Opaque token for non-blocking inference. Returned by `submit()`, used with `try_recv()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferenceToken(pub u64);

thread_local! {
    static PENDING: RefCell<HashMap<InferenceToken, Receiver<InferenceResult>>> =
        RefCell::new(HashMap::new());
}

/// Trait for GPU inference clients (both SHM and Burn).
pub trait InferenceClient: Send + Sync {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String>;
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String>;
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String>;
    fn batch_epoch(&self) -> u64;
    fn wait_for_batch(&self, since: u64);
    fn h_dim(&self) -> usize;
}

pub struct GpuInferenceClient {
    request_tx: Sender<BatchItem>,
    next_token: AtomicU64,
    /// Batch completion counter — incremented by batcher after each response dispatch.
    /// Threads call `wait_for_batch()` to park until this changes.
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
    /// GRU hidden state dimension (0 = no GRU).
    h_dim: usize,
    /// Aggregate feature dimension (0 = no aggregate features).
    agg_dim: usize,
}

/// Condvar-based notification for batch completion.
struct BatchNotify {
    mutex: std::sync::Mutex<()>,
    condvar: std::sync::Condvar,
}

impl BatchNotify {
    fn new() -> Self {
        Self {
            mutex: std::sync::Mutex::new(()),
            condvar: std::sync::Condvar::new(),
        }
    }

    fn notify_all(&self) {
        let _lock = self.mutex.lock().unwrap();
        self.condvar.notify_all();
    }

    fn wait_timeout(&self, timeout: std::time::Duration) {
        let lock = self.mutex.lock().unwrap();
        let _ = self.condvar.wait_timeout(lock, timeout);
    }
}

impl GpuInferenceClient {
    pub fn new(
        shm_path: &str,
        max_batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Result<Arc<Self>, String> {
        let file = {
            let mut attempts = 0;
            loop {
                match std::fs::OpenOptions::new().read(true).write(true).open(shm_path) {
                    Ok(f) => break f,
                    Err(e) => {
                        attempts += 1;
                        if attempts > 100 {
                            return Err(format!("Timed out waiting for SHM {shm_path}: {e}"));
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            }
        };

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file) }
            .map_err(|e| format!("Failed to mmap {shm_path}: {e}"))?;

        let magic = u32::from_le_bytes(mmap[OFF_MAGIC..OFF_MAGIC+4].try_into().unwrap());
        if magic != SHM_MAGIC {
            return Err(format!("Bad SHM magic: {magic:#x}"));
        }
        let cls_dim = u32::from_le_bytes(mmap[OFF_CLS_DIM..OFF_CLS_DIM+4].try_into().unwrap()) as usize;
        let server_max_batch = u32::from_le_bytes(mmap[OFF_MAX_BATCH..OFF_MAX_BATCH+4].try_into().unwrap()) as usize;
        let sample_size = u32::from_le_bytes(mmap[OFF_SAMPLE_SIZE..OFF_SAMPLE_SIZE+4].try_into().unwrap()) as usize;
        let response_sample_size = u32::from_le_bytes(mmap[OFF_RESPONSE_SAMPLE_SIZE..OFF_RESPONSE_SAMPLE_SIZE+4].try_into().unwrap()) as usize;
        let h_dim = u32::from_le_bytes(mmap[OFF_H_DIM..OFF_H_DIM+4].try_into().unwrap()) as usize;
        let agg_dim = u32::from_le_bytes(mmap[OFF_AGG_DIM..OFF_AGG_DIM+4].try_into().unwrap()) as usize;
        let effective_max_batch = max_batch_size.min(server_max_batch);

        eprintln!("GPU SHM connected: cls_dim={cls_dim}, max_batch={effective_max_batch}, sample_size={sample_size}, response_sample_size={response_sample_size}, h_dim={h_dim}, agg_dim={agg_dim}");

        let (request_tx, request_rx) = bounded::<BatchItem>(effective_max_batch * 32);
        let batch_epoch = Arc::new(AtomicU64::new(0));
        let batch_notify = Arc::new(BatchNotify::new());

        let batcher = BatcherThread {
            mmap,
            request_rx,
            cls_dim,
            h_dim,
            agg_dim,
            sample_size,
            response_sample_size,
            max_batch_size: effective_max_batch,
            batch_timeout_ms,
            batch_epoch: batch_epoch.clone(),
            batch_notify: batch_notify.clone(),
        };
        std::thread::spawn(move || batcher.run());

        Ok(Arc::new(Self {
            request_tx,
            next_token: AtomicU64::new(0),
            batch_epoch,
            batch_notify,
            h_dim,
            agg_dim,
        }))
    }

    /// Blocking inference: submit request and wait for result.
    pub fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> {
        let (response_tx, response_rx) = bounded(1);
        self.request_tx
            .send(BatchItem { request, response_tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        response_rx
            .recv()
            .map_err(|_| "No response from batcher".to_string())
    }

    /// Non-blocking submit: enqueue request, return token for later polling.
    pub fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> {
        let token = InferenceToken(self.next_token.fetch_add(1, Ordering::Relaxed));
        let (response_tx, response_rx) = bounded(1);
        self.request_tx
            .send(BatchItem { request, response_tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        PENDING.with(|p| p.borrow_mut().insert(token, response_rx));
        Ok(token)
    }

    /// Poll for a submitted request's result.
    /// Returns `Ok(Some(result))` if ready, `Ok(None)` if pending, `Err` on disconnect.
    pub fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String> {
        PENDING.with(|p| {
            let mut map = p.borrow_mut();
            let rx = match map.get(&token) {
                Some(rx) => rx,
                None => return Err("Unknown inference token".to_string()),
            };
            match rx.try_recv() {
                Ok(result) => { map.remove(&token); Ok(Some(result)) }
                Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    map.remove(&token);
                    Err("Batcher disconnected".to_string())
                }
            }
        })
    }

    /// Get the current batch epoch (monotonically increasing counter).
    pub fn batch_epoch(&self) -> u64 {
        self.batch_epoch.load(Ordering::Acquire)
    }

    /// Block until the batch epoch advances past `since`, or timeout (10ms).
    /// Use this instead of busy-polling to avoid wasting CPU.
    pub fn wait_for_batch(&self, since: u64) {
        if self.batch_epoch.load(Ordering::Acquire) != since {
            return; // already advanced
        }
        self.batch_notify.wait_timeout(std::time::Duration::from_millis(10));
    }

    /// GRU hidden state dimension (0 = no GRU).
    pub fn h_dim(&self) -> usize {
        self.h_dim
    }

    /// Aggregate feature dimension (0 = no aggregate features).
    pub fn agg_dim(&self) -> usize {
        self.agg_dim
    }
}

impl InferenceClient for GpuInferenceClient {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> { self.infer(request) }
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> { self.submit(request) }
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String> { self.try_recv(token) }
    fn batch_epoch(&self) -> u64 { self.batch_epoch() }
    fn wait_for_batch(&self, since: u64) { self.wait_for_batch(since) }
    fn h_dim(&self) -> usize { self.h_dim() }
}

struct BatcherThread {
    mmap: memmap2::MmapMut,
    request_rx: Receiver<BatchItem>,
    cls_dim: usize,
    h_dim: usize,
    agg_dim: usize,
    sample_size: usize,
    response_sample_size: usize,
    max_batch_size: usize,
    batch_timeout_ms: u64,
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
}

impl BatcherThread {
    fn run(mut self) {
        let dummy = InferenceResult {
            move_dx: 0.0, move_dy: 0.0, combat_type: 1, target_idx: 0,
            lp_move: -2.2, lp_combat: -0.7, lp_pointer: 0.0,
            hidden_state_out: vec![0.0; self.h_dim],
        };

        loop {
            let mut batch: Vec<BatchItem> = Vec::with_capacity(self.max_batch_size);

            match self.request_rx.recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }

            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(self.batch_timeout_ms);
            while batch.len() < self.max_batch_size {
                match self.request_rx.recv_deadline(deadline) {
                    Ok(item) => batch.push(item),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }

            let batch_size = batch.len();
            if batch_size == 0 { continue; }

            // Serialize batch into SHM request region
            let req_offset = HEADER_SIZE;
            let mut offset = req_offset;
            for item in &batch {
                let data = serialize_sample(&item.request, self.cls_dim, self.h_dim, self.agg_dim);
                self.mmap[offset..offset + data.len()].copy_from_slice(&data);
                offset += self.sample_size; // stride by sample_size for alignment
            }

            // Write batch_size, then signal request_ready
            self.mmap[OFF_BATCH_SIZE..OFF_BATCH_SIZE+4]
                .copy_from_slice(&(batch_size as u32).to_le_bytes());
            std::sync::atomic::fence(Ordering::Release);
            self.write_flag(1);

            // Spin-poll for response
            let timeout_at = std::time::Instant::now() + std::time::Duration::from_secs(30);
            let got_response = loop {
                if self.read_flag() == 2 { break true; }
                if std::time::Instant::now() > timeout_at {
                    eprintln!("GPU SHM timeout");
                    break false;
                }
                std::hint::spin_loop();
            };

            if !got_response {
                for item in &batch {
                    let _ = item.response_tx.send(dummy.clone());
                }
                self.write_flag(0);
                continue;
            }

            std::sync::atomic::fence(Ordering::Acquire);

            // Read responses
            let resp_base = HEADER_SIZE + self.max_batch_size * self.sample_size;
            for (i, item) in batch.iter().enumerate() {
                let off = resp_base + i * self.response_sample_size;
                let resp = &self.mmap[off..off + self.response_sample_size];
                let hidden_state_out = if self.h_dim > 0 {
                    let h_off = 24; // hidden state starts after the 24-byte fixed fields
                    (0..self.h_dim)
                        .map(|j| f32::from_le_bytes(resp[h_off + j*4..h_off + j*4 + 4].try_into().unwrap()))
                        .collect()
                } else {
                    Vec::new()
                };
                let _ = item.response_tx.send(InferenceResult {
                    move_dx: f32::from_le_bytes(resp[0..4].try_into().unwrap()),
                    move_dy: f32::from_le_bytes(resp[4..8].try_into().unwrap()),
                    combat_type: resp[8],
                    target_idx: u16::from_le_bytes([resp[10], resp[11]]),
                    lp_move: f32::from_le_bytes(resp[12..16].try_into().unwrap()),
                    lp_combat: f32::from_le_bytes(resp[16..20].try_into().unwrap()),
                    lp_pointer: f32::from_le_bytes(resp[20..24].try_into().unwrap()),
                    hidden_state_out,
                });
            }

            self.write_flag(0);

            // Signal all waiting threads that a batch completed
            self.batch_epoch.fetch_add(1, Ordering::Release);
            self.batch_notify.notify_all();
        }
    }

    fn read_flag(&self) -> u32 {
        unsafe {
            let ptr = self.mmap.as_ptr().add(OFF_FLAG) as *const AtomicU32;
            (*ptr).load(Ordering::Acquire)
        }
    }

    fn write_flag(&mut self, val: u32) {
        unsafe {
            let ptr = self.mmap.as_mut_ptr().add(OFF_FLAG) as *mut AtomicU32;
            (*ptr).store(val, Ordering::Release);
        }
    }
}

fn serialize_sample(req: &InferenceRequest, cls_dim: usize, h_dim: usize, agg_dim: usize) -> Vec<u8> {
    let ent_mask_padded = (MAX_ENTITIES + 3) & !3;
    let thr_mask_padded = (MAX_THREATS + 3) & !3;
    let pos_mask_padded = (MAX_POSITIONS + 3) & !3;
    let zone_mask_padded = (MAX_ZONES + 3) & !3;
    let sample_size = 12 + MAX_ENTITIES * ENTITY_DIM * 4 + MAX_ENTITIES * 4 + ent_mask_padded
        + MAX_THREATS * THREAT_DIM * 4 + thr_mask_padded
        + MAX_POSITIONS * POSITION_DIM * 4 + pos_mask_padded
        + MAX_ZONES * ZONE_DIM * 4 + zone_mask_padded
        + 12 + MAX_ABILITIES + MAX_ABILITIES * cls_dim * 4;

    let mut data = Vec::with_capacity(sample_size);

    let n_ent = req.entities.len().min(MAX_ENTITIES) as u16;
    let n_thr = req.threats.len().min(MAX_THREATS) as u16;
    let n_pos = req.positions.len().min(MAX_POSITIONS) as u16;
    let n_zones = req.zones.len().min(MAX_ZONES) as u16;
    data.extend_from_slice(&n_ent.to_le_bytes());
    data.extend_from_slice(&n_thr.to_le_bytes());
    data.extend_from_slice(&n_pos.to_le_bytes());
    data.extend_from_slice(&n_zones.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // padding
    data.extend_from_slice(&0u16.to_le_bytes()); // pad to 12 bytes (4-byte aligned)

    for i in 0..MAX_ENTITIES {
        if i < req.entities.len() {
            for j in 0..ENTITY_DIM {
                let v = if j < req.entities[i].len() { req.entities[i][j] } else { 0.0 };
                data.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            data.extend_from_slice(&[0u8; ENTITY_DIM * 4]);
        }
    }

    for i in 0..MAX_ENTITIES {
        let t = if i < req.entity_types.len() { req.entity_types[i] as i32 } else { 0 };
        data.extend_from_slice(&t.to_le_bytes());
    }

    for i in 0..ent_mask_padded {
        if i < MAX_ENTITIES { data.push(if i >= req.entities.len() { 1 } else { 0 }); }
        else { data.push(1); }
    }

    for i in 0..MAX_THREATS {
        if i < req.threats.len() {
            for j in 0..THREAT_DIM {
                let v = if j < req.threats[i].len() { req.threats[i][j] } else { 0.0 };
                data.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            data.extend_from_slice(&[0u8; THREAT_DIM * 4]);
        }
    }

    for i in 0..thr_mask_padded {
        if i < MAX_THREATS { data.push(if i >= req.threats.len() { 1 } else { 0 }); }
        else { data.push(1); }
    }

    for i in 0..MAX_POSITIONS {
        if i < req.positions.len() {
            for j in 0..POSITION_DIM {
                let v = if j < req.positions[i].len() { req.positions[i][j] } else { 0.0 };
                data.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            data.extend_from_slice(&[0u8; POSITION_DIM * 4]);
        }
    }

    for i in 0..pos_mask_padded {
        if i < MAX_POSITIONS { data.push(if i >= req.positions.len() { 1 } else { 0 }); }
        else { data.push(1); }
    }

    // Zone features
    for i in 0..MAX_ZONES {
        if i < req.zones.len() {
            for j in 0..ZONE_DIM {
                let v = if j < req.zones[i].len() { req.zones[i][j] } else { 0.0 };
                data.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            data.extend_from_slice(&[0u8; ZONE_DIM * 4]);
        }
    }

    // Zone mask
    for i in 0..zone_mask_padded {
        if i < MAX_ZONES { data.push(if i >= req.zones.len() { 1 } else { 0 }); }
        else { data.push(1); }
    }

    for i in 0..NUM_COMBAT_TYPES {
        data.push(if i < req.combat_mask.len() && req.combat_mask[i] { 1 } else { 0 });
    }
    data.push(0);
    data.push(0);

    for i in 0..MAX_ABILITIES {
        data.push(if i < req.ability_cls.len() && req.ability_cls[i].is_some() { 1 } else { 0 });
    }
    for i in 0..MAX_ABILITIES {
        if i < req.ability_cls.len() {
            if let Some(ref cls) = req.ability_cls[i] {
                for j in 0..cls_dim {
                    let v = if j < cls.len() { cls[j] } else { 0.0 };
                    data.extend_from_slice(&v.to_le_bytes());
                }
            } else {
                data.extend_from_slice(&vec![0u8; cls_dim * 4]);
            }
        } else {
            data.extend_from_slice(&vec![0u8; cls_dim * 4]);
        }
    }

    // Append GRU hidden state (h_dim floats, or nothing if h_dim=0)
    if h_dim > 0 {
        for j in 0..h_dim {
            let v = if j < req.hidden_state.len() { req.hidden_state[j] } else { 0.0 };
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Append aggregate features (agg_dim floats, or nothing if agg_dim=0)
    if agg_dim > 0 {
        for j in 0..agg_dim {
            let v = if j < req.aggregate_features.len() { req.aggregate_features[j] } else { 0.0 };
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    data
}
