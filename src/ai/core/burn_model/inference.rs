//! In-process GPU inference client using Burn/LibTorch.
//!
//! Double-buffered design: sim threads submit requests into buffer A while
//! the GPU processes buffer B. When the GPU finishes, it swaps buffers
//! immediately — no idle time on either side.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

use burn::prelude::*;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;

use super::actor_critic::ActorCriticV5;
use super::config::*;

// Inference types (previously in gpu_client, now canonical here)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InferenceToken(pub u64);

pub trait InferenceClient: Send + Sync {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String>;
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String>;
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String>;
    fn batch_epoch(&self) -> u64;
    fn wait_for_batch(&self, since: u64);
    fn h_dim(&self) -> usize;
}

#[derive(Clone)]
pub struct InferenceRequest {
    pub entities: Vec<Vec<f32>>,
    pub entity_types: Vec<u8>,
    pub zones: Vec<Vec<f32>>,
    pub combat_mask: Vec<bool>,
    pub ability_cls: Vec<Option<Vec<f32>>>,
    pub hidden_state: Vec<f32>,
    pub aggregate_features: Vec<f32>,
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

type B = LibTorch;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct BatchItem {
    request: InferenceRequest,
    response_tx: Sender<InferenceResult>,
}

thread_local! {
    static PENDING: RefCell<HashMap<InferenceToken, Receiver<InferenceResult>>> =
        RefCell::new(HashMap::new());
}

struct BatchNotify {
    mutex: std::sync::Mutex<()>,
    condvar: std::sync::Condvar,
}

impl BatchNotify {
    fn new() -> Self {
        Self { mutex: std::sync::Mutex::new(()), condvar: std::sync::Condvar::new() }
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

// ---------------------------------------------------------------------------
// Public client
// ---------------------------------------------------------------------------

pub struct BurnInferenceClient {
    request_tx: Sender<BatchItem>,
    next_token: AtomicU64,
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
    h_dim: usize,
}

impl BurnInferenceClient {
    /// Create a new double-buffered inference client.
    pub fn new(
        model: ActorCriticV5<B>,
        device: LibTorchDevice,
        max_batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Arc<Self> {
        let h_dim = model.temporal_cell.h_dim();
        // Large channel: sim threads can keep submitting while GPU works
        let (request_tx, request_rx) = bounded::<BatchItem>(max_batch_size * 4);
        let batch_epoch = Arc::new(AtomicU64::new(0));
        let batch_notify = Arc::new(BatchNotify::new());

        let batcher = DoubleBufferBatcher {
            model,
            device,
            request_rx,
            max_batch_size,
            batch_timeout_ms,
            batch_epoch: batch_epoch.clone(),
            batch_notify: batch_notify.clone(),
        };
        std::thread::spawn(move || batcher.run());

        Arc::new(Self {
            request_tx,
            next_token: AtomicU64::new(0),
            batch_epoch,
            batch_notify,
            h_dim,
        })
    }

    /// Blocking inference.
    pub fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> {
        let (tx, rx) = bounded(1);
        self.request_tx.send(BatchItem { request, response_tx: tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        rx.recv().map_err(|_| "No response from batcher".to_string())
    }

    /// Non-blocking submit.
    pub fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> {
        let token = InferenceToken(self.next_token.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = bounded(1);
        self.request_tx.send(BatchItem { request, response_tx: tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        PENDING.with(|p| p.borrow_mut().insert(token, rx));
        Ok(token)
    }

    /// Poll for result.
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

    pub fn batch_epoch(&self) -> u64 {
        self.batch_epoch.load(Ordering::Acquire)
    }

    pub fn wait_for_batch(&self, since: u64) {
        if self.batch_epoch.load(Ordering::Acquire) != since {
            return;
        }
        self.batch_notify.wait_timeout(std::time::Duration::from_millis(10));
    }

    pub fn h_dim(&self) -> usize {
        self.h_dim
    }
}

impl InferenceClient for BurnInferenceClient {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> { self.infer(request) }
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> { self.submit(request) }
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String> { self.try_recv(token) }
    fn batch_epoch(&self) -> u64 { self.batch_epoch() }
    fn wait_for_batch(&self, since: u64) { self.wait_for_batch(since) }
    fn h_dim(&self) -> usize { self.h_dim() }
}

// ---------------------------------------------------------------------------
// Double-buffered batcher: GPU never waits for requests, sims never wait for GPU
// ---------------------------------------------------------------------------

/// Max entity tokens per sample (self + allies + enemies + zones + agg).
const MAX_ENTITY_TOKENS: usize = 16;
/// Max zone tokens per sample.
const MAX_ZONE_TOKENS: usize = 12;

struct DoubleBufferBatcher {
    model: ActorCriticV5<B>,
    device: LibTorchDevice,
    request_rx: Receiver<BatchItem>,
    max_batch_size: usize,
    batch_timeout_ms: u64,
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
}

/// Preallocated packing buffers — reused across batches to avoid allocation.
struct PackBuffers {
    ent_data: Vec<f32>,
    ent_type_data: Vec<i64>,
    ent_mask_data: Vec<bool>,
    zone_data: Vec<f32>,
    zone_mask_data: Vec<bool>,
    agg_data: Vec<f32>,
    combat_mask_data: Vec<bool>,
    h_prev_data: Vec<f32>,
}

impl PackBuffers {
    fn new(max_batch: usize, h_dim: usize) -> Self {
        Self {
            ent_data: vec![0.0f32; max_batch * MAX_ENTITY_TOKENS * ENTITY_DIM],
            ent_type_data: vec![0i64; max_batch * MAX_ENTITY_TOKENS],
            ent_mask_data: vec![false; max_batch * MAX_ENTITY_TOKENS],
            zone_data: vec![0.0f32; max_batch * MAX_ZONE_TOKENS * ZONE_DIM],
            zone_mask_data: vec![false; max_batch * MAX_ZONE_TOKENS],
            agg_data: vec![0.0f32; max_batch * AGG_DIM],
            combat_mask_data: vec![false; max_batch * NUM_COMBAT_TYPES],
            h_prev_data: vec![0.0f32; max_batch * h_dim],
        }
    }

    /// Zero all buffers up to `n` samples with given dimensions.
    fn clear(&mut self, n: usize, max_ent: usize, max_zones: usize, h_dim: usize) {
        self.ent_data[..n * max_ent * ENTITY_DIM].fill(0.0);
        self.ent_type_data[..n * max_ent].fill(0);
        self.ent_mask_data[..n * max_ent].fill(false);
        self.zone_data[..n * max_zones * ZONE_DIM].fill(0.0);
        self.zone_mask_data[..n * max_zones].fill(false);
        self.agg_data[..n * AGG_DIM].fill(0.0);
        self.combat_mask_data[..n * NUM_COMBAT_TYPES].fill(false);
        self.h_prev_data[..n * h_dim].fill(0.0);
    }
}

impl DoubleBufferBatcher {
    fn run(self) {
        let mut buffer_a: Vec<BatchItem> = Vec::with_capacity(self.max_batch_size);
        let mut buffer_b: Vec<BatchItem> = Vec::with_capacity(self.max_batch_size);
        let h_dim = self.model.temporal_cell.h_dim();
        let mut pack = PackBuffers::new(self.max_batch_size, h_dim);

        // Fill initial batch (blocking wait for first request)
        match self.request_rx.recv() {
            Ok(item) => buffer_a.push(item),
            Err(_) => return,
        }
        self.drain_into(&mut buffer_a);

        let mut batch_count = 0u64;
        let mut total_forward_ns = 0u128;
        let mut total_pack_ns = 0u128;
        let mut total_fwd_ns = 0u128;
        let mut total_unpack_ns = 0u128;
        let mut total_items = 0u64;
        let mut max_batch = 0usize;
        let run_start = std::time::Instant::now();

        loop {
            // Process buffer_a on GPU
            let t0 = std::time::Instant::now();
            let (results, p_ns, f_ns, u_ns) = self.forward_batch(&buffer_a, &mut pack);
            total_forward_ns += t0.elapsed().as_nanos();
            total_pack_ns += p_ns;
            total_fwd_ns += f_ns;
            total_unpack_ns += u_ns;
            total_items += buffer_a.len() as u64;
            if buffer_a.len() > max_batch { max_batch = buffer_a.len(); }
            batch_count += 1;

            if batch_count % 500 == 0 {
                let avg_ms = total_forward_ns as f64 / batch_count as f64 / 1e6;
                let avg_size = total_items as f64 / batch_count as f64;
                let throughput = total_items as f64 / run_start.elapsed().as_secs_f64();
                let pack_ms = total_pack_ns as f64 / batch_count as f64 / 1e6;
                let fwd_ms = total_fwd_ns as f64 / batch_count as f64 / 1e6;
                let unpack_ms = total_unpack_ns as f64 / batch_count as f64 / 1e6;
                eprintln!("[burn] {} batches, avg {:.1}ms/batch (pack={:.1} fwd={:.1} unpack={:.1}), size {:.0}, {:.0} items/s",
                    batch_count, avg_ms, pack_ms, fwd_ms, unpack_ms, avg_size, throughput);
            }

            // While dispatching results, drain channel into buffer_b
            // (sim threads that got results will immediately extract + submit)
            for (item, result) in buffer_a.drain(..).zip(results) {
                let _ = item.response_tx.send(result);
            }
            self.batch_epoch.fetch_add(1, Ordering::Release);
            self.batch_notify.notify_all();

            // Drain any requests that arrived during GPU execution into buffer_b
            self.drain_into(&mut buffer_b);

            if buffer_b.is_empty() {
                // No requests arrived during GPU work — wait for next
                match self.request_rx.recv() {
                    Ok(item) => buffer_b.push(item),
                    Err(_) => break,
                }
                self.drain_into(&mut buffer_b);
            }

            // Swap: buffer_b becomes the active batch, buffer_a is the drain target
            std::mem::swap(&mut buffer_a, &mut buffer_b);
        }

        if batch_count > 0 {
            let avg_ms = total_forward_ns as f64 / batch_count as f64 / 1e6;
            let avg_size = total_items as f64 / batch_count as f64;
            let throughput = total_items as f64 / run_start.elapsed().as_secs_f64();
            eprintln!("[burn] FINAL: {} batches, avg {:.1}ms/batch, avg size {:.0}, max {}, {:.0} items/s",
                batch_count, avg_ms, avg_size, max_batch, throughput);
        }
    }

    /// Drain all pending requests from channel (non-blocking), up to max_batch_size.
    fn drain_into(&self, buffer: &mut Vec<BatchItem>) {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(self.batch_timeout_ms);
        while buffer.len() < self.max_batch_size {
            match self.request_rx.recv_deadline(deadline) {
                Ok(item) => buffer.push(item),
                Err(_) => break,
            }
        }
    }

    fn forward_batch(&self, batch: &[BatchItem], _pack: &mut PackBuffers) -> (Vec<InferenceResult>, u128, u128, u128) {
        let bs = batch.len();
        if bs == 0 { return (Vec::new(), 0, 0, 0); }
        let dev = &self.device;
        let t_pack = std::time::Instant::now();

        // Find max sequence lengths for padding
        let max_ent = batch.iter().map(|b| b.request.entities.len()).max().unwrap_or(1).max(1);
        let max_zones = batch.iter().map(|b| b.request.zones.len()).max().unwrap_or(1).max(1);
        let h_dim = self.model.temporal_cell.h_dim();

        // Build flat f32 buffers — types and masks encoded as f32 to avoid TensorData allocs
        let n_ent = bs * max_ent;
        let n_zones = bs * max_zones;
        let mut ent_data = vec![0.0f32; n_ent * ENTITY_DIM];
        let mut etype_f32 = vec![0.0f32; n_ent];
        let mut emask_f32 = vec![0.0f32; n_ent];
        let mut zone_data = vec![0.0f32; n_zones * ZONE_DIM];
        let mut zmask_f32 = vec![0.0f32; n_zones];
        let mut agg_data = vec![0.0f32; bs * AGG_DIM];
        let mut h_prev_data = vec![0.0f32; bs * h_dim];

        for (i, item) in batch.iter().enumerate() {
            let req = &item.request;

            for (j, ent) in req.entities.iter().enumerate().take(max_ent) {
                let base = (i * max_ent + j) * ENTITY_DIM;
                let len = ent.len().min(ENTITY_DIM);
                ent_data[base..base + len].copy_from_slice(&ent[..len]);
                etype_f32[i * max_ent + j] = req.entity_types.get(j).copied().unwrap_or(0) as f32;
                emask_f32[i * max_ent + j] = 1.0;
            }

            for (j, zone) in req.zones.iter().enumerate().take(max_zones) {
                let base = (i * max_zones + j) * ZONE_DIM;
                let len = zone.len().min(ZONE_DIM);
                zone_data[base..base + len].copy_from_slice(&zone[..len]);
                zmask_f32[i * max_zones + j] = 1.0;
            }

            let alen = req.aggregate_features.len().min(AGG_DIM);
            if alen > 0 {
                agg_data[i * AGG_DIM..i * AGG_DIM + alen]
                    .copy_from_slice(&req.aggregate_features[..alen]);
            }

            let hlen = req.hidden_state.len().min(h_dim);
            if hlen > 0 {
                h_prev_data[i * h_dim..i * h_dim + hlen]
                    .copy_from_slice(&req.hidden_state[..hlen]);
            }
        }

        // All from_floats — no TensorData::new, no .to_vec()
        let ent_feat = Tensor::<B, 1>::from_floats(ent_data.as_slice(), dev)
            .reshape([bs, max_ent, ENTITY_DIM]);
        let ent_types = Tensor::<B, 1>::from_floats(etype_f32.as_slice(), dev)
            .reshape([bs, max_ent]).int();
        let ent_mask = Tensor::<B, 1>::from_floats(emask_f32.as_slice(), dev)
            .reshape([bs, max_ent]).greater_elem(0.5);
        let zone_feat = Tensor::<B, 1>::from_floats(zone_data.as_slice(), dev)
            .reshape([bs, max_zones, ZONE_DIM]);
        let zone_mask = Tensor::<B, 1>::from_floats(zmask_f32.as_slice(), dev)
            .reshape([bs, max_zones]).greater_elem(0.5);
        let agg_feat = Tensor::<B, 1>::from_floats(agg_data.as_slice(), dev)
            .reshape([bs, AGG_DIM]);
        let h_prev = Tensor::<B, 1>::from_floats(h_prev_data.as_slice(), dev)
            .reshape([bs, h_dim]);

        // Pack ability CLS embeddings: per-ability [bs, cls_dim] tensors
        let ability_cls: Vec<Option<Tensor<B, 2>>> = {
            let cls_dim = batch.iter()
                .flat_map(|b| b.request.ability_cls.iter())
                .find_map(|opt| opt.as_ref().map(|v| v.len()))
                .unwrap_or(0);

            if cls_dim == 0 {
                vec![None; MAX_ABILITIES]
            } else {
                (0..MAX_ABILITIES).map(|ab_idx| {
                    let has_any = batch.iter().any(|b| {
                        b.request.ability_cls.get(ab_idx)
                            .and_then(|opt| opt.as_ref())
                            .is_some()
                    });
                    if !has_any {
                        return None;
                    }
                    let mut data = vec![0.0f32; bs * cls_dim];
                    for (i, item) in batch.iter().enumerate() {
                        if let Some(Some(cls)) = item.request.ability_cls.get(ab_idx) {
                            let len = cls.len().min(cls_dim);
                            data[i * cls_dim..i * cls_dim + len].copy_from_slice(&cls[..len]);
                        }
                    }
                    Some(Tensor::<B, 1>::from_floats(data.as_slice(), dev)
                        .reshape([bs, cls_dim]))
                }).collect()
            }
        };
        let pack_ns = t_pack.elapsed().as_nanos();

        // Forward pass
        let t_fwd = std::time::Instant::now();
        let (output, h_new) = self.model.forward(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg_feat), Some(h_prev),
        );

        let n_tokens = output.combat.attack_ptr.dims()[1];
        let fwd_ns = t_fwd.elapsed().as_nanos();

        // Extract results — single GPU→CPU transfer
        let t_unpack = std::time::Instant::now();
        let pos_2d = output.target_pos;              // [bs, 2]
        let combat_2d = output.combat.combat_logits; // [bs, 10]
        let ptr_2d = output.combat.attack_ptr;       // [bs, n_tokens]
        let h_2d = h_new;                            // [bs, h_dim]

        let combined = Tensor::cat(vec![pos_2d, combat_2d, ptr_2d, h_2d], 1);
        let combined_data = combined.to_data();
        let all_f: &[f32] = combined_data.as_slice().unwrap();
        let row_len = 2 + NUM_COMBAT_TYPES + n_tokens + h_dim;

        // Offsets within each row
        let o_pos = 0;
        let o_combat = 2;
        let o_ptr = o_combat + NUM_COMBAT_TYPES;
        let o_h = o_ptr + n_tokens;

        let mut results = Vec::with_capacity(bs);
        for i in 0..bs {
            let row = &all_f[i * row_len..(i + 1) * row_len];
            let req = &batch[i].request;

            let tx = row[o_pos] * 20.0;
            let ty = row[o_pos + 1] * 20.0;

            let mut best_ct = 1usize;
            let mut best_score = f32::NEG_INFINITY;
            for j in 0..NUM_COMBAT_TYPES {
                if req.combat_mask.get(j).copied().unwrap_or(false) {
                    let score = row[o_combat + j];
                    if score > best_score { best_score = score; best_ct = j; }
                }
            }

            let mut best_ti = 0usize;
            let mut best_ptr = f32::NEG_INFINITY;
            for j in 0..n_tokens {
                let score = row[o_ptr + j];
                if score > best_ptr { best_ptr = score; best_ti = j; }
            }

            let h_out = row[o_h..o_h + h_dim].to_vec();

            results.push(InferenceResult {
                move_dx: tx, move_dy: ty,
                combat_type: best_ct as u8, target_idx: best_ti as u16,
                lp_move: 0.0, lp_combat: 0.0, lp_pointer: 0.0,
                hidden_state_out: h_out,
            });
        }

        (results, pack_ns, fwd_ns, t_unpack.elapsed().as_nanos())
    }
}
