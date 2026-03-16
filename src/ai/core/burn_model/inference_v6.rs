//! V6 GPU inference client using Burn/LibTorch.
//!
//! Same double-buffered design as V5, but uses ActorCriticV6 with
//! spatial cross-attention and latent interface.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

use burn::prelude::*;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;

use super::actor_critic_v6::ActorCriticV6;
use super::config::*;
use super::spatial_cross_attn::CORNER_DIM;

pub use crate::ai::core::ability_transformer::gpu_client::{
    InferenceToken, InferenceClient, InferenceRequest, InferenceResult,
};

type B = LibTorch;

struct BatchItem {
    request: InferenceRequest,
    response_tx: Sender<InferenceResult>,
}

thread_local! {
    static PENDING_V6: RefCell<HashMap<InferenceToken, Receiver<InferenceResult>>> =
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

pub struct BurnInferenceClientV6 {
    request_tx: Sender<BatchItem>,
    next_token: AtomicU64,
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
    h_dim: usize,
}

impl BurnInferenceClientV6 {
    pub fn new(
        model: ActorCriticV6<B>,
        device: LibTorchDevice,
        max_batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Arc<Self> {
        let h_dim = model.temporal_cell.h_dim();
        let (request_tx, request_rx) = bounded::<BatchItem>(max_batch_size * 4);
        let batch_epoch = Arc::new(AtomicU64::new(0));
        let batch_notify = Arc::new(BatchNotify::new());

        let batcher = DoubleBufferBatcherV6 {
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

    pub fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> {
        let (tx, rx) = bounded(1);
        self.request_tx.send(BatchItem { request, response_tx: tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        rx.recv().map_err(|_| "No response from batcher".to_string())
    }

    pub fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> {
        let token = InferenceToken(self.next_token.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = bounded(1);
        self.request_tx.send(BatchItem { request, response_tx: tx })
            .map_err(|_| "Batcher thread died".to_string())?;
        PENDING_V6.with(|p| p.borrow_mut().insert(token, rx));
        Ok(token)
    }

    pub fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String> {
        PENDING_V6.with(|p| {
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

impl InferenceClient for BurnInferenceClientV6 {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String> { self.infer(request) }
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String> { self.submit(request) }
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String> { self.try_recv(token) }
    fn batch_epoch(&self) -> u64 { self.batch_epoch() }
    fn wait_for_batch(&self, since: u64) { self.wait_for_batch(since) }
    fn h_dim(&self) -> usize { self.h_dim() }
}

// ---------------------------------------------------------------------------
// Double-buffered batcher for V6
// ---------------------------------------------------------------------------

struct DoubleBufferBatcherV6 {
    model: ActorCriticV6<B>,
    device: LibTorchDevice,
    request_rx: Receiver<BatchItem>,
    max_batch_size: usize,
    batch_timeout_ms: u64,
    batch_epoch: Arc<AtomicU64>,
    batch_notify: Arc<BatchNotify>,
}

impl DoubleBufferBatcherV6 {
    fn run(self) {
        let mut buffer_a: Vec<BatchItem> = Vec::with_capacity(self.max_batch_size);
        let mut buffer_b: Vec<BatchItem> = Vec::with_capacity(self.max_batch_size);

        match self.request_rx.recv() {
            Ok(item) => buffer_a.push(item),
            Err(_) => return,
        }
        self.drain_into(&mut buffer_a);

        let mut batch_count = 0u64;
        let mut total_forward_ns = 0u128;
        let mut total_items = 0u64;
        let mut max_batch = 0usize;
        let run_start = std::time::Instant::now();

        loop {
            let t0 = std::time::Instant::now();
            let results = self.forward_batch(&buffer_a);
            total_forward_ns += t0.elapsed().as_nanos();
            total_items += buffer_a.len() as u64;
            if buffer_a.len() > max_batch { max_batch = buffer_a.len(); }
            batch_count += 1;

            if batch_count % 500 == 0 {
                let avg_ms = total_forward_ns as f64 / batch_count as f64 / 1e6;
                let avg_size = total_items as f64 / batch_count as f64;
                let throughput = total_items as f64 / run_start.elapsed().as_secs_f64();
                eprintln!("[burn-v6] {} batches, avg {:.1}ms/batch, size {:.0}, {:.0} items/s",
                    batch_count, avg_ms, avg_size, throughput);
            }

            for (item, result) in buffer_a.drain(..).zip(results) {
                let _ = item.response_tx.send(result);
            }
            self.batch_epoch.fetch_add(1, Ordering::Release);
            self.batch_notify.notify_all();

            self.drain_into(&mut buffer_b);
            if buffer_b.is_empty() {
                match self.request_rx.recv() {
                    Ok(item) => buffer_b.push(item),
                    Err(_) => break,
                }
                self.drain_into(&mut buffer_b);
            }
            std::mem::swap(&mut buffer_a, &mut buffer_b);
        }

        if batch_count > 0 {
            let avg_ms = total_forward_ns as f64 / batch_count as f64 / 1e6;
            let throughput = total_items as f64 / run_start.elapsed().as_secs_f64();
            eprintln!("[burn-v6] FINAL: {} batches, avg {:.1}ms/batch, max {}, {:.0} items/s",
                batch_count, avg_ms, max_batch, throughput);
        }
    }

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

    fn forward_batch(&self, batch: &[BatchItem]) -> Vec<InferenceResult> {
        let bs = batch.len();
        if bs == 0 { return Vec::new(); }
        let dev = &self.device;
        let h_dim = self.model.temporal_cell.h_dim();

        // Find max sequence lengths
        let max_ent = batch.iter().map(|b| b.request.entities.len()).max().unwrap_or(1).max(1);
        let max_zones = batch.iter().map(|b| b.request.zones.len()).max().unwrap_or(1).max(1);
        let max_corners = batch.iter().map(|b| b.request.corner_tokens.len()).max().unwrap_or(0);
        let has_corners = max_corners > 0;
        let max_corners = max_corners.max(1); // At least 1 for tensor shape

        // Allocate flat buffers
        let n_ent = bs * max_ent;
        let n_zones = bs * max_zones;
        let n_corners = bs * max_corners;
        let mut ent_data = vec![0.0f32; n_ent * ENTITY_DIM];
        let mut etype_f32 = vec![0.0f32; n_ent];
        let mut emask_f32 = vec![0.0f32; n_ent];
        let mut zone_data = vec![0.0f32; n_zones * ZONE_DIM];
        let mut zmask_f32 = vec![0.0f32; n_zones];
        let mut agg_data = vec![0.0f32; bs * AGG_DIM];
        let mut h_prev_data = vec![0.0f32; bs * h_dim];
        let mut corner_data = vec![0.0f32; n_corners * CORNER_DIM];
        let mut cmask_f32 = vec![0.0f32; n_corners];

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

            for (j, corner) in req.corner_tokens.iter().enumerate().take(max_corners) {
                let base = (i * max_corners + j) * CORNER_DIM;
                let len = corner.len().min(CORNER_DIM);
                corner_data[base..base + len].copy_from_slice(&corner[..len]);
                cmask_f32[i * max_corners + j] = 1.0;
            }
        }

        // Build tensors
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

        let (corner_tokens, corner_mask) = if has_corners {
            let ct = Tensor::<B, 1>::from_floats(corner_data.as_slice(), dev)
                .reshape([bs, max_corners, CORNER_DIM]);
            let cm = Tensor::<B, 1>::from_floats(cmask_f32.as_slice(), dev)
                .reshape([bs, max_corners]).greater_elem(0.5);
            (Some(ct), Some(cm))
        } else {
            (None, None)
        };

        // Pack ability CLS embeddings: per-ability [bs, cls_dim] tensors
        let ability_cls: Vec<Option<Tensor<B, 2>>> = {
            // Detect CLS dimension from first non-None embedding
            let cls_dim = batch.iter()
                .flat_map(|b| b.request.ability_cls.iter())
                .find_map(|opt| opt.as_ref().map(|v| v.len()))
                .unwrap_or(0);

            if cls_dim == 0 {
                vec![None; MAX_ABILITIES]
            } else {
                (0..MAX_ABILITIES).map(|ab_idx| {
                    // Check if any sample in this batch has this ability
                    let has_any = batch.iter().any(|b| {
                        b.request.ability_cls.get(ab_idx)
                            .and_then(|opt| opt.as_ref())
                            .is_some()
                    });
                    if !has_any {
                        return None;
                    }
                    // Build [bs, cls_dim] tensor for this ability slot
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

        // Forward pass (inference mode — no value head)
        let (output, h_new) = self.model.forward_inference(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            &ability_cls, Some(agg_feat), Some(h_prev),
            corner_tokens, corner_mask, None,
        );

        let n_tokens = output.combat.attack_ptr.dims()[1];

        // Extract results — single GPU→CPU transfer
        let pos_2d = output.target_pos;
        let combat_2d = output.combat.combat_logits;
        let ptr_2d = output.combat.attack_ptr;
        let h_2d = h_new;

        let combined = Tensor::cat(vec![pos_2d, combat_2d, ptr_2d, h_2d], 1);
        let combined_data = combined.to_data();
        let all_f: &[f32] = combined_data.as_slice().unwrap();
        let row_len = 2 + NUM_COMBAT_TYPES + n_tokens + h_dim;

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

        results
    }
}
