# Killing the Python Sidecar: Migrating GPU Inference from Shared Memory to Burn

*How we deleted 4,667 lines of cross-language serialization code by defining the model once in Rust — and got 1.9x faster in the process.*

---

## The Problem

Our IMPALA training loop runs 4,096 parallel combat simulations across 64 threads. Every sim tick, every hero needs a neural network forward pass to pick actions. That inference happens on a GPU.

For six months, the GPU lived in a separate Python process. Rust sims serialized game state into a memory-mapped file, set a flag, spin-polled for the response, and deserialized the result. Python read the same file, ran PyTorch, wrote back. It worked. It was fast enough. And every time we changed the model, we wanted to set it on fire.

This is the story of replacing that entire stack with Burn — a Rust ML framework that let us define the model once and run it on both CPU (for tests) and CUDA (for training), with no Python process, no shared memory file, and no byte-level serialization.

---

## The Old Architecture: Shared Memory Protocol

The bridge between Rust and Python was a memory-mapped file at `/dev/shm/impala_inf`. The layout was, charitably, hand-rolled:

```
┌────────────────────────────────────────────────────────────────┐
│                    512-byte Global Header                       │
│                                                                 │
│  [0x00] magic: u32 = 0x47505549 ("GPUI")                      │
│  [0x04] version: u32 = 1                                       │
│  [0x08] cls_dim: u32 (ability embedding dimension)             │
│  [0x0C] max_batch_size: u32                                    │
│  [0x10] sample_size: u32 (bytes per request)                   │
│  [0x14] response_sample_size: u32                              │
│  [0x18] h_dim: u32 (hidden state dimension)                    │
│  [0x1C] agg_dim: u32 (aggregate feature dimension)             │
│  ...                                                            │
│  [0x40] flag: u32 (0=idle, 1=request_ready, 2=response_ready) │
│  [0x44] batch_size: u32                                        │
│  [0x80] reload_path: 256 bytes (null-terminated)               │
│  [0x180] reload_request: u32                                   │
│  [0x184] reload_ack: u32                                       │
│                                                                 │
│  [512..] Request region: max_batch × sample_size               │
│  [512 + req_region..] Response region: max_batch × resp_size   │
└────────────────────────────────────────────────────────────────┘
```

Each request sample was a densely packed binary blob. The serialization function in Rust (`serialize_sample`, 138 lines) wrote fields in a specific order with manual padding:

```
Per-sample request layout (~7KB):
┌──────────────────────────────────────────────────────────────┐
│ 12-byte header: n_entities(u16), n_threats(u16),             │
│                 n_positions(u16), n_zones(u16), pad(4)       │
├──────────────────────────────────────────────────────────────┤
│ Entity features:    20 × 34 × f32 = 2,720 bytes             │
│ Entity type IDs:    20 × i32       =    80 bytes             │
│ Entity mask:        20 bytes (padded to 4-byte alignment)    │
├──────────────────────────────────────────────────────────────┤
│ Threat features:     6 × 10 × f32 =   240 bytes             │
│ Threat mask:         8 bytes (padded)                        │
├──────────────────────────────────────────────────────────────┤
│ Position features:   8 ×  8 × f32 =   256 bytes             │
│ Position mask:       8 bytes (padded)                        │
├──────────────────────────────────────────────────────────────┤
│ Zone features:      10 × 12 × f32 =   480 bytes             │
│ Zone mask:          12 bytes (padded)                        │
├──────────────────────────────────────────────────────────────┤
│ Combat mask:        10 bytes + 2 pad                         │
│ Ability presence:    8 bytes                                 │
│ Ability CLS:         8 × 128 × f32 = 4,096 bytes            │
├──────────────────────────────────────────────────────────────┤
│ Hidden state:       h_dim × f32                              │
│ Aggregate features: agg_dim × f32                            │
└──────────────────────────────────────────────────────────────┘
```

The response was tighter — 24 bytes of fixed fields plus the hidden state:

```
Per-sample response:
┌──────────┬──────────┬──────────────┬───────────┬──────────┬───────────┬─────────┐
│ move_dx  │ move_dy  │ combat_type  │ target_id │ lp_move  │ lp_combat │ lp_ptr  │
│ f32 LE   │ f32 LE   │ u8           │ u16 LE    │ f32 LE   │ f32 LE    │ f32 LE  │
└──────────┴──────────┴──────────────┴───────────┴──────────┴───────────┴─────────┘
│ hidden_state_out: h_dim × f32                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Flag-Based Synchronization

The communication protocol was primitive:

1. Rust batcher writes batch into the request region
2. Sets `flag = 1` (request ready)
3. Python GPU server polls `flag`, sees 1, reads batch, runs forward pass
4. Writes results into response region, sets `flag = 2`
5. Rust batcher polls `flag`, sees 2, reads responses
6. Sets `flag = 0` (idle)

Both sides spin-poll. The Rust batcher used `std::hint::spin_loop()` with a 30-second timeout. The Python server polled in a tight loop with `time.sleep(0.0001)` backoff.

### The Alignment Bug

When we added zone tokens to the model, each sample gained a 5-zone header: `n_zones(u16)` packed alongside the existing entity/threat/position counts. Five `u16` values = 10 bytes. But every subsequent float array expected 4-byte alignment.

10 bytes is not 4-byte aligned.

The Rust side added explicit 2-byte padding to bring the header to 12 bytes. But the Python side used numpy's `frombuffer` with computed offsets — and those offsets were wrong by 2 bytes. Every float array after the header was misaligned, causing numpy stride errors that only manifested with certain batch sizes where the misalignment didn't happen to land on a 4-byte boundary anyway.

This took the better part of a day to track down, and the fix was adding `pad(2)` in two places. The kind of bug that only exists because two languages are independently computing byte offsets into the same memory region.

---

## The Crossbeam Batching Pattern

The SHM protocol was the transport layer. Above it sat a batching system that coordinated 4,096 concurrent simulations:

```
┌─────────────────────────────────────────────────────────────┐
│                  64 Rayon Threads                             │
│                                                              │
│  Thread 0: sim[0], sim[1], ..., sim[63]                     │
│  Thread 1: sim[64], sim[65], ..., sim[127]                  │
│  ...                                                         │
│  Thread 63: sim[4032], sim[4033], ..., sim[4095]            │
│                                                              │
│  Each sim submits InferenceRequest via crossbeam channel     │
│  Gets back InferenceToken for non-blocking polling           │
└────────────────────────┬────────────────────────────────────┘
                         │ bounded(max_batch * 32)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Batcher Thread                             │
│                                                              │
│  loop {                                                      │
│    // Block until first request arrives                      │
│    batch.push(request_rx.recv());                           │
│                                                              │
│    // Drain more with 1ms deadline                          │
│    while batch.len() < max_batch {                          │
│      match request_rx.recv_deadline(deadline) {             │
│        Ok(item) => batch.push(item),                        │
│        Err(Timeout) => break,                               │
│      }                                                       │
│    }                                                         │
│                                                              │
│    // Forward pass (SHM or Burn)                             │
│    results = transport.forward(batch);                       │
│                                                              │
│    // Dispatch via per-request oneshot channels              │
│    for (item, result) in zip(batch, results) {              │
│      item.response_tx.send(result);                         │
│    }                                                         │
│                                                              │
│    // Wake parked sim threads                                │
│    batch_epoch.fetch_add(1);                                │
│    condvar.notify_all();                                     │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

The key patterns:

**Non-blocking pipelining.** Each sim calls `submit()` to enqueue a request and gets back an `InferenceToken`. It can then advance other work before calling `try_recv(token)` to check for results. Pending responses are stored in thread-local `HashMap<InferenceToken, Receiver<InferenceResult>>`.

**Deadline-based batching.** The batcher blocks on the first request, then drains the channel for up to 1ms to build a batch. This balances latency (don't wait forever for a full batch) against throughput (don't run the GPU on single samples).

**Condvar parking.** When all of a thread's sims are waiting on inference, the thread calls `wait_for_batch(epoch)` which parks on a condvar instead of busy-polling. The batcher wakes everyone after each batch completes.

This entire pattern survived the migration unchanged. Only the transport inside the batcher changed — from "serialize bytes into mmap, set flag, spin-poll" to "build Burn tensors, call `model.forward()`".

---

## The Pain of Dual Maintenance

Every model architecture change touched a minimum of 6 files across 2 languages:

```
Model change (e.g., add zone tokens):
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Python side:                                                │
│  ├─ training/models/encoder_v5.py      Model definition      │
│  ├─ training/models/actor_critic_v5.py Full actor-critic     │
│  ├─ training/gpu_inference_server.py   SHM parsing + forward │
│  ├─ training/export_actor_critic_v5.py Weight export script  │
│  └─ training/impala_learner_v5.py      Training loop         │
│                                                               │
│  Rust side:                                                  │
│  ├─ weights_actor_critic_v5.rs    Hand-rolled inference      │
│  │  (674 lines of manual matmul, softmax, layer norm)        │
│  └─ gpu_client.rs                 SHM serialization          │
│     (528 lines of byte packing/unpacking)                    │
│                                                               │
│  Shared:                                                     │
│  └─ Implicit contract: field order, byte offsets, padding    │
│     (documented nowhere, verified by "does it crash")        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

The Rust "inference" code in `weights_actor_critic_v5.rs` was 674 lines of `f32` array operations implementing multi-head self-attention, layer norm, GELU, and pointer attention by hand. No autodiff. No operator fusion. Just loops over arrays. It existed only for CPU-side evaluation (running scenarios without a GPU), but it had to exactly match the PyTorch model's numerics or the trained weights would produce wrong actions.

The zone token change was a representative example of the pain:

1. Add `ZoneEncoder` to the Python model class
2. Update `gpu_inference_server.py` to parse zone features from new SHM offsets
3. Update `export_actor_critic_v5.py` to include zone encoder weights
4. Update Rust `serialize_sample()` with new byte layout
5. Update Rust `weights_actor_critic_v5.rs` with CPU zone encoder
6. Update the Python training scripts for the new architecture
7. Debug the alignment bug from step 4

Every one of these steps was a potential source of silent disagreement between Python and Rust.

---

## The Burn Solution

Burn lets you define a neural network module in Rust that is generic over its compute backend:

```rust
#[derive(Module, Debug)]
pub struct ActorCriticV5<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoderV5<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub temporal_cell: CfCCell<B>,
    pub position_head_l1: Linear<B>,
    pub position_head_l2: Linear<B>,
    pub combat_head: CombatPointerHead<B>,
    pub external_cls_proj: Option<Linear<B>>,
    gelu: Gelu,
    d_model: usize,
}
```

The `B: Backend` parameter means this exact code runs on:
- `NdArray` — pure Rust CPU, no dependencies, used in unit tests
- `LibTorch` — libtorch CUDA backend, used for GPU inference and training

Same struct. Same `forward()`. Same weights. No export step, no serialization disagreement, no alignment bugs.

### The Components

Each piece of the architecture is a separate Burn module:

**EntityEncoderV5** — projects entities (34-dim), zones (12-dim), and aggregate features (16-dim) to d_model, adds learned type embeddings, runs self-attention:

```rust
#[derive(Module, Debug)]
pub struct EntityEncoderV5<B: Backend> {
    entity_proj: Linear<B>,
    zone_proj: Linear<B>,
    agg_proj: Linear<B>,
    type_emb: Embedding<B>,
    input_norm: LayerNorm<B>,
    encoder: TransformerEncoder<B>,
    out_norm: LayerNorm<B>,
    d_model: usize,
}
```

**CfCCell** — closed-form continuous-time temporal cell. Replaces GRU with input-dependent time constants, so the cell learns different integration speeds for different game states:

```rust
#[derive(Module, Debug)]
pub struct CfCCell<B: Backend> {
    f_gate: Linear<B>,   // forget gate
    h_gate: Linear<B>,   // candidate gate
    t_a: Linear<B>,      // time constant A
    t_b: Linear<B>,      // time constant B
    proj: Linear<B>,     // output projection
    h_dim: usize,
}
```

**CombatPointerHead** — dual-output head for combat type classification (10-way softmax) and pointer-based target selection (scaled dot-product attention over entity tokens):

```rust
#[derive(Module, Debug)]
pub struct CombatPointerHead<B: Backend> {
    type_l1: Linear<B>,
    type_l2: Linear<B>,
    pointer_key: Linear<B>,
    attack_query: Linear<B>,
    ability_queries: Vec<Linear<B>>,  // one per ability slot
    gelu: Gelu,
    d_model: usize,
    scale: f32,
}
```

### The InferenceClient Trait

The migration was designed as a drop-in replacement. Both the old SHM client and the new Burn client implement the same trait:

```rust
pub trait InferenceClient: Send + Sync {
    fn infer(&self, request: InferenceRequest) -> Result<InferenceResult, String>;
    fn submit(&self, request: InferenceRequest) -> Result<InferenceToken, String>;
    fn try_recv(&self, token: InferenceToken) -> Result<Option<InferenceResult>, String>;
    fn batch_epoch(&self) -> u64;
    fn wait_for_batch(&self, since: u64);
    fn h_dim(&self) -> usize;
}
```

The `GpuInferenceClient` (SHM) and `BurnInferenceClient` both implement this. The sim threads, the episode generator, the evaluation harness — none of them know or care which backend is running. A single feature flag (`burn-gpu`) switches between them.

### BurnInferenceClient: Same Pattern, No Serialization

The new client uses the exact same crossbeam batching pattern. The difference is what happens inside the batcher thread:

```
OLD (SHM):                              NEW (Burn):

request → serialize_sample()             request → build Burn tensors
        → copy bytes into mmap                   → model.forward(tensors)
        → set flag=1                             → extract f32 from output
        → spin-poll flag==2                      → (done)
        → read response bytes
        → deserialize InferenceResult
```

The Burn batcher builds padded tensors directly from the request structs:

```rust
fn forward_batch(&self, batch: &[BatchItem]) -> Vec<InferenceResult> {
    // Build padded f32 arrays from requests
    let mut ent_data = vec![0.0f32; bs * max_ent * ENTITY_DIM];
    for (i, item) in batch.iter().enumerate() {
        for (j, ent) in item.request.entities.iter().enumerate() {
            let base = (i * max_ent + j) * ENTITY_DIM;
            ent_data[base..base + ent.len()].copy_from_slice(ent);
        }
    }

    // Convert to Burn tensors on GPU
    let ent_feat = Tensor::<B, 1>::from_floats(ent_data.as_slice(), dev)
        .reshape([bs, max_ent, ENTITY_DIM]);

    // Forward pass — same code as training
    let (output, h_new) = self.model.forward(
        ent_feat, ent_types, zone_feat,
        ent_mask, zone_mask,
        &ability_cls, Some(agg_feat), Some(h_prev),
    );

    // Extract results (greedy argmax over logits)
    // ...
}
```

No byte packing. No manual offsets. No alignment padding. No flag polling. The model runs directly on the GPU in the same process, and the output comes back as Burn tensors that we index into.

---

## What Got Deleted

```
Deleted files:                                          Lines
────────────────────────────────────────────────────────────────
training/gpu_inference_server.py      SHM server          ~850
training/export_actor_critic_v5.py    weight export        ~320
training/models/encoder_v5.py         Python encoder       ~410
training/models/actor_critic_v5.py    Python actor-critic  ~580
src/.../weights_actor_critic_v5.rs    hand-rolled CPU inf   674
src/.../gpu_client.rs (serialize_*)   byte serialization   ~138
scripts/start_gpu_server.sh           launcher              ~25
scripts/stop_gpu_server.sh            stopper                ~15
docs/shm_protocol.md                  protocol docs        ~200
────────────────────────────────────────────────────────────────
Total deleted:                                          ~4,667

Replacement:
────────────────────────────────────────────────────────────────
src/ai/core/burn_model/actor_critic.rs                    321
src/ai/core/burn_model/entity_encoder.rs                  115
src/ai/core/burn_model/cross_attention.rs                  ~80
src/ai/core/burn_model/cfc_cell.rs                         80
src/ai/core/burn_model/combat_head.rs                     116
src/ai/core/burn_model/inference.rs                       374
src/ai/core/burn_model/config.rs                           23
────────────────────────────────────────────────────────────────
Total added:                                            ~1,109
────────────────────────────────────────────────────────────────
Net: -3,558 lines
```

More importantly, the model definition went from **2 (Python + Rust CPU)** to **1 (Rust, generic over backend)**. There is no longer a class of bugs where Python and Rust disagree about the model architecture.

---

## Benchmark Results

Same workload: 217 drill scenarios, 10 episodes each = 2,170 episodes generating ~143K inference steps. 64 rayon threads, each running 64 concurrent sims (4,096 total).

```
                           SHM (old)         Burn (new)
────────────────────────────────────────────────────────
Python server startup      ~4.0s              0s (no Python)
Episode generation         9.18s              6.94s
────────────────────────────────────────────────────────
Total wall clock           ~13.2s             6.94s
────────────────────────────────────────────────────────
Speedup                                       1.9×
```

The wall clock improvement comes from two sources:

1. **No startup cost.** The SHM path required launching a Python process, loading PyTorch, loading model weights, allocating the SHM file, and waiting for a magic-byte handshake. Burn loads weights at process start and the model is ready immediately.

2. **No serialization overhead.** The SHM path serialized every request into a packed byte buffer, copied it into the mmap, then the Python side deserialized it into numpy arrays, converted to PyTorch tensors, ran the forward pass, serialized the output back into bytes, and Rust deserialized those. Burn builds tensors directly from the request structs and runs the forward pass in-process. The only data movement is the CPU-to-GPU transfer that both paths must do.

The GPU forward pass itself is roughly the same speed — libtorch is libtorch whether you call it from Python or Rust. The gains are all in eliminating the ceremony around it.

---

## Double-Buffered Batching: Eliminating Poll Waste

With the Burn migration done, we profiled the new inference path. The GPU was fast. The model was correct. And 72% of the work the sim threads were doing was wasted.

### The Problem

The crossbeam batching pattern from the previous section has a structural inefficiency. While the GPU processes batch N, the batcher thread is blocked inside `model.forward()`. It cannot collect new requests. The 64 sim threads that have already extracted state for the next tick and called `submit()` are fine — their requests sit in the channel buffer. But the threads that called `try_recv()` to poll for batch N's results are spinning:

```
Profiling (143K steps, 1.9M try_recv polls):

  Polls that returned Some(result):   530K   (28%)
  Polls that returned None:          1,370K  (72%)  <-- waste
```

530K successful polls out of 1.9M total. The other 72% are sim threads calling `try_recv()` in a loop, finding nothing, and calling again. Each poll is cheap — it's just a `crossbeam_channel::TryRecvError::Empty` — but the CPU time adds up across 64 threads.

The timeline looks like this:

```
Single-buffered (current):

Threads:   [extract][submit][poll..poll..poll..][extract][submit][poll..poll..]
                            ^^^^^^^^^^^^^^^^^^^^
                            GPU is busy, no results yet

GPU:              [idle][======= batch N =======][idle][==== batch N+1 ====]
                        ^                       ^
                        batcher blocked         batcher drains channel,
                        in forward()            builds next batch

Batcher:   [drain][forward.................][drain][forward...............]
```

The problem is the gap after the GPU finishes: the batcher has to drain new requests from the channel *before* it can start the next forward pass. During `forward()`, it can't drain. During draining, the GPU sits idle. And the whole time the GPU is running, sim threads that need results are spin-polling with nothing to find.

### Alternative Considered: Lock-Step Synchronous Batching

The simplest fix would be to drop the asynchronous batcher entirely. Use a barrier:

1. All sim threads extract state and submit requests
2. Barrier — wait for all 4,096 sims to submit
3. Run one big GPU batch
4. Scatter results back to threads
5. All threads advance one tick
6. Repeat

No polling. No wasted cycles. Every thread does useful work or is parked on the barrier. Simple.

It would also be slower.

```
Lock-step (considered):

Threads:   [===== extract =====][barrier][idle..........][===== extract =====]
GPU:                                     [=== batch ===]
                                 ^^^^^^^^               ^^^^^^^^
                                 GPU idle               CPU idle
                                 (waiting for           (waiting for
                                  all sims)              GPU)
```

The GPU and CPU never overlap. The fastest sim thread waits for the slowest (barrier), then the CPU waits for the GPU, then the GPU waits for the CPU. With 4,096 sims at varying tick counts (some fights are shorter, some units die early), the straggler problem is severe. The sim extraction time variance across threads is roughly 2x, so the barrier would waste half the CPU time on the fast threads.

The current approach, despite the poll waste, at least lets the GPU and CPU overlap. Sim threads that finish extracting can submit immediately for the next batch without waiting for slower threads. The GPU runs on whatever it has. The throughput is higher even though individual threads waste cycles polling.

### The Solution: Double-Buffered Batcher

The fix is to let the batcher collect the next batch while the GPU processes the current one. Two buffers, two phases, full overlap:

```
Double-buffered (new):

Thread 1:  [extract][sub->B1][extract][sub->B2][extract][sub->B1][extract]
Thread 2:  [extract][sub->B1][extract][sub->B2][extract][sub->B1][extract]
  ...
Thread 64: [extract][sub->B1][extract][sub->B2][extract][sub->B1][extract]

GPU:              [idle][==== batch B1 ====][==== batch B2 ====][==== B1 ====]
                        ^                   ^                   ^
                        swap                swap                swap

Batcher:   [drain B1][forward B1 + drain B2][forward B2 + drain B1][...]
                      ^^^^^^^^^^^^^^^^^^^^^^
                      GPU runs B1 while channel fills B2
```

vs the old single-buffer:

```
Single-buffered (old):

Threads:   [extract][submit][poll..poll..poll..][extract][submit][poll..poll..]

GPU:              [idle][====== batch ======][idle][====== batch ======]

Batcher:   [drain][forward.................][drain][forward...............]
                                             ^^^^^
                                             GPU idle while draining
```

The implementation splits the batcher loop into two concurrent activities:

1. **GPU thread** owns the model and runs `forward_batch()`. When it finishes, it sends completed results to a dispatch channel and swaps to the next pending batch.
2. **Collector thread** drains the request channel continuously, building the next batch. When the GPU thread signals it's ready, the collector hands over the accumulated batch and starts a fresh one.

The key constraint: the collector and GPU thread share no mutable state. The collector fills a `Vec<BatchItem>` and hands ownership to the GPU thread via a channel swap. No locks on the hot path.

From the sim threads' perspective, nothing changes. They still call `submit()` and `try_recv()` on the same `InferenceClient` trait. But now `try_recv()` finds a result waiting almost immediately — because the next batch's results were computed while the sim was extracting state, not after.

The condvar `wait_for_batch` still exists for the case where a thread has no other work to do and all its sims are blocked on inference. But it fires much less often: with double buffering, results arrive while sims are still submitting their next requests, so most `try_recv()` calls succeed on the first or second attempt instead of the thirtieth.

---

## What's Next

**Weight loading from PyTorch checkpoints.** We have months of trained checkpoints saved as `.pt` files. Burn has its own serialization format, so we need a bridge to load existing PyTorch state dicts into Burn modules. This is the immediate blocker for using trained models with the new inference path.

**Training loop in Rust.** The IMPALA V-trace computation and gradient updates still happen in Python. With the model defined in Burn and autodiff available via the `Autodiff<LibTorch>` backend, the entire training loop can move to Rust. No more exporting weights between training iterations — the learner and the inference server share the same model instance.

**Movement curriculum with zone tokens.** The zone token feature that caused the alignment bug? It's what lets the model see objective zones (capture points, extraction areas) in the game state. With the SHM protocol gone, adding new token types to the model is a single-file change instead of a six-file ordeal. The movement curriculum can now teach the model to navigate toward objectives, not just fight.

---

*The SHM protocol served us well for six months and through five model versions. It enabled the first IMPALA training runs that produced real learning curves. But the maintenance tax grew with every architecture change, and the alignment bug that ate a day of debugging was the final straw. Sometimes the best optimization is deleting the abstraction boundary entirely.*
