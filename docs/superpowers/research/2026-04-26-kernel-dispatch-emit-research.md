# Kernel Dispatch-Emit Research (2026-04-26)

Design-input document for the upcoming brainstorm on a "dispatch-emit" abstraction that
would have the DSL compiler emit not just WGSL kernel source but also the Rust wrapper
struct, buffer-ownership declaration, and the `step_batch` dispatch sequencing for the 14
hand-written kernels in `crates/engine_gpu/` (with 15-25 more queued, including the 8 fold
kernels currently planned as one-per-view wrappers).

The goal of this document is **not** to pick a design. It is to surface (a) what the
literature does for analogous problems, (b) which patterns map cleanly onto our wgpu /
WebGPU constraint, and (c) what we already have in the codebase so the brainstorm starts
from the actual surface area, not a sketch of it.

Cited literature is concentrated under each section; full URL list is at the end of every
section.

---

## TL;DR

- **Render-graph patterns (Frostbite FrameGraph, Themaister Granite, Falcor) are the
  closest analog**, but most of their machinery — pass culling, transient texture aliasing,
  layout transitions, async-queue scheduling — is overkill for our compute-only,
  WGSL-driven, single-queue, fixed-pipeline shape. The parts that DO transfer cleanly:
  declarative resource declarations, retained graph structure (build once, replay many),
  and "lifetime classification" (persistent vs transient vs per-iteration). Don't try to
  steal the whole concept.
- **The 8 fold kernels are already a templated form of dispatch-emit** (see
  `view_storage.rs:323-458` `fold_pair_events` + `fold_slot_events`): one shared
  `cs_fold` entry-point name, one BGL-per-shape, one `bind_group_for` helper. The
  per-tick dispatcher walks the same path for every view, branching on `ViewShape`. This
  is the strongest evidence that homogeneous kernel families benefit from generation.
  The other 13 kernels are heterogeneous enough that a single generator template would be
  procrustean — the abstraction needs to be parameterised by *kernel family*, not by
  individual kernel.
- **DirectX 12 Work Graphs (2024) and VK_EXT_device_generated_commands solve a different
  problem** (GPU-driven dispatch generation for very wide irregular workloads). They are
  not available on wgpu/WebGPU and the model — nodes that recursively launch other nodes
  — is more dynamic than our 100ms fixed-tick deterministic schedule needs. CUDA Graphs
  ("capture once, replay N times") is the closer analog conceptually, but the *replay*
  benefit is already approximated by our resident `step_batch` building one encoder for N
  ticks.
- **`wgsl_bindgen` / `wgsl_to_wgpu` exist and largely solve the codegen-binding problem**
  (Pod struct emission, bind-group layout, compile-time alignment assertions via naga
  reflection). The dispatch-emit abstraction can rest on top of either of these instead
  of inventing the WGSL-Rust bridge from scratch — but they only generate the *binding*
  glue, not the dispatch sequencing or the buffer ownership decisions.
- **The biggest unanswered design question is buffer ownership.** Our codebase has 4 distinct
  ownership styles already (per-kernel `BufferPool`, caller-supplied `&wgpu::Buffer`,
  shared resident context via `ResidentPathContext`, and pool-cached
  `Vec<wgpu::Buffer>` view-buffer handles). Picking one is the load-bearing decision,
  not the WGSL emission part.

---

## 1. GPU Work Graph Survey

### DirectX 12 Work Graphs (Microsoft / 2024-03)

A graph of compute nodes where each shader can request additional invocations of other
nodes without CPU round-trip. Authored against the `lib_6_8` HLSL shader target. Acyclic
with one exception: a node may output to itself; depth limit 32 including recursion.
Nodes can be thread-launch / thread-group-launch / variable-grid. Compiled into a state
object, executed by `DispatchGraph`. AMD has driver support (Adrenalin 24.x for RX 7000).
Key value proposition: eliminates CPU launch overhead for irregular workloads where the
work shape is determined on the GPU.

What this solves:
- Producer-consumer chains where the consumer's workgroup count depends on producer
  output.
- "Mega-kernel" decomposition without the latency of indirect dispatch.

What does NOT fit our context:
- **Not exposed in wgpu.** WebGPU has no work-graph proposal. Adding it would mean a
  D3D12-specific path, abandoning Vulkan/Metal/web parity.
- We don't have irregular work. Our pipeline is fixed: mask → scoring → apply → movement
  → cascade × 8 iterations. The shape is known at scenario start (agent count fixes
  workgroup count for everything except per-event physics).
- The model assumes nodes "request" other nodes; ours has CPU-scheduled fixed order with
  small numbers of indirect dispatches (cascade per-iter event count).

Useful concept to steal: **the explicit *node* abstraction** — a kernel's identity in the
graph is decoupled from its WGSL source. We could expose the same idea (a `Node` value
that carries its dispatch shape, bind group, output buffers) without adopting GPU-side
chaining.

Sources:
- [D3D12 Work Graphs — DirectX Developer Blog](https://devblogs.microsoft.com/directx/d3d12-work-graphs/)
- [Advancing GPU-Driven Rendering with Work Graphs — NVIDIA](https://developer.nvidia.com/blog/advancing-gpu-driven-rendering-with-work-graphs-in-direct3d-12/)
- [DirectX-Specs — WorkGraphs.html](https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs.html)
- [GPU Work Graphs in D3D12 — AMD GPUOpen](https://gpuopen.com/learn/gpu-work-graphs/gpu-work-graphs-intro/)

### VK_EXT_device_generated_commands (Vulkan / promoted from NV in 2024)

Lets a compute shader write a stream of "indirect tokens" describing
state changes + dispatches; a generation pass turns that stream into
executable command buffer state. For compute, supported tokens include
`DISPATCH`, `EXECUTION_SET` (switch shaders mid-stream), `PUSH_CONSTANT`.
Works against an `IndirectExecutionSet` of pre-compiled pipelines.

What this solves:
- Removes CPU bottleneck for streams of mixed-shader dispatches. Original NV motivation
  was draw-call generation; the EXT extends it to compute.
- Allows the application's own GPU code to schedule the work graph.

What does NOT fit:
- **Not in wgpu.** Cross-platform availability is also still patchy (the EXT promotion
  was 2024).
- We have ≤ 14 dispatches per tick. CPU overhead is not the bottleneck — kernel
  execution is. Generating commands on-GPU adds complexity for no savings.

Useful concept: the **indirect token stream** is essentially a serialised graph that
the GPU can replay. If we ever need to amortise CPU encoder cost (at very high tick rates
or many parallel scenarios), our compiler-emitted dispatch sequencer is morally the same
thing — just CPU-side.

Sources:
- [VK_EXT_device_generated_commands proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_device_generated_commands.html)
- [VK_EXT_device_generated_commands appendix](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/VK_EXT_device_generated_commands.adoc)
- [New: Vulkan Device Generated Commands — NVIDIA](https://developer.nvidia.com/blog/new-vulkan-device-generated-commands/)

### CUDA Graphs (`cudaGraph_t`)

Capture-then-replay model: record CUDA stream operations between
`cudaStreamBeginCapture` and `cudaStreamEndCapture`, instantiate via
`cudaGraphInstantiate`, launch with `cudaGraphLaunch`. Reported speedups
1.12× (BERT) to 2.3× (LLaMA-7B) come from amortising launch overhead
across replays.

What this solves:
- Repetitive same-shape workloads: the CPU pays the dispatch cost once.
- Optimiser can do graph-wide transforms (constant-time launch for straight-line graphs,
  see NVIDIA's 2024 post).

What does NOT fit:
- **No wgpu equivalent.** Closest wgpu primitive is the encoder itself, which is
  reusable within a submit but not retained across submits.
- Our `step_batch` already approximates this: build one encoder for N ticks, submit once.
  The CUDA-Graph win — skipping per-launch overhead — is partially captured.

Useful concept to steal: **"a static graph is just a value"**. CUDA Graphs is the
strongest existing-system evidence that retained dispatch graphs scale well. Our
compiler-emitted sequencer would be a Rust-level analog.

Sources:
- [Getting Started with CUDA Graphs — NVIDIA](https://developer.nvidia.com/blog/cuda-graphs/)
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [CUDA Programming Guide — Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

### Metal Performance Shaders Graph (MPSGraph)

Multi-dimensional tensor compute graph. Three edge kinds: input,
output, control-dependency (must execute before, even if no data
dependency). The compiler fuses adjacent ops into single Metal shaders.

Key idea worth lifting: **explicit control-dependency edges** that don't carry data.
Useful when you want to express "reset the apply_event_ring tail before the next tick's
mask kernel runs" without inventing a phantom buffer to depend on.

Sources:
- [Metal Performance Shaders Graph — Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Build customized ML models with MPSGraph — WWDC20](https://developer.apple.com/videos/play/wwdc2020/10677/)

### WebGPU `RenderBundle` and the absent compute equivalent

`RenderBundle` lets you pre-record a sequence of render-pass commands
and replay them with reduced state-set cost. **There is no
`ComputeBundle`** — and the GPU-Web group has explicitly said this is
intentional: compute workloads have far less state churn (a few big jobs
versus thousands of draws), so the win wouldn't justify the spec
surface (see toji.dev best-practices).

Implication for us: WebGPU does not give us a "record once, replay every tick" primitive
on the compute side. Our retention strategy has to live above wgpu — either by reusing
the same encoder across ticks within a `step_batch` (current resident pattern) or by
building a *Rust-level* retained dispatch list that re-encodes cheaply per submit.

Sources:
- [WebGPU Render Bundle best practices — Toji](https://toji.dev/webgpu-best-practices/render-bundles.html)
- [GPURenderBundle — MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPURenderBundle)
- [Are command buffers reusable? — gpuweb#1971](https://github.com/gpuweb/gpuweb/issues/1971)

### Cross-system synthesis

| System | Model | What we can steal |
|---|---|---|
| D3D12 Work Graphs | GPU-side node spawning | Node-as-value abstraction |
| VK_EXT_device_generated_commands | Indirect token stream | Concept of serialised dispatch graph |
| CUDA Graphs | CPU capture-replay | Retained graph structure, instantiate once |
| MPSGraph | Tensor DAG with control deps | Control-only edges (non-data dependencies) |
| WebGPU RenderBundle | Reusable command lump | Confirms compute-side retention has to be CPU |

The shared idea across every system: **separate "describe the graph" from "execute the
graph"**. Our current `step_batch` mixes both.

---

## 2. Render / Compute Graph Libraries

### Bevy `RenderGraph` (closest in-Rust + wgpu reference)

Three-component model: **Nodes** (own draw/dispatch logic), **Edges** (ordering: `NodeEdge`
for pure ordering, `SlotEdge` for data dependency + ordering), **Slots** (typed
input/output channels). Resources persist across frames; the graph itself is retained and
stateless during execution. Each node has its own internal state. Runner topologically
sorts and executes in dependency order.

What's good:
- Slot-typed edges catch "wired the wrong buffer to the wrong input" at registration time.
- Retained structure: a frame just walks the graph; addition/removal happens out-of-band.
- Nodes are external to the runner: third-party plugins drop in nodes without touching
  the graph runner.

What's debated in the Bevy community:
- The `Slot` system is being revisited (see "Render graph slots revival" discussion). Slots
  add ceremony for simple cases. Multiple Bevy contributors have noted that the slot model
  is awkward for compute-heavy workloads.
- Edges are unordered when both are `NodeEdge` to the same target; deterministic ordering
  needs explicit edges.

What does NOT translate cleanly:
- Bevy's graph re-walks every frame. We want to walk it `N` times per `step_batch` —
  potentially thousands of times — so per-walk overhead matters more.
- Bevy nodes are graphics-first (they encode draw-call state). Our nodes are pure compute.

Sources:
- [RenderGraph in bevy_render::render_graph](https://docs.rs/bevy_render/latest/bevy_render/render_graph/struct.RenderGraph.html)
- [Render Architecture Overview — Bevy Cheat Book](https://bevy-cheatbook.github.io/gpu/intro.html)
- [Render graph slots revival proposal](https://github.com/bevyengine/bevy/discussions/8644)
- [A walkthrough of Bevy's rendering — Discussion #9897](https://github.com/bevyengine/bevy/discussions/9897)

### Falcor (NVIDIA)

Render-graph-as-API library used by NVIDIA research. Each `RenderPass` declares input,
output, and *internal* fields (`Field::Flags::Persistent` opts a resource out of
aliasing; default is transient). The graph compiler chooses memory aliasing and inserts
transitions automatically. Three-phase execution: setup → compile → execute.

Steal-worthy: the **`Persistent` flag as opt-out** (default = transient). Our world is
inverted — most buffers are persistent. But the *flag-based ownership* model maps onto
our four-tiered ownership scheme cleanly.

Sources:
- [Falcor render-passes docs](https://github.com/NVIDIAGameWorks/Falcor/blob/master/docs/usage/render-passes.md)
- [Falcor RenderGraph.cpp](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/RenderGraph/RenderGraph.cpp)
- [Falcor RenderPass.h](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/RenderGraph/RenderPass.h/)

### Frostbite FrameGraph (Yuriy O'Donnell, GDC 2017)

The "first triple-A render graph" that everyone now copies. Three phases:
1. **Setup** — declarative pass description, declares which resources each pass reads/writes.
2. **Compile** — lifetime analysis, allocation, aliasing, barrier insertion.
3. **Execute** — walk and dispatch.

Memory model: heaps survive across frames, allocator tracks high-water mark, blocks
released only after several frames of low demand. This avoids alloc/free churn on
transient resources without permanent over-allocation. Resource aliasing reuses memory
across non-overlapping lifetimes.

Steal-worthy parts for us:
- **Declarative pass description.** Today our `step_batch` is fully imperative.
- **Lifetime analysis as a separate pass** that can be inspected, tested, asserted-on.
- **High-water mark + lazy-release** for our growing buffers (event rings, view buffers).

Don't steal:
- Aliasing transients across passes. Our biggest buffers (`resident_agents_buf`,
  `view_storage` cells, `apply_event_ring`) are persistent across the whole batch by
  design — there's nothing to alias.

Sources:
- [FrameGraph: Extensible Rendering Architecture in Frostbite — slides](https://www.slideshare.net/slideshow/framegraph-extensible-rendering-architecture-in-frostbite/72795495)
- [Frame Graph: Production Engines — stoleckipawel.dev](https://stoleckipawel.dev/posts/frame-graph-production/)
- [Render Graphs — Riccardo Loggini](https://logins.github.io/graphics/2021/05/31/RenderGraphs.html)

### Themaister Granite (open-source Vulkan reference)

Comprehensive render graph in `renderer/render_graph.{cpp,hpp}`. Compilation phases:
dependency analysis → topological sort → lifetime calculation → pass culling → barrier
insertion. Uses `VkEvent` for in-queue resources, `VkSemaphore` for cross-queue.

Themaister's blog posts (2017 and 2019) are the de-facto written introduction to render
graphs and the explicit pitfalls section is gold:
- "Doing async compute is hard precisely because non-local frame knowledge is required to
  place barriers."
- "Render graphs that try to handle both compute and rasterization uniformly always
  underdeliver on one of them."

Steal-worthy:
- The **compile-then-replay** structure (Granite recompiles the graph only on topology
  change).
- **Pass culling**: passes whose outputs aren't consumed get dropped. Useful for us if a
  scenario doesn't actually use a particular view — we could skip its fold kernel
  entirely.

Sources:
- [Granite — GitHub](https://github.com/Themaister/Granite)
- [Render graphs and Vulkan — a deep dive (Themaister)](https://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/)
- [A tour of Granite's Vulkan backend Part 1](https://themaister.net/blog/2019/04/14/a-tour-of-granites-vulkan-backend-part-1/)

### What separates GOOD render graphs from bad ones

Cross-cutting findings from Themaister, Frostbite, Falcor, Granite, Tony Adriansen's 2025
write-up, "apoorvaj.io/render-graphs-1", and Traverse Research's "Render Graph 101":

1. **Make the resource declaration the source of truth.** Every pass declares its inputs
   and outputs. The graph is everything else.
2. **Don't try to alias persistent resources** — only alias transients. Distinguishing
   them requires explicit lifetime markers (Falcor's `Persistent` flag, Granite's
   `add_imported_*`).
3. **Build the graph once, execute many times.** Reconstruction per frame is the
   anti-pattern that kills throughput.
4. **Topological sort + barrier insertion is mechanical** — once dependencies are
   declared, the rest is solved by a textbook algorithm. The hard part is making the
   declarations idiomatic.
5. **Type-check edges at registration time.** Slot types catch wiring errors before they
   become silent driver crashes.
6. **Avoid over-generalising compute and graphics.** The render-graph antipattern most
   often cited is "we tried to make one abstraction for both, neither worked well."

Sources:
- [Render graphs — apoorvaj.io](https://apoorvaj.io/render-graphs-1)
- [Render Graph 101 — Traverse Research](https://blog.traverseresearch.nl/render-graph-101-f42646255636)
- [Building a Vulkan Render Graph — Tony Adriansen 2025](https://tadriansen.dev/2025-04-21-building-a-vulkan-render-graph/)
- [Rendergraphs and how to implement one — Ponies & Light](https://www.poniesandlight.co.uk/reflect/island_rendergraph_1/)

### The wgpu-ecosystem state of the art

There is no in-tree wgpu render-graph crate. Closest things:
- **`wgsl_bindgen`** — generates Rust types/bind-group-layouts from WGSL using naga
  reflection, but doesn't model the dispatch graph.
- **`wgsl_to_wgpu`** — same idea, simpler scope.
- **`renderling`** — schell's GPU-driven renderer; uses a hand-written graph internally,
  not a reusable one.
- **`Vello`** — 2D compute renderer; has an internal compute pipeline, not a graph
  abstraction.

Implication: nothing off-the-shelf to drop in. Anything we build is bespoke.

Sources:
- [wgsl_bindgen — crates.io](https://crates.io/crates/wgsl_bindgen)
- [wgsl_to_wgpu — GitHub](https://github.com/ScanMountGoat/wgsl_to_wgpu)
- [renderling — GitHub](https://github.com/schell/renderling)
- [Vello — GitHub](https://github.com/linebender/vello)

---

## 3. Memory & Buffer Lifetime Patterns

### Lifetime classes in production engines

Every render graph distinguishes:
- **Persistent / imported**: lives beyond the graph (e.g. swapchain images,
  application-managed buffers). Graph cannot allocate, alias, or release these.
- **Transient (frame-scoped)**: graph owns and may alias across non-overlapping passes.
- **Per-iteration / scratch**: lives for one node only. Sometimes implemented as a
  ring with explicit ping-pong slots.

Frostbite's lifetime calculation walks the DAG, computes `(first_write, last_read)` for
every transient, and bucket-sorts by lifetime to discover aliasing opportunities (see
GDC 2017 slides 35-50, summarised in the dev.to and stoleckipawel write-ups).

### Mapping to our codebase

The 14 existing kernels and their backing buffers, classified:

| Buffer | Owner | Class | Notes |
|---|---|---|---|
| `resident_agents_buf` | `ResidentPathContext` | persistent (batch) | Mutated each tick, no scratch |
| `sim_cfg_buf` | `ResidentPathContext` | persistent (batch) | Atomic tick lives here |
| `mask_bitmaps_buf` | `FusedMaskKernel` | persistent (batch) | Re-cleared each tick |
| `agent_data_buf` (scoring) | `ScoringPool` | per-tick scratch | Re-uploaded by `upload_soa_from_state` |
| `scoring_out_buf` | `ScoringPool` | per-tick scratch | Producer for apply/movement |
| `apply_event_ring` | `CascadeCtx` | per-tick scratch | Tail cleared each tick |
| `physics_ring_a / _b` | `CascadeResidentCtx` | ping-pong | Per-iter, swapped by indirect args |
| `batch_events_ring` | `CascadeResidentCtx` | persistent (batch) | Append-only across ticks |
| `chronicle_ring` | `CascadeResidentCtx` | persistent (batch) | Snapshot watermark |
| `indirect_args` | `ResidentPathContext` | persistent (batch) | Cascade indirect dispatch |
| `view_storage` cells/anchors/ids | `ViewStorage` | persistent (batch) | 144 MB at N=100k |
| `gold_buf`, `standing_storage`, `memory_storage` | `ResidentPathContext` | persistent (batch) | Read by scoring, mutated by cold-state replay |
| `chosen_ability_buf` | `ResidentPathContext` | per-tick scratch | Producer for apply_actions |
| Spatial query outputs | `SpatialOutputs` | per-tick scratch | 80 MB; 3 queries/tick |

**Observation:** very few "transient" buffers in the render-graph sense. Most are
persistent across the batch. The ones that ARE per-tick (`scoring_out_buf`,
`apply_event_ring`, `chosen_ability_buf`, spatial outputs) could *in principle* alias —
but they're already small relative to the persistent set, so aliasing is unlikely to
win meaningful memory.

What the dispatch-emit abstraction needs to model is therefore **less about aliasing**
(low payoff) and **more about declaring which class each buffer belongs to** (high payoff
for correctness & debuggability).

### Buffer-pool patterns we already use

Inspecting `scoring.rs`, `apply_actions.rs`, `movement.rs`, the four ownership styles
currently coexist:

1. **Per-kernel `BufferPool`** (`scoring.rs:440`, `apply_actions.rs:198`,
   `movement.rs`). Each kernel owns its scratch buffers; resized via `ensure_pool` on
   `agent_cap` change.
2. **Caller-supplied buffer handles** (`run_resident` signatures pass
   `agents_buf: &wgpu::Buffer`, `mask_bitmaps_buf: &wgpu::Buffer`, etc.). The kernel
   doesn't own these, just binds them.
3. **Shared `ResidentPathContext`** holds the cross-kernel persistent buffers
   (`resident_agents_buf`, `sim_cfg_buf`, `gold_buf`, etc.). This is the de-facto
   "graph-owned persistent set."
4. **Pool-cached `Vec<wgpu::Buffer>` view-buffer handles** (`scoring.rs:468`
   `view_buf_handles`). Snapshots from `ViewStorage` at upload time, since `wgpu::Buffer`
   is Arc-backed and cheap to clone.

The proliferation is the smell. A dispatch-emit abstraction that pulled all four into a
single declarative model — every buffer has an owner, an access mode, and a lifetime
class — would eliminate ~half the bookkeeping.

### Bind-group lifetime: the `cached_resident_bg` pattern

Every kernel re-uses an interesting trick: cache the resident bind-group keyed by the
caller-supplied buffer identities (`scoring.rs:514-516`, `apply_actions.rs:182-196`,
`movement.rs` similar). The cache hits 100% after tick 1 because resident buffers don't
swap. This is a hand-rolled optimisation that the abstraction should capture:
**bind-groups are pure functions of (BGL, [Buffer])** and should be memoised at the
graph level, not per-kernel.

### WebGPU automatic synchronization

Important constraint: WebGPU auto-inserts barriers between dispatches in the same encoder
(see "The case for passes" — gpuweb#64). We rely on this. The abstraction must not break
it. In particular, if we ever batch multiple dispatches into a single compute pass for
fewer pass-begin/pass-end overheads, we lose the auto-barrier between them and need to
prove that's safe per pair.

Sources:
- [WebGPU spec — synchronization](https://www.w3.org/TR/webgpu/)
- [The case for passes — gpuweb#64](https://github.com/gpuweb/gpuweb/issues/64)
- [Vulkan Memory Allocator — Resource aliasing](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/resource_aliasing.html)
- [Aliasing transient textures in DirectX 12 — Pavel Šmejkal](https://pavelsmejkal.net/Posts/TransientResourceManagement)

---

## 4. Dispatch Sequencing Patterns

### Today's imperative sequencing in `step_batch`

`crates/engine_gpu/src/lib.rs:1080-1450` is the canonical example. Per tick:
1. `fused_unpack_kernel.encode_unpack(...)` — unpacks SoA + clears mask bitmaps.
2. `alive_pack_kernel.encode_pack(...)` — alive-bitmap pack.
3. `run_spatial_resident_pre_scoring(...)` — 3 spatial queries.
4. `mask_kernel.run_resident(...)` (or `.run_resident_split(...)` under env-var).
5. `scoring_kernel.run_resident(...)`.
6. `apply_actions.run_resident(...)`.
7. `movement.run_resident(...)`.
8. `encode_append_apply_events(...)`.
9. `run_cascade_resident(...)` — internally seed + 8 iterations of physics +
   indirect-arg ping-pong.
10. Cold-state replay (CPU side, post-readback).

Plus profiler `mark` and `write_between_pass_timestamp` calls between every pair, used by
the perf harness.

The whole thing is ~370 lines of imperative encoding inside one giant per-tick `for`
loop. Ordering between kernels is encoded only as comments and as the Rust source line
order. A reader has to manually verify "does scoring run before apply_actions?" by
reading the file.

### Patterns from the literature

#### a) **Pure imperative + comments** (current state)

Pros: zero indirection; debugger steps through the actual order; the comments
double as the dependency documentation.
Cons: dependencies are implicit; refactoring is risky; can't dump or visualize the
schedule; can't insert/remove a kernel without surgery.

#### b) **Declarative DAG** (Bevy, Falcor, Frostbite, Granite)

Each kernel declares `inputs: [Buffer], outputs: [Buffer]`; a topological sort produces
the schedule. Dependencies are *inferred* from buffer access.

Pros: removable nodes (pass culling); inspectable; refactor-safe.
Cons: the sort can produce a non-deterministic order when there are independent passes
(important for our determinism contract — see P5 in the constitution), so we'd need to
fix tie-breaking explicitly. Also: when *every* pass writes a different buffer and
reads the previous, the sort just reproduces the imperative order, so the abstraction
adds ceremony without simplifying.

#### c) **Channel-typed flow** (Bevy slot edges, MPSGraph)

Stronger typing than (b): outputs are *typed slots* (`scoring_out: ScoreOutputBuffer`)
that connect to typed inputs. Catches "wired wrong buffer" at registration. Cost is more
boilerplate per node, and mismatched types between WGSL and Rust become a third axis to
keep in sync (DSL → WGSL → Rust slot type).

#### d) **Explicit barriers vs automatic dependency inference**

Frostbite, Granite, Bevy: dependencies inferred from buffer access mode, then barriers
inserted by the compiler.
WebGPU: barriers inserted by the *driver* between every pair of dispatches in the same
encoder (no explicit pipeline-stage barriers).
**Implication: our compiler doesn't need to insert barriers.** We get them for free.
This significantly reduces the ROI of a full-featured dependency-inference engine.

#### e) **Indirect dispatch** (cascade physics, current)

`cascade_resident.rs` already does this: kernel A writes the workgroup count for kernel
B's `dispatch_workgroups_indirect` call. Means the abstraction has to handle two kinds of
dispatch: direct (workgroup count is a function of agent_cap, computed in Rust) and
indirect (workgroup count is in a buffer, written by an upstream kernel). The cascade's
8-iter ping-pong is essentially a hand-rolled tail-recursive node.

WebGPU `dispatchWorkgroupsIndirect` signature (`docs.rs/wgpu` / MDN):
- Indirect buffer must have `INDIRECT` usage flag.
- Args: 3× `u32` (X, Y, Z workgroup counts), tightly packed.
- `wgpu` 0.23+ silently inserts a validating dispatch around each indirect call (see
  gfx-rs/wgpu#6567) — note for the perf harness.

Sources:
- [GPUComputePassEncoder.dispatchWorkgroupsIndirect — MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUComputePassEncoder/dispatchWorkgroupsIndirect)
- [WebGPU Indirect Draw best practices — Toji](https://toji.dev/webgpu-best-practices/indirect-draws.html)
- [wgpu 23 indirect-dispatch validation regression — gfx-rs/wgpu#6567](https://github.com/gfx-rs/wgpu/issues/6567)

### Where we already have semi-declarative dispatch

`view_storage.rs:323-458` (`fold_pair_events` + `fold_slot_events`) is the closest thing
in our codebase to a declarative dispatcher. The caller picks the view by name; the
helper resolves shape, picks the right pipeline, builds the BGL, dispatches. The shape
and BGL are emitted by `dsl_compiler::emit_view_wgsl::emit_view_fold_wgsl` for each
view. **8 fold kernels share one `cs_fold` entry-point name and one
`build_fold_module_wgsl` template** (`view_storage.rs:854`, `999-1043`), parameterised by
`ViewStorageSpec`.

This is exactly the pattern that scales. The brainstorm question is: should the other 13
kernels be coerced into this shape (one-template-per-family), or should we accept that
they're heterogeneous and the abstraction only helps when a *family* of N homogeneous
kernels exists?

---

## 5. Codegen Patterns (Rust + wgpu specific)

### `wgsl_bindgen` / `wgsl_to_wgpu` — the off-the-shelf option

Both crates parse WGSL via `naga` and emit Rust that includes:
- `#[derive(bytemuck::Pod)]` structs matching every WGSL `struct` with explicit padding.
- `const _: () = assert!(...)` static assertions for offset / size / alignment using
  `naga`'s layout calculations. (This catches the bytemuck-with-implicit-padding
  footgun; see `Lokathor/bytemuck` discussion #86.)
- One `WgpuBindGroupN` type per bind group, with a typed `WgpuBindGroupNEntries` builder
  whose field names match the WGSL bindings exactly. Compile-time check that you supplied
  every binding.
- Pipeline-layout / pipeline construction helpers.

Implication: **we don't need to write the WGSL→Rust binding bridge ourselves.** What
we'd write is the layer above — which kernel exists, what dispatch shape, what
sequencing position.

Caveat: `wgsl_bindgen` runs at *build time* via `build.rs`. Our compiler runs the WGSL
emission at *runtime* (from `.sim`/`.ability` files in `assets/`). The two need to be
reconciled: either move WGSL emission to build time (loses hot-reload of `.sim` files
without a rebuild), or run a `wgsl_bindgen`-equivalent at runtime over our emitted WGSL
(harder; would need to vendor naga's reflection path).

There's a third option: **have the dsl_compiler emit the bindgen-equivalent Rust as
strings**, sidestepping the runtime/build-time question. This is what
`emit_scoring.rs` / `emit_physics.rs` / etc. already do for the *CPU* path. Extending
it to the GPU wrapper struct is the obvious incremental step.

Sources:
- [wgsl_bindgen — docs.rs](https://docs.rs/wgsl_bindgen/)
- [wgsl-bindgen — Swoorup GitHub](https://github.com/Swoorup/wgsl-bindgen)
- [wgsl_to_wgpu — ScanMountGoat GitHub](https://github.com/ScanMountGoat/wgsl_to_wgpu)
- [Memory Layout in WGSL — Learn Wgpu](https://sotrh.github.io/learn-wgpu/showcase/alignment/)
- [bytemuck implicit-padding footgun — discussion #86](https://github.com/Lokathor/bytemuck/discussions/86)

### Naga reflection — the underlying primitive

`naga` parses WGSL into an IR that exposes:
- Module-level `EntryPoint` list (so we can verify `cs_scoring` is actually defined).
- `GlobalVariable` list with binding/group/access-mode (so we can derive the BGL from the
  WGSL itself, instead of hand-syncing as today's code does).
- Type registry with computed layout (offset/size/alignment per field, including padding
  for `vec3` / `mat3x3`).

If we don't want to depend on `wgsl_bindgen`, we can call into `naga` directly the way
`wgsl_bindgen` does. `wgpu` already has `naga` as a dep, so adding the reflection path
costs nothing.

The current `dsl_compiler` does *not* use `naga` at all — it emits WGSL as strings and
trusts that the human-written WGSL prefix constants and the human-written Rust BGL stay
in sync. **Closing this gap is independent of the dispatch-emit abstraction** but the
abstraction is the natural place to land it.

Sources:
- [naga — gfx-rs/naga GitHub](https://github.com/gfx-rs/naga)
- [naga-cli — lib.rs](https://lib.rs/crates/naga-cli)
- [naga_oil — lib.rs](https://lib.rs/crates/naga_oil)

### Error-reporting patterns

Today (`scoring.rs:536-545`, `view_storage.rs:855-863`, `apply_actions.rs:215-223`,
`movement.rs` similar):

```rust
device.push_error_scope(wgpu::ErrorFilter::Validation);
let shader = device.create_shader_module(...);
if let Some(err) = pollster::block_on(device.pop_error_scope()) {
    return Err(...::ShaderCompile(format!(
        "{err}\n--- WGSL source ---\n{wgsl}"
    )));
}
```

Every kernel has the same error-scope boilerplate. The pattern is correct (push scope,
compile, pop scope, embed WGSL in error) but copy-pasted. Trivial to extract; minor
quality-of-life rather than a load-bearing decision.

For *runtime* WGSL parse errors, naga gives location info (`naga::front::wgsl::ParseError::emit_to_string`)
that's much nicer than wgpu's wrapped `wgpu::Error`. The dispatch-emit abstraction is the
natural place to surface that.

### `include_str!` vs string emission at build time

Today the engine has both: many kernels compose WGSL from string-emit functions
(`emit_scoring_wgsl_atomic_views`, `emit_view_fold_wgsl`, etc.) at runtime; some have
hand-written WGSL. Either is fine. The dispatch-emit abstraction probably wants to keep
runtime emission so that hot-reload of `.sim` files keeps working. (Build-time emission
would force a rebuild on every DSL edit.)

---

## 6. Implications for Our Engine

This section sketches potential shapes for the abstraction. **It does not pick one** —
that's the brainstorm. It surfaces the tradeoffs and the load-bearing decisions.

### What family of kernels does this serve?

Looking at our 14 kernels by structural similarity:

**Family A — homogeneous fold-style** (8 fold kernels, projected): drain events from a
ring → reduce per-agent → write into `view_storage`. **Strong codegen win: one template,
8 instantiations.**

**Family B — homogeneous per-agent** (mask, scoring, apply_actions, movement): one thread
per agent, read SoA, read views, write per-agent output buffer. **Medium codegen win:
shared structure, but each kernel's body is unique.**

**Family C — irregular** (cascade physics with 8-iter indirect ping-pong, spatial hash
rebuild + per-agent queries, alive_bitmap pack, fused_unpack). **Low codegen win:
each is its own thing.**

The brainstorm's first decision: scope to A only (clear win, low risk), A+B (more
ambitious, useful for the 15-25 new ones), or A+B+C (full graph abstraction;
likely overkill for our shape).

### What would the DSL author write?

Today (the fold case, e.g.):
```
view threat_level = pair_map<f32> @decay 0.97 {
    fold AgentAttacked(actor, target) -> { self += 1.0 }
}
```

The compiler already lowers this to:
- WGSL `cs_fold_threat_level` (via `emit_view_fold_wgsl`).
- A `ViewStorageSpec` that drives the BGL/buffer choices in `view_storage.rs`.

For the brainstorm, the question is: do we want the DSL author to write *anything* about
the dispatch? Or should the compiler infer dispatch position purely from the lowering?

Three sketch options to debate:

**Option α — pure inference.** Author writes only the DSL (today's syntax). Compiler
classifies (this is a fold kernel, fold kernels run in phase 5; this is a mask rule,
mask runs in phase 1) and slots it into a fixed phase-ordered schedule. The schedule is
an implementation detail.

**Option β — explicit phase tag.** Author writes `@phase(view_fold)` (or similar) on
each kernel definition. Compiler validates that the kernel's I/O is consistent with the
phase. Phase ordering is fixed, but the assignment is explicit.

**Option γ — dataflow declaration.** Author writes `reads: [agent_data, threat_level]
writes: [score_out]`. Schedule is inferred from dataflow per Frostbite/Bevy. Most
flexible; biggest authoring cost.

Decision input: how often do we add a kernel that doesn't fit one of the existing phases?
If the answer is "never" then α; if "yearly" then β; if "monthly" then γ.

### Where do buffer-ownership decisions live?

Three sketch options:

**(i)** In the DSL: `view threat_level = pair_map<f32> @persistent { ... }`, etc.
**(ii)** In a sidecar config: a separate `kernels.toml` (or compiler-IR-driven) declares
the ownership for each kernel.
**(iii)** In the compiler IR: ownership is *derived* from the kernel's role — fold
kernels write to view storage (always persistent), scoring writes to scratch (always
per-tick), etc.

Option (iii) is the cleanest if our four ownership classes really do correspond
one-to-one with kernel families. Worth checking on the brainstorm: are there any
kernels in flight whose ownership doesn't follow from their family?

### What does the compiler emit?

Today (per kernel):
- WGSL source (e.g. `emit_view_fold_wgsl`).
- (Sometimes) a CPU-side counterpart (e.g. `emit_pick_ability_cpu`).
- The Rust wrapper struct is hand-written.

In the dispatch-emit world, the compiler additionally emits:
- A Rust wrapper struct (today's `ScoringKernel`, `ApplyActionsKernel`, etc.).
- A bind-group layout descriptor (currently hand-written but mechanical).
- A run-method (today's `run_resident(...)`).
- A `Schedule` value naming the kernel's position relative to others.
- A `Pod` struct for the kernel's input/output types matching WGSL alignment.

The wgsl_bindgen-flavoured pieces (Pod struct, BGL, bind-group builder) are the most
mechanical and least controversial — they're the lowest-risk first deliverable.

### What invariants does the abstraction need to preserve?

From the constitution and existing engine status:
1. **Cross-backend parity.** Whatever the GPU emits, the CPU must produce identically.
   The DSL's CPU emission already does this; the abstraction must not introduce a
   GPU-only path.
2. **Deterministic dispatch order.** Per P3 and P5, two `step_batch(N)` calls with the
   same input must encode the same kernels in the same order. Topological-sort tie-
   breaking must be stable (alphabetical, or insertion-order; not hash-order).
3. **No GPU panics on schema mismatch.** Today the schema-hash protects struct shapes.
   The abstraction should fail-stop at compile (or at minimum at startup), not at
   dispatch-time as a wgpu validation error.
4. **No regression in `step_batch` perf.** Recent perf work (5.4× scoring speedup,
   batch path validated at N=200k agents) raised the bar. The abstraction must not add
   per-tick allocation, BGL rebuilds, or extra encoder splits.
5. **Hot-reload still works** (`.sim` edits without recompile). Argues against
   compile-time codegen for the dispatch part.

### Migration story for the 14 existing kernels

Three plausible paths:

**Big-bang.** Convert all 14 simultaneously. Risky given the perf bar and the spec
requirement that backends stay parity-tested across the migration.
**Strangler.** New kernels (fold family + the 15-25 upcoming) use the abstraction;
existing ones keep their hand-written wrappers. Eventually the new pattern becomes the
norm and old ones get ported opportunistically. Lowest risk; lets the abstraction prove
itself on the easy cases first.
**Family-by-family.** Convert Family A first (8 fold kernels — clear win, low risk),
then Family B (4 per-agent kernels — medium win), then Family C only if the abstraction
is ergonomic enough that it pulls them in.

Strangler is what the codebase's history suggests works — recent migrations (resident
path, scoring K=32 spatial-within, dsl_compiler improvements) have all been incremental.

### Concrete near-term questions to settle in the brainstorm

1. **Scope.** Family A only, A+B, or A+B+C?
2. **Codegen surface.** Does the compiler emit just WGSL+Pod (today plus a bit), or
   WGSL+Pod+BGL+wrapper-struct, or WGSL+Pod+BGL+wrapper+schedule?
3. **Naga dependency.** Use it for reflection, or keep emitting strings and trust the
   layout via convention + tests?
4. **Build-time vs runtime emission.** Build-time wins compile-time validation; runtime
   wins hot-reload. Pick one or layer both?
5. **Where does ownership live.** DSL annotation, sidecar config, or compiler-derived
   from kernel role?
6. **Schedule shape.** Imperative-but-typed (today, just nicer wrappers), phase-tagged
   (β above), or full dataflow (γ)?
7. **Migration path.** Strangler vs family-by-family vs big-bang?
8. **First user.** Almost certainly the 8 fold kernels. The plan should probably ship
   with all 8 working before any existing kernel is touched.

### Smallest sensible deliverable

A "dispatch-emit prototype scoped to the 8 fold kernels" looks like:
- Compiler emits `Fold<ViewName>Kernel` Rust struct + WGSL (already mostly done).
- A shared `FoldDispatcher` runs each one in phase 5 of the schedule, parameterised by
  `ViewStorageSpec`.
- The 8-kernel set is the first user; result is removing ~8× the per-kernel
  boilerplate that would otherwise be hand-written.
- No change to the other 13 kernels.

Even at this scope, the prototype answers most of the questions above — proving (or
disproving) that the abstraction has legs without committing to a full migration.

---

## Appendix: Current Engine Shape

### Inventory of the 14 hand-written kernels

Source: `crates/engine_gpu/src/`, line counts via `wc -l`:

| Kernel | File | LOC | Family | Ownership style |
|---|---|---:|---|---|
| `FusedMaskKernel` | `mask.rs` | 2230 (incl. 3 kernels) | B | per-kernel pool |
| `MaskUnpackKernel` | `mask.rs` | (inside mask.rs) | C | per-kernel pool |
| `FusedAgentUnpackKernel` | `mask.rs` | (inside mask.rs) | C | per-kernel pool |
| `ScoringKernel` | `scoring.rs` | 2795 | B | per-kernel pool + cached view handles |
| `ScoringUnpackKernel` | `scoring.rs` | (inside scoring.rs) | C | per-kernel pool |
| `ApplyActionsKernel` | `apply_actions.rs` | 904 | B | per-kernel pool + cached resident BG |
| `MovementKernel` | `movement.rs` | 823 | B | per-kernel pool + cached resident BG |
| Cascade physics | `physics.rs` + `cascade*.rs` | 3228 + 981 + 1602 | C | shared `CascadeResidentCtx` |
| `GpuSpatialHash` | `spatial_gpu.rs` | 2103 | C | `SpatialOutputs` |
| `ViewStorage` (8 fold kernels via templates) | `view_storage.rs` + `view_storage_*.rs` | 1504 + 574 + 495 | A | shared `ViewStorage` |
| `EventRing` (apply / batch / chronicle / physics) | `event_ring.rs` | 2087 | shared infrastructure | n/a |
| `AliveBitmapKernel` | `alive_bitmap.rs` | 383 | C | per-kernel |

Total `engine_gpu` Rust LOC: **23,646**.

### What kernels share

- **All** use `device.push_error_scope(wgpu::ErrorFilter::Validation)` around shader
  compile, embed the WGSL in the error on failure.
- **All** have a `BindGroupLayoutEntry` builder helper (`storage_rw`, `storage_ro`,
  `uniform`) — copy-pasted across `apply_actions.rs:226`, `movement.rs:188`,
  `mask.rs:575`, etc.
- **All resident-path kernels** maintain a `cached_resident_bg` keyed by caller-supplied
  buffer identities.
- **Most** maintain a `BufferPool` resized via `ensure_pool(agent_cap)`.
- **All resident-path kernels** have a `run_resident(...)` taking
  `(&Device, &Queue, &mut CommandEncoder, agent_cap, ...buffer-handles...)`.
- **Some** also expose a sync `run_and_readback(...)` for parity testing.

### What kernels differ

- **Bind group layouts** vary in core arity (5 for apply_actions, 5+N-views for scoring,
  3 for movement, 3-5 for fold kernels depending on view shape).
- **Workgroup size** is uniformly 64 except for some legacy 1×1 fold variants in fallback
  paths (`view_storage.rs:1043`).
- **Dispatch shape**: most are direct `ceil(agent_cap / 64)`, some are
  `ceil(events / 64)` (fold kernels), one is indirect (cascade physics).
- **Buffer ownership style**, as discussed in §3 — four distinct flavours.

### What's NOT in scope today

- No render-graph-style retained graph data structure. Schedule is hardcoded in
  `step_batch`.
- No `naga` reflection. BGL ↔ WGSL sync is by hand.
- No buffer aliasing. Persistent buffers dominate.
- No async compute / multi-queue. Everything runs on the default queue.
- No GPU-side dispatch generation (no Work Graphs, no DGC). Indirect dispatch is used in
  the cascade only.

### What's already moved partly toward dispatch-emit

The DSL compiler's `emit_*_wgsl` family lives at `crates/dsl_compiler/src/` (~21k LOC):
- `emit_scoring_wgsl.rs` (2193 LOC) — the largest. Includes `emit_kernel`,
  `emit_bindings`, `emit_view_bindings_for_mode`, `scoring_view_binding_order`.
- `emit_view_wgsl.rs` (2234 LOC) — `emit_view_fold_wgsl` and `emit_view_read_wgsl`.
- `emit_mask_wgsl.rs` (1836), `emit_physics_wgsl.rs` (2695).
- `emit_scoring.rs` (1857) emits CPU pick_ability; `emit_pick_ability_wgsl` is in
  `emit_scoring_wgsl.rs` (line 1893).

The compiler already has all the per-kernel structural metadata — `ScoringIR`,
`ViewStorageSpec`, `ScoreEntry`, etc. — that a dispatch-emit abstraction would need to
read. The infrastructure to *generate* Rust wrapper code already exists for the CPU path
(`emit_scoring.rs` emits `pub fn pick_ability(...)`); extending it to GPU wrapper
structs is a strict superset, not a new direction.
