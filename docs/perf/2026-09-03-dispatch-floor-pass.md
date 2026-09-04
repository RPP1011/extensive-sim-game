# 2026-09-03 — dispatch-floor pass (webband_colony)

Measured on `webband_colony` (cap 512, 119 live) through Webband's
`webband_bench --mode both`; every step reproduces Webband's recorded
16-day digests bit for bit and the compiler's unit + integration tests and
every fixture's `cargo check` pass. Numbers: 3.8 → 2.9 ms per tick
(265 → 341 ticks/s); host encode 1.3 → 0.75 ms; schedule 136 → 123 stages.

The tick was LATENCY-bound: ~136 kernel dispatches plus ~300 tiny
transfer commands per tick, and the chronicle ring is empty on the median
tick. What changed, and where:

| change | where | effect |
|---|---|---|
| Event-ring sort: 15 kernels (five dispatched over the 1M-slot ring CAPACITY) + a 46 MB copy-back → ONE single-workgroup stable LSD radix dispatch, identical permutation | `cg/emit/sort_kernel.rs`, `build_helper.rs` (`run_radix_sort`) | −14 dispatches, −46 MB copy per tick |
| One cfg uniform buffer per fixture (256-byte slot per kernel, host mirror `cfg_shadow`, one `write_buffer` per tick); kernels bind a `BufferBinding` slice; `pub const CFG_OFF_<kernel>` | `build_helper.rs`, `kernel_lowerings.rs`, `emit/program.rs`, `tom_probe_runtime` | −~200 queued writes |
| ViewFold consumers read `event_tail[0u]` from a binding pointed at `prev_event_tail_buf` instead of a cfg word filled by a copy | `build_helper.rs` (`view_fold_prev_tail_kernel_names`) | −43 copies |
| One command encoder / one submit per tick; tail reset is an in-encoder copy from `pending_tail_buf` | `build_helper.rs` step() preamble | −1 submit |
| Bind groups cached per kernel across ticks, keyed on (buffer, offset, size) | `engine::gpu::cached_bind_group`, `emit/program.rs` | host |
| All mask bitmaps in one buffer (`mask_bitmaps_buf`, 256-aligned slots, sliced bindings) → one `clear_buffer` | `build_helper.rs`, `kernel_binding_ir.rs` (`is_mask_bitmap_binding`), lowerings | −44 clears |
| Scheduler: ready-set walk over an ordering DAG (Bernstein conflicts on projected handles, ring drain = write); ONLY view folds and view decays may move, every other op kind is a barrier; like-with-like tie-break | `cg/schedule/fusion.rs` (`OrderingDag`) | decays gather |
| Plain (ungated, unpacked) decays over distinct views fuse into one `decays_<a>_to_<b>` kernel with per-member `cfg.slot_count_<i>` | `cg/schedule/fusion.rs`, `emit/kernel.rs` (`ViewDecayFused`), `build_helper.rs` | 15 → 2 dispatches |
| Pair-keyed f32 serial-scan folds run one thread per observer ROW (`// fold-rows` marker, dispatch by rows, cfg keeps the slot domain) | `emit/wgsl_body.rs`, `emit/kernel.rs`, `build_helper.rs` | ~5× on those kernels |
| Spatial scatter: single-thread serial loop → one 256-thread workgroup, block-rank stable placement (same in-cell order), no one-shot guard | `emit/kernel.rs` (`SPATIAL_BUILD_HASH_SCATTER_BODY`), `emit/spatial.rs`, `emit/program.rs` | ~4× on that kernel |

Why physics rules do NOT move in the scheduler: with them reorderable, a
per-agent rule that sums thought views was hoisted above the folds that
fill them and the fixture digest drifted while the campaign/log digests did
not — the derived read set of a physics rule is not a complete description
of its body. Folds and decays declare their storage explicitly.

Kernel ids inside the chronicle `seq` trailer are dense over PRODUCER
kernels in schedule order; moving non-producers (folds, decays) leaves them
unchanged, which is why the reorder is digest-neutral.

Not done, in order of expected value on this fixture:

1. A per-tick compact live-agent index list for every full-population scan
   (`physics_Steer` alone is ~17% of GPU time and scales with cap, not with
   the live count; `scoring` carries 44 such loops).
2. Dead bodies out of the spatial grid (needs every walk body alive-gated;
   the emitted neighbourhood walks are, `FilteredWalk` is not audited).
3. Cross-view fold fusion (the emitter's fold binding layout is per view).

Latent, unchanged, pinned by digests: per-event consumer kernels dispatch
`agent_cap` threads, so ring records past slot `agent_cap` on a busy tick
are never consumed.
