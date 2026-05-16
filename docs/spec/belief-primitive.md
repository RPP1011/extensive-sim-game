# Belief Primitive — Maintenance Cheatsheet

> Reference for the `belief <name>(observer: Agent[, subject: Agent]) -> T { ... }` DSL surface. Plan I shipped this primitive across 10 slices + polish — this doc captures the current shape so contributors can extend it without re-deriving the architecture.

## Surface

```
[ @dispatch(per_agent_event_scan) ]
[ @decay(rate = <f32>, per = tick) ]
[ @per_entity_ring(K = <int>) ]
[ @belief_gated ]
belief <name>(observer: Agent [, subject: Agent]) -> <T> {
  initial: <expr>,
  [ on <Event> { <pattern> } [where <expr>] { <body> } … ]
  [ on <Event> { <pattern> } [where <expr>] merge from <ident>: <op> … ]
  [ clamp: [<lo>, <hi>], ]
}
```

**Param shapes.** First param must be `Agent`. Second param (optional) is `Agent` (pair-keyed) or one of `u8|u32|i32` (key-typed second param — deferred, surfaces `LoweringError::UnsupportedBeliefShape` with slice pointer I.3b).

**Return types.** `bool`, `u8`, `u32`, `i32`, `f32`, or a registered struct entity (for struct-cell ring storage with `@per_entity_ring`).

**Merge ops.** `bit_or`, `max`, `min`, `replace` — all four wired end-to-end with the matching WGSL atomic primitive.

## Storage shapes

| Signature | Inferred storage | Buffer size | Test |
|---|---|---|---|
| `(observer: Agent) -> T` | PairMap (collapses to single-key sizing) | N cells | `room_known_pattern_probe_pin` |
| `(observer: Agent, subject: Agent) -> T` | PairMap | N² cells | `belief_merge_propagation_probe_pin` |
| `(observer: Agent[, subject: Agent]) -> T` + `@per_entity_ring(K=…)` | PerEntityRing | N × K cells | `threats_struct_probe_pos_keyed_pin` |
| `(observer: Agent, key: u32)` | Not yet | — | (I.3b deferred) |

The lowering infers storage at `crates/dsl_compiler/src/cg/lower/view.rs::infer_belief_storage_hint`.

## End-to-end pipeline

```
.sim source
  ↓ parser (dsl_ast/src/parser.rs::belief_decl)
Decl::Belief(BeliefDecl)
  ↓ resolver (dsl_ast/src/resolve.rs Pass-1 + Pass-2)
ViewIR { kind: ViewKind::Belief, social_merges: Vec<SocialMergeHandler> }
  ↓ lowering (dsl_compiler/src/cg/lower/view.rs::lower_view's Belief arm)
ComputeOpKind::ViewFold {…}  (one per propagation handler)
ComputeOpKind::BeliefSocialMerge { view, on_event, op }  (one per `merge from` clause)
  ↓ classifier + binding builder (dsl_compiler/src/cg/emit/kernel.rs)
KernelKindClass::BeliefSocialMerge { view_name, view_id, on_event_kind_id, op }
  ↓ WGSL emit (build_belief_social_merge_wgsl_body)
WGSL kernel with: bounds check, kind filter, source_agent read, per-cell merge loop
  ↓ runtime
GPU dispatch — atomicOr/Max/Min/Store into view_storage_<view>_primary_buf
```

## Test pin map

Compiler-layer (`crates/dsl_compiler/tests/`):
- `belief_lower_pair_map.rs` — 7 tests covering pair-keyed + single-key lowering, op variants, view-namespace sharing, BeliefSocialMerge IR shape
- `belief_smoke_probe.rs` — full pipeline + naga validation on the merge kernel WGSL
- `belief_migration_pin.rs` — `view beliefs_flags` → `belief beliefs_flags` migration across 4 fixtures (tom_probe, dungeon_horde, dungeon_stealth, plague_city) keeps `ViewKind::Belief` post-resolve
- `belief_fuzz_round_trip.rs` — 10K random valid `.sim` snippets with belief decls all parse + resolve + lower
- `ability_grammar_walker_lower.rs` — 9 lowering tests for the tree-walker's emitted ability shapes

GPU runtime (`crates/sims/tests/`):
- `belief_smoke_probe_pin.rs` — merge kernel dispatches cleanly at runtime; naga-validated WGSL body
- `belief_merge_propagation_probe_pin.rs` — pair-keyed `bit_or` merge on AllyDied: every receiver inherits agent 0's belief row
- `room_known_pattern_probe_pin.rs` — single-key `bit_or` (the dungeon_horde gossip pattern stand-in): every hero inherits hero 0's room knowledge
- `belief_merge_ops_probe_pin.rs` — `max`/`min`/`replace` ops verified with known inputs and computed expected outputs

Threats migration (`crates/sims/tests/threat_*.rs`):
- All 5 threats fixtures (`threats_view_probe`, `threats_with_decay_probe`, `threat_stresstest`, `dodger_probe`, `threats_struct_probe`) now use the `belief threats(…)` keyword.
- Behavioural pins (`threat_stresstest_pin`, `threats_struct_probe_pos_keyed_pin`, `tom_probe_belief_gated_threat_pin`) green.

Multi-horizon stresstest (`crates/sims/tests/threat_horizon_stresstest_pin.rs`):
- Mixed long-fuse + short-fuse threat sources coexist in one sim; effectiveness report with crossover-tick + sustained-Flee-% metric.

## Known limitations / future polish

| Item | Status | Path forward |
|---|---|---|
| `i32` / `u8` return types | Grammar-valid, lowering-rejected on type-mismatch | DSL literal-suffix surface needs `1i` / `1u8` parser support |
| Key-typed second param (`(Agent, u32)`) | Surfaces `UnsupportedBeliefShape` | Slice I.3b — needs SingleKey-extended storage variant sized `agent_cap × key_pop` |
| IR-level source-agent field offset lookup | Hardcoded offset 2 (works for single-Agent-field events like `AllyDied { dead: Agent }`) | Compute from `social_merge.source_agent: LocalRef` + event field layout |
| Plan I.6 viewer migration | Pattern probe shipped (`room_known_pattern_probe`); full dungeon_horde viewer rewrite deferred | Replace `hero_known_rooms: [u64; 5]` host field with GPU readback when the migration value justifies the Plan E hook bypass |

## Adding a new merge op

1. Extend `dsl_ast::ir::MergeOp` enum + `SocialMergeOpName` parser keyword.
2. Add discriminant + atomic primitive in `cg/emit/kernel.rs::build_belief_social_merge_wgsl_body`'s `atomic_op` match.
3. Add a fuzz row in `belief_fuzz_round_trip.rs` so the round-trip exercises the new op.
4. Add a runtime e2e test row in `belief_merge_ops_probe.sim` + pin.

The 4-op set today (`bit_or`, `max`, `min`, `replace`) covers the four idempotent + commutative atomic primitives WGSL exposes. Adding an op outside that set requires designing a new dispatch shape (no-longer-trivial-commutative ops need different concurrency handling).
