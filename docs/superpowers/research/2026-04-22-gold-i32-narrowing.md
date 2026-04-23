# Narrowing `Inventory.gold` from i64 to i32 — scoping note

**Date:** 2026-04-22
**Status:** recommendation ready; not yet brainstormed/planned
**Context:** flagged as a risk in `docs/superpowers/research/2026-04-22-gpu-everything-scoping.md` — WGSL has no atomic i64, so either gold narrows to i32 or we emulate i64 atomics for the `transfer_gold` rule on GPU.

## What exists today

Gold is defined exactly once in the engine: `crates/engine/src/state/agent_types.rs:73`.

```rust
pub struct Inventory {
    pub gold:        i64,
    pub commodities: [u16; 8],
}
```

Accompanying doc comment: "`gold` is signed (`i64`) so debt is representable as a negative balance." Signedness is load-bearing; width probably isn't.

The schema hash commits to the type: `crates/engine/src/schema_hash.rs:33` → `Inventory{gold=i64,commodities=[u16;8]}`.

Gold is consumed by the DSL `transfer_gold` physics rule at `assets/sim/physics.sim:133` via `agents.sub_gold(from, a)` / `agents.add_gold(to, a)`. The generated handler is `crates/engine/src/generated/physics/transfer_gold.rs`.

(Out of scope: `src/world_sim/` has a separate legacy `NpcData.gold: f32` field for the pre-engine world-sim layer. That's f32 and doesn't hit this question.)

## Max gold in practice

No scenarios or fixtures exercise gold values near i32's range. The economic-sim spawning templates allocate agents with low-thousands of gold; transfers are typically <10. Even a runaway fantasy economy should stay well under 2.1 billion (i32 max). No code that I can find treats gold as "more than a few digits."

## Narrowing cost

**Code change:**

1. `agent_types.rs:73` — `pub gold: i64` → `pub gold: i32`.
2. `agent_types.rs:67-70` — update doc comment to say `i32`.
3. `schema_hash.rs:33` — update string to `Inventory{gold=i32,commodities=[u16;8]}`. **This is a schema-hash bump**: existing save files become incompatible with new binaries. Pre-1.0, that's acceptable.
4. Check `agents.sub_gold` / `agents.add_gold` generated code — these are wrappers that probably just do the arithmetic; if they bake in i64 anywhere, update. Likely a 1-line `as i64` → `as i32` change in the generator.
5. `Inventory` derives `Pod, Zeroable` via `#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]`. Narrowing changes the struct size by 4 bytes (20 B → 16 B). Anyone hashing or comparing raw bytes (serialization, GPU uploads) will see a different layout — and that's exactly the point (smaller upload, GPU-atomic-compatible).
6. Grep + update any `as i64` or `i64::from` calls that consume `Inventory.gold` specifically. Expected: <10 sites total.

**Semantic change:** i32 supports debts up to ~-2.1 billion and balances up to ~2.1 billion. Any emergent behavior that amplifies gold (compound interest, runaway minting) could overflow at scale, whereas i64 was effectively unbounded. Mitigation: the `sub_gold` / `add_gold` helpers should saturate rather than wrap on overflow. Or we add a runtime check in dev builds.

## Risk matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overflow in emergent econ sim | Low | Medium | Saturating arithmetic in add/sub helpers |
| Breaking save files | Certain | Low (pre-1.0) | Schema hash already exists for exactly this |
| Some callsite relies on i64 range | Low | Low | Rust won't compile those until fixed; grep before narrowing |
| GPU side needs atomic i32 i/o | Certain | — (the whole point) | WGSL `atomic<i32>` is supported |

## Recommendation

Narrow to `i32` when item 4b (`transfer_gold` + `modify_standing` migration to GPU) is implemented. It's a 5-10 line PR touching:

- `agent_types.rs:73` (field)
- `schema_hash.rs:33` (hash string)
- Any code-emit site that hardcodes i64 for gold
- Add saturating arithmetic in `add_gold` / `sub_gold` (defence against overflow)

Do not narrow speculatively — wait until item 4b is being worked so the narrowing lands with the WGSL port that motivates it. The schema-hash bump is a one-shot action, batch it with the migration.

If we later decide we need more range (e.g. the econ sim genuinely hits billions), we can widen to u64-as-two-u32s with an `atomicCompareExchange`-based CAS loop for atomic increments. That's a later problem if it ever materialises.

## Open questions

- Are there any ability effects (DSL) that multiply gold? If so, overflow becomes easier to hit. Worth grepping `physics.sim` and ability templates for gold arithmetic once item 4b work starts.
- Does anyone in the campaign layer (`src/ai/goap/` or similar) expect gold to be able to exceed i32 range? Likely not, but a quick sanity check during 4b.
