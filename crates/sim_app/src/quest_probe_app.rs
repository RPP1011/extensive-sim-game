//! Quest probe harness — drives `quest_probe_runtime` for N ticks and
//! reports the observed per-Adventurer `progress` view.
//!
//! ## Discovery surfaces
//!
//! Two surfaces are documented in spec but never exercised by any
//! prior fixture (full report:
//! `docs/superpowers/notes/2026-05-04-quest_probe.md`):
//!
//!   1. `entity X : Quest { ... }` — the parser today only accepts
//!      `Agent`, `Item`, `Group` as entity roots
//!      (`crates/dsl_ast/src/parser.rs:374-385`). The fixture
//!      includes a commented-out `entity Mission : Quest` line that
//!      would parse-fail; falls back to `entity Mission : Item`
//!      (Item is fully wired) so the rest of the program compiles.
//!   2. `quests.*` namespace — the resolver routes `quests` through
//!      `NamespaceId::Quests` (`resolve.rs:117`) but the CG namespace
//!      registry (`cg/lower/driver.rs:672`) registers ZERO methods.
//!      Any call falls through to `LoweringError::UnsupportedName
//!      spaceCall`. The fixture includes a commented-out `quests.is_
//!      active(0u)` line for reference.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — `+= 1u` on a u32 view accumulates correctly
//!
//! With AGENT_COUNT=32, TICKS=100: every alive Adventurer emits one
//! ProgressTick per tick → `progress[N] = TICKS = 100u` for every N.
//! This shape would require the WGSL emitter to route `+=` on a
//! u32 view through `atomicAdd(&storage[idx], rhs)` (commutative +
//! associative — same P11-trivial guarantee as `atomicOr`).
//!
//! ### (b) GAP CONFIRMED — `+= 1u` silently lowers to `atomicOr`
//!
//! The current emitter branches solely on the view's result type
//! (`crates/dsl_compiler/src/cg/emit/wgsl_body.rs:1326-1338`):
//! `u32` → `atomicOr`; otherwise → CAS+add. The OPERATOR is ignored
//! by the emit-time check — `+=` and `|=` produce identical WGSL.
//! With rhs = `1u` constant, `atomicOr` is idempotent, so per-slot
//! value stays at `1u` after any number of fires.
//!
//! Predicted: `progress[N] = 1u` for every N. The harness's OUTCOME
//! line classifies the observed pattern.

use engine::CompiledSim;
use quest_probe_runtime::QuestProbeState;

const SEED: u64 = 0x9E5_71_71_5E_E5_DA_DA;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;

fn main() {
    let mut sim = QuestProbeState::new(SEED, AGENT_COUNT);
    println!(
        "quest_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let progress = sim.progress().to_vec();
    println!(
        "quest_probe_app: finished — final tick={} agents={} progress.len()={}",
        sim.tick(),
        sim.agent_count(),
        progress.len(),
    );

    let (min, max, sum, zero_slots, ones_slots, ticks_slots) = stats(&progress, TICKS as u32);
    let nonzero_slots = progress.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (progress.len().max(1) as f32);
    let mean = (sum as f64) / (progress.len().max(1) as f64);

    println!(
        "quest_probe_app: progress readback — min={} mean={:.3} max={} sum={}",
        min, mean, max, sum,
    );
    println!(
        "quest_probe_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        progress.len(),
        observed_fraction * 100.0,
    );
    println!(
        "quest_probe_app: predicted shapes — \
         (a) FULL FIRE: progress[N] = {} for every N (atomicAdd-style); \
         (b) GAP: progress[N] = 1 for every N (atomicOr idempotent)",
        TICKS,
    );
    println!(
        "quest_probe_app: ones_slots = {}/{} (would be 100% under GAP); \
         ticks_slots = {}/{} (would be 100% under FULL FIRE)",
        ones_slots,
        progress.len(),
        ticks_slots,
        progress.len(),
    );

    let preview = progress.iter().take(8).copied().collect::<Vec<_>>();
    println!("quest_probe_app: preview progress[0..8] = {:?}", preview);

    // OUTCOME classification.
    if ticks_slots == progress.len() {
        println!(
            "quest_probe_app: OUTCOME = (a) FULL FIRE — every slot accumulated to TICKS={}.\n  \
             This means `+= 1u` on a u32 view DOES route through atomicAdd-style accumulation \
             — the gap predicted in the discovery doc was already closed. Update \
             docs/superpowers/notes/2026-05-04-quest_probe.md to reflect the new state.",
            TICKS,
        );
    } else if ones_slots == progress.len() {
        println!(
            "quest_probe_app: OUTCOME = (b) GAP CONFIRMED — every slot stuck at 1u.\n  \
             `self += 1u` on a `u32`-result view silently lowered to `atomicOr(&storage[idx], 1u)` \
             per the result-type-only branch in cg/emit/wgsl_body.rs:1326-1338. The OR is \
             idempotent, so per-slot value stays at 1u regardless of how many ProgressTick \
             events the fold consumed. See docs/superpowers/notes/2026-05-04-quest_probe.md.",
        );
    } else if max == 0 {
        println!(
            "quest_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0. The producer \
             rule's emit never reached the fold storage. Likely gap: the per-event tag-filter \
             dropped every event, OR the event ring's tail clear isn't sequenced before the \
             fold dispatch.",
        );
    } else {
        println!(
            "quest_probe_app: OUTCOME = (b) PARTIAL — neither pure-1u nor pure-TICKS. \
             min={min} max={max} sum={sum}; investigate per-slot pattern.",
        );
    }
}

/// Returns `(min, max, sum, zero_slots, ones_slots, ticks_slots)`.
/// `ticks_target` is the per-slot value under the OPERATOR-INTENT
/// (full fire) shape — caller passes `TICKS as u32`.
fn stats(v: &[u32], ticks_target: u32) -> (u32, u32, u64, usize, usize, usize) {
    let mut min = u32::MAX;
    let mut max = 0u32;
    let mut sum: u64 = 0;
    let mut zero = 0usize;
    let mut ones = 0usize;
    let mut full = 0usize;
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as u64;
        if x == 0 {
            zero += 1;
        }
        if x == 1 {
            ones += 1;
        }
        if x == ticks_target {
            full += 1;
        }
    }
    if v.is_empty() {
        min = 0;
    }
    (min, max, sum, zero, ones, full)
}
