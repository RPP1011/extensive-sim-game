//! Seeded, resumable RNG — a bit-exact port of `F:\MB\src\core\rng.ts`
//! (mulberry32 over a seed⊕counter stream). The counter is persisted with the
//! save so a loaded game resumes the stream mid-flight. All JS `| 0` /
//! `Math.imul` / `>>> 0` semantics are mod-2^32 arithmetic, which u32
//! wrapping ops reproduce exactly; the final division produces the identical
//! f64. Cross-language pins live in `tests.rs`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngState {
    pub seed: u32,
    /// Persisted draw counter — resuming from (seed, counter) continues the
    /// exact stream. TS stores this as a (possibly negative) i32; the u32 bit
    /// pattern is identical under wrapping arithmetic.
    pub rng_counter: u32,
}

impl RngState {
    pub fn new(seed: u32) -> Self {
        RngState { seed, rng_counter: 0 }
    }
}

fn mulberry32(a: u32) -> f64 {
    let mut t = a.wrapping_add(0x6d2b_79f5);
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    f64::from(t ^ (t >> 14)) / 4_294_967_296.0
}

/// Next float in `[0, 1)`; advances the persisted counter.
pub fn rng_float(r: &mut RngState) -> f64 {
    r.rng_counter = r.rng_counter.wrapping_add(1);
    mulberry32(r.seed ^ r.rng_counter.wrapping_mul(0x9e37_79b9))
}

/// Integer in `[min, max]` inclusive.
pub fn rng_int(r: &mut RngState, min: i64, max: i64) -> i64 {
    min + (rng_float(r) * ((max - min + 1) as f64)).floor() as i64
}

/// Index into a slice of `len` items — the `rngPick` idiom
/// (`arr[Math.floor(rngFloat * arr.length)]`).
pub fn rng_pick_idx(r: &mut RngState, len: usize) -> usize {
    (rng_float(r) * len as f64).floor() as usize
}

/// Pick a reference out of a slice (panics on empty input like the TS would
/// return `undefined` and crash downstream — generation never picks from an
/// empty pool by construction).
pub fn rng_pick<'a, T>(r: &mut RngState, arr: &'a [T]) -> &'a T {
    &arr[rng_pick_idx(r, arr.len())]
}
