//! Deterministic 2D value noise + fbm — a bit-exact port of
//! `F:\MB\src\core\noise.ts`. Worldgen samples `fbm` through `world_field`
//! for terrain-aware landmark placement; castgen's look pass uses
//! `id_seed`/`feature_roll` (the busts' headwear-preference hash).
//!
//! JS `Math.imul(x, c)` first coerces via ToInt32 (truncate toward zero, then
//! wrap mod 2^32) — `to_i32` reproduces that for the f64 inputs `hash` takes.

/// ToInt32 for the domain this crate feeds it (|x| well below 2^63).
fn to_i32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    (x.trunc() as i64) as u32 as i32
}

/// Integer-lattice hash in `[0, 1]` (divisor 2^32 − 1, per the TS).
pub fn hash(x: f64, y: f64) -> f64 {
    let xi = to_i32(x) as u32;
    let yi = to_i32(y) as u32;
    let mut h = xi
        .wrapping_mul(374_761_393)
        .wrapping_add(yi.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    f64::from(h) / 4_294_967_295.0
}

/// Deterministic string→seed for per-character feature rolls (headwear).
/// Iterates UTF-16 code units exactly as JS `charCodeAt` does.
pub fn id_seed(id: &str) -> u32 {
    let mut h: u32 = 7;
    for cu in id.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(cu));
    }
    h
}

/// Deterministic feature roll in `[0, 1)` for a given seed + salt.
pub fn feature_roll(seed: u32, salt: u32) -> f64 {
    hash(f64::from(seed), f64::from(salt) * 2_654_435_761.0)
}

fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

pub fn value_noise(x: f64, y: f64) -> f64 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let a = hash(xi, yi);
    let b = hash(xi + 1.0, yi);
    let c = hash(xi, yi + 1.0);
    let d = hash(xi + 1.0, yi + 1.0);
    let u = smooth(xf);
    let v = smooth(yf);
    a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

/// Fractal noise in `[0, 1]` (default octaves 4, lacunarity 2, gain 0.5).
pub fn fbm(x: f64, y: f64, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut fx = x;
    let mut fy = y;
    for _ in 0..octaves {
        sum += value_noise(fx, fy) * amp;
        norm += amp;
        amp *= 0.5;
        fx *= 2.0;
        fy *= 2.0;
    }
    sum / norm
}
