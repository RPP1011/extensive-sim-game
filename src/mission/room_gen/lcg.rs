/// LCG pseudo-random number generator (no external crates needed).
pub(crate) struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        // Mix the seed so that seed=0 still gives a useful sequence.
        let s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut lcg = Self(s);
        // Warm up the generator.
        for _ in 0..8 {
            lcg.next_u64();
        }
        lcg
    }

    /// Advance and return the next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        // Knuth's multiplicative LCG (modulus 2^64).
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Return a value in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
    }

    /// Return a value in [lo, hi).
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Return a usize in [lo, hi] (inclusive).
    pub fn next_usize_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        let range = (hi - lo + 1) as u64;
        lo + (self.next_u64() % range) as usize
    }
}

// ---------------------------------------------------------------------------
// Internal geometry records (used for both nav and visuals)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct ObstacleRegion {
    pub col0: usize,
    pub col1: usize,
    pub row0: usize,
    pub row1: usize,
    /// Visual height in world units (0.5-2.0).
    pub height: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct RampRegion {
    pub col0: usize,
    pub col1: usize,
    pub row0: usize,
    pub row1: usize,
    /// Elevation delta at the high end (0.5-1.5 m).
    pub elevation: f32,
}
