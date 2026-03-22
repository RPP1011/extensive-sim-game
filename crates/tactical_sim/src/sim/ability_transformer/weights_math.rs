//! Math utility functions for transformer weight inference.

// ---------------------------------------------------------------------------
// SIMD-friendly dot product -- auto-vectorized by LLVM
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Process in chunks of 8 for autovectorization
    let n = a.len();
    let chunks = n / 8;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    for c in 0..chunks {
        let base = c * 8;
        sum0 += a[base] * b[base] + a[base + 1] * b[base + 1];
        sum1 += a[base + 2] * b[base + 2] + a[base + 3] * b[base + 3];
        sum2 += a[base + 4] * b[base + 4] + a[base + 5] * b[base + 5];
        sum3 += a[base + 6] * b[base + 6] + a[base + 7] * b[base + 7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-10.0, 10.0)).exp())
}

#[inline]
pub(super) fn gelu(x: f32) -> f32 {
    // Approximation: x * s(1.702 * x)
    x * sigmoid(1.702 * x)
}

pub(super) fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
