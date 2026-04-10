/// Murmur3-based deterministic hash using only u32 arithmetic so the same
/// algorithm can be ported byte-for-byte to GLSL (no int64 extension needed).
/// Uses the standard Murmur3 fmix32 finalizer for full avalanche.
#[inline]
pub fn hash_u32(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed.wrapping_add(0x9e3779b9); // break fixed-point at zero
    h = h.wrapping_add(x as u32).wrapping_mul(0x9e3779b9);
    h = h.wrapping_add(y as u32).wrapping_mul(0x85ebca6b);
    h = h.wrapping_add(z as u32).wrapping_mul(0xc2b2ae35);
    // Murmur3 fmix32 finalizer
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// u32 hash → f32 in [0, 1).
#[inline]
pub fn hash_u32_to_f32(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    hash_u32(x, y, z, seed) as f32 / u32::MAX as f32
}

/// Deterministic 3D hash → u32. Ported from voxel.rs.
pub fn hash_3d(x: i32, y: i32, z: i32, seed: u64) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(x as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(y as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(z as u64);
    h = h ^ (h >> 33);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h = h ^ (h >> 33);
    (h >> 32) as u32
}

/// Hash to float in [0, 1).
pub fn hash_f32(x: i32, y: i32, z: i32, seed: u64) -> f32 {
    hash_3d(x, y, z, seed) as f32 / u32::MAX as f32
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 2D value noise with smoothstep interpolation. Returns [0, 1].
pub fn value_noise_2d(x: f32, y: f32, seed: u64, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let fx = smoothstep(sx - sx.floor());
    let fy = smoothstep(sy - sy.floor());
    let s32 = seed as u32 ^ (seed >> 32) as u32;
    let h00 = hash_u32_to_f32(ix, iy, 0, s32);
    let h10 = hash_u32_to_f32(ix + 1, iy, 0, s32);
    let h01 = hash_u32_to_f32(ix, iy + 1, 0, s32);
    let h11 = hash_u32_to_f32(ix + 1, iy + 1, 0, s32);
    let a = h00 + (h10 - h00) * fx;
    let b = h01 + (h11 - h01) * fx;
    a + (b - a) * fy
}

/// 3D value noise with trilinear smoothstep. Returns [0, 1].
pub fn value_noise_3d(x: f32, y: f32, z: f32, seed: u64, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let sz = z / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let iz = sz.floor() as i32;
    let fx = smoothstep(sx - sx.floor());
    let fy = smoothstep(sy - sy.floor());
    let fz = smoothstep(sz - sz.floor());
    let s32 = seed as u32 ^ (seed >> 32) as u32;
    let c000 = hash_u32_to_f32(ix, iy, iz, s32);
    let c100 = hash_u32_to_f32(ix + 1, iy, iz, s32);
    let c010 = hash_u32_to_f32(ix, iy + 1, iz, s32);
    let c110 = hash_u32_to_f32(ix + 1, iy + 1, iz, s32);
    let c001 = hash_u32_to_f32(ix, iy, iz + 1, s32);
    let c101 = hash_u32_to_f32(ix + 1, iy, iz + 1, s32);
    let c011 = hash_u32_to_f32(ix, iy + 1, iz + 1, s32);
    let c111 = hash_u32_to_f32(ix + 1, iy + 1, iz + 1, s32);
    let a0 = c000 + (c100 - c000) * fx;
    let b0 = c010 + (c110 - c010) * fx;
    let a1 = c001 + (c101 - c001) * fx;
    let b1 = c011 + (c111 - c011) * fx;
    let c0 = a0 + (b0 - a0) * fy;
    let c1 = a1 + (b1 - a1) * fy;
    c0 + (c1 - c0) * fz
}

/// 2D Fractal Brownian Motion. Returns [0, 1].
pub fn fbm_2d(x: f32, y: f32, seed: u64, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut max_amp = 0.0f32;
    for i in 0..octaves {
        sum += amp * value_noise_2d(x * freq, y * freq, seed.wrapping_add(i as u64 * 31337), 1.0);
        max_amp += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_amp
}

/// 3D Fractal Brownian Motion. Returns [0, 1].
pub fn fbm_3d(x: f32, y: f32, z: f32, seed: u64, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut max_amp = 0.0f32;
    for i in 0..octaves {
        let s = seed.wrapping_add(i as u64 * 31337);
        sum += amp * value_noise_3d(x * freq, y * freq, z * freq, s, 1.0);
        max_amp += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_amp
}

/// Worm cave test: returns true if position should be carved.
pub fn worm_cave(x: f32, y: f32, z: f32, seed_a: u64, seed_b: u64, threshold: f32) -> bool {
    let a = value_noise_3d(x, y, z, seed_a, 16.0);
    let b = value_noise_3d(x, y, z, seed_b, 16.0);
    (a - 0.5).abs() < threshold && (b - 0.5).abs() < threshold
}

/// Domain warp: offset input coordinates by noise for organic shapes.
pub fn domain_warp_2d(x: f32, y: f32, seed: u64, scale: f32, strength: f32) -> (f32, f32) {
    let wx = value_noise_2d(x, y, seed, scale) * 2.0 - 1.0;
    let wy = value_noise_2d(x, y, seed.wrapping_add(77777), scale) * 2.0 - 1.0;
    (x + wx * strength, y + wy * strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        assert_eq!(hash_3d(10, 20, 30, 42), hash_3d(10, 20, 30, 42));
        assert_ne!(hash_3d(10, 20, 30, 42), hash_3d(11, 20, 30, 42));
    }

    #[test]
    fn hash_f32_in_range() {
        for i in 0..100 {
            let v = hash_f32(i, i * 7, 0, 999);
            assert!(v >= 0.0 && v < 1.0, "hash_f32 out of range: {v}");
        }
    }

    #[test]
    fn value_noise_2d_smooth() {
        let a = value_noise_2d(100.0, 100.0, 42, 16.0);
        let b = value_noise_2d(101.0, 100.0, 42, 16.0);
        assert!((a - b).abs() < 0.2, "value_noise_2d not smooth: {a} vs {b}");
    }

    #[test]
    fn value_noise_3d_smooth() {
        let a = value_noise_3d(100.0, 100.0, 100.0, 42, 16.0);
        let b = value_noise_3d(101.0, 100.0, 100.0, 42, 16.0);
        assert!((a - b).abs() < 0.2, "value_noise_3d not smooth: {a} vs {b}");
    }

    #[test]
    fn fbm_2d_in_range() {
        for i in 0..50 {
            let v = fbm_2d(i as f32 * 10.0, i as f32 * 7.0, 42, 5, 2.0, 0.5);
            assert!(v >= 0.0 && v <= 1.0, "fbm_2d out of range: {v}");
        }
    }

    #[test]
    fn fbm_3d_in_range() {
        for i in 0..50 {
            let v = fbm_3d(i as f32 * 10.0, 0.0, i as f32 * 7.0, 42, 4, 2.0, 0.5);
            assert!(v >= 0.0 && v <= 1.0, "fbm_3d out of range: {v}");
        }
    }

    #[test]
    fn hash_u32_deterministic_and_uniform() {
        // Determinism: same input → same output
        assert_eq!(hash_u32(10, 20, 30, 42), hash_u32(10, 20, 30, 42));

        // Origin is not a fixed point
        assert_ne!(hash_u32(0, 0, 0, 0), 0, "hash_u32(0,0,0,0) must not be zero");

        // Different inputs → different outputs (10 different perturbations)
        let base = hash_u32(100, 200, 300, 42);
        for &(dx, dy, dz) in &[(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1),(2,0,0),(0,2,0),(7,11,13),(1,1,1)] {
            assert_ne!(hash_u32(100 + dx, 200 + dy, 300 + dz, 42), base,
                "perturbation ({},{},{}) collided with base", dx, dy, dz);
        }

        // Seed independence: different seeds → different outputs
        assert_ne!(hash_u32(10, 20, 30, 0), hash_u32(10, 20, 30, 1));
        assert_ne!(hash_u32(0, 0, 0, 1), hash_u32(0, 0, 0, 2));

        // Avalanche: changing one input bit should flip ~16 output bits on average
        // (i.e. each output bit has ~50% chance of flipping). Sample 256 pairs.
        let mut total_diff_bits = 0u32;
        let n_pairs = 256;
        for i in 0..n_pairs {
            let a = hash_u32(i, i * 13, i * 7, 999);
            let b = hash_u32(i + 1, i * 13, i * 7, 999); // adjacent in x
            total_diff_bits += (a ^ b).count_ones();
        }
        let avg_flipped = total_diff_bits as f32 / n_pairs as f32;
        assert!(avg_flipped > 12.0 && avg_flipped < 20.0,
            "avalanche poor: avg {} bits flipped per adjacent input (expected ~16)",
            avg_flipped);

        // Distribution: mean of 10K samples should be near 0.5
        let mut sum = 0.0f64;
        let n = 10_000;
        for i in 0..n {
            sum += hash_u32_to_f32(i, i * 7, i / 3, 999) as f64;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.05, "hash_u32 distribution skewed: mean={mean}");
    }

    #[test]
    fn worm_noise_produces_caves() {
        let mut cave_count = 0;
        for x in 0..100 {
            for y in 0..100 {
                if worm_cave(x as f32, y as f32, 50.0, 42, 55555, 0.06) {
                    cave_count += 1;
                }
            }
        }
        assert!(cave_count > 0, "worm_noise produced no caves");
        assert!(cave_count < 5000, "worm_noise produced too many caves: {cave_count}");
    }
}
