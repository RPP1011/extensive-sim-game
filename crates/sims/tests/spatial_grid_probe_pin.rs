//! `spatial_grid_probe` pin — regression coverage for the uniform
//! spatial grid behind every `spatial.<query>(self)` walk
//! (`crates/dsl_compiler/src/cg/emit/spatial.rs` + the walk emitters in
//! `cg/emit/wgsl_body.rs` / `cg/emit/kernel.rs`).
//!
//! Every output of `assets/sim/spatial_grid_probe.sim` is checked
//! against a CPU reference, per agent, bit-for-bit:
//!
//! * **Declared radius honoured** (`@spatial(radius = R)`): the fold and
//!   body forms see exactly the agents with `|d|² <= R²` — including a
//!   watched agent 25 units away (4 grid cells), which the old fixed
//!   27-cell walk could not reach, and nothing beyond `R`.
//! * **Dense cells**: 101 agents share one grid cell (> MAX_PER_CELL =
//!   32). The un-annotated query runs through the tiled PerCell kernel
//!   (the path that used to truncate each neighbour cell at 32); every
//!   one of them must be counted.
//! * **No duplicate visits**: every `seen_*` output is a sum of 1.0 over
//!   a unique-agent predicate, for agents in the world interior, at the
//!   world edge (±63) and beyond it (±100, positions clamp into edge
//!   cells). Before the fix the edge cells were walked up to 4× (clamped
//!   neighbour offsets) and, from the second tick on, stale grid entries
//!   were re-read because the per-cell counts were never reset.
//! * **Determinism**: two same-seed runs are byte-identical.
//!
//! Positions are integers with `(p + 64) % 6 != 0` so no position lies
//! exactly on a cell boundary (GPU f32 division is not required to be
//! correctly rounded, so a boundary position could land in either cell);
//! radii are x.5 so no pair sits exactly on a radius boundary.

use sims::spatial_grid_probe::GeneratedRuntime;

const SEED: u64 = 0x5EA7_1A1_6121D;
const TICKS: u32 = 4;
const R_FAR: f32 = 30.5;
const R_NEAR: f32 = 2.5;
const NO_WATCH: u32 = 0xFFFF_FFFF;

// Mirrors `crates/dsl_compiler/src/cg/emit/spatial.rs`.
const CELL_SIZE: f32 = 6.0;
const HALF_EXTENT: f32 = 64.0;
const GRID_DIM: i32 = 22;
/// Loose OOB-pool gate of the one-ring (un-annotated) body walk:
/// `3 * (r+1)^2 * CELL_SIZE^2` with r = 1.
const LEGACY_GATE_SQ: f32 = 3.0 * 4.0 * CELL_SIZE * CELL_SIZE;

struct Layout {
    pos: Vec<[f32; 3]>,
    tag: Vec<u32>,
    watch: Vec<u32>,
}

/// Deterministic LCG for the scattered part of the layout (no RNG crate).
fn lcg(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// Nudge an integer coordinate off an exact cell boundary.
fn off_boundary(v: i32) -> i32 {
    if (v + 64).rem_euclid(6) == 0 { v + 1 } else { v }
}

fn build_layout() -> Layout {
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut watch_slot: Vec<Option<usize>> = Vec::new();
    let mut push = |p: [i32; 2], w: Option<usize>, pos: &mut Vec<[f32; 3]>| {
        pos.push([off_boundary(p[0]) as f32, off_boundary(p[1]) as f32, 0.0]);
        watch_slot.push(w);
    };

    // (a) Radius larger than the old 1-ring reach: slot 0 watches slot 1,
    //     25 units away (cells 10 and 14 on x).
    push([0, 0], Some(1), &mut pos);
    push([25, 0], Some(0), &mut pos);

    // (b) Dense cell: 101 agents in cell (11, 11, 10) (world x,y in 3..=5).
    //     Slot 2 watches the LAST of them (well past slot 32 of the cell).
    let dense_start = pos.len();
    for k in 0..101usize {
        let x = 3 + (k % 3) as i32;
        let y = 3 + ((k / 3) % 3) as i32;
        let w = if k == 0 { Some(dense_start + 100) } else { Some(dense_start + (k * 7) % 101) };
        push([x, y], w, &mut pos);
    }

    // (c) World edge and beyond: pairs that watch each other.
    for &(a, b) in &[
        ([63, 63], [62, 61]),     // corner cell (21, 21)
        ([-63, -63], [-61, -62]), // corner cell (0, 0)
        ([100, 100], [99, 97]),   // both clamp into corner (21, 21)
        ([-90, 10], [-80, 12]),   // both clamp into column 0
        ([63, 0], [70, 1]),       // one in-world edge, one beyond
        ([0, -100], [5, -95]),    // clamp into row 0
    ] {
        let i = pos.len();
        push(a, Some(i + 1), &mut pos);
        push(b, Some(i), &mut pos);
    }

    // (d) Scatter across ±110 so totals exercise cells everywhere,
    //     including beyond the ±64 world extent.
    let mut s = 0x1234_5678_9abc_def0u64;
    let scatter_start = pos.len();
    for k in 0..80usize {
        let x = (lcg(&mut s) % 221) as i32 - 110;
        let y = (lcg(&mut s) % 221) as i32 - 110;
        let w = if k % 5 == 4 { None } else { Some(scatter_start + (lcg(&mut s) as usize % 80)) };
        push([x, y], w, &mut pos);
    }

    let n = pos.len();
    let tag: Vec<u32> = (0..n as u32).map(|i| 1000 + i * 3).collect();
    let watch: Vec<u32> = watch_slot.iter().map(|w| w.map_or(NO_WATCH, |j| tag[j])).collect();
    Layout { pos, tag, watch }
}

fn cell_coord(v: f32) -> i32 {
    let f = (v + HALF_EXTENT) / CELL_SIZE;
    (f.max(0.0) as i32).clamp(0, GRID_DIM - 1)
}

fn cell_of(p: &[f32; 3]) -> [i32; 3] {
    [cell_coord(p[0]), cell_coord(p[1]), cell_coord(p[2])]
}

fn dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// One-ring window membership (distinct in-grid cells around self's
/// clamped cell — no cell counted twice).
fn in_legacy_window(a: &[f32; 3], b: &[f32; 3]) -> bool {
    let (ca, cb) = (cell_of(a), cell_of(b));
    (0..3).all(|k| (ca[k] - cb[k]).abs() <= 1)
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Outputs {
    seen_far: Vec<f32>,
    total_far: Vec<f32>,
    seen_near: Vec<f32>,
    total_near: Vec<f32>,
    seen_body: Vec<f32>,
    total_body: Vec<f32>,
    seen_legacy: Vec<f32>,
    total_legacy: Vec<f32>,
    seen_legacy_body: Vec<f32>,
}

fn expected(l: &Layout) -> Outputs {
    let n = l.pos.len();
    let mut o = Outputs::default();
    for i in 0..n {
        let p = &l.pos[i];
        let (mut sf, mut tf, mut sn, mut tn, mut sl, mut tl, mut slb) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for j in 0..n {
            let q = &l.pos[j];
            let d2 = dist_sq(p, q);
            let watched = l.tag[j] == l.watch[i];
            if d2 <= R_FAR * R_FAR {
                tf += 1.0;
                if watched { sf += 1.0; }
            }
            if d2 <= R_NEAR * R_NEAR {
                tn += 1.0;
                if watched { sn += 1.0; }
            }
            if in_legacy_window(p, q) {
                // Tiled PerCell fold: no distance gate.
                tl += 1.0;
                if watched { sl += 1.0; }
                // Per-agent body walk: loose OOB-pool gate.
                if watched && d2 <= LEGACY_GATE_SQ { slb += 1.0; }
            }
        }
        o.seen_far.push(sf);
        o.total_far.push(tf);
        o.seen_near.push(sn);
        o.total_near.push(tn);
        o.seen_body.push(sf);
        o.total_body.push(tf);
        o.seen_legacy.push(sl);
        o.total_legacy.push(tl);
        o.seen_legacy_body.push(slb);
    }
    o
}

fn run(l: &Layout) -> Option<Outputs> {
    let n = l.pos.len() as u32;
    let mut state = GeneratedRuntime::try_new(SEED, n)?;
    let pos4: Vec<[f32; 4]> = l.pos.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    let alive = vec![1u32; n as usize];
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&pos4));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    state.gpu.queue.write_buffer(&state.agent_tag_buf, 0, bytemuck::cast_slice(&l.tag));
    state.gpu.queue.write_buffer(&state.agent_watch_buf, 0, bytemuck::cast_slice(&l.watch));
    for _ in 0..TICKS {
        state.step();
    }
    let n = n as usize;
    let mut rd = |pick: fn(&GeneratedRuntime) -> &wgpu::Buffer| {
        let buf = pick(&state).clone();
        read_f32(&mut state, &buf, n)
    };
    Some(Outputs {
        seen_far: rd(|s| &s.agent_seen_far_buf),
        total_far: rd(|s| &s.agent_total_far_buf),
        seen_near: rd(|s| &s.agent_seen_near_buf),
        total_near: rd(|s| &s.agent_total_near_buf),
        seen_body: rd(|s| &s.agent_seen_body_buf),
        total_body: rd(|s| &s.agent_total_body_buf),
        seen_legacy: rd(|s| &s.agent_seen_legacy_buf),
        total_legacy: rd(|s| &s.agent_total_legacy_buf),
        seen_legacy_body: rd(|s| &s.agent_seen_legacy_body_buf),
    })
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial_grid_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spatial_grid_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

/// Compare one output column; report every mismatching agent.
fn check(name: &str, got: &[f32], want: &[f32], l: &Layout, errors: &mut Vec<String>) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            errors.push(format!(
                "{name}[{i}] = {g} (expected {w}); pos = {:?}, cell = {:?}",
                l.pos[i],
                cell_of(&l.pos[i])
            ));
        }
    }
}

#[test]
fn spatial_grid_walks_honour_radius_density_and_edges() {
    let layout = build_layout();
    let Some(got) = run(&layout) else {
        eprintln!("[spatial_grid_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let want = expected(&layout);

    // Sanity on the reference itself: the scenario-specific claims hold.
    assert_eq!(want.seen_far[0], 1.0, "reference: slot 1 (25u away) is within R_FAR of slot 0");
    assert_eq!(want.seen_legacy[0], 0.0, "reference: slot 1 is outside slot 0's one-ring window");
    assert!(want.total_legacy[2] >= 101.0, "reference: dense cell holds >= 101 agents");
    for (name, col) in [
        ("seen_far", &want.seen_far),
        ("seen_near", &want.seen_near),
        ("seen_legacy", &want.seen_legacy),
        ("seen_legacy_body", &want.seen_legacy_body),
    ] {
        assert!(col.iter().all(|&v| v == 0.0 || v == 1.0), "reference {name} must be 0/1");
    }

    let mut errors = Vec::new();
    check("seen_far", &got.seen_far, &want.seen_far, &layout, &mut errors);
    check("total_far", &got.total_far, &want.total_far, &layout, &mut errors);
    check("seen_near", &got.seen_near, &want.seen_near, &layout, &mut errors);
    check("total_near", &got.total_near, &want.total_near, &layout, &mut errors);
    check("seen_body", &got.seen_body, &want.seen_body, &layout, &mut errors);
    check("total_body", &got.total_body, &want.total_body, &layout, &mut errors);
    check("seen_legacy", &got.seen_legacy, &want.seen_legacy, &layout, &mut errors);
    check("total_legacy", &got.total_legacy, &want.total_legacy, &layout, &mut errors);
    check("seen_legacy_body", &got.seen_legacy_body, &want.seen_legacy_body, &layout, &mut errors);
    if !errors.is_empty() {
        let limit = std::env::var("SPATIAL_PROBE_SHOW").ok().and_then(|v| v.parse().ok()).unwrap_or(40usize);
        let shown: Vec<&String> = errors.iter().take(limit).collect();
        panic!(
            "{} spatial-grid mismatches (first {} shown):\n{}",
            errors.len(),
            shown.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }

    // Determinism: a second same-seed run is byte-identical.
    let again = run(&layout).expect("adapter present on first run");
    assert!(got == again, "same-seed reruns diverged");
    println!(
        "spatial_grid_probe: {} agents x {} ticks match the CPU reference; dense-cell total = {}",
        layout.pos.len(),
        TICKS,
        got.total_legacy[2]
    );
}

/// Perf bench (not a correctness pin): ms/tick of the probe fixture with
/// `n` agents scattered uniformly over ±100 x/y (z = 0) — the game-scale
/// world, larger than the grid's ±64 extent. The fixture runs two
/// declared-radius folds pairs (R = 30.5 and R = 2.5), one declared-radius
/// body walk (R = 30.5) and two one-ring walks per agent per tick.
///
/// `cargo test -p sims --release --test spatial_grid_probe_pin -- --ignored --nocapture`
#[test]
#[ignore = "perf bench; run explicitly with --ignored --nocapture"]
fn spatial_grid_probe_bench() {
    for &n in &[1000u32, 4000, 16000] {
        let mut s = 0xBE4C_0000_0000_0001u64 ^ n as u64;
        let pos4: Vec<[f32; 4]> = (0..n)
            .map(|_| {
                let x = (lcg(&mut s) % 20001) as f32 * 0.01 - 100.0;
                let y = (lcg(&mut s) % 20001) as f32 * 0.01 - 100.0;
                [x, y, 0.0, 0.0]
            })
            .collect();
        let tag: Vec<u32> = (0..n).collect();
        let watch: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % n).collect();
        let Some(mut state) = GeneratedRuntime::try_new(SEED, n) else {
            eprintln!("[spatial_grid_probe] skipping bench: no wgpu adapter on host.");
            return;
        };
        state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&pos4));
        state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n as usize]));
        state.gpu.queue.write_buffer(&state.agent_tag_buf, 0, bytemuck::cast_slice(&tag));
        state.gpu.queue.write_buffer(&state.agent_watch_buf, 0, bytemuck::cast_slice(&watch));
        for _ in 0..10 {
            state.step();
        }
        state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
        const T: u32 = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..T {
            state.step();
        }
        state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / T as f64;
        println!("spatial_grid_probe bench: n = {n:>6}  {ms:8.3} ms/tick");
    }
}
