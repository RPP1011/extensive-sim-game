//! webband_bench — tick-rate scaling harness for the Webband port
//! (perf slice, 2026-07-22).
//!
//! Answers two DIFFERENT questions that the port's own notes keep
//! separate, and so does this harness:
//!
//!   (a) AGENT CAP — `try_new(seed, cap)`. The cap sizes every per-agent
//!       kernel's dispatch domain AND the pair-keyed belief storage
//!       (`cap * cap * 4` bytes each, folded + decayed at `cap*cap`
//!       threads every tick). Live population is untouched.
//!   (b) LIVE POPULATION — how many colonists actually work. Dialled at
//!       runtime by writing `agent_alive_buf` (the S6 roster-pool idiom),
//!       so `webband_bench.sim` (500 colonists spawned) serves the whole
//!       sweep from one build.
//!
//! Methodology follows docs/perf/2026-05-09-stress-ceilings.md:
//!   * warmup ticks are stripped (tick 0 pays the GPU pipeline-cache
//!     miss; the perf doc's p99 column is polluted by exactly that);
//!   * `--mode latency` (default) times `step()` + `device.poll(Wait)`
//!     per tick — the full host encoder build + submit + GPU + poll
//!     round trip, i.e. what a real-time game loop pays;
//!   * `--mode pipelined` submits N ticks and polls ONCE at the end,
//!     which measures the host-side encoder+submit cost per tick with
//!     the GPU overlapped. latency − pipelined ≈ the exposed GPU time.
//!   * `try_new` wall is reported SEPARATELY (shader compilation, ~1
//!     minute on this fixture because of the AoE dispatcher) and never
//!     folded into per-tick numbers.
//!
//! Usage:
//!   cargo run --release -p sims --example webband_bench -- \
//!       <fixture> <cap> <ticks> [--pop N] [--seed X] [--warmup N]
//!       [--mode latency|pipelined|both] [--raid] [--ring] [--digest]
//!
//! Fixtures: webband_colony (production), webband_bench (500-colonist
//! copy), webband_bench_nopair (bench minus the pair beliefs).

use std::time::Instant;

const DEFAULT_SEED: u64 = 0xC0_10_11_5E_ED;
const CT_COLONIST: u32 = 0;
const CT_RAIDER: u32 = 13;
const CT_WARLORD: u32 = 14;
const TICKS_PER_DAY: f64 = 600.0;

struct Args {
    fixture: String,
    cap: u32,
    ticks: usize,
    pop: Option<usize>,
    seed: u64,
    warmup: usize,
    mode: String,
    raid: bool,
    ring: bool,
    digest: bool,
    kernels: bool,
    dump: Option<String>,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let mut out = Args {
        fixture: a.get(1).cloned().unwrap_or_else(|| "webband_colony".into()),
        cap: a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512),
        ticks: a.get(3).and_then(|s| s.parse().ok()).unwrap_or(300),
        pop: None,
        seed: DEFAULT_SEED,
        warmup: 20,
        mode: "latency".into(),
        raid: false,
        ring: false,
        digest: false,
        kernels: false,
        dump: None,
    };
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--pop" => {
                out.pop = a.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--seed" => {
                out.seed = a
                    .get(i + 1)
                    .and_then(|s| {
                        s.strip_prefix("0x")
                            .map(|h| u64::from_str_radix(h, 16).ok())
                            .unwrap_or_else(|| s.parse().ok())
                    })
                    .unwrap_or(DEFAULT_SEED);
                i += 1;
            }
            "--warmup" => {
                out.warmup = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
                i += 1;
            }
            "--mode" => {
                out.mode = a.get(i + 1).cloned().unwrap_or_else(|| "latency".into());
                i += 1;
            }
            "--dump" => {
                out.dump = a.get(i + 1).cloned();
                i += 1;
            }
            "--raid" => out.raid = true,
            "--ring" => out.ring = true,
            "--digest" => out.digest = true,
            "--kernels" => out.kernels = true,
            _ => {}
        }
        i += 1;
    }
    out
}

fn pct(sorted: &[u128], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx] as f64 / 1000.0 // µs
}

/// Everything a run reports. Printed as one JSON line so a sweep can be
/// concatenated into NDJSON like the stress fixtures' output.
#[derive(Default)]
struct Report {
    fixture: String,
    cap: u32,
    live: usize,
    colonists: usize,
    ticks: usize,
    try_new_ms: f64,
    median_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
    min_us: f64,
    pipelined_us: f64,
    first_tick_ms: f64,
    // Tick CLASSES. The fixture's event volume is not uniform: the
    // jostle/brawl storm fires on `tick % 20 == 10` and the dawn fold on
    // `tick % 600 == 0`. Those are the ticks where the chronicle ring is
    // deep, and therefore the ticks where every serial-scan fold (the
    // pair-keyed beliefs above all) actually does its O(cells x events)
    // work. Separating them is the difference between "the median tick"
    // and "the tick the player feels as a hitch".
    // The perf doc's host-vs-GPU split, measured directly: `step()`
    // returns as soon as the encoder is built and submitted (the
    // generated step never polls), so timing step() alone — with the
    // queue drained by the PREVIOUS tick's poll — is the host-side
    // encoder+write_buffer cost, and the poll that follows is the
    // exposed GPU time.
    host_us: f64,
    gpu_us: f64,
    ordinary_us: f64,
    storm_us: f64,
    dawn_us: f64,
    ring_max: u32,
    ring_median: u32,
    digest: u64,
    pair_bytes: u64,
}

impl Report {
    fn print(&self) {
        let tps = if self.median_us > 0.0 { 1e6 / self.median_us } else { 0.0 };
        println!(
            "{{\"fixture\":\"{}\",\"cap\":{},\"live_agents\":{},\"colonists\":{},\
             \"ticks\":{},\"try_new_ms\":{:.0},\"median_us\":{:.1},\"p95_us\":{:.1},\
             \"p99_us\":{:.1},\"mean_us\":{:.1},\"min_us\":{:.1},\"pipelined_us\":{:.1},\
             \"host_us\":{:.1},\"gpu_us\":{:.1},\"first_tick_ms\":{:.0},\"ticks_per_s\":{:.1},\"days_per_s\":{:.3},\"ring_max\":{},\
             \"ring_median\":{},\"ordinary_us\":{:.1},\"storm_us\":{:.1},\"dawn_us\":{:.1},\
             \"pair_bytes\":{},\"digest\":\"{:#018x}\"}}",
            self.fixture,
            self.cap,
            self.live,
            self.colonists,
            self.ticks,
            self.try_new_ms,
            self.median_us,
            self.p95_us,
            self.p99_us,
            self.mean_us,
            self.min_us,
            self.pipelined_us,
            self.host_us,
            self.gpu_us,
            self.first_tick_ms,
            tps,
            tps / TICKS_PER_DAY,
            self.ring_max,
            self.ring_median,
            self.ordinary_us,
            self.storm_us,
            self.dawn_us,
            self.pair_bytes,
            self.digest,
        );
    }
}

macro_rules! bench_fixture {
    ($modpath:path, $args:expr) => {{
        use $modpath as fx;
        let args: &Args = $args;
        let mut rep = Report {
            fixture: args.fixture.clone(),
            cap: args.cap,
            ticks: args.ticks,
            ..Default::default()
        };

        let t0 = Instant::now();
        let Some(mut rt) = fx::GeneratedRuntime::try_new(args.seed, args.cap) else {
            eprintln!("[bench] try_new failed (no adapter?)");
            std::process::exit(2);
        };
        rep.try_new_ms = t0.elapsed().as_secs_f64() * 1000.0;
        {
            let info = rt.gpu.adapter.get_info();
            let lim = rt.gpu.device.limits();
            eprintln!(
                "[bench] adapter: {} ({:?}, {:?}) driver={} | max_buffer_size={}                  max_workgroups_per_dim={}",
                info.name, info.device_type, info.backend, info.driver,
                lim.max_buffer_size, lim.max_compute_workgroups_per_dimension,
            );
        }

        let n = rt.agent_count as usize;
        // Read the spawn layout back so population dialling is
        // discovered, never assumed (the cast/pool layout is fixture
        // data, and the two bench copies differ from the production
        // fixture only in colonist count).
        let types = read_u32(&rt, &rt.agent_creature_type_buf.clone(), n);
        let alive0 = read_u32(&rt, &rt.agent_alive_buf.clone(), n);
        let colonist_slots: Vec<usize> = (0..n)
            .filter(|&i| alive0[i] != 0 && types[i] == CT_COLONIST)
            .collect();
        let raid_slots: Vec<usize> = (0..n)
            .filter(|&i| alive0[i] != 0 && (types[i] == CT_RAIDER || types[i] == CT_WARLORD))
            .collect();

        // (b) LIVE POPULATION: deactivate colonist slots past `pop`.
        if let Some(pop) = args.pop {
            for &s in colonist_slots.iter().skip(pop) {
                rt.gpu
                    .queue
                    .write_buffer(&rt.agent_alive_buf, (s as u64) * 4, bytemuck::cast_slice(&[0u32]));
            }
        }
        // Optional: muster the whole raid pool so the combat verbs +
        // the AoE ability dispatcher are actually exercised.
        if args.raid {
            let muster = 612u32; // dawn+12, the fixture's own convention
            for (i, &r) in raid_slots.iter().enumerate() {
                rt.gpu.queue.write_buffer(
                    &rt.agent_musters_at_buf,
                    (r as u64) * 4,
                    bytemuck::cast_slice(&[muster]),
                );
                let x = -9.0 + 18.0 * (i as f32) / (raid_slots.len().max(2) - 1) as f32;
                let y = 24.0 + 2.0 * ((i % 3) as f32);
                rt.gpu.queue.write_buffer(
                    &rt.agent_pos_buf,
                    (r as u64) * 16,
                    bytemuck::cast_slice(&[x, y, 0.0f32, 0.0f32]),
                );
            }
        }

        let live_after = read_u32(&rt, &rt.agent_alive_buf.clone(), n);
        rep.live = live_after.iter().filter(|&&v| v != 0).count();
        rep.colonists = (0..n)
            .filter(|&i| live_after[i] != 0 && types[i] == CT_COLONIST)
            .count();
        rep.pair_bytes = (args.cap as u64) * (args.cap as u64) * 4;

        // THE FIRST TICK pays every kernel's lazy shader compile
        // (`cache.get_or_insert_with` in the dispatch helpers) — that is
        // where this fixture's AoE-dispatcher compile lands, NOT in
        // try_new. Timed on its own and never folded into per-tick numbers.
        let tf = Instant::now();
        rt.step();
        let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
        rep.first_tick_ms = tf.elapsed().as_secs_f64() * 1000.0;

        // If a raid is staged, fast-forward to the muster before timing
        // so the measured window is the FIGHT, not the quiet morning.
        let skip = if args.raid { 640usize } else { 0 };
        for _ in 0..skip {
            rt.step();
        }
        let _ = rt.gpu.device.poll(wgpu::PollType::Wait);

        // Warmup — strip the pipeline-cache miss (perf doc's advice).
        for _ in 0..args.warmup {
            rt.step();
            let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
        }

        // --- mode: latency (step + poll per tick) ---
        if args.mode != "pipelined" {
            let mut samples: Vec<u128> = Vec::with_capacity(args.ticks);
            let mut ordinary: Vec<u128> = Vec::new();
            let mut storm: Vec<u128> = Vec::new();
            let mut dawn: Vec<u128> = Vec::new();
            let mut host: Vec<u128> = Vec::new();
            let mut gpu: Vec<u128> = Vec::new();
            let dump = args.dump.clone();
            let mut trace: Vec<(u64, u128, u128)> = Vec::new();
            for _ in 0..args.ticks {
                // `rt.tick` is the tick ABOUT to be computed.
                let tk = rt.tick;
                let t = Instant::now();
                rt.step();
                let submitted = t.elapsed().as_nanos();
                let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
                let ns = t.elapsed().as_nanos();
                host.push(submitted);
                gpu.push(ns - submitted);
                if dump.is_some() {
                    trace.push((tk, ns, submitted));
                }
                samples.push(ns);
                if tk % 600 == 0 {
                    dawn.push(ns);
                } else if tk % 20 == 10 {
                    storm.push(ns);
                } else {
                    ordinary.push(ns);
                }
            }
            if let Some(path) = dump {
                use std::io::Write as _;
                let mut f = std::fs::File::create(&path).expect("dump file");
                for (t, ns, sub) in &trace {
                    writeln!(f, "{},{},{}", t, ns, sub).ok();
                }
            }
            for (v, slot) in [
                (&mut host, &mut rep.host_us),
                (&mut gpu, &mut rep.gpu_us),
                (&mut ordinary, &mut rep.ordinary_us),
                (&mut storm, &mut rep.storm_us),
                (&mut dawn, &mut rep.dawn_us),
            ] {
                v.sort_unstable();
                *slot = pct(v, 0.50);
            }
            let sum: u128 = samples.iter().sum();
            rep.mean_us = sum as f64 / samples.len() as f64 / 1000.0;
            samples.sort_unstable();
            rep.median_us = pct(&samples, 0.50);
            rep.p95_us = pct(&samples, 0.95);
            rep.p99_us = pct(&samples, 0.99);
            rep.min_us = samples[0] as f64 / 1000.0;
        }

        // --- mode: pipelined (submit N, poll once) ---
        if args.mode == "pipelined" || args.mode == "both" {
            let t = Instant::now();
            for _ in 0..args.ticks {
                rt.step();
            }
            let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
            rep.pipelined_us = t.elapsed().as_secs_f64() * 1e6 / args.ticks as f64;
        }

        // --- capacity instrumentation: the chronicle ring high-water ---
        if args.ring {
            let mut tails: Vec<u32> = Vec::new();
            for _ in 0..60 {
                rt.step();
                let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
                tails.push(read_one_u32(&rt, &rt.event_ring.tail().clone()));
            }
            rep.ring_max = tails.iter().copied().max().unwrap_or(0);
            tails.sort_unstable();
            rep.ring_median = tails[tails.len() / 2];
        }

        // --- did the raid actually happen? (never claim combat cost
        // without proving combat ran) ---
        if args.raid {
            let al = read_u32(&rt, &rt.agent_alive_buf.clone(), n);
            let dn = read_u32(&rt, &rt.agent_downed_buf.clone(), n);
            let ra = read_u32(&rt, &rt.agent_raid_active_buf.clone(), n);
            let raiders_alive = (0..n)
                .filter(|&i| al[i] != 0 && (types[i] == CT_RAIDER || types[i] == CT_WARLORD))
                .count();
            let downed = (0..n).filter(|&i| types[i] == CT_COLONIST && dn[i] != 0).count();
            let engaged = (0..n).filter(|&i| ra[i] == 1).count();
            eprintln!(
                "[bench] raid check: raiders {} -> {} alive, colonists downed {}, raid_active(1) {}",
                raid_slots.len(), raiders_alive, downed, engaged
            );
        }

        // --- determinism digest over the state a run ends on ---
        if args.digest {
            let mut h: u64 = 0xcbf29ce484222325;
            let bufs: Vec<(&str, wgpu::Buffer)> = vec![
                ("hp", rt.agent_hp_buf.clone()),
                ("mood", rt.agent_mood_buf.clone()),
                ("need_food", rt.agent_need_food_buf.clone()),
                ("pos", rt.agent_pos_buf.clone()),
                ("claimed_job", rt.agent_claimed_job_buf.clone()),
                ("alive", rt.agent_alive_buf.clone()),
                ("inv_timber", rt.agent_inv_timber_buf.clone()),
                ("inv_meal", rt.agent_inv_meal_buf.clone()),
            ];
            for (_, b) in &bufs {
                let words = read_u32(&rt, b, n);
                for w in words {
                    h ^= w as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            rep.digest = h;
        }
        rep
    }};
}

fn main() {
    // Generated runtimes hold large stack values at construction
    // (S1 report: windows-gnu's 2 MiB default overflows in wgpu init).
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

/// PER-KERNEL ATTRIBUTION — the compiler's D1-D4 `DebugTimings` facility,
/// which had no call site in any generated runtime until 2026-07-22. Only
/// `webband_bench` is lowered at D1+ (it declares `debug { depth: kernel }`),
/// so this is a fixture-specific routine rather than part of the generic
/// bench macro: at D0 the generated runtime carries no timing API at all,
/// which is exactly the "costs nothing when off" property.
///
/// Any other fixture can be raised for one build with `SIM_DEBUG_DEPTH=3`.
/// The query set itself is only allocated when the PROCESS also sets
/// `SIM_KERNEL_TIMINGS=1`.
fn kernel_table(args: &Args) {
    use sims::webband_bench as fx;
    let Some(mut rt) = fx::GeneratedRuntime::try_new(args.seed, args.cap) else {
        eprintln!("[bench] try_new failed (no adapter?)");
        std::process::exit(2);
    };
    if !rt.kernel_timings_enabled() {
        eprintln!(
            "[bench] --kernels: timings unavailable — set SIM_KERNEL_TIMINGS=1              (and the adapter must expose TIMESTAMP_QUERY)"
        );
        std::process::exit(3);
    }
    let n = rt.agent_count as usize;
    let types = read_u32(&rt, &rt.agent_creature_type_buf.clone(), n);
    let alive0 = read_u32(&rt, &rt.agent_alive_buf.clone(), n);
    let colonist_slots: Vec<usize> = (0..n)
        .filter(|&i| alive0[i] != 0 && types[i] == CT_COLONIST)
        .collect();
    if let Some(pop) = args.pop {
        for &s in colonist_slots.iter().skip(pop) {
            rt.gpu
                .queue
                .write_buffer(&rt.agent_alive_buf, (s as u64) * 4, bytemuck::cast_slice(&[0u32]));
        }
    }
    rt.step();
    let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
    for _ in 0..args.warmup {
        rt.step();
        let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
    }
    // Accumulate over the window so one tick's scheduling noise does not
    // decide the table. The window spans whole storm cycles (%20) on purpose.
    let mut acc: std::collections::BTreeMap<String, (u64, u64)> = std::collections::BTreeMap::new();
    let window = args.ticks.max(20);
    for _ in 0..window {
        rt.step();
        let _ = rt.gpu.device.poll(wgpu::PollType::Wait);
        for t in rt.kernel_timings() {
            let e = acc.entry(t.kernel).or_insert((0, 0));
            e.0 += t.wall_ns;
            e.1 += 1;
        }
    }
    let mut rows: Vec<(String, u64, u64)> = acc
        .into_iter()
        .map(|(k, (ns, cnt))| (k, ns / cnt.max(1), cnt))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = rows.iter().map(|r| r.1).sum();
    println!(
        "[kernels] fixture=webband_bench cap={} pop={:?} ticks={} kernels_timed={}          mean_gpu_ns_per_tick={}",
        args.cap,
        args.pop,
        window,
        rows.len(),
        total
    );
    for (k, ns, cnt) in rows.iter().take(30) {
        println!(
            "[kernels] {:<54} {:>9} ns  {:>5.1}%  (dispatched {}/{} ticks)",
            k,
            ns,
            100.0 * (*ns as f64) / (total.max(1) as f64),
            cnt,
            window
        );
    }
}

fn run() {
    let args = parse_args();
    if args.kernels {
        kernel_table(&args);
        return;
    }
    eprintln!(
        "[bench] fixture={} cap={} ticks={} pop={:?} mode={} raid={} seed={:#x}",
        args.fixture, args.cap, args.ticks, args.pop, args.mode, args.raid, args.seed
    );
    let rep = match args.fixture.as_str() {
        "webband_colony" => bench_fixture!(sims::webband_colony, &args),
        "webband_bench" => bench_fixture!(sims::webband_bench, &args),
        "webband_bench_nopair" => bench_fixture!(sims::webband_bench_nopair, &args),
        other => {
            eprintln!("[bench] unknown fixture {other:?}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "[bench] DONE try_new {:.1}s first_tick {:.1}s | median {:.0} µs = {:.0} ticks/s = {:.2} days/s | \
         p95 {:.0} µs | pipelined {:.0} µs | live {} ({} colonists)",
        rep.try_new_ms / 1000.0,
        rep.first_tick_ms / 1000.0,
        rep.median_us,
        if rep.median_us > 0.0 { 1e6 / rep.median_us } else { 0.0 },
        if rep.median_us > 0.0 { 1e6 / rep.median_us / TICKS_PER_DAY } else { 0.0 },
        rep.p95_us,
        rep.pipelined_us,
        rep.live,
        rep.colonists,
    );
    rep.print();
}

// ---------------------------------------------------------------------
// Readback helpers. Generic over the generated runtime types by taking
// the `GpuContext` directly — every generated runtime exposes `pub gpu`.
// ---------------------------------------------------------------------

trait HasGpu {
    fn gpu(&self) -> &engine::GpuContext;
}
impl HasGpu for sims::webband_colony::GeneratedRuntime {
    fn gpu(&self) -> &engine::GpuContext {
        &self.gpu
    }
}
impl HasGpu for sims::webband_bench::GeneratedRuntime {
    fn gpu(&self) -> &engine::GpuContext {
        &self.gpu
    }
}
impl HasGpu for sims::webband_bench_nopair::GeneratedRuntime {
    fn gpu(&self) -> &engine::GpuContext {
        &self.gpu
    }
}

fn read_u32<R: HasGpu>(rt: &R, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let gpu = rt.gpu();
    // Staging is always >= 16 B (wgpu's MAP_ALIGNMENT); the COPY is
    // clamped to what the source actually holds (the event-ring tail is
    // a 4-byte buffer).
    let want = ((count as u64) * 4).max(16);
    let copy_bytes = want.min(buf.size()) & !3;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_bench::u32_staging"),
        size: want,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bytes = want;
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, copy_bytes);
    gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        let take = count.min(words.len());
        let mut v = words[..take].to_vec();
        v.resize(count, 0);
        v
    };
    staging.unmap();
    out
}

fn read_one_u32<R: HasGpu>(rt: &R, buf: &wgpu::Buffer) -> u32 {
    read_u32(rt, buf, 4)[0]
}
