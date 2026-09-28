//! Deterministic contention resolution (`@serial(...)` chronicle
//! consumers) — end-to-end GPU pins for the two contention patterns
//! game fixtures need:
//!
//! 1. EXCLUSIVE RESERVATION — `assets/sim/contention_claim_probe.sim`:
//!    ~1000 claimants contend for 10 targets EVERY tick. Exactly one
//!    claim per target wins (highest priority, ties lowest claimant id);
//!    every loser takes the consumer's `else` branch.
//! 2. ORDER-INDEPENDENT ACCUMULATION + GUARDED DECREMENT —
//!    `assets/sim/contention_stock_probe.sim`: ~1000 eaters hit ONE
//!    holder every tick; an f32 ledger accumulates a non-integral share
//!    per event and a u32 stock is decremented only while >= 1 remains
//!    ("who gets the last unit" is decided by hunger desc, id asc).
//!
//! Each pin checks the GPU result against a host-side SERIAL reference
//! model of the documented `@serial` semantics, bit for bit, on every
//! run; re-runs the scenario several times in-process and compares state
//! hashes; and re-runs it in a second PROCESS (this test binary
//! re-invoked on the `*_child_hash` test) and compares hashes again.
//!
//! Both fixtures also carry an un-annotated CONTROL consumer (the same
//! body, default parallel apply) writing mirrored `_u` fields. The
//! control is reported (how far it strays from the serial reference,
//! and whether it diverged across runs) but never asserted on — it is
//! the scheduling-dependent behaviour `@serial` exists to replace.
//!
//! GPU-backed: skips (passes silently) when no wgpu adapter exists.

use std::collections::BTreeMap;

const CLAIM_AGENTS: u32 = 1024;
const CLAIM_TARGETS: u32 = 10; // config.cp.targets default
const CLAIM_TICKS: u32 = 40;
const STOCK_AGENTS: u32 = 1024;
const STOCK_INIT: u32 = 2000; // init stock
const STOCK_TICKS: u32 = 6;
const RUNS: usize = 3;
const SEED: u64 = 0xC0_47E4_7105;

// ---------------------------------------------------------------------------
// GPU readback helpers (generic over any fixture runtime's buffers)
// ---------------------------------------------------------------------------

fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contention_probe::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("contention_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback map_async failed"));
    device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn fnv1a(fields: &BTreeMap<&'static str, Vec<u32>>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, words) in fields {
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for w in words {
            for b in w.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

fn count_mismatches(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

// ---------------------------------------------------------------------------
// Scenario 1 — exclusive reservation
// ---------------------------------------------------------------------------

struct ClaimRun {
    serial: BTreeMap<&'static str, Vec<u32>>,
    control: BTreeMap<&'static str, Vec<u32>>,
    step_ms: f64,
}

fn run_claim() -> Option<ClaimRun> {
    use sims::contention_claim_probe::GeneratedRuntime;
    let mut st = GeneratedRuntime::try_new(SEED, CLAIM_AGENTS)?;
    let t0 = std::time::Instant::now();
    for _ in 0..CLAIM_TICKS {
        st.step();
    }
    // Drain the queue so the timing covers the GPU work too.
    st.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let step_ms = t0.elapsed().as_secs_f64() * 1e3 / CLAIM_TICKS as f64;
    let n = CLAIM_AGENTS as usize;
    let mut serial = BTreeMap::new();
    serial.insert("claim_tick", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_claim_tick_buf, n));
    serial.insert("claimed_by", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_claimed_by_buf, n));
    serial.insert("wins", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_wins_buf, n));
    serial.insert("losses", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_losses_buf, n));
    serial.insert("last_win", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_last_win_buf, n));
    let mut control = BTreeMap::new();
    control.insert("claim_tick", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_claim_tick_u_buf, n));
    control.insert("claimed_by", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_claimed_by_u_buf, n));
    control.insert("wins", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_wins_u_buf, n));
    control.insert("losses", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_losses_u_buf, n));
    Some(ClaimRun { serial, control, step_ms })
}

/// Host reference: the documented `@serial(key = t, order = [p, w],
/// desc = [p])` semantics, applied literally.
fn claim_reference() -> BTreeMap<&'static str, Vec<u32>> {
    let n = CLAIM_AGENTS as usize;
    let t_count = CLAIM_TARGETS;
    let mut claim_tick = vec![0u32; n];
    let mut claimed_by = vec![0u32; n];
    let mut wins = vec![0u32; n];
    let mut losses = vec![0u32; n];
    let mut last_win = vec![0u32; n];
    for tick in 0..CLAIM_TICKS {
        // verb Claim: unique argmax target (c + x + tick) % T == T - 1.
        let mut groups: Vec<Vec<(u32, u32)>> = vec![Vec::new(); t_count as usize];
        for c in t_count..CLAIM_AGENTS {
            let x = (2 * t_count - 1 - (c + tick) % t_count) % t_count;
            let p = (c * 7 + tick) % 13;
            groups[x as usize].push((p, c));
        }
        for (x, g) in groups.iter_mut().enumerate() {
            g.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            for &(_p, c) in g.iter() {
                if claim_tick[x] != tick + 1 {
                    claim_tick[x] = tick + 1;
                    claimed_by[x] = c;
                    wins[c as usize] += 1;
                    last_win[c as usize] = x as u32 + 1;
                } else {
                    losses[c as usize] += 1;
                }
            }
        }
    }
    let mut m = BTreeMap::new();
    m.insert("claim_tick", claim_tick);
    m.insert("claimed_by", claimed_by);
    m.insert("wins", wins);
    m.insert("losses", losses);
    m.insert("last_win", last_win);
    m
}

// ---------------------------------------------------------------------------
// Scenario 2 — accumulation + guarded decrement on one holder
// ---------------------------------------------------------------------------

struct StockRun {
    serial: BTreeMap<&'static str, Vec<u32>>,
    control: BTreeMap<&'static str, Vec<u32>>,
    step_ms: f64,
}

fn run_stock() -> Option<StockRun> {
    use sims::contention_stock_probe::GeneratedRuntime;
    let mut st = GeneratedRuntime::try_new(SEED, STOCK_AGENTS)?;
    let t0 = std::time::Instant::now();
    for _ in 0..STOCK_TICKS {
        st.step();
    }
    st.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let step_ms = t0.elapsed().as_secs_f64() * 1e3 / STOCK_TICKS as f64;
    let n = STOCK_AGENTS as usize;
    let mut serial = BTreeMap::new();
    serial.insert("stock", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_stock_buf, n));
    serial.insert("ledger", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_ledger_buf, n));
    serial.insert("fed", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_fed_buf, n));
    serial.insert("missed", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_missed_buf, n));
    let mut control = BTreeMap::new();
    control.insert("stock", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_stock_u_buf, n));
    control.insert("ledger", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_ledger_u_buf, n));
    control.insert("fed", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_fed_u_buf, n));
    control.insert("missed", read_words(&st.gpu.device, &st.gpu.queue, &st.agent_missed_u_buf, n));
    Some(StockRun { serial, control, step_ms })
}

/// Host reference: `@serial(key = t, order = [h, w], desc = [h])`.
fn stock_reference() -> BTreeMap<&'static str, Vec<u32>> {
    let n = STOCK_AGENTS as usize;
    let mut stock = vec![STOCK_INIT; n];
    let mut ledger = vec![0.0f32; n];
    let mut fed = vec![0u32; n];
    let mut missed = vec![0u32; n];
    for tick in 0..STOCK_TICKS {
        let mut evs: Vec<(u32, u32, f32)> = (1..STOCK_AGENTS)
            .map(|e| {
                let h = (e * 11 + tick) % 17;
                let s = 0.1f32 * ((1 + (e + tick) % 7) as f32);
                (h, e, s)
            })
            .collect();
        evs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_h, e, s) in evs {
            ledger[0] += s;
            if stock[0] >= 1 {
                stock[0] -= 1;
                fed[e as usize] += 1;
            } else {
                missed[e as usize] += 1;
            }
        }
    }
    let mut m = BTreeMap::new();
    m.insert("stock", stock);
    m.insert("ledger", ledger.iter().map(|f| f.to_bits()).collect());
    m.insert("fed", fed);
    m.insert("missed", missed);
    m
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn assert_matches_reference(what: &str, got: &BTreeMap<&'static str, Vec<u32>>, want: &BTreeMap<&'static str, Vec<u32>>) {
    for (name, w) in want {
        let g = &got[name];
        let bad: Vec<(usize, u32, u32)> = g
            .iter()
            .zip(w)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .take(8)
            .collect();
        assert!(
            bad.is_empty(),
            "[{what}] field `{name}` differs from the host serial reference \
             (first mismatches as (slot, gpu, reference)): {bad:?}"
        );
    }
}

#[test]
fn claim_contention_is_deterministic_and_matches_serial_reference() {
    let reference = claim_reference();
    let mut hashes = Vec::new();
    let mut control_hashes = Vec::new();
    for run in 0..RUNS {
        let Some(r) = run_claim() else {
            eprintln!("[contention_claim_probe] skipping: no wgpu adapter on host.");
            return;
        };
        assert_matches_reference("claim", &r.serial, &reference);
        // Exclusivity: exactly one win per target per tick, every other
        // claim is an observed loss.
        let wins: u64 = r.serial["wins"].iter().map(|&w| w as u64).sum();
        let losses: u64 = r.serial["losses"].iter().map(|&w| w as u64).sum();
        assert_eq!(wins, (CLAIM_TARGETS * CLAIM_TICKS) as u64);
        assert_eq!(wins + losses, ((CLAIM_AGENTS - CLAIM_TARGETS) * CLAIM_TICKS) as u64);
        let ctrl_wins: u64 = r.control["wins"].iter().map(|&w| w as u64).sum();
        eprintln!(
            "[contention_claim_probe] run {run}: {:.3} ms/step; serial hash {:016x}; \
             control: {} total wins (exclusive = {}), claimed_by mismatches vs reference {}/{}, hash {:016x}",
            r.step_ms,
            fnv1a(&r.serial),
            ctrl_wins,
            CLAIM_TARGETS * CLAIM_TICKS,
            count_mismatches(&r.control["claimed_by"][..CLAIM_TARGETS as usize], &reference["claimed_by"][..CLAIM_TARGETS as usize]),
            CLAIM_TARGETS,
            fnv1a(&r.control),
        );
        hashes.push(fnv1a(&r.serial));
        control_hashes.push(fnv1a(&r.control));
    }
    assert!(hashes.windows(2).all(|w| w[0] == w[1]), "serial state hashes differ across runs: {hashes:x?}");
    let control_diverged = !control_hashes.windows(2).all(|w| w[0] == w[1]);
    eprintln!("[contention_claim_probe] control (unannotated) diverged across runs: {control_diverged}");
}

#[test]
fn stock_contention_is_deterministic_and_matches_serial_reference() {
    let reference = stock_reference();
    let mut hashes = Vec::new();
    let mut control_hashes = Vec::new();
    for run in 0..RUNS {
        let Some(r) = run_stock() else {
            eprintln!("[contention_stock_probe] skipping: no wgpu adapter on host.");
            return;
        };
        assert_matches_reference("stock", &r.serial, &reference);
        let fed: u64 = r.serial["fed"].iter().map(|&w| w as u64).sum();
        assert_eq!(fed, STOCK_INIT as u64, "every unit is eaten exactly once");
        eprintln!(
            "[contention_stock_probe] run {run}: {:.3} ms/step; serial hash {:016x}; \
             control: stock {} / fed {} / ledger {} (reference stock {} / fed {} / ledger {}), hash {:016x}",
            r.step_ms,
            fnv1a(&r.serial),
            r.control["stock"][0],
            r.control["fed"].iter().map(|&w| w as u64).sum::<u64>(),
            f32::from_bits(r.control["ledger"][0]),
            reference["stock"][0],
            fed,
            f32::from_bits(reference["ledger"][0]),
            fnv1a(&r.control),
        );
        hashes.push(fnv1a(&r.serial));
        control_hashes.push(fnv1a(&r.control));
    }
    assert!(hashes.windows(2).all(|w| w[0] == w[1]), "serial state hashes differ across runs: {hashes:x?}");
    let control_diverged = !control_hashes.windows(2).all(|w| w[0] == w[1]);
    eprintln!("[contention_stock_probe] control (unannotated) diverged across runs: {control_diverged}");
}

/// Child half of the cross-process check: only does work when
/// re-invoked by `contention_is_deterministic_across_processes`.
#[test]
fn contention_child_hash() {
    if std::env::var_os("CONTENTION_CHILD").is_none() {
        return;
    }
    let (Some(c), Some(s)) = (run_claim(), run_stock()) else {
        println!("CONTENTION_HASH=none");
        return;
    };
    println!("CONTENTION_HASH={:016x}:{:016x}", fnv1a(&c.serial), fnv1a(&s.serial));
}

#[test]
fn contention_is_deterministic_across_processes() {
    let (Some(c), Some(s)) = (run_claim(), run_stock()) else {
        eprintln!("[contention_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let here = format!("{:016x}:{:016x}", fnv1a(&c.serial), fnv1a(&s.serial));
    let exe = std::env::current_exe().expect("current_exe");
    for child in 0..2 {
        let out = std::process::Command::new(&exe)
            .args(["contention_child_hash", "--exact", "--nocapture", "--test-threads=1"])
            .env("CONTENTION_CHILD", "1")
            .output()
            .expect("spawn child test process");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.split_once("CONTENTION_HASH=").map(|(_, h)| h.trim()))
            .unwrap_or_else(|| panic!("child produced no hash line; stdout:\n{stdout}\nstderr:\n{}", String::from_utf8_lossy(&out.stderr)));
        eprintln!("[contention_probe] parent {here} / child {child} {line}");
        if line == "none" {
            continue;
        }
        assert_eq!(line, here, "state hash differs across processes");
    }
}

/// Per-kernel GPU timings for the two scenarios (the `@serial` kernel vs
/// the un-annotated control kernel). Prints only; needs
/// `SIM_KERNEL_TIMINGS=1` and a TIMESTAMP_QUERY-capable adapter.
#[test]
fn contention_kernel_timings() {
    fn report(label: &str, timings: &[(String, u64)]) {
        for (k, ns) in timings {
            eprintln!("[contention timings] {label}: {k} {:.1} us/tick", *ns as f64 / 1e3);
        }
    }
    fn avg(acc: &mut BTreeMap<String, (u64, u64)>, t: Vec<(String, u64)>) {
        for (k, ns) in t {
            let e = acc.entry(k).or_insert((0, 0));
            e.0 += ns;
            e.1 += 1;
        }
    }
    {
        use sims::contention_claim_probe::GeneratedRuntime;
        let Some(mut st) = GeneratedRuntime::try_new(SEED, CLAIM_AGENTS) else { return };
        if !st.kernel_timings_enabled() {
            eprintln!("[contention timings] disabled (set SIM_KERNEL_TIMINGS=1)");
            return;
        }
        let mut acc = BTreeMap::new();
        for i in 0..CLAIM_TICKS {
            st.step();
            st.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            if i >= 4 {
                avg(&mut acc, st.kernel_timings().into_iter().map(|t| (t.kernel, t.wall_ns)).collect());
            }
        }
        let v: Vec<(String, u64)> = acc.into_iter().map(|(k, (s, n))| (k, s / n.max(1))).collect();
        report("claim (1014 claimants x 10 targets)", &v);
    }
    {
        use sims::contention_stock_probe::GeneratedRuntime;
        let Some(mut st) = GeneratedRuntime::try_new(SEED, STOCK_AGENTS) else { return };
        let mut acc = BTreeMap::new();
        for _ in 0..STOCK_TICKS {
            st.step();
            st.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            avg(&mut acc, st.kernel_timings().into_iter().map(|t| (t.kernel, t.wall_ns)).collect());
        }
        let v: Vec<(String, u64)> = acc.into_iter().map(|(k, (s, n))| (k, s / n.max(1))).collect();
        report("stock (1023 eaters x 1 holder)", &v);
    }
}
