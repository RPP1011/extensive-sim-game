//! `wave_defense_app` — driver binary for Stress Fixture C (task #243).
//!
//! ## CLI
//!
//! ```text
//! wave_defense_app [seed] [max_ticks]
//! ```
//!
//! Defaults: seed=0, max_ticks=2000.
//!
//! ## Output
//!
//! Per-tick NDJSON lines:
//!
//!   `{"tick": N, "alive_settlers": M, "alive_monsters": K,
//!     "score": S, "wave_size": W}`
//!
//! Final summary line:
//!
//!   `{"summary": true, "died_at_tick": T, "score": S,
//!     "max_wave_size": W, "panic": null|"<msg>"}`
//!
//! Wraps `step()` in `std::panic::catch_unwind` per P10. To save
//! per-tick GPU sync overhead, only every 50th tick reads back the
//! settler/monster counters; the others just re-emit the last sample.
//! The summary line always reads the final score after termination.

use std::io::Write;
use std::panic::AssertUnwindSafe;

use wave_defense_runtime::{WaveDefenseState, DEFAULT_MAX_TICKS, WAVE_SIZE};

const TICK_SAMPLE_PERIOD: u64 = 50;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_ticks: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_TICKS);

    eprintln!(
        "[wave_defense_app] seed={} max_ticks={}",
        seed, max_ticks,
    );

    let mut stdout = std::io::stdout().lock();

    let init = std::panic::catch_unwind(AssertUnwindSafe(|| {
        WaveDefenseState::new(seed)
    }));
    let mut state = match init {
        Ok(s) => s,
        Err(payload) => {
            let msg = panic_payload_msg(payload.as_ref());
            eprintln!("[wave_defense_app] PANIC at construction: {msg}");
            emit_summary(&mut stdout, 0, 0.0, WAVE_SIZE, Some(msg));
            return;
        }
    };

    // Track last-seen counts so non-sample ticks can repeat them.
    let mut last_alive_settlers: u32 = 0;
    let mut last_alive_monsters: u32 = 0;
    let mut last_score: f32 = 0.0;
    let wave_size: u32 = WAVE_SIZE; // constant per Task #243 deferral

    let mut died_at_tick: u64 = max_ticks;
    let mut panic_msg: Option<String> = None;

    for t in 0..max_ticks {
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            state.step_and_check_termination()
        }));
        let terminated = match outcome {
            Ok(b) => b,
            Err(payload) => {
                let msg = panic_payload_msg(payload.as_ref());
                eprintln!(
                    "[wave_defense_app] PANIC at tick {}: {msg}",
                    t,
                );
                panic_msg = Some(msg);
                died_at_tick = t;
                break;
            }
        };

        // Per-tick NDJSON sample. Only every TICK_SAMPLE_PERIOD does a
        // GPU readback so we don't murder perf at high tick counts.
        if t % TICK_SAMPLE_PERIOD == 0 || terminated {
            last_alive_settlers = state.alive_settler_count();
            last_alive_monsters = state.alive_monster_count();
            last_score = state.read_score();
        }
        emit_per_tick(
            &mut stdout,
            t,
            last_alive_settlers,
            last_alive_monsters,
            last_score,
            wave_size,
        );

        if terminated {
            died_at_tick = t;
            break;
        }
    }

    // Always re-read score at termination (covers max_ticks-without-
    // termination case AND the panic break).
    let final_score = std::panic::catch_unwind(AssertUnwindSafe(|| state.read_score()))
        .unwrap_or(last_score);
    emit_summary(&mut stdout, died_at_tick, final_score, wave_size, panic_msg);
}

fn emit_per_tick(
    w: &mut impl Write,
    tick: u64,
    alive_settlers: u32,
    alive_monsters: u32,
    score: f32,
    wave_size: u32,
) {
    let line = format!(
        "{{\"tick\":{},\"alive_settlers\":{},\"alive_monsters\":{},\"score\":{},\"wave_size\":{}}}\n",
        tick, alive_settlers, alive_monsters, score, wave_size,
    );
    let _ = w.write_all(line.as_bytes());
    let _ = w.flush();
}

fn emit_summary(
    w: &mut impl Write,
    died_at_tick: u64,
    score: f32,
    max_wave_size: u32,
    panic_msg: Option<String>,
) {
    let panic_field = match &panic_msg {
        Some(m) => format!("\"{}\"", json_escape(m)),
        None => "null".to_string(),
    };
    let line = format!(
        "{{\"summary\":true,\"died_at_tick\":{},\"score\":{},\"max_wave_size\":{},\"panic\":{}}}\n",
        died_at_tick, score, max_wave_size, panic_field,
    );
    let _ = w.write_all(line.as_bytes());
    let _ = w.flush();
}

fn panic_payload_msg(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<panic with non-string payload>".to_string()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}
