//! webband_fields_probe — behavioural pin for the single question:
//! do custom agent `field` declarations reach the verb surface (`when`
//! masks + `score` expressions), and can per-agent physics write them?
//!
//! The fixture (assets/sim/webband_fields_probe.sim) rises `mood`
//! 1.0/tick and `skill` 0.5/tick via `agents.set_*` and races three
//! self-only verbs through the per-agent argmax:
//!   * Work  — when mood >= 60, score 1000 (mask reads mood)
//!   * Study — always legal, score self.skill (score reads skill)
//!   * Loaf  — always legal, score 20 constant (the control band)
//!
//! Over 100 ticks a control agent's expected action counts are
//! Loaf ≈ 40 / Study ≈ 20 / Work ≈ 40 — a split only explicable if
//! BOTH custom-field reads lower and execute. The causal flip the
//! probe brief asks for is done per-agent by direct buffer seeding:
//! one slot's mood is preloaded to -1000 (the gate can never pass in
//! 100 ticks → Work must be 0 for that slot ONLY) and one to +1000
//! (gate open from tick 0 → Work ≈ 100, Loaf ≈ 0). Same runtime, same
//! rules — only the field value differs, so any Work-count difference
//! is BECAUSE of the mask's field read.
//!
//! Requires a GPU adapter; without one `try_new` returns `None` and
//! the test skips (corpus convention).

#![allow(non_snake_case)]

use sims::webband_fields_probe::GeneratedRuntime;

const SEED: u64 = 0xF1E1D5_F00D;
const AGENTS: u32 = 16; // 8 spawned workers land in slots 1..=8.
const TICKS: usize = 100;

// Fixture constants (config probe { ... } in the .sim).
const MOOD_RISE: f32 = 1.0;
const SKILL_RISE: f32 = 0.5;

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_fields_probe::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("webband_fields_probe::f32_readback"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_fields_probe::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("webband_fields_probe::u32_readback"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

#[test]
fn custom_fields_reach_masks_scores_and_physics_writes() {
    let mut state = match GeneratedRuntime::try_new(SEED, AGENTS) {
        Some(s) => s,
        None => {
            eprintln!("[webband_fields_probe] skipping: no wgpu adapter");
            return;
        }
    };
    let n = state.agent_count as usize;

    // Discover the spawned workers (slot 0 is the AgentId sentinel).
    let alive0 = {
        let b = state.agent_alive_buf.clone();
        read_u32(&mut state, &b, n)
    };
    let workers: Vec<usize> = (0..n).filter(|&i| alive0[i] != 0).collect();
    assert_eq!(workers.len(), 8, "founding seeds 8 workers; got {workers:?}");

    // Causal flip via per-agent field seeding (same runtime, same
    // rules; only the custom-field VALUE differs between cohorts):
    //   suppressed: mood preloaded to -1000 → gate unreachable → Work=0
    //   boosted:    mood preloaded to +1000 → gate open at tick 0
    let suppressed = workers[0];
    let boosted = workers[1];
    let controls: Vec<usize> = workers[2..].to_vec();
    state.gpu.queue.write_buffer(
        &state.agent_mood_buf,
        (suppressed as u64) * 4,
        bytemuck::cast_slice(&[-1000.0f32]),
    );
    state.gpu.queue.write_buffer(
        &state.agent_mood_buf,
        (boosted as u64) * 4,
        bytemuck::cast_slice(&[1000.0f32]),
    );

    for _ in 0..TICKS {
        state.step();
    }

    let mood = {
        let b = state.agent_mood_buf.clone();
        read_f32(&mut state, &b, n)
    };
    let skill = {
        let b = state.agent_skill_buf.clone();
        read_f32(&mut state, &b, n)
    };
    let worked = {
        let b = state.view_storage_worked_total_primary_buf.clone();
        read_f32(&mut state, &b, n)
    };
    let studied = {
        let b = state.view_storage_studied_total_primary_buf.clone();
        read_f32(&mut state, &b, n)
    };
    let loafed = {
        let b = state.view_storage_loafed_total_primary_buf.clone();
        read_f32(&mut state, &b, n)
    };

    for &i in &workers {
        println!(
            "[webband_fields_probe] slot {i}{}: mood={:.1} skill={:.1} \
             worked={} studied={} loafed={}",
            if i == suppressed {
                " (mood -1000)"
            } else if i == boosted {
                " (mood +1000)"
            } else {
                ""
            },
            mood[i],
            skill[i],
            worked[i],
            studied[i],
            loafed[i],
        );
    }

    // --- Sub-question 2: physics WRITES custom fields.
    // Control slots: mood 0 → ~100, skill 0 → ~50 after 100 ticks of
    // agents.set_mood / agents.set_skill.
    for &i in &controls {
        assert!(
            (mood[i] - TICKS as f32 * MOOD_RISE).abs() <= 1.5,
            "slot {i}: mood={} after {TICKS} ticks (expected ~{}) — \
             physics write to custom field `mood` is dead",
            mood[i],
            TICKS as f32 * MOOD_RISE
        );
        assert!(
            (skill[i] - TICKS as f32 * SKILL_RISE).abs() <= 1.5,
            "slot {i}: skill={} after {TICKS} ticks (expected ~{}) — \
             physics write to custom field `skill` is dead",
            skill[i],
            TICKS as f32 * SKILL_RISE
        );
    }

    // --- Sub-question 1a: the MASK reads `mood` (per-agent).
    // Control timeline: gate 60 crosses at tick ~60 → Work ≈ 40.
    for &i in &controls {
        assert!(
            (32.0..=48.0).contains(&worked[i]),
            "slot {i}: worked={} (expected ~40 of {TICKS}) — the `when mood >= 60` \
             gate did not open at the field-crossing tick",
            worked[i]
        );
    }
    // Suppressed cohort: mood -1000 +100 rise = -900 < 60 forever → 0.
    assert_eq!(
        worked[suppressed], 0.0,
        "suppressed slot {suppressed} worked {} times with mood {} — \
         the mask is NOT reading the per-agent custom field",
        worked[suppressed], mood[suppressed]
    );
    // Boosted cohort: gate open from tick 0 → Work wins every tick.
    assert!(
        worked[boosted] >= (TICKS as f32) - 2.0,
        "boosted slot {boosted} worked only {} of {TICKS} ticks with mood {} — \
         the mask is NOT reading the per-agent custom field",
        worked[boosted],
        mood[boosted]
    );
    assert!(
        loafed[boosted] <= 2.0,
        "boosted slot {boosted} loafed {} times despite a permanently open \
         top-band gate",
        loafed[boosted]
    );

    // --- Sub-question 1b: the SCORE reads `skill`.
    // Control timeline: Study (score = skill, 0.5/tick) overtakes
    // Loaf's constant 20 at tick ~40 and loses to Work at ~60 →
    // Study ≈ 20, Loaf ≈ 40. If the score read were dead (0 or NaN),
    // Study would be ~0; if it dominated wrongly, Loaf would be ~0.
    for &i in &controls {
        assert!(
            (12.0..=28.0).contains(&studied[i]),
            "slot {i}: studied={} (expected ~20) — `score self.skill` is not \
             tracking the custom field's trajectory",
            studied[i]
        );
        assert!(
            (32.0..=48.0).contains(&loafed[i]),
            "slot {i}: loafed={} (expected ~40) — the constant band lost to a \
             broken skill score",
            loafed[i]
        );
        let total = worked[i] + studied[i] + loafed[i];
        assert!(
            (95.0..=(TICKS as f32 + 1.0)).contains(&total),
            "slot {i}: action total {total} over {TICKS} ticks — the argmax \
             did not fire once per tick"
        );
    }
}

#[test]
fn probe_is_deterministic_across_runs() {
    // Two identical runs must produce identical counters (the fixture
    // draws no rng; determinism here pins the phase ordering the
    // band assertions rely on).
    let mut totals: Vec<Vec<f32>> = Vec::new();
    for _ in 0..2 {
        let mut state = match GeneratedRuntime::try_new(SEED, AGENTS) {
            Some(s) => s,
            None => {
                eprintln!("[webband_fields_probe] skipping: no wgpu adapter");
                return;
            }
        };
        let n = state.agent_count as usize;
        for _ in 0..TICKS {
            state.step();
        }
        let worked = {
            let b = state.view_storage_worked_total_primary_buf.clone();
            read_f32(&mut state, &b, n)
        };
        let studied = {
            let b = state.view_storage_studied_total_primary_buf.clone();
            read_f32(&mut state, &b, n)
        };
        let loafed = {
            let b = state.view_storage_loafed_total_primary_buf.clone();
            read_f32(&mut state, &b, n)
        };
        totals.push(
            worked
                .into_iter()
                .chain(studied)
                .chain(loafed)
                .collect(),
        );
    }
    assert_eq!(
        totals[0], totals[1],
        "non-deterministic action counts across identical runs"
    );
}
