//! maze_explorer_multi behavioural pin — N Adventurers explore
//! the same maze in parallel, sharing visited-room knowledge each
//! tick via a Gossip-driven social-merge on the I.3b belief.
//!
//! Exercises the full I.3b stack as a system:
//!   * Per-tick propagation handler writes (fold_visited)
//!   * Per-tick social-merge writes (merge_visited_gossip_bit_or)
//!     — every receiver OR-merges every source's 16-cell row
//!   * Per-tick pair-keyed view-read in the per-agent rule
//!     (`visited(self, next_room)` resolves to
//!     `view_storage_visited_primary[agent_id * 16u + next_room]`)
//!
//! Outcome metric: tick-to-first-found-key. With 4 sharing agents
//! the team should find the key meaningfully faster than the
//! single-agent baseline (median 172 for belief_smart; 128 for
//! soa-smart).

#![allow(non_snake_case)]

use sims::maze_explorer_multi::GeneratedRuntime;

const SEED: u64 = 0xB1_BB_BB_BB_BB_BB_BB_77u64;
const N_AGENTS: u32 = 4;
const TICK_BUDGET: u64 = 5000;
const K: u32 = 16;

fn read_u32_slice(state: &GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<u32> {
    let bytes = (n as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multi::read_slice"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multi::readback"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let v: Vec<u32> = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..n as usize].to_vec()
    };
    staging.unmap();
    v
}

fn run_one(seed: u64) -> Option<(u32, Vec<u32>)> {
    let mut state = GeneratedRuntime::try_new(seed, N_AGENTS)?;
    let alive: Vec<u32> = vec![1u32; N_AGENTS as usize];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    let mut ticks = 0u64;
    while ticks < TICK_BUDGET {
        state.step();
        ticks += 1;
        let found = read_u32_slice(&state, &state.agent_found_key_buf, N_AGENTS);
        if found.iter().any(|f| *f == 1) {
            // Read tick_found from each agent that has found_key=1
            let tf = read_u32_slice(&state, &state.agent_tick_found_buf, N_AGENTS);
            // Earliest non-zero tick_found across agents (every
            // agent that found the key recorded the tick they
            // entered it; min finds the first).
            let earliest = found
                .iter()
                .zip(tf.iter())
                .filter(|(f, _)| **f == 1)
                .map(|(_, t)| *t)
                .min()
                .unwrap_or(0);
            return Some((earliest, found));
        }
    }
    None
}

#[test]
fn multi_agent_team_finds_key_within_budget() {
    let (earliest_tick, found) = match run_one(SEED) {
        Some(x) => x,
        None => {
            if GeneratedRuntime::try_new(SEED, N_AGENTS).is_none() {
                eprintln!("[maze_multi] skipping: no wgpu adapter");
                return;
            }
            panic!("4-agent team did not find key within {TICK_BUDGET} ticks");
        }
    };
    let finders = found.iter().filter(|f| **f == 1).count();
    println!(
        "[maze_multi] earliest tick_found={earliest_tick} ({finders}/{N_AGENTS} agents found)"
    );
    assert!(earliest_tick < TICK_BUDGET as u32);
}

#[test]
fn multi_agent_gossip_propagates_visited_rows() {
    // The social-merge clause `on Gossip { source: s } merge from s:
    // bit_or` SHOULD make every agent's row converge on the union
    // of all agents' visited rooms within a few ticks. With the
    // current dispatch infrastructure it converges only partially:
    //
    // Known gap (#multi-K-dispatch): the merge kernel dispatches
    // (N/8, N/8) workgroups via DispatchShape::PerAgentEventScan,
    // which sizes BOTH axes by agent_cap. For an I.3b key-typed
    // belief with K > N (here K=16, N=4) the y axis covers only
    // [0..N), missing cells s ∈ [N..K). Agents that have visited
    // rooms in the s ∈ [N..K) range never propagate those bits via
    // gossip — the merge thread for that (r, s) cell never runs.
    //
    // Fixing this properly needs a new DispatchShape::PerAgentKeyScan
    // { k } variant (24 match sites to update across the dispatch
    // module + emit + schedule synthesis), or threading per-op K
    // into DispatchCtx. Deferred to a follow-up; pin asserts the
    // weaker convergence-up-to-N property here.

    let mut state = GeneratedRuntime::try_new(SEED, N_AGENTS)
        .expect("wgpu adapter");
    let alive: Vec<u32> = vec![1u32; N_AGENTS as usize];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    for _ in 0..30 {
        state.step();
    }
    let rows = read_u32_slice(
        &state,
        &state.view_storage_visited_primary_buf,
        N_AGENTS * K,
    );
    let row_for = |a: u32| -> u32 {
        let mut mask = 0u32;
        for k in 0..K {
            if rows[(a * K + k) as usize] != 0 {
                mask |= 1 << k;
            }
        }
        mask
    };
    let masks: Vec<u32> = (0..N_AGENTS).map(row_for).collect();
    let union = masks.iter().fold(0u32, |a, b| a | b);
    println!(
        "[maze_multi] after 30 ticks: per-agent visited masks = {:04X?}, union = 0x{union:04X}",
        masks
    );
    // Sanity: every agent's mask is a subset of the union (gossip
    // only ADDS bits, never invents them).
    for (a, m) in masks.iter().enumerate() {
        assert_eq!(
            *m & union, *m,
            "agent {a}'s mask 0x{m:04X} has bits not in union 0x{union:04X} — merge invented bits?"
        );
    }
    // Strict invariant (post K-aware dispatch fix): every agent's
    // visited row equals the union across ALL K cells. The
    // KernelSpec.y_dim_override = Some(K) routes the merge kernel
    // through `dispatch_workgroups(N/8, K/8, 1)` instead of
    // `(N/8, N/8, 1)`, so cells s ∈ [N..K) now have a thread.
    for (a, m) in masks.iter().enumerate() {
        assert_eq!(
            *m, union,
            "agent {a}'s mask 0x{m:04X} ≠ union 0x{union:04X} — gossip merge incomplete (K-dispatch gap regressed?)"
        );
    }
}
