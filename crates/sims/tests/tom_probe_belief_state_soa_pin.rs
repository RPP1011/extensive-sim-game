//! Mega-crate-hosted port of the Wave 3 ToM Phase 2 BeliefState SoA pin
//! (formerly `crates/tom_probe_runtime/tests/belief_state_soa_pin.rs`).
//!
//! Round-trip + isolation checks across all 6 BeliefState columns.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const N: u32 = 4;

fn expected_flags(observer: usize, target: usize) -> u32 {
    ((observer as u32) << 16) | (target as u32)
}

fn expected_pos(observer: usize, target: usize) -> [f32; 4] {
    [
        observer as f32,
        target as f32,
        (observer * target) as f32,
        1.0,
    ]
}

fn expected_type(observer: usize, target: usize) -> u8 {
    ((observer * 7 + target) & 0xFF) as u8
}

fn expected_tick(observer: usize, target: usize) -> u32 {
    (100 + observer * 1000 + target) as u32
}

fn expected_confidence(observer: usize, target: usize) -> u8 {
    ((observer * 11 + target * 3) & 0xFF) as u8
}

fn expected_suspicion(observer: usize, target: usize) -> u8 {
    ((observer * 13 + target * 5) & 0xFF) as u8
}

#[test]
fn belief_state_soa_round_trips_all_columns() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    let flags: Vec<u32> = (0..cells).map(|i| expected_flags(i / n, i % n)).collect();
    let pos: Vec<[f32; 4]> = (0..cells).map(|i| expected_pos(i / n, i % n)).collect();
    let creature_types: Vec<u8> = (0..cells).map(|i| expected_type(i / n, i % n)).collect();
    let ticks: Vec<u32> = (0..cells).map(|i| expected_tick(i / n, i % n)).collect();
    let confidence: Vec<u8> = (0..cells).map(|i| expected_confidence(i / n, i % n)).collect();
    let suspicion: Vec<u8> = (0..cells).map(|i| expected_suspicion(i / n, i % n)).collect();

    h::seed_beliefs_flags(&mut sim, &flags);
    h::seed_beliefs_pos(&mut sim, &pos);
    h::seed_beliefs_type(&mut sim, &creature_types);
    h::seed_beliefs_tick(&mut sim, &ticks);
    h::seed_beliefs_confidence(&mut sim, &confidence);
    h::seed_beliefs_suspicion(&mut sim, &suspicion);

    let read_flags = h::read_beliefs_flags(&sim);
    let read_pos = h::read_beliefs_pos(&sim);
    let read_type = h::read_beliefs_type(&sim);
    let read_tick = h::read_beliefs_tick(&sim);
    let read_confidence = h::read_beliefs_confidence(&sim);
    let read_suspicion = h::read_beliefs_suspicion(&sim);

    assert_eq!(read_flags.len(), cells);
    assert_eq!(read_pos.len(), cells);
    assert_eq!(read_type.len(), cells);
    assert_eq!(read_tick.len(), cells);
    assert_eq!(read_confidence.len(), cells);
    assert_eq!(read_suspicion.len(), cells);

    for observer in 0..n {
        for target in 0..n {
            let idx = observer * n + target;
            assert_eq!(read_flags[idx], expected_flags(observer, target));
            assert_eq!(read_pos[idx], expected_pos(observer, target));
            assert_eq!(read_type[idx], expected_type(observer, target));
            assert_eq!(read_tick[idx], expected_tick(observer, target));
            assert_eq!(read_confidence[idx], expected_confidence(observer, target));
            assert_eq!(read_suspicion[idx], expected_suspicion(observer, target));
        }
    }
}

#[test]
fn belief_state_soa_per_observer_isolation() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    let mut flags = vec![0u32; cells];
    let mut pos = vec![[0.0f32; 4]; cells];
    let mut creature_types = vec![0u8; cells];
    let mut ticks = vec![0u32; cells];
    let mut confidence = vec![0u8; cells];
    let mut suspicion = vec![0u8; cells];

    for target in 0..n {
        let idx = target;
        flags[idx] = 0xDEAD_0000 | (target as u32);
        pos[idx] = [99.0, target as f32, -1.0, 7.0];
        creature_types[idx] = 0xA0 | (target as u8 & 0x0F);
        ticks[idx] = 5000 + target as u32;
        confidence[idx] = 0xC0 | (target as u8 & 0x0F);
        suspicion[idx] = 0x50 | (target as u8 & 0x0F);
    }

    h::seed_beliefs_flags(&mut sim, &flags);
    h::seed_beliefs_pos(&mut sim, &pos);
    h::seed_beliefs_type(&mut sim, &creature_types);
    h::seed_beliefs_tick(&mut sim, &ticks);
    h::seed_beliefs_confidence(&mut sim, &confidence);
    h::seed_beliefs_suspicion(&mut sim, &suspicion);

    let read_flags = h::read_beliefs_flags(&sim);
    let read_pos = h::read_beliefs_pos(&sim);
    let read_type = h::read_beliefs_type(&sim);
    let read_tick = h::read_beliefs_tick(&sim);
    let read_confidence = h::read_beliefs_confidence(&sim);
    let read_suspicion = h::read_beliefs_suspicion(&sim);

    for target in 0..n {
        let idx = target;
        assert_eq!(read_flags[idx], 0xDEAD_0000 | (target as u32));
        assert_eq!(read_pos[idx], [99.0, target as f32, -1.0, 7.0]);
        assert_eq!(read_type[idx], 0xA0 | (target as u8 & 0x0F));
        assert_eq!(read_tick[idx], 5000 + target as u32);
        assert_eq!(read_confidence[idx], 0xC0 | (target as u8 & 0x0F));
        assert_eq!(read_suspicion[idx], 0x50 | (target as u8 & 0x0F));
    }

    for observer in 1..n {
        for target in 0..n {
            let idx = observer * n + target;
            assert_eq!(read_flags[idx], 0, "flags[{observer},{target}]");
            assert_eq!(read_pos[idx], [0.0; 4], "pos[{observer},{target}]");
            assert_eq!(read_type[idx], 0, "type[{observer},{target}]");
            assert_eq!(read_tick[idx], 0, "tick[{observer},{target}]");
            assert_eq!(read_confidence[idx], 0, "conf[{observer},{target}]");
            assert_eq!(read_suspicion[idx], 0, "susp[{observer},{target}]");
        }
    }
}

#[test]
fn belief_state_soa_per_column_independence() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    let only_ticks: Vec<u32> = (0..cells).map(|i| 1000 + i as u32).collect();
    h::seed_beliefs_tick(&mut sim, &only_ticks);

    let read_flags = h::read_beliefs_flags(&sim);
    let read_pos = h::read_beliefs_pos(&sim);
    let read_type = h::read_beliefs_type(&sim);
    let read_tick = h::read_beliefs_tick(&sim);
    let read_confidence = h::read_beliefs_confidence(&sim);
    let read_suspicion = h::read_beliefs_suspicion(&sim);

    for i in 0..cells {
        assert_eq!(read_flags[i], 0, "flags[{i}]");
        assert_eq!(read_pos[i], [0.0; 4], "pos[{i}]");
        assert_eq!(read_type[i], 0, "type[{i}]");
        assert_eq!(read_tick[i], 1000 + i as u32, "tick[{i}]");
        assert_eq!(read_confidence[i], 0, "conf[{i}]");
        assert_eq!(read_suspicion[i], 0, "susp[{i}]");
    }
}
