use crate::sim::{step, ReplayResult, SimEvent, UnitIntent};
use crate::pathing::GridNav;

pub fn event_row(event: &SimEvent) -> (u64, String, String, String, String, String) {
    match *event {
        SimEvent::Moved {
            tick,
            unit_id,
            from_x100,
            from_y100,
            to_x100,
            to_y100,
        } => (
            tick,
            "Moved".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            format!(
                "({}, {}) -> ({}, {})",
                from_x100, from_y100, to_x100, to_y100
            ),
        ),
        SimEvent::CastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "CastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "attack cast start".to_string(),
        ),
        SimEvent::AbilityCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "ability cast start".to_string(),
        ),
        SimEvent::HealCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "heal cast start".to_string(),
        ),
        SimEvent::ControlCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "control cast start".to_string(),
        ),
        SimEvent::ControlApplied {
            tick,
            source_id,
            target_id,
            duration_ms,
        } => (
            tick,
            "ControlApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            duration_ms.to_string(),
            format!("control for {}ms", duration_ms),
        ),
        SimEvent::UnitControlled { tick, unit_id } => (
            tick,
            "UnitControlled".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            "unit action locked".to_string(),
        ),
        SimEvent::DamageApplied {
            tick,
            source_id,
            target_id,
            amount,
            target_hp_after,
            ..
        } => (
            tick,
            "DamageApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            amount.to_string(),
            format!("target hp -> {}", target_hp_after),
        ),
        SimEvent::HealApplied {
            tick,
            source_id,
            target_id,
            amount,
            target_hp_after,
            ..
        } => (
            tick,
            "HealApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            amount.to_string(),
            format!("target hp -> {}", target_hp_after),
        ),
        SimEvent::UnitDied { tick, unit_id } => (
            tick,
            "UnitDied".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            "unit died".to_string(),
        ),
        SimEvent::AttackBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "AttackBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::AbilityBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "AbilityBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::HealBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "HealBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::ControlBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "ControlBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::AttackBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AttackBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::AbilityBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::HealBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::ControlBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::AttackRepositioned {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AttackRepositioned".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "reposition".to_string(),
        ),
        SimEvent::AbilityBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
        SimEvent::CastFailedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "CastFailedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "cast failed".to_string(),
        ),
        SimEvent::HealBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
        SimEvent::ControlBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
        // Hero ability engine events -- generic row
        _ => (0, format!("{:?}", event).chars().take(40).collect(), "-".into(), "-".into(), "-".into(), "-".into()),
    }
}

pub fn build_event_rows(replay: &ReplayResult) -> String {
    let mut rows = String::new();
    for event in &replay.events {
        let (tick, kind, src, dst, value, detail) = event_row(event);
        rows.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            tick, kind, src, dst, value, detail
        ));
    }
    rows
}

pub fn build_frame_rows(
    initial: &crate::sim::SimState,
    script: &[Vec<UnitIntent>],
    dt_ms: u32,
) -> String {
    let mut frame_rows = String::new();
    let mut state = initial.clone();
    for unit in &state.units {
        frame_rows.push_str(&format!(
            "{}\t{}\t{:?}\t{}\t{:.3}\t{:.3}\n",
            state.tick, unit.id, unit.team, unit.hp, unit.position.x, unit.position.y
        ));
    }
    for intents in script {
        let (new_state, _) = step(state, intents, dt_ms);
        state = new_state;
        for unit in &state.units {
            frame_rows.push_str(&format!(
                "{}\t{}\t{:?}\t{}\t{:.3}\t{:.3}\n",
                state.tick, unit.id, unit.team, unit.hp, unit.position.x, unit.position.y
            ));
        }
    }
    frame_rows
}

pub fn obstacle_rows_from_nav_cells(nav: &GridNav) -> String {
    let mut cells = nav.blocked.iter().copied().collect::<Vec<_>>();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut rows = String::new();
    for (cx, cy) in cells {
        let min_x = nav.min_x + cx as f32 * nav.cell_size;
        let max_x = min_x + nav.cell_size;
        let min_y = nav.min_y + cy as f32 * nav.cell_size;
        let max_y = min_y + nav.cell_size;
        rows.push_str(&format!(
            "{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            min_x, max_x, min_y, max_y
        ));
    }
    rows
}
