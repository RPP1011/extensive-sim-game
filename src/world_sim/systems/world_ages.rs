#![allow(unused)]
//! World ages — named historical eras that emerge from event density.
//!
//! Every 2400 ticks (1 game-year), analyzes recent chronicle entries to
//! determine the character of the era. When the dominant event type shifts,
//! a new age is proclaimed.
//!
//! Age types:
//! - Age of Blood: dominated by Death/war events
//! - Age of Discovery: dominated by Achievement/exploration events
//! - Age of Peace: low event density, high morale
//! - Age of Legends: multiple legends exist
//! - Age of Expansion: settlements founded
//! - Age of Corruption: betrayals and broken oaths
//!
//! Cadence: every 2400 ticks.

use crate::world_sim::state::*;

const AGE_INTERVAL: u64 = 2400;

pub fn advance_world_ages(state: &mut WorldState) {
    if state.tick % AGE_INTERVAL != 0 || state.tick == 0 { return; }
    if state.tick < AGE_INTERVAL { return; }

    let tick = state.tick;
    let year = tick / AGE_INTERVAL;

    // Count chronicle events in the last age.
    let period_start = tick.saturating_sub(AGE_INTERVAL);
    let recent: Vec<&ChronicleEntry> = state.chronicle.iter()
        .filter(|e| e.tick >= period_start && e.tick < tick)
        .collect();

    if recent.is_empty() { return; }

    let deaths = recent.iter().filter(|e| e.category == ChronicleCategory::Death).count();
    let achievements = recent.iter().filter(|e| e.category == ChronicleCategory::Achievement).count();
    let narratives = recent.iter().filter(|e| e.category == ChronicleCategory::Narrative).count();

    // Check for specific themes in narrative text.
    let wars = recent.iter().filter(|e| e.text.contains("WAR") || e.text.contains("war")).count();
    let peace = recent.iter().filter(|e| e.text.contains("PEACE") || e.text.contains("peace")).count();
    let betrayals = recent.iter().filter(|e| e.text.contains("betray") || e.text.contains("Betray")).count();
    let legends = recent.iter().filter(|e| e.text.contains("LEGEND") || e.text.contains("Legendary")).count();
    let foundings = recent.iter().filter(|e| e.text.contains("found") || e.text.contains("colonist")).count();
    let prophecies = recent.iter().filter(|e| e.text.contains("PROPHECY")).count();

    // Determine era character.
    let total = recent.len();
    let age_name = if prophecies > 0 {
        "The Age of Prophecy"
    } else if wars > 2 {
        "The Age of Blood"
    } else if legends > 1 {
        "The Age of Legends"
    } else if betrayals > 2 {
        "The Age of Corruption"
    } else if foundings > 1 {
        "The Age of Expansion"
    } else if deaths > total / 2 {
        "The Age of Sorrow"
    } else if achievements > total / 2 {
        "The Age of Discovery"
    } else if peace > 0 && deaths < 3 {
        "The Long Peace"
    } else {
        // Generic age name from year number.
        match year % 8 {
            0 => "The Age of Iron",
            1 => "The Age of Storms",
            2 => "The Silver Age",
            3 => "The Dark Age",
            4 => "The Golden Age",
            5 => "The Age of Embers",
            6 => "The Age of Winds",
            _ => "The Quiet Age",
        }
    };

    // Population snapshot.
    let pop = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();
    let settlements = state.settlements.len();

    state.chronicle.push(ChronicleEntry {
        tick,
        category: ChronicleCategory::Narrative,
        text: format!(
            "Year {} begins. {} — {} souls across {} settlements. {} chronicle events shaped the previous era.",
            year, age_name, pop, settlements, total),
        entity_ids: vec![],
    });
}
