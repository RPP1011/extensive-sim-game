//! Advanced generation strategies (5-7) and disk write functions.
//!
//! Split from `strategies.rs` to keep files under 500 lines.

use std::path::Path;

use super::super::types::ScenarioFile;
use super::coverage::CoverageTracker;
use super::metadata::{
    ALL_HEROES, ALL_LOL_HEROES, ROOM_TYPES, DedupSet, Lcg,
};
use super::strategies::{build_scenario, build_hvh_scenario, emit, to_toml};
use super::GenConfig;

// ---------------------------------------------------------------------------
// 5. Team size spectrum
// ---------------------------------------------------------------------------

pub(super) fn generate_size_spectrum(
    config: &GenConfig,
    rng: &mut Lcg,
    next_seed: &mut impl FnMut() -> u64,
    hero_names: &[&str],
    out: &mut Vec<ScenarioFile>,
    dedup: &mut DedupSet,
    coverage: &mut CoverageTracker,
) {
    let sizes: &[(usize, usize)] = &[
        (1, 2), (1, 3),
        (2, 2), (2, 3), (2, 4),
        (3, 3), (3, 4), (3, 5),
        (4, 4), (4, 5), (4, 6), (4, 8),
        (5, 5), (5, 6), (5, 7),
        (6, 6), (6, 8),
        (8, 8), (8, 10),
    ];

    let sv = config.seed_variants;
    for (hc, ec) in sizes {
        for _ in 0..2 {
            let party: Vec<String> = rng.sample_n(hero_names, *hc)
                .into_iter().map(|s| s.to_string()).collect();
            let diff = (rng.next_usize(3) + 1) as u32;
            let room = *rng.choose(ROOM_TYPES);
            let hp = if *ec > *hc + 2 { 2.0 } else { 1.0 };
            let seed = next_seed();
            let name = format!("size_{hc}v{ec}_{}", out.len());
            let s = build_scenario(name, party, *ec, diff, hp, room, seed, 10_000);
            emit(s, sv, out, dedup, coverage);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Hero vs Hero
// ---------------------------------------------------------------------------

pub(super) fn generate_hero_vs_hero(
    config: &GenConfig,
    rng: &mut Lcg,
    next_seed: &mut impl FnMut() -> u64,
    out: &mut Vec<ScenarioFile>,
    dedup: &mut DedupSet,
    coverage: &mut CoverageTracker,
) {
    let all_names: Vec<&str> = ALL_HEROES.iter().chain(ALL_LOL_HEROES.iter())
        .map(|h| h.name).collect();
    let std_names: Vec<&str> = ALL_HEROES.iter().map(|h| h.name).collect();
    let lol_names: Vec<&str> = ALL_LOL_HEROES.iter().map(|h| h.name).collect();
    let sv = config.seed_variants;

    for i in 0..config.hvh_count {
        let team_size = match rng.next_usize(6) {
            0 => 2,
            1..=2 => 3,
            3..=4 => 4,
            _ => 5,
        };

        // Composition strategy for matchup variety
        let (team_a, team_b) = match rng.next_usize(5) {
            0 => {
                // Standard vs Standard
                let a: Vec<String> = rng.sample_n(&std_names, team_size).into_iter().map(|s| s.to_string()).collect();
                let b: Vec<String> = rng.sample_n(&std_names, team_size).into_iter().map(|s| s.to_string()).collect();
                (a, b)
            }
            1 => {
                // LoL vs LoL
                let a: Vec<String> = rng.sample_n(&lol_names, team_size).into_iter().map(|s| s.to_string()).collect();
                let b: Vec<String> = rng.sample_n(&lol_names, team_size).into_iter().map(|s| s.to_string()).collect();
                (a, b)
            }
            2 => {
                // Standard vs LoL (cross-universe)
                let a: Vec<String> = rng.sample_n(&std_names, team_size).into_iter().map(|s| s.to_string()).collect();
                let b: Vec<String> = rng.sample_n(&lol_names, team_size).into_iter().map(|s| s.to_string()).collect();
                (a, b)
            }
            _ => {
                // Mixed pool — both teams from all heroes
                let a: Vec<String> = rng.sample_n(&all_names, team_size).into_iter().map(|s| s.to_string()).collect();
                let b: Vec<String> = rng.sample_n(&all_names, team_size).into_iter().map(|s| s.to_string()).collect();
                (a, b)
            }
        };

        let room = *rng.choose(ROOM_TYPES);
        let hp = *rng.choose(&[1.0f32, 2.0, 3.0]);
        let seed = next_seed();
        let name = format!("hvh_{i:04}_{team_size}v{team_size}");
        let s = build_hvh_scenario(name, team_a, team_b, hp, room, seed, 8_000);
        emit(s, sv, out, dedup, coverage);
    }
}

// ---------------------------------------------------------------------------
// 7. Coverage-driven random fill
// ---------------------------------------------------------------------------

pub(super) fn generate_random_fill(
    config: &GenConfig,
    rng: &mut Lcg,
    next_seed: &mut impl FnMut() -> u64,
    hero_names: &[&str],
    out: &mut Vec<ScenarioFile>,
    dedup: &mut DedupSet,
    coverage: &mut CoverageTracker,
) {
    let sv = config.seed_variants;
    for i in 0..config.extra_random {
        let party_size = match rng.next_usize(10) {
            0 => 2,
            1..=2 => 3,
            3..=6 => 4,
            7..=8 => 5,
            _ => 6,
        };

        // Anchor on least-seen hero
        let anchor = coverage.least_seen_hero().to_string();
        let mut party = vec![anchor];

        for _ in 1..party_size {
            let h = if rng.next_usize(3) == 0 {
                // Prefer undertested
                let mut candidates: Vec<&str> = hero_names.iter()
                    .filter(|h| !party.iter().any(|p| p == **h))
                    .copied()
                    .collect();
                candidates.sort_by_key(|h| coverage.hero_appearances(h));
                candidates.first().unwrap_or(&"warrior").to_string()
            } else {
                loop {
                    let h = rng.choose(hero_names);
                    if !party.iter().any(|p| p == *h) { break h.to_string(); }
                }
            };
            party.push(h);
        }

        let ec = party.len() + rng.next_usize(4);
        let diff = (rng.next_usize(5) + 1) as u32;
        let room = if rng.next_usize(3) == 0 {
            coverage.least_seen_room()
        } else {
            rng.choose(ROOM_TYPES)
        };
        let hp = *rng.choose(&[1.0f32, 1.0, 2.0, 3.0, 5.0]);
        let seed = next_seed();

        let name = format!("rand_{i:04}_{party_size}v{ec}_d{diff}");
        let s = build_scenario(name, party, ec, diff, hp, room, seed, 10_000);
        emit(s, sv, out, dedup, coverage);
    }
}

// ---------------------------------------------------------------------------
// Write to disk
// ---------------------------------------------------------------------------

pub fn write_scenarios(scenarios: &[ScenarioFile], output_dir: &Path) -> Result<usize, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create {}: {e}", output_dir.display()))?;

    // Clean old generated files (only gen_*.toml to be safe)
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("gen_") && name.ends_with(".toml") {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    for (i, scenario) in scenarios.iter().enumerate() {
        let filename = format!("gen_{:04}.toml", i);
        let path = output_dir.join(filename);
        let content = to_toml(scenario);
        std::fs::write(&path, content)
            .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
    }

    Ok(scenarios.len())
}
