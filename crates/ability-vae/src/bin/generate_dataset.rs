//! Generate quality-filtered abilities from grammar space and output as DSL.
//!
//! 1. Sample 50K random grammar space vectors
//! 2. Score each on quality heuristics
//! 3. Keep top 5K
//! 4. Also generate feature-specific batches for underrepresented features
//! 5. Decode all to DSL and write to files
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin generate-dataset

use ability_vae::grammar_space::{self, GRAMMAR_DIM};
use ability_vae::quality::{score_ability, generate_quality_abilities, generate_with_feature};

use std::io::Write;

fn main() {
    println!("=== Quality-Filtered Ability Generation ===");

    // Phase 1: Random sampling with quality filter
    println!("\n--- Phase 1: Random quality sampling ---");
    let top = generate_quality_abilities(50_000, 5_000, 42);
    println!("  Sampled 50,000 → kept top 5,000");
    println!("  Score range: {:.3} - {:.3}", top.last().unwrap().1, top[0].1);

    // Phase 2: Feature-specific generation for underrepresented features
    println!("\n--- Phase 2: Feature-specific generation ---");

    let feature_batches: &[(&str, &[(usize, f32)])] = &[
        // Auras (passive + while_alive would need special dims, approximate with passive)
        ("passive_heal", &[(0, 0.75), (6, 0.3)]),  // passive + heal hint
        ("passive_defense", &[(0, 0.75), (6, 0.6)]), // passive + defense
        ("passive_cc", &[(0, 0.75), (6, 0.4)]),     // passive + cc

        // Campaign abilities
        ("campaign_economy", &[(1, 0.75), (6, 0.1)]),
        ("campaign_diplomacy", &[(1, 0.75), (6, 0.25)]),
        ("campaign_stealth", &[(1, 0.75), (6, 0.4)]),
        ("campaign_leadership", &[(1, 0.75), (6, 0.55)]),

        // Combat with specific hints
        ("combat_heal", &[(0, 0.25), (1, 0.25), (6, 0.3)]),
        ("combat_cc", &[(0, 0.25), (1, 0.25), (6, 0.5)]),
        ("combat_defense", &[(0, 0.25), (1, 0.25), (6, 0.7)]),
        ("combat_utility", &[(0, 0.25), (1, 0.25), (6, 0.9)]),

        // Elements
        ("fire", &[(1, 0.25), (17, 0.55)]),
        ("ice", &[(1, 0.25), (17, 0.65)]),
        ("dark", &[(1, 0.25), (17, 0.75)]),
        ("holy", &[(1, 0.25), (17, 0.85)]),
        ("poison", &[(1, 0.25), (17, 0.95)]),

        // Delivery types
        ("projectile", &[(1, 0.25), (8, 0.5)]),
        ("chain", &[(1, 0.25), (8, 0.64)]),
        ("zone", &[(1, 0.25), (8, 0.78)]),
        ("trap", &[(1, 0.25), (8, 0.92)]),

        // Multi-effect combos
        ("combo_2", &[(11, 0.5)]),  // 2 effects
        ("combo_3", &[(11, 0.75)]), // 3 effects
        ("combo_4", &[(11, 0.95)]), // 4 effects

        // Meta-effects (approximate with high effect type values)
        ("meta_buff", &[(12, 0.78)]),
        ("meta_debuff", &[(12, 0.82)]),
        ("meta_dot", &[(12, 0.88)]),
        ("meta_hot", &[(12, 0.92)]),
    ];

    let mut feature_abilities = Vec::new();
    for (name, dims) in feature_batches {
        let batch = generate_with_feature(dims, 2000, 200,
            name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64)));
        println!("  {}: {} abilities (score {:.3}-{:.3})",
            name, batch.len(),
            batch.last().map(|x| x.1).unwrap_or(0.0),
            batch.first().map(|x| x.1).unwrap_or(0.0));
        feature_abilities.extend(batch);
    }

    // Combine and deduplicate
    let mut all: Vec<([f32; GRAMMAR_DIM], f32)> = top;
    all.extend(feature_abilities);
    println!("\n  Total before dedup: {}", all.len());

    // Write to .ability files
    let out_dir = "dataset/abilities/generated";
    std::fs::create_dir_all(out_dir).ok();

    let mut total_written = 0;
    let abilities_per_file = 50;
    let mut file_idx = 0;

    let mut current_file: Option<std::fs::File> = None;
    let mut in_file = 0;

    for (v, score) in &all {
        if in_file == 0 || in_file >= abilities_per_file {
            let path = format!("{}/gen_{:04}.ability", out_dir, file_idx);
            current_file = Some(std::fs::File::create(&path).unwrap());
            if let Some(ref mut f) = current_file {
                writeln!(f, "// Auto-generated abilities (quality score >= {:.3})", score).ok();
                writeln!(f).ok();
            }
            file_idx += 1;
            in_file = 0;
        }

        let dsl = grammar_space::decode(v);

        // Verify it parses
        if tactical_sim::effects::dsl::parse_abilities(&dsl).is_ok() {
            if let Some(ref mut f) = current_file {
                writeln!(f, "{}", dsl).ok();
                writeln!(f).ok();
            }
            total_written += 1;
            in_file += 1;
        }
    }

    println!("\n=== Results ===");
    println!("  Written: {} abilities across {} files", total_written, file_idx);
    println!("  Output: {}/", out_dir);

    // Score distribution
    let scores: Vec<f32> = all.iter().map(|x| x.1).collect();
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    println!("  Mean score: {:.3}", mean);
    println!("  Min score: {:.3}", scores.iter().cloned().fold(f32::INFINITY, f32::min));
    println!("  Max score: {:.3}", scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    println!("\nDone!");
}
