//! BFS JSONL analysis report.
//!
//! Reads BFS exploration output and prints a structured analysis including
//! overview stats, action distribution, class system analysis (decoded from
//! aggregate tokens using the same layout as `tokens.rs`), skill diversity,
//! diagnostics, and progression timeline.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use bevy_game::headless_campaign::bfs_explore::BfsSample;
use bevy_game::headless_campaign::tokens::EntityToken;

// ---------------------------------------------------------------------------
// Aggregate token (type_id=9) feature layout — mirrors tokens.rs exactly.
// Keep in sync with CampaignState::aggregate_token().
// ---------------------------------------------------------------------------

// Basic features (0-9)
#[allow(dead_code)]
const AGG_TICK_NORM: usize = 0;

// Extended state (10-21)
// System trackers pass 3 (22-50)
// Pass 4 expanded systems (51-71)

// Layout: 0-9 basic (10), 10-21 extended (12), 22-67 pass 3 trackers (46),
//         68-97 pass 4 expanded (30), 98-112 class system (15),
//         113-118 class progression detail (6).
//
// Class System — 15 features starting at index 98
const AGG_CLASS_TOTAL_CLASSES: usize = 98;
const AGG_CLASS_MEAN_LEVEL: usize = 99;
const AGG_CLASS_MAX_LEVEL: usize = 100;
const AGG_CLASS_TOTAL_SKILLS: usize = 101;
const AGG_CLASS_UNIQUE_CLASSES: usize = 102;
const AGG_CLASS_SHAME_CLASSES: usize = 103;
#[allow(dead_code)]
const AGG_CLASS_CRISIS_UNIQUE: usize = 104;
#[allow(dead_code)]
const AGG_CLASS_CONSOLIDATION: usize = 105;
#[allow(dead_code)]
const AGG_CLASS_IDENTITY_COHERENCE: usize = 106;
#[allow(dead_code)]
const AGG_CLASS_STAGNATION: usize = 107;
#[allow(dead_code)]
const AGG_CLASS_DIVERSITY_RATIO: usize = 108;
#[allow(dead_code)]
const AGG_CLASS_BEHAVIOR_INTENSITY: usize = 109;
#[allow(dead_code)]
const AGG_CLASS_MULTICLASS_RATIO: usize = 110;
#[allow(dead_code)]
const AGG_CLASS_EVOLUTION_COUNT: usize = 111;

// Class progression detail — 6 features starting at index 113
const AGG_CLASS_SKILLS_WITH_DSL: usize = 113;
#[allow(dead_code)]
const AGG_CLASS_RARE_SKILL_RATIO: usize = 114;
const AGG_CLASS_STARTER_COUNT: usize = 115;
const AGG_CLASS_EVOLVED_COUNT: usize = 116;
const AGG_CLASS_BEHAVIOR_SPEC: usize = 117;
const AGG_CLASS_LV5_RATIO: usize = 118;

// Denormalization constants (inverse of what tokens.rs applies)
const TOTAL_CLASSES_NORM: f32 = 50.0;
const MEAN_LEVEL_NORM: f32 = 25.0;
const MAX_LEVEL_NORM: f32 = 30.0;
const TOTAL_SKILLS_NORM: f32 = 100.0;
const UNIQUE_CLASSES_NORM: f32 = 20.0;
const SHAME_CLASSES_NORM: f32 = 10.0;
const SKILLS_WITH_DSL_NORM: f32 = 50.0;
const STARTER_COUNT_NORM: f32 = 20.0;
const EVOLVED_COUNT_NORM: f32 = 20.0;

// ---------------------------------------------------------------------------
// Helper: get aggregate token from a token list
// ---------------------------------------------------------------------------

fn find_aggregate(tokens: &[EntityToken]) -> Option<&EntityToken> {
    tokens.iter().find(|t| t.type_id == 9)
}

fn agg_f(tok: &EntityToken, idx: usize) -> f32 {
    tok.features.get(idx).copied().unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn run(path: &str) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Error: cannot open {}: {}", path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);

    let mut samples: Vec<BfsSample> = Vec::new();
    let mut skipped_first = false;

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Warning: read error at line {}: {}", line_no + 1, e);
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Skip config header (first non-empty line)
        if !skipped_first {
            // Try to parse as BfsSample; if it fails, it's the config line
            if serde_json::from_str::<BfsSample>(trimmed).is_err() {
                skipped_first = true;
                continue;
            }
            skipped_first = true;
            // Falls through to parse as sample
        }

        match serde_json::from_str::<BfsSample>(trimmed) {
            Ok(s) => samples.push(s),
            Err(e) => {
                if line_no < 5 {
                    eprintln!("Warning: parse error at line {}: {}", line_no + 1, e);
                }
            }
        }
    }

    if samples.is_empty() {
        eprintln!("No samples found in {}", path);
        return;
    }

    print_overview(&samples);
    print_actions(&samples);
    print_class_system(&samples);
    print_skills_used(&samples);
    print_diagnostics(&samples);
    print_progression_timeline(&samples);
}

// ---------------------------------------------------------------------------
// 1. Overview
// ---------------------------------------------------------------------------

fn print_overview(samples: &[BfsSample]) {
    let n = samples.len();
    let ticks: Vec<u64> = samples.iter().map(|s| s.root_tick).collect();
    let tick_min = ticks.iter().copied().min().unwrap_or(0);
    let tick_max = ticks.iter().copied().max().unwrap_or(0);

    let values: Vec<f32> = samples.iter().map(|s| s.leaf_value).collect();
    let val_min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let val_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let val_mean = values.iter().sum::<f32>() / n as f32;

    let victories = samples
        .iter()
        .filter(|s| {
            s.leaf_outcome
                .as_deref()
                .map_or(false, |o| o.contains("Victory") || o.contains("victory"))
        })
        .count();
    let defeats = samples
        .iter()
        .filter(|s| {
            s.leaf_outcome
                .as_deref()
                .map_or(false, |o| o.contains("Defeat") || o.contains("defeat") || o.contains("Loss"))
        })
        .count();

    println!("=== BFS Analysis Report ===");
    println!("Samples: {}", n);
    println!("Tick range: {} - {}", tick_min, tick_max);
    println!(
        "Value: min={:.2} max={:.2} mean={:.2}",
        val_min, val_max, val_mean
    );

    // Tick distribution by 5000-tick buckets (no phase tags)
    let mut tick_buckets: HashMap<u64, usize> = HashMap::new();
    for s in samples.iter() {
        *tick_buckets.entry(s.root_tick / 5000).or_default() += 1;
    }
    let mut sorted: Vec<_> = tick_buckets.iter().collect();
    sorted.sort_by_key(|(&k, _)| k);
    print!("Ticks:");
    for (&bucket, &count) in &sorted {
        print!(" {}-{}={:.0}%", bucket * 5000, (bucket + 1) * 5000, count as f64 / n as f64 * 100.0);
    }
    println!();
    {
    }
    println!("Terminal: {} victories, {} defeats", victories, defeats);
    println!();
}

// ---------------------------------------------------------------------------
// 2. Action distribution
// ---------------------------------------------------------------------------

fn print_actions(samples: &[BfsSample]) {
    let n = samples.len();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in samples {
        *counts.entry(s.action_type.as_str()).or_default() += 1;
    }

    let mut sorted: Vec<(&&str, &usize)> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("=== Actions ===");
    for (action, count) in &sorted {
        println!(
            "  {:24} {:>6} ({:>5.1}%)",
            action,
            count,
            **count as f64 / n as f64 * 100.0
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// 3. Class system analysis
// ---------------------------------------------------------------------------

fn print_class_system(samples: &[BfsSample]) {
    // Collect aggregate token features from leaf tokens
    let mut total_classes_sum = 0.0_f64;
    let mut mean_level_sum = 0.0_f64;
    let mut max_level_max = 0.0_f32;
    let mut total_skills_sum = 0.0_f64;
    let mut unique_classes_sum = 0.0_f64;
    let mut shame_classes_sum = 0.0_f64;
    let mut evolved_sum = 0.0_f64;
    let mut starter_sum = 0.0_f64;
    let mut skills_dsl_sum = 0.0_f64;
    let mut lv5_ratio_sum = 0.0_f64;
    let mut behavior_spec_sum = 0.0_f64;
    let mut count = 0u64;

    for s in samples {
        if let Some(agg) = find_aggregate(&s.leaf_tokens) {
            // Denormalize
            let total_classes = agg_f(agg, AGG_CLASS_TOTAL_CLASSES) * TOTAL_CLASSES_NORM;
            let mean_level = agg_f(agg, AGG_CLASS_MEAN_LEVEL) * MEAN_LEVEL_NORM;
            let max_level = agg_f(agg, AGG_CLASS_MAX_LEVEL) * MAX_LEVEL_NORM;
            let total_skills = agg_f(agg, AGG_CLASS_TOTAL_SKILLS) * TOTAL_SKILLS_NORM;
            let unique_classes = agg_f(agg, AGG_CLASS_UNIQUE_CLASSES) * UNIQUE_CLASSES_NORM;
            let shame_classes = agg_f(agg, AGG_CLASS_SHAME_CLASSES) * SHAME_CLASSES_NORM;
            let evolved = agg_f(agg, AGG_CLASS_EVOLVED_COUNT) * EVOLVED_COUNT_NORM;
            let starters = agg_f(agg, AGG_CLASS_STARTER_COUNT) * STARTER_COUNT_NORM;
            let skills_dsl = agg_f(agg, AGG_CLASS_SKILLS_WITH_DSL) * SKILLS_WITH_DSL_NORM;
            let lv5_ratio = agg_f(agg, AGG_CLASS_LV5_RATIO);
            let behavior_spec = agg_f(agg, AGG_CLASS_BEHAVIOR_SPEC);

            total_classes_sum += total_classes as f64;
            mean_level_sum += mean_level as f64;
            if max_level > max_level_max {
                max_level_max = max_level;
            }
            total_skills_sum += total_skills as f64;
            unique_classes_sum += unique_classes as f64;
            shame_classes_sum += shame_classes as f64;
            evolved_sum += evolved as f64;
            starter_sum += starters as f64;
            skills_dsl_sum += skills_dsl as f64;
            lv5_ratio_sum += lv5_ratio as f64;
            behavior_spec_sum += behavior_spec as f64;
            count += 1;
        }
    }

    if count == 0 {
        println!("=== Class System ===");
        println!("  (no aggregate tokens found)");
        println!();
        return;
    }

    let n = count as f64;
    println!("=== Class System ===");
    println!("  Total classes:     {:>6.1}", total_classes_sum / n);
    println!("  Mean class level:  {:>6.1}", mean_level_sum / n);
    println!("  Max class level:   {:>6.0}", max_level_max);
    println!("  Total skills:      {:>6.1}", total_skills_sum / n);
    println!("  Unique classes:    {:>6.1}", unique_classes_sum / n);
    println!("  Shame classes:     {:>6.1}", shame_classes_sum / n);
    println!("  Evolved classes:   {:>6.1}", evolved_sum / n);
    println!("  Starter classes:   {:>6.1}", starter_sum / n);
    println!("  Skills with DSL:   {:>6.1}", skills_dsl_sum / n);
    println!("  Lv5+ ratio:        {:>6.2}", lv5_ratio_sum / n);
    println!(
        "  Behavior axes active: {:.0}/12",
        behavior_spec_sum / n * 12.0
    );
    println!();
}

// ---------------------------------------------------------------------------
// 4. Skills used
// ---------------------------------------------------------------------------

fn print_skills_used(samples: &[BfsSample]) {
    let mut skill_counts: HashMap<String, usize> = HashMap::new();
    let mut total_uses = 0usize;

    for s in samples {
        if s.action_type == "UseClassSkill" {
            // action_detail typically contains the skill name
            let name = s.action_detail.clone();
            *skill_counts.entry(name).or_default() += 1;
            total_uses += 1;
        }
    }

    println!("=== Skills Used ===");
    if skill_counts.is_empty() {
        println!("  (none)");
    } else {
        let mut sorted: Vec<(String, usize)> = skill_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let display_count = sorted.len().min(20);
        for (name, count) in &sorted[..display_count] {
            println!("  {}: {}", name, count);
        }
        if sorted.len() > 20 {
            println!("  ... and {} more", sorted.len() - 20);
        }
        println!(
            "  Unique: {}, Total uses: {}",
            sorted.len(),
            total_uses
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// 5. Diagnostics
// ---------------------------------------------------------------------------

fn print_diagnostics(samples: &[BfsSample]) {
    let n = samples.len();
    let mut flag_counts: HashMap<String, usize> = HashMap::new();

    for s in samples {
        if s.diagnostics.is_empty() {
            continue;
        }
        for flag in s.diagnostics.split(',') {
            let flag = flag.trim();
            if !flag.is_empty() {
                *flag_counts.entry(flag.to_string()).or_default() += 1;
            }
        }
    }

    println!("=== Diagnostics ===");
    if flag_counts.is_empty() {
        println!("  (no diagnostic flags set)");
    } else {
        let mut sorted: Vec<(String, usize)> = flag_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (flag, count) in &sorted {
            println!(
                "  {}: {}/{} ({:.1}%)",
                flag,
                count,
                n,
                *count as f64 / n as f64 * 100.0
            );
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// 6. Progression timeline
// ---------------------------------------------------------------------------

fn print_progression_timeline(samples: &[BfsSample]) {
    let ranges: &[(u64, u64, &str)] = &[
        (0, 2000, "    0-2000"),
        (2000, 5000, " 2000-5000"),
        (5000, 10000, " 5000-10000"),
        (10000, u64::MAX, "10000+"),
    ];

    println!("=== Progression Timeline ===");
    println!(
        "  {:>14}  {:>8}  {:>4}  {:>6}  {:>6}",
        "Tick", "mean_lv", "max", "skills", "unique"
    );

    for &(lo, hi, label) in ranges {
        let bucket: Vec<&BfsSample> = samples
            .iter()
            .filter(|s| s.root_tick >= lo && s.root_tick < hi)
            .collect();

        if bucket.is_empty() {
            continue;
        }

        let mut mean_levels = Vec::new();
        let mut max_level: f32 = 0.0;
        let mut skills_sum = 0.0_f64;
        let mut unique_sum = 0.0_f64;
        let mut agg_count = 0u64;

        for s in &bucket {
            // Use root tokens for "state at this tick range"
            if let Some(agg) = find_aggregate(&s.root_tokens) {
                let ml = agg_f(agg, AGG_CLASS_MEAN_LEVEL) * MEAN_LEVEL_NORM;
                let mx = agg_f(agg, AGG_CLASS_MAX_LEVEL) * MAX_LEVEL_NORM;
                let sk = agg_f(agg, AGG_CLASS_TOTAL_SKILLS) * TOTAL_SKILLS_NORM;
                let uq = agg_f(agg, AGG_CLASS_UNIQUE_CLASSES) * UNIQUE_CLASSES_NORM;

                mean_levels.push(ml);
                if mx > max_level {
                    max_level = mx;
                }
                skills_sum += sk as f64;
                unique_sum += uq as f64;
                agg_count += 1;
            }
        }

        if agg_count == 0 {
            continue;
        }

        let avg_level =
            mean_levels.iter().sum::<f32>() / mean_levels.len() as f32;
        let avg_skills = skills_sum / agg_count as f64;
        let avg_unique = unique_sum / agg_count as f64;

        println!(
            "  Tick {:>10}: mean_lv={:.1}, max={:.0}, skills={:.0}, unique={:.0}",
            label, avg_level, max_level, avg_skills, avg_unique
        );
    }
    println!();
}
