//! Generate scenario TOML files for training corpus.
//!
//! Usage:
//!   cargo run --bin gen_scenarios -- [--count N] [--output-dir DIR]

use clap::Parser;
use std::fs;
use std::path::PathBuf;

const ROOM_TYPES: [&str; 6] = ["Entry", "Pressure", "Pivot", "Setpiece", "Recovery", "Climax"];
const ENEMY_COUNTS: [u32; 7] = [2, 3, 4, 5, 6, 8, 10];
const HERO_TEMPLATES: [&[&str]; 12] = [
    // Mixed parties
    &["Warrior", "Cleric", "Rogue"],
    &["Warrior", "Mage", "Ranger", "Cleric"],
    &["Paladin", "Mage", "Rogue", "Cleric"],
    &["Warrior", "Ranger", "Mage"],
    &["Paladin", "Cleric", "Ranger", "Rogue"],
    &["Warrior", "Paladin", "Mage", "Cleric", "Rogue", "Ranger"],
    // Duplicate-hero parties
    &["Warrior", "Cleric", "Cleric", "Cleric"],
    &["Mage", "Mage", "Cleric", "Rogue"],
    &["Paladin", "Paladin", "Cleric", "Mage"],
    &["Warrior", "Warrior", "Warrior"],
    &["Rogue", "Rogue", "Rogue", "Cleric"],
    &["Ranger", "Ranger", "Mage", "Cleric"],
];

#[derive(Parser)]
struct Args {
    /// Number of scenarios to generate.
    #[arg(long, default_value_t = 50)]
    count: usize,

    /// Directory to write generated TOML files into.
    #[arg(long, default_value = "scenarios/generated")]
    output_dir: PathBuf,
}

fn main() {
    let args = Args::parse();

    fs::create_dir_all(&args.output_dir).expect("failed to create output directory");

    for i in 0..args.count {
        let room_type = ROOM_TYPES[i % 6];
        let seed = i * 7919 + 1;
        // Use coprime-ish strides so template, enemy count, and difficulty
        // cycle independently instead of locking to the same index.
        let templates = HERO_TEMPLATES[i % HERO_TEMPLATES.len()];
        let hero_count = templates.len() as u32;
        let enemy_count = ENEMY_COUNTS[(i * 3) % 7];
        let difficulty = (i * 2 % 3) + 1; // 1-3; heroes don't scale so cap difficulty
        let max_ticks = (2000 + (hero_count + enemy_count) as usize * 200).clamp(2000, 5000);
        let name = format!("Gen {room_type} {hero_count}v{enemy_count} d{difficulty} s{seed}");
        let templates_toml: Vec<String> = templates.iter().map(|t| format!("\"{}\"", t)).collect();
        let templates_line = format!("hero_templates = [{}]", templates_toml.join(", "));

        let toml = format!(
            "\
[scenario]
name        = \"{name}\"
seed        = {seed}
hero_count  = {hero_count}
enemy_count = {enemy_count}
difficulty  = {difficulty}
max_ticks   = {max_ticks}
room_type   = \"{room_type}\"
{templates_line}

[assert]
outcome = \"Either\"
"
        );

        let path = args.output_dir.join(format!("gen_{i:04}.toml"));
        fs::write(&path, toml).expect("failed to write scenario file");
    }

    println!(
        "Generated {} scenarios in {}",
        args.count,
        args.output_dir.display()
    );
}
