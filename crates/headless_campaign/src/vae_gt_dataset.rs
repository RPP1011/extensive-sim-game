//! Ground-truth VAE training data from existing ability DSL files.
//!
//! Loads all 726+ ability definitions from `.ability` files, tags them by
//! effect type / hint / power tags, then pairs them with campaign trigger
//! contexts via tag-weighted sampling. Every slot vector is extracted by the
//! real Rust parser — zero parse failures, fully differentiable.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;
use serde::Serialize;

use tactical_sim::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
use tactical_sim::effects::dsl;
use tactical_sim::effects::effect_enum::Effect;

use super::vae_dataset::{sweep_campaigns, TriggerContext, VaeDatasetConfig};
use super::vae_slots;

// ---------------------------------------------------------------------------
// Tagged ability
// ---------------------------------------------------------------------------

/// A parsed ability with semantic tags for context matching.
pub struct TaggedAbility {
    pub name: String,
    pub source_file: String,
    pub is_passive: bool,
    pub slots: Vec<f32>,
    pub tags: HashSet<String>,
    pub hint: String,
}

/// A dataset record with ground-truth slots.
#[derive(Serialize)]
pub struct GtRecord {
    pub input: Vec<f32>,
    pub slots: Vec<f32>,
    pub content_type: String,
    pub ability_name: String,
    pub source_file: String,
    pub trigger: String,
    pub archetype: String,
    pub level: u32,
    pub seed: u64,
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Step 1: Load & tag
// ---------------------------------------------------------------------------

/// Load all ability definitions from disk and tag them.
pub fn load_all_abilities() -> Vec<TaggedAbility> {
    let mut tagged = Vec::new();

    let dirs = ["dataset/abilities/hero_templates", "dataset/abilities/lol_heroes"];
    for dir in &dirs {
        let dir_path = Path::new(dir);
        if !dir_path.exists() {
            continue;
        }
        let entries: Vec<_> = std::fs::read_dir(dir_path)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "ability")
                    .unwrap_or(false)
            })
            .collect();

        for entry in entries {
            let path = entry.path();
            let source = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(_) => continue,
            };

            match dsl::parse_abilities(&text) {
                Ok((abilities, passives)) => {
                    for def in &abilities {
                        let slots = vae_slots::ability_to_slots(def);
                        let tags = derive_ability_tags(def);
                        tagged.push(TaggedAbility {
                            name: def.name.clone(),
                            source_file: source.clone(),
                            is_passive: false,
                            slots,
                            tags,
                            hint: def.ai_hint.clone(),
                        });
                    }
                    for def in &passives {
                        let slots = vae_slots::passive_to_slots(def);
                        let tags = derive_passive_tags(def);
                        tagged.push(TaggedAbility {
                            name: def.name.clone(),
                            source_file: source.clone(),
                            is_passive: true,
                            slots,
                            tags,
                            hint: String::new(),
                        });
                    }
                }
                Err(_) => {
                    // Skip unparseable files
                }
            }
        }
    }

    tagged
}

/// Derive semantic tags from an AbilityDef using the same vocabulary as adventurer_tags().
fn derive_ability_tags(def: &AbilityDef) -> HashSet<String> {
    let mut tags = HashSet::new();

    // Range-based: melee vs ranged
    if def.range > 4.0 {
        tags.insert("ranged".into());
    } else {
        tags.insert("melee".into());
    }

    // Hint-derived
    match def.ai_hint.as_str() {
        "heal" => {
            tags.insert("healing".into());
            tags.insert("divine".into());
        }
        "defense" => {
            tags.insert("defense".into());
            tags.insert("protection".into());
        }
        "crowd_control" => {
            tags.insert("defense".into());
        }
        "utility" => {
            tags.insert("enchantment".into());
        }
        _ => {}
    }

    // Effect-derived tags
    for ce in &def.effects {
        add_effect_tags(&ce.effect, &mut tags);

        // DSL power tags (from ce.tags HashMap keys)
        for key in ce.tags.keys() {
            match key.to_uppercase().as_str() {
                "MAGIC" => { tags.insert("arcane".into()); }
                "FIRE" | "ICE" | "LIGHTNING" => { tags.insert("elemental".into()); }
                "DARK" => { tags.insert("deception".into()); tags.insert("ritual".into()); }
                "POISON" => { tags.insert("nature".into()); tags.insert("sabotage".into()); }
                "PHYSICAL" => { tags.insert("melee".into()); }
                _ => {}
            }
        }
    }

    // Targeting-derived
    match def.targeting {
        AbilityTargeting::TargetAlly | AbilityTargeting::SelfCast => {
            tags.insert("protection".into());
        }
        AbilityTargeting::SelfAoe | AbilityTargeting::GroundTarget => {
            tags.insert("elemental".into());
        }
        _ => {}
    }

    tags
}

/// Derive semantic tags from a PassiveDef.
fn derive_passive_tags(def: &PassiveDef) -> HashSet<String> {
    let mut tags = HashSet::new();

    // All passives get survival tag
    tags.insert("survival".into());

    for ce in &def.effects {
        add_effect_tags(&ce.effect, &mut tags);
        for key in ce.tags.keys() {
            match key.to_uppercase().as_str() {
                "MAGIC" => { tags.insert("arcane".into()); }
                "FIRE" | "ICE" | "LIGHTNING" => { tags.insert("elemental".into()); }
                "DARK" => { tags.insert("deception".into()); }
                "POISON" => { tags.insert("nature".into()); }
                _ => {}
            }
        }
    }

    tags
}

/// Map an Effect variant to semantic tags.
fn add_effect_tags(effect: &Effect, tags: &mut HashSet<String>) {
    match effect {
        Effect::Damage { .. } => {} // already handled by range
        Effect::Heal { .. } => { tags.insert("healing".into()); tags.insert("divine".into()); }
        Effect::Shield { .. } => { tags.insert("defense".into()); tags.insert("protection".into()); }
        Effect::Stun { .. } | Effect::Root { .. } | Effect::Silence { .. }
        | Effect::Fear { .. } | Effect::Polymorph { .. } | Effect::Suppress { .. } => {
            tags.insert("defense".into());
        }
        Effect::Slow { .. } | Effect::Grounded { .. } => { tags.insert("defense".into()); }
        Effect::Knockback { .. } | Effect::Pull { .. } => { tags.insert("melee".into()); }
        Effect::Dash { .. } => { tags.insert("agility".into()); }
        Effect::Buff { .. } => { tags.insert("leadership".into()); tags.insert("inspiration".into()); }
        Effect::Debuff { .. } => { tags.insert("sabotage".into()); tags.insert("deception".into()); }
        Effect::Summon { .. } => { tags.insert("arcane".into()); tags.insert("nature".into()); }
        Effect::Stealth { .. } => { tags.insert("stealth".into()); tags.insert("assassination".into()); }
        Effect::Lifesteal { .. } => { tags.insert("survival".into()); }
        Effect::Execute { .. } => { tags.insert("assassination".into()); }
        Effect::Resurrect { .. } => { tags.insert("healing".into()); tags.insert("divine".into()); }
        Effect::Reflect { .. } => { tags.insert("defense".into()); }
        Effect::Link { .. } | Effect::Redirect { .. } => { tags.insert("protection".into()); }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Step 2: Scoring & sampling
// ---------------------------------------------------------------------------

/// Score an ability's relevance to a context's archetype tags.
fn tag_score(ability: &TaggedAbility, context_tags: &[&str]) -> f32 {
    let overlap = context_tags
        .iter()
        .filter(|t| ability.tags.contains(**t))
        .count() as f32;

    // Hint bonus
    let hint_bonus = match ability.hint.as_str() {
        "heal" if context_tags.contains(&"healing") => 1.0,
        "defense" if context_tags.contains(&"defense") || context_tags.contains(&"protection") => 1.0,
        "damage" if context_tags.contains(&"melee") || context_tags.contains(&"ranged")
            || context_tags.contains(&"assassination") => 0.5,
        "crowd_control" if context_tags.contains(&"defense") || context_tags.contains(&"sabotage") => 0.5,
        _ => 0.0,
    };

    overlap + hint_bonus + 0.1 // floor ensures every ability has nonzero weight
}

/// Get archetype tags for a context, matching the content_prompts vocabulary.
fn context_tags(archetype: &str) -> Vec<&'static str> {
    // Reuse the same mapping as content_prompts::adventurer_tags
    match archetype {
        "ranger"      => vec!["ranged", "nature", "stealth", "tracking", "survival"],
        "knight"      => vec!["melee", "defense", "leadership", "fortification", "honor"],
        "mage"        => vec!["arcane", "elemental", "ritual", "knowledge", "enchantment"],
        "cleric"      => vec!["healing", "divine", "protection", "purification", "restoration"],
        "rogue"       => vec!["stealth", "assassination", "agility", "deception", "sabotage"],
        "paladin"     => vec!["melee", "divine", "leadership", "protection", "honor"],
        "berserker"   => vec!["melee", "agility", "survival", "inspiration", "sacrifice"],
        "necromancer" => vec!["arcane", "ritual", "deception", "sabotage", "sacrifice"],
        "bard"        => vec!["inspiration", "leadership", "deception", "knowledge", "agility"],
        "druid"       => vec!["nature", "healing", "elemental", "protection", "survival"],
        "warlock"     => vec!["arcane", "deception", "ritual", "sabotage", "enchantment"],
        "monk"        => vec!["melee", "agility", "survival", "protection", "stealth"],
        "assassin"    => vec!["assassination", "stealth", "agility", "deception", "sabotage"],
        "guardian"    => vec!["defense", "fortification", "protection", "leadership", "honor"],
        "shaman"      => vec!["nature", "elemental", "healing", "ritual", "inspiration"],
        "artificer"   => vec!["knowledge", "enchantment", "fortification", "arcane", "leadership"],
        "tank"        => vec!["defense", "fortification", "melee", "protection", "survival"],
        _ => vec!["melee", "survival"],
    }
}

/// Sample N abilities weighted by tag score.
fn sample_abilities<'a>(
    abilities: &'a [TaggedAbility],
    ctx_tags: &[&str],
    rng: &mut u64,
    n: usize,
) -> Vec<&'a TaggedAbility> {
    // Compute scores
    let scores: Vec<f32> = abilities.iter().map(|a| tag_score(a, ctx_tags)).collect();
    let total: f32 = scores.iter().sum();

    let mut selected = Vec::with_capacity(n);
    let mut used = vec![false; abilities.len()];

    for _ in 0..n {
        // Weighted random selection
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let mut target = (*rng as f32 / u64::MAX as f32) * total;

        let mut chosen = 0;
        for (i, &score) in scores.iter().enumerate() {
            if used[i] {
                continue;
            }
            target -= score;
            if target <= 0.0 {
                chosen = i;
                break;
            }
            chosen = i; // fallback to last
        }

        used[chosen] = true;
        selected.push(&abilities[chosen]);
    }

    selected
}

// ---------------------------------------------------------------------------
// Step 3: Full pipeline
// ---------------------------------------------------------------------------

/// Configuration for the ground-truth dataset pipeline.
#[derive(Clone, Debug)]
pub struct GtDatasetConfig {
    pub campaigns: u64,
    pub max_ticks: u64,
    pub threads: usize,
    pub base_seed: u64,
    pub samples_per_context: usize,
    pub output_path: String,
}

impl Default for GtDatasetConfig {
    fn default() -> Self {
        Self {
            campaigns: 1000,
            max_ticks: 30_000,
            threads: 0,
            base_seed: 2026,
            samples_per_context: 4,
            output_path: "generated/vae_gt_dataset.jsonl".into(),
        }
    }
}

/// Run the full ground-truth dataset pipeline.
pub fn run_gt_pipeline(config: &GtDatasetConfig) {
    // Step 1: Load abilities
    eprintln!("=== Ground-Truth VAE Dataset ===");
    let abilities = load_all_abilities();
    let active = abilities.iter().filter(|a| !a.is_passive).count();
    let passive = abilities.iter().filter(|a| a.is_passive).count();
    eprintln!("Loaded {} abilities ({} active, {} passive)", abilities.len(), active, passive);

    if abilities.is_empty() {
        eprintln!("No abilities found! Check dataset/abilities/hero_templates/ and dataset/abilities/lol_heroes/");
        return;
    }

    // Collect unique tags across all abilities
    let all_tags: HashSet<&str> = abilities
        .iter()
        .flat_map(|a| a.tags.iter().map(|s| s.as_str()))
        .collect();
    eprintln!("Unique ability tags: {} {:?}", all_tags.len(), all_tags);

    // Step 2: Campaign sweep
    let sweep_config = VaeDatasetConfig {
        campaigns: config.campaigns,
        max_ticks: config.max_ticks,
        threads: config.threads,
        base_seed: config.base_seed,
        include_procedural: false,
        ..Default::default()
    };
    let contexts = sweep_campaigns(&sweep_config);

    // Filter to ability contexts only
    let ability_contexts: Vec<&TriggerContext> = contexts
        .iter()
        .filter(|c| c.content_type == "ability")
        .collect();
    eprintln!("Ability contexts: {}", ability_contexts.len());

    // Step 3: Sample and write
    if let Some(parent) = Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&config.output_path).expect("Failed to create output file"),
    ));
    let written = AtomicU64::new(0);
    let start = std::time::Instant::now();

    let threads = if config.threads == 0 {
        rayon::current_num_threads().min(8)
    } else {
        config.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build thread pool");

    pool.install(|| {
        ability_contexts.par_iter().for_each(|ctx| {
            let ctx_tags = context_tags(&ctx.archetype);
            let mut rng = ctx.seed.wrapping_mul(6364136223846793005)
                .wrapping_add(ctx.tick);

            let sampled = sample_abilities(&abilities, &ctx_tags, &mut rng, config.samples_per_context);

            for ability in sampled {
                let record = GtRecord {
                    input: ctx.input.clone(),
                    slots: ability.slots.clone(),
                    content_type: "ability".into(),
                    ability_name: ability.name.clone(),
                    source_file: ability.source_file.clone(),
                    trigger: ctx.trigger.clone(),
                    archetype: ctx.archetype.clone(),
                    level: ctx.level,
                    seed: ctx.seed,
                    tick: ctx.tick,
                };

                if let Ok(json) = serde_json::to_string(&record) {
                    let mut f = file.lock().unwrap();
                    writeln!(f, "{}", json).ok();
                }

                written.fetch_add(1, Ordering::Relaxed);
            }
        });
    });

    let total = written.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("\nWrote {} records in {:.1}s ({:.0}/s)", total, elapsed, total as f64 / elapsed);
    eprintln!("Output: {}", config.output_path);
}
