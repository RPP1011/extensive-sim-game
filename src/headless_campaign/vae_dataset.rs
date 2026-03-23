//! VAE dataset collection pipeline.
//!
//! Three-step process:
//! 1. **Sweep**: Run campaigns, record trigger contexts + 124-dim input vectors
//! 2. **Generate**: Batch-call Ollama to produce ability/class DSL for each context
//! 3. **Extract**: Parse LLM outputs, extract slot vectors, write final dataset
//!
//! Items and quests use procedural generation (no LLM needed).

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::actions::*;
use super::batch::heuristic_policy;
use super::config::CampaignConfig;
use super::llm::{self, ContentStore, LlmConfig};
use super::state::*;
use super::step::step_campaign;
use super::vae_features;
use super::vae_slots;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A trigger context recorded during campaign sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriggerContext {
    pub input: Vec<f32>,
    pub trigger: String,
    pub content_type: String,
    pub seed: u64,
    pub tick: u64,
    pub adv_id: u32,
    pub archetype: String,
    pub level: u32,
    pub context_text: String,
}

/// A final dataset record with input vector + slot vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub input: Vec<f32>,
    pub slots: Vec<f32>,
    pub content_type: String,
    pub raw_dsl: String,
    pub valid: bool,
    pub score: f32,
    pub trigger: String,
    pub seed: u64,
    pub tick: u64,
}

/// Configuration for the dataset pipeline.
#[derive(Clone, Debug)]
pub struct VaeDatasetConfig {
    pub campaigns: u64,
    pub max_ticks: u64,
    pub threads: usize,
    pub base_seed: u64,
    pub campaign_config: CampaignConfig,
    /// Output paths.
    pub contexts_path: String,
    pub raw_path: String,
    pub dataset_path: String,
    /// LLM config (None = skip generation step).
    pub llm_config: Option<LlmConfig>,
    /// Workers for parallel LLM calls.
    pub llm_workers: usize,
    /// Candidates per LLM call.
    pub llm_candidates: usize,
    /// Also generate procedural items/quests.
    pub include_procedural: bool,
}

impl Default for VaeDatasetConfig {
    fn default() -> Self {
        Self {
            campaigns: 100,
            max_ticks: 30_000,
            threads: 0,
            base_seed: 2026,
            campaign_config: CampaignConfig::default(),
            contexts_path: "generated/vae_contexts.jsonl".into(),
            raw_path: "generated/vae_raw.jsonl".into(),
            dataset_path: "generated/vae_dataset.jsonl".into(),
            llm_config: None,
            llm_workers: 4,
            llm_candidates: 3,
            include_procedural: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Step 1: Campaign Sweep
// ---------------------------------------------------------------------------

/// Run campaigns and record all trigger contexts.
pub fn sweep_campaigns(config: &VaeDatasetConfig) -> Vec<TriggerContext> {
    let threads = if config.threads == 0 {
        rayon::current_num_threads().min(8)
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build thread pool");

    let completed = AtomicU64::new(0);
    let total_triggers = AtomicU64::new(0);
    let all_contexts: Mutex<Vec<TriggerContext>> = Mutex::new(Vec::new());
    let start = std::time::Instant::now();

    eprintln!("=== VAE Dataset: Campaign Sweep ===");
    eprintln!("Campaigns: {}, threads: {}, max_ticks: {}",
        config.campaigns, threads, config.max_ticks);

    pool.install(|| {
        (0..config.campaigns).into_par_iter().for_each(|i| {
            let seed = config.base_seed.wrapping_add(i.wrapping_mul(7919));
            let contexts = sweep_single_campaign(seed, config);
            let n = contexts.len();

            total_triggers.fetch_add(n as u64, Ordering::Relaxed);
            if !contexts.is_empty() {
                all_contexts.lock().unwrap().extend(contexts);
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 || done == config.campaigns {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("[{done}/{}] {:.0}/s, {} triggers collected",
                    config.campaigns, done as f64 / elapsed,
                    total_triggers.load(Ordering::Relaxed));
            }
        });
    });

    let contexts = all_contexts.into_inner().unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    // Write to JSONL
    if let Some(parent) = std::path::Path::new(&config.contexts_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = std::fs::File::create(&config.contexts_path)
        .expect("Failed to create contexts file");
    for ctx in &contexts {
        if let Ok(json) = serde_json::to_string(ctx) {
            writeln!(file, "{}", json).ok();
        }
    }

    eprintln!("\nSweep complete: {} triggers from {} campaigns in {:.1}s",
        contexts.len(), config.campaigns, elapsed);
    eprintln!("Mean {:.1} triggers/campaign", contexts.len() as f64 / config.campaigns as f64);
    eprintln!("Written to: {}", config.contexts_path);

    // Type distribution
    let mut by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for ctx in &contexts {
        *by_type.entry(ctx.content_type.clone()).or_default() += 1;
    }
    for (t, n) in &by_type {
        eprintln!("  {}: {}", t, n);
    }

    contexts
}

/// Run one campaign, returning trigger contexts.
fn sweep_single_campaign(seed: u64, config: &VaeDatasetConfig) -> Vec<TriggerContext> {
    let mut state = CampaignState::with_config(seed, config.campaign_config.clone());
    // No LLM — template fallback only
    let mut contexts = Vec::new();
    let mut seen_triggers: std::collections::HashSet<String> = std::collections::HashSet::new();

    for _ in 0..config.max_ticks {
        let action = heuristic_policy(&state);

        // Check for new progression items BEFORE stepping
        // (they get added during step by progression_triggers)
        let prev_count = state.pending_progression.len();
        let result = step_campaign(&mut state, action);

        // Record any new progression triggers
        for prog in state.pending_progression.iter().skip(prev_count) {
            let trigger_key = format!("{}_{}", prog.adventurer_id.unwrap_or(0), prog.trigger);
            if !seen_triggers.insert(trigger_key) {
                continue; // already recorded this trigger
            }

            let adv_id = prog.adventurer_id.unwrap_or(0);
            let content_type = match prog.kind {
                ProgressionKind::Ability => "ability",
                ProgressionKind::ClassOffer | ProgressionKind::HeroCandidacy => "class",
                ProgressionKind::QuestHook => "quest_hook",
                _ => continue, // skip LevelUp, ItemReward
            };

            let input = vae_features::assemble_input(
                &state, adv_id, &prog.trigger, prog.kind,
            );

            let adv = state.adventurers.iter().find(|a| a.id == adv_id);
            let archetype = adv.map(|a| a.archetype.clone()).unwrap_or_default();
            let level = adv.map(|a| a.level).unwrap_or(0);

            // Build context text for LLM prompt
            let context_text = super::content_prompts::ability_prompt(&state, adv_id, &prog.trigger);

            contexts.push(TriggerContext {
                input: input.to_vec(),
                trigger: prog.trigger.clone(),
                content_type: content_type.to_string(),
                seed,
                tick: state.tick,
                adv_id,
                archetype,
                level,
                context_text,
            });
        }

        if result.outcome.is_some() {
            break;
        }
    }

    // Optionally generate procedural items and quests
    if config.include_procedural {
        // Record quest contexts from completed quests
        for quest in &state.completed_quests {
            // Use the quest itself as a "trigger" context
            let input = vae_features::assemble_input(
                &state, 0, &format!("quest_{}", quest.quest_type as u8), ProgressionKind::QuestHook,
            );
            contexts.push(TriggerContext {
                input: input.to_vec(),
                trigger: format!("procedural_quest_{}", quest.id),
                content_type: "quest".to_string(),
                seed,
                tick: quest.completed_at_ms / 100, // approx tick
                adv_id: 0,
                archetype: String::new(),
                level: 0,
                context_text: String::new(), // no LLM needed
            });
        }
    }

    contexts
}

// ---------------------------------------------------------------------------
// Step 2: Batch LLM Generation
// ---------------------------------------------------------------------------

/// Read trigger contexts and generate content via LLM.
pub fn generate_content(config: &VaeDatasetConfig, contexts: &[TriggerContext]) {
    let llm_cfg = match &config.llm_config {
        Some(c) => c,
        None => {
            eprintln!("LLM not configured, skipping generation step");
            return;
        }
    };

    if !llm::check_ollama(llm_cfg) {
        eprintln!("Ollama not reachable, skipping generation");
        return;
    }

    let store = std::sync::Arc::new(llm::create_store(llm_cfg));

    // Filter to only ability and class contexts (quests are procedural)
    let llm_contexts: Vec<&TriggerContext> = contexts.iter()
        .filter(|c| c.content_type == "ability" || c.content_type == "class")
        .collect();

    eprintln!("\n=== VAE Dataset: LLM Generation ===");
    eprintln!("{} contexts to generate ({} workers, {} candidates each)",
        llm_contexts.len(), config.llm_workers, config.llm_candidates);

    let completed = AtomicU64::new(0);
    let valid = AtomicU64::new(0);
    let start = std::time::Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.llm_workers)
        .build()
        .expect("Failed to build LLM thread pool");

    let mut llm_cfg_with_candidates = llm_cfg.clone();
    llm_cfg_with_candidates.candidates = config.llm_candidates;

    pool.install(|| {
        llm_contexts.par_iter().for_each(|ctx| {
            let result = match ctx.content_type.as_str() {
                "ability" => llm::generate_ability(&llm_cfg_with_candidates, &store, &ctx.context_text),
                "class" => llm::generate_class(&llm_cfg_with_candidates, &store, &ctx.context_text),
                _ => None,
            };

            if result.is_some() {
                valid.fetch_add(1, Ordering::Relaxed);
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("[{done}/{}] {:.2}/s, {} valid",
                    llm_contexts.len(), done as f64 / elapsed,
                    valid.load(Ordering::Relaxed));
            }
        });
    });

    let elapsed = start.elapsed().as_secs_f64();
    let (total, hits, valid_count) = store.stats();
    eprintln!("\nGeneration complete in {:.1}s", elapsed);
    eprintln!("Total calls: {}, cache hits: {}, valid: {}", total, hits, valid_count);
}

// ---------------------------------------------------------------------------
// Step 3: Parse + Extract Slots
// ---------------------------------------------------------------------------

/// Parse LLM outputs and extract slot vectors, writing the final dataset.
pub fn extract_dataset(config: &VaeDatasetConfig, contexts: &[TriggerContext]) {
    use tactical_sim::effects::dsl;

    eprintln!("\n=== VAE Dataset: Extract Slots ===");

    // Load the content store to get LLM outputs
    let store_path = config.llm_config.as_ref()
        .and_then(|c| c.store_path.as_deref())
        .unwrap_or("generated/llm_content_store.jsonl");

    let store_records: Vec<llm::ContentRecord> = if let Ok(contents) = std::fs::read_to_string(store_path) {
        contents.lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    } else {
        vec![]
    };

    // Build lookup: context_text hash → best content
    let mut content_lookup: std::collections::HashMap<u64, (String, f32)> = std::collections::HashMap::new();
    for record in &store_records {
        if let Some(ref selected) = record.selected {
            content_lookup.insert(record.prompt_hash, (selected.clone(), record.score));
        }
    }

    if let Some(parent) = std::path::Path::new(&config.dataset_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = std::fs::File::create(&config.dataset_path)
        .expect("Failed to create dataset file");

    let mut total = 0u64;
    let mut valid_count = 0u64;
    let mut by_type: std::collections::HashMap<String, (u64, u64)> = std::collections::HashMap::new();

    for ctx in contexts {
        match ctx.content_type.as_str() {
            "ability" | "class" => {
                // Look up LLM-generated content by prompt hash
                let prompt_hash = hash_prompt(&ctx.context_text);
                let (content, score) = match content_lookup.get(&prompt_hash) {
                    Some((c, s)) => (c.clone(), *s),
                    None => continue, // No LLM output for this context
                };

                // Parse and extract slots
                let (slots, valid) = if ctx.content_type == "ability" {
                    parse_ability_slots(&content)
                } else {
                    parse_class_slots(&content)
                };

                let entry = by_type.entry(ctx.content_type.clone()).or_default();
                entry.0 += 1;
                if valid { entry.1 += 1; valid_count += 1; }

                let record = DatasetRecord {
                    input: ctx.input.clone(),
                    slots,
                    content_type: ctx.content_type.clone(),
                    raw_dsl: content,
                    valid,
                    score,
                    trigger: ctx.trigger.clone(),
                    seed: ctx.seed,
                    tick: ctx.tick,
                };

                if let Ok(json) = serde_json::to_string(&record) {
                    writeln!(file, "{}", json).ok();
                }
                total += 1;
            }
            "quest" => {
                // Procedural quest — slot extract directly from the context
                // We need the original QuestRequest, but we only have the context vector.
                // For now, skip procedural quests in the slot extraction.
                // They can be added by running sweep with quest data attached.
            }
            _ => {}
        }
    }

    eprintln!("Dataset: {} records ({} valid)", total, valid_count);
    for (t, (n, v)) in &by_type {
        eprintln!("  {}: {}/{} valid ({:.0}%)", t, v, n,
            if *n > 0 { *v as f64 / *n as f64 * 100.0 } else { 0.0 });
    }
    eprintln!("Written to: {}", config.dataset_path);
}

fn parse_ability_slots(dsl_text: &str) -> (Vec<f32>, bool) {
    use tactical_sim::effects::dsl;

    // Try parsing as ability DSL
    match dsl::parse_abilities(dsl_text) {
        Ok((abilities, passives)) => {
            if let Some(def) = abilities.first() {
                (vae_slots::ability_to_slots(def), true)
            } else if let Some(def) = passives.first() {
                (vae_slots::passive_to_slots(def), true)
            } else {
                (vec![0.0; vae_slots::ABILITY_SLOT_DIM], false)
            }
        }
        Err(_) => (vec![0.0; vae_slots::ABILITY_SLOT_DIM], false),
    }
}

fn parse_class_slots(dsl_text: &str) -> (Vec<f32>, bool) {
    use super::class_dsl;

    match class_dsl::parse_class(dsl_text) {
        Ok(def) => (vae_slots::class_to_slots(&def), true),
        Err(_) => (vec![0.0; vae_slots::CLASS_SLOT_DIM], false),
    }
}

/// FNV-1a hash (same as in llm.rs).
fn hash_prompt(prompt: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in prompt.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

/// Run the complete dataset pipeline: sweep → generate → extract.
pub fn run_full_pipeline(config: &VaeDatasetConfig) {
    let contexts = sweep_campaigns(config);
    generate_content(config, &contexts);
    extract_dataset(config, &contexts);
}
