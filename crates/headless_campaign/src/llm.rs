//! LLM content generation via Ollama HTTP API.
//!
//! Blocking client that generates abilities and classes by calling a local
//! Ollama server. Falls back to template generators when the server is
//! unavailable or returns invalid output.
//!
//! All requests and responses are logged to a `ContentStore` for:
//! - Deduplication across BFS branches (same prompt → cache hit)
//! - Audit/replay of generated content
//! - Persistence across runs via JSONL

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Content store
// ---------------------------------------------------------------------------

/// A single LLM request+response record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentRecord {
    /// Hash of the prompt (for dedup).
    pub prompt_hash: u64,
    /// Generation type.
    pub gen_type: String,
    /// The context passed to the generator.
    pub context: String,
    /// The full prompt sent to the LLM.
    pub prompt: String,
    /// All raw responses from the LLM (one per candidate).
    pub raw_responses: Vec<String>,
    /// The extracted+scored candidates (block text, score).
    pub candidates: Vec<(String, f32)>,
    /// The selected best content, or None if all candidates failed.
    pub selected: Option<String>,
    /// Quality score of the selected content.
    pub score: f32,
    /// Whether this was a cache hit.
    pub cache_hit: bool,
    /// Wall-clock time for this generation (seconds).
    pub time_s: f64,
    /// Model used.
    pub model: String,
}

/// Thread-safe content store for all LLM interactions.
pub struct ContentStore {
    /// Prompt hash → best content (for dedup).
    cache: Mutex<HashMap<u64, String>>,
    /// All records, appended in order.
    records: Mutex<Vec<ContentRecord>>,
    /// Optional JSONL file for persistent logging.
    log_file: Option<Mutex<std::fs::File>>,
}

impl std::fmt::Debug for ContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (total, hits, valid) = self.stats();
        write!(f, "ContentStore({} records, {} hits, {} valid)", total, hits, valid)
    }
}

impl ContentStore {
    /// Create a new store, optionally logging to a JSONL file.
    pub fn new(log_path: Option<&str>) -> Self {
        let log_file = log_path.map(|p| {
            if let Some(parent) = std::path::Path::new(p).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .expect("Failed to open LLM log file"),
            )
        });
        ContentStore {
            cache: Mutex::new(HashMap::new()),
            records: Mutex::new(Vec::new()),
            log_file,
        }
    }

    /// Load cache from an existing JSONL file (warm start).
    pub fn load_cache(&self, path: &str) {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let mut cache = self.cache.lock().unwrap();
            for line in contents.lines() {
                if let Ok(record) = serde_json::from_str::<ContentRecord>(line) {
                    if let Some(ref content) = record.selected {
                        cache.insert(record.prompt_hash, content.clone());
                    }
                }
            }
            eprintln!("LLM store: loaded {} cached entries from {}", cache.len(), path);
        }
    }

    /// Look up a cached result by prompt hash.
    fn get_cached(&self, hash: u64) -> Option<String> {
        self.cache.lock().unwrap().get(&hash).cloned()
    }

    /// Store a record and update the cache.
    fn store(&self, record: ContentRecord) {
        // Live logging — LLM is slow, user needs to see progress
        if record.cache_hit {
            eprintln!(
                "  LLM [cache] {} — {}",
                record.gen_type,
                record.context.chars().take(60).collect::<String>(),
            );
        } else if let Some(ref content) = record.selected {
            // Extract the name from the generated content for display
            let name = content.split_whitespace().nth(1).unwrap_or("?");
            eprintln!(
                "  LLM [{}] {} \"{}\" score={:.0} ({}/{} valid, {:.1}s)",
                record.gen_type, name, record.context.chars().take(40).collect::<String>(),
                record.score, record.candidates.len(),
                record.raw_responses.len(), record.time_s,
            );
        } else {
            eprintln!(
                "  LLM [FAIL] {} — {} (0/{} valid, {:.1}s)",
                record.gen_type,
                record.context.chars().take(50).collect::<String>(),
                record.raw_responses.len(), record.time_s,
            );
        }

        if let Some(ref content) = record.selected {
            self.cache
                .lock()
                .unwrap()
                .insert(record.prompt_hash, content.clone());
        }

        // Append to log file
        if let Some(ref log) = self.log_file {
            if let Ok(json) = serde_json::to_string(&record) {
                if let Ok(mut f) = log.lock() {
                    writeln!(f, "{}", json).ok();
                    f.flush().ok();
                }
            }
        }

        self.records.lock().unwrap().push(record);
    }

    /// Get summary statistics.
    pub fn stats(&self) -> (usize, usize, usize) {
        let records = self.records.lock().unwrap();
        let total = records.len();
        let hits = records.iter().filter(|r| r.cache_hit).count();
        let valid = records.iter().filter(|r| r.selected.is_some()).count();
        (total, hits, valid)
    }
}

// ---------------------------------------------------------------------------
// Config & globals
// ---------------------------------------------------------------------------

/// Global flag to disable LLM calls (e.g., after connection failure).
static LLM_DISABLED: AtomicBool = AtomicBool::new(false);
static LLM_CALLS: AtomicU64 = AtomicU64::new(0);
static LLM_HITS: AtomicU64 = AtomicU64::new(0);

/// Configuration for the LLM client.
#[derive(Clone, Debug)]
pub struct LlmConfig {
    /// Ollama server URL.
    pub base_url: String,
    /// Model name.
    pub model: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Number of candidates to generate per request (best-of-N).
    pub candidates: usize,
    /// Max tokens for ability generation.
    pub ability_max_tokens: u32,
    /// Max tokens for class generation.
    pub class_max_tokens: u32,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Path to content store JSONL log.
    pub store_path: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            model: std::env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "qwen35-9b".into()),
            temperature: 0.7,
            candidates: 3,
            ability_max_tokens: 300,
            class_max_tokens: 500,
            timeout_secs: 90,
            store_path: Some("generated/llm_content_store.jsonl".into()),
        }
    }
}

fn make_agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(timeout_secs)))
        .build()
        .new_agent()
}

/// Check if Ollama is reachable. Disables LLM globally if not.
pub fn check_ollama(config: &LlmConfig) -> bool {
    if LLM_DISABLED.load(Ordering::Relaxed) {
        return false;
    }
    let url = format!("{}/api/tags", config.base_url);
    let agent = make_agent(3);
    match agent.get(&url).call() {
        Ok(_) => true,
        Err(_) => {
            LLM_DISABLED.store(true, Ordering::Relaxed);
            eprintln!("LLM: Ollama not reachable at {}, disabling", config.base_url);
            false
        }
    }
}

/// Reset the disabled flag (e.g., after restarting Ollama).
pub fn enable_llm() {
    LLM_DISABLED.store(false, Ordering::Relaxed);
}

/// Get LLM call statistics.
pub fn llm_stats() -> (u64, u64) {
    (
        LLM_CALLS.load(Ordering::Relaxed),
        LLM_HITS.load(Ordering::Relaxed),
    )
}

/// Create a content store from config, loading existing cache if available.
pub fn create_store(config: &LlmConfig) -> ContentStore {
    let store = ContentStore::new(config.store_path.as_deref());
    if let Some(ref path) = config.store_path {
        store.load_cache(path);
    }
    store
}

// ---------------------------------------------------------------------------
// Prompt templates (must match scripts/generate_content.py)
// ---------------------------------------------------------------------------

const ABILITY_SPEC: &str = r#"Output ONLY one ability block in this exact format, nothing else.

ability shield_bash {
    type: active
    cooldown: 10s
    effect: stun 2s + damage 30
    tag: melee
    description: "A devastating shield strike"
}

ability battle_sense {
    type: passive
    trigger: on_damage_taken
    effect: buff defense 15% 5s
    tag: defense
    description: "Pain sharpens focus"
}

Valid tags: ranged, nature, stealth, tracking, survival, melee, defense, leadership, fortification, honor, arcane, elemental, ritual, knowledge, enchantment, healing, divine, protection, purification, restoration, assassination, agility, deception, sabotage
Valid stats: attack, defense, speed, max_hp, ability_power
Valid triggers (passive only): on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged, on_death, on_ability_used, on_hp_below, on_hp_above, on_shield_broken, periodic
Valid effects: damage <N>, heal <N>, shield <N>, stun <N>s, slow <factor> <N>s, knockback <N>, dash, buff <stat> <N>% <N>s, debuff <stat> <N>% <N>s, stealth <N>s, evasion <N>%, tenacity <N>, teleport, aura <stat> +<N>

Generate ONE ability for: "#;

const CLASS_SPEC: &str = r#"Output ONLY one class definition, nothing else.

class Ranger {
    stat_growth: +2 attack, +1 defense, +3 speed, +1 ability_power per level
    tags: ranged, nature, stealth
    scaling party_alive_count {
        when party_members >= 2: +10% speed
        always: aura morale +1
    }
    abilities {
        level 1: keen_eye "Improved scouting range"
        level 5: multishot "Hit multiple targets"
        level 10: camouflage "Escape losing battles"
        level 20: deadeye "Double critical hit chance"
    }
    requirements: level 1
}

Rules:
- stat_growth uses ONLY: attack, defense, speed, max_hp, ability_power (total per level <= 10)
- tags from: ranged, nature, stealth, melee, defense, leadership, arcane, elemental, healing, divine, assassination, agility, deception, sabotage
- scaling sources: party_alive_count, faction_allied_count, crisis_active, threat_level, fame
- requirements: level <N> or quest_count <N> or fame <N>
- ability names must be snake_case, 3-5 abilities spread across levels 1-40

Generate ONE class for: "#;

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

fn hash_prompt(prompt: &str) -> u64 {
    // FNV-1a hash
    let mut h: u64 = 0xcbf29ce484222325;
    for b in prompt.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Extraction & scoring
// ---------------------------------------------------------------------------

fn extract_ability(text: &str) -> Option<String> {
    let start = text.find("ability ")?;
    let brace_start = text[start..].find('{')?;
    let brace_end = text[start + brace_start..].find('}')?;
    Some(text[start..start + brace_start + brace_end + 1].to_string())
}

fn extract_class(text: &str) -> Option<String> {
    let start = text.find("class ")?;
    let brace_start = text[start..].find('{')?;
    let mut depth = 0;
    for (i, c) in text[start + brace_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + brace_start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_think(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            break;
        }
    }
    result.trim().to_string()
}

fn valid_tags() -> HashSet<&'static str> {
    [
        "ranged", "nature", "stealth", "tracking", "survival", "melee",
        "defense", "leadership", "fortification", "honor", "arcane",
        "elemental", "ritual", "knowledge", "enchantment", "healing",
        "divine", "protection", "purification", "restoration",
        "assassination", "agility", "deception", "sabotage", "crisis",
        "legendary", "inspiration", "sacrifice",
    ]
    .into_iter()
    .collect()
}

fn score_ability(text: &str) -> f32 {
    let tags = valid_tags();
    let mut score = 0.0f32;

    if text.contains("type: active") || text.contains("type: passive") {
        score += 1.0;
    }
    if text.contains("type: active") && text.contains("cooldown:") {
        score += 1.0;
    }
    if text.contains("type: passive") && text.contains("trigger:") {
        score += 1.0;
    }
    if let Some(tag_pos) = text.find("tag:") {
        let tag_line = &text[tag_pos + 4..];
        let tag_end = tag_line.find('\n').unwrap_or(tag_line.len());
        let tag = tag_line[..tag_end].trim().trim_end_matches(',');
        if tags.contains(tag) {
            score += 3.0;
        }
    }
    if text.contains("effect:") {
        score += 1.0;
    }
    if text.contains("description: \"") {
        score += 1.0;
    }
    if let Some(name_start) = text.find("ability ") {
        let rest = &text[name_start + 8..];
        if let Some(space) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
            let name = &rest[..space];
            if name == name.to_lowercase() && name.contains('_') {
                score += 2.0;
            }
        }
    }
    score
}

fn score_class(text: &str) -> f32 {
    let tags = valid_tags();
    let mut score = 0.0f32;

    if text.contains("stat_growth:") {
        score += 1.0;
        for stat in ["attack", "defense", "speed", "max_hp", "ability_power"] {
            if text.contains(stat) {
                score += 0.3;
            }
        }
    }
    if let Some(tags_pos) = text.find("tags:") {
        let tag_line = &text[tags_pos + 5..];
        let tag_end = tag_line.find('\n').unwrap_or(tag_line.len());
        let tag_str = &tag_line[..tag_end];
        let found_tags: Vec<&str> = tag_str.split(',').map(|t| t.trim()).collect();
        let valid = found_tags.iter().filter(|t| tags.contains(**t)).count();
        score += valid as f32 * 0.5;
    }
    if text.contains("scaling ") {
        score += 1.0;
    }
    let level_count = text.matches("level ").count().saturating_sub(1);
    if (3..=6).contains(&level_count) {
        score += 2.0;
    } else if level_count > 0 {
        score += 1.0;
    }
    if text.contains("requirements:") {
        score += 1.0;
    }
    if let Some(name_start) = text.find("class ") {
        let rest = &text[name_start + 6..];
        if let Some(space) = rest.find(|c: char| !c.is_alphanumeric()) {
            let name = &rest[..space];
            if !name.is_empty()
                && name.chars().next().unwrap().is_uppercase()
                && !name.contains('_')
            {
                score += 2.0;
            }
        }
    }
    score
}

// ---------------------------------------------------------------------------
// HTTP calls
// ---------------------------------------------------------------------------

fn call_ollama(
    config: &LlmConfig,
    messages: &[serde_json::Value],
    max_tokens: u32,
) -> Option<String> {
    let body = serde_json::json!({
        "model": config.model,
        "messages": messages,
        "stream": false,
        "options": {
            "temperature": config.temperature,
            "num_predict": max_tokens,
        }
    });

    let url = format!("{}/api/chat", config.base_url);
    let agent = make_agent(config.timeout_secs);
    let mut resp = agent.post(&url).send_json(&body).ok()?;

    let json: serde_json::Value = resp.body_mut().read_json().ok()?;
    let content = json["message"]["content"].as_str()?;
    Some(strip_think(content))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate an ability via LLM with content store. Returns the DSL text or None.
pub fn generate_ability(
    config: &LlmConfig,
    store: &ContentStore,
    context: &str,
) -> Option<String> {
    if LLM_DISABLED.load(Ordering::Relaxed) {
        return None;
    }

    let prompt = format!("{}{}", ABILITY_SPEC, context);
    let prompt_hash = hash_prompt(&prompt);

    // Check cache first
    if let Some(cached) = store.get_cached(prompt_hash) {
        store.store(ContentRecord {
            prompt_hash,
            gen_type: "ability".into(),
            context: context.to_string(),
            prompt: prompt.clone(),
            raw_responses: vec![],
            candidates: vec![(cached.clone(), 0.0)],
            selected: Some(cached.clone()),
            score: 0.0,
            cache_hit: true,
            time_s: 0.0,
            model: config.model.clone(),
        });
        LLM_CALLS.fetch_add(1, Ordering::Relaxed);
        LLM_HITS.fetch_add(1, Ordering::Relaxed);
        return Some(cached);
    }

    LLM_CALLS.fetch_add(1, Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];

    let mut raw_responses = Vec::new();
    let mut scored_candidates = Vec::new();

    for _ in 0..config.candidates {
        if let Some(raw) = call_ollama(config, &messages, config.ability_max_tokens) {
            if let Some(block) = extract_ability(&raw) {
                let s = score_ability(&block);
                scored_candidates.push((block, s));
            }
            raw_responses.push(raw);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected = scored_candidates.first().map(|(b, _)| b.clone());
    let best_score = scored_candidates.first().map(|(_, s)| *s).unwrap_or(0.0);

    if selected.is_some() {
        LLM_HITS.fetch_add(1, Ordering::Relaxed);
    }

    store.store(ContentRecord {
        prompt_hash,
        gen_type: "ability".into(),
        context: context.to_string(),
        prompt: format!("{}...", &ABILITY_SPEC[..50]),
        raw_responses,
        candidates: scored_candidates,
        selected: selected.clone(),
        score: best_score,
        cache_hit: false,
        time_s: elapsed,
        model: config.model.clone(),
    });

    selected
}

/// Generate a class via LLM with content store. Returns the DSL text or None.
pub fn generate_class(
    config: &LlmConfig,
    store: &ContentStore,
    context: &str,
) -> Option<String> {
    if LLM_DISABLED.load(Ordering::Relaxed) {
        return None;
    }

    let prompt = format!("{}{}", CLASS_SPEC, context);
    let prompt_hash = hash_prompt(&prompt);

    // Check cache first
    if let Some(cached) = store.get_cached(prompt_hash) {
        store.store(ContentRecord {
            prompt_hash,
            gen_type: "class".into(),
            context: context.to_string(),
            prompt: prompt.clone(),
            raw_responses: vec![],
            candidates: vec![(cached.clone(), 0.0)],
            selected: Some(cached.clone()),
            score: 0.0,
            cache_hit: true,
            time_s: 0.0,
            model: config.model.clone(),
        });
        LLM_CALLS.fetch_add(1, Ordering::Relaxed);
        LLM_HITS.fetch_add(1, Ordering::Relaxed);
        return Some(cached);
    }

    LLM_CALLS.fetch_add(1, Ordering::Relaxed);
    let t0 = std::time::Instant::now();

    // Multi-turn to avoid prompt completion
    let words: Vec<String> = context
        .split_whitespace()
        .filter(|w| w.chars().all(|c| c.is_alphabetic()))
        .filter(|w| {
            !["a", "an", "the", "who", "that", "and", "or", "with", "in", "on", "for"]
                .contains(w)
        })
        .take(3)
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(first) => first.to_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect();
    let seed_name = if words.is_empty() {
        "Custom".into()
    } else {
        words.join("")
    };

    let messages = vec![
        serde_json::json!({"role": "user", "content": format!("Generate a class definition for {}. Use the exact DSL format I'll show you.", context)}),
        serde_json::json!({"role": "assistant", "content": format!("class {} {{", seed_name)}),
        serde_json::json!({"role": "user", "content": prompt}),
    ];

    let mut raw_responses = Vec::new();
    let mut scored_candidates = Vec::new();

    for _ in 0..config.candidates {
        if let Some(raw) = call_ollama(config, &messages, config.class_max_tokens) {
            if let Some(block) = extract_class(&raw) {
                let s = score_class(&block);
                scored_candidates.push((block, s));
            }
            raw_responses.push(raw);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected = scored_candidates.first().map(|(b, _)| b.clone());
    let best_score = scored_candidates.first().map(|(_, s)| *s).unwrap_or(0.0);

    if selected.is_some() {
        LLM_HITS.fetch_add(1, Ordering::Relaxed);
    }

    store.store(ContentRecord {
        prompt_hash,
        gen_type: "class".into(),
        context: context.to_string(),
        prompt: format!("{}...", &CLASS_SPEC[..50]),
        raw_responses,
        candidates: scored_candidates,
        selected: selected.clone(),
        score: best_score,
        cache_hit: false,
        time_s: elapsed,
        model: config.model.clone(),
    });

    selected
}
