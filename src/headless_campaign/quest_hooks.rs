//! Wildermyth-style quest hook system.
//!
//! Quest hooks are pre-authored story beats defined in TOML that fire when
//! trigger conditions are met against game state. The system checks all
//! loaded hooks every N ticks and generates quests/choices when conditions
//! match.
//!
//! Hook triggers can reference:
//! - Adventurer traits, stats, levels, status
//! - Faction relations and stances
//! - Campaign progress and global threat
//! - Locations (scouted, threat level)
//! - Quest completion counts
//! - Guild resources (gold, reputation, adventurer count)
//! - Adventurer pair relationships (shared quest count)

use std::collections::HashMap;
use serde::Deserialize;

use super::choice_templates::{instantiate_template, ChoiceTemplate, TemplateContext};
use super::state::*;
use super::actions::WorldEvent;

// ---------------------------------------------------------------------------
// Hook definition (deserialized from TOML)
// ---------------------------------------------------------------------------

/// A quest hook loaded from a TOML file.
#[derive(Clone, Debug, Deserialize)]
pub struct QuestHook {
    /// Unique identifier (filename without extension).
    #[serde(default)]
    pub id: String,

    /// Human-readable name for debugging.
    pub name: String,

    /// What triggers this hook.
    pub trigger: HookTrigger,

    /// Conditions that must ALL be true for the hook to fire.
    #[serde(default)]
    pub conditions: Vec<HookCondition>,

    /// The choice template to instantiate when the hook fires.
    /// Inline — same format as choice_templates/*.toml.
    pub choice: ChoiceTemplate,

    /// Can this hook fire more than once per campaign?
    #[serde(default)]
    pub repeatable: bool,

    /// Minimum ticks between firings (if repeatable).
    #[serde(default = "default_cooldown")]
    pub cooldown_ticks: u64,

    /// Priority — higher priority hooks are checked first.
    #[serde(default)]
    pub priority: i32,
}

fn default_cooldown() -> u64 {
    3000 // ~5 minutes
}

/// What game event or state triggers this hook.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum HookTrigger {
    /// An adventurer has a specific trait.
    AdventurerTrait { trait_name: String },

    /// An adventurer reaches a level threshold.
    AdventurerLevel { min_level: u32 },

    /// Faction relation crosses a threshold.
    FactionRelation {
        #[serde(default)]
        faction_id: Option<usize>,
        /// "above" or "below"
        direction: String,
        threshold: f32,
    },

    /// A location has been scouted.
    LocationScouted {
        #[serde(default)]
        location_type: Option<String>,
    },

    /// Campaign progress crosses a threshold.
    CampaignProgress { min_progress: f32 },

    /// Guild resource threshold.
    GuildResource {
        resource: String,
        direction: String,
        threshold: f32,
    },

    /// Total quests completed reaches a count.
    QuestsCompleted { min_count: usize },

    /// Global threat level crosses a threshold.
    ThreatLevel { min_threat: f32 },

    /// Two adventurers have been on N+ quests together.
    AdventurerBond { min_shared_quests: usize },

    /// Periodic — fires on a timer regardless of state.
    Periodic { interval_ticks: u64 },
}

/// Additional conditions that must be true alongside the trigger.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum HookCondition {
    /// Adventurer stat must be above/below threshold.
    AdventurerStat {
        stat: String,
        direction: String,
        threshold: f32,
    },
    /// Guild must have at least this much gold.
    MinGold { amount: f32 },
    /// Guild must have at least N adventurers alive.
    MinAdventurers { count: usize },
    /// No pending choice from this hook.
    NoPendingChoice,
}

// ---------------------------------------------------------------------------
// Hook registry
// ---------------------------------------------------------------------------

/// All loaded quest hooks.
#[derive(Clone, Debug, Default)]
pub struct QuestHookRegistry {
    pub hooks: Vec<QuestHook>,
}

impl QuestHookRegistry {
    /// Load all hooks from a directory of TOML files.
    pub fn load_from_dir(dir: &std::path::Path) -> Self {
        let mut hooks = Vec::new();

        if !dir.exists() {
            return Self { hooks };
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Self { hooks },
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str::<QuestHook>(&content) {
                    Ok(mut hook) => {
                        hook.id = id;
                        hooks.push(hook);
                    }
                    Err(e) => eprintln!("Warning: failed to parse hook {}: {}", path.display(), e),
                },
                Err(e) => eprintln!("Warning: failed to read {}: {}", path.display(), e),
            }
        }

        // Sort by priority (higher first)
        hooks.sort_by(|a, b| b.priority.cmp(&a.priority));

        Self { hooks }
    }
}

// ---------------------------------------------------------------------------
// Hook state tracking (per-campaign)
// ---------------------------------------------------------------------------

/// Tracks which hooks have fired and when, stored in CampaignState.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HookState {
    /// Hook ID → last tick it fired.
    pub last_fired: HashMap<String, u64>,
    /// Hook ID → total times fired.
    pub fire_count: HashMap<String, u32>,
}

// ---------------------------------------------------------------------------
// Hook evaluation
// ---------------------------------------------------------------------------

/// Lazily loaded hook registry.
static HOOKS: std::sync::OnceLock<QuestHookRegistry> = std::sync::OnceLock::new();

pub fn get_or_load_hooks() -> &'static QuestHookRegistry {
    HOOKS.get_or_init(|| {
        let dir = std::path::Path::new("dataset/campaign/quest_hooks");
        let registry = QuestHookRegistry::load_from_dir(dir);
        if !registry.hooks.is_empty() {
            eprintln!("Quest hooks: loaded {} from {}", registry.hooks.len(), dir.display());
        }
        registry
    })
}

/// Check all hooks against current game state and fire any that match.
/// Returns the generated choice events.
pub fn evaluate_hooks(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    let registry = get_or_load_hooks();
    if registry.hooks.is_empty() {
        return;
    }

    // Max 1 hook fires per evaluation (prevent floods)
    // Debug: log first evaluation
    for hook in &registry.hooks {
        // Check if already fired and not repeatable
        if !hook.repeatable && state.hook_state.fire_count.get(&hook.id).copied().unwrap_or(0) > 0 {
            continue;
        }

        // Check cooldown
        if let Some(&last) = state.hook_state.last_fired.get(&hook.id) {
            if state.tick < last + hook.cooldown_ticks {
                continue;
            }
        }

        // Check trigger
        let trigger_ctx = match check_trigger(&hook.trigger, state) {
            Some(ctx) => ctx,
            None => continue,
        };

        // Check additional conditions
        if !check_conditions(&hook.conditions, state) {
            continue;
        }

        // Fire the hook!
        let choice_id = state.next_event_id;
        state.next_event_id += 1;

        let choice = instantiate_template(
            &hook.choice,
            &trigger_ctx,
            choice_id,
            state.elapsed_ms,
        );

        events.push(WorldEvent::ChoicePresented {
            choice_id,
            prompt: choice.prompt.clone(),
            num_options: choice.options.len(),
        });

        state.pending_choices.push(choice);

        // Record firing
        state.hook_state.last_fired.insert(hook.id.clone(), state.tick);
        *state.hook_state.fire_count.entry(hook.id.clone()).or_default() += 1;

        // Max 2 hooks per evaluation to prevent floods
        if state.pending_choices.len() >= 5 {
            break;
        }
    }
}

/// Check if a trigger condition is met. Returns context variables if so.
fn check_trigger(trigger: &HookTrigger, state: &CampaignState) -> Option<TemplateContext> {
    match trigger {
        HookTrigger::AdventurerTrait { trait_name } => {
            for adv in &state.adventurers {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                if adv.traits.iter().any(|t| t == trait_name) {
                    let mut ctx = TemplateContext::new();
                    ctx.insert("adventurer_name".into(), adv.name.clone());
                    ctx.insert("adventurer_id".into(), adv.id.to_string());
                    ctx.insert("trait_name".into(), trait_name.clone());
                    return Some(ctx);
                }
            }
            None
        }

        HookTrigger::AdventurerLevel { min_level } => {
            for adv in &state.adventurers {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                if adv.level >= *min_level {
                    let mut ctx = TemplateContext::new();
                    ctx.insert("adventurer_name".into(), adv.name.clone());
                    ctx.insert("adventurer_id".into(), adv.id.to_string());
                    ctx.insert("level".into(), adv.level.to_string());
                    return Some(ctx);
                }
            }
            None
        }

        HookTrigger::FactionRelation {
            faction_id,
            direction,
            threshold,
        } => {
            for faction in &state.factions {
                if let Some(fid) = faction_id {
                    if faction.id != *fid {
                        continue;
                    }
                }
                let rel = faction.relationship_to_guild;
                let matches = match direction.as_str() {
                    "above" => rel > *threshold,
                    "below" => rel < *threshold,
                    _ => false,
                };
                if matches {
                    let mut ctx = TemplateContext::new();
                    ctx.insert("faction_name".into(), faction.name.clone());
                    ctx.insert("faction_id".into(), faction.id.to_string());
                    ctx.insert("relation".into(), format!("{:.0}", rel));
                    return Some(ctx);
                }
            }
            None
        }

        HookTrigger::LocationScouted { location_type } => {
            for loc in &state.overworld.locations {
                if !loc.scouted {
                    continue;
                }
                if let Some(ref lt) = location_type {
                    let loc_type_str = format!("{:?}", loc.location_type);
                    if !loc_type_str.eq_ignore_ascii_case(lt) {
                        continue;
                    }
                }
                let mut ctx = TemplateContext::new();
                ctx.insert("location_name".into(), loc.name.clone());
                ctx.insert("location_id".into(), loc.id.to_string());
                ctx.insert("threat".into(), format!("{:.0}", loc.threat_level));
                return Some(ctx);
            }
            None
        }

        HookTrigger::CampaignProgress { min_progress } => {
            if state.overworld.campaign_progress >= *min_progress {
                let mut ctx = TemplateContext::new();
                ctx.insert(
                    "progress".into(),
                    format!("{:.0}", state.overworld.campaign_progress * 100.0),
                );
                Some(ctx)
            } else {
                None
            }
        }

        HookTrigger::GuildResource {
            resource,
            direction,
            threshold,
        } => {
            let value = match resource.as_str() {
                "gold" => state.guild.gold,
                "supplies" => state.guild.supplies,
                "reputation" => state.guild.reputation,
                "adventurers" => state
                    .adventurers
                    .iter()
                    .filter(|a| a.status != AdventurerStatus::Dead)
                    .count() as f32,
                _ => return None,
            };
            let matches = match direction.as_str() {
                "above" => value > *threshold,
                "below" => value < *threshold,
                _ => false,
            };
            if matches {
                let mut ctx = TemplateContext::new();
                ctx.insert("resource".into(), resource.clone());
                ctx.insert("value".into(), format!("{:.0}", value));
                Some(ctx)
            } else {
                None
            }
        }

        HookTrigger::QuestsCompleted { min_count } => {
            let victories = state
                .completed_quests
                .iter()
                .filter(|q| q.result == QuestResult::Victory)
                .count();
            if victories >= *min_count {
                let mut ctx = TemplateContext::new();
                ctx.insert("quests_completed".into(), victories.to_string());
                Some(ctx)
            } else {
                None
            }
        }

        HookTrigger::ThreatLevel { min_threat } => {
            if state.overworld.global_threat_level >= *min_threat {
                let mut ctx = TemplateContext::new();
                ctx.insert(
                    "threat_level".into(),
                    format!("{:.0}", state.overworld.global_threat_level),
                );
                Some(ctx)
            } else {
                None
            }
        }

        HookTrigger::AdventurerBond { min_shared_quests } => {
            // Count shared quests between adventurer pairs
            let alive: Vec<&Adventurer> = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .collect();

            for i in 0..alive.len() {
                for j in (i + 1)..alive.len() {
                    let shared = state
                        .completed_quests
                        .iter()
                        .filter(|q| {
                            // Check if both adventurers were in this quest's party
                            // (approximation: check if party_id matches)
                            q.result == QuestResult::Victory
                        })
                        .count();
                    // Rough heuristic: shared quests ≈ total / adventurer_count
                    let approx_shared = shared / alive.len().max(1);
                    if approx_shared >= *min_shared_quests {
                        let mut ctx = TemplateContext::new();
                        ctx.insert("adventurer1_name".into(), alive[i].name.clone());
                        ctx.insert("adventurer2_name".into(), alive[j].name.clone());
                        ctx.insert("adventurer1_id".into(), alive[i].id.to_string());
                        ctx.insert("adventurer2_id".into(), alive[j].id.to_string());
                        ctx.insert("shared_quests".into(), approx_shared.to_string());
                        return Some(ctx);
                    }
                }
            }
            None
        }

        HookTrigger::Periodic { interval_ticks } => {
            if state.tick % interval_ticks == 0 && state.tick > 0 {
                Some(TemplateContext::new())
            } else {
                None
            }
        }
    }
}

/// Check if all additional conditions are met.
fn check_conditions(conditions: &[HookCondition], state: &CampaignState) -> bool {
    conditions.iter().all(|cond| match cond {
        HookCondition::AdventurerStat {
            stat,
            direction,
            threshold,
        } => state.adventurers.iter().any(|a| {
            if a.status == AdventurerStatus::Dead {
                return false;
            }
            let value = match stat.as_str() {
                "loyalty" => a.loyalty,
                "stress" => a.stress,
                "fatigue" => a.fatigue,
                "injury" => a.injury,
                "resolve" => a.resolve,
                "morale" => a.morale,
                "level" => a.level as f32,
                _ => return false,
            };
            match direction.as_str() {
                "above" => value > *threshold,
                "below" => value < *threshold,
                _ => false,
            }
        }),

        HookCondition::MinGold { amount } => state.guild.gold >= *amount,

        HookCondition::MinAdventurers { count } => {
            state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .count()
                >= *count
        }

        HookCondition::NoPendingChoice => state.pending_choices.is_empty(),
    })
}
