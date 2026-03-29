//! Reputation stories — tales of guild deeds spread across regions.
//!
//! Stories about the guild's actions propagate through the world via region
//! adjacency, degrading in accuracy as they travel (Chinese telephone effect).
//! Distant NPCs and factions react to stories before the guild arrives.
//!
//! Fires every 500 ticks (~50s game time).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the reputation stories system ticks (in ticks).
const STORY_INTERVAL: u64 = 17;

/// Maximum number of active stories at once.
const MAX_ACTIVE_STORIES: usize = 10;

/// Ticks before a story fades away.
const STORY_LIFETIME: u64 = 3000;

/// Accuracy multiplier per region hop (Chinese telephone degradation).
const ACCURACY_DECAY: f32 = 0.8;

/// Tick reputation stories: generate new stories from events, spread existing
/// ones to adjacent regions, apply effects, and decay old stories.
pub fn tick_reputation_stories(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % STORY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Generate new stories from recent events ---
    generate_stories(state, events);

    // --- Phase 2: Spread existing stories to adjacent regions ---
    spread_stories(state, events);

    // --- Phase 3: Apply story effects in reached regions ---
    apply_story_effects(state);

    // --- Phase 4: Decay old stories ---
    decay_stories(state, events);
}

// ---------------------------------------------------------------------------
// Story generation — significant events create stories
// ---------------------------------------------------------------------------

fn generate_stories(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.reputation_stories.len() >= MAX_ACTIVE_STORIES {
        return;
    }

    let mut new_stories: Vec<ReputationStory> = Vec::new();

    // Check recently completed quests for story-worthy events.
    // Collect quest data first to avoid borrow conflicts with state.rng.
    let recent_cutoff = state.tick.saturating_sub(STORY_INTERVAL);
    let quest_candidates: Vec<(u32, QuestResult, f32, u32)> = state
        .completed_quests
        .iter()
        .filter(|q| {
            let quest_tick = q.completed_at_ms / (CAMPAIGN_TURN_SECS as u64 * 1000);
            quest_tick >= recent_cutoff
        })
        .map(|q| {
            let significance = q.reward_applied.gold + q.reward_applied.reputation * 2.0;
            (q.id, q.result, significance, q.casualties)
        })
        .collect();

    for (quest_id, result, quest_significance, casualties) in quest_candidates {
        // Already at capacity?
        if state.reputation_stories.len() + new_stories.len() >= MAX_ACTIVE_STORIES {
            break;
        }

        // Already have a story about this quest? (dedup by checking text prefix)
        let quest_tag = format!("quest#{}", quest_id);
        let already_has = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| s.text.contains(&quest_tag));
        if already_has {
            continue;
        }

        match result {
            QuestResult::Victory if quest_significance > 50.0 => {
                // Heroic victory against a significant quest
                let impact = (20.0 + quest_significance * 0.3).min(100.0);
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text = format!(
                    "The guild triumphed over a fearsome threat! [quest#{}]",
                    quest_id
                );
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::HeroicVictory,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
            QuestResult::Defeat => {
                if casualties > 0 {
                    // Heroic last stand
                    let impact = (10.0 + casualties as f32 * 5.0).min(100.0);
                    let origin = pick_origin_region(state);
                    let id = state.next_event_id;
                    state.next_event_id += 1;
                    let text = format!(
                        "Guild adventurers made a heroic last stand against impossible odds. [quest#{}]",
                        quest_id
                    );
                    new_stories.push(ReputationStory {
                        id,
                        story_type: StoryType::HeroicVictory,
                        text: text.clone(),
                        origin_region_id: origin,
                        spread_to: vec![origin],
                        created_tick: state.tick,
                        accuracy: 1.0,
                        impact,
                    });
                    events.push(WorldEvent::StoryCreated { text, impact });
                } else {
                    // Cruel defeat — guild failed
                    let impact = (-15.0 - quest_significance * 0.2).max(-100.0);
                    let origin = pick_origin_region(state);
                    let id = state.next_event_id;
                    state.next_event_id += 1;
                    let text = format!(
                        "The guild was defeated and fled in disgrace. [quest#{}]",
                        quest_id
                    );
                    new_stories.push(ReputationStory {
                        id,
                        story_type: StoryType::CruelDefeat,
                        text: text.clone(),
                        origin_region_id: origin,
                        spread_to: vec![origin],
                        created_tick: state.tick,
                        accuracy: 1.0,
                        impact,
                    });
                    events.push(WorldEvent::StoryCreated { text, impact });
                }
            }
            _ => {}
        }
    }

    // Check for adventurer deaths (recent)
    // We track this via event_log — look for death events in the recent window.
    let dead_count = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .count();
    if dead_count > 0 && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES {
        // Only generate a death story if guild reputation is high (implies well-known)
        // and we haven't already generated one recently.
        let has_death_story = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| {
                s.story_type == StoryType::CruelDefeat
                    && state.tick.saturating_sub(s.created_tick) < STORY_INTERVAL * 2
            });
        if !has_death_story && state.guild.reputation > 40.0 {
            // Roll for it
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.3 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text =
                    "Tales spread of adventurers who fell in service of the guild.".to_string();
                let impact = -10.0;
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::CruelDefeat,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
        }
    }

    // Check for generous deeds — high reputation + large gold reserves
    if state.guild.reputation > 60.0
        && state.guild.gold > 200.0
        && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES
    {
        let has_generous = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| s.story_type == StoryType::GenerousDeed);
        if !has_generous {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.2 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text = "The guild is known for its generosity and fair dealings.".to_string();
                let impact = 15.0;
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::GenerousDeed,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
        }
    }

    // Check for war-related stories
    let at_war_factions: Vec<String> = state
        .factions
        .iter()
        .filter(|f| f.relationship_to_guild < -50.0)
        .map(|f| f.name.clone())
        .collect();
    if !at_war_factions.is_empty()
        && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES
    {
        let has_war_story = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| {
                s.story_type == StoryType::WarAtrocity
                    && state.tick.saturating_sub(s.created_tick) < STORY_INTERVAL * 3
            });
        if !has_war_story {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.25 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                // Determine if this is justified defense or atrocity based on guild rep
                if state.guild.reputation > 50.0 {
                    let text = format!(
                        "The guild defends the realm against hostile forces from {}.",
                        at_war_factions[0]
                    );
                    let impact = 12.0;
                    new_stories.push(ReputationStory {
                        id,
                        story_type: StoryType::PeacemakerTale,
                        text: text.clone(),
                        origin_region_id: origin,
                        spread_to: vec![origin],
                        created_tick: state.tick,
                        accuracy: 1.0,
                        impact,
                    });
                    events.push(WorldEvent::StoryCreated { text, impact });
                } else {
                    let text = format!(
                        "Rumors of the guild's brutal war with {} spread fear.",
                        at_war_factions[0]
                    );
                    let impact = -18.0;
                    new_stories.push(ReputationStory {
                        id,
                        story_type: StoryType::WarAtrocity,
                        text: text.clone(),
                        origin_region_id: origin,
                        spread_to: vec![origin],
                        created_tick: state.tick,
                        accuracy: 1.0,
                        impact,
                    });
                    events.push(WorldEvent::StoryCreated { text, impact });
                }
            }
        }
    }

    // Check for peace treaties (high faction relations)
    let peaceful_factions = state
        .factions
        .iter()
        .filter(|f| f.relationship_to_guild > 70.0 && f.coalition_member)
        .count();
    if peaceful_factions > 0
        && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES
    {
        let has_peace = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| s.story_type == StoryType::PeacemakerTale);
        if !has_peace {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.15 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text =
                    "The guild has forged lasting peace through diplomacy and coalition."
                        .to_string();
                let impact = 20.0;
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::PeacemakerTale,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
        }
    }

    // Rags to riches — guild went from low gold to high gold
    if state.guild.gold > 500.0
        && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES
    {
        let has_riches = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| s.story_type == StoryType::RagsToRiches);
        if !has_riches {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.1 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text =
                    "From humble beginnings, the guild has amassed great wealth.".to_string();
                let impact = 10.0;
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::RagsToRiches,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
        }
    }

    // Mysterious event — random chance
    if state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES {
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.05 {
            let origin = pick_origin_region(state);
            let id = state.next_event_id;
            state.next_event_id += 1;
            let mysteries = [
                "Strange lights were seen near the guild hall at midnight.",
                "A mysterious figure was spotted leaving the guild under cover of darkness.",
                "Travelers whisper of uncanny events surrounding the guild.",
                "The guild's banner was seen flying in places they have never visited.",
            ];
            let idx = (lcg_next(&mut state.rng) as usize) % mysteries.len();
            let text = mysteries[idx].to_string();
            // Mysterious events have small random impact
            let impact = (lcg_f32(&mut state.rng) * 10.0) - 5.0; // -5 to +5
            new_stories.push(ReputationStory {
                id,
                story_type: StoryType::MysteriousEvent,
                text: text.clone(),
                origin_region_id: origin,
                spread_to: vec![origin],
                created_tick: state.tick,
                accuracy: 1.0,
                impact,
            });
            events.push(WorldEvent::StoryCreated { text, impact });
        }
    }

    // Greedy act — low reputation + black market activity implied
    if state.guild.reputation < 30.0
        && state.reputation_stories.len() + new_stories.len() < MAX_ACTIVE_STORIES
    {
        let has_greedy = state
            .reputation_stories
            .iter()
            .chain(new_stories.iter())
            .any(|s| {
                s.story_type == StoryType::GreedyAct
                    && state.tick.saturating_sub(s.created_tick) < STORY_INTERVAL * 3
            });
        if !has_greedy {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.2 {
                let origin = pick_origin_region(state);
                let id = state.next_event_id;
                state.next_event_id += 1;
                let text = "Rumors circulate of the guild's unsavory dealings and profiteering."
                    .to_string();
                let impact = -12.0;
                new_stories.push(ReputationStory {
                    id,
                    story_type: StoryType::GreedyAct,
                    text: text.clone(),
                    origin_region_id: origin,
                    spread_to: vec![origin],
                    created_tick: state.tick,
                    accuracy: 1.0,
                    impact,
                });
                events.push(WorldEvent::StoryCreated { text, impact });
            }
        }
    }

    state.reputation_stories.extend(new_stories);
}

// ---------------------------------------------------------------------------
// Story spread — each tick, stories propagate to 1 adjacent region
// ---------------------------------------------------------------------------

fn spread_stories(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.overworld.regions.is_empty() {
        return;
    }

    // Build adjacency snapshot to avoid borrow issues.
    let region_neighbors: Vec<Vec<usize>> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.neighbors.clone())
        .collect();
    let region_count = state.overworld.regions.len();

    for story in &mut state.reputation_stories {
        // Find regions adjacent to current spread that haven't been reached yet.
        let mut candidates: Vec<usize> = Vec::new();
        for &region_id in &story.spread_to {
            if region_id < region_count {
                for &neighbor in &region_neighbors[region_id] {
                    if neighbor < region_count && !story.spread_to.contains(&neighbor) {
                        candidates.push(neighbor);
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        if candidates.is_empty() {
            continue;
        }

        // Pick one adjacent region to spread to.
        // Use a deterministic selection based on story id + tick to avoid needing &mut rng.
        let pick_idx = ((story.id as u64).wrapping_mul(31) + state.tick) as usize % candidates.len();
        let new_region = candidates[pick_idx];
        story.spread_to.push(new_region);
        story.accuracy *= ACCURACY_DECAY;

        let region_name = if new_region < state.overworld.regions.len() {
            state.overworld.regions[new_region].name.clone()
        } else {
            format!("region {}", new_region)
        };

        events.push(WorldEvent::StorySpread {
            story_text: story.text.clone(),
            region_name,
        });
    }
}

// ---------------------------------------------------------------------------
// Story effects — modify faction relations, prices, recruitment in regions
// ---------------------------------------------------------------------------

fn apply_story_effects(state: &mut CampaignState) {
    // Aggregate impact per region from all active stories.
    let mut region_impacts: Vec<f32> = vec![0.0; state.overworld.regions.len()];

    for story in &state.reputation_stories {
        for &region_id in &story.spread_to {
            if region_id < region_impacts.len() {
                // Impact scaled by accuracy — distorted stories have less effect
                region_impacts[region_id] += story.impact * story.accuracy * 0.01;
            }
        }
    }

    // Apply effects to factions and regions.
    for (region_id, &impact) in region_impacts.iter().enumerate() {
        if impact.abs() < 0.001 {
            continue;
        }

        // Find faction owning this region and adjust relationship.
        let owner_id = if region_id < state.overworld.regions.len() {
            state.overworld.regions[region_id].owner_faction_id
        } else {
            continue;
        };

        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == owner_id) {
            // Positive stories improve relations, negative stories worsen them.
            // Small incremental effect per tick to avoid wild swings.
            let delta = impact.clamp(-2.0, 2.0);
            faction.relationship_to_guild =
                (faction.relationship_to_guild + delta).clamp(-100.0, 100.0);
        }

        // Positive stories reduce unrest, negative stories increase it.
        if region_id < state.overworld.regions.len() {
            let unrest_delta = -impact * 0.5; // positive impact reduces unrest
            state.overworld.regions[region_id].unrest =
                (state.overworld.regions[region_id].unrest + unrest_delta).clamp(0.0, 100.0);
        }
    }

    // Positive stories across the board improve guild reputation slightly.
    let total_positive: f32 = state
        .reputation_stories
        .iter()
        .filter(|s| s.impact > 0.0)
        .map(|s| s.impact * s.accuracy * 0.001)
        .sum();
    let total_negative: f32 = state
        .reputation_stories
        .iter()
        .filter(|s| s.impact < 0.0)
        .map(|s| s.impact * s.accuracy * 0.001)
        .sum();

    state.guild.reputation =
        (state.guild.reputation + total_positive + total_negative).clamp(0.0, 100.0);

    // Market price effects — positive reputation = better prices.
    let net_reputation_impact = total_positive + total_negative;
    if net_reputation_impact.abs() > 0.01 {
        let price_modifier = -net_reputation_impact * 0.01; // positive rep → lower prices
        state.guild.market_prices.supply_multiplier =
            (state.guild.market_prices.supply_multiplier + price_modifier).max(0.5);
        state.guild.market_prices.recruitment_multiplier =
            (state.guild.market_prices.recruitment_multiplier + price_modifier).max(0.5);
    }
}

// ---------------------------------------------------------------------------
// Story decay — remove stories older than STORY_LIFETIME
// ---------------------------------------------------------------------------

fn decay_stories(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let current_tick = state.tick;
    let mut faded_texts: Vec<String> = Vec::new();

    state.reputation_stories.retain(|story| {
        if current_tick.saturating_sub(story.created_tick) >= STORY_LIFETIME {
            faded_texts.push(story.text.clone());
            false
        } else {
            true
        }
    });

    for text in faded_texts {
        events.push(WorldEvent::StoryFaded { text });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pick a region to originate a story from. Prefers the guild's home region.
fn pick_origin_region(state: &mut CampaignState) -> usize {
    if state.overworld.regions.is_empty() {
        return 0;
    }

    // Find region closest to guild base position.
    // Since regions don't have positions directly, use index 0 as default
    // and pick randomly with a bias toward the first region.
    let roll = lcg_next(&mut state.rng) as usize;
    if state.overworld.regions.len() <= 1 {
        0
    } else {
        // 50% chance of region 0 (guild home), otherwise random
        if roll % 2 == 0 {
            0
        } else {
            roll % state.overworld.regions.len()
        }
    }
}
