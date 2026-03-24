//! Folk hero reputation system — tracks how common people view the guild.
//!
//! Separate from faction reputation, this measures the guild's standing among
//! ordinary citizens. Fame grows from heroic deeds (rescuing civilians, defending
//! settlements, curing disease) and decays from taxation, corruption, and neglect.
//!
//! Regional fame thresholds unlock benefits (cheap prices, volunteers, militia)
//! or penalties (suspicion, hostility).
//!
//! Fires every 500 ticks (~50s game time).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the folk hero system ticks (in ticks).
const FOLK_HERO_INTERVAL: u64 = 500;

/// Maximum number of active folk tales.
const MAX_FOLK_TALES: usize = 20;

/// Ticks before a folk tale fades from memory.
const TALE_LIFETIME: u64 = 5000;

/// Fame threshold for positive regional effects.
const FAME_POSITIVE_THRESHOLD: f32 = 50.0;

/// Fame threshold for folk hero status.
const FAME_HERO_THRESHOLD: f32 = 75.0;

/// Fame threshold below which suspicion and hostility occur.
const FAME_SUSPICION_THRESHOLD: f32 = 20.0;

/// Passive fame decay per tick interval (people forget).
const FAME_DECAY_RATE: f32 = 0.5;

/// Tick the folk hero reputation system: detect fame-worthy events, update
/// regional fame, generate folk tales, and apply fame effects.
pub fn tick_folk_hero(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FOLK_HERO_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Detect fame-generating and fame-reducing events ---
    let fame_deltas = compute_fame_deltas(state);

    // --- Phase 2: Apply fame deltas to regional fame ---
    apply_fame_deltas(state, &fame_deltas, events);

    // --- Phase 3: Generate folk tales from significant events ---
    generate_folk_tales(state, &fame_deltas, events);

    // --- Phase 4: Spread folk tales to adjacent regions ---
    spread_folk_tales(state, events);

    // --- Phase 5: Apply fame threshold effects ---
    apply_fame_effects(state, events);

    // --- Phase 6: Passive decay + recalculate overall fame ---
    decay_and_recalculate(state);

    // --- Phase 7: Expire old tales ---
    expire_folk_tales(state);
}

// ---------------------------------------------------------------------------
// Fame delta computation
// ---------------------------------------------------------------------------

/// Per-region fame changes detected this tick.
struct FameDeltas {
    /// (region_id, delta, reason)
    changes: Vec<(usize, f32, &'static str)>,
}

fn compute_fame_deltas(state: &CampaignState) -> FameDeltas {
    let mut changes: Vec<(usize, f32, &'static str)> = Vec::new();
    let recent_cutoff = state.tick.saturating_sub(FOLK_HERO_INTERVAL);
    let region_count = state.overworld.regions.len();

    if region_count == 0 {
        return FameDeltas { changes };
    }

    // --- Positive fame sources ---

    // Rescuing civilians (from evacuations completed recently)
    for evac in &state.evacuations {
        if evac.completed && evac.source_region_id < region_count {
            // Check if completed in recent window (use tick heuristic)
            if state.tick.saturating_sub(evac.started_tick) < FOLK_HERO_INTERVAL * 2 {
                changes.push((evac.source_region_id, 10.0, "rescued civilians"));
            }
        }
    }

    // Defending settlements — completed quests with victories in recent window
    for quest in &state.completed_quests {
        let quest_tick = quest.completed_at_ms / CAMPAIGN_TICK_MS as u64;
        if quest_tick >= recent_cutoff {
            match quest.result {
                QuestResult::Victory => {
                    // Defending settlements (+8)
                    let region = quest_region(state, quest);
                    if region < region_count {
                        changes.push((region, 8.0, "defended settlement"));
                    }
                }
                QuestResult::Defeat => {
                    // Ignoring crises / failing to protect
                    let region = quest_region(state, quest);
                    if region < region_count {
                        changes.push((region, -10.0, "failed to protect"));
                    }
                }
                _ => {}
            }
        }
    }

    // Curing disease — high containment means the guild helped
    for disease in &state.diseases {
        if disease.containment > 80.0 {
            // Apply fame to all affected regions
            for &rid in &disease.affected_regions {
                if rid < region_count {
                    if state.tick.saturating_sub(disease.started_tick) < FOLK_HERO_INTERVAL * 3 {
                        changes.push((rid, 12.0, "cured disease"));
                    }
                }
            }
        }
    }

    // Charity/donations — guild has high gold and reputation
    if state.guild.gold > 300.0 && state.guild.reputation > 50.0 {
        // Spread charity fame to the guild's primary region
        changes.push((0, 5.0, "charitable reputation"));
    }

    // Completing bounties
    for bounty in &state.bounty_board {
        if bounty.completed {
            if let Some(rid) = bounty.region_id {
                if rid < region_count {
                    changes.push((rid, 3.0, "completed bounty"));
                }
            }
        }
    }

    // Defeating monsters — recently hunted monster populations
    for pop in &state.monster_populations {
        if pop.last_hunted_tick > 0
            && state.tick.saturating_sub(pop.last_hunted_tick) < FOLK_HERO_INTERVAL * 2
            && pop.region_id < region_count
        {
            changes.push((pop.region_id, 4.0, "defeated monsters"));
        }
    }

    // --- Negative fame sources ---

    // High taxation reduces fame
    for region in &state.overworld.regions {
        if region.tax_rate > 0.3 {
            let penalty = (region.tax_rate - 0.3) * 30.0; // up to -21 at max tax
            changes.push((region.id, -penalty, "high taxation"));
        }
    }

    // Corruption scandals
    if state.corruption.level > 30.0 {
        let penalty = state.corruption.level * 0.5; // up to -50
        changes.push((0, -penalty.min(15.0), "corruption scandal"));
    }

    // Civilian casualties from battles (approximated by recent quest casualties)
    for quest in &state.completed_quests {
        let quest_tick = quest.completed_at_ms / CAMPAIGN_TICK_MS as u64;
        if quest_tick >= recent_cutoff && quest.casualties > 1 {
            let region = quest_region(state, quest);
            if region < region_count {
                let penalty = (quest.casualties as f32 * 5.0).min(20.0);
                changes.push((region, -penalty, "civilian casualties"));
            }
        }
    }

    // Active crises that are being ignored (high threat, no active quests dispatched)
    let dispatched_count = state.active_quests.len();
    for region in &state.overworld.regions {
        if region.threat_level > 60.0 && dispatched_count == 0 {
            changes.push((region.id, -10.0, "ignored crisis"));
        }
    }

    FameDeltas { changes }
}

/// Determine which region a completed quest was associated with.
/// Falls back to region 0 (guild home).
fn quest_region(state: &CampaignState, _quest: &CompletedQuest) -> usize {
    // Use a deterministic fallback: spread across regions based on quest ID
    if state.overworld.regions.is_empty() {
        return 0;
    }
    // Quests don't directly store region_id, so distribute based on quest id
    _quest.id as usize % state.overworld.regions.len()
}

// ---------------------------------------------------------------------------
// Apply fame deltas
// ---------------------------------------------------------------------------

fn apply_fame_deltas(
    state: &mut CampaignState,
    deltas: &FameDeltas,
    events: &mut Vec<WorldEvent>,
) {
    for &(region_id, delta, _reason) in &deltas.changes {
        let current = state
            .folk_reputation
            .regional_fame
            .entry(region_id)
            .or_insert(50.0);
        let old = *current;
        *current = (*current + delta).clamp(0.0, 100.0);
        let new = *current;

        if (new - old).abs() > 2.0 {
            events.push(WorldEvent::FolkFameChanged {
                region: region_id,
                amount: new - old,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Folk tale generation
// ---------------------------------------------------------------------------

fn generate_folk_tales(
    state: &mut CampaignState,
    deltas: &FameDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.folk_reputation.folk_tales.len() >= MAX_FOLK_TALES {
        return;
    }

    // Find the most impactful positive event to generate a tale about
    let mut best_positive: Option<(usize, f32, &str)> = None;
    for &(region_id, delta, reason) in &deltas.changes {
        if delta > 5.0 {
            match &best_positive {
                None => best_positive = Some((region_id, delta, reason)),
                Some((_, best_delta, _)) if delta > *best_delta => {
                    best_positive = Some((region_id, delta, reason));
                }
                _ => {}
            }
        }
    }

    if let Some((region_id, impact, reason)) = best_positive {
        if state.folk_reputation.folk_tales.len() < MAX_FOLK_TALES {
            // Pick an adventurer to feature in the tale
            let adventurer_id = pick_featured_adventurer(state);
            let tale_text = generate_tale_text(state, reason, adventurer_id);
            let id = state.next_event_id;
            state.next_event_id += 1;

            state.folk_reputation.folk_tales.push(FolkTale {
                id,
                adventurer_id,
                tale: tale_text.clone(),
                region_id,
                fame_impact: impact,
                created_tick: state.tick,
            });

            events.push(WorldEvent::FolkTaleCreated {
                tale: tale_text,
                region: region_id,
                adventurer_id,
                fame_impact: impact,
            });
        }
    }

    // Also generate a negative tale if there's a big negative event
    let mut worst_negative: Option<(usize, f32, &str)> = None;
    for &(region_id, delta, reason) in &deltas.changes {
        if delta < -8.0 {
            match &worst_negative {
                None => worst_negative = Some((region_id, delta, reason)),
                Some((_, best_delta, _)) if delta < *best_delta => {
                    worst_negative = Some((region_id, delta, reason));
                }
                _ => {}
            }
        }
    }

    if let Some((region_id, impact, reason)) = worst_negative {
        if state.folk_reputation.folk_tales.len() < MAX_FOLK_TALES {
            let tale_text = generate_negative_tale_text(state, reason);
            let id = state.next_event_id;
            state.next_event_id += 1;

            state.folk_reputation.folk_tales.push(FolkTale {
                id,
                adventurer_id: 0, // no specific adventurer for negative tales
                tale: tale_text.clone(),
                region_id,
                fame_impact: impact,
                created_tick: state.tick,
            });

            events.push(WorldEvent::FolkTaleCreated {
                tale: tale_text,
                region: region_id,
                adventurer_id: 0,
                fame_impact: impact,
            });
        }
    }
}

/// Pick an adventurer to feature in a folk tale, preferring active high-level ones.
fn pick_featured_adventurer(state: &mut CampaignState) -> u32 {
    let alive: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    if alive.is_empty() {
        return 0;
    }

    let idx = lcg_next(&mut state.rng) as usize % alive.len();
    alive[idx]
}

fn generate_tale_text(state: &mut CampaignState, reason: &str, adventurer_id: u32) -> String {
    let name = state
        .adventurers
        .iter()
        .find(|a| a.id == adventurer_id)
        .map(|a| a.name.as_str())
        .unwrap_or("a brave adventurer");

    let templates = match reason {
        "rescued civilians" => &[
            "They say {} carried children to safety through flame and ruin.",
            "Folk whisper how {} shielded the helpless when all seemed lost.",
            "The people sing of {} who stood between danger and the innocent.",
        ][..],
        "defended settlement" => &[
            "{} held the gate when the horde came, and not a soul was lost.",
            "The village still stands because {} refused to retreat.",
            "Farmers raise a toast to {} who drove the threat from their lands.",
        ][..],
        "cured disease" => &[
            "{} brought the remedy when healers had given up hope.",
            "The plague lifted after {} arrived — some call it a miracle.",
            "Children who would have perished now play, thanks to {}.",
        ][..],
        "charitable reputation" => &[
            "The guild's generosity is spoken of in every tavern.",
            "Even the poorest know the guild shares its fortune freely.",
            "{} ensures the guild never turns away those in need.",
        ][..],
        "completed bounty" => &[
            "{} tracked the menace and ended it — the roads are safe again.",
            "The bounty is claimed and the people rest easier, thanks to {}.",
        ][..],
        "defeated monsters" => &[
            "{} slew the beasts that terrorized the countryside.",
            "The monster threat faded after {} led the hunt.",
            "Shepherds no longer fear the night since {} cleared the hills.",
        ][..],
        _ => &[
            "The deeds of {} are spoken of with admiration.",
            "{} has earned the gratitude of common folk everywhere.",
        ][..],
    };

    let idx = lcg_next(&mut state.rng) as usize % templates.len();
    templates[idx].replace("{}", name)
}

fn generate_negative_tale_text(state: &mut CampaignState, reason: &str) -> String {
    let templates = match reason {
        "high taxation" => &[
            "The guild squeezes coin from those who have nothing left.",
            "Peasants grumble that the guild's taxes leave them hungry.",
        ][..],
        "corruption scandal" => &[
            "Word spreads of backroom deals and guild coffers lining private pockets.",
            "Trust erodes as corruption within the guild comes to light.",
        ][..],
        "civilian casualties" => &[
            "Innocents died in the crossfire, and the guild did nothing to prevent it.",
            "The market square still bears scars from the guild's reckless battle.",
        ][..],
        "ignored crisis" => &[
            "When the people cried for help, the guild was nowhere to be found.",
            "The guild turned a blind eye while danger consumed the region.",
        ][..],
        "failed to protect" => &[
            "The guild's adventurers fell, and the settlement paid the price.",
            "Hope faded when the guild could not hold the line.",
        ][..],
        _ => &[
            "Dark rumors circulate about the guild's indifference.",
            "The people's faith in the guild wavers.",
        ][..],
    };

    let idx = lcg_next(&mut state.rng) as usize % templates.len();
    templates[idx].to_string()
}

// ---------------------------------------------------------------------------
// Folk tale spreading
// ---------------------------------------------------------------------------

fn spread_folk_tales(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.overworld.regions.is_empty() {
        return;
    }

    let region_neighbors: Vec<Vec<usize>> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.neighbors.clone())
        .collect();
    let region_count = state.overworld.regions.len();

    for tale in &state.folk_reputation.folk_tales {
        // Spread fame impact to adjacent regions (diminished)
        let tale_region = tale.region_id;
        if tale_region >= region_count {
            continue;
        }

        // Only spread tales that are relatively fresh
        if state.tick.saturating_sub(tale.created_tick) > TALE_LIFETIME / 2 {
            continue;
        }

        for &neighbor in &region_neighbors[tale_region] {
            if neighbor < region_count {
                let spread_impact = tale.fame_impact * 0.3; // diminished in adjacent regions
                let current = state
                    .folk_reputation
                    .regional_fame
                    .entry(neighbor)
                    .or_insert(50.0);
                *current = (*current + spread_impact * 0.1).clamp(0.0, 100.0);
            }
        }
    }

    // Emit an event if any tale is still actively spreading
    let spreading_count = state
        .folk_reputation
        .folk_tales
        .iter()
        .filter(|t| state.tick.saturating_sub(t.created_tick) < TALE_LIFETIME / 2)
        .count();
    if spreading_count > 0 {
        // No individual event per spread — the fame changes are tracked via FolkFameChanged
        let _ = events; // already emitted via apply_fame_deltas
    }
}

// ---------------------------------------------------------------------------
// Fame threshold effects
// ---------------------------------------------------------------------------

fn apply_fame_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let region_count = state.overworld.regions.len();

    for region_id in 0..region_count {
        let fame = *state
            .folk_reputation
            .regional_fame
            .get(&region_id)
            .unwrap_or(&50.0);

        if fame > FAME_HERO_THRESHOLD {
            // Folk hero status: inspired militia (+defense), tribute gifts
            // Apply defense bonus via reduced unrest
            if state.overworld.regions[region_id].unrest > 5.0 {
                state.overworld.regions[region_id].unrest -= 2.0;
            }
            // Tribute gifts: small gold bonus
            let tribute = 2.0;
            state.guild.gold += tribute;

            // Check if we should emit a folk hero event
            // Only emit once per hero threshold crossing (use a simple heuristic)
            let prev_fame = fame - 1.0; // approximate previous
            if prev_fame <= FAME_HERO_THRESHOLD {
                // Find the most famous adventurer for this region
                let hero_id = pick_region_hero(state, region_id);
                events.push(WorldEvent::FolkHeroStatus {
                    region: region_id,
                    adventurer: hero_id,
                });
            }
        } else if fame > FAME_POSITIVE_THRESHOLD {
            // Positive fame: cheaper prices (handled via market modifier), volunteers
            // Small unrest reduction
            if state.overworld.regions[region_id].unrest > 10.0 {
                state.overworld.regions[region_id].unrest -= 0.5;
            }
        } else if fame < FAME_SUSPICION_THRESHOLD {
            // Suspicion: higher prices, increased unrest, potential hostility
            state.overworld.regions[region_id].unrest =
                (state.overworld.regions[region_id].unrest + 1.5).min(100.0);
        }
    }

    // Apply price modifiers based on overall fame
    let overall = state.folk_reputation.overall_fame;
    if overall > FAME_POSITIVE_THRESHOLD {
        // Cheaper prices
        let discount = (overall - FAME_POSITIVE_THRESHOLD) * 0.002; // up to ~10% at 100 fame
        state.guild.market_prices.supply_multiplier =
            (state.guild.market_prices.supply_multiplier - discount * 0.1).max(0.5);
    } else if overall < FAME_SUSPICION_THRESHOLD {
        // More expensive
        let markup = (FAME_SUSPICION_THRESHOLD - overall) * 0.003;
        state.guild.market_prices.supply_multiplier =
            (state.guild.market_prices.supply_multiplier + markup * 0.1).min(3.0);
    }
}

/// Pick the most relevant adventurer for a region's folk hero status.
fn pick_region_hero(state: &mut CampaignState, _region_id: usize) -> u32 {
    // Find adventurers featured in folk tales for this region
    let featured: Vec<u32> = state
        .folk_reputation
        .folk_tales
        .iter()
        .filter(|t| t.region_id == _region_id && t.adventurer_id != 0)
        .map(|t| t.adventurer_id)
        .collect();

    if let Some(&id) = featured.last() {
        id
    } else {
        pick_featured_adventurer(state)
    }
}

// ---------------------------------------------------------------------------
// Passive decay and overall fame recalculation
// ---------------------------------------------------------------------------

fn decay_and_recalculate(state: &mut CampaignState) {
    // Passive decay: fame drifts toward 50 (neutral)
    for fame in state.folk_reputation.regional_fame.values_mut() {
        if *fame > 50.0 {
            *fame = (*fame - FAME_DECAY_RATE).max(50.0);
        } else if *fame < 50.0 {
            *fame = (*fame + FAME_DECAY_RATE * 0.5).min(50.0); // recover from bad rep slower
        }
    }

    // Recalculate overall fame as weighted average across all regions
    if state.folk_reputation.regional_fame.is_empty() {
        state.folk_reputation.overall_fame = 50.0;
    } else {
        let sum: f32 = state.folk_reputation.regional_fame.values().sum();
        let count = state.folk_reputation.regional_fame.len() as f32;
        state.folk_reputation.overall_fame = sum / count;
    }
}

// ---------------------------------------------------------------------------
// Folk tale expiry
// ---------------------------------------------------------------------------

fn expire_folk_tales(state: &mut CampaignState) {
    let current_tick = state.tick;
    state.folk_reputation.folk_tales.retain(|tale| {
        current_tick.saturating_sub(tale.created_tick) < TALE_LIFETIME
    });
}
