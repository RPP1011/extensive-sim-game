//! Adventurer fears and phobias system — fires every 300 ticks.
//!
//! Adventurers can develop fears from traumatic experiences that affect their
//! effectiveness. Fears have severity (0-100), can be triggered in relevant
//! situations, and can be overcome through successful exposure.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to tick the fear system (in ticks).
const FEAR_INTERVAL: u64 = 300;

/// Chance of developing Darkness fear after near-death in dungeon.
const DARKNESS_FEAR_CHANCE: f32 = 0.30;

/// Chance of developing fear of specific monster type after defeat.
const MONSTER_FEAR_CHANCE: f32 = 0.20;

/// Chance of developing Crowds fear after contracting disease.
const CROWDS_FEAR_CHANCE: f32 = 0.15;

/// Chance of developing Authority fear after betrayal events.
const AUTHORITY_FEAR_CHANCE: f32 = 0.25;

/// Fear contagion chance from a fearful adventurer to party members.
const FEAR_CONTAGION_CHANCE: f32 = 0.05;

/// Number of times a fear must be overcome before it converts to a conquered bonus.
const OVERCOME_THRESHOLD: u32 = 3;

/// Severity reduction when successfully completing a feared activity.
const OVERCOME_SEVERITY_REDUCTION: f32 = 10.0;

/// Severity reduction per tick from mentorship by a fearless adventurer.
const MENTORSHIP_SEVERITY_REDUCTION: f32 = 5.0;

/// Conquered fear bonus effectiveness multiplier.
const CONQUERED_BONUS: f32 = 0.05;

/// Tick the fear system. Called every tick, but only processes every
/// `FEAR_INTERVAL` ticks.
pub fn tick_fears(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FEAR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Phase 1: Check for fear acquisition from traumatic events
    acquire_fears(state, events);

    // Phase 2: Apply fear effects (morale, effectiveness penalties)
    apply_fear_effects(state, events);

    // Phase 3: Process fear overcoming (successful exposure)
    process_overcoming(state, events);

    // Phase 4: Mentorship-based fear reduction
    mentorship_reduction(state, events);

    // Phase 5: Fear contagion within parties
    spread_fear_contagion(state, events);
}

/// Check for fear acquisition based on recent traumatic experiences.
fn acquire_fears(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Collect adventurer data for fear acquisition checks
    let mut new_fears: Vec<(u32, FearType, f32)> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // Already has max 4 fears — don't pile on
        if adv.fears.len() >= 4 {
            continue;
        }

        // Near-death check: injury >= 80 suggests near-death experience
        // If currently on a dungeon-related quest → Darkness fear
        if adv.injury >= 80.0 && !has_fear(&adv.fears, &FearType::Darkness) {
            // Check if adventurer is in a dungeon-like context (exploration or combat quest)
            let in_dungeon = adv.status == AdventurerStatus::Fighting
                || adv.status == AdventurerStatus::OnMission;
            if in_dungeon {
                let roll = lcg_f32(&mut state.rng);
                if roll < DARKNESS_FEAR_CHANCE {
                    let severity = 20.0 + lcg_f32(&mut state.rng) * 40.0;
                    new_fears.push((adv.id, FearType::Darkness, severity));
                }
            }
        }

        // Defeat by undead-related quests → Undead fear
        // Approximate: high injury + combat quest
        if adv.injury >= 60.0
            && adv.status == AdventurerStatus::Fighting
            && !has_fear(&adv.fears, &FearType::Undead)
        {
            let roll = lcg_f32(&mut state.rng);
            if roll < MONSTER_FEAR_CHANCE {
                let severity = 15.0 + lcg_f32(&mut state.rng) * 35.0;
                new_fears.push((adv.id, FearType::Undead, severity));
            }
        }

        // Disease → Crowds fear
        if adv.disease_status != DiseaseStatus::Healthy
            && !has_fear(&adv.fears, &FearType::Crowds)
        {
            let roll = lcg_f32(&mut state.rng);
            if roll < CROWDS_FEAR_CHANCE {
                let severity = 15.0 + lcg_f32(&mut state.rng) * 30.0;
                new_fears.push((adv.id, FearType::Crowds, severity));
            }
        }

        // Low loyalty (betrayal proxy) → Authority fear
        if adv.loyalty < 20.0 && !has_fear(&adv.fears, &FearType::Authority) {
            let roll = lcg_f32(&mut state.rng);
            if roll < AUTHORITY_FEAR_CHANCE {
                let severity = 20.0 + lcg_f32(&mut state.rng) * 30.0;
                new_fears.push((adv.id, FearType::Authority, severity));
            }
        }

        // High stress in solo assignment → Isolation fear
        if adv.stress >= 70.0
            && adv.party_id.is_none()
            && !has_fear(&adv.fears, &FearType::Isolation)
        {
            let roll = lcg_f32(&mut state.rng);
            if roll < MONSTER_FEAR_CHANCE {
                let severity = 15.0 + lcg_f32(&mut state.rng) * 30.0;
                new_fears.push((adv.id, FearType::Isolation, severity));
            }
        }

        // High fatigue in traveling → Heights/Water fears (terrain-based)
        if adv.fatigue >= 70.0 && adv.status == AdventurerStatus::Traveling {
            if !has_fear(&adv.fears, &FearType::Heights) {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.10 {
                    let severity = 15.0 + lcg_f32(&mut state.rng) * 25.0;
                    new_fears.push((adv.id, FearType::Heights, severity));
                }
            }
            if !has_fear(&adv.fears, &FearType::Water) {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.10 {
                    let severity = 15.0 + lcg_f32(&mut state.rng) * 25.0;
                    new_fears.push((adv.id, FearType::Water, severity));
                }
            }
        }

        // Fire fear from high-injury combat situations
        if adv.injury >= 50.0
            && adv.stress >= 50.0
            && !has_fear(&adv.fears, &FearType::Fire)
        {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.08 {
                let severity = 15.0 + lcg_f32(&mut state.rng) * 30.0;
                new_fears.push((adv.id, FearType::Fire, severity));
            }
        }
    }

    // Apply new fears
    for (adv_id, fear_type, severity) in new_fears {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            // Don't add duplicates
            if has_fear(&adv.fears, &fear_type) {
                continue;
            }
            let fear = Fear {
                fear_type: fear_type.clone(),
                severity,
                acquired_tick: tick,
                times_triggered: 0,
                times_overcome: 0,
            };
            adv.fears.push(fear);

            events.push(WorldEvent::FearDeveloped {
                adventurer_id: adv_id,
                fear_type,
                severity,
            });
        }
    }
}

/// Apply fear effects: morale penalties and effectiveness reduction.
fn apply_fear_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect trigger info to avoid borrow issues
    let mut triggers: Vec<(u32, usize)> = Vec::new(); // (adv_id, fear_index)

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        for (fi, fear) in adv.fears.iter().enumerate() {
            if is_fear_relevant(adv, fear) {
                triggers.push((adv.id, fi));
            }
        }
    }

    // Apply effects
    for (adv_id, fear_idx) in triggers {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if fear_idx >= adv.fears.len() {
                continue;
            }

            let severity = adv.fears[fear_idx].severity;
            let fear_type = adv.fears[fear_idx].fear_type.clone();

            // Track triggering
            adv.fears[fear_idx].times_triggered += 1;

            // Morale penalty proportional to severity: -5 to -15
            let morale_penalty = 5.0 + (severity / 100.0) * 10.0;
            adv.morale = (adv.morale - morale_penalty).max(0.0);

            // Stress increase proportional to severity
            let stress_increase = 3.0 + (severity / 100.0) * 7.0;
            adv.stress = (adv.stress + stress_increase).min(100.0);

            events.push(WorldEvent::FearTriggered {
                adventurer_id: adv_id,
                fear_type,
                severity,
            });
        }
    }
}

/// Process fear overcoming: successful completion while feared reduces severity.
fn process_overcoming(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Adventurers who are idle (returned from quest) with fears that were triggered
    // have a chance to overcome if they survived
    let mut overcomes: Vec<(u32, usize)> = Vec::new();
    let mut conquests: Vec<(u32, usize)> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        // Idle adventurers who recently returned — check if fears were triggered
        if adv.status != AdventurerStatus::Idle {
            continue;
        }

        for (fi, fear) in adv.fears.iter().enumerate() {
            // Only process fears that have been triggered at least once
            if fear.times_triggered == 0 {
                continue;
            }
            // Check if this adventurer has low-enough injury to count as "survived"
            if adv.injury < 50.0 && adv.stress < 60.0 {
                overcomes.push((adv.id, fi));
            }
        }
    }

    for (adv_id, fear_idx) in overcomes {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if fear_idx >= adv.fears.len() {
                continue;
            }

            adv.fears[fear_idx].severity =
                (adv.fears[fear_idx].severity - OVERCOME_SEVERITY_REDUCTION).max(0.0);
            adv.fears[fear_idx].times_overcome += 1;

            let fear_type = adv.fears[fear_idx].fear_type.clone();
            let severity = adv.fears[fear_idx].severity;

            if adv.fears[fear_idx].times_overcome >= OVERCOME_THRESHOLD {
                conquests.push((adv_id, fear_idx));
            } else {
                events.push(WorldEvent::FearOvercome {
                    adventurer_id: adv_id,
                    fear_type,
                    new_severity: severity,
                    times_overcome: adv.fears[fear_idx].times_overcome,
                });
            }
        }
    }

    // Process conquests (remove fear, add conquered trait)
    // Sort descending to remove by index safely
    conquests.sort_by(|a, b| b.1.cmp(&a.1));
    for (adv_id, fear_idx) in conquests {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if fear_idx >= adv.fears.len() {
                continue;
            }
            let fear_type = adv.fears[fear_idx].fear_type.clone();
            adv.fears.remove(fear_idx);

            // Add a conquered trait for the bonus
            let trait_name = format!("conquered_{}", fear_type.trait_suffix());
            if !adv.traits.contains(&trait_name) {
                adv.traits.push(trait_name);
            }

            events.push(WorldEvent::FearConquered {
                adventurer_id: adv_id,
                fear_type,
            });
        }
    }
}

/// Mentorship-based fear reduction: fearless adventurers in the same party
/// can help reduce fear severity.
fn mentorship_reduction(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find parties with both fearful and fearless adventurers
    let mut reductions: Vec<(u32, usize)> = Vec::new(); // (adv_id, fear_index)

    for party in &state.parties {
        if party.member_ids.len() < 2 {
            continue;
        }

        // Check if any party member has no fears (the mentor)
        let has_fearless = party.member_ids.iter().any(|&mid| {
            state
                .adventurers
                .iter()
                .find(|a| a.id == mid)
                .map(|a| a.fears.is_empty() && a.status != AdventurerStatus::Dead)
                .unwrap_or(false)
        });

        if !has_fearless {
            continue;
        }

        // Fearful members get severity reduction
        for &mid in &party.member_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == mid) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                for fi in 0..adv.fears.len() {
                    reductions.push((mid, fi));
                }
            }
        }
    }

    for (adv_id, fear_idx) in reductions {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if fear_idx >= adv.fears.len() {
                continue;
            }
            let old_severity = adv.fears[fear_idx].severity;
            adv.fears[fear_idx].severity =
                (old_severity - MENTORSHIP_SEVERITY_REDUCTION).max(0.0);

            // If severity dropped to zero, remove the fear
            if adv.fears[fear_idx].severity <= 0.0 {
                let fear_type = adv.fears[fear_idx].fear_type.clone();
                adv.fears.remove(fear_idx);
                events.push(WorldEvent::FearOvercome {
                    adventurer_id: adv_id,
                    fear_type,
                    new_severity: 0.0,
                    times_overcome: 0,
                });
            }
        }
    }
}

/// Fear contagion: fearful adventurers can spread fear to party members.
fn spread_fear_contagion(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut new_fears: Vec<(u32, FearType, f32)> = Vec::new();

    for party in &state.parties {
        if party.member_ids.len() < 2 {
            continue;
        }

        // Find the most severe fear in the party
        let mut worst_fear: Option<(FearType, f32)> = None;
        for &mid in &party.member_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == mid) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                // Only spread from high-stress adventurers
                if adv.stress < 60.0 {
                    continue;
                }
                for fear in &adv.fears {
                    if fear.severity > 50.0 {
                        match &worst_fear {
                            None => worst_fear = Some((fear.fear_type.clone(), fear.severity)),
                            Some((_, s)) if fear.severity > *s => {
                                worst_fear = Some((fear.fear_type.clone(), fear.severity))
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let (contagion_type, source_severity) = match worst_fear {
            Some(f) => f,
            None => continue,
        };

        // Try to spread to other party members who don't have this fear
        for &mid in &party.member_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == mid) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                if has_fear(&adv.fears, &contagion_type) {
                    continue;
                }
                if adv.fears.len() >= 4 {
                    continue;
                }

                let roll = lcg_f32(&mut state.rng);
                if roll < FEAR_CONTAGION_CHANCE {
                    // Contagion fears are weaker than the source
                    let severity = source_severity * 0.5;
                    new_fears.push((mid, contagion_type.clone(), severity));
                }
            }
        }
    }

    for (adv_id, fear_type, severity) in new_fears {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if has_fear(&adv.fears, &fear_type) {
                continue;
            }
            let fear = Fear {
                fear_type: fear_type.clone(),
                severity,
                acquired_tick: tick,
                times_triggered: 0,
                times_overcome: 0,
            };
            adv.fears.push(fear);

            events.push(WorldEvent::FearDeveloped {
                adventurer_id: adv_id,
                fear_type,
                severity,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if an adventurer already has a specific fear type.
fn has_fear(fears: &[Fear], fear_type: &FearType) -> bool {
    fears.iter().any(|f| f.fear_type == *fear_type)
}

/// Check if a fear is relevant to the adventurer's current situation.
fn is_fear_relevant(adv: &Adventurer, fear: &Fear) -> bool {
    match &fear.fear_type {
        FearType::Darkness => {
            // Relevant when on dungeon-like missions
            adv.status == AdventurerStatus::OnMission || adv.status == AdventurerStatus::Fighting
        }
        FearType::Heights => {
            // Relevant when traveling
            adv.status == AdventurerStatus::Traveling
        }
        FearType::Water => {
            // Relevant when traveling
            adv.status == AdventurerStatus::Traveling
        }
        FearType::Undead => {
            // Relevant during combat
            adv.status == AdventurerStatus::Fighting
        }
        FearType::Fire => {
            // Relevant during combat or missions
            adv.status == AdventurerStatus::Fighting || adv.status == AdventurerStatus::OnMission
        }
        FearType::Crowds => {
            // Relevant when idle in settlement
            adv.status == AdventurerStatus::Idle
        }
        FearType::Isolation => {
            // Relevant when alone (no party)
            adv.party_id.is_none()
                && (adv.status == AdventurerStatus::Traveling
                    || adv.status == AdventurerStatus::OnMission)
        }
        FearType::Authority => {
            // Relevant when assigned to faction-interaction quests
            adv.status == AdventurerStatus::Assigned || adv.status == AdventurerStatus::OnMission
        }
    }
}
