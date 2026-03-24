//! Weather hazards system — fires every 200 ticks.
//!
//! Generates season-dependent weather events (storms, blizzards, floods, etc.)
//! that block travel routes, damage buildings, strand parties, and disrupt
//! economies. Integrates with the seasonal cycle from `seasons.rs`.
//!
//! Weather severity (0–100) scales all effects linearly: severity 50 = half
//! the listed base effect.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for new weather and apply ongoing effects (in ticks).
const WEATHER_INTERVAL: u64 = 200;

// ---------------------------------------------------------------------------
// Season-dependent weather probabilities
// ---------------------------------------------------------------------------

/// (WeatherType, probability) pairs for each season.
fn weather_probabilities(season: Season) -> &'static [(WeatherType, f32)] {
    match season {
        Season::Winter => &[
            (WeatherType::Blizzard, 0.20),
            (WeatherType::Storm, 0.10),
        ],
        Season::Spring => &[
            (WeatherType::Flood, 0.15),
            (WeatherType::Storm, 0.10),
            (WeatherType::Fog, 0.10),
        ],
        Season::Summer => &[
            (WeatherType::Heatwave, 0.15),
            (WeatherType::Drought, 0.10),
        ],
        Season::Autumn => &[
            (WeatherType::Storm, 0.15),
            (WeatherType::Fog, 0.15),
        ],
    }
}

/// Base duration range (min, max) in ticks for each weather type.
fn duration_range(wt: WeatherType) -> (u64, u64) {
    match wt {
        WeatherType::Storm => (300, 600),
        WeatherType::Blizzard => (400, 800),
        WeatherType::Flood => (500, 800),
        WeatherType::Drought => (600, 800),
        WeatherType::Fog => (300, 500),
        WeatherType::Heatwave => (400, 700),
    }
}

// ---------------------------------------------------------------------------
// Main tick entry point
// ---------------------------------------------------------------------------

/// Advance weather: expire old events, apply ongoing effects, maybe spawn new ones.
pub fn tick_weather(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % WEATHER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire finished weather ---
    expire_weather(state, events);

    // --- Apply ongoing weather effects ---
    apply_weather_effects(state, events);

    // --- Maybe spawn new weather ---
    maybe_spawn_weather(state, events);
}

// ---------------------------------------------------------------------------
// Expiration
// ---------------------------------------------------------------------------

fn expire_weather(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut expired = Vec::new();

    state.overworld.active_weather.retain(|w| {
        let alive = tick < w.started_tick + w.duration;
        if !alive {
            expired.push(w.weather_type);
        }
        alive
    });

    for wt in expired {
        events.push(WorldEvent::WeatherEnded { weather_type: wt });
    }
}

// ---------------------------------------------------------------------------
// Ongoing effects
// ---------------------------------------------------------------------------

fn apply_weather_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect weather info so we don't borrow state mutably twice.
    let weather_snapshot: Vec<(WeatherType, Vec<usize>, f32)> = state
        .overworld
        .active_weather
        .iter()
        .map(|w| (w.weather_type, w.affected_regions.clone(), w.severity))
        .collect();

    for (wt, regions, severity) in &weather_snapshot {
        let scale = *severity / 100.0;
        match wt {
            WeatherType::Storm => apply_storm(state, events, regions, scale),
            WeatherType::Blizzard => apply_blizzard(state, events, regions, scale),
            WeatherType::Flood => apply_flood(state, events, regions, scale),
            WeatherType::Drought => apply_drought(state, events, regions, scale),
            WeatherType::Fog => apply_fog(state, events, regions, scale),
            WeatherType::Heatwave => apply_heatwave(state, events, regions, scale),
        }
    }
}

/// Storm: -30% travel speed, 5% party damage, -10 civilian morale.
fn apply_storm(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    // Slow traveling parties in affected regions.
    for party in &mut state.parties {
        if party.status != PartyStatus::Traveling && party.status != PartyStatus::Returning {
            continue;
        }
        if party_in_regions(party, &state.overworld.regions, regions) {
            // Reduce speed for this tick interval (applied as a temporary debuff).
            let speed_penalty = 0.30 * scale;
            party.speed *= 1.0 - speed_penalty;

            // 5% chance of party damage per weather tick.
            let dmg_roll = lcg_f32(&mut state.rng);
            if dmg_roll < 0.05 * scale {
                for &mid in &party.member_ids {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                        adv.injury = (adv.injury + 5.0 * scale).clamp(0.0, 100.0);
                    }
                }
                events.push(WorldEvent::WeatherDamage {
                    weather_type: WeatherType::Storm,
                    description: format!(
                        "Party {} battered by storm winds",
                        party.id
                    ),
                });
            }
        }
    }

    // Reduce morale in affected regions (via unrest increase).
    for &rid in regions {
        if let Some(region) = state.overworld.regions.get_mut(rid) {
            region.unrest = (region.unrest + 10.0 * scale * 0.01).clamp(0.0, 100.0);
        }
    }
}

/// Blizzard: -50% travel speed, +20 supply drain, buildings damaged (-5 building level).
fn apply_blizzard(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    for party in &mut state.parties {
        if party.status != PartyStatus::Traveling && party.status != PartyStatus::Returning {
            continue;
        }
        if party_in_regions(party, &state.overworld.regions, regions) {
            let speed_penalty = 0.50 * scale;
            party.speed *= 1.0 - speed_penalty;

            // Extra supply drain (+20 per weather interval at full severity).
            let drain = 20.0 * scale * 0.01; // per-tick fraction
            party.supply_level = (party.supply_level - drain).max(0.0);
        }
    }

    // Damage guild buildings if guild base region is affected.
    // We use region 0 as the guild base region heuristic.
    if regions.contains(&0) {
        let dmg_roll = lcg_f32(&mut state.rng);
        if dmg_roll < 0.10 * scale {
            let b = &mut state.guild_buildings;
            let bldgs = [
                ("training_grounds", &mut b.training_grounds),
                ("watchtower", &mut b.watchtower),
                ("trade_post", &mut b.trade_post),
                ("barracks", &mut b.barracks),
                ("infirmary", &mut b.infirmary),
                ("war_room", &mut b.war_room),
            ];
            for (name, tier) in bldgs {
                if *tier > 0 {
                    *tier = tier.saturating_sub(1);
                    events.push(WorldEvent::WeatherDamage {
                        weather_type: WeatherType::Blizzard,
                        description: format!(
                            "Blizzard damaged {} (now tier {})",
                            name, *tier
                        ),
                    });
                    break; // Only damage one building per interval.
                }
            }
        }
    }
}

/// Flood: trade routes disrupted (-50% income), population -2% (modeled as unrest).
fn apply_flood(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    // Reduce trade income.
    let income_penalty = 0.50 * scale * 0.01; // per-tick fraction
    state.guild.total_trade_income *= 1.0 - income_penalty;

    // Increase unrest in affected regions (proxy for population loss).
    for &rid in regions {
        if let Some(region) = state.overworld.regions.get_mut(rid) {
            region.unrest = (region.unrest + 2.0 * scale * 0.01).clamp(0.0, 100.0);
            region.control = (region.control - 1.0 * scale * 0.01).clamp(0.0, 100.0);
        }
    }

    // Block travel through flooded regions for parties.
    for party in &mut state.parties {
        if party.status != PartyStatus::Traveling && party.status != PartyStatus::Returning {
            continue;
        }
        if party_in_regions(party, &state.overworld.regions, regions) {
            // Severe slowdown.
            party.speed *= 1.0 - 0.40 * scale;
        }
    }

    if !regions.is_empty() && lcg_f32(&mut state.rng) < 0.05 * scale {
        events.push(WorldEvent::WeatherDamage {
            weather_type: WeatherType::Flood,
            description: "Flooding has disrupted trade routes".into(),
        });
    }
}

/// Drought: -30% population growth (unrest), supply costs +50%.
fn apply_drought(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    // Increase supply costs (market multiplier).
    let cost_increase = 0.50 * scale * 0.01; // per-tick fraction
    state.guild.market_prices.supply_multiplier *= 1.0 + cost_increase;
    // Cap to prevent runaway inflation.
    state.guild.market_prices.supply_multiplier =
        state.guild.market_prices.supply_multiplier.min(5.0);

    // Increase unrest in affected regions (proxy for growth reduction).
    for &rid in regions {
        if let Some(region) = state.overworld.regions.get_mut(rid) {
            region.unrest = (region.unrest + 1.5 * scale * 0.01).clamp(0.0, 100.0);
        }
    }

    if !regions.is_empty() && lcg_f32(&mut state.rng) < 0.03 * scale {
        events.push(WorldEvent::WeatherDamage {
            weather_type: WeatherType::Drought,
            description: "Drought is driving up supply costs".into(),
        });
    }
}

/// Fog: -50% scouting visibility, ambush chance +20%.
fn apply_fog(
    state: &mut CampaignState,
    _events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    // Reduce visibility in affected regions.
    for &rid in regions {
        if let Some(region) = state.overworld.regions.get_mut(rid) {
            let vis_penalty = 0.50 * scale * 0.01; // per-tick fraction
            region.visibility = (region.visibility - vis_penalty).clamp(0.0, 1.0);
        }
    }

    // Increase threat level in foggy regions (proxy for ambush chance).
    for &rid in regions {
        if let Some(region) = state.overworld.regions.get_mut(rid) {
            region.threat_level = (region.threat_level + 0.20 * scale * 0.1).clamp(0.0, 100.0);
        }
    }
}

/// Heatwave: +fatigue for traveling parties, -10% combat effectiveness.
fn apply_heatwave(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    regions: &[usize],
    scale: f32,
) {
    for party in &mut state.parties {
        if party.status == PartyStatus::Idle {
            continue;
        }
        if party_in_regions(party, &state.overworld.regions, regions) {
            // Increase fatigue for party members.
            for &mid in &party.member_ids {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                    adv.fatigue = (adv.fatigue + 2.0 * scale * 0.01).clamp(0.0, 100.0);
                }
            }

            // Reduce party morale (proxy for combat effectiveness).
            party.morale = (party.morale - 0.10 * scale * 0.1).max(0.0);
        }
    }

    if !regions.is_empty() && lcg_f32(&mut state.rng) < 0.03 * scale {
        events.push(WorldEvent::WeatherDamage {
            weather_type: WeatherType::Heatwave,
            description: "Heatwave is exhausting traveling parties".into(),
        });
    }
}

// ---------------------------------------------------------------------------
// Spawning new weather
// ---------------------------------------------------------------------------

fn maybe_spawn_weather(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let season = state.overworld.season;
    let probs = weather_probabilities(season);
    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return;
    }

    for &(wt, prob) in probs {
        let roll = lcg_f32(&mut state.rng);
        if roll >= prob {
            continue;
        }

        // Don't stack same weather type.
        if state
            .overworld
            .active_weather
            .iter()
            .any(|w| w.weather_type == wt)
        {
            continue;
        }

        // Pick 1–3 affected regions.
        let count = 1 + (lcg_next(&mut state.rng) as usize % 3).min(num_regions);
        let mut affected = Vec::with_capacity(count);
        for _ in 0..count {
            let rid = lcg_next(&mut state.rng) as usize % num_regions;
            if !affected.contains(&rid) {
                affected.push(rid);
            }
        }

        // Severity 20–100.
        let severity = 20.0 + lcg_f32(&mut state.rng) * 80.0;

        // Duration.
        let (min_dur, max_dur) = duration_range(wt);
        let dur_range = max_dur - min_dur;
        let duration = min_dur + (lcg_next(&mut state.rng) as u64 % (dur_range + 1));

        let id = state.overworld.next_weather_id;
        state.overworld.next_weather_id += 1;

        events.push(WorldEvent::WeatherStarted {
            weather_type: wt,
            regions: affected.clone(),
            severity,
        });

        state.overworld.active_weather.push(WeatherEvent {
            id,
            weather_type: wt,
            affected_regions: affected,
            severity,
            started_tick: state.tick,
            duration,
        });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a party's current position falls within any of the given region indices.
/// Uses a simple spatial heuristic: region index maps to a grid cell.
fn party_in_regions(party: &Party, regions: &[Region], affected: &[usize]) -> bool {
    if regions.is_empty() || affected.is_empty() {
        return false;
    }
    // Simple heuristic: divide world into grid cells matching region count.
    // Each region covers a strip of the world.
    let num_regions = regions.len();
    let region_width = 100.0 / num_regions as f32;
    let party_region = ((party.position.0 / region_width) as usize).min(num_regions - 1);
    affected.contains(&party_region)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn test_state_with_regions(num_regions: usize, season: Season) -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.overworld.season = season;
        state.overworld.regions = (0..num_regions)
            .map(|i| Region {
                id: i,
                name: format!("Region {}", i),
                owner_faction_id: 0,
                neighbors: vec![],
                unrest: 10.0,
                control: 80.0,
                threat_level: 20.0,
                visibility: 0.5,
                population: 500,
                civilian_morale: 50.0,
                tax_rate: 0.1,
                growth_rate: 0.0,
            })
            .collect();
        state
    }

    #[test]
    fn weather_does_not_fire_at_tick_zero() {
        let mut state = test_state_with_regions(5, Season::Winter);
        state.tick = 0;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_weather(&mut state, &mut deltas, &mut events);
        assert!(state.overworld.active_weather.is_empty());
    }

    #[test]
    fn weather_fires_at_interval() {
        let mut state = test_state_with_regions(5, Season::Winter);
        state.tick = WEATHER_INTERVAL;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        // Run many ticks to get at least one weather event (probabilistic but
        // deterministic with the seed).
        for i in 1..=50 {
            state.tick = WEATHER_INTERVAL * i;
            tick_weather(&mut state, &mut deltas, &mut events);
        }
        // With 50 rolls at 20%+10% chance, we should have spawned something.
        assert!(
            !events.is_empty(),
            "Expected at least one weather event after 50 intervals"
        );
    }

    #[test]
    fn weather_expires() {
        let mut state = test_state_with_regions(3, Season::Winter);
        state.overworld.active_weather.push(WeatherEvent {
            id: 0,
            weather_type: WeatherType::Storm,
            affected_regions: vec![0],
            severity: 50.0,
            started_tick: 100,
            duration: 200,
        });
        state.tick = 400; // past started_tick + duration
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_weather(&mut state, &mut deltas, &mut events);
        assert!(state.overworld.active_weather.is_empty());
        assert!(events.iter().any(|e| matches!(e, WorldEvent::WeatherEnded { .. })));
    }

    #[test]
    fn no_duplicate_weather_type() {
        let mut state = test_state_with_regions(5, Season::Winter);
        // Pre-seed a Blizzard so the spawner shouldn't add another.
        state.overworld.active_weather.push(WeatherEvent {
            id: 0,
            weather_type: WeatherType::Blizzard,
            affected_regions: vec![0],
            severity: 80.0,
            started_tick: 0,
            duration: 999_999, // Won't expire during the test
        });
        state.overworld.next_weather_id = 1;
        for i in 1..=20 {
            state.tick = WEATHER_INTERVAL * i;
            let mut deltas = StepDeltas::default();
            let mut events = Vec::new();
            tick_weather(&mut state, &mut deltas, &mut events);
        }
        let blizzard_count = state
            .overworld
            .active_weather
            .iter()
            .filter(|w| w.weather_type == WeatherType::Blizzard)
            .count();
        assert_eq!(blizzard_count, 1);
    }

    #[test]
    fn severity_scales_effects() {
        // At severity 0 (min clamp is 20, but let's test with a manual value),
        // effects should be minimal.
        let mut state = test_state_with_regions(3, Season::Winter);
        state.overworld.active_weather.push(WeatherEvent {
            id: 0,
            weather_type: WeatherType::Drought,
            affected_regions: vec![0, 1],
            severity: 1.0, // Very low severity
            started_tick: 100,
            duration: 10000,
        });
        let initial_multiplier = state.guild.market_prices.supply_multiplier;
        state.tick = WEATHER_INTERVAL;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_weather(&mut state, &mut deltas, &mut events);
        // Supply multiplier should barely change at severity 1.
        let diff = (state.guild.market_prices.supply_multiplier - initial_multiplier).abs();
        assert!(diff < 0.01, "Low severity should cause minimal effect, got diff={}", diff);
    }
}
