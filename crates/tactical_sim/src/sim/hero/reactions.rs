use crate::effects::{
    AbilitySlot, Area, ConditionalEffect, DamageType, Effect, Stacking,
};

use super::super::types::*;
use super::super::events::SimEvent;

/// Apply ability morphing after an ability resolves.
pub fn apply_morph(caster_idx: usize, ability_index: usize, slot: &AbilitySlot, state: &mut SimState) {
    let Some(ability_slot) = state.units[caster_idx].abilities.get_mut(ability_index) else {
        return;
    };
    if ability_slot.base_def.is_some() {
        if let Some(base) = ability_slot.base_def.take() {
            ability_slot.def = *base;
            ability_slot.morph_remaining_ms = 0;
        }
    } else if let Some(ref morph_into) = slot.def.morph_into {
        ability_slot.base_def = Some(Box::new(slot.def.clone()));
        ability_slot.def = *morph_into.clone();
        if slot.def.morph_duration_ms > 0 {
            ability_slot.morph_remaining_ms = slot.def.morph_duration_ms;
        }
        ability_slot.cooldown_remaining_ms = ability_slot.def.cooldown_ms;
    }
}

/// Form swap: morph all abilities that have a matching `form` tag.
/// Each swapped ability toggles between its base and morph_into.
pub fn apply_form_swap(caster_idx: usize, form_tag: &str, state: &mut SimState) {
    for slot in &mut state.units[caster_idx].abilities {
        let matches_form = slot.def.form.as_deref() == Some(form_tag);
        if !matches_form {
            continue;
        }
        // Toggle: if already morphed, revert to base; otherwise morph into alternate
        if slot.base_def.is_some() {
            if let Some(base) = slot.base_def.take() {
                slot.def = *base;
                slot.morph_remaining_ms = 0;
            }
        } else if let Some(ref morph_into) = slot.def.morph_into.clone() {
            slot.base_def = Some(Box::new(slot.def.clone()));
            slot.def = *morph_into.clone();
        }
        slot.cooldown_remaining_ms = 0; // Reset cooldown on form swap
    }
}

// ---------------------------------------------------------------------------
// Zone reactions: overlapping tagged zones produce combo effects
// ---------------------------------------------------------------------------

fn zone_radius(area: &Area) -> f32 {
    match area {
        Area::Circle { radius } => *radius,
        Area::Cone { radius, .. } => *radius,
        Area::Spread { radius, .. } => *radius,
        _ => 2.0,
    }
}

/// Check if a newly placed tagged zone overlaps any existing tagged zone from
/// the same caster. If so, spawn a combo zone at the midpoint.
pub fn check_zone_reactions(
    state: &mut SimState,
    source_id: u32,
    source_team: Team,
    new_pos: SimVec2,
    new_tag: &str,
    tick: u64,
    events: &mut Vec<SimEvent>,
) {
    use super::super::math::distance;
    use std::collections::HashMap;

    // Find overlapping tagged zones from the same caster (exclude the one just placed)
    let mut reaction_partner: Option<(String, SimVec2, f32)> = None;
    for zone in state.zones.iter() {
        if zone.source_id != source_id {
            continue;
        }
        let Some(ref tag) = zone.zone_tag else { continue };
        if tag == new_tag {
            continue; // Same element doesn't react with itself
        }
        let r = zone_radius(&zone.area);
        let dist = distance(new_pos, zone.position);
        if dist <= r + 2.5 {
            // Zones overlap (within combined radius + tolerance)
            reaction_partner = Some((tag.clone(), zone.position, r));
            break;
        }
    }

    let Some((partner_tag, partner_pos, _partner_r)) = reaction_partner else {
        return;
    };

    // Sort tags alphabetically for consistent lookup
    let (tag_a, tag_b) = if new_tag < partner_tag.as_str() {
        (new_tag.to_string(), partner_tag.clone())
    } else {
        (partner_tag.clone(), new_tag.to_string())
    };

    // Combo zone spawns at the midpoint
    let combo_pos = SimVec2 {
        x: (new_pos.x + partner_pos.x) / 2.0,
        y: (new_pos.y + partner_pos.y) / 2.0,
    };

    // Look up reaction: (tag_a, tag_b) -> (combo_name, effects, radius, duration_ms, tick_interval_ms)
    let reaction = match (tag_a.as_str(), tag_b.as_str()) {
        // Fire + Frost = Steam Cloud (blind + damage)
        ("fire", "frost") => Some((
            "SteamCloud",
            vec![
                ConditionalEffect {
                    effect: Effect::Blind { miss_chance: 0.4, duration_ms: 3000 },
                    condition: None,
                    area: Some(Area::Circle { radius: 3.0 }),
                    tags: HashMap::new(),
                    stacking: Stacking::default(), chance: 0.0, else_effects: vec![], targeting_filter: None,
                },
                ConditionalEffect {
                    effect: Effect::Damage {
                        amount: 15, amount_per_tick: 0, tick_interval_ms: 0,
                        duration_ms: 0, scaling_stat: None, scaling_percent: 0.0,
                        damage_type: DamageType::Magic,
                        bonus: vec![],
                    },
                    condition: None,
                    area: Some(Area::Circle { radius: 3.0 }),
                    tags: HashMap::new(),
                    stacking: Stacking::default(), chance: 0.0, else_effects: vec![], targeting_filter: None,
                },
            ],
            3.0, 4000, 500,
        )),
        // Fire + Lightning = Plasma Storm (high damage, big radius)
        ("fire", "lightning") => Some((
            "PlasmaStorm",
            vec![
                ConditionalEffect {
                    effect: Effect::Damage {
                        amount: 30, amount_per_tick: 0, tick_interval_ms: 0,
                        duration_ms: 0, scaling_stat: None, scaling_percent: 0.0,
                        damage_type: DamageType::Magic,
                        bonus: vec![],
                    },
                    condition: None,
                    area: Some(Area::Circle { radius: 3.5 }),
                    tags: HashMap::new(),
                    stacking: Stacking::default(), chance: 0.0, else_effects: vec![], targeting_filter: None,
                },
            ],
            3.5, 3000, 500,
        )),
        // Frost + Lightning = Shatter (burst damage + stun)
        ("frost", "lightning") => Some((
            "Shatter",
            vec![
                ConditionalEffect {
                    effect: Effect::Damage {
                        amount: 40, amount_per_tick: 0, tick_interval_ms: 0,
                        duration_ms: 0, scaling_stat: None, scaling_percent: 0.0,
                        damage_type: DamageType::Magic,
                        bonus: vec![],
                    },
                    condition: None,
                    area: Some(Area::Circle { radius: 2.5 }),
                    tags: HashMap::new(),
                    stacking: Stacking::default(), chance: 0.0, else_effects: vec![], targeting_filter: None,
                },
                ConditionalEffect {
                    effect: Effect::Stun { duration_ms: 1500 },
                    condition: None,
                    area: Some(Area::Circle { radius: 2.5 }),
                    tags: HashMap::new(),
                    stacking: Stacking::default(), chance: 0.0, else_effects: vec![], targeting_filter: None,
                },
            ],
            2.5, 1, 0, // duration=1ms, tick=0 → instant burst (applied once on creation by tick_zones)
        )),
        _ => None,
    };

    let Some((combo_name, effects, radius, duration_ms, tick_interval_ms)) = reaction else {
        return;
    };

    let combo_id = tick as u32 * 1000 + source_id + 500;
    state.zones.push(ActiveZone {
        id: combo_id,
        source_id,
        source_team,
        position: combo_pos,
        area: Area::Circle { radius },
        effects,
        remaining_ms: duration_ms,
        tick_interval_ms,
        tick_elapsed_ms: tick_interval_ms, // Fire immediately on next tick
        trigger_on_enter: false,
        invisible: false,
        triggered: false,
        arm_time_ms: 0,
        blocked_cells: Vec::new(),
        zone_tag: Some(format!("combo_{}", combo_name.to_lowercase())),
    });

    events.push(SimEvent::ZoneReaction {
        tick,
        source_id,
        tag_a,
        tag_b,
        combo_name: combo_name.to_string(),
    });
    events.push(SimEvent::ZoneCreated { tick, zone_id: combo_id, source_id });
}
