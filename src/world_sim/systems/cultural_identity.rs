#![allow(unused)]
//! Cultural identity — settlements develop emergent culture from NPC experiences.
//!
//! Every 500 ticks, aggregate the behavior profiles of all NPCs at a settlement.
//! The dominant tag cluster determines the settlement's cultural identity, which
//! provides passive bonuses and shapes chronicle flavor.
//!
//! Culture types (emergent, not assigned):
//! - Warrior Culture (combat+defense dominant) → +defense tags for residents
//! - Scholar Culture (research+lore dominant) → +research tags, class XP bonus
//! - Merchant Culture (trade+negotiation) → price bonuses, caravan funding
//! - Artisan Culture (crafting+smithing) → item quality bonus
//! - Farming Culture (farming+labor) → food production bonus
//! - Survivor Culture (resilience+survival+endurance) → morale resistance
//! - Seafaring Culture (seafaring+navigation) → sea travel speed bonus
//! - Storytelling Culture (teaching+diplomacy) → belief spread bonus
//!
//! Cadence: every 500 ticks.

use crate::world_sim::state::*;

const CULTURE_UPDATE_INTERVAL: u64 = 500;

/// A settlement's emergent cultural identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CultureType {
    Warrior,
    Scholar,
    Merchant,
    Artisan,
    Farming,
    Survivor,
    Seafaring,
    Storytelling,
    Undefined,
}

impl CultureType {
    pub fn name(self) -> &'static str {
        match self {
            CultureType::Warrior => "Warrior Culture",
            CultureType::Scholar => "Scholar Culture",
            CultureType::Merchant => "Merchant Culture",
            CultureType::Artisan => "Artisan Culture",
            CultureType::Farming => "Farming Culture",
            CultureType::Survivor => "Survivor Culture",
            CultureType::Seafaring => "Seafaring Culture",
            CultureType::Storytelling => "Storytelling Culture",
            CultureType::Undefined => "No distinct culture",
        }
    }
}

pub fn advance_cultural_identity(state: &mut WorldState) {
    if state.tick % CULTURE_UPDATE_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    for si in 0..state.settlements.len() {
        let sid = state.settlements[si].id;

        // Aggregate behavior profiles of all NPCs at this settlement.
        let mut tag_sums: Vec<(u32, f32)> = Vec::new();
        let mut npc_count = 0u32;

        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(sid) { continue; }

            npc_count += 1;
            for &(tag, value) in &npc.behavior_profile {
                if let Some(entry) = tag_sums.iter_mut().find(|(t, _)| *t == tag) {
                    entry.1 += value;
                } else {
                    tag_sums.push((tag, value));
                }
            }
        }

        if npc_count < 5 { continue; } // too small for distinct culture

        // Determine dominant culture from tag clusters.
        let score = |tag_hashes: &[u32]| -> f32 {
            tag_hashes.iter()
                .filter_map(|h| tag_sums.iter().find(|(t, _)| t == h))
                .map(|(_, v)| v / npc_count as f32)
                .sum()
        };

        let cultures = [
            (CultureType::Warrior, score(&[tags::COMBAT, tags::DEFENSE, tags::MELEE, tags::TACTICS])),
            (CultureType::Scholar, score(&[tags::RESEARCH, tags::LORE, tags::DISCIPLINE])),
            (CultureType::Merchant, score(&[tags::TRADE, tags::NEGOTIATION, tags::DIPLOMACY])),
            (CultureType::Artisan, score(&[tags::CRAFTING, tags::SMITHING, tags::MASONRY])),
            (CultureType::Farming, score(&[tags::FARMING, tags::LABOR, tags::WOODWORK])),
            (CultureType::Survivor, score(&[tags::RESILIENCE, tags::SURVIVAL, tags::ENDURANCE])),
            (CultureType::Seafaring, score(&[tags::SEAFARING, tags::NAVIGATION])),
            (CultureType::Storytelling, score(&[tags::TEACHING, tags::DIPLOMACY, tags::LEADERSHIP])),
        ];

        let (best_culture, best_score) = cultures.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if *best_score < 1.0 { continue; } // no clear culture

        // Apply cultural bonuses to settlement NPCs.
        let bonus_tag = match best_culture {
            CultureType::Warrior => tags::DEFENSE,
            CultureType::Scholar => tags::RESEARCH,
            CultureType::Merchant => tags::TRADE,
            CultureType::Artisan => tags::CRAFTING,
            CultureType::Farming => tags::FARMING,
            CultureType::Survivor => tags::RESILIENCE,
            CultureType::Seafaring => tags::SEAFARING,
            CultureType::Storytelling => tags::TEACHING,
            CultureType::Undefined => continue,
        };

        // Apply small cultural bonus to all residents (reinforcing feedback loop).
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(sid) { continue; }

            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(bonus_tag, 0.5); // gentle cultural reinforcement
                a
            });
        }

        // Chronicle first time a culture solidifies (or changes).
        // Use a simple flag: only chronicle if settlement context_tags don't already have this.
        let culture_tag_hash = crate::world_sim::state::tag(best_culture.name().as_bytes());
        let already_has = state.settlements[si].context_tags.iter()
            .any(|(h, _)| *h == culture_tag_hash);

        if !already_has {
            // Clear old culture tags and set new one.
            state.settlements[si].context_tags.retain(|(h, _)| {
                // Remove previous culture entries.
                let culture_names: Vec<u32> = [
                    CultureType::Warrior, CultureType::Scholar, CultureType::Merchant,
                    CultureType::Artisan, CultureType::Farming, CultureType::Survivor,
                    CultureType::Seafaring, CultureType::Storytelling,
                ].iter().map(|c| crate::world_sim::state::tag(c.name().as_bytes())).collect();
                !culture_names.contains(h)
            });
            state.settlements[si].context_tags.push((culture_tag_hash, 1.0));

            let settlement_name = state.settlements[si].name.clone();
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Achievement,
                text: format!("{} has developed a {}. Its people are shaped by their shared experiences.",
                    settlement_name, best_culture.name()),
                entity_ids: vec![],
            });
        }
    }
}
