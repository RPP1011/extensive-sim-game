//! NPC biography generator — pure read-only function that produces a
//! multi-paragraph human-readable biography from an NPC's memory,
//! personality, and life events.
//!
//! This is NOT a per-tick system. It is called on demand (e.g., from the UI
//! or CLI) to render a biography for a specific NPC entity.

use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::{tag, Entity, EntityKind, MemEventType, WorldState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a tick to (year, season_name). One year = 4800 ticks, one season = 1200 ticks.
fn tick_to_date(tick: u64) -> (u64, &'static str) {
    let year = tick / 4800;
    let season_idx = (tick / 1200) % 4;
    let season = match season_idx {
        0 => "Spring",
        1 => "Summer",
        2 => "Autumn",
        _ => "Winter",
    };
    (year, season)
}

/// Map a behavior tag hash to a human-readable display name.
fn tag_display_name(tag_hash: u32) -> &'static str {
    // We use a match against compile-time hashes so this stays in sync with the
    // tag constants defined in state::tags.
    match tag_hash {
        x if x == tag(b"melee") => "Melee Combat",
        x if x == tag(b"ranged") => "Ranged Combat",
        x if x == tag(b"combat") => "Combat",
        x if x == tag(b"defense") => "Defense",
        x if x == tag(b"tactics") => "Tactics",
        x if x == tag(b"mining") => "Mining",
        x if x == tag(b"smithing") => "Smithing",
        x if x == tag(b"crafting") => "Crafting",
        x if x == tag(b"enchantment") => "Enchantment",
        x if x == tag(b"alchemy") => "Alchemy",
        x if x == tag(b"trade") => "Trade",
        x if x == tag(b"diplomacy") => "Diplomacy",
        x if x == tag(b"leadership") => "Leadership",
        x if x == tag(b"negotiation") => "Negotiation",
        x if x == tag(b"deception") => "Deception",
        x if x == tag(b"research") => "Research",
        x if x == tag(b"lore") => "Lore",
        x if x == tag(b"medicine") => "Medicine",
        x if x == tag(b"herbalism") => "Herbalism",
        x if x == tag(b"navigation") => "Navigation",
        x if x == tag(b"endurance") => "Endurance",
        x if x == tag(b"resilience") => "Resilience",
        x if x == tag(b"stealth") => "Stealth",
        x if x == tag(b"survival") => "Survival",
        x if x == tag(b"awareness") => "Awareness",
        x if x == tag(b"faith") => "Faith",
        x if x == tag(b"ritual") => "Ritual",
        x if x == tag(b"labor") => "Labor",
        x if x == tag(b"teaching") => "Teaching",
        x if x == tag(b"discipline") => "Discipline",
        x if x == tag(b"construction") => "Construction",
        x if x == tag(b"architecture") => "Architecture",
        x if x == tag(b"masonry") => "Masonry",
        x if x == tag(b"farming") => "Farming",
        x if x == tag(b"woodwork") => "Woodwork",
        x if x == tag(b"exploration") => "Exploration",
        x if x == tag(b"compassion") => "Compassion",
        x if x == tag(b"seafaring") => "Seafaring",
        x if x == tag(b"dungeoneering") => "Dungeoneering",
        _ => "Unknown",
    }
}

/// Map a mood u8 to a prose description. Mirrors the Mood enum in moods.rs.
fn mood_description(mood: u8) -> &'static str {
    match mood {
        0 => "feeling composed",
        1 => "feeling excited",
        2 => "feeling inspired",
        3 => "feeling angry",
        4 => "feeling fearful",
        5 => "grieving",
        6 => "feeling melancholic",
        7 => "feeling determined",
        _ => "in an indescribable mood",
    }
}

// ---------------------------------------------------------------------------
// Entity name resolution helper
// ---------------------------------------------------------------------------

/// Resolve an entity ID to a display name, falling back to "Entity #ID".
fn resolve_name(id: u32, state: &WorldState) -> String {
    state
        .entity(id)
        .map(entity_display_name)
        .unwrap_or_else(|| format!("Entity #{}", id))
}

/// Resolve a settlement ID to its name, falling back to "an unknown settlement".
fn resolve_settlement_name(id: u32, state: &WorldState) -> String {
    state
        .settlement(id)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "an unknown settlement".to_string())
}

// ---------------------------------------------------------------------------
// Biography generator
// ---------------------------------------------------------------------------

/// Generate a multi-paragraph human-readable biography for an NPC entity.
///
/// Returns an empty string if the entity is not an NPC.
pub fn generate_biography(entity: &Entity, state: &WorldState) -> String {
    if entity.kind != EntityKind::Npc {
        return String::new();
    }

    let npc = match entity.npc.as_ref() {
        Some(n) => n,
        None => return String::new(),
    };

    let name = entity_display_name(entity);
    let mut bio = String::with_capacity(1024);

    // --- Opening ---
    {
        let (year, season) = tick_to_date(npc.born_tick);
        let birthplace = npc
            .home_settlement_id
            .map(|sid| resolve_settlement_name(sid, state))
            .unwrap_or_else(|| "parts unknown".to_string());

        bio.push_str(&format!(
            "{} was born in {} during {} of Year {}.",
            name, birthplace, season, year
        ));

        if !npc.archetype.is_empty() {
            let article = if matches!(
                npc.archetype.as_bytes().first(),
                Some(b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U')
            ) {
                "An"
            } else {
                "A"
            };
            bio.push_str(&format!(" {} {} by trade.", article, npc.archetype));
        }

        bio.push('\n');
    }

    // --- Career ---
    {
        bio.push('\n');
        if !npc.classes.is_empty() {
            let class_names: Vec<&str> = npc
                .classes
                .iter()
                .map(|c| {
                    if c.display_name.is_empty() {
                        "an unnamed class"
                    } else {
                        c.display_name.as_str()
                    }
                })
                .collect();
            bio.push_str(&format!("Trained as {}.", class_names.join(", ")));
        }

        // Mentor lineage
        if !npc.mentor_lineage.is_empty() {
            let mentor_names: Vec<String> = npc
                .mentor_lineage
                .iter()
                .map(|&mid| resolve_name(mid, state))
                .collect();
            bio.push_str(&format!(" Learned under {}.", mentor_names.join(", then ")));
        }

        // Top 3 behavior tags
        if !npc.behavior_profile.is_empty() {
            let mut sorted: Vec<(u32, f32)> = npc.behavior_profile.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<&str> = sorted
                .iter()
                .take(3)
                .map(|&(hash, _)| tag_display_name(hash))
                .filter(|&n| n != "Unknown")
                .collect();
            if !top.is_empty() {
                bio.push_str(&format!(" Known for {}.", top.join(", ")));
            }
        }

        bio.push('\n');
    }

    // --- Life events ---
    if !npc.memory.events.is_empty() {
        bio.push('\n');
        for event in &npc.memory.events {
            let (year, season) = tick_to_date(event.tick);
            let desc = match &event.event_type {
                MemEventType::WasAttacked => "Survived an attack".to_string(),
                MemEventType::AttackedEnemy => "Fought against a foe".to_string(),
                MemEventType::WonFight => "Emerged victorious in battle".to_string(),
                MemEventType::FriendDied(id) => {
                    format!("Mourned the loss of {}", resolve_name(*id, state))
                }
                MemEventType::MadeNewFriend(id) => {
                    format!("Befriended {}", resolve_name(*id, state))
                }
                MemEventType::TradedWith(id) => {
                    format!(
                        "Established trade with {}",
                        resolve_settlement_name(*id, state)
                    )
                }
                MemEventType::CompletedQuest => "Completed a quest".to_string(),
                MemEventType::LearnedSkill => "Mastered a new skill".to_string(),
                MemEventType::WasHealed => "Was nursed back to health".to_string(),
                MemEventType::WasBetrayedBy(id) => {
                    format!("Was betrayed by {}", resolve_name(*id, state))
                }
                MemEventType::BuiltSomething => "Helped build a new structure".to_string(),
                MemEventType::LostHome => "Lost their home".to_string(),
                MemEventType::WasRescuedBy(id) => {
                    format!("Was rescued by {}", resolve_name(*id, state))
                }
                MemEventType::Starved => "Endured famine".to_string(),
                MemEventType::FoundShelter => "Found shelter in dire times".to_string(),
                MemEventType::BecameApprentice(id) => {
                    format!("Became apprentice to {}", resolve_name(*id, state))
                }
                MemEventType::CompletedApprenticeship(id) => {
                    format!("Completed apprenticeship under {}", resolve_name(*id, state))
                }
                MemEventType::TrainedApprentice(id) => {
                    format!("Took on {} as an apprentice", resolve_name(*id, state))
                }
            };
            bio.push_str(&format!("In {} of Year {}, {}.\n", season, year, desc));
        }
    }

    // --- Personality ---
    {
        let personality = &npc.personality;
        let mut traits = Vec::new();
        if personality.risk_tolerance > 0.7 {
            traits.push("Bold");
        }
        if personality.compassion > 0.7 {
            traits.push("Compassionate");
        }
        if personality.ambition > 0.7 {
            traits.push("Ambitious");
        }
        if personality.social_drive > 0.7 {
            traits.push("Gregarious");
        }
        if personality.curiosity > 0.7 {
            traits.push("Curious");
        }

        // Also note low values as personality color
        if personality.risk_tolerance < 0.3 {
            traits.push("Cautious");
        }
        if personality.compassion < 0.3 {
            traits.push("Cold-Hearted");
        }
        if personality.ambition < 0.3 {
            traits.push("Content");
        }
        if personality.social_drive < 0.3 {
            traits.push("Reclusive");
        }
        if personality.curiosity < 0.3 {
            traits.push("Set in Their Ways");
        }

        if !traits.is_empty() {
            bio.push_str(&format!(
                "\nThose who know {} describe them as {}.\n",
                name,
                traits.join(", ")
            ));
        }
    }

    // --- Current state ---
    {
        bio.push('\n');
        let class_label = npc
            .classes
            .first()
            .and_then(|c| {
                if c.display_name.is_empty() {
                    None
                } else {
                    Some(c.display_name.as_str())
                }
            })
            .unwrap_or(&npc.archetype);

        let settlement_desc = npc
            .home_settlement_id
            .map(|sid| format!("living in {}", resolve_settlement_name(sid, state)))
            .unwrap_or_else(|| "wandering the land".to_string());

        let mood_desc = mood_description(npc.mood);

        bio.push_str(&format!(
            "Currently a level {} {}, {}. {}.\n",
            entity.level,
            class_label,
            settlement_desc,
            capitalize_first(mood_desc),
        ));
    }

    bio
}

/// Capitalize the first character of a string slice.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;
    use std::collections::VecDeque;

    /// Create a minimal NPC entity for biography testing.
    fn make_test_npc(id: u32) -> Entity {
        let mut entity = Entity::new_npc(id, (100.0, 200.0));
        entity.level = 5;
        let npc = entity.npc.as_mut().unwrap();
        npc.name = "Korrin".to_string();
        npc.archetype = "knight".to_string();
        npc.born_tick = 6000; // Year 1, Summer (6000/4800=1, (6000/1200)%4=1)
        npc.home_settlement_id = Some(10);

        // Add a class
        npc.classes.push(ClassSlot {
            class_name_hash: tag(b"warrior"),
            level: 3,
            xp: 0.0,
            display_name: "Iron-Fisted Warrior".to_string(),
        });

        // Mentor lineage (entity IDs that may or may not resolve)
        npc.mentor_lineage = vec![99];

        // Behavior profile — top tags
        npc.behavior_profile = vec![
            (tag(b"combat"), 150.0),
            (tag(b"defense"), 80.0),
            (tag(b"leadership"), 45.0),
        ];

        // Personality — bold and ambitious
        npc.personality = Personality {
            risk_tolerance: 0.85,
            compassion: 0.4,
            ambition: 0.9,
            social_drive: 0.5,
            curiosity: 0.5,
        };

        // Mood: Determined (7)
        npc.mood = 7;

        // Memory events
        let mut events = VecDeque::new();
        events.push_back(MemoryEvent {
            tick: 7200,
            event_type: MemEventType::WonFight,
            location: (100.0, 200.0),
            entity_ids: vec![],
            emotional_impact: 0.5,
        });
        events.push_back(MemoryEvent {
            tick: 9600,
            event_type: MemEventType::FriendDied(42),
            location: (100.0, 200.0),
            entity_ids: vec![42],
            emotional_impact: -0.8,
        });
        events.push_back(MemoryEvent {
            tick: 12000,
            event_type: MemEventType::CompletedQuest,
            location: (100.0, 200.0),
            entity_ids: vec![],
            emotional_impact: 0.6,
        });
        events.push_back(MemoryEvent {
            tick: 14400,
            event_type: MemEventType::Starved,
            location: (100.0, 200.0),
            entity_ids: vec![],
            emotional_impact: -0.7,
        });
        events.push_back(MemoryEvent {
            tick: 15600,
            event_type: MemEventType::MadeNewFriend(55),
            location: (100.0, 200.0),
            entity_ids: vec![55],
            emotional_impact: 0.4,
        });
        npc.memory.events = events;

        entity
    }

    #[test]
    fn biography_contains_expected_sections() {
        let mut state = WorldState::new(42);

        // Add settlement so birth location resolves
        state
            .settlements
            .push(SettlementState::new(10, "Ironhaven".into(), (0.0, 0.0)));

        // Add the NPC itself
        let npc_entity = make_test_npc(1);
        state.entities.push(npc_entity);

        // Add a friend entity so FriendDied and MadeNewFriend can resolve names
        let mut friend = Entity::new_npc(42, (50.0, 50.0));
        friend.npc.as_mut().unwrap().name = "Thessa".to_string();
        state.entities.push(friend);

        let mut new_friend = Entity::new_npc(55, (60.0, 60.0));
        new_friend.npc.as_mut().unwrap().name = "Bregan".to_string();
        state.entities.push(new_friend);

        // Rebuild the entity index for O(1) lookups
        state.rebuild_entity_cache();

        let entity = state.entity(1).unwrap();
        let bio = generate_biography(entity, &state);

        // Opening
        assert!(
            bio.contains("Korrin was born in Ironhaven during Summer of Year 1"),
            "Opening should mention name, settlement, season, year. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("knight by trade"),
            "Opening should mention archetype. Got:\n{}",
            bio
        );

        // Career
        assert!(
            bio.contains("Iron-Fisted Warrior"),
            "Career should mention class display name. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Known for Combat"),
            "Career should mention top behavior tag. Got:\n{}",
            bio
        );

        // Life events
        assert!(
            bio.contains("Emerged victorious in battle"),
            "Events should include WonFight. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Mourned the loss of Thessa"),
            "Events should resolve FriendDied name. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Completed a quest"),
            "Events should include CompletedQuest. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Endured famine"),
            "Events should include Starved. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Befriended Bregan"),
            "Events should resolve MadeNewFriend name. Got:\n{}",
            bio
        );

        // Personality
        assert!(
            bio.contains("Bold"),
            "Personality should note high risk_tolerance. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Ambitious"),
            "Personality should note high ambition. Got:\n{}",
            bio
        );

        // Current state
        assert!(
            bio.contains("level 5"),
            "Current state should mention level. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("living in Ironhaven"),
            "Current state should mention settlement. Got:\n{}",
            bio
        );
        assert!(
            bio.contains("Feeling determined"),
            "Current state should describe mood. Got:\n{}",
            bio
        );
    }

    #[test]
    fn biography_empty_for_non_npc() {
        let state = WorldState::new(42);
        let mut entity = Entity::new_npc(1, (0.0, 0.0));
        entity.kind = EntityKind::Monster;
        let bio = generate_biography(&entity, &state);
        assert!(bio.is_empty(), "Non-NPC should produce empty biography");
    }

    #[test]
    fn tick_to_date_conversion() {
        assert_eq!(tick_to_date(0), (0, "Spring"));
        assert_eq!(tick_to_date(1200), (0, "Summer"));
        assert_eq!(tick_to_date(2400), (0, "Autumn"));
        assert_eq!(tick_to_date(3600), (0, "Winter"));
        assert_eq!(tick_to_date(4800), (1, "Spring"));
        assert_eq!(tick_to_date(6000), (1, "Summer"));
    }

    #[test]
    fn tag_display_names_resolve() {
        assert_eq!(tag_display_name(tag(b"combat")), "Combat");
        assert_eq!(tag_display_name(tag(b"smithing")), "Smithing");
        assert_eq!(tag_display_name(tag(b"faith")), "Faith");
        assert_eq!(tag_display_name(0xDEAD), "Unknown");
    }
}
