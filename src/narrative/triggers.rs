//! Narrative trigger definitions and matching logic.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Trigger kinds
// ---------------------------------------------------------------------------

/// Battle outcome for post-battle triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BattleOutcome {
    Victory,
    Defeat,
    CloseFight,
}

/// Condition kinds for hero-specific triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeroConditionKind {
    LowHp,
    HighStress,
    LowLoyalty,
    Injured,
    Promoted,
}

/// The kind of event that can fire a narrative trigger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NarrativeTriggerKind {
    /// After a battle completes.
    PostBattle { outcome: BattleOutcome },
    /// When the player enters a region.
    RegionEnter { region_id: usize },
    /// When a faction performs a strategic action.
    FactionEvent { faction_id: usize, action: String },
    /// When a specific campaign turn is reached.
    TurnThreshold { turn: u32 },
    /// When a hero meets a condition.
    HeroCondition { hero_id: u32, condition: HeroConditionKind },
    /// When a quest updates.
    QuestUpdate { quest_id: u32 },
}

impl NarrativeTriggerKind {
    /// Check if this trigger kind matches an incoming event.
    pub fn matches(&self, event: &NarrativeTriggerKind) -> bool {
        match (self, event) {
            (
                NarrativeTriggerKind::PostBattle { outcome: a },
                NarrativeTriggerKind::PostBattle { outcome: b },
            ) => a == b,
            (
                NarrativeTriggerKind::RegionEnter { region_id: a },
                NarrativeTriggerKind::RegionEnter { region_id: b },
            ) => a == b,
            (
                NarrativeTriggerKind::FactionEvent { faction_id: a, action: aa },
                NarrativeTriggerKind::FactionEvent { faction_id: b, action: ba },
            ) => a == b && aa == ba,
            (
                NarrativeTriggerKind::TurnThreshold { turn: a },
                NarrativeTriggerKind::TurnThreshold { turn: b },
            ) => b >= a, // Fires when current turn >= threshold
            (
                NarrativeTriggerKind::HeroCondition { hero_id: a, condition: ac },
                NarrativeTriggerKind::HeroCondition { hero_id: b, condition: bc },
            ) => a == b && ac == bc,
            (
                NarrativeTriggerKind::QuestUpdate { quest_id: a },
                NarrativeTriggerKind::QuestUpdate { quest_id: b },
            ) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger definition
// ---------------------------------------------------------------------------

/// A registered narrative trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeTrigger {
    /// Unique tag for cooldown tracking.
    pub tag: String,
    /// The kind of event that activates this trigger.
    pub kind: NarrativeTriggerKind,
    /// Priority (higher = evaluated first).
    pub priority: u8,
    /// Minimum turns between consecutive firings.
    pub cooldown_turns: u32,
    /// Key into the template registry for generating output.
    pub template_key: String,
}

// ---------------------------------------------------------------------------
// Trigger registry
// ---------------------------------------------------------------------------

/// Registry of narrative triggers.
#[derive(Debug, Clone, Default)]
pub struct TriggerRegistry {
    triggers: Vec<NarrativeTrigger>,
}

impl TriggerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new trigger.
    pub fn register(&mut self, trigger: NarrativeTrigger) {
        self.triggers.push(trigger);
        // Keep sorted by priority (highest first)
        self.triggers.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Find all triggers that match a given event kind.
    pub fn find_matching(&self, event: &NarrativeTriggerKind) -> Vec<&NarrativeTrigger> {
        self.triggers
            .iter()
            .filter(|t| t.kind.matches(event))
            .collect()
    }

    /// Number of registered triggers.
    pub fn len(&self) -> usize {
        self.triggers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triggers.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_matching() {
        let trigger = NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Victory,
        };
        let event = NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Victory,
        };
        assert!(trigger.matches(&event));

        let different = NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Defeat,
        };
        assert!(!trigger.matches(&different));
    }

    #[test]
    fn test_turn_threshold_matching() {
        let trigger = NarrativeTriggerKind::TurnThreshold { turn: 10 };
        let event_before = NarrativeTriggerKind::TurnThreshold { turn: 5 };
        let event_after = NarrativeTriggerKind::TurnThreshold { turn: 15 };

        assert!(!trigger.matches(&event_before));
        assert!(trigger.matches(&event_after));
    }

    #[test]
    fn test_registry_priority_sorting() {
        let mut registry = TriggerRegistry::new();
        registry.register(NarrativeTrigger {
            tag: "low".to_string(),
            kind: NarrativeTriggerKind::PostBattle {
                outcome: BattleOutcome::Victory,
            },
            priority: 1,
            cooldown_turns: 0,
            template_key: "low_template".to_string(),
        });
        registry.register(NarrativeTrigger {
            tag: "high".to_string(),
            kind: NarrativeTriggerKind::PostBattle {
                outcome: BattleOutcome::Victory,
            },
            priority: 10,
            cooldown_turns: 0,
            template_key: "high_template".to_string(),
        });

        let event = NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Victory,
        };
        let matches = registry.find_matching(&event);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].tag, "high"); // Higher priority first
    }
}
