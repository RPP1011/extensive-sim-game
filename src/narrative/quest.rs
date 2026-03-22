//! Quest tracking system — create, update objectives, and complete quests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Quest types
// ---------------------------------------------------------------------------

/// State of a quest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestState {
    Active,
    Completed,
    Failed,
}

/// A single quest objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestObjective {
    pub description: String,
    pub completed: bool,
}

/// Reward for completing a quest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestReward {
    #[serde(default)]
    pub supply: u32,
    #[serde(default)]
    pub relation_faction_id: Option<usize>,
    #[serde(default)]
    pub relation_change: i32,
    #[serde(default)]
    pub description: String,
}

impl Default for QuestReward {
    fn default() -> Self {
        Self {
            supply: 0,
            relation_faction_id: None,
            relation_change: 0,
            description: String::new(),
        }
    }
}

/// A quest with objectives and rewards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u32,
    pub title: String,
    pub description: String,
    pub objectives: Vec<QuestObjective>,
    pub state: QuestState,
    pub reward: QuestReward,
    pub source_trigger_tag: String,
}

/// Update notification for quest state changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestUpdate {
    pub quest_id: u32,
    pub new_state: QuestState,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Quest tracker
// ---------------------------------------------------------------------------

/// Tracks all active and completed quests.
#[derive(Debug, Clone, Default)]
pub struct QuestTracker {
    pub quests: HashMap<u32, Quest>,
    pub next_id: u32,
}

impl QuestTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new quest and return its ID.
    pub fn create_quest(
        &mut self,
        title: String,
        description: String,
        objectives: Vec<String>,
        reward: QuestReward,
        source_trigger_tag: String,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let quest = Quest {
            id,
            title,
            description,
            objectives: objectives
                .into_iter()
                .map(|desc| QuestObjective {
                    description: desc,
                    completed: false,
                })
                .collect(),
            state: QuestState::Active,
            reward,
            source_trigger_tag,
        };
        self.quests.insert(id, quest);
        id
    }

    /// Complete a specific objective of a quest.
    pub fn complete_objective(&mut self, quest_id: u32, objective_idx: usize) -> Option<QuestUpdate> {
        let quest = self.quests.get_mut(&quest_id)?;
        if quest.state != QuestState::Active {
            return None;
        }

        if let Some(obj) = quest.objectives.get_mut(objective_idx) {
            obj.completed = true;
        }

        // Check if all objectives are complete
        let all_complete = quest.objectives.iter().all(|o| o.completed);
        if all_complete {
            quest.state = QuestState::Completed;
            Some(QuestUpdate {
                quest_id,
                new_state: QuestState::Completed,
                message: format!("Quest completed: {}", quest.title),
            })
        } else {
            let completed_count = quest.objectives.iter().filter(|o| o.completed).count();
            Some(QuestUpdate {
                quest_id,
                new_state: QuestState::Active,
                message: format!(
                    "Quest progress: {} ({}/{})",
                    quest.title,
                    completed_count,
                    quest.objectives.len()
                ),
            })
        }
    }

    /// Fail a quest.
    pub fn fail_quest(&mut self, quest_id: u32) -> Option<QuestUpdate> {
        let quest = self.quests.get_mut(&quest_id)?;
        quest.state = QuestState::Failed;
        Some(QuestUpdate {
            quest_id,
            new_state: QuestState::Failed,
            message: format!("Quest failed: {}", quest.title),
        })
    }

    /// Get all active quests.
    pub fn active_quests(&self) -> Vec<&Quest> {
        self.quests
            .values()
            .filter(|q| q.state == QuestState::Active)
            .collect()
    }

    /// Get a quest by ID.
    pub fn get(&self, quest_id: u32) -> Option<&Quest> {
        self.quests.get(&quest_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quest_creation() {
        let mut tracker = QuestTracker::new();
        let id = tracker.create_quest(
            "Test Quest".to_string(),
            "A test quest.".to_string(),
            vec!["Objective 1".to_string(), "Objective 2".to_string()],
            QuestReward::default(),
            "test_trigger".to_string(),
        );
        assert_eq!(id, 0);
        assert_eq!(tracker.active_quests().len(), 1);
    }

    #[test]
    fn test_objective_completion_advances_quest() {
        let mut tracker = QuestTracker::new();
        let id = tracker.create_quest(
            "Test".to_string(),
            "Desc".to_string(),
            vec!["A".to_string(), "B".to_string()],
            QuestReward::default(),
            "tag".to_string(),
        );

        // Complete first objective
        let update = tracker.complete_objective(id, 0).unwrap();
        assert_eq!(update.new_state, QuestState::Active);
        assert!(update.message.contains("1/2"));

        // Complete second objective → quest completes
        let update = tracker.complete_objective(id, 1).unwrap();
        assert_eq!(update.new_state, QuestState::Completed);
        assert!(update.message.contains("completed"));

        // No more active quests
        assert_eq!(tracker.active_quests().len(), 0);
    }

    #[test]
    fn test_quest_failure() {
        let mut tracker = QuestTracker::new();
        let id = tracker.create_quest(
            "Test".to_string(),
            "Desc".to_string(),
            vec!["A".to_string()],
            QuestReward::default(),
            "tag".to_string(),
        );

        let update = tracker.fail_quest(id).unwrap();
        assert_eq!(update.new_state, QuestState::Failed);
        assert_eq!(tracker.active_quests().len(), 0);
    }
}
