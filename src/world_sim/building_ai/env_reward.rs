//! Objective scores and Chebyshev scalarization for BuildingEnv reward.

use crate::world_sim::state::{EntityKind, WorldState};
use super::types::ChallengeCategory;

/// Snapshot of objective scores at a decision point.
#[derive(Debug, Clone, Default)]
pub struct ObjectiveScores {
    pub defense: f32,
    pub economy: f32,
    pub population: f32,
    pub connectivity: f32,
    pub garrison: f32,
    pub spatial: f32,
}

impl ObjectiveScores {
    /// Snapshot current objective scores from world state.
    pub fn snapshot(state: &WorldState) -> Self {
        let total_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc)
            .count() as f32;
        let alive_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive)
            .count() as f32;
        let defense = if total_npcs > 0.0 { alive_npcs / total_npcs } else { 1.0 };

        let economy = if state.settlements.is_empty() {
            0.5
        } else {
            let total: f32 = state.settlements.iter()
                .map(|s| {
                    let sum: f32 = s.stockpile.iter().sum();
                    let cap = s.stockpile.len() as f32 * 100.0;
                    (sum / cap).min(1.0)
                })
                .sum();
            total / state.settlements.len() as f32
        };

        let housing_cap: f32 = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter_map(|e| e.building.as_ref())
            .map(|b| b.residential_capacity as f32)
            .sum();
        let population = if housing_cap > 0.0 { (alive_npcs / housing_cap).min(1.0) } else { 0.0 };

        let total_buildings = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .count() as f32;
        let connectivity = if total_buildings > 0.0 { 1.0 } else { 0.0 };

        let combat_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive && e.level >= 2)
            .count() as f32;
        let garrison = if alive_npcs > 0.0 { (combat_npcs / alive_npcs).min(1.0) } else { 0.0 };

        let spatial = (total_buildings / 50.0).min(1.0);

        Self { defense, economy, population, connectivity, garrison, spatial }
    }

    fn as_array(&self) -> [f32; 6] {
        [self.defense, self.economy, self.population, self.connectivity, self.garrison, self.spatial]
    }
}

/// Challenge-category-dependent objective weights.
pub fn category_weights(category: ChallengeCategory) -> [f32; 6] {
    match category {
        ChallengeCategory::Military      => [0.40, 0.05, 0.10, 0.15, 0.20, 0.10],
        ChallengeCategory::Environmental => [0.15, 0.10, 0.15, 0.20, 0.05, 0.35],
        ChallengeCategory::Economic      => [0.05, 0.40, 0.10, 0.25, 0.05, 0.15],
        ChallengeCategory::Population    => [0.05, 0.10, 0.40, 0.20, 0.05, 0.20],
        _                                => [0.20, 0.15, 0.15, 0.20, 0.15, 0.15],
    }
}

/// Chebyshev scalarization: -max_i(w_i * |R_i - R*_i| / R*_i).
pub fn chebyshev_score(scores: &ObjectiveScores, ideal: &ObjectiveScores, weights: &[f32; 6]) -> f32 {
    let s = scores.as_array();
    let r = ideal.as_array();
    let mut worst = 0.0f32;
    for i in 0..6 {
        let ri = r[i].max(0.01);
        let deviation = weights[i] * (s[i] - ri).abs() / ri;
        worst = worst.max(deviation);
    }
    -worst
}

/// Per-step reward as Chebyshev delta between pre and post decision scores.
pub fn compute_reward(
    pre: &ObjectiveScores,
    post: &ObjectiveScores,
    ideal: &ObjectiveScores,
    category: ChallengeCategory,
) -> f32 {
    let w = category_weights(category);
    chebyshev_score(post, ideal, &w) - chebyshev_score(pre, ideal, &w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_scores_zero_reward() {
        let scores = ObjectiveScores {
            defense: 0.8, economy: 0.6, population: 0.7,
            connectivity: 0.9, garrison: 0.5, spatial: 0.4,
        };
        let ideal = ObjectiveScores {
            defense: 1.0, economy: 1.0, population: 1.0,
            connectivity: 1.0, garrison: 1.0, spatial: 1.0,
        };
        let reward = compute_reward(&scores, &scores, &ideal, ChallengeCategory::Military);
        assert!((reward).abs() < 1e-6);
    }

    #[test]
    fn improvement_gives_positive_reward() {
        let pre = ObjectiveScores { defense: 0.5, economy: 0.5, population: 0.5, connectivity: 0.5, garrison: 0.5, spatial: 0.5 };
        let post = ObjectiveScores { defense: 0.8, ..pre.clone() };
        let ideal = ObjectiveScores { defense: 1.0, economy: 1.0, population: 1.0, connectivity: 1.0, garrison: 1.0, spatial: 1.0 };
        let reward = compute_reward(&pre, &post, &ideal, ChallengeCategory::Military);
        assert!(reward > 0.0, "got {}", reward);
    }

    #[test]
    fn degradation_gives_negative_reward() {
        let pre = ObjectiveScores { defense: 0.8, economy: 0.5, population: 0.5, connectivity: 0.5, garrison: 0.5, spatial: 0.5 };
        let post = ObjectiveScores { defense: 0.3, ..pre.clone() };
        let ideal = ObjectiveScores { defense: 1.0, economy: 1.0, population: 1.0, connectivity: 1.0, garrison: 1.0, spatial: 1.0 };
        let reward = compute_reward(&pre, &post, &ideal, ChallengeCategory::Military);
        assert!(reward < 0.0, "got {}", reward);
    }

    #[test]
    fn chebyshev_at_ideal_is_zero() {
        let ideal = ObjectiveScores { defense: 1.0, economy: 1.0, population: 1.0, connectivity: 1.0, garrison: 1.0, spatial: 1.0 };
        let w = category_weights(ChallengeCategory::Military);
        let score = chebyshev_score(&ideal, &ideal, &w);
        assert!((score).abs() < 1e-6);
    }
}
