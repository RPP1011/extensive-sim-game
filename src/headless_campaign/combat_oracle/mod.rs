//! Combat oracle for fast battle outcome prediction.
//!
//! During MCTS rollouts, running the full deterministic combat sim for each
//! battle is too slow. The oracle predicts outcomes in microseconds from
//! party composition and enemy strength.
//!
//! Two implementations:
//! - `HeuristicOracle`: deterministic formula-based (no training data needed)
//! - `MlpOracle`: small trained MLP (future — requires combat sim dataset)

use super::state::*;

/// Predicted combat outcome.
#[derive(Clone, Debug)]
pub struct CombatOracleResult {
    /// Probability of victory (0–1).
    pub victory_probability: f32,
    /// Expected fraction of party HP remaining after battle (0–1).
    pub expected_hp_remaining: f32,
    /// Expected battle duration in campaign ticks.
    pub expected_ticks: u64,
    /// Expected number of party casualties.
    pub expected_casualties: f32,
}

/// Trait for combat outcome prediction.
pub trait CombatOracle: Send + Sync {
    fn predict(
        &self,
        party_members: &[&Adventurer],
        enemy_strength: f32,
        threat_level: f32,
    ) -> CombatOracleResult;
}

// ---------------------------------------------------------------------------
// Heuristic oracle
// ---------------------------------------------------------------------------

/// Formula-based oracle: no training data needed.
///
/// Compares aggregate party power (stats + level + equipment) against
/// enemy threat level. Uses deterministic formulas calibrated to produce
/// reasonable outcomes across the threat range.
pub struct HeuristicOracle;

impl HeuristicOracle {
    /// Compute aggregate party power from member stats.
    fn party_power(members: &[&Adventurer]) -> f32 {
        members
            .iter()
            .map(|a| {
                let base = a.stats.attack + a.stats.defense + a.stats.ability_power;
                let level_bonus = a.level as f32 * 5.0;
                let hp_factor = a.stats.max_hp / 100.0;
                let condition = 1.0 - (a.injury / 200.0 + a.fatigue / 200.0);
                let morale_factor = 0.8 + a.morale / 500.0; // 0.8–1.0

                (base + level_bonus) * hp_factor * condition.max(0.1) * morale_factor
            })
            .sum()
    }

    /// Estimate victory probability from power ratio.
    fn win_probability(power_ratio: f32) -> f32 {
        // Sigmoid centered at ratio=1.0, steepness=2.5
        // ratio > 1.0 → favorable, ratio < 1.0 → unfavorable
        let x = (power_ratio - 1.0) * 2.5;
        1.0 / (1.0 + (-x).exp())
    }
}

impl CombatOracle for HeuristicOracle {
    fn predict(
        &self,
        party_members: &[&Adventurer],
        enemy_strength: f32,
        threat_level: f32,
    ) -> CombatOracleResult {
        if party_members.is_empty() {
            return CombatOracleResult {
                victory_probability: 0.0,
                expected_hp_remaining: 0.0,
                expected_ticks: 1,
                expected_casualties: 0.0,
            };
        }

        let party_pow = Self::party_power(party_members);
        let enemy_pow = enemy_strength * (threat_level / 50.0).max(0.5);
        let ratio = party_pow / enemy_pow.max(1.0);

        let win_prob = Self::win_probability(ratio);

        // HP remaining: scales with power advantage
        let hp_remaining = if win_prob > 0.5 {
            // Victory: keep more HP with stronger advantage
            (0.3 + 0.5 * (ratio - 1.0).max(0.0).min(1.0)).min(0.95)
        } else {
            // Defeat: party gets worn down
            (0.1 * ratio).min(0.3)
        };

        // Duration: harder fights last longer
        let base_duration = 30; // ticks (~3s game time)
        let difficulty_factor = (2.0 - ratio).max(0.5).min(3.0);
        let duration = (base_duration as f32 * difficulty_factor) as u64;

        // Casualties: probability per member based on difficulty
        let casualty_rate = (1.0 - win_prob) * 0.5;
        let expected_casualties = party_members.len() as f32 * casualty_rate;

        CombatOracleResult {
            victory_probability: win_prob,
            expected_hp_remaining: hp_remaining,
            expected_ticks: duration.max(5),
            expected_casualties,
        }
    }
}

// ---------------------------------------------------------------------------
// Default oracle
// ---------------------------------------------------------------------------

/// Create the default combat oracle (heuristic for now).
pub fn default_oracle() -> Box<dyn CombatOracle> {
    Box::new(HeuristicOracle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adventurer(level: u32, attack: f32, defense: f32) -> Adventurer {
        Adventurer {
            id: 1,
            name: "Test".into(),
            archetype: "knight".into(),
            level,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 100.0,
                attack,
                defense,
                speed: 10.0,
                ability_power: 10.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 70.0,
            stress: 10.0,
            fatigue: 10.0,
            injury: 0.0,
            resolve: 60.0,
            morale: 80.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
        }
    }

    #[test]
    fn test_strong_party_wins() {
        let oracle = HeuristicOracle;
        let a1 = make_adventurer(5, 20.0, 15.0);
        let a2 = make_adventurer(5, 18.0, 12.0);
        let members: Vec<&Adventurer> = vec![&a1, &a2];

        let result = oracle.predict(&members, 30.0, 20.0);
        assert!(
            result.victory_probability > 0.7,
            "Strong party should have high win prob: {}",
            result.victory_probability
        );
        assert!(result.expected_hp_remaining > 0.3);
    }

    #[test]
    fn test_weak_party_loses() {
        let oracle = HeuristicOracle;
        let a1 = make_adventurer(1, 5.0, 5.0);
        let members: Vec<&Adventurer> = vec![&a1];

        let result = oracle.predict(&members, 80.0, 70.0);
        assert!(
            result.victory_probability < 0.3,
            "Weak party should have low win prob: {}",
            result.victory_probability
        );
    }

    #[test]
    fn test_empty_party() {
        let oracle = HeuristicOracle;
        let result = oracle.predict(&[], 50.0, 50.0);
        assert_eq!(result.victory_probability, 0.0);
    }
}
