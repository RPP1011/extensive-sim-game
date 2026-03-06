use serde::{Deserialize, Serialize};

use crate::ai::effects::{AbilityTarget, AbilityTargeting, Effect};

// ---------------------------------------------------------------------------
// Ability categories
// ---------------------------------------------------------------------------

/// Categories that group abilities by targeting mode and purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbilityCategory {
    DamageUnit,   // Single-target damage (TargetEnemy + damage hint)
    DamageAoe,    // AoE damage (SelfAoe/GroundTarget + damage hint)
    CcUnit,       // Crowd control (TargetEnemy + crowd_control hint)
    HealUnit,     // Single-target heal (TargetAlly + heal hint)
    HealAoe,      // AoE/self heal (SelfCast/SelfAoe + heal hint)
    Defense,      // Shields, damage reduction (defense hint)
    Utility,      // Dashes, buffs (utility hint)
    Summon,       // Creates ally units (summon effects)
    Obstacle,     // Terrain/wall creation (obstacle effects)
}

impl AbilityCategory {
    pub fn count() -> usize {
        9
    }

    /// Classify an ability by hint, targeting, AND effect types.
    /// Effect-based detection takes priority for summons/obstacles since
    /// those are often tagged as "utility" in ai_hint.
    pub fn from_ability_full(
        hint: &str,
        targeting: &AbilityTargeting,
        effects: &[crate::ai::effects::ConditionalEffect],
        delivery: Option<&crate::ai::effects::Delivery>,
    ) -> Self {
        // Check effects for summon/obstacle — these override ai_hint
        let has_summon = effects.iter().any(|ce| matches!(ce.effect, Effect::Summon { .. }));
        let has_obstacle = effects.iter().any(|ce| matches!(ce.effect, Effect::Obstacle { .. }));

        // Also check delivery on_hit/on_arrival for summon/obstacle
        let delivery_effects: Vec<&crate::ai::effects::ConditionalEffect> = match delivery {
            Some(crate::ai::effects::Delivery::Projectile { on_hit, on_arrival, .. }) => {
                on_hit.iter().chain(on_arrival.iter()).collect()
            }
            Some(crate::ai::effects::Delivery::Chain { on_hit, .. }) => {
                on_hit.iter().collect()
            }
            Some(crate::ai::effects::Delivery::Tether { on_complete, .. }) => {
                on_complete.iter().collect()
            }
            _ => Vec::new(),
        };
        let delivery_has_summon = delivery_effects.iter().any(|ce| matches!(ce.effect, Effect::Summon { .. }));
        let delivery_has_obstacle = delivery_effects.iter().any(|ce| matches!(ce.effect, Effect::Obstacle { .. }));

        if has_obstacle || delivery_has_obstacle {
            return Self::Obstacle;
        }
        if has_summon || delivery_has_summon {
            return Self::Summon;
        }

        // Fall back to hint-based classification
        Self::from_ability(hint, targeting)
    }

    /// Simple classification from ai_hint + targeting only (backwards compatible).
    pub fn from_ability(hint: &str, targeting: &AbilityTargeting) -> Self {
        match (hint, targeting) {
            ("damage", AbilityTargeting::TargetEnemy) => Self::DamageUnit,
            ("damage", _) => Self::DamageAoe,
            ("crowd_control", _) => Self::CcUnit,
            ("heal", AbilityTargeting::TargetAlly) => Self::HealUnit,
            ("heal", _) => Self::HealAoe,
            ("defense", _) => Self::Defense,
            ("utility", _) => Self::Utility,
            // Default: if it targets enemies it's damage, otherwise utility
            (_, AbilityTargeting::TargetEnemy) => Self::DamageUnit,
            _ => Self::Utility,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::DamageUnit => "damage_unit",
            Self::DamageAoe => "damage_aoe",
            Self::CcUnit => "cc_unit",
            Self::HealUnit => "heal_unit",
            Self::HealAoe => "heal_aoe",
            Self::Defense => "defense",
            Self::Utility => "utility",
            Self::Summon => "summon",
            Self::Obstacle => "obstacle",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "damage_unit" => Some(Self::DamageUnit),
            "damage_aoe" => Some(Self::DamageAoe),
            "cc_unit" => Some(Self::CcUnit),
            "heal_unit" => Some(Self::HealUnit),
            "heal_aoe" => Some(Self::HealAoe),
            "defense" => Some(Self::Defense),
            "utility" => Some(Self::Utility),
            "summon" => Some(Self::Summon),
            "obstacle" => Some(Self::Obstacle),
            _ => None,
        }
    }

    /// Maximum feature count for any category.
    pub fn max_features() -> usize {
        28 // With terrain features added
    }
}

// ---------------------------------------------------------------------------
// Evaluator result
// ---------------------------------------------------------------------------

/// Output from an ability evaluator.
#[derive(Debug, Clone)]
pub struct AbilityEvalResult {
    pub ability_index: usize,
    pub category: AbilityCategory,
    pub urgency: f32,
    pub target: AbilityTarget,
}

/// Threshold above which an ability interrupts normal combat.
pub const URGENCY_THRESHOLD: f32 = 0.4;
