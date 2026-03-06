use serde::{Deserialize, Serialize};

use crate::ai::core::UnitState;

// ---------------------------------------------------------------------------
// Personality -- who the unit IS (7 character traits, 0.0-1.0)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Personality {
    pub aggression: f32,
    pub compassion: f32,
    pub caution: f32,
    pub discipline: f32,
    pub cunning: f32,
    pub tenacity: f32,
    pub patience: f32,
}

impl Default for Personality {
    fn default() -> Self {
        Self::default_balanced()
    }
}

impl Personality {
    pub fn default_balanced() -> Self {
        Self {
            aggression: 0.5,
            compassion: 0.5,
            caution: 0.5,
            discipline: 0.5,
            cunning: 0.5,
            tenacity: 0.5,
            patience: 0.5,
        }
    }

    /// Returns a zeroed personality (used as initial drift).
    pub(super) fn zero() -> Self {
        Self {
            aggression: 0.0,
            compassion: 0.0,
            caution: 0.0,
            discipline: 0.0,
            cunning: 0.0,
            tenacity: 0.0,
            patience: 0.0,
        }
    }

    /// Combine base personality + drift, clamping each trait to 0.0..1.0.
    pub fn effective(&self, drift: &Personality) -> Personality {
        Personality {
            aggression: (self.aggression + drift.aggression).clamp(0.0, 1.0),
            compassion: (self.compassion + drift.compassion).clamp(0.0, 1.0),
            caution: (self.caution + drift.caution).clamp(0.0, 1.0),
            discipline: (self.discipline + drift.discipline).clamp(0.0, 1.0),
            cunning: (self.cunning + drift.cunning).clamp(0.0, 1.0),
            tenacity: (self.tenacity + drift.tenacity).clamp(0.0, 1.0),
            patience: (self.patience + drift.patience).clamp(0.0, 1.0),
        }
    }
}

/// Infer a personality from a unit's stat block.
pub fn infer_personality(unit: &UnitState) -> Personality {
    let heal_ability_count = unit.abilities.iter().filter(|a| a.def.ai_hint == "heal").count();
    let cc_ability_count = unit.abilities.iter().filter(|a| a.def.ai_hint == "control").count();
    let is_melee = unit.attack_range <= 1.5;
    let is_tanky = unit.max_hp >= 140;
    let has_legacy_heal = unit.heal_amount > 0;

    // Graduated personality inference -- not hard cutoffs.
    let mut p = Personality::default_balanced();

    // Dedicated healer: high compassion, high caution, high patience, low aggression
    if heal_ability_count >= 2 || has_legacy_heal {
        let heal_strength = if heal_ability_count >= 2 { 1.0 } else { 0.6 };
        p.compassion = 0.5 + 0.4 * heal_strength;
        p.caution = 0.5 + 0.3 * heal_strength;
        p.patience = 0.5 + 0.3 * heal_strength;
        p.aggression = 0.5 - 0.3 * heal_strength;
    }

    // Tanky melee: high aggression, high discipline, low caution
    if is_tanky && is_melee {
        p.aggression = p.aggression.max(0.8);
        p.discipline = p.discipline.max(0.75);
        p.caution = p.caution.min(0.25);
        p.tenacity = p.tenacity.max(0.7);
    } else if is_tanky {
        // Tanky ranged -- moderate aggression, higher discipline
        p.aggression = p.aggression.max(0.6);
        p.discipline = p.discipline.max(0.7);
        p.caution = p.caution.min(0.4);
    }

    // Ranged DPS: moderate aggression, moderate cunning, moderate caution
    if !is_tanky && !is_melee && heal_ability_count < 2 && !has_legacy_heal {
        p.aggression = 0.55;
        p.cunning = 0.65;
        p.caution = 0.55;
    }

    // CC specialist: high cunning, high discipline
    if cc_ability_count >= 1 {
        p.cunning = p.cunning.max(0.75);
        p.discipline = p.discipline.max(0.65);
    }

    p
}
