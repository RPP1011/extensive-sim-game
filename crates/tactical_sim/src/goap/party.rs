//! PartyCulture: group-level behavioral identity that modifies GOAP parameters.

use std::collections::HashMap;

/// Cultural traits that influence how a party behaves.
#[derive(Debug, Clone)]
pub struct CultureTraits {
    /// [-0.3, 0.3] — shifts goal insistence toward attack (positive) or defense (negative).
    pub aggression_bias: f32,
    /// [0, 1] — high = focus fire same target, low = spread damage.
    pub coordination: f32,
    /// [0, 1] — high = use abilities earlier, low = conserve cooldowns.
    pub ability_eagerness: f32,
    /// [0, 1] — HP% below which survival goals spike in priority.
    pub retreat_threshold: f32,
    /// [0, 1] — how strongly allies respond to low-HP teammates.
    pub protect_instinct: f32,
}

impl Default for CultureTraits {
    fn default() -> Self {
        Self {
            aggression_bias: 0.0,
            coordination: 0.5,
            ability_eagerness: 0.5,
            retreat_threshold: 0.3,
            protect_instinct: 0.5,
        }
    }
}

/// Party-wide behavioral modifiers applied to GOAP parameters.
#[derive(Debug, Clone)]
pub struct PartyCulture {
    pub name: String,
    /// goal_name → insistence multiplier.
    pub goal_insistence_modifiers: HashMap<String, f32>,
    /// action_name → cost multiplier.
    pub action_cost_modifiers: HashMap<String, f32>,
    /// Override default 0.15 hysteresis threshold.
    pub replan_hysteresis: f32,
    pub traits: CultureTraits,
}

impl Default for PartyCulture {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            goal_insistence_modifiers: HashMap::new(),
            action_cost_modifiers: HashMap::new(),
            replan_hysteresis: 0.15,
            traits: CultureTraits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Archetype {
    Aggressive,
    Defensive,
    Balanced,
    Tricky,
}

const ADJECTIVES: &[&str] = &[
    "Iron", "Shadow", "Blood", "Storm", "Crimson",
    "Ashen", "Silent", "Frost", "Ember", "Void",
    "Golden", "Dark", "Wild", "Pale", "Dire",
];

const NOUNS: &[&str] = &[
    "Vanguard", "Covenant", "Legion", "Watch", "Pact",
    "Order", "Horde", "Syndicate", "Circle", "Band",
    "Guard", "Brotherhood", "Coven", "Warband", "Pack",
];

impl PartyCulture {
    /// Generate a random culture with coherent trait combinations.
    pub fn generate(rng: &mut impl FnMut() -> u64) -> Self {
        let roll = (rng() % 4) as u8;
        let archetype = match roll {
            0 => Archetype::Aggressive,
            1 => Archetype::Defensive,
            2 => Archetype::Balanced,
            _ => Archetype::Tricky,
        };

        let mut traits = match archetype {
            Archetype::Aggressive => CultureTraits {
                aggression_bias: 0.2,
                coordination: 0.7,
                ability_eagerness: 0.8,
                retreat_threshold: 0.15,
                protect_instinct: 0.3,
            },
            Archetype::Defensive => CultureTraits {
                aggression_bias: -0.15,
                coordination: 0.5,
                ability_eagerness: 0.3,
                retreat_threshold: 0.45,
                protect_instinct: 0.8,
            },
            Archetype::Balanced => CultureTraits {
                aggression_bias: 0.0,
                coordination: 0.5,
                ability_eagerness: 0.5,
                retreat_threshold: 0.3,
                protect_instinct: 0.5,
            },
            Archetype::Tricky => CultureTraits {
                aggression_bias: 0.05,
                coordination: 0.3,
                ability_eagerness: 0.9,
                retreat_threshold: 0.25,
                protect_instinct: 0.4,
            },
        };

        // Random perturbation ±0.1
        let perturb = |rng: &mut dyn FnMut() -> u64, val: &mut f32, lo: f32, hi: f32| {
            let r = ((rng() % 201) as f32 / 1000.0) - 0.1; // [-0.1, 0.1]
            *val = (*val + r).clamp(lo, hi);
        };
        perturb(&mut *rng, &mut traits.aggression_bias, -0.3, 0.3);
        perturb(&mut *rng, &mut traits.coordination, 0.0, 1.0);
        perturb(&mut *rng, &mut traits.ability_eagerness, 0.0, 1.0);
        perturb(&mut *rng, &mut traits.retreat_threshold, 0.0, 1.0);
        perturb(&mut *rng, &mut traits.protect_instinct, 0.0, 1.0);

        // Build goal/action modifiers from traits
        let mut goal_mods = HashMap::new();
        let mut action_mods = HashMap::new();

        // Aggression bias shifts attack goals up and defense goals down
        if traits.aggression_bias > 0.05 {
            goal_mods.insert("engage".to_string(), 1.0 + traits.aggression_bias);
            goal_mods.insert("kill_target".to_string(), 1.0 + traits.aggression_bias);
            goal_mods.insert("stay_safe".to_string(), 1.0 - traits.aggression_bias * 0.5);
        } else if traits.aggression_bias < -0.05 {
            goal_mods.insert("engage".to_string(), 1.0 + traits.aggression_bias);
            goal_mods.insert("protect_ally".to_string(), 1.0 - traits.aggression_bias);
            goal_mods.insert("stay_safe".to_string(), 1.0 - traits.aggression_bias);
        }

        // Protect instinct boosts heal/protect goals
        if traits.protect_instinct > 0.6 {
            goal_mods.insert("keep_team_alive".to_string(), 1.0 + (traits.protect_instinct - 0.5) * 0.5);
        }

        // Ability eagerness reduces ability action costs
        if traits.ability_eagerness > 0.6 {
            action_mods.insert("cc_interrupt".to_string(), 1.0 - (traits.ability_eagerness - 0.5) * 0.3);
        }

        // Generate thematic name
        let adj_idx = (rng() as usize) % ADJECTIVES.len();
        let noun_idx = (rng() as usize) % NOUNS.len();
        let name = format!("{} {}", ADJECTIVES[adj_idx], NOUNS[noun_idx]);

        // Hysteresis varies with archetype
        let hysteresis = match archetype {
            Archetype::Aggressive => 0.1,  // quicker to switch goals
            Archetype::Defensive => 0.2,   // more committed
            Archetype::Balanced => 0.15,
            Archetype::Tricky => 0.08,     // very reactive
        };

        PartyCulture {
            name,
            goal_insistence_modifiers: goal_mods,
            action_cost_modifiers: action_mods,
            replan_hysteresis: hysteresis,
            traits,
        }
    }

    /// Parse culture traits from scenario TOML fields.
    pub fn from_toml_fields(
        name: Option<&str>,
        aggression_bias: Option<f32>,
        coordination: Option<f32>,
        ability_eagerness: Option<f32>,
        retreat_threshold: Option<f32>,
        protect_instinct: Option<f32>,
    ) -> Self {
        let traits = CultureTraits {
            aggression_bias: aggression_bias.unwrap_or(0.0),
            coordination: coordination.unwrap_or(0.5),
            ability_eagerness: ability_eagerness.unwrap_or(0.5),
            retreat_threshold: retreat_threshold.unwrap_or(0.3),
            protect_instinct: protect_instinct.unwrap_or(0.5),
        };

        let mut culture = PartyCulture {
            name: name.unwrap_or("Custom").to_string(),
            traits,
            ..Default::default()
        };

        // Derive modifiers from traits (same logic as generate)
        if culture.traits.aggression_bias.abs() > 0.05 {
            let bias = culture.traits.aggression_bias;
            culture.goal_insistence_modifiers.insert("engage".to_string(), 1.0 + bias);
            if bias > 0.0 {
                culture.goal_insistence_modifiers.insert("stay_safe".to_string(), 1.0 - bias * 0.5);
            } else {
                culture.goal_insistence_modifiers.insert("protect_ally".to_string(), 1.0 - bias);
            }
        }

        culture
    }
}
