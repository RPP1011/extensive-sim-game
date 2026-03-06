//! Hero template loading — converts TOML hero definitions into SimState units.

use std::collections::{HashMap, VecDeque};

use crate::ai::core::{sim_vec2, SimState, SimVec2, Team, UnitState};
use crate::ai::effects::{AbilitySlot, HeroToml, PassiveSlot};

/// All built-in hero templates, loadable by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeroTemplate {
    Warrior,
    Ranger,
    Mage,
    Cleric,
    Rogue,
    Paladin,
}

impl HeroTemplate {
    pub const ALL: &'static [HeroTemplate] = &[
        HeroTemplate::Warrior,
        HeroTemplate::Ranger,
        HeroTemplate::Mage,
        HeroTemplate::Cleric,
        HeroTemplate::Rogue,
        HeroTemplate::Paladin,
    ];

    pub fn file_name(self) -> &'static str {
        match self {
            HeroTemplate::Warrior => "warrior.toml",
            HeroTemplate::Ranger => "ranger.toml",
            HeroTemplate::Mage => "mage.toml",
            HeroTemplate::Cleric => "cleric.toml",
            HeroTemplate::Rogue => "rogue.toml",
            HeroTemplate::Paladin => "paladin.toml",
        }
    }
}

/// Load a hero TOML string into a HeroToml struct.
pub fn parse_hero_toml(toml_str: &str) -> Result<HeroToml, String> {
    toml::from_str(toml_str).map_err(|e| format!("hero TOML parse error: {e}"))
}

/// Convert a HeroToml into a UnitState ready for simulation.
pub fn hero_toml_to_unit(toml: &HeroToml, id: u32, team: Team, position: SimVec2) -> UnitState {
    let atk = toml.attack.clone().unwrap_or_default();
    let abilities: Vec<AbilitySlot> = toml
        .abilities
        .iter()
        .map(|def| AbilitySlot::new(def.clone()))
        .collect();
    let passives: Vec<PassiveSlot> = toml
        .passives
        .iter()
        .map(|def| PassiveSlot::new(def.clone()))
        .collect();

    UnitState {
        id,
        team,
        hp: toml.stats.hp,
        max_hp: toml.stats.hp,
        position,
        move_speed_per_sec: toml.stats.move_speed,
        attack_damage: atk.damage,
        attack_range: atk.range,
        attack_cooldown_ms: atk.cooldown,
        attack_cast_time_ms: atk.cast_time,
        cooldown_remaining_ms: 0,
        // Hero abilities replace the old ability/heal/control system
        ability_damage: 0,
        ability_range: 0.0,
        ability_cooldown_ms: 0,
        ability_cast_time_ms: 0,
        ability_cooldown_remaining_ms: 0,
        heal_amount: 0,
        heal_range: 0.0,
        heal_cooldown_ms: 0,
        heal_cast_time_ms: 0,
        heal_cooldown_remaining_ms: 0,
        control_range: 0.0,
        control_duration_ms: 0,
        control_cooldown_ms: 0,
        control_cast_time_ms: 0,
        control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0,
        casting: None,
        abilities,
        passives,
        status_effects: Vec::new(),
        shield_hp: 0,
        resistance_tags: toml.stats.tags.clone(),
        state_history: VecDeque::new(),
        channeling: None,
        resource: toml.stats.resource,
        max_resource: toml.stats.max_resource,
        resource_regen_per_sec: toml.stats.resource_regen_per_sec,
        owner_id: None,
        directed: false,
        armor: toml.stats.armor,
        magic_resist: toml.stats.magic_resist,
        cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    }
}

/// Load all embedded hero templates (compile-time).
/// Returns a map from template name to HeroToml.
pub fn load_embedded_templates() -> HashMap<HeroTemplate, HeroToml> {
    let templates = [
        (HeroTemplate::Warrior, include_str!("../../assets/hero_templates/warrior.toml")),
        (HeroTemplate::Ranger, include_str!("../../assets/hero_templates/ranger.toml")),
        (HeroTemplate::Mage, include_str!("../../assets/hero_templates/mage.toml")),
        (HeroTemplate::Cleric, include_str!("../../assets/hero_templates/cleric.toml")),
        (HeroTemplate::Rogue, include_str!("../../assets/hero_templates/rogue.toml")),
        (HeroTemplate::Paladin, include_str!("../../assets/hero_templates/paladin.toml")),
    ];

    templates
        .into_iter()
        .map(|(key, toml_str)| {
            let toml = parse_hero_toml(toml_str)
                .unwrap_or_else(|e| panic!("failed to parse {}: {e}", key.file_name()));
            (key, toml)
        })
        .collect()
}

/// Build a default 3-hero party SimState using the standard templates.
pub fn default_hero_party(seed: u64) -> SimState {
    let templates = load_embedded_templates();
    let party = [HeroTemplate::Warrior, HeroTemplate::Cleric, HeroTemplate::Rogue];
    let positions = [
        sim_vec2(-2.0, 0.0),
        sim_vec2(-4.0, 1.0),
        sim_vec2(-3.0, -1.0),
    ];

    let units: Vec<UnitState> = party
        .iter()
        .zip(positions.iter())
        .enumerate()
        .map(|(i, (template, pos))| {
            let toml = &templates[template];
            hero_toml_to_unit(toml, (i + 1) as u32, Team::Hero, *pos)
        })
        .collect();

    SimState {
        tick: 0,
        rng_state: seed,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_templates_parse() {
        let templates = load_embedded_templates();
        assert_eq!(templates.len(), 6);
        for (key, toml) in &templates {
            assert!(
                !toml.hero.name.is_empty(),
                "{:?} has empty name",
                key
            );
            assert!(toml.stats.hp > 0, "{:?} has no HP", key);
        }
    }

    #[test]
    fn warrior_has_correct_stats() {
        let templates = load_embedded_templates();
        let w = &templates[&HeroTemplate::Warrior];
        assert_eq!(w.stats.hp, 180);
        assert_eq!(w.abilities.len(), 8);
        assert_eq!(w.passives.len(), 2);
        assert!(w.stats.tags.contains_key("CROWD_CONTROL"));
    }

    #[test]
    fn hero_toml_converts_to_unit() {
        let templates = load_embedded_templates();
        let w = &templates[&HeroTemplate::Warrior];
        let unit = hero_toml_to_unit(w, 1, Team::Hero, sim_vec2(0.0, 0.0));
        assert_eq!(unit.hp, 180);
        assert_eq!(unit.max_hp, 180);
        assert_eq!(unit.abilities.len(), 8);
        assert_eq!(unit.passives.len(), 2);
        assert!(unit.resistance_tags.contains_key("CROWD_CONTROL"));
    }

    #[test]
    fn default_party_creates_valid_state() {
        let state = default_hero_party(42);
        assert_eq!(state.units.len(), 3);
        assert!(state.units.iter().all(|u| u.hp > 0));
        assert!(state.units.iter().all(|u| u.team == Team::Hero));
    }

    #[test]
    fn ranger_has_projectile_delivery() {
        let templates = load_embedded_templates();
        let r = &templates[&HeroTemplate::Ranger];
        let power_shot = &r.abilities[0];
        assert!(power_shot.delivery.is_some(), "Power Shot should have projectile delivery");
    }

    #[test]
    fn cleric_has_heal_ability() {
        let templates = load_embedded_templates();
        let c = &templates[&HeroTemplate::Cleric];
        assert_eq!(c.abilities.len(), 8);
        let holy_light = &c.abilities[0];
        assert_eq!(holy_light.name, "HolyLight");
    }
}
