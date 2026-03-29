//! Trait interfaces for class and ability generation.
//!
//! Implementations are swappable -- the world sim depends on traits, not concrete types.
//! The `DefaultClassGenerator` uses a hardcoded template table of 12 base classes.
//! The `DefaultAbilityGenerator` produces placeholder tier-scaled buff abilities.

use crate::world_sim::state::tag;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of matching a behavior profile against class templates.
pub struct ClassMatch {
    pub class_name_hash: u32,
    pub display_name: String,
    pub score: f32,
}

/// A procedurally generated class definition.
pub struct ClassDef {
    pub name_hash: u32,
    pub display_name: String,
    pub tag_requirements: Vec<(u32, f32)>,
}

/// A generated ability for a class level-up.
pub struct GeneratedAbility {
    pub name: String,
    pub dsl_text: String,
    pub is_passive: bool,
    pub tier: u32,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Generates class grants from NPC behavior profiles.
pub trait ClassGenerator: Send + Sync {
    /// Match behavior profile against known class templates.
    /// Returns classes the NPC qualifies for (score >= threshold).
    fn match_classes(&self, behavior_tags: &[u32], behavior_values: &[f32]) -> Vec<ClassMatch>;

    /// Generate a unique class when no template matches but behavior is significant.
    fn generate_unique_class(
        &self,
        behavior_tags: &[u32],
        behavior_values: &[f32],
        seed: u64,
    ) -> Option<ClassDef>;
}

/// Generates abilities when NPCs level up in a class.
pub trait AbilityGenerator: Send + Sync {
    /// Generate an ability for the given class at the given tier.
    fn generate_ability(
        &self,
        class_name_hash: u32,
        archetype: &str,
        tier: u32,
        behavior_tags: &[u32],
        behavior_values: &[f32],
        seed: u64,
    ) -> GeneratedAbility;
}

// ---------------------------------------------------------------------------
// Class template table
// ---------------------------------------------------------------------------

struct ClassTemplate {
    name_hash: u32,
    display_name: &'static str,
    requirements: &'static [(u32, f32)],
    score_tags: &'static [(u32, f32)],
}

// Tag constants (computed at compile time via const fn).
const TAG_MELEE: u32 = tag(b"melee");
const TAG_RANGED: u32 = tag(b"ranged");
const TAG_COMBAT: u32 = tag(b"combat");
const TAG_DEFENSE: u32 = tag(b"defense");
const TAG_TACTICS: u32 = tag(b"tactics");
const TAG_ENDURANCE: u32 = tag(b"endurance");
const TAG_AWARENESS: u32 = tag(b"awareness");
const TAG_SURVIVAL: u32 = tag(b"survival");
const TAG_NAVIGATION: u32 = tag(b"navigation");
const TAG_MEDICINE: u32 = tag(b"medicine");
const TAG_FAITH: u32 = tag(b"faith");
const TAG_HERBALISM: u32 = tag(b"herbalism");
const TAG_RESILIENCE: u32 = tag(b"resilience");
const TAG_TRADE: u32 = tag(b"trade");
const TAG_NEGOTIATION: u32 = tag(b"negotiation");
const TAG_DIPLOMACY: u32 = tag(b"diplomacy");
const TAG_RESEARCH: u32 = tag(b"research");
const TAG_LORE: u32 = tag(b"lore");
const TAG_DISCIPLINE: u32 = tag(b"discipline");
const TAG_STEALTH: u32 = tag(b"stealth");
const TAG_DECEPTION: u32 = tag(b"deception");
const TAG_CRAFTING: u32 = tag(b"crafting");
const TAG_SMITHING: u32 = tag(b"smithing");
const TAG_LABOR: u32 = tag(b"labor");
const TAG_LEADERSHIP: u32 = tag(b"leadership");
const TAG_FARMING: u32 = tag(b"farming");
const TAG_MINING: u32 = tag(b"mining");

/// Minimum dot-product score for a class match.
const SCORE_THRESHOLD: f32 = 0.3;

static TEMPLATES: &[ClassTemplate] = &[
    ClassTemplate {
        name_hash: tag(b"Warrior"),
        display_name: "Warrior",
        requirements: &[(TAG_MELEE, 100.0)],
        score_tags: &[(TAG_MELEE, 0.4), (TAG_DEFENSE, 0.3), (TAG_ENDURANCE, 0.2), (TAG_COMBAT, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Ranger"),
        display_name: "Ranger",
        requirements: &[(TAG_RANGED, 100.0)],
        score_tags: &[(TAG_RANGED, 0.4), (TAG_AWARENESS, 0.2), (TAG_SURVIVAL, 0.2), (TAG_NAVIGATION, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Guardian"),
        display_name: "Guardian",
        requirements: &[(TAG_DEFENSE, 100.0), (TAG_ENDURANCE, 50.0)],
        score_tags: &[(TAG_DEFENSE, 0.4), (TAG_ENDURANCE, 0.3), (TAG_RESILIENCE, 0.2), (TAG_COMBAT, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Healer"),
        display_name: "Healer",
        requirements: &[(TAG_MEDICINE, 50.0)],
        score_tags: &[(TAG_MEDICINE, 0.4), (TAG_FAITH, 0.2), (TAG_HERBALISM, 0.2), (TAG_RESILIENCE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Merchant"),
        display_name: "Merchant",
        requirements: &[(TAG_TRADE, 100.0)],
        score_tags: &[(TAG_TRADE, 0.4), (TAG_NEGOTIATION, 0.3), (TAG_DIPLOMACY, 0.2), (TAG_NAVIGATION, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Scholar"),
        display_name: "Scholar",
        requirements: &[(TAG_RESEARCH, 50.0), (TAG_LORE, 30.0)],
        score_tags: &[(TAG_RESEARCH, 0.4), (TAG_LORE, 0.3), (TAG_MEDICINE, 0.15), (TAG_DISCIPLINE, 0.15)],
    },
    ClassTemplate {
        name_hash: tag(b"Rogue"),
        display_name: "Rogue",
        requirements: &[(TAG_STEALTH, 50.0)],
        score_tags: &[(TAG_STEALTH, 0.4), (TAG_DECEPTION, 0.3), (TAG_AWARENESS, 0.2), (TAG_SURVIVAL, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Artisan"),
        display_name: "Artisan",
        requirements: &[(TAG_CRAFTING, 100.0)],
        score_tags: &[(TAG_CRAFTING, 0.3), (TAG_SMITHING, 0.3), (TAG_LABOR, 0.2), (TAG_ENDURANCE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Diplomat"),
        display_name: "Diplomat",
        requirements: &[(TAG_DIPLOMACY, 100.0)],
        score_tags: &[(TAG_DIPLOMACY, 0.4), (TAG_NEGOTIATION, 0.2), (TAG_LEADERSHIP, 0.2), (TAG_TRADE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Commander"),
        display_name: "Commander",
        requirements: &[(TAG_LEADERSHIP, 50.0), (TAG_TACTICS, 30.0)],
        score_tags: &[(TAG_LEADERSHIP, 0.3), (TAG_TACTICS, 0.3), (TAG_COMBAT, 0.2), (TAG_DISCIPLINE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Farmer"),
        display_name: "Farmer",
        requirements: &[(TAG_FARMING, 100.0)],
        score_tags: &[(TAG_FARMING, 0.4), (TAG_LABOR, 0.3), (TAG_ENDURANCE, 0.2), (TAG_SURVIVAL, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Miner"),
        display_name: "Miner",
        requirements: &[(TAG_MINING, 100.0)],
        score_tags: &[(TAG_MINING, 0.4), (TAG_ENDURANCE, 0.3), (TAG_LABOR, 0.2), (TAG_SMITHING, 0.1)],
    },
];

// ---------------------------------------------------------------------------
// Tag hash -> display name table (for variant naming)
// ---------------------------------------------------------------------------

/// Resolve a tag hash to its display name. Returns None for unknown hashes.
fn tag_display_name(hash: u32) -> Option<&'static str> {
    static TABLE: &[(u32, &str)] = &[
        (TAG_MELEE, "Melee"),
        (TAG_RANGED, "Ranged"),
        (TAG_COMBAT, "Combat"),
        (TAG_DEFENSE, "Defense"),
        (TAG_TACTICS, "Tactics"),
        (TAG_ENDURANCE, "Endurance"),
        (TAG_AWARENESS, "Awareness"),
        (TAG_SURVIVAL, "Survival"),
        (TAG_NAVIGATION, "Navigation"),
        (TAG_MEDICINE, "Medicine"),
        (TAG_FAITH, "Faith"),
        (TAG_HERBALISM, "Herbalism"),
        (TAG_RESILIENCE, "Resilience"),
        (TAG_TRADE, "Trade"),
        (TAG_NEGOTIATION, "Negotiation"),
        (TAG_DIPLOMACY, "Diplomacy"),
        (TAG_RESEARCH, "Research"),
        (TAG_LORE, "Lore"),
        (TAG_DISCIPLINE, "Discipline"),
        (TAG_STEALTH, "Stealth"),
        (TAG_DECEPTION, "Deception"),
        (TAG_CRAFTING, "Crafting"),
        (TAG_SMITHING, "Smithing"),
        (TAG_LABOR, "Labor"),
        (TAG_LEADERSHIP, "Leadership"),
        (TAG_FARMING, "Farming"),
        (TAG_MINING, "Mining"),
    ];
    TABLE.iter().find(|&&(h, _)| h == hash).map(|&(_, name)| name)
}

// ---------------------------------------------------------------------------
// Helper: look up a tag value in sorted parallel arrays (O(log n))
// ---------------------------------------------------------------------------

fn lookup_tag(behavior_tags: &[u32], behavior_values: &[f32], tag_hash: u32) -> f32 {
    match behavior_tags.binary_search(&tag_hash) {
        Ok(idx) => behavior_values[idx],
        Err(_) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// DefaultClassGenerator
// ---------------------------------------------------------------------------

pub struct DefaultClassGenerator;

impl DefaultClassGenerator {
    pub fn new() -> Self {
        DefaultClassGenerator
    }
}

impl ClassGenerator for DefaultClassGenerator {
    fn match_classes(&self, behavior_tags: &[u32], behavior_values: &[f32]) -> Vec<ClassMatch> {
        let mut matches = Vec::new();

        for tmpl in TEMPLATES {
            // Check all requirements are met.
            let mut qualified = true;
            for &(req_tag, min_val) in tmpl.requirements {
                if lookup_tag(behavior_tags, behavior_values, req_tag) < min_val {
                    qualified = false;
                    break;
                }
            }
            if !qualified {
                continue;
            }

            // Compute weighted dot product of behavior profile x score weights.
            let mut score = 0.0f32;
            let mut best_tag_hash = 0u32;
            let mut best_weighted = 0.0f32;
            let mut second_tag_hash = 0u32;
            let mut second_weighted = 0.0f32;

            for &(score_tag, weight) in tmpl.score_tags {
                let val = lookup_tag(behavior_tags, behavior_values, score_tag);
                let weighted = val * weight;
                score += weighted;

                if weighted > best_weighted {
                    second_tag_hash = best_tag_hash;
                    second_weighted = best_weighted;
                    best_tag_hash = score_tag;
                    best_weighted = weighted;
                } else if weighted > second_weighted {
                    second_tag_hash = score_tag;
                    second_weighted = weighted;
                }
            }

            if score < SCORE_THRESHOLD {
                continue;
            }

            // Variant naming: if second-highest weighted tag exceeds primary * 0.8, append suffix.
            let display_name = if best_weighted > 0.0
                && second_weighted > best_weighted * 0.8
                && second_tag_hash != 0
            {
                if let Some(suffix) = tag_display_name(second_tag_hash) {
                    format!("{} of {}", tmpl.display_name, suffix)
                } else {
                    tmpl.display_name.to_string()
                }
            } else {
                tmpl.display_name.to_string()
            };

            matches.push(ClassMatch {
                class_name_hash: tmpl.name_hash,
                display_name,
                score,
            });
        }

        matches
    }

    fn generate_unique_class(
        &self,
        _behavior_tags: &[u32],
        _behavior_values: &[f32],
        _seed: u64,
    ) -> Option<ClassDef> {
        // Placeholder: no procedural class generation yet.
        None
    }
}

// ---------------------------------------------------------------------------
// DefaultAbilityGenerator
// ---------------------------------------------------------------------------

pub struct DefaultAbilityGenerator;

impl AbilityGenerator for DefaultAbilityGenerator {
    fn generate_ability(
        &self,
        _class_name_hash: u32,
        _archetype: &str,
        tier: u32,
        _behavior_tags: &[u32],
        _behavior_values: &[f32],
        _seed: u64,
    ) -> GeneratedAbility {
        let power = tier as f32 * 5.0;
        GeneratedAbility {
            name: format!("Skill T{}", tier),
            dsl_text: format!("buff attack {} for 10s", power),
            is_passive: tier % 2 == 0,
            tier,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(pairs: &[(&[u8], f32)]) -> (Vec<u32>, Vec<f32>) {
        let mut entries: Vec<(u32, f32)> = pairs.iter().map(|(name, val)| (tag(name), *val)).collect();
        entries.sort_by_key(|&(h, _)| h);
        let tags = entries.iter().map(|&(h, _)| h).collect();
        let values = entries.iter().map(|&(_, v)| v).collect();
        (tags, values)
    }

    #[test]
    fn warrior_matches_with_high_melee() {
        let (tags, values) = make_profile(&[
            (b"melee", 200.0),
            (b"defense", 80.0),
            (b"endurance", 60.0),
            (b"combat", 40.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&tags, &values);
        assert!(
            matches.iter().any(|m| m.display_name.starts_with("Warrior")),
            "Expected Warrior class match, got: {:?}",
            matches.iter().map(|m| &m.display_name).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn no_match_below_requirements() {
        let (tags, values) = make_profile(&[
            (b"melee", 50.0),
            (b"defense", 10.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&tags, &values);
        assert!(
            matches.iter().all(|m| !m.display_name.starts_with("Warrior")),
            "Warrior should not match with melee=50",
        );
    }

    #[test]
    fn variant_naming_when_secondary_high() {
        let (tags, values) = make_profile(&[
            (b"crafting", 200.0),
            (b"smithing", 200.0),
            (b"labor", 50.0),
            (b"endurance", 50.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&tags, &values);
        let artisan = matches.iter().find(|m| m.class_name_hash == tag(b"Artisan"));
        assert!(artisan.is_some(), "Artisan should match");
        let name = &artisan.unwrap().display_name;
        assert!(name.contains(" of "), "Expected variant name, got: {}", name);
    }

    #[test]
    fn multiple_classes_can_match() {
        let (tags, values) = make_profile(&[
            (b"melee", 200.0),
            (b"defense", 200.0),
            (b"endurance", 200.0),
            (b"combat", 100.0),
            (b"resilience", 100.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&tags, &values);
        let names: Vec<_> = matches.iter().map(|m| m.display_name.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("Warrior")), "Missing Warrior in {:?}", names);
        assert!(names.iter().any(|n| n.starts_with("Guardian")), "Missing Guardian in {:?}", names);
    }

    #[test]
    fn default_ability_generator_scales_by_tier() {
        let gen = DefaultAbilityGenerator;
        let a1 = gen.generate_ability(0, "test", 1, &[], &[], 42);
        let a3 = gen.generate_ability(0, "test", 3, &[], &[], 42);
        assert!(a1.dsl_text.contains("5"), "Tier 1 should have power 5");
        assert!(a3.dsl_text.contains("15"), "Tier 3 should have power 15");
        assert!(!a1.is_passive, "Tier 1 (odd) should not be passive");
        assert!(!a3.is_passive, "Tier 3 (odd) should not be passive");
    }
}
