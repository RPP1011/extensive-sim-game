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
    fn match_classes(&self, behavior_profile: &[(u32, f32)]) -> Vec<ClassMatch>;

    /// Generate a unique class when no template matches but behavior is significant.
    fn generate_unique_class(
        &self,
        behavior_profile: &[(u32, f32)],
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
        behavior_profile: &[(u32, f32)],
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
const TAG_WOODWORK: u32 = tag(b"woodwork");
const TAG_ALCHEMY: u32 = tag(b"alchemy");
const TAG_EXPLORATION: u32 = tag(b"exploration");
const TAG_TEACHING: u32 = tag(b"teaching");
const TAG_CONSTRUCTION: u32 = tag(b"construction");
const TAG_ARCHITECTURE: u32 = tag(b"architecture");
const TAG_MASONRY: u32 = tag(b"masonry");

/// Minimum normalized alignment score for a class match.
/// Score uses sigmoid: raw/(raw+100). Score=100→0.5, 500→0.83.
/// Threshold 0.3 requires raw score ~43 (modest behavior alignment).
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
    ClassTemplate {
        name_hash: tag(b"Woodsman"),
        display_name: "Woodsman",
        requirements: &[(TAG_WOODWORK, 100.0)],
        score_tags: &[(TAG_WOODWORK, 0.4), (TAG_ENDURANCE, 0.3), (TAG_LABOR, 0.2), (TAG_SURVIVAL, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Alchemist"),
        display_name: "Alchemist",
        requirements: &[(TAG_ALCHEMY, 100.0)],
        score_tags: &[(TAG_ALCHEMY, 0.4), (TAG_RESEARCH, 0.2), (TAG_HERBALISM, 0.2), (TAG_MEDICINE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Herbalist"),
        display_name: "Herbalist",
        requirements: &[(TAG_HERBALISM, 100.0)],
        score_tags: &[(TAG_HERBALISM, 0.4), (TAG_MEDICINE, 0.3), (TAG_SURVIVAL, 0.2), (TAG_LORE, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Explorer"),
        display_name: "Explorer",
        requirements: &[(TAG_EXPLORATION, 50.0), (TAG_SURVIVAL, 30.0)],
        score_tags: &[(TAG_EXPLORATION, 0.3), (TAG_NAVIGATION, 0.3), (TAG_SURVIVAL, 0.2), (TAG_AWARENESS, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Mentor"),
        display_name: "Mentor",
        requirements: &[(TAG_TEACHING, 50.0), (TAG_LEADERSHIP, 30.0)],
        score_tags: &[(TAG_TEACHING, 0.4), (TAG_LEADERSHIP, 0.3), (TAG_DISCIPLINE, 0.2), (TAG_LABOR, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Builder"),
        display_name: "Builder",
        requirements: &[(TAG_CONSTRUCTION, 20.0), (TAG_LABOR, 10.0)],
        score_tags: &[(TAG_CONSTRUCTION, 0.4), (TAG_MASONRY, 0.3), (TAG_WOODWORK, 0.2), (TAG_LABOR, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Architect"),
        display_name: "Architect",
        requirements: &[(TAG_CONSTRUCTION, 50.0), (TAG_ARCHITECTURE, 30.0)],
        score_tags: &[(TAG_ARCHITECTURE, 0.4), (TAG_CONSTRUCTION, 0.3), (TAG_MASONRY, 0.2), (TAG_LEADERSHIP, 0.1)],
    },
    // --- Experience-driven classes (from surviving hardship, not work) ---
    ClassTemplate {
        name_hash: tag(b"Sentinel"),
        display_name: "Sentinel",
        requirements: &[(TAG_DEFENSE, 50.0), (TAG_RESILIENCE, 30.0)],
        score_tags: &[(TAG_DEFENSE, 0.3), (TAG_RESILIENCE, 0.3), (TAG_ENDURANCE, 0.2), (TAG_AWARENESS, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Survivor"),
        display_name: "Survivor",
        requirements: &[(TAG_SURVIVAL, 50.0), (TAG_ENDURANCE, 30.0)],
        score_tags: &[(TAG_SURVIVAL, 0.4), (TAG_ENDURANCE, 0.3), (TAG_RESILIENCE, 0.2), (TAG_AWARENESS, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Warden"),
        display_name: "Warden",
        requirements: &[(TAG_DEFENSE, 80.0), (TAG_COMBAT, 50.0)],
        score_tags: &[(TAG_DEFENSE, 0.3), (TAG_COMBAT, 0.3), (TAG_RESILIENCE, 0.2), (TAG_LEADERSHIP, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Veteran"),
        display_name: "Veteran",
        requirements: &[(TAG_COMBAT, 80.0), (TAG_TACTICS, 30.0)],
        score_tags: &[(TAG_COMBAT, 0.3), (TAG_TACTICS, 0.3), (TAG_MELEE, 0.2), (TAG_ENDURANCE, 0.2)],
    },
    ClassTemplate {
        name_hash: tag(b"Stalwart"),
        display_name: "Stalwart",
        requirements: &[(TAG_RESILIENCE, 80.0), (TAG_ENDURANCE, 50.0)],
        score_tags: &[(TAG_RESILIENCE, 0.4), (TAG_ENDURANCE, 0.3), (TAG_DEFENSE, 0.2), (TAG_SURVIVAL, 0.1)],
    },
    // --- Storytelling class ---
    ClassTemplate {
        name_hash: tag(b"Bard"),
        display_name: "Bard",
        requirements: &[(TAG_TEACHING, 100.0), (TAG_DIPLOMACY, 50.0)],
        score_tags: &[(TAG_TEACHING, 0.3), (TAG_DIPLOMACY, 0.3), (TAG_LEADERSHIP, 0.2), (TAG_TRADE, 0.2)],
    },
    // --- Seafaring classes ---
    ClassTemplate {
        name_hash: tag(b"Mariner"),
        display_name: "Mariner",
        requirements: &[(TAG_SEAFARING, 50.0), (TAG_NAVIGATION, 30.0)],
        score_tags: &[(TAG_SEAFARING, 0.4), (TAG_NAVIGATION, 0.3), (TAG_SURVIVAL, 0.2), (TAG_TRADE, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Sea Captain"),
        display_name: "Sea Captain",
        requirements: &[(TAG_SEAFARING, 100.0), (TAG_LEADERSHIP, 50.0)],
        score_tags: &[(TAG_SEAFARING, 0.3), (TAG_LEADERSHIP, 0.3), (TAG_NAVIGATION, 0.2), (TAG_COMBAT, 0.2)],
    },
    // --- Dungeoneering classes ---
    ClassTemplate {
        name_hash: tag(b"Delver"),
        display_name: "Delver",
        requirements: &[(TAG_DUNGEONEERING, 50.0), (TAG_SURVIVAL, 30.0)],
        score_tags: &[(TAG_DUNGEONEERING, 0.4), (TAG_SURVIVAL, 0.3), (TAG_AWARENESS, 0.2), (TAG_COMBAT, 0.1)],
    },
    ClassTemplate {
        name_hash: tag(b"Dungeon Master"),
        display_name: "Dungeon Master",
        requirements: &[(TAG_DUNGEONEERING, 100.0), (TAG_TACTICS, 50.0)],
        score_tags: &[(TAG_DUNGEONEERING, 0.3), (TAG_TACTICS, 0.3), (TAG_COMBAT, 0.2), (TAG_LEADERSHIP, 0.2)],
    },
    // --- Oath classes ---
    ClassTemplate {
        name_hash: tag(b"Oathkeeper"),
        display_name: "Oathkeeper",
        requirements: &[(TAG_FAITH, 50.0), (TAG_DISCIPLINE, 30.0)],
        score_tags: &[(TAG_FAITH, 0.4), (TAG_DISCIPLINE, 0.3), (TAG_RESILIENCE, 0.2), (TAG_LEADERSHIP, 0.1)],
    },
    // --- Villainy classes ---
    ClassTemplate {
        name_hash: tag(b"Betrayer"),
        display_name: "Betrayer",
        requirements: &[(TAG_DECEPTION, 80.0), (TAG_STEALTH, 50.0)],
        score_tags: &[(TAG_DECEPTION, 0.4), (TAG_STEALTH, 0.3), (TAG_AWARENESS, 0.2), (TAG_COMBAT, 0.1)],
    },
];

const TAG_SEAFARING: u32 = tag(b"seafaring");
const TAG_DUNGEONEERING: u32 = tag(b"dungeoneering");
const TAG_COMPASSION: u32 = tag(b"compassion");

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
        (TAG_WOODWORK, "Woodwork"),
        (TAG_ALCHEMY, "Alchemy"),
        (TAG_EXPLORATION, "Exploration"),
        (TAG_TEACHING, "Teaching"),
        (TAG_CONSTRUCTION, "Construction"),
        (TAG_ARCHITECTURE, "Architecture"),
        (TAG_MASONRY, "Masonry"),
        (TAG_COMPASSION, "Compassion"),
        (TAG_SEAFARING, "Seafaring"),
        (TAG_DUNGEONEERING, "Dungeoneering"),
    ];
    TABLE.iter().find(|&&(h, _)| h == hash).map(|&(_, name)| name)
}

// ---------------------------------------------------------------------------
// Helper: look up a tag value in sorted parallel arrays (O(log n))
// ---------------------------------------------------------------------------

fn lookup_tag(behavior_profile: &[(u32, f32)], tag_hash: u32) -> f32 {
    match behavior_profile.binary_search_by_key(&tag_hash, |&(t, _)| t) {
        Ok(idx) => behavior_profile[idx].1,
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
    fn match_classes(&self, behavior_profile: &[(u32, f32)]) -> Vec<ClassMatch> {
        let mut matches = Vec::new();

        for tmpl in TEMPLATES {
            // Check all requirements are met.
            let mut qualified = true;
            for &(req_tag, min_val) in tmpl.requirements {
                if lookup_tag(behavior_profile, req_tag) < min_val {
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
                let val = lookup_tag(behavior_profile, score_tag);
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

            // Normalize: raw score is a dot product (can be thousands).
            // Compress to a 0–1 alignment quality using sigmoid-like scaling.
            // score=100 → 0.5, score=500 → 0.83, score=1000 → 0.91
            let normalized_score = score / (score + 100.0);

            if normalized_score < SCORE_THRESHOLD {
                continue;
            }
            let score = normalized_score;

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
        _behavior_profile: &[(u32, f32)],
        _seed: u64,
    ) -> Option<ClassDef> {
        // Placeholder: no procedural class generation yet.
        None
    }
}

// ---------------------------------------------------------------------------
// RegistryClassGenerator — reads class templates from the data-driven registry
// ---------------------------------------------------------------------------

use std::sync::Arc;

/// Class generator that reads templates from the data-driven registry.
pub struct RegistryClassGenerator {
    registry: Arc<super::registry::Registry>,
}

impl RegistryClassGenerator {
    pub fn new(registry: Arc<super::registry::Registry>) -> Self {
        Self { registry }
    }
}

impl ClassGenerator for RegistryClassGenerator {
    fn match_classes(&self, behavior_profile: &[(u32, f32)]) -> Vec<ClassMatch> {
        let mut matches = Vec::new();

        for def in self.registry.classes.values() {
            // Check all behavior requirements are met.
            let mut qualified = true;
            for (tag_name, &min_val) in &def.requirements.behavior {
                let tag_hash = tag(tag_name.as_bytes());
                if lookup_tag(behavior_profile, tag_hash) < min_val {
                    qualified = false;
                    break;
                }
            }
            if !qualified {
                continue;
            }

            // Compute weighted dot product using score_weights.
            let mut score = 0.0f32;
            let mut best_tag_hash = 0u32;
            let mut best_weighted = 0.0f32;
            let mut second_tag_hash = 0u32;
            let mut second_weighted = 0.0f32;

            if def.score_weights.is_empty() {
                // No score_weights: use tags with equal weights.
                let w = if def.tags.is_empty() { 1.0 } else { 1.0 / def.tags.len() as f32 };
                for tag_name in &def.tags {
                    let tag_hash = tag(tag_name.as_bytes());
                    let val = lookup_tag(behavior_profile, tag_hash);
                    let weighted = val * w;
                    score += weighted;
                    if weighted > best_weighted {
                        second_tag_hash = best_tag_hash;
                        second_weighted = best_weighted;
                        best_tag_hash = tag_hash;
                        best_weighted = weighted;
                    } else if weighted > second_weighted {
                        second_tag_hash = tag_hash;
                        second_weighted = weighted;
                    }
                }
            } else {
                for (tag_name, &weight) in &def.score_weights {
                    let tag_hash = tag(tag_name.as_bytes());
                    let val = lookup_tag(behavior_profile, tag_hash);
                    let weighted = val * weight;
                    score += weighted;
                    if weighted > best_weighted {
                        second_tag_hash = best_tag_hash;
                        second_weighted = best_weighted;
                        best_tag_hash = tag_hash;
                        best_weighted = weighted;
                    } else if weighted > second_weighted {
                        second_tag_hash = tag_hash;
                        second_weighted = weighted;
                    }
                }
            }

            // Normalize with sigmoid: raw/(raw+100).
            let normalized_score = score / (score + 100.0);
            if normalized_score < SCORE_THRESHOLD {
                continue;
            }

            // Variant naming.
            let display_name = if best_weighted > 0.0
                && second_weighted > best_weighted * 0.8
                && second_tag_hash != 0
            {
                if let Some(suffix) = tag_display_name(second_tag_hash) {
                    format!("{} of {}", def.name, suffix)
                } else {
                    def.name.clone()
                }
            } else {
                def.name.clone()
            };

            matches.push(ClassMatch {
                class_name_hash: tag(def.name.as_bytes()),
                display_name,
                score: normalized_score,
            });
        }

        matches
    }

    fn generate_unique_class(
        &self,
        _behavior_profile: &[(u32, f32)],
        _seed: u64,
    ) -> Option<ClassDef> {
        None
    }
}

// ---------------------------------------------------------------------------
// DefaultAbilityGenerator
// ---------------------------------------------------------------------------

/// Ability generator that walks the DSL grammar tree with archetype-conditioned
/// probability distributions, then scores the result with the grammar-space
/// quality metric. Generates N candidates and keeps the best one.
pub struct DefaultAbilityGenerator {
    candidates: usize,
}

impl DefaultAbilityGenerator {
    pub fn new() -> Self {
        Self { candidates: 1 }
    }
}

impl AbilityGenerator for DefaultAbilityGenerator {
    fn generate_ability(
        &self,
        _class_name_hash: u32,
        archetype: &str,
        tier: u32,
        _behavior_profile: &[(u32, f32)],
        seed: u64,
    ) -> GeneratedAbility {
        let history = std::collections::HashMap::new();
        let mut best_dsl = String::new();
        let mut best_name = String::new();
        let mut best_score = -1.0f32;
        let mut best_passive = false;

        for i in 0..self.candidates {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            let (ability, dsl) = super::ability_gen::generate_tiered_ability(
                archetype, tier, &mut rng, &history,
            );

            // Score via grammar space encode → quality metric
            let score = super::grammar_space::encode(&dsl)
                .map(|v| super::ability_quality::score_ability(&v))
                .unwrap_or(0.0);

            if score > best_score {
                best_score = score;
                best_name = ability.name.clone();
                best_dsl = dsl;
                best_passive = ability.cast_time_ms == 0 && ability.cooldown_ms == 0;
            }
        }

        // Fallback if all candidates failed to encode
        if best_dsl.is_empty() {
            let mut rng = seed;
            let (ability, dsl) = super::ability_gen::generate_tiered_ability(
                archetype, tier, &mut rng, &history,
            );
            best_name = ability.name;
            best_dsl = dsl;
        }

        GeneratedAbility {
            name: best_name,
            dsl_text: best_dsl,
            is_passive: best_passive,
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

    fn make_profile(pairs: &[(&[u8], f32)]) -> Vec<(u32, f32)> {
        let mut entries: Vec<(u32, f32)> = pairs.iter().map(|(name, val)| (tag(name), *val)).collect();
        entries.sort_by_key(|&(h, _)| h);
        entries
    }

    #[test]
    fn warrior_matches_with_high_melee() {
        let profile = make_profile(&[
            (b"melee", 200.0),
            (b"defense", 80.0),
            (b"endurance", 60.0),
            (b"combat", 40.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&profile);
        assert!(
            matches.iter().any(|m| m.display_name.starts_with("Warrior")),
            "Expected Warrior class match, got: {:?}",
            matches.iter().map(|m| &m.display_name).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn no_match_below_requirements() {
        let profile = make_profile(&[
            (b"melee", 50.0),
            (b"defense", 10.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&profile);
        assert!(
            matches.iter().all(|m| !m.display_name.starts_with("Warrior")),
            "Warrior should not match with melee=50",
        );
    }

    #[test]
    fn variant_naming_when_secondary_high() {
        let profile = make_profile(&[
            (b"crafting", 200.0),
            (b"smithing", 200.0),
            (b"labor", 50.0),
            (b"endurance", 50.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&profile);
        let artisan = matches.iter().find(|m| m.class_name_hash == tag(b"Artisan"));
        assert!(artisan.is_some(), "Artisan should match");
        let name = &artisan.unwrap().display_name;
        assert!(name.contains(" of "), "Expected variant name, got: {}", name);
    }

    #[test]
    fn multiple_classes_can_match() {
        let profile = make_profile(&[
            (b"melee", 200.0),
            (b"defense", 200.0),
            (b"endurance", 200.0),
            (b"combat", 100.0),
            (b"resilience", 100.0),
        ]);
        let gen = DefaultClassGenerator::new();
        let matches = gen.match_classes(&profile);
        let names: Vec<_> = matches.iter().map(|m| m.display_name.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("Warrior")), "Missing Warrior in {:?}", names);
        assert!(names.iter().any(|n| n.starts_with("Guardian")), "Missing Guardian in {:?}", names);
    }

    #[test]
    fn ability_generator_produces_valid_dsl() {
        let gen = DefaultAbilityGenerator::new();
        let a1 = gen.generate_ability(0, "knight", 1, &[], 42);
        let a3 = gen.generate_ability(0, "mage", 3, &[], 99);
        // Both should produce non-empty DSL that contains "ability" keyword
        assert!(!a1.dsl_text.is_empty(), "Tier 1 should produce DSL");
        assert!(!a3.dsl_text.is_empty(), "Tier 3 should produce DSL");
        assert!(a1.dsl_text.contains("ability"), "DSL should contain ability block: {}", a1.dsl_text);
        assert!(a3.dsl_text.contains("ability"), "DSL should contain ability block: {}", a3.dsl_text);
        assert!(!a1.name.is_empty(), "Should have a name");
    }
}

// ---------------------------------------------------------------------------
// Public API: class→building relevance for production quality (Phase 6)
// ---------------------------------------------------------------------------

/// Compute relevance of a class (by name_hash) to a set of building production tags.
/// Returns [0.0, 1.0] — dot product of class score_tags vs building tags, normalized.
pub fn class_building_relevance(class_name_hash: u32, building_tags: &[(u32, f32)]) -> f32 {
    if building_tags.is_empty() { return 0.0; }

    let tmpl = match TEMPLATES.iter().find(|t| t.name_hash == class_name_hash) {
        Some(t) => t,
        None => return 0.0, // runtime-generated class — fall back to 0
    };

    let mut dot = 0.0f32;
    for &(btag, bweight) in building_tags {
        for &(ctag, cweight) in tmpl.score_tags {
            if ctag == btag {
                dot += bweight * cweight;
            }
        }
    }
    // Normalize: max possible dot is ~1.0 (both sides sum to ~1.0).
    dot.min(1.0)
}
