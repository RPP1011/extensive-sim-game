//! Procedural class/ability naming + LFM naming queue.
//!
//! Generates flavorful names from behavior tags and settlement context.
//! Names are procedural by default. An LFM naming queue allows async
//! replacement with higher-quality model-generated names.

use std::collections::HashMap;

use super::state::tag;

// ---------------------------------------------------------------------------
// Procedural naming from behavior tags
// ---------------------------------------------------------------------------

/// Adjective pools keyed by dominant tag hash.
struct TagFlavor {
    tag_hash: u32,
    adjectives: &'static [&'static str],
}

static TAG_FLAVORS: &[TagFlavor] = &[
    TagFlavor { tag_hash: tag(b"melee"),      adjectives: &["Battle-Forged", "Iron-Fisted", "Steel-Clad", "War-Scarred", "Brawling"] },
    TagFlavor { tag_hash: tag(b"ranged"),     adjectives: &["Sharp-Eyed", "Wind-Footed", "Far-Sighted", "Hawk-Born", "True-Aimed"] },
    TagFlavor { tag_hash: tag(b"combat"),     adjectives: &["Blooded", "Battle-Tested", "War-Hardened", "Grim", "Relentless"] },
    TagFlavor { tag_hash: tag(b"defense"),    adjectives: &["Stalwart", "Unyielding", "Iron-Willed", "Shieldbearing", "Steadfast"] },
    TagFlavor { tag_hash: tag(b"tactics"),    adjectives: &["Cunning", "Strategic", "Sharp-Minded", "Calculating", "Methodical"] },
    TagFlavor { tag_hash: tag(b"mining"),     adjectives: &["Stoneheart", "Deep-Delving", "Ore-Touched", "Tunnel-Born", "Ember Vein"] },
    TagFlavor { tag_hash: tag(b"smithing"),   adjectives: &["Forge-Tempered", "Anvil-Born", "Hammer-Sworn", "Fire-Kissed", "Steel-Souled"] },
    TagFlavor { tag_hash: tag(b"crafting"),   adjectives: &["Deft-Handed", "Master-Wrought", "Fine-Tooled", "True-Crafted", "Guild-Marked"] },
    TagFlavor { tag_hash: tag(b"enchantment"), adjectives: &["Rune-Touched", "Glyph-Scarred", "Spell-Forged", "Arcane", "Ether-Bound"] },
    TagFlavor { tag_hash: tag(b"alchemy"),    adjectives: &["Vial-Keeper", "Essence-Stained", "Brew-Wise", "Tincture-Born", "Flux-Touched"] },
    TagFlavor { tag_hash: tag(b"trade"),      adjectives: &["Silver-Tongued", "Coin-Wise", "Far-Traded", "Market-Sharp", "Deal-Sworn"] },
    TagFlavor { tag_hash: tag(b"diplomacy"),  adjectives: &["Peace-Weaver", "Word-Sworn", "Treaty-Bound", "Crown-Spoken", "Court-Wise"] },
    TagFlavor { tag_hash: tag(b"leadership"), adjectives: &["Banner-Born", "Rally-Voiced", "Command-Forged", "Oath-Keeper", "Vanguard"] },
    TagFlavor { tag_hash: tag(b"negotiation"), adjectives: &["Shrewd", "Bargain-Struck", "Price-Wise", "Barter-Born", "Deal-Maker"] },
    TagFlavor { tag_hash: tag(b"research"),   adjectives: &["Lore-Deep", "Tome-Bound", "Ink-Stained", "Knowledge-Hungry", "Archive-Keeper"] },
    TagFlavor { tag_hash: tag(b"lore"),       adjectives: &["Legend-Keeper", "Saga-Teller", "Memory-Woven", "Truth-Seeker", "Chronicle-Born"] },
    TagFlavor { tag_hash: tag(b"medicine"),   adjectives: &["Gentle-Handed", "Life-Sworn", "Wound-Mender", "Balm-Bearer", "Mercy-Touched"] },
    TagFlavor { tag_hash: tag(b"herbalism"),  adjectives: &["Root-Wise", "Green-Fingered", "Bloom-Keeper", "Leaf-Touched", "Garden-Born"] },
    TagFlavor { tag_hash: tag(b"endurance"),  adjectives: &["Iron-Willed", "Stone-Souled", "Never-Yielding", "Tireless", "Unbreaking"] },
    TagFlavor { tag_hash: tag(b"resilience"), adjectives: &["Scar-Worn", "Twice-Risen", "Storm-Weathered", "Phoenix-Born", "Hard-Tempered"] },
    TagFlavor { tag_hash: tag(b"stealth"),    adjectives: &["Shadow-Stepped", "Unseen", "Ghost-Walker", "Night-Cloaked", "Whisper-Footed"] },
    TagFlavor { tag_hash: tag(b"survival"),   adjectives: &["Frontier", "Wild-Born", "Trail-Hardened", "Outland", "Wilderness"] },
    TagFlavor { tag_hash: tag(b"awareness"),  adjectives: &["Keen-Sensed", "Ever-Watchful", "Hawk-Eyed", "Alert", "Sentinel"] },
    TagFlavor { tag_hash: tag(b"faith"),      adjectives: &["Devout", "Blessed", "Sanctified", "Holy", "Pious"] },
    TagFlavor { tag_hash: tag(b"ritual"),     adjectives: &["Rite-Keeper", "Circle-Bound", "Chant-Woven", "Ceremony-Wise", "Oath-Spoken"] },
    TagFlavor { tag_hash: tag(b"labor"),      adjectives: &["Toil-Worn", "Steady-Handed", "Work-Hardened", "Dutiful", "Tireless"] },
    TagFlavor { tag_hash: tag(b"teaching"),   adjectives: &["Sage", "Patient", "Guide-Born", "Lesson-Sworn", "Wisdom-Keeper"] },
    TagFlavor { tag_hash: tag(b"farming"),    adjectives: &["Earth-Touched", "Harvest-Born", "Field-Sworn", "Grain-Keeper", "Soil-Wise"] },
    TagFlavor { tag_hash: tag(b"woodwork"),   adjectives: &["Oak-Hearted", "Timber-Born", "Grain-Reader", "Forest-Sworn", "Bark-Scarred"] },
    TagFlavor { tag_hash: tag(b"exploration"), adjectives: &["Wandering", "Pathfinder", "Horizon-Chaser", "Uncharted", "Wayfarer"] },
    TagFlavor { tag_hash: tag(b"deception"),  adjectives: &["Mask-Worn", "Two-Faced", "Veil-Dancer", "Lie-Smith", "Shadow-Spoken"] },
    TagFlavor { tag_hash: tag(b"discipline"), adjectives: &["Tempered", "Ordered", "Rule-Bound", "Focused", "Measured"] },
    TagFlavor { tag_hash: tag(b"navigation"), adjectives: &["Star-Guided", "Path-Wise", "Chart-Reader", "Compass-Born", "Way-Finder"] },
];

/// Generate a procedural class name from the NPC's behavior profile.
///
/// Format: "[Adjective] [BaseClassName]"
/// Adjective comes from the NPC's strongest non-class tag.
/// E.g., a Miner with high resilience → "Scar-Worn Miner"
///       a Farmer with high faith → "Devout Farmer"
pub fn procedural_class_name(
    base_name: &str,
    behavior_tags: &[u32],
    behavior_values: &[f32],
    seed: u64,
) -> String {
    if behavior_tags.is_empty() {
        return base_name.to_string();
    }

    // Find the tag with the highest value that has a flavor adjective.
    let mut best_tag = 0u32;
    let mut best_val = 0.0f32;
    for (i, &tag_hash) in behavior_tags.iter().enumerate() {
        let val = behavior_values[i];
        if val > best_val && TAG_FLAVORS.iter().any(|f| f.tag_hash == tag_hash) {
            best_tag = tag_hash;
            best_val = val;
        }
    }

    if best_tag == 0 {
        return base_name.to_string();
    }

    // Pick adjective deterministically from seed.
    if let Some(flavor) = TAG_FLAVORS.iter().find(|f| f.tag_hash == best_tag) {
        let idx = (seed as usize) % flavor.adjectives.len();
        format!("{} {}", flavor.adjectives[idx], base_name)
    } else {
        base_name.to_string()
    }
}

// ---------------------------------------------------------------------------
// LFM Naming Queue
// ---------------------------------------------------------------------------

/// A pending name request for the LFM model.
#[derive(Debug, Clone)]
pub struct NamingRequest {
    /// Entity ID that needs the name.
    pub entity_id: u32,
    /// Class slot index on the entity.
    pub class_index: usize,
    /// Base class name (e.g., "Miner").
    pub base_name: String,
    /// Top 5 behavior tags with values for context.
    pub top_tags: Vec<(String, f32)>,
    /// Settlement name for context.
    pub settlement_name: String,
    /// NPC archetype for context.
    pub archetype: String,
}

/// Async naming queue + resolved name cache.
pub struct NamingService {
    /// Pending requests waiting to be sent to LFM.
    pub queue: Vec<NamingRequest>,
    /// Resolved names: (entity_id, class_index) → display_name.
    /// Checked each tick and written to ClassSlot.display_name.
    pub resolved: HashMap<(u32, usize), String>,
}

impl NamingService {
    pub fn new() -> Self {
        NamingService {
            queue: Vec::with_capacity(64),
            resolved: HashMap::new(),
        }
    }

    /// Queue a name request. Will be resolved by an external caller
    /// (e.g., LFM batch inference) and placed into `resolved`.
    pub fn request_name(&mut self, req: NamingRequest) {
        // Don't duplicate requests.
        let key = (req.entity_id, req.class_index);
        if self.resolved.contains_key(&key) { return; }
        if self.queue.iter().any(|r| r.entity_id == req.entity_id && r.class_index == req.class_index) {
            return;
        }
        self.queue.push(req);
    }

    /// Drain resolved names into entity ClassSlots.
    /// Returns number of names applied.
    pub fn apply_resolved(&mut self, entities: &mut [super::state::Entity]) -> usize {
        let mut applied = 0;
        for (&(entity_id, class_idx), name) in &self.resolved {
            if let Some(entity) = entities.iter_mut().find(|e| e.id == entity_id) {
                if let Some(npc) = entity.npc.as_mut() {
                    if class_idx < npc.classes.len() && npc.classes[class_idx].display_name.is_empty() {
                        npc.classes[class_idx].display_name = name.clone();
                        applied += 1;
                    }
                }
            }
        }
        self.resolved.clear();
        applied
    }
}
