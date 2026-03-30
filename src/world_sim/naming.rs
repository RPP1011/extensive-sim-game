//! Procedural class/ability naming + LFM naming queue.
//!
//! Generates flavorful names from behavior tags and settlement context.
//! Names are procedural by default. An LFM naming queue allows async
//! replacement with higher-quality model-generated names.

use std::collections::HashMap;

use super::state::{tag, Entity, EntityKind};

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
    behavior_profile: &[(u32, f32)],
    seed: u64,
) -> String {
    if behavior_profile.is_empty() {
        return base_name.to_string();
    }

    // Find the tag with the highest value that has a flavor adjective.
    let mut best_tag = 0u32;
    let mut best_val = 0.0f32;
    for &(tag_hash, val) in behavior_profile {
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
// Personal name generation — deterministic fantasy names from entity ID + seed
// ---------------------------------------------------------------------------

static PREFIXES: &[&str] = &[
    "Kor", "Thes", "Bre", "Val", "Mor", "Eld", "Gar", "Fen", "Ash", "Dra",
    "Kel", "Tor", "Lyn", "Har", "Sar", "Wren", "Zev", "Nol", "Cael", "Iri",
    "Oth", "Rav", "Sel", "Tam", "Ul", "Vex", "Yar", "Dun", "Grim", "Rhi",
];

static SUFFIXES: &[&str] = &[
    "rin", "sa", "gan", "ek", "don", "wen", "ax", "is", "ra", "os",
    "ia", "en", "ul", "ith", "ard", "ley", "on", "us", "an", "ael",
    "ik", "or", "eth", "ius", "ara", "olt", "ynn", "ash", "mir", "tek",
];

/// Generate a deterministic fantasy personal name from an entity ID and world seed.
///
/// Uses LCG-style hashing to pick from prefix/suffix syllable tables, producing
/// names like "Korrin", "Thessa", "Bregan", etc.
pub fn generate_personal_name(entity_id: u32, seed: u64) -> String {
    let h = (entity_id as u64).wrapping_mul(6364136223846793005u64).wrapping_add(seed);
    let prefix_idx = (h as usize) % PREFIXES.len();
    let suffix_idx = ((h >> 16) as usize) % SUFFIXES.len();
    format!("{}{}", PREFIXES[prefix_idx], SUFFIXES[suffix_idx])
}

/// Generate a settlement name from a seed.
pub fn generate_settlement_name_from_seed(seed: u64) -> String {
    const PREFIXES: &[&str] = &[
        "Iron", "Oak", "Grey", "Black", "Red", "South", "North", "East", "West",
        "Old", "New", "High", "Low", "Silver", "Gold", "Blue", "Green", "White",
    ];
    const SUFFIXES: &[&str] = &[
        "haven", "port", "wick", "dale", "field", "moor", "crest", "hollow",
        "bridge", "watch", "keep", "grove", "vale", "ford", "stead", "gate",
    ];
    let h = seed.wrapping_mul(6364136223846793005);
    let p = (h as usize) % PREFIXES.len();
    let s = ((h >> 20) as usize) % SUFFIXES.len();
    format!("{}{}", PREFIXES[p], SUFFIXES[s])
}

/// Return a display name for an entity. NPCs use their personal name,
/// monsters get evocative names based on level, and buildings/other types
/// use "Entity #ID".
pub fn entity_display_name(entity: &Entity) -> String {
    match entity.kind {
        EntityKind::Npc => {
            if let Some(npc) = entity.npc.as_ref() {
                if !npc.name.is_empty() {
                    return npc.name.clone();
                }
            }
            format!("Entity #{}", entity.id)
        }
        EntityKind::Monster => monster_display_name(entity.id, entity.level),
        _ => format!("Entity #{}", entity.id),
    }
}

// ---------------------------------------------------------------------------
// Monster name generation — deterministic evocative names from ID + level
// ---------------------------------------------------------------------------

static MONSTER_ADJECTIVES: &[&str] = &[
    "Rabid", "Feral", "Diseased", "Frenzied", "Lurking",
    "Twisted", "Hollow", "Bloated", "Starving", "Withered",
    "Venomous", "Rotting", "Savage", "Cursed", "Blighted",
];

static MONSTER_CREATURES: &[&str] = &[
    "Wolf", "Spider", "Troll", "Wyrm", "Ghoul",
    "Ogre", "Drake", "Warg", "Fiend", "Crawler",
    "Stalker", "Brute", "Horror", "Shade", "Beast",
];

static MONSTER_TITLES: &[&str] = &[
    "Iron", "Shadow", "Blood", "Stone", "Bone",
    "Dread", "Night", "Rust", "Ash", "Doom",
    "Blight", "Plague", "Rot", "Storm", "Frost",
];

static MONSTER_NAMES: &[&str] = &[
    "Grath", "Skarn", "Vex", "Thok", "Murg",
    "Krell", "Drex", "Zul", "Nyx", "Gor",
    "Blight", "Char", "Skar", "Dusk", "Kraven",
];

/// Generate a deterministic evocative monster name from entity ID and level.
///
/// - Level 1-5:  "{adjective} {creature}" (e.g., "a Rabid Wolf")
/// - Level 6-15: "{title} {creature}" (e.g., "the Iron Troll")
/// - Level 16+:  "{name} the {title}" (e.g., "Grath the Devourer")
fn monster_display_name(id: u32, level: u32) -> String {
    // LCG-style deterministic hash mixing ID and level tier for variety across seeds.
    let tier_salt = if level >= 16 { 2 } else if level >= 6 { 1 } else { 0 };
    let h = (id as u64).wrapping_mul(6364136223846793005u64)
        .wrapping_add(1442695040888963407u64)
        .wrapping_add(tier_salt * 2862933555777941757);

    let creature_idx = (h as usize) % MONSTER_CREATURES.len();
    let adj_idx = ((h >> 16) as usize) % MONSTER_ADJECTIVES.len();
    let title_idx = ((h >> 32) as usize) % MONSTER_TITLES.len();
    let name_idx = ((h >> 48) as usize) % MONSTER_NAMES.len();

    match level {
        0..=5 => format!("a {} {}", MONSTER_ADJECTIVES[adj_idx], MONSTER_CREATURES[creature_idx]),
        6..=15 => format!("the {} {}", MONSTER_TITLES[title_idx], MONSTER_CREATURES[creature_idx]),
        _ => format!("{} the {}", MONSTER_NAMES[name_idx], MONSTER_TITLES[title_idx]),
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
