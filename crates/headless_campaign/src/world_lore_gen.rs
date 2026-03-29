//! Procedural world lore generator for headless campaigns.
//!
//! Each campaign gets unique narrative context generated deterministically
//! from the campaign RNG at initialization time. All generation uses
//! template banks and the campaign's LCG — no external randomness.

use super::state::{
    lcg_f32, lcg_next, CampaignState, DiplomaticStance,
};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Lore structs
// ---------------------------------------------------------------------------

/// Top-level lore context for a campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldLore {
    /// Campaign-level name, e.g. "The Age of Iron".
    pub campaign_name: String,
    /// One-line era context.
    pub era_description: String,
    /// Why the guild was founded.
    pub guild_origin: String,
    /// Per-faction backstory.
    pub faction_lore: Vec<FactionLore>,
    /// Per-region history/legends.
    pub region_lore: Vec<RegionLore>,
    /// What the endgame crisis means narratively.
    pub crisis_lore: CrisisLore,
}

/// Narrative backstory for a single faction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionLore {
    /// Which faction this describes.
    pub faction_id: usize,
    /// How this faction came to be.
    pub origin_story: String,
    /// Name of the faction's current leader.
    pub leader_name: String,
    /// Personality descriptor for the leader.
    pub leader_personality: String,
    /// Faction IDs this faction bears historical grudges against.
    pub historical_grudges: Vec<usize>,
}

/// Narrative context for a single region.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionLore {
    /// Which region this describes.
    pub region_id: usize,
    /// A legend associated with the region.
    pub legend: String,
    /// A notable landmark in the region.
    pub notable_landmark: String,
    /// A historical event that occurred here.
    pub historical_event: String,
}

/// Narrative identity for the endgame crisis.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CrisisLore {
    /// Who the Sleeping King is, narratively.
    pub sleeping_king_identity: String,
    /// Where the Corruption originated from.
    pub blight_origin: String,
    /// What the Breach looks like.
    pub breach_description: String,
}

// ---------------------------------------------------------------------------
// Template banks
// ---------------------------------------------------------------------------

const ERA_NOUNS: &[&str] = &[
    "Iron", "Ashes", "Thorns", "Silence", "Embers", "Ruin", "Splendor",
    "Bones", "Twilight", "Storms", "Frost", "Shadows", "Flame", "Dust",
    "Glass", "Serpents", "Ravens", "Stone", "Chains", "Stars",
];

const ERA_ADJECTIVES: &[&str] = &[
    "Twilight", "Broken", "Crimson", "Last", "Dying", "Hollow",
    "Sunken", "Bitter", "Waning", "Forgotten", "Silent", "Ashen",
    "Golden", "Iron", "Silver",
];

const ERA_ABSTRACT_NOUNS: &[&str] = &[
    "Compact", "Accord", "Dominion", "Covenant", "Chronicle",
    "Reckoning", "Convergence", "Eclipse", "Requiem", "Vigil",
    "Crusade", "Exodus", "Mandate", "Pact", "Tribunal",
];

const GUILD_LEADERS: &[&str] = &[
    "Captain Aldric", "the scholar Venn", "Marshal Thessa",
    "an unknown benefactor", "three exiled knights", "the war-priest Ondra",
    "a merchant consortium", "the retired champion Korr",
    "Dame Elspeth of the Grey", "the wanderer Fael", "Archon Seris",
    "Lord Maren the Younger", "the surgeon Yulda", "Warden Brask",
    "the refugees of Oldwall",
];

const GUILD_EVENTS: &[&str] = &[
    "the fall of the old kingdom", "the second monster surge",
    "a devastating plague", "the betrayal at Thorngate",
    "the collapse of the Merchant League", "the siege of Whitespire",
    "a terrible famine", "the dragon's last attack",
    "the dissolution of the royal guard", "the night of red stars",
    "the flooding of the lowlands", "the sundering of the roads",
    "the exile of the mage council", "the burning of the great library",
    "the vanishing of the old gods",
];

const GUILD_MISSIONS: &[&str] = &[
    "protect the frontier settlements", "reclaim lost territory",
    "hunt the creatures of the deep wilds", "guard the trade routes",
    "train a new generation of defenders", "broker peace between warring factions",
    "investigate the spreading corruption", "serve as neutral arbiters",
    "preserve what knowledge remains", "stand against the coming darkness",
    "rebuild what was lost", "unite the fractured provinces",
    "provide refuge to the displaced", "keep the old roads open",
    "ensure no power goes unchecked",
];

const LEADER_FIRST_NAMES: &[&str] = &[
    "Aldric", "Brenna", "Corvus", "Dara", "Elric", "Freya", "Gareth",
    "Hilde", "Ivor", "Jessa", "Kevan", "Lira", "Magnus", "Nessa",
    "Orin", "Petra", "Quinn", "Rhiannon", "Soren", "Thalia",
    "Ulric", "Vera", "Wynn", "Xara", "Yoren", "Zara",
];

const LEADER_SURNAMES: &[&str] = &[
    "Blackthorn", "Ironheart", "Ashford", "Stormcrow", "Duskwalker",
    "Frostbane", "Hearthstone", "Nightfall", "Oakenshield", "Ravencrest",
    "Silvervein", "Thornwall", "Voidkeeper", "Windhelm", "Goleli",
    "Redmane", "Deepwell", "Farstrider", "Grimhold", "Longbow",
];

const FRIENDLY_PERSONALITIES: &[&str] = &[
    "a cautious diplomat who values stability",
    "a pragmatic trader with a generous streak",
    "a weary peacekeeper who has seen too many wars",
    "a visionary reformer building alliances",
    "an aging idealist clinging to honor",
];

const NEUTRAL_PERSONALITIES: &[&str] = &[
    "a calculating opportunist who sides with the strong",
    "a reclusive scholar concerned only with knowledge",
    "a stoic administrator who follows the letter of law",
    "an indifferent merchant lord focused on profit",
    "a young inheritor uncertain of their path",
];

const HOSTILE_PERSONALITIES: &[&str] = &[
    "a bitter warlord nursing old grievances",
    "a zealot who sees the guild as a threat to order",
    "a cunning expansionist with imperial ambitions",
    "a paranoid strategist who trusts no one",
    "a ruthless survivor who respects only strength",
];

const CREATURES: &[&str] = &[
    "dragon", "wyrm", "giant", "troll king", "shadow beast",
    "iron golem", "dire wolf", "basilisk", "lich", "hydra",
    "manticore", "griffon", "wyvern", "chimera", "kraken",
];

const BATTLE_NAMES: &[&str] = &[
    "Battle of the Red Ford", "Siege of the Black Tower",
    "Night of Broken Shields", "Rout at Cinder Pass",
    "Skirmish of the Three Bridges", "Last Stand at the Old Wall",
    "Burning of the Grove", "Charge at Thornfield",
    "Fall of the Iron Gate", "Storm of the High Keep",
    "Massacre at Dusk Hollow", "Retreat from Ashen Dale",
];

const LANDMARKS: &[&str] = &[
    "the Shattered Obelisk", "an ancient stone circle",
    "the Weeping Statue", "a collapsed mine entrance",
    "the Hanging Gardens", "a petrified forest",
    "the Sunken Temple", "a rusted war memorial",
    "the Crystal Cavern", "an overgrown amphitheater",
    "the Bone Arch", "a crumbling watchtower",
    "the Mirror Lake", "a sealed vault door",
    "the Whispering Stones", "an abandoned observatory",
];

const HISTORICAL_EVENTS: &[&str] = &[
    "a great flood reshaped the landscape two centuries ago",
    "an empire fell here, leaving only ruins",
    "a magical catastrophe scarred the land generations past",
    "invaders burned every settlement to the ground a century ago",
    "a plague swept through, and the region was quarantined for decades",
    "a dragon roosted here until driven off by an unknown hero",
    "the old road was built through here by a now-forgotten civilization",
    "this was the breadbasket of a kingdom that no longer exists",
    "a meteorite struck the valley, leaving strange minerals behind",
    "two armies clashed here and neither side won",
    "a holy order once maintained a fortress here before their dissolution",
    "the original inhabitants vanished overnight without explanation",
];

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Pick a random element from a slice using LCG.
fn pick<'a>(rng: &mut u64, bank: &'a [&str]) -> &'a str {
    let idx = (lcg_next(rng) as usize) % bank.len();
    bank[idx]
}

/// Generate a campaign name from template patterns.
fn gen_campaign_name(rng: &mut u64) -> String {
    let pattern = lcg_next(rng) % 3;
    match pattern {
        0 => format!("The Age of {}", pick(rng, ERA_NOUNS)),
        1 => format!("The {} {}", pick(rng, ERA_ADJECTIVES), pick(rng, ERA_ABSTRACT_NOUNS)),
        _ => format!("An Era of {}", pick(rng, ERA_NOUNS)),
    }
}

/// Generate a one-line era description.
fn gen_era_description(rng: &mut u64, campaign_name: &str) -> String {
    let templates: &[&str] = &[
        "The old powers have crumbled and new ones rise to fill the void.",
        "War has left the land scarred, and the survivors struggle to rebuild.",
        "Magic fades from the world as something ancient stirs beneath.",
        "Trade routes collapse as monsters grow bolder beyond the walls.",
        "A fragile peace holds, but every faction prepares for what comes next.",
        "The frontier expands as desperate folk push into untamed wilderness.",
        "Prophecies speak of a reckoning, and the wise take heed.",
        "The guilds are the last institution the common folk can trust.",
    ];
    let desc = pick(rng, templates);
    format!("{campaign_name}: {desc}")
}

/// Generate a guild origin story.
fn gen_guild_origin(rng: &mut u64) -> String {
    let leader = pick(rng, GUILD_LEADERS);
    let event = pick(rng, GUILD_EVENTS);
    let mission = pick(rng, GUILD_MISSIONS);
    format!("Founded by {leader} after {event} to {mission}.")
}

/// Generate a faction leader name.
fn gen_leader_name(rng: &mut u64) -> String {
    let first = pick(rng, LEADER_FIRST_NAMES);
    let last = pick(rng, LEADER_SURNAMES);
    format!("{first} {last}")
}

/// Pick a personality description based on diplomatic stance.
fn gen_leader_personality(rng: &mut u64, stance: DiplomaticStance) -> String {
    let bank = match stance {
        DiplomaticStance::Friendly | DiplomaticStance::Coalition => FRIENDLY_PERSONALITIES,
        DiplomaticStance::Neutral => NEUTRAL_PERSONALITIES,
        DiplomaticStance::Hostile | DiplomaticStance::AtWar => HOSTILE_PERSONALITIES,
    };
    pick(rng, bank).to_string()
}

/// Derive historical grudges from diplomacy matrix.
fn derive_grudges(faction_id: usize, relations: &[Vec<i32>]) -> Vec<usize> {
    if faction_id >= relations.len() {
        return Vec::new();
    }
    relations[faction_id]
        .iter()
        .enumerate()
        .filter(|&(other, &score)| other != faction_id && score < -10)
        .map(|(other, _)| other)
        .collect()
}

/// Generate lore for a single faction.
fn gen_faction_lore(
    rng: &mut u64,
    faction_id: usize,
    stance: DiplomaticStance,
    faction_name: &str,
    relations: &[Vec<i32>],
) -> FactionLore {
    let leader_name = gen_leader_name(rng);
    let leader_personality = gen_leader_personality(rng, stance);
    let grudges = derive_grudges(faction_id, relations);

    let origin_templates: &[&str] = &[
        "rose from the ashes of a collapsed duchy",
        "was forged in the crucible of the border wars",
        "traces its lineage to a band of mercenary captains",
        "grew from a prosperous trading post into a regional power",
        "was founded by refugees fleeing a far-off catastrophe",
        "emerged when local warlords united under a single banner",
        "began as a religious order that turned to secular rule",
        "coalesced around a mining consortium that outgrew its charter",
    ];

    let origin = pick(rng, origin_templates);
    let origin_story = format!("{faction_name} {origin}.");

    FactionLore {
        faction_id,
        origin_story,
        leader_name,
        leader_personality,
        historical_grudges: grudges,
    }
}

/// Generate lore for a single region.
fn gen_region_lore(rng: &mut u64, region_id: usize, region_name: &str) -> RegionLore {
    let creature = pick(rng, CREATURES);
    let battle = pick(rng, BATTLE_NAMES);
    let landmark = pick(rng, LANDMARKS);
    let event = pick(rng, HISTORICAL_EVENTS);

    let legend_pattern = lcg_next(rng) % 3;
    let legend = match legend_pattern {
        0 => format!("They say a {creature} was slain in the heart of {region_name} long ago."),
        1 => format!(
            "Legends tell of the {battle}, which shaped {region_name}'s borders forever."
        ),
        _ => format!(
            "The people of {region_name} still whisper of the {creature} that sleeps beneath the hills."
        ),
    };

    RegionLore {
        region_id,
        legend,
        notable_landmark: format!("{landmark} stands as a silent witness in {region_name}."),
        historical_event: format!("In {region_name}, {event}."),
    }
}

/// Generate crisis lore tied to faction and region names.
fn gen_crisis_lore(
    rng: &mut u64,
    faction_names: &[String],
    region_names: &[String],
) -> CrisisLore {
    // Sleeping King identity
    let king_adj = pick(rng, &[
        "the Once and Future", "the Undying", "the Betrayed",
        "the Dreaming", "the Forgotten", "the Entombed",
        "the Twice-Crowned", "the Hollow",
    ]);
    let king_base = pick(rng, &[
        "King", "Emperor", "Warlord", "Sovereign", "Archon",
        "Tyrant", "Monarch", "Regent",
    ]);
    let king_region = if !region_names.is_empty() {
        let idx = (lcg_next(rng) as usize) % region_names.len();
        &region_names[idx]
    } else {
        "the lost realm"
    };
    let sleeping_king_identity = format!(
        "{king_adj} {king_base}, who ruled {king_region} before the age turned."
    );

    // Blight / Corruption origin
    let blight_src = pick(rng, &[
        "a shattered seal beneath the oldest dungeon",
        "the corpse of a god buried under the mountains",
        "a failed alchemical experiment that seeped into the water table",
        "the vengeful curse of a dying nature spirit",
        "a rift torn open by forbidden summoning magic",
        "the rotting core of an ancient world-tree",
        "a meteor that struck centuries ago, now finally cracking open",
        "the accumulated despair of a hundred years of war",
    ]);
    let blight_origin = format!("The corruption seeps from {blight_src}.");

    // Breach description
    let breach_loc = if !region_names.is_empty() {
        let idx = (lcg_next(rng) as usize) % region_names.len();
        region_names[idx].clone()
    } else {
        "the borderlands".to_string()
    };
    let breach_desc = pick(rng, &[
        "a yawning chasm that exhales hot, sulfurous wind",
        "a crack in reality where the sky turns the color of blood",
        "a pit ringed with black glass where nothing grows",
        "a shimmering tear in the air from which creatures pour endlessly",
        "a collapsed cathedral whose basement opens into something vast",
        "an ancient gate that no ward can seal for long",
        "a wound in the earth that grows wider with each passing moon",
        "a portal of swirling darkness at the base of a dead volcano",
    ]);
    let breach_description = format!(
        "In {breach_loc}, the Breach manifests as {breach_desc}."
    );

    // Use faction names in flavor if available
    let _ = faction_names; // already consumed via region context
    let _ = lcg_f32(rng); // advance RNG for determinism stability

    CrisisLore {
        sleeping_king_identity,
        blight_origin,
        breach_description,
    }
}

/// Generate complete world lore for a campaign. Called once at initialization.
///
/// Uses the campaign's deterministic LCG so identical seeds produce identical lore.
pub fn generate_world_lore(state: &mut CampaignState) -> WorldLore {
    let mut rng = state.rng;

    let campaign_name = gen_campaign_name(&mut rng);
    let era_description = gen_era_description(&mut rng, &campaign_name);
    let guild_origin = gen_guild_origin(&mut rng);

    // Snapshot faction/region data to avoid borrow conflicts
    let faction_snapshots: Vec<(usize, DiplomaticStance, String)> = state
        .factions
        .iter()
        .map(|f| (f.id, f.diplomatic_stance, f.name.clone()))
        .collect();
    let relations = state.diplomacy.relations.clone();
    let region_snapshots: Vec<(usize, String)> = state
        .overworld
        .regions
        .iter()
        .map(|r| (r.id, r.name.clone()))
        .collect();

    // Faction lore
    let faction_lore: Vec<FactionLore> = faction_snapshots
        .iter()
        .map(|(id, stance, name)| {
            gen_faction_lore(&mut rng, *id, *stance, name, &relations)
        })
        .collect();

    // Region lore
    let region_lore: Vec<RegionLore> = region_snapshots
        .iter()
        .map(|(id, name)| gen_region_lore(&mut rng, *id, name))
        .collect();

    // Crisis lore
    let faction_names: Vec<String> = faction_snapshots.iter().map(|(_, _, n)| n.clone()).collect();
    let region_names: Vec<String> = region_snapshots.iter().map(|(_, n)| n.clone()).collect();
    let crisis_lore = gen_crisis_lore(&mut rng, &faction_names, &region_names);

    // Write back RNG state
    state.rng = rng;

    WorldLore {
        campaign_name,
        era_description,
        guild_origin,
        faction_lore,
        region_lore,
        crisis_lore,
    }
}
