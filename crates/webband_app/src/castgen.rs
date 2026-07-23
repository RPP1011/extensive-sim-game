//! Procedural cast generation — port of `F:\MB\src\campaign\castgen.ts`.
//! Every founding rolls a fresh company of bands, companions, and freelancers
//! from the guild's seeded rng (never ambient randomness — same seed, same
//! people). One fixed pass; the draw order is frozen. Coherence comes from two
//! rolls per companion — a TEMPER and a BACKSTORY — from which line, persona,
//! title, and hooks all derive. `assert_cast` errors at founding on any broken
//! invariant (the catalog validateSpec trust-boundary pattern).
//!
//! Draw-order fidelity notes (all mirrored from the TS exactly):
//! - short-circuit draws (`sizes[b] === 3 && rngFloat < 0.3`, the poach roll,
//!   the b>0 hire-cost pick, the hookless ground-want tag) only fire when their
//!   guards pass;
//! - argument evaluation order (a freelancer's hireCost pick happens BEFORE
//!   the name rolls inside rollCompanion);
//! - the look pass (`assign_looks`) is pure and DRAW-FREE — preferences come
//!   from the id hash, so the seeded stream feeding worldgen/goals is
//!   untouched.

use serde::{Deserialize, Serialize};

use crate::error::GenError;
use crate::noise::{feature_roll, id_seed};
use crate::rng::{rng_float, rng_int, rng_pick, rng_pick_idx, RngState};
use crate::worldgen::hook_vocab;

// ---------- Public state structs ----------

/// The identity kit: the axes that make a companion tellable-apart in every
/// scene. Assigned by `assign_looks` with uniqueness rules.
///
/// NOTE: unlike the TS (`look?:`), the Rust type makes absence
/// unrepresentable — every generated companion carries a look by construction,
/// so the TS assert's "companion has no look" branch has no Rust equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompanionLook {
    /// 0 bare, 1 hood, 2 kettle helm, 3 coif, 4 wide-brim hat, 5 fur cap.
    pub headwear: u8,
    /// 0 slight, 1 standard, 2 heavy.
    pub build: u8,
}

pub const HEADWEAR_FAMILIES: u8 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerkId {
    Tracker,
    Trader,
    Surgeon,
    Skirmisher,
    Drillmaster,
    Sutler,
    Engineer,
}

/// What a freelancer waits for before signing cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FreelancerWant {
    Renown { need: i64 },
    Ground { tag: String },
    Band { band_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalKind {
    Guild,
    Deed,
    Prosperity,
    Debt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompanionDef {
    /// Unique, id-safe (`^[a-z][a-z0-9-]*$` — the TS DOM-safety constraint,
    /// kept as a uniqueness/charset guarantee).
    pub id: String,
    pub name: String,
    pub title: String,
    pub hp: f64,
    pub damage: f64,
    pub speed: f64,
    pub reach: f64,
    pub block_chance: f64,
    pub color: u32,
    pub perk: Option<PerkId>,
    pub perk_desc: String,
    pub line: String,
    /// Short personality descriptor (distilled trait descriptors for the LLM).
    pub persona: String,
    pub hire_cost: i64,
    /// Band this companion arrives with; `None` = freelancer.
    pub band: Option<String>,
    /// Narrative hook tags from the backstory; `None` is honest hooklessness.
    pub hooks: Option<Vec<String>>,
    /// Signature ability spec ids (CATALOG keys).
    pub kit: Vec<String>,
    /// Freelancers only.
    pub want: Option<FreelancerWant>,
    pub look: CompanionLook,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandDef {
    pub id: String,
    pub name: String,
    pub blurb: String,
    pub color: u32,
    pub hooks: Vec<String>,
    pub founders: bool,
    pub goal_kind: GoalKind,
    pub want_poach_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cast {
    /// Arrays, not maps — iteration order carries seeded draws (worldgen
    /// binds hooks in cast order).
    pub companions: Vec<CompanionDef>,
    pub bands: Vec<BandDef>,
}

// ---------- Names ----------

/// Ids that would shadow enemy kits, troop ids, or party machinery.
const RESERVED: [&str; 7] = [
    "looter", "bandit", "raider", "warlord", "elite_leader", "hero", "founding",
];

/// Syllable collisions that read as the wrong word entirely.
const BANNED_NAMES: [&str; 2] = ["alien", "marys"];

struct Culture {
    starts: &'static [&'static str],
    ends: &'static [&'static str],
}

/// Order matters: western, norse, steppe, southern (CULTURE_IDS).
const CULTURES: [Culture; 4] = [
    Culture {
        starts: &["Al", "Bran", "Ced", "Gar", "Hale", "Mar", "Rol", "Wal", "Ede", "Ler", "Cor", "Ger"],
        ends: &["ric", "win", "and", "mund", "bert", "ien", "ys", "ard"],
    },
    Culture {
        starts: &["Sig", "Thor", "Bjor", "Hal", "Ing", "Rag", "Sol", "Gud", "Ey", "Ulf", "Ast", "Kel"],
        ends: &["run", "vald", "grim", "dis", "na", "leif", "frid", "mar"],
    },
    Culture {
        starts: &["Ba", "Tor", "Ke", "Sar", "Ula", "Chag", "No", "Del", "Qara", "Tem"],
        ends: &["dai", "ghan", "chek", "gis", "nai", "tugh", "mish", "tar"],
    },
    Culture {
        starts: &["Az", "Fah", "Mas", "Ras", "Zay", "Nur", "Kha", "Sha", "Ibra", "Yal"],
        ends: &["im", "ir", "oud", "ida", "ana", "iq", "ez", "ara"],
    },
];

fn prefix4(s: &str) -> &str {
    &s[..s.len().min(4)]
}

/// The three name-surface constraints: exact-match surfaces, 4-char-prefix
/// speaker recovery, substring prose lookup. Band fores share the pool.
fn accept_name(name: &str, taken: &[String]) -> bool {
    let n = name.to_lowercase();
    if RESERVED.contains(&n.as_str()) || BANNED_NAMES.contains(&n.as_str()) {
        return false;
    }
    taken.iter().all(|t| {
        let s = t.to_lowercase();
        prefix4(&n) != prefix4(&s) && !n.contains(&s) && !s.contains(&n)
    })
}

/// Bounded seeded retries, then a deterministic fixed-order scan.
fn roll_name(g: &mut RngState, culture: usize, taken: &[String]) -> Result<String, GenError> {
    let t = &CULTURES[culture];
    for _ in 0..40 {
        let name = format!("{}{}", rng_pick(g, t.starts), rng_pick(g, t.ends));
        if accept_name(&name, taken) {
            return Ok(name);
        }
    }
    let mut order: Vec<usize> = vec![culture];
    order.extend(0..CULTURES.len());
    for cid in order {
        let c = &CULTURES[cid];
        for s in c.starts {
            for e in c.ends {
                let name = format!("{s}{e}");
                if accept_name(&name, taken) {
                    return Ok(name);
                }
            }
        }
    }
    Err(GenError::NameSpaceExhausted("name"))
}

// ---------- Archetypes ----------

pub const ARCH_SKIRMISHER: usize = 0;
pub const ARCH_ARCHER: usize = 1;
pub const ARCH_BRUISER: usize = 2;
pub const ARCH_DUELIST: usize = 3;
pub const ARCH_WARDEN: usize = 4;
pub const ARCH_HEALER: usize = 5;
pub const ARCH_ASSASSIN: usize = 6;
const N_ARCH: usize = 7;

struct Envelope {
    hp: (f64, f64),
    damage: (f64, f64),
    speed: (f64, f64),
    reach: (f64, f64),
    block: (f64, f64),
    titles: &'static [&'static str],
    flavor: &'static [&'static str],
    kits: &'static [&'static [&'static str]],
}

const ARCHETYPES: [Envelope; N_ARCH] = [
    // skirmisher
    Envelope {
        hp: (80.0, 95.0), damage: (20.0, 24.0), speed: (6.0, 6.4), reach: (1.9, 2.1), block: (0.25, 0.4),
        titles: &["the Quick", "the Wader", "the Grey"],
        flavor: &["Moves quiet and strikes first", "Faster than the argument", "Never where the swing lands"],
        kits: &[&["lunge"], &["lunge", "expose_weakness"], &["pocket_sand", "lunge"]],
    },
    // archer
    Envelope {
        hp: (80.0, 95.0), damage: (22.0, 26.0), speed: (5.8, 6.2), reach: (2.0, 2.1), block: (0.25, 0.35),
        titles: &["the Fletcher", "the Bow", "the Hawk-eyed"],
        flavor: &["Her arrows arrive before the argument does", "Counts the wind like coin", "Kills politely, from far away"],
        kits: &[&["crippling_shot"], &["crippling_shot", "expose_weakness"]],
    },
    // bruiser
    Envelope {
        hp: (125.0, 150.0), damage: (24.0, 30.0), speed: (5.5, 5.9), reach: (2.2, 2.4), block: (0.45, 0.55),
        titles: &["the Bear", "the Ox", "the Breaker"],
        flavor: &["A fearsome front-line fighter", "Built like a gatehouse, swings like one falling", "The press parts around them"],
        kits: &[&["cleaving_blow"], &["whirlwind"], &["power_strike", "cleaving_blow"]],
    },
    // duelist
    Envelope {
        hp: (105.0, 125.0), damage: (26.0, 30.0), speed: (5.8, 6.2), reach: (2.2, 2.4), block: (0.4, 0.55),
        titles: &["the Swordsman", "the Blade", "the Duelist"],
        flavor: &["A master of the blade", "Fights like the ballads demand", "One clean answer to every question"],
        kits: &[&["heroic_strike"], &["steppe_charge"], &["power_strike", "lunge"]],
    },
    // warden
    Envelope {
        hp: (130.0, 150.0), damage: (22.0, 26.0), speed: (5.4, 5.7), reach: (2.2, 2.3), block: (0.5, 0.55),
        titles: &["the Shield", "the Steady", "the Gate"],
        flavor: &["The shield the line stands behind", "Holds ground like it owes them rent", "The last one standing, by design"],
        kits: &[&["shield_wall"], &["shield_wall", "rally"], &["shield_wall", "power_strike"]],
    },
    // healer
    Envelope {
        hp: (78.0, 100.0), damage: (16.0, 19.0), speed: (5.4, 5.6), reach: (2.0, 2.1), block: (0.2, 0.35),
        titles: &["the Leech", "the Mender", "the Physician"],
        flavor: &["Your friends need not die of a scratch", "Sews flesh and tempers alike", "Keeps the whole company on its feet"],
        kits: &[&["field_dressing"], &["field_dressing", "quartermasters_brew"], &["field_dressing", "rally"]],
    },
    // assassin
    Envelope {
        hp: (68.0, 80.0), damage: (28.0, 32.0), speed: (6.2, 6.5), reach: (1.9, 1.9), block: (0.25, 0.35),
        titles: &["the Knife", "the Shadow", "the Quiet"],
        flavor: &["Fragile, fast, and astonishingly deadly", "You will not hear the second step", "Ends fights that never started"],
        kits: &[&["sisters_knife"], &["sisters_knife", "pocket_sand"]],
    },
];

/// Every spec id the ability catalog knows (`catalog.ts`) — the kit
/// validation vocabulary. The full IR port is a later slice; the ID SET is
/// what `assert_cast` needs.
pub const CATALOG_IDS: [&str; 17] = [
    "power_strike", "lunge", "whirlwind", "second_wind", "cleaving_blow",
    "crippling_shot", "field_dressing", "shield_wall", "warlord_sweep",
    "ballista_bolt", "expose_weakness", "pocket_sand", "quartermasters_brew",
    "sisters_knife", "rally", "steppe_charge", "heroic_strike",
];

// ---------- Perks ----------

/// PERK_DESCS key order in the TS — the scatter loop iterates it.
const PERK_ORDER: [PerkId; 7] = [
    PerkId::Tracker, PerkId::Trader, PerkId::Surgeon, PerkId::Skirmisher,
    PerkId::Drillmaster, PerkId::Sutler, PerkId::Engineer,
];

fn perk_desc(p: PerkId) -> &'static str {
    match p {
        PerkId::Tracker => "Enemy warband numbers are revealed on the map",
        PerkId::Trader => "+25% gold from every source",
        PerkId::Surgeon => "Knocked-out companions wake at 60% health; the party mends 15% after each battle",
        PerkId::Skirmisher => "The whole party fights 10% faster on foot",
        PerkId::Drillmaster => "+10% damage for the whole party",
        PerkId::Sutler => "Tavern services cost half",
        PerkId::Engineer => "+12% max health for the whole party",
    }
}

fn perk_title(p: PerkId) -> &'static str {
    match p {
        PerkId::Tracker => "the Tracker",
        PerkId::Trader => "the Factor",
        PerkId::Surgeon => "the Surgeon",
        PerkId::Skirmisher => "the Outrider",
        PerkId::Drillmaster => "the Drillmaster",
        PerkId::Sutler => "the Sutler",
        PerkId::Engineer => "the Engineer",
    }
}

/// Archetypes a perk sits well on (`None` = anyone).
fn perk_fit(p: PerkId) -> Option<&'static [usize]> {
    match p {
        PerkId::Tracker => Some(&[ARCH_SKIRMISHER, ARCH_ARCHER]),
        PerkId::Surgeon => Some(&[ARCH_HEALER]),
        PerkId::Skirmisher => Some(&[ARCH_SKIRMISHER, ARCH_ARCHER, ARCH_ASSASSIN]),
        PerkId::Drillmaster => Some(&[ARCH_BRUISER, ARCH_WARDEN, ARCH_DUELIST]),
        PerkId::Engineer => Some(&[ARCH_ARCHER, ARCH_WARDEN, ARCH_DUELIST, ARCH_BRUISER, ARCH_SKIRMISHER]),
        PerkId::Trader | PerkId::Sutler => None,
    }
}

// ---------- Tempers & backstories (the coherence scheme) ----------

struct Temper {
    adj: &'static str,
    quirk: &'static str,
    touch: &'static str,
}

const TEMPERS: [Temper; 12] = [
    Temper { adj: "gruff, plain-spoken", quirk: "distrusts nobles", touch: "And keep the lords out of it." },
    Temper { adj: "boastful", quirk: "takes credit loudly", touch: "No one does it better." },
    Temper { adj: "dry, scholarly", quirk: "scorns superstition", touch: "That is simply arithmetic." },
    Temper { adj: "pious, formal", quirk: "seeks penance", touch: "The rest I leave to heaven." },
    Temper { adj: "quiet", quirk: "morbid humor", touch: "I'm still deciding how I feel about that." },
    Temper { adj: "grandiose", quirk: "claims implausible pedigree", touch: "My grandfather held three castles. Allegedly." },
    Temper { adj: "laconic", quirk: "counts everything", touch: "That is the whole of it." },
    Temper { adj: "jovial", quirk: "misquotes scripture on purpose", touch: "As the good book almost says." },
    Temper { adj: "grave", quirk: "weighs every promise like coin", touch: "I do not say such things lightly." },
    Temper { adj: "wary", quirk: "reads omens in weather", touch: "The crows agreed, for what that is worth." },
    Temper { adj: "penny-counting", quirk: "bitter about lost goods", touch: "Someone still owes me for the last one." },
    Temper { adj: "proud", quirk: "scorns fools", touch: "Try to keep up." },
];

struct Backstory {
    hooks: &'static [&'static str],
    frag: &'static str,
    titles: &'static [&'static str],
    lines: &'static [&'static str],
}

const BACKSTORIES: [Backstory; 12] = [
    Backstory {
        hooks: &["roads", "trade"], frag: "years guarding caravans",
        titles: &["the Outrider", "the Factor", "the Wheel"],
        lines: &[
            "Ten years walking other men's goods down bad roads. Yours will be the first I keep.",
            "I know every toll, ford, and ambush from here to the coast. Ask me the price of any of them.",
        ],
    },
    Backstory {
        hooks: &["marsh"], frag: "kept the marsh paths",
        titles: &["the Fen-walker", "the Quiet"],
        lines: &[
            "The fen took my company's pay and half its boots. I left before it took the rest.",
            "Solid ground is a rumor where I come from. I learned to stand on rumors.",
        ],
    },
    Backstory {
        hooks: &["war"], frag: "twelve years in a pike line",
        titles: &["the Sergeant", "the Pike"],
        lines: &[
            "I held a line for twelve years and one bad morning. Don't ask about the morning.",
            "Wars end. The men who drilled you into the mud find other work.",
        ],
    },
    Backstory {
        hooks: &["faith", "learning"], frag: "put out of an abbey",
        titles: &["the Friar", "the Scribe"],
        lines: &[
            "The abbey and I disagreed on one point of doctrine. The point was mine to keep.",
            "I copied books until they burned the wrong one. Now I make my letters in other ways.",
        ],
    },
    Backstory {
        hooks: &["dead"], frag: "dug graves for coin",
        titles: &["the Sexton", "the Pale"],
        lines: &[
            "I buried better men than the ones who paid me to. I keep the shovel out of sentiment.",
            "The dead are easy company. It is the living who short your wages.",
        ],
    },
    Backstory {
        hooks: &["sea"], frag: "lost a ship, misses salt water",
        titles: &["the Oarless", "the Salt"],
        lines: &[
            "My ship rots in a port I can't show my face in. Pay well and I'll row for you anyway.",
            "I traded a deck for a saddle and I regret it daily. The pay had better be worth it.",
        ],
    },
    Backstory {
        hooks: &["mountain", "high"], frag: "hunted the high passes",
        titles: &["the Wayfarer", "the Falconer"],
        lines: &[
            "Above the treeline everything worth killing can see you coming. You learn patience or you learn falling.",
            "I hunted for lords who never climbed past their own gates. Their coin spent fine, though.",
        ],
    },
    Backstory {
        hooks: &["fallen-house"], frag: "served a house that fell",
        titles: &["the Banneret", "the Lance", "the Sworn"],
        lines: &[
            "My lord is dead and the oath is not. It needs somewhere to live. Perhaps your banner.",
            "I carried a banner at a field nobody sings about. It fell. I did not.",
        ],
    },
    Backstory {
        hooks: &["old", "dead"], frag: "grew up beside the barrows",
        titles: &["the Barrow-born", "the Elder"],
        lines: &[
            "We plowed around the barrows and never after dark. I know which rules were real.",
            "Old ground remembers. I was raised to listen to it.",
        ],
    },
    Backstory {
        hooks: &["hearth", "trade"], frag: "raised at a mill, left it",
        titles: &["the Miller", "the Stone"],
        lines: &[
            "Twenty years of flour and I never once got to hit anyone. I am making up the count.",
            "The mill wanted a son and got me. We parted on honest terms: I took nothing but the dog.",
        ],
    },
    Backstory {
        hooks: &["water", "roads"], frag: "poled a ferry, heard everything",
        titles: &["the Ferryman", "the Wader"],
        lines: &[
            "Everyone crosses the river eventually, and everyone talks on the water. I remember all of it.",
            "I know who paid, who swam, and who went in wearing stones. Useful trade, ferrying.",
        ],
    },
    Backstory {
        // The honest-hookless story — some people the map can't hold.
        hooks: &[], frag: "from nowhere in particular",
        titles: &["the Stray", "the Far-come"],
        lines: &[
            "Where am I from? The road behind me. Where am I going? The one in front.",
            "I have been everywhere twice and belonged nowhere once. Your fire looked warm.",
        ],
    },
];
const WANDERER: usize = BACKSTORIES.len() - 1;

// ---------- Band identity ----------

struct Concept {
    hooks: &'static [&'static str],
    suffixes: &'static [&'static str],
    blurbs: &'static [&'static str],
}

const CONCEPTS: [Concept; 8] = [
    Concept {
        hooks: &["trade", "roads"], suffixes: &["Company", "Train"],
        blurbs: &[
            "trail-wise and thrifty, veterans of the caravan roads.",
            "they have walked every toll road twice and been cheated on most of them.",
        ],
    },
    Concept {
        hooks: &["marsh"], suffixes: &["Watch", "Wardens"],
        blurbs: &[
            "quiet blades who kept the marsh roads until the marsh kept their pay.",
            "they know the fens by smell and hold grudges like standing water.",
        ],
    },
    Concept {
        hooks: &["war", "roads"], suffixes: &["Train", "Column"],
        blurbs: &[
            "quartermasters, pike-drill, and siegecraft — an army's spine for hire.",
            "what remains of a company that outlived three paymasters.",
        ],
    },
    Concept {
        hooks: &["mountain", "high"], suffixes: &["Bows", "Hunt"],
        blurbs: &[
            "hunters of the high passes, hooded, weathered, and owed money.",
            "they read ridgelines the way clerks read ledgers.",
        ],
    },
    Concept {
        hooks: &["fallen-house"], suffixes: &["Lances", "Sworn"],
        blurbs: &[
            "sworn lances of a fallen house, seeking a banner worth the oath.",
            "their lord is ash; the oath is looking for lodging.",
        ],
    },
    Concept {
        hooks: &["sea"], suffixes: &["Oars", "Tide"],
        blurbs: &[
            "salt-cured and shipless, they fight like the sea owes them passage.",
            "a crew without a keel, selling boarding-work on dry land.",
        ],
    },
    Concept {
        hooks: &["faith"], suffixes: &["Vigil", "Candles"],
        blurbs: &[
            "blades sworn at an altar nobody tends anymore.",
            "they keep a saint the church misplaced, and their swords keep them.",
        ],
    },
    Concept {
        hooks: &["dead", "old"], suffixes: &["Wake", "Wards"],
        blurbs: &[
            "they mind what sleeps under the old ground, for a fee.",
            "grave-quiet and patient — the dead taught them both.",
        ],
    },
];

const FORE_ADJ: [&str; 12] = [
    "Grey", "Black", "Red", "Long", "Cold", "White", "Old", "Broken", "Salt", "Iron", "Winter", "Low",
];
const FORE_NOUN: [&str; 12] = [
    "road", "march", "field", "water", "gate", "bridge", "hill", "fen", "shore", "wood", "tower", "candle",
];

fn roll_fore(g: &mut RngState, taken: &[String]) -> Result<String, GenError> {
    for _ in 0..40 {
        let fore = format!("{}{}", rng_pick(g, &FORE_ADJ), rng_pick(g, &FORE_NOUN));
        if accept_name(&fore, taken) {
            return Ok(fore);
        }
    }
    for a in FORE_ADJ {
        for n in FORE_NOUN {
            let fore = format!("{a}{n}");
            if accept_name(&fore, taken) {
                return Ok(fore);
            }
        }
    }
    Err(GenError::NameSpaceExhausted("band"))
}

// ---------- Colors ----------

pub fn hsl_to_hex(h: f64, s: f64, l: f64) -> u32 {
    let hue = ((h % 360.0) + 360.0) % 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (((hue / 60.0) % 2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let (r, gg, b) = if hue < 60.0 {
        (c, x, 0.0)
    } else if hue < 120.0 {
        (x, c, 0.0)
    } else if hue < 180.0 {
        (0.0, c, x)
    } else if hue < 240.0 {
        (0.0, x, c)
    } else if hue < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let to255 = |v: f64| -> u32 { ((v + m) * 255.0).round() as u32 };
    (to255(r) << 16) | (to255(gg) << 8) | to255(b)
}

fn hue_of_hex(color: u32) -> f64 {
    let r = f64::from((color >> 16) & 255) / 255.0;
    let g = f64::from((color >> 8) & 255) / 255.0;
    let b = f64::from(color & 255) / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max == min {
        return 0.0;
    }
    let d = max - min;
    let h = if max == r {
        ((g - b) / d + 6.0) % 6.0
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h * 60.0
}

fn hue_delta(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

/// Two companions closer than this in hue must differ in silhouette.
const HUE_BAR: f64 = 25.0;

fn build_for(hp: f64) -> u8 {
    if hp < 88.0 {
        0
    } else if hp < 112.0 {
        1
    } else {
        2
    }
}

// ---------- Generation ----------

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn round_step(v: f64, step: f64) -> f64 {
    (v / step).round() * step
}

/// (hp, damage, speed, reach, block_chance) — five draws, in this order.
fn roll_stats(g: &mut RngState, e: &Envelope) -> (f64, f64, f64, f64, f64) {
    let hp = round_step(lerp(e.hp.0, e.hp.1, rng_float(g)), 5.0);
    let damage = lerp(e.damage.0, e.damage.1, rng_float(g)).round();
    let speed = round_step(lerp(e.speed.0, e.speed.1, rng_float(g)), 0.1);
    let reach = round_step(lerp(e.reach.0, e.reach.1, rng_float(g)), 0.1);
    let block = round_step(lerp(e.block.0, e.block.1, rng_float(g)), 0.05);
    (hp, damage, speed, reach, block)
}

/// Archetype draw with a per-band cap of 2 repeats.
fn roll_archetype(g: &mut RngState, in_band: &[usize], pool: &[usize]) -> usize {
    for _ in 0..12 {
        let a = *rng_pick(g, pool);
        if in_band.iter().filter(|&&x| x == a).count() < 2 {
            return a;
        }
    }
    pool.iter()
        .copied()
        .find(|&a| in_band.iter().filter(|&&x| x == a).count() < 2)
        .unwrap_or(pool[0])
}

/// A rolled companion plus the generation-only rolls the later passes need.
struct Rolled {
    def: CompanionDef,
    arch: usize,
    story: usize,
}

#[allow(clippy::too_many_arguments)]
fn roll_companion(
    g: &mut RngState,
    arch: usize,
    culture: usize,
    taken: &mut Vec<String>,
    hue: f64,
    band_id: Option<String>,
    band_hooks: &'static [&'static str],
    hire_cost: i64,
    dedup_tempers: &mut Vec<usize>,
    dedup_stories: &mut Vec<usize>,
) -> Result<Rolled, GenError> {
    let name = roll_name(g, culture, taken)?;
    taken.push(name.clone());
    // No temper or backstory repeats within a band.
    let mut temper = rng_pick_idx(g, TEMPERS.len());
    let mut i = 0;
    while dedup_tempers.contains(&temper) && i < 8 {
        temper = rng_pick_idx(g, TEMPERS.len());
        i += 1;
    }
    if dedup_tempers.contains(&temper) {
        temper = (0..TEMPERS.len()).find(|t| !dedup_tempers.contains(t)).unwrap_or(temper);
    }
    dedup_tempers.push(temper);
    // Backstories sharing a hook with the band ride at double weight; the
    // wanderer always does.
    let mut story_pool: Vec<usize> = Vec::new();
    for (si, s) in BACKSTORIES.iter().enumerate() {
        let kin = s.hooks.iter().any(|h| band_hooks.contains(h));
        story_pool.push(si);
        if si == WANDERER || kin {
            story_pool.push(si);
        }
    }
    let mut story = story_pool[rng_pick_idx(g, story_pool.len())];
    let mut i = 0;
    while dedup_stories.contains(&story) && i < 8 {
        story = story_pool[rng_pick_idx(g, story_pool.len())];
        i += 1;
    }
    if dedup_stories.contains(&story) {
        story = (0..BACKSTORIES.len()).find(|s| !dedup_stories.contains(s)).unwrap_or(story);
    }
    dedup_stories.push(story);
    let base = *rng_pick(g, BACKSTORIES[story].lines);
    let e = &ARCHETYPES[arch];
    let (hp, damage, speed, reach, block_chance) = roll_stats(g, e);
    let sat = 0.28 + rng_float(g) * 0.14;
    let light = 0.35 + rng_float(g) * 0.15;
    let color = hsl_to_hex(hue, sat, light);
    let kit: Vec<String> = rng_pick(g, e.kits).iter().map(|s| (*s).to_string()).collect();
    let t = &TEMPERS[temper];
    let s = &BACKSTORIES[story];
    Ok(Rolled {
        def: CompanionDef {
            id: name.to_lowercase(),
            name,
            title: String::new(), // the title pass runs after perks land
            hp,
            damage,
            speed,
            reach,
            block_chance,
            color,
            perk: None,
            perk_desc: String::new(), // the perk pass fills it
            line: format!("\"{} {}\"", base, t.touch),
            persona: format!("{}; {}; {}", t.adj, s.frag, t.quirk),
            hire_cost,
            band: band_id,
            hooks: if s.hooks.is_empty() {
                None
            } else {
                Some(s.hooks.iter().map(|h| (*h).to_string()).collect())
            },
            kit,
            want: None,
            look: CompanionLook { headwear: 0, build: 0 }, // assign_looks overwrites
        },
        arch,
        story,
    })
}

pub fn roll_cast(g: &mut RngState) -> Result<Cast, GenError> {
    // 1 ── shape
    let n_bands = rng_int(g, 4, 5) as usize;
    let mut sizes: Vec<i64> = vec![rng_int(g, 3, 4)];
    for _ in 1..n_bands {
        sizes.push(*rng_pick(g, &[2i64, 3, 3, 4]));
    }
    let mut n_free = rng_int(g, 2, 3);
    // Deterministic clamp to 14–18 total (no draws).
    let total = |sizes: &[i64], n_free: i64| sizes.iter().sum::<i64>() + n_free;
    while total(&sizes, n_free) > 18 {
        let mx = sizes[1..].iter().copied().max().unwrap();
        let i = sizes.iter().rposition(|&s| s == mx).unwrap();
        if i > 0 && sizes[i] > 2 {
            sizes[i] -= 1;
        } else {
            n_free -= 1;
        }
    }
    while total(&sizes, n_free) < 14 {
        let mn = sizes[1..].iter().copied().min().unwrap();
        let i = sizes.iter().position(|&s| s == mn).unwrap();
        sizes[i] += 1;
    }

    let mut taken: Vec<String> = Vec::new();
    let mut bands: Vec<BandDef> = Vec::new();
    let mut comps: Vec<Rolled> = Vec::new();
    let base_hue = rng_float(g) * 360.0;
    let mut concept_pool: Vec<usize> = (0..CONCEPTS.len()).collect();
    let mut poach: Option<(usize, i64)> = None;

    // 2 ── bands, each followed by its members (one pass, fixed order)
    for b in 0..n_bands {
        let ci = concept_pool.remove((rng_float(g) * concept_pool.len() as f64).floor() as usize);
        let concept = &CONCEPTS[ci];
        let fore = roll_fore(g, &taken)?;
        taken.push(fore.clone());
        // Short-circuit fidelity: the 0.3 roll only happens for size-3 bands.
        let suffix: String = if sizes[b] == 3 && rng_float(g) < 0.3 {
            "Three".to_string()
        } else {
            (*rng_pick(g, concept.suffixes)).to_string()
        };
        let hue = (base_hue + (b as f64 * 360.0) / (n_bands as f64 + 1.0)) % 360.0;
        let goal_kind = if b == 0 {
            GoalKind::Guild
        } else {
            *rng_pick(g, &[GoalKind::Deed, GoalKind::Deed, GoalKind::Prosperity, GoalKind::Debt])
        };
        if b > 0 && poach.is_none() && rng_float(g) < 0.35 {
            poach = Some((b, rng_int(g, 0, n_free - 1)));
        }
        bands.push(BandDef {
            id: fore.to_lowercase(),
            name: format!("The {fore} {suffix}"),
            blurb: (*rng_pick(g, concept.blurbs)).to_string(),
            color: hsl_to_hex(hue, 0.3, 0.34),
            hooks: concept.hooks.iter().map(|h| (*h).to_string()).collect(),
            founders: b == 0,
            goal_kind,
            want_poach_id: None,
        });

        let culture = rng_pick_idx(g, CULTURES.len());
        let mut in_band: Vec<usize> = Vec::new();
        let mut dedup_tempers: Vec<usize> = Vec::new();
        let mut dedup_stories: Vec<usize> = Vec::new();
        // The founders always field exactly one healer.
        let healer_slot: i64 = if b == 0 { rng_int(g, 0, sizes[b] - 1) } else { -1 };
        for m in 0..sizes[b] {
            let member_culture = if rng_float(g) < 0.7 {
                culture
            } else {
                rng_pick_idx(g, CULTURES.len())
            };
            let arch = if m == healer_slot {
                ARCH_HEALER
            } else if b == 0 {
                let pool: Vec<usize> = (0..N_ARCH).filter(|&a| a != ARCH_HEALER).collect();
                roll_archetype(g, &in_band, &pool)
            } else {
                let pool: Vec<usize> = (0..N_ARCH).collect();
                roll_archetype(g, &in_band, &pool)
            };
            in_band.push(arch);
            let hue2 = hue + (m as f64 - (sizes[b] - 1) as f64 / 2.0) * 24.0;
            let cost = if b == 0 {
                100
            } else {
                100 + (b as i64) * 50 + *rng_pick(g, &[0i64, 50])
            };
            let band_id = bands[b].id.clone();
            comps.push(roll_companion(
                g, arch, member_culture, &mut taken, hue2, Some(band_id),
                concept.hooks, cost, &mut dedup_tempers, &mut dedup_stories,
            )?);
        }
    }

    // 3 ── freelancers (exotic-leaning archetypes, each with a want)
    let mut free_ids: Vec<String> = Vec::new();
    let free_hue_base = (base_hue + (n_bands as f64 * 360.0) / (n_bands as f64 + 1.0)) % 360.0;
    let mut free_dedup_tempers: Vec<usize> = Vec::new();
    let mut free_dedup_stories: Vec<usize> = Vec::new();
    for f in 0..n_free {
        let arch = *rng_pick(g, &[ARCH_ASSASSIN, ARCH_DUELIST, ARCH_ARCHER, ARCH_ASSASSIN, ARCH_DUELIST]);
        let culture = rng_pick_idx(g, CULTURES.len());
        // Argument-evaluation fidelity: the hireCost pick draws BEFORE the
        // name rolls inside rollCompanion.
        let hire = *rng_pick(g, &[150i64, 200, 250, 300]);
        let mut c = roll_companion(
            g, arch, culture, &mut taken, free_hue_base + (f as f64) * 24.0, None,
            &[], hire, &mut free_dedup_tempers, &mut free_dedup_stories,
        )?;
        let kind = *rng_pick(g, &[0usize, 1, 2]); // renown / ground / band
        c.def.want = Some(match kind {
            0 => FreelancerWant::Renown { need: rng_int(g, 4, 8) * 25 },
            1 => FreelancerWant::Ground {
                tag: match &c.def.hooks {
                    Some(h) => h[0].clone(),
                    None => (*rng_pick(g, &["dead", "war", "trade", "old"])).to_string(),
                },
            },
            _ => FreelancerWant::Band {
                band_id: bands[1 + rng_pick_idx(g, bands.len() - 1)].id.clone(),
            },
        });
        free_ids.push(c.def.id.clone());
        comps.push(c);
    }

    // 4 ── resolve the poach want; guarantee a camped deed-goal band
    if let Some((band_index, free_index)) = poach {
        let fi = (free_index.min(free_ids.len() as i64 - 1)).max(0) as usize;
        bands[band_index].want_poach_id = Some(free_ids[fi].clone());
    }
    if !bands.iter().enumerate().any(|(i, b)| i > 0 && b.goal_kind == GoalKind::Deed) {
        bands[1].goal_kind = GoalKind::Deed;
    }

    // 5 ── perks: founders get one of the utility three + one other; the rest
    // scatter archetype-fitted at p=0.7 each
    let founders_band_id = bands[0].id.clone();
    let founder_idx: Vec<usize> = comps
        .iter()
        .enumerate()
        .filter(|(_, c)| c.def.band.as_deref() == Some(founders_band_id.as_str()))
        .map(|(i, _)| i)
        .collect();
    let other_idx: Vec<usize> = comps
        .iter()
        .enumerate()
        .filter(|(_, c)| c.def.band.as_deref() != Some(founders_band_id.as_str()))
        .map(|(i, _)| i)
        .collect();
    fn give(g: &mut RngState, perk: PerkId, pool: &[usize], comps: &mut [Rolled]) {
        let fit: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|&i| {
                comps[i].def.perk.is_none()
                    && perk_fit(perk).map_or(true, |f| f.contains(&comps[i].arch))
            })
            .collect();
        let fallback: Vec<usize> = pool.iter().copied().filter(|&i| comps[i].def.perk.is_none()).collect();
        let who = if !fit.is_empty() {
            fit[rng_pick_idx(g, fit.len())]
        } else if !fallback.is_empty() {
            fallback[rng_pick_idx(g, fallback.len())]
        } else {
            return;
        };
        comps[who].def.perk = Some(perk);
        comps[who].def.perk_desc = perk_desc(perk).to_string();
        if perk == PerkId::Engineer && comps[who].def.kit.len() < 2 {
            comps[who].def.kit.push("ballista_bolt".to_string());
        }
    }
    let first = *rng_pick(g, &[PerkId::Tracker, PerkId::Trader, PerkId::Surgeon]);
    give(g, first, &founder_idx, &mut comps);
    let rest: Vec<PerkId> = PERK_ORDER.iter().copied().filter(|&p| p != first).collect();
    let second = *rng_pick(g, &rest);
    give(g, second, &founder_idx, &mut comps);
    for &perk in &rest {
        if comps.iter().any(|c| c.def.perk == Some(perk)) {
            continue;
        }
        if rng_float(g) < 0.7 {
            give(g, perk, &other_idx, &mut comps);
        }
    }
    for c in comps.iter_mut() {
        if c.def.perk_desc.is_empty() {
            c.def.perk_desc = (*rng_pick(g, ARCHETYPES[c.arch].flavor)).to_string();
        }
    }

    // 6 ── titles: perk holders claim theirs first, then the perkless draw
    // from backstory/archetype pools with a uniqueness reject
    let mut used_titles: Vec<String> = Vec::new();
    for c in comps.iter_mut() {
        if let Some(p) = c.def.perk {
            c.def.title = perk_title(p).to_string();
            used_titles.push(c.def.title.clone());
        }
    }
    for ci in 0..comps.len() {
        if comps[ci].def.perk.is_some() {
            continue;
        }
        let pool: Vec<&'static str> = BACKSTORIES[comps[ci].story]
            .titles
            .iter()
            .chain(ARCHETYPES[comps[ci].arch].titles.iter())
            .copied()
            .collect();
        let mut t = *rng_pick(g, &pool);
        let mut guard = 20;
        while used_titles.iter().any(|u| u.as_str() == t) && guard > 0 {
            t = *rng_pick(g, &pool);
            guard -= 1;
        }
        if used_titles.iter().any(|u| u.as_str() == t) {
            t = pool
                .iter()
                .copied()
                .find(|x| !used_titles.iter().any(|u| u.as_str() == *x))
                .unwrap_or(t);
        }
        comps[ci].def.title = t.to_string();
        used_titles.push(comps[ci].def.title.clone());
    }

    let mut cast = Cast {
        bands,
        companions: comps.into_iter().map(|r| r.def).collect(),
    };
    assign_looks(&mut cast);
    assert_cast(&cast)?;
    Ok(cast)
}

// ---------- The identity kit (looks) ----------

/// The identifiability pass: every companion pair must differ on a visible
/// axis — hue (≥ HUE_BAR°), headwear family, or build. Pure and DRAW-FREE
/// (preferences come from the id hash), so the seeded rng stream feeding
/// worldgen/goals is untouched.
pub fn assign_looks(cast: &mut Cast) {
    // Group indices in insertion order (band id, or the freelancer bucket).
    let mut group_keys: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of: Vec<usize> = vec![0; cast.companions.len()];
    for (i, c) in cast.companions.iter().enumerate() {
        let key = c.band.clone().unwrap_or_else(|| "_freelancers".to_string());
        let gi = match group_keys.iter().position(|k| *k == key) {
            Some(gi) => gi,
            None => {
                group_keys.push(key);
                groups.push(Vec::new());
                group_keys.len() - 1
            }
        };
        groups[gi].push(i);
        group_of[i] = gi;
    }

    // Pass 1: hash-preferred headwear, deduped within the group; hp build.
    let hwf = HEADWEAR_FAMILIES;
    for members in &groups {
        let mut taken_hw: Vec<u8> = Vec::new();
        for &i in members {
            let c = &mut cast.companions[i];
            let pref = (feature_roll(id_seed(&c.id), 5) * f64::from(hwf)).floor() as u8;
            let mut hw = pref;
            let mut k: u8 = 1;
            while taken_hw.contains(&hw) && k <= hwf {
                hw = (pref + k) % hwf;
                k += 1;
            }
            taken_hw.push(hw);
            c.look = CompanionLook { headwear: hw, build: build_for(c.hp) };
        }
    }

    // Pass 2: each companion must not share (headwear, build) with any
    // EARLIER near-hue companion. One left-to-right pass settles.
    for j in 0..cast.companions.len() {
        let b_hue = hue_of_hex(cast.companions[j].color);
        let near: Vec<CompanionLook> = (0..j)
            .filter(|&a| hue_delta(hue_of_hex(cast.companions[a].color), b_hue) < HUE_BAR)
            .map(|a| cast.companions[a].look)
            .collect();
        let collides = |hw: u8, build: u8| near.iter().any(|l| l.headwear == hw && l.build == build);
        let cur = cast.companions[j].look;
        if !collides(cur.headwear, cur.build) {
            continue;
        }
        let mates: Vec<u8> = groups[group_of[j]]
            .iter()
            .copied()
            .filter(|&m| m != j)
            .map(|m| cast.companions[m].look.headwear)
            .collect();
        let mut free_hw: Vec<u8> = vec![cur.headwear];
        for hw in 0..hwf {
            if hw != cur.headwear && !mates.contains(&hw) {
                free_hw.push(hw);
            }
        }
        // Prefer keeping the headwear and shifting build; then new combos.
        'outer: for &hw in &free_hw {
            for build in [cur.build, (cur.build + 1) % 3, (cur.build + 2) % 3] {
                if hw == cur.headwear && build == cur.build {
                    continue;
                }
                if !collides(hw, build) {
                    cast.companions[j].look = CompanionLook { headwear: hw, build };
                    break 'outer;
                }
            }
        }
    }
}

// ---------- Generation-time invariants ----------

fn id_safe(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Errors at founding on any broken invariant — the same trust boundary as
/// the TS `assertCast`: bad generation is loud, never a quietly wrong save.
pub fn assert_cast(cast: &Cast) -> Result<(), GenError> {
    let fail = |msg: String| Err(GenError::Invariant(msg));
    let vocab = hook_vocab();
    let mut ids: Vec<&str> = Vec::new();
    for c in &cast.companions {
        if !id_safe(&c.id) {
            return fail(format!("id '{}' not DOM-safe", c.id));
        }
        if RESERVED.contains(&c.id.as_str()) {
            return fail(format!("id '{}' is reserved", c.id));
        }
        if ids.contains(&c.id.as_str()) {
            return fail(format!("duplicate id '{}'", c.id));
        }
        ids.push(&c.id);
        for k in &c.kit {
            if !CATALOG_IDS.contains(&k.as_str()) {
                return fail(format!("kit spec '{k}' not in CATALOG"));
            }
        }
        if let Some(hooks) = &c.hooks {
            for h in hooks {
                if !vocab.contains(&h.as_str()) {
                    return fail(format!("hook '{h}' not in KIND_TAGS vocabulary"));
                }
            }
        }
        if let Some(FreelancerWant::Band { band_id }) = &c.want {
            if !cast.bands.iter().any(|b| !b.founders && b.id == *band_id) {
                return fail(format!("want.bandId '{band_id}' is not a non-founding band"));
            }
        }
    }
    let names: Vec<String> = cast.companions.iter().map(|c| c.name.to_lowercase()).collect();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let a = &names[i];
            let b = &names[j];
            if prefix4(a) == prefix4(b) {
                return fail(format!(
                    "names '{}'/'{}' share a 4-char prefix",
                    cast.companions[i].name, cast.companions[j].name
                ));
            }
            if a.contains(b.as_str()) || b.contains(a.as_str()) {
                return fail(format!(
                    "names '{}'/'{}' nest",
                    cast.companions[i].name, cast.companions[j].name
                ));
            }
        }
    }
    // Identifiability: every pair differs on a visible axis.
    for i in 0..cast.companions.len() {
        for j in (i + 1)..cast.companions.len() {
            let a = &cast.companions[i];
            let b = &cast.companions[j];
            if hue_delta(hue_of_hex(a.color), hue_of_hex(b.color)) >= HUE_BAR {
                continue;
            }
            if a.look.headwear != b.look.headwear || a.look.build != b.look.build {
                continue;
            }
            return fail(format!("'{}'/'{}' are look-twins (near hue, same headwear+build)", a.id, b.id));
        }
    }
    for bnd in &cast.bands {
        let hw: Vec<u8> = cast
            .companions
            .iter()
            .filter(|c| c.band.as_deref() == Some(bnd.id.as_str()))
            .map(|c| c.look.headwear)
            .collect();
        let mut uniq = hw.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() != hw.len() {
            return fail(format!("band '{}' repeats a headwear family", bnd.id));
        }
    }
    let founders: Vec<&BandDef> = cast.bands.iter().filter(|b| b.founders).collect();
    if founders.len() != 1 {
        return fail(format!("{} founding bands", founders.len()));
    }
    if founders[0].goal_kind != GoalKind::Guild {
        return fail("founders must hold the guild goal".to_string());
    }
    for b in &cast.bands {
        if ids.contains(&b.id.as_str()) || RESERVED.contains(&b.id.as_str()) {
            return fail(format!("band id '{}' collides", b.id));
        }
        let members: Vec<&CompanionDef> = cast
            .companions
            .iter()
            .filter(|c| c.band.as_deref() == Some(b.id.as_str()))
            .collect();
        if members.len() < 2 {
            return fail(format!("band '{}' has {} members", b.id, members.len()));
        }
        if b.founders && (members.len() < 3 || members.len() > 4) {
            return fail("founding band must be 3-4".to_string());
        }
        if b.founders && !members.iter().any(|c| c.kit.iter().any(|k| k == "field_dressing")) {
            return fail("founders need a healer".to_string());
        }
        if let Some(poach) = &b.want_poach_id {
            if !cast.companions.iter().any(|c| c.id == *poach && c.band.is_none()) {
                return fail(format!("poach target '{poach}' is not a freelancer"));
            }
        }
        for h in &b.hooks {
            if !vocab.contains(&h.as_str()) {
                return fail(format!("band hook '{h}' not in vocabulary"));
            }
        }
    }
    if !cast.bands.iter().any(|b| !b.founders && b.goal_kind == GoalKind::Deed) {
        return fail("no camped deed-goal band".to_string());
    }
    let n = cast.companions.len();
    if !(14..=18).contains(&n) {
        return fail(format!("cast size {n} outside 14-18"));
    }
    Ok(())
}
