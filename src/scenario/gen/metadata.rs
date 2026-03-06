//! Hero metadata constants, role definitions, LCG RNG, and DedupSet.

use std::collections::HashSet;

use super::super::types::ScenarioCfg;

// ---------------------------------------------------------------------------
// Hero metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Tank,
    Healer,
    MeleeDps,
    RangedDps,
    Hybrid,
}

pub struct HeroMeta {
    pub name: &'static str,
    pub role: Role,
}

pub const ALL_HEROES: &[HeroMeta] = &[
    HeroMeta { name: "warrior", role: Role::Tank },
    HeroMeta { name: "knight", role: Role::Tank },
    HeroMeta { name: "paladin", role: Role::Tank },
    HeroMeta { name: "warden", role: Role::Tank },
    HeroMeta { name: "templar", role: Role::Tank },
    HeroMeta { name: "cleric", role: Role::Healer },
    HeroMeta { name: "druid", role: Role::Healer },
    HeroMeta { name: "bard", role: Role::Healer },
    HeroMeta { name: "shaman", role: Role::Healer },
    HeroMeta { name: "alchemist", role: Role::Healer },
    HeroMeta { name: "rogue", role: Role::MeleeDps },
    HeroMeta { name: "assassin", role: Role::MeleeDps },
    HeroMeta { name: "berserker", role: Role::MeleeDps },
    HeroMeta { name: "samurai", role: Role::MeleeDps },
    HeroMeta { name: "shadow_dancer", role: Role::MeleeDps },
    HeroMeta { name: "monk", role: Role::MeleeDps },
    HeroMeta { name: "mage", role: Role::RangedDps },
    HeroMeta { name: "ranger", role: Role::RangedDps },
    HeroMeta { name: "pyromancer", role: Role::RangedDps },
    HeroMeta { name: "cryomancer", role: Role::RangedDps },
    HeroMeta { name: "elementalist", role: Role::RangedDps },
    HeroMeta { name: "engineer", role: Role::RangedDps },
    HeroMeta { name: "arcanist", role: Role::RangedDps },
    HeroMeta { name: "blood_mage", role: Role::Hybrid },
    HeroMeta { name: "necromancer", role: Role::Hybrid },
    HeroMeta { name: "warlock", role: Role::Hybrid },
    HeroMeta { name: "witch_doctor", role: Role::Hybrid },
];

pub const ALL_LOL_HEROES: &[HeroMeta] = &[
    HeroMeta { name: "Aatrox", role: Role::MeleeDps },
    HeroMeta { name: "Ahri", role: Role::RangedDps },
    HeroMeta { name: "Akali", role: Role::MeleeDps },
    HeroMeta { name: "Akshan", role: Role::RangedDps },
    HeroMeta { name: "Alistar", role: Role::Tank },
    HeroMeta { name: "Ambessa", role: Role::MeleeDps },
    HeroMeta { name: "Amumu", role: Role::Tank },
    HeroMeta { name: "Anivia", role: Role::RangedDps },
    HeroMeta { name: "Annie", role: Role::RangedDps },
    HeroMeta { name: "Aphelios", role: Role::RangedDps },
    HeroMeta { name: "Ashe", role: Role::RangedDps },
    HeroMeta { name: "AurelionSol", role: Role::RangedDps },
    HeroMeta { name: "Aurora", role: Role::RangedDps },
    HeroMeta { name: "Azir", role: Role::RangedDps },
    HeroMeta { name: "Bard", role: Role::RangedDps },
    HeroMeta { name: "Belveth", role: Role::MeleeDps },
    HeroMeta { name: "Blitzcrank", role: Role::Tank },
    HeroMeta { name: "Brand", role: Role::RangedDps },
    HeroMeta { name: "Braum", role: Role::Tank },
    HeroMeta { name: "Briar", role: Role::MeleeDps },
    HeroMeta { name: "Caitlyn", role: Role::RangedDps },
    HeroMeta { name: "Camille", role: Role::MeleeDps },
    HeroMeta { name: "Cassiopeia", role: Role::RangedDps },
    HeroMeta { name: "Chogath", role: Role::Tank },
    HeroMeta { name: "Corki", role: Role::RangedDps },
    HeroMeta { name: "Darius", role: Role::Tank },
    HeroMeta { name: "Diana", role: Role::MeleeDps },
    HeroMeta { name: "DrMundo", role: Role::Tank },
    HeroMeta { name: "Draven", role: Role::RangedDps },
    HeroMeta { name: "Ekko", role: Role::MeleeDps },
    HeroMeta { name: "Elise", role: Role::MeleeDps },
    HeroMeta { name: "Evelynn", role: Role::MeleeDps },
    HeroMeta { name: "Ezreal", role: Role::RangedDps },
    HeroMeta { name: "Fiddlesticks", role: Role::RangedDps },
    HeroMeta { name: "Fiora", role: Role::MeleeDps },
    HeroMeta { name: "Fizz", role: Role::MeleeDps },
    HeroMeta { name: "Galio", role: Role::Tank },
    HeroMeta { name: "Gangplank", role: Role::Hybrid },
    HeroMeta { name: "Garen", role: Role::Tank },
    HeroMeta { name: "Gnar", role: Role::Tank },
    HeroMeta { name: "Gragas", role: Role::MeleeDps },
    HeroMeta { name: "Graves", role: Role::RangedDps },
    HeroMeta { name: "Gwen", role: Role::MeleeDps },
    HeroMeta { name: "Hecarim", role: Role::Tank },
    HeroMeta { name: "Heimerdinger", role: Role::RangedDps },
    HeroMeta { name: "Hwei", role: Role::RangedDps },
    HeroMeta { name: "Illaoi", role: Role::Tank },
    HeroMeta { name: "Irelia", role: Role::MeleeDps },
    HeroMeta { name: "Ivern", role: Role::RangedDps },
    HeroMeta { name: "Janna", role: Role::RangedDps },
    HeroMeta { name: "JarvanIV", role: Role::Tank },
    HeroMeta { name: "Jax", role: Role::MeleeDps },
    HeroMeta { name: "Jayce", role: Role::RangedDps },
    HeroMeta { name: "Jhin", role: Role::RangedDps },
    HeroMeta { name: "Jinx", role: Role::RangedDps },
    HeroMeta { name: "KSante", role: Role::Tank },
    HeroMeta { name: "Kaisa", role: Role::RangedDps },
    HeroMeta { name: "Kalista", role: Role::RangedDps },
    HeroMeta { name: "Karma", role: Role::RangedDps },
    HeroMeta { name: "Karthus", role: Role::RangedDps },
    HeroMeta { name: "Kassadin", role: Role::MeleeDps },
    HeroMeta { name: "Katarina", role: Role::MeleeDps },
    HeroMeta { name: "Kayle", role: Role::RangedDps },
    HeroMeta { name: "Kayn", role: Role::MeleeDps },
    HeroMeta { name: "Kennen", role: Role::RangedDps },
    HeroMeta { name: "Khazix", role: Role::MeleeDps },
    HeroMeta { name: "Kindred", role: Role::RangedDps },
    HeroMeta { name: "Kled", role: Role::MeleeDps },
    HeroMeta { name: "KogMaw", role: Role::RangedDps },
    HeroMeta { name: "Leblanc", role: Role::MeleeDps },
    HeroMeta { name: "LeeSin", role: Role::MeleeDps },
    HeroMeta { name: "Leona", role: Role::Tank },
    HeroMeta { name: "Lillia", role: Role::MeleeDps },
    HeroMeta { name: "Lissandra", role: Role::RangedDps },
    HeroMeta { name: "Lucian", role: Role::RangedDps },
    HeroMeta { name: "Lulu", role: Role::RangedDps },
    HeroMeta { name: "Lux", role: Role::RangedDps },
    HeroMeta { name: "Malphite", role: Role::Tank },
    HeroMeta { name: "Malzahar", role: Role::RangedDps },
    HeroMeta { name: "Maokai", role: Role::Tank },
    HeroMeta { name: "MasterYi", role: Role::Hybrid },
    HeroMeta { name: "Mel", role: Role::RangedDps },
    HeroMeta { name: "Milio", role: Role::RangedDps },
    HeroMeta { name: "MissFortune", role: Role::RangedDps },
    HeroMeta { name: "MonkeyKing", role: Role::Tank },
    HeroMeta { name: "Mordekaiser", role: Role::MeleeDps },
    HeroMeta { name: "Morgana", role: Role::RangedDps },
    HeroMeta { name: "Naafiri", role: Role::MeleeDps },
    HeroMeta { name: "Nami", role: Role::RangedDps },
    HeroMeta { name: "Nasus", role: Role::Tank },
    HeroMeta { name: "Nautilus", role: Role::Tank },
    HeroMeta { name: "Neeko", role: Role::RangedDps },
    HeroMeta { name: "Nidalee", role: Role::MeleeDps },
    HeroMeta { name: "Nilah", role: Role::MeleeDps },
    HeroMeta { name: "Nocturne", role: Role::MeleeDps },
    HeroMeta { name: "Nunu", role: Role::Tank },
    HeroMeta { name: "Olaf", role: Role::Tank },
    HeroMeta { name: "Orianna", role: Role::RangedDps },
    HeroMeta { name: "Ornn", role: Role::Tank },
    HeroMeta { name: "Pantheon", role: Role::MeleeDps },
    HeroMeta { name: "Poppy", role: Role::Tank },
    HeroMeta { name: "Pyke", role: Role::RangedDps },
    HeroMeta { name: "Qiyana", role: Role::MeleeDps },
    HeroMeta { name: "Quinn", role: Role::RangedDps },
    HeroMeta { name: "Rakan", role: Role::RangedDps },
    HeroMeta { name: "Rammus", role: Role::Tank },
    HeroMeta { name: "RekSai", role: Role::Tank },
    HeroMeta { name: "Rell", role: Role::Tank },
    HeroMeta { name: "Renata", role: Role::RangedDps },
    HeroMeta { name: "Renekton", role: Role::Tank },
    HeroMeta { name: "Rengar", role: Role::MeleeDps },
    HeroMeta { name: "Riven", role: Role::MeleeDps },
    HeroMeta { name: "Rumble", role: Role::MeleeDps },
    HeroMeta { name: "Ryze", role: Role::RangedDps },
    HeroMeta { name: "Samira", role: Role::RangedDps },
    HeroMeta { name: "Sejuani", role: Role::Tank },
    HeroMeta { name: "Senna", role: Role::RangedDps },
    HeroMeta { name: "Seraphine", role: Role::RangedDps },
    HeroMeta { name: "Sett", role: Role::Tank },
    HeroMeta { name: "Shaco", role: Role::MeleeDps },
    HeroMeta { name: "Shen", role: Role::Tank },
    HeroMeta { name: "Shyvana", role: Role::MeleeDps },
    HeroMeta { name: "Singed", role: Role::Tank },
    HeroMeta { name: "Sion", role: Role::Tank },
    HeroMeta { name: "Sivir", role: Role::RangedDps },
    HeroMeta { name: "Skarner", role: Role::Tank },
    HeroMeta { name: "Smolder", role: Role::RangedDps },
    HeroMeta { name: "Sona", role: Role::RangedDps },
    HeroMeta { name: "Soraka", role: Role::Healer },
    HeroMeta { name: "Swain", role: Role::RangedDps },
    HeroMeta { name: "Sylas", role: Role::RangedDps },
    HeroMeta { name: "Syndra", role: Role::RangedDps },
    HeroMeta { name: "TahmKench", role: Role::Tank },
    HeroMeta { name: "Taliyah", role: Role::RangedDps },
    HeroMeta { name: "Talon", role: Role::MeleeDps },
    HeroMeta { name: "Taric", role: Role::Hybrid },
    HeroMeta { name: "Teemo", role: Role::RangedDps },
    HeroMeta { name: "Thresh", role: Role::RangedDps },
    HeroMeta { name: "Tristana", role: Role::RangedDps },
    HeroMeta { name: "Trundle", role: Role::Tank },
    HeroMeta { name: "Tryndamere", role: Role::MeleeDps },
    HeroMeta { name: "TwistedFate", role: Role::RangedDps },
    HeroMeta { name: "Twitch", role: Role::RangedDps },
    HeroMeta { name: "Udyr", role: Role::Tank },
    HeroMeta { name: "Urgot", role: Role::Tank },
    HeroMeta { name: "Varus", role: Role::RangedDps },
    HeroMeta { name: "Vayne", role: Role::RangedDps },
    HeroMeta { name: "Veigar", role: Role::RangedDps },
    HeroMeta { name: "Velkoz", role: Role::RangedDps },
    HeroMeta { name: "Vex", role: Role::RangedDps },
    HeroMeta { name: "Vi", role: Role::MeleeDps },
    HeroMeta { name: "Viego", role: Role::MeleeDps },
    HeroMeta { name: "Viktor", role: Role::RangedDps },
    HeroMeta { name: "Vladimir", role: Role::RangedDps },
    HeroMeta { name: "Volibear", role: Role::Tank },
    HeroMeta { name: "Warwick", role: Role::Tank },
    HeroMeta { name: "Xayah", role: Role::RangedDps },
    HeroMeta { name: "Xerath", role: Role::RangedDps },
    HeroMeta { name: "XinZhao", role: Role::Tank },
    HeroMeta { name: "Yasuo", role: Role::MeleeDps },
    HeroMeta { name: "Yone", role: Role::MeleeDps },
    HeroMeta { name: "Yorick", role: Role::Tank },
    HeroMeta { name: "Yunara", role: Role::RangedDps },
    HeroMeta { name: "Yuumi", role: Role::RangedDps },
    HeroMeta { name: "Zaahen", role: Role::Healer },
    HeroMeta { name: "Zac", role: Role::Tank },
    HeroMeta { name: "Zed", role: Role::MeleeDps },
    HeroMeta { name: "Zeri", role: Role::RangedDps },
    HeroMeta { name: "Ziggs", role: Role::RangedDps },
    HeroMeta { name: "Zilean", role: Role::RangedDps },
    HeroMeta { name: "Zoe", role: Role::RangedDps },
    HeroMeta { name: "Zyra", role: Role::RangedDps },
];

/// Combined pool of all standard + LoL heroes.
pub fn all_heroes_combined() -> Vec<&'static HeroMeta> {
    ALL_HEROES.iter().chain(ALL_LOL_HEROES.iter()).collect()
}

pub const ROOM_TYPES: &[&str] = &["Entry", "Pressure", "Pivot", "Setpiece", "Recovery", "Climax"];

pub fn heroes_by_role(role: Role) -> Vec<&'static str> {
    ALL_HEROES.iter().filter(|h| h.role == role).map(|h| h.name).collect()
}


// ---------------------------------------------------------------------------
// LCG RNG
// ---------------------------------------------------------------------------

pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() % bound as u64) as usize
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }

    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        &slice[self.next_usize(slice.len())]
    }

    pub fn sample_n<T: Clone>(&mut self, pool: &[T], n: usize) -> Vec<T> {
        let mut buf: Vec<T> = pool.to_vec();
        self.shuffle(&mut buf);
        buf.truncate(n.min(buf.len()));
        buf
    }
}

// ---------------------------------------------------------------------------
// Dedup set — canonical key is sorted hero names + enemy_count + difficulty + room
// ---------------------------------------------------------------------------

pub struct DedupSet {
    seen: HashSet<u64>,
}

impl DedupSet {
    pub fn new() -> Self { Self { seen: HashSet::new() } }

    pub fn key(cfg: &ScenarioCfg) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut names = cfg.hero_templates.clone();
        names.sort();
        names.hash(&mut hasher);
        if !cfg.enemy_hero_templates.is_empty() {
            let mut enemy_names = cfg.enemy_hero_templates.clone();
            enemy_names.sort();
            enemy_names.hash(&mut hasher);
        }
        cfg.enemy_count.hash(&mut hasher);
        cfg.difficulty.hash(&mut hasher);
        cfg.room_type.hash(&mut hasher);
        // hp_multiplier bucketed to avoid float issues
        ((cfg.hp_multiplier * 10.0) as u32).hash(&mut hasher);
        hasher.finish()
    }

    pub fn insert(&mut self, cfg: &ScenarioCfg) -> bool {
        self.seen.insert(Self::key(cfg))
    }
}
