use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::fidelity::Fidelity;
use super::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// WorldState — the complete snapshot, immutable during COMPUTE phase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub tick: u64,
    pub rng_state: u64,

    /// All entities — backward-compatible combined view.
    /// Kept in sync with hot/cold arrays. Use `hot`/`cold` for perf-critical paths.
    pub entities: Vec<Entity>,

    /// Cache-line-friendly entity data for high-frequency iteration.
    /// Parallel array: `hot[i]` corresponds to `entities[i]`.
    #[serde(skip)]
    pub hot: Vec<HotEntity>,
    /// Heap-heavy entity data accessed only when needed.
    /// Parallel array: `cold[i]` corresponds to `entities[i]`.
    #[serde(skip)]
    pub cold: Vec<ColdEntity>,

    /// Secondary index: entity_id → index into entities/hot/cold.
    /// Rebuilt by `rebuild_entity_cache()`. Enables O(1) lookup by ID.
    #[serde(skip)]
    pub entity_index: Vec<u32>,
    /// Max entity ID seen (entity_index is sized to max_id+1).
    #[serde(skip)]
    pub max_entity_id: u32,

    /// Group index: contiguous entity ranges by settlement/party.
    /// Entities are sorted by (settlement_id, party_id) so that all entities
    /// at the same settlement are adjacent. Systems iterate a slice instead of
    /// scanning all entities.
    #[serde(skip)]
    pub group_index: GroupIndex,

    /// Active local grids (settlements, encounter zones).
    pub grids: Vec<LocalGrid>,

    /// Regional data (monster populations, faction control).
    pub regions: Vec<RegionState>,

    /// Per-settlement economy (stockpiles, prices, treasury).
    pub settlements: Vec<SettlementState>,

    /// Global economy (total gold supply, trade routes).
    pub economy: EconomyState,

    // --- Campaign-level collections (migrated from headless_campaign) ---

    /// Faction states (governments, guilds, cults, etc.).
    pub factions: Vec<FactionState>,

    /// Active quests currently being pursued.
    pub quests: Vec<Quest>,

    /// Quest board — available quests not yet accepted.
    pub quest_board: Vec<QuestPosting>,

    /// Bond graph between entities. Key = (min_id, max_id), value = strength 0–100.
    pub adventurer_bonds: HashMap<(u32, u32), f32>,

    /// Player guild state.
    pub guild: GuildState,

    /// Narrative chronicle log (bounded ring buffer).
    pub chronicle: Vec<ChronicleEntry>,

    /// Relation graph between entities. Key = (entity_a, entity_b, kind), value = strength.
    pub relations: HashMap<(u32, u32, u8), f32>,

    /// World events log (recent events for system queries, bounded).
    pub world_events: Vec<WorldEvent>,
}

impl WorldState {
    pub fn new(seed: u64) -> Self {
        Self {
            tick: 0,
            rng_state: seed,
            entities: Vec::new(),
            hot: Vec::new(),
            cold: Vec::new(),
            entity_index: Vec::new(),
            max_entity_id: 0,
            group_index: GroupIndex::default(),
            grids: Vec::new(),
            regions: Vec::new(),
            settlements: Vec::new(),
            economy: EconomyState::default(),
            factions: Vec::new(),
            quests: Vec::new(),
            quest_board: Vec::new(),
            adventurer_bonds: HashMap::new(),
            guild: GuildState::default(),
            chronicle: Vec::new(),
            relations: HashMap::new(),
            world_events: Vec::new(),
        }
    }

    /// Full rebuild: sort entities by group, then rebuild hot/cold/index.
    /// Call after structural changes (push, remove).
    pub fn rebuild_all_indices(&mut self) {
        self.rebuild_group_index();
        self.rebuild_entity_cache();
    }

    /// Rebuild hot/cold arrays and entity_index from `entities`.
    /// Does NOT re-sort. Call `rebuild_all_indices` if order may have changed.
    pub fn rebuild_entity_cache(&mut self) {
        let len = self.entities.len();
        self.hot.resize(len, HotEntity {
            id: 0, kind: EntityKind::Npc, team: WorldTeam::Neutral, alive: false,
            level: 0, pos: (0.0, 0.0), hp: 0.0, max_hp: 0.0,
            attack_damage: 0.0, attack_range: 0.0, move_speed: 0.0, grid_id: None,
        });
        self.cold.resize_with(len, ColdEntity::default);

        self.max_entity_id = self.entities.iter().map(|e| e.id).max().unwrap_or(0);
        let idx_len = self.max_entity_id as usize + 1;
        self.entity_index.resize(idx_len, u32::MAX);
        for v in &mut self.entity_index { *v = u32::MAX; }

        for (i, e) in self.entities.iter().enumerate() {
            self.hot[i] = HotEntity {
                id: e.id, kind: e.kind, team: e.team, alive: e.alive,
                level: e.level, pos: e.pos, hp: e.hp, max_hp: e.max_hp,
                attack_damage: e.attack_damage, attack_range: e.attack_range,
                move_speed: e.move_speed, grid_id: e.grid_id,
            };
            self.cold[i] = ColdEntity {
                shield_hp: e.shield_hp, armor: e.armor, magic_resist: e.magic_resist,
                local_pos: e.local_pos,
                status_effects: std::mem::take(&mut self.cold[i].status_effects),
                npc: std::mem::take(&mut self.cold[i].npc),
            };
            // Re-populate cold from entity (only on full rebuild).
            self.cold[i].status_effects.clear();
            self.cold[i].status_effects.extend_from_slice(&e.status_effects);
            self.cold[i].npc = e.npc.clone();

            if (e.id as usize) < idx_len {
                self.entity_index[e.id as usize] = i as u32;
            }
        }
    }

    /// Fast sync: update hot array from entities without touching cold or index.
    /// O(n) flat copy of scalar fields only. No allocations.
    pub fn sync_hot_from_entities(&mut self) {
        for i in 0..self.entities.len().min(self.hot.len()) {
            let e = &self.entities[i];
            let h = &mut self.hot[i];
            h.alive = e.alive;
            h.hp = e.hp;
            h.max_hp = e.max_hp;
            h.pos = e.pos;
            h.attack_damage = e.attack_damage;
            h.attack_range = e.attack_range;
            h.move_speed = e.move_speed;
            h.grid_id = e.grid_id;
            h.level = e.level;
            h.team = e.team;
        }
    }

    /// Sync `entities[i]` back from `hot[i]` and `cold[i]`.
    /// Call after mutating hot/cold arrays directly.
    pub fn sync_entity(&mut self, i: usize) {
        let h = &self.hot[i];
        let c = &self.cold[i];
        let e = &mut self.entities[i];
        e.id = h.id; e.kind = h.kind; e.team = h.team; e.alive = h.alive;
        e.level = h.level; e.pos = h.pos; e.hp = h.hp; e.max_hp = h.max_hp;
        e.attack_damage = h.attack_damage; e.attack_range = h.attack_range;
        e.move_speed = h.move_speed; e.grid_id = h.grid_id;
        e.shield_hp = c.shield_hp; e.armor = c.armor; e.magic_resist = c.magic_resist;
        e.local_pos = c.local_pos;
        // status_effects and npc are reference types — entities[i] keeps its own copy.
        // Only sync scalar fields.
    }

    /// O(1) entity lookup by ID using the secondary index.
    pub fn entity(&self, id: u32) -> Option<&Entity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.entities.len() {
                return Some(&self.entities[idx]);
            }
        }
        None
    }

    pub fn entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.entities.len() {
                return Some(&mut self.entities[idx]);
            }
        }
        None
    }

    /// O(1) hot entity lookup by ID.
    pub fn hot_entity(&self, id: u32) -> Option<&HotEntity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.hot.len() {
                return Some(&self.hot[idx]);
            }
        }
        None
    }

    /// O(1) cold entity lookup by ID.
    pub fn cold_entity(&self, id: u32) -> Option<&ColdEntity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.cold.len() {
                return Some(&self.cold[idx]);
            }
        }
        None
    }

    pub fn settlement(&self, id: u32) -> Option<&SettlementState> {
        self.settlements.iter().find(|s| s.id == id)
    }

    pub fn grid(&self, id: u32) -> Option<&LocalGrid> {
        self.grids.iter().find(|g| g.id == id)
    }

    pub fn faction(&self, id: u32) -> Option<&FactionState> {
        self.factions.iter().find(|f| f.id == id)
    }

    pub fn faction_mut(&mut self, id: u32) -> Option<&mut FactionState> {
        self.factions.iter_mut().find(|f| f.id == id)
    }

    pub fn quest(&self, id: u32) -> Option<&Quest> {
        self.quests.iter().find(|q| q.id == id)
    }

    pub fn quest_mut(&mut self, id: u32) -> Option<&mut Quest> {
        self.quests.iter_mut().find(|q| q.id == id)
    }

    pub fn region_mut(&mut self, id: u32) -> Option<&mut RegionState> {
        self.regions.iter_mut().find(|r| r.id == id)
    }

    pub fn settlement_mut(&mut self, id: u32) -> Option<&mut SettlementState> {
        self.settlements.iter_mut().find(|s| s.id == id)
    }

    /// Look up bond strength between two entities.
    pub fn bond_strength(&self, a: u32, b: u32) -> f32 {
        if a == b { return 0.0; }
        let key = (a.min(b), a.max(b));
        self.adventurer_bonds.get(&key).copied().unwrap_or(0.0)
    }

    /// Look up a relation value between two entities.
    pub fn relation(&self, a: u32, b: u32, kind: RelationKind) -> f32 {
        self.relations.get(&(a, b, kind as u8)).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// GroupIndex — contiguous ranges for per-settlement / per-party iteration
// ---------------------------------------------------------------------------

/// Stores (start, end) ranges into the entity arrays for each group.
/// After `rebuild_group_index()`, entities are sorted by settlement, then party.
/// `settlement_ranges[settlement_id] = (start, end)` means
/// `entities[start..end]` are all entities at that settlement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupIndex {
    /// (start, end) index range per settlement_id.
    pub settlement_ranges: Vec<(u32, u32)>,
    /// (start, end) index range per party_id. 0 = no party.
    pub party_ranges: Vec<(u32, u32)>,
    /// Entities not assigned to any settlement (monsters, travelers).
    pub unaffiliated_range: (u32, u32),
}

impl GroupIndex {
    /// Iterate entity indices for a given settlement.
    pub fn settlement_entities(&self, settlement_id: u32) -> std::ops::Range<usize> {
        let i = settlement_id as usize;
        if i < self.settlement_ranges.len() {
            let (start, end) = self.settlement_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate entity indices for a given party.
    pub fn party_entities(&self, party_id: u32) -> std::ops::Range<usize> {
        let i = party_id as usize;
        if i < self.party_ranges.len() {
            let (start, end) = self.party_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate entity indices not belonging to any settlement.
    pub fn unaffiliated_entities(&self) -> std::ops::Range<usize> {
        self.unaffiliated_range.0 as usize..self.unaffiliated_range.1 as usize
    }
}

impl WorldState {
    /// Sort entities by (settlement_id, party_id) and rebuild all indices.
    ///
    /// After this call:
    /// - `entities`, `hot`, `cold` are sorted so settlement members are contiguous
    /// - `group_index.settlement_ranges` gives slice ranges per settlement
    /// - `entity_index` is rebuilt for O(1) ID lookup into the new order
    ///
    /// Call at init and after structural changes (spawn/despawn).
    pub fn rebuild_group_index(&mut self) {
        let n = self.entities.len();
        if n == 0 { return; }

        // Build sort keys: (settlement_id or MAX, party_id or MAX, original_index).
        // Entities without a settlement sort to the end.
        let mut order: Vec<(u32, u32, usize)> = Vec::with_capacity(n);
        for (i, e) in self.entities.iter().enumerate() {
            let sid = e.npc.as_ref()
                .and_then(|npc| npc.home_settlement_id)
                .unwrap_or(u32::MAX);
            let pid = e.npc.as_ref()
                .and_then(|npc| npc.party_id)
                .unwrap_or(u32::MAX);
            order.push((sid, pid, i));
        }
        order.sort_unstable();

        // Apply the permutation to entities, hot, cold.
        let mut new_entities = Vec::with_capacity(n);
        let mut new_hot = Vec::with_capacity(n);
        let mut new_cold = Vec::with_capacity(n);
        for &(_, _, old_i) in &order {
            new_entities.push(self.entities[old_i].clone());
            if old_i < self.hot.len() {
                new_hot.push(self.hot[old_i]);
            }
            if old_i < self.cold.len() {
                new_cold.push(self.cold[old_i].clone());
            }
        }
        self.entities = new_entities;
        self.hot = new_hot;
        self.cold = new_cold;

        // Rebuild entity_index.
        self.max_entity_id = self.entities.iter().map(|e| e.id).max().unwrap_or(0);
        let idx_len = self.max_entity_id as usize + 1;
        self.entity_index.resize(idx_len, u32::MAX);
        for v in &mut self.entity_index { *v = u32::MAX; }
        for (i, e) in self.entities.iter().enumerate() {
            if (e.id as usize) < idx_len {
                self.entity_index[e.id as usize] = i as u32;
            }
        }

        // Build settlement ranges.
        let max_sid = self.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;
        self.group_index.settlement_ranges.clear();
        self.group_index.settlement_ranges.resize(max_sid, (0, 0));

        let mut i = 0;
        while i < n {
            let sid = order[i].0;
            if sid == u32::MAX { break; } // rest are unaffiliated
            let start = i;
            while i < n && order[i].0 == sid { i += 1; }
            let sid_idx = sid as usize;
            if sid_idx < max_sid {
                self.group_index.settlement_ranges[sid_idx] = (start as u32, i as u32);
            }
        }
        self.group_index.unaffiliated_range = (i as u32, n as u32);

        // Build party ranges.
        // Scan entities for distinct party_ids and record ranges.
        let max_pid = self.entities.iter()
            .filter_map(|e| e.npc.as_ref().and_then(|n| n.party_id))
            .max()
            .unwrap_or(0) as usize + 1;
        self.group_index.party_ranges.clear();
        self.group_index.party_ranges.resize(max_pid, (0, 0));

        // Party ranges require a secondary sort within each settlement group.
        // Since we sorted by (sid, pid), entities with the same pid are contiguous
        // within each settlement. But the same pid could appear in multiple settlements
        // if parties span settlements. For now, just record first/last occurrence.
        let mut party_first = vec![u32::MAX; max_pid];
        let mut party_last = vec![0u32; max_pid];
        for (i, e) in self.entities.iter().enumerate() {
            if let Some(pid) = e.npc.as_ref().and_then(|n| n.party_id) {
                let p = pid as usize;
                if p < max_pid {
                    if party_first[p] == u32::MAX { party_first[p] = i as u32; }
                    party_last[p] = (i + 1) as u32;
                }
            }
        }
        for p in 0..max_pid {
            if party_first[p] != u32::MAX {
                self.group_index.party_ranges[p] = (party_first[p], party_last[p]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entity — unified representation for NPCs, monsters, buildings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    Npc,
    Monster,
    Building,
    Projectile,
}

/// Team affiliation for combat. NPCs on the same team don't attack each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldTeam {
    /// Player-aligned faction.
    Friendly,
    /// Hostile monsters / enemy faction.
    Hostile,
    /// Neutral (won't attack, can be attacked).
    Neutral,
}

impl Default for WorldTeam {
    fn default() -> Self {
        WorldTeam::Neutral
    }
}

/// Hot entity data — 48 bytes, cache-line friendly.
/// Iterated by every system every tick. Keep this small.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct HotEntity {
    pub id: u32,
    pub kind: EntityKind,
    pub team: WorldTeam,
    pub alive: bool,
    pub level: u32,
    pub pos: (f32, f32),
    pub hp: f32,
    pub max_hp: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    pub grid_id: Option<u32>,
}

/// Cold entity data — large, heap-heavy. Only accessed when a system
/// needs NPC-specific or detailed combat data. Indexed in parallel with HotEntity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdEntity {
    pub shield_hp: f32,
    pub armor: f32,
    pub magic_resist: f32,
    pub local_pos: Option<(f32, f32)>,
    pub status_effects: Vec<StatusEffect>,
    pub npc: Option<NpcData>,
}

/// Combined view of an entity for backward-compatible access.
/// This is NOT stored — it's constructed on demand from hot + cold refs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub team: WorldTeam,
    pub pos: (f32, f32),
    pub grid_id: Option<u32>,
    pub local_pos: Option<(f32, f32)>,
    pub alive: bool,
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub armor: f32,
    pub magic_resist: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    pub level: u32,
    pub status_effects: Vec<StatusEffect>,
    pub npc: Option<NpcData>,
}

impl Entity {
    /// Split into hot + cold components.
    pub fn split(&self) -> (HotEntity, ColdEntity) {
        (
            HotEntity {
                id: self.id, kind: self.kind, team: self.team, alive: self.alive,
                level: self.level, pos: self.pos, hp: self.hp, max_hp: self.max_hp,
                attack_damage: self.attack_damage, attack_range: self.attack_range,
                move_speed: self.move_speed, grid_id: self.grid_id,
            },
            ColdEntity {
                shield_hp: self.shield_hp, armor: self.armor, magic_resist: self.magic_resist,
                local_pos: self.local_pos, status_effects: self.status_effects.clone(),
                npc: self.npc.clone(),
            },
        )
    }

    /// Reconstruct from hot + cold.
    pub fn from_parts(hot: &HotEntity, cold: &ColdEntity) -> Self {
        Entity {
            id: hot.id, kind: hot.kind, team: hot.team, pos: hot.pos,
            grid_id: hot.grid_id, local_pos: cold.local_pos, alive: hot.alive,
            hp: hot.hp, max_hp: hot.max_hp, shield_hp: cold.shield_hp,
            armor: cold.armor, magic_resist: cold.magic_resist,
            attack_damage: hot.attack_damage, attack_range: hot.attack_range,
            move_speed: hot.move_speed, level: hot.level,
            status_effects: cold.status_effects.clone(), npc: cold.npc.clone(),
        }
    }

    pub fn new_npc(id: u32, pos: (f32, f32)) -> Self {
        Self {
            id,
            kind: EntityKind::Npc,
            team: WorldTeam::Friendly,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 100.0,
            max_hp: 100.0,
            shield_hp: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
            attack_damage: 10.0,
            attack_range: 1.5,
            move_speed: 3.0,
            level: 1,
            status_effects: Vec::new(),
            npc: Some(NpcData::default()),
        }
    }

    pub fn new_monster(id: u32, pos: (f32, f32), level: u32) -> Self {
        Self {
            id,
            kind: EntityKind::Monster,
            team: WorldTeam::Hostile,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 50.0 + level as f32 * 20.0,
            max_hp: 50.0 + level as f32 * 20.0,
            shield_hp: 0.0,
            armor: 5.0,
            magic_resist: 0.0,
            attack_damage: 5.0 + level as f32 * 3.0,
            attack_range: 1.5,
            move_speed: 2.0,
            level,
            status_effects: Vec::new(),
            npc: None,
        }
    }

    pub fn new_building(id: u32, pos: (f32, f32)) -> Self {
        Self {
            id,
            kind: EntityKind::Building,
            team: WorldTeam::Neutral,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 500.0,
            max_hp: 500.0,
            shield_hp: 0.0,
            armor: 20.0,
            magic_resist: 0.0,
            attack_damage: 0.0,
            attack_range: 0.0,
            move_speed: 0.0,
            level: 0,
            status_effects: Vec::new(),
            npc: None,
        }
    }
}

// ---------------------------------------------------------------------------
// StatusEffect — live status effect on an entity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEffect {
    pub kind: StatusEffectKind,
    pub source_id: u32,
    pub remaining_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatusEffectKind {
    Stun,
    Slow { factor: f32 },
    Root,
    Silence,
    Dot { damage_per_tick: f32, tick_interval_ms: u32, tick_elapsed_ms: u32 },
    Hot { heal_per_tick: f32, tick_interval_ms: u32, tick_elapsed_ms: u32 },
    Buff { stat: String, factor: f32 },
    Debuff { stat: String, factor: f32 },
}

impl StatusEffectKind {
    /// Discriminant for dedup: two effects with the same discriminant are the "same kind".
    pub fn discriminant(&self) -> u8 {
        match self {
            StatusEffectKind::Stun => 0,
            StatusEffectKind::Slow { .. } => 1,
            StatusEffectKind::Root => 2,
            StatusEffectKind::Silence => 3,
            StatusEffectKind::Dot { .. } => 4,
            StatusEffectKind::Hot { .. } => 5,
            StatusEffectKind::Buff { .. } => 6,
            StatusEffectKind::Debuff { .. } => 7,
        }
    }
}

// ---------------------------------------------------------------------------
// NpcData — NPC-specific economic/social data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcData {
    /// Links to campaign Adventurer id.
    pub adventurer_id: u32,
    pub gold: f32,
    pub home_settlement_id: Option<u32>,
    pub economic_intent: EconomicIntent,
    pub price_knowledge: Vec<PriceReport>,
    pub carried_goods: [f32; NUM_COMMODITIES],
    pub class_tags: Vec<String>,
    /// What commodities this NPC produces (commodity_index, rate_per_tick).
    pub behavior_production: Vec<(usize, f32)>,

    // --- Campaign system fields (migrated from headless_campaign::Adventurer) ---

    /// 0–100. Combat morale.
    pub morale: f32,
    /// 0–100. Psychological stress.
    pub stress: f32,
    /// 0–100. Physical fatigue.
    pub fatigue: f32,
    /// 0–100. Loyalty to guild/faction.
    pub loyalty: f32,
    /// 0–100. Injury severity (>=90 incapacitated).
    pub injury: f32,
    /// Experience points.
    pub xp: u32,
    /// Hero archetype name (e.g. "knight", "ranger", "mage").
    pub archetype: String,
    /// Party membership, if any.
    pub party_id: Option<u32>,
    /// Faction membership, if any.
    pub faction_id: Option<u32>,
    /// Current mood (index into a mood enum). 0 = Neutral.
    pub mood: u8,
    /// Active fear type indices.
    pub fears: Vec<u8>,
    /// Legendary deed type indices.
    pub deeds: Vec<u8>,
    /// 0–100. Resolve under pressure.
    pub resolve: f32,
    /// Guild relationship score (-100 to 100).
    pub guild_relationship: f32,
}

impl Default for NpcData {
    fn default() -> Self {
        Self {
            adventurer_id: 0,
            gold: 0.0,
            home_settlement_id: None,
            economic_intent: EconomicIntent::Idle,
            price_knowledge: Vec::new(),
            carried_goods: [0.0; NUM_COMMODITIES],
            class_tags: Vec::new(),
            behavior_production: Vec::new(),
            morale: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            loyalty: 50.0,
            injury: 0.0,
            xp: 0,
            archetype: String::new(),
            party_id: None,
            faction_id: None,
            mood: 0,
            fears: Vec::new(),
            deeds: Vec::new(),
            resolve: 50.0,
            guild_relationship: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EconomicIntent {
    Idle,
    Produce,
    Trade { destination_settlement_id: u32 },
    Buy { commodity: usize },
    Sell { commodity: usize },
    Travel { destination: (f32, f32) },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceReport {
    pub settlement_id: u32,
    pub prices: [f32; NUM_COMMODITIES],
    pub tick_observed: u64,
}

// ---------------------------------------------------------------------------
// LocalGrid — a spatial grid for a settlement or encounter zone
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalGrid {
    pub id: u32,
    pub fidelity: Fidelity,
    pub center: (f32, f32),
    pub radius: f32,
    /// Entity IDs currently on this grid.
    pub entity_ids: Vec<u32>,
}

impl LocalGrid {
    pub fn has_hostiles(&self, state: &super::WorldState) -> bool {
        self.entity_ids.iter().any(|id| {
            state.entity(*id)
                .map(|e| e.kind == EntityKind::Monster && e.alive)
                .unwrap_or(false)
        })
    }

    pub fn has_friendlies(&self, state: &super::WorldState) -> bool {
        self.entity_ids.iter().any(|id| {
            state.entity(*id)
                .map(|e| e.kind == EntityKind::Npc && e.alive)
                .unwrap_or(false)
        })
    }
}

// ---------------------------------------------------------------------------
// SettlementState — per-settlement economy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementState {
    pub id: u32,
    pub name: String,
    pub pos: (f32, f32),
    pub grid_id: Option<u32>,
    pub stockpile: [f32; NUM_COMMODITIES],
    pub prices: [f32; NUM_COMMODITIES],
    pub treasury: f32,
    pub population: u32,

    // --- Campaign system fields ---

    /// Owning faction, if any.
    pub faction_id: Option<u32>,
    /// 0–1. Threat from nearby monsters/enemies.
    pub threat_level: f32,
    /// 0–5. Building/upgrade level.
    pub infrastructure_level: f32,
}

impl SettlementState {
    pub fn new(id: u32, name: String, pos: (f32, f32)) -> Self {
        Self {
            id,
            name,
            pos,
            grid_id: None,
            stockpile: [0.0; NUM_COMMODITIES],
            prices: [1.0; NUM_COMMODITIES],
            treasury: 0.0,
            population: 0,
            faction_id: None,
            threat_level: 0.0,
            infrastructure_level: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RegionState — regional monster population and faction control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionState {
    pub id: u32,
    pub name: String,
    pub monster_density: f32,
    pub faction_id: Option<u32>,
    pub threat_level: f32,

    // --- Campaign system fields ---

    /// 0–1. Civil unrest level.
    pub unrest: f32,
    /// 0–1. Faction control strength.
    pub control: f32,
}

// ---------------------------------------------------------------------------
// EconomyState — global economy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EconomyState {
    pub total_gold_supply: f32,
    pub total_commodities: [f32; NUM_COMMODITIES],
}

// ---------------------------------------------------------------------------
// FactionState — political/military faction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionState {
    pub id: u32,
    pub name: String,
    /// Relationship to the player guild (-100 to 100).
    pub relationship_to_guild: f32,
    pub military_strength: f32,
    pub max_military_strength: f32,
    pub territory_size: u32,
    pub diplomatic_stance: DiplomaticStance,
    /// Treasury (faction gold).
    pub treasury: f32,
    /// Faction-to-faction war targets.
    pub at_war_with: Vec<u32>,
    /// Running coup-risk score (0.0–1.0).
    pub coup_risk: f32,
    /// Escalation level (0=normal, 5=maximum).
    pub escalation_level: u32,
    /// Tech/research level.
    pub tech_level: u32,
    /// Recent actions log (bounded).
    pub recent_actions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiplomaticStance {
    Friendly,
    Neutral,
    Hostile,
    AtWar,
    Coalition,
}

impl Default for DiplomaticStance {
    fn default() -> Self {
        DiplomaticStance::Neutral
    }
}

// ---------------------------------------------------------------------------
// GuildState — player guild
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildState {
    pub gold: f32,
    pub supplies: f32,
    /// 0–100. Affects quest quality and pricing.
    pub reputation: f32,
    /// Guild tier (0–5).
    pub tier: u32,
    /// Credit rating for loans.
    pub credit_rating: f32,
    /// Active quest capacity.
    pub active_quest_capacity: u32,
}

impl Default for GuildState {
    fn default() -> Self {
        Self {
            gold: 100.0,
            supplies: 50.0,
            reputation: 10.0,
            tier: 0,
            credit_rating: 50.0,
            active_quest_capacity: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Quest — an active quest being pursued
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u32,
    pub name: String,
    pub quest_type: QuestType,
    /// Assigned party entity IDs.
    pub party_member_ids: Vec<u32>,
    /// Destination position.
    pub destination: (f32, f32),
    /// 0–1. Progress toward completion.
    pub progress: f32,
    /// Current quest phase.
    pub status: QuestStatus,
    /// Tick when quest was accepted.
    pub accepted_tick: u64,
    /// Tick deadline (0 = no deadline).
    pub deadline_tick: u64,
    /// Threat level of the quest.
    pub threat_level: f32,
    /// Gold reward on completion.
    pub reward_gold: f32,
    /// XP reward on completion.
    pub reward_xp: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestType {
    Hunt,
    Escort,
    Deliver,
    Explore,
    Defend,
    Gather,
    Rescue,
    Assassinate,
    Diplomacy,
    Custom,
}

impl Default for QuestType {
    fn default() -> Self {
        QuestType::Hunt
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestStatus {
    /// Quest accepted, party traveling to destination.
    Traveling,
    /// Party is at the quest location, working on it.
    InProgress,
    /// Quest completed successfully.
    Completed,
    /// Quest failed.
    Failed,
    /// Party is returning to base.
    Returning,
}

impl Default for QuestStatus {
    fn default() -> Self {
        QuestStatus::Traveling
    }
}

// ---------------------------------------------------------------------------
// QuestPosting — a quest available on the board
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestPosting {
    pub id: u32,
    pub name: String,
    pub quest_type: QuestType,
    pub destination: (f32, f32),
    pub threat_level: f32,
    pub reward_gold: f32,
    pub reward_xp: u32,
    /// Tick when posting expires.
    pub expires_tick: u64,
}

// ---------------------------------------------------------------------------
// ChronicleEntry — narrative log entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub tick: u64,
    pub category: ChronicleCategory,
    pub text: String,
    /// Entity IDs involved.
    pub entity_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChronicleCategory {
    Battle,
    Quest,
    Diplomacy,
    Economy,
    Death,
    Discovery,
    Crisis,
    Achievement,
    Narrative,
}

// ---------------------------------------------------------------------------
// WorldEvent — events recorded during a tick
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldEvent {
    /// Generic event with a category tag and description.
    Generic { category: ChronicleCategory, text: String },
    /// Entity died.
    EntityDied { entity_id: u32, cause: String },
    /// Quest status changed.
    QuestChanged { quest_id: u32, new_status: QuestStatus },
    /// Faction relation changed.
    FactionRelationChanged { faction_id: u32, old: f32, new: f32 },
    /// Region ownership changed.
    RegionOwnerChanged { region_id: u32, old_owner: Option<u32>, new_owner: Option<u32> },
    /// Bond grief event.
    BondGrief { entity_id: u32, dead_id: u32, bond_strength: f32 },
    /// Season changed.
    SeasonChanged { new_season: u8 },
    /// Battle started.
    BattleStarted { grid_id: u32, participants: Vec<u32> },
    /// Battle ended.
    BattleEnded { grid_id: u32, victor_team: WorldTeam },
}

// ---------------------------------------------------------------------------
// EntityField — fields on Entity/NpcData addressable by delta
// ---------------------------------------------------------------------------

/// Addressable scalar fields on an entity's NpcData.
/// Used by `UpdateEntityField` delta for consolidated per-entity updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityField {
    Morale,
    Stress,
    Fatigue,
    Loyalty,
    Injury,
    Resolve,
    GuildRelationship,
    Gold,
    Hp,
    MaxHp,
    ShieldHp,
    Armor,
    MagicResist,
    AttackDamage,
    AttackRange,
    MoveSpeed,
}

// ---------------------------------------------------------------------------
// FactionField — addressable fields on FactionState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactionField {
    RelationshipToGuild,
    MilitaryStrength,
    Treasury,
    CoupRisk,
    EscalationLevel,
    TechLevel,
}

// ---------------------------------------------------------------------------
// RegionField — addressable fields on RegionState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionField {
    MonsterDensity,
    ThreatLevel,
    Unrest,
    Control,
}

// ---------------------------------------------------------------------------
// SettlementField — addressable fields on SettlementState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettlementField {
    Treasury,
    Population,
    ThreatLevel,
    InfrastructureLevel,
}

// ---------------------------------------------------------------------------
// RelationKind — types of entity-to-entity relations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RelationKind {
    /// General relationship (-100 to 100).
    Relationship = 0,
    /// Bond strength (0–100).
    Bond = 1,
    /// Romance level (0–100).
    Romance = 2,
    /// Rivalry intensity (0–100).
    Rivalry = 3,
    /// Grudge intensity (0–100).
    Grudge = 4,
    /// Mentor-mentee (0–100).
    Mentorship = 5,
}

// ---------------------------------------------------------------------------
// QuestDelta — quest lifecycle changes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestDelta {
    /// Advance progress by a fraction.
    AdvanceProgress { amount: f32 },
    /// Change quest status.
    SetStatus { status: QuestStatus },
    /// Add a member to the quest party.
    AddMember { entity_id: u32 },
    /// Remove a member from the quest party.
    RemoveMember { entity_id: u32 },
    /// Complete the quest (triggers rewards).
    Complete,
    /// Fail the quest.
    Fail,
}
