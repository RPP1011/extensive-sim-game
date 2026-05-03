//! Snapshot file format. 64-byte header + field blocks.
//!
//! Layout:
//! - magic            : [u8; 8]   offset 0..8
//! - schema_hash      : [u8; 32]  offset 8..40
//! - tick             : u32 LE    offset 40..44
//! - seed             : u64 LE    offset 44..52
//! - format_version   : u16 LE    offset 52..54
//! - reserved         : [u8; 10]  offset 54..64  (future fields)
//!
//! After the header, a body block serialises the `SimState` SoA + pool +
//! the pod-shaped cold collections, followed by `EventRing` metadata (cap +
//! monotonic cursors; entry contents are intentionally **not** snapshotted
//! for v1 — see "Coverage gaps" below).
//!
//! # Coverage gaps (v1)
//!
//! Fields intentionally NOT snapshotted by v1. On load these are default-
//! constructed or rebuilt from what was serialised. Add coverage by bumping
//! `FORMAT_VERSION` and registering a migration.
//!
//! - `cold_channels` — `Vec<Option<ChannelSet>>`. Restored empty; callers
//!   re-attach species capabilities from `CreatureType` if needed.
//! - `EventRing` entries — only the monotonic cursors (`cap`, `current_tick`,
//!   `next_seq`, `total_pushed`, `dispatched`) are restored. Historical
//!   events are dropped; `replayable_sha256()` will differ from a
//!   straight-run equivalent. Task 12 acceptance documents this.
//! - `SimState::views` — derived per-tick via view folds; reset empty.
//! - `SimState::ability_registry` — caller-supplied; reset empty.
//! - `SimState::terrain` — `Arc<dyn>` injected by the caller; resets to
//!   `FlatPlane`.
//! - `SimState::config` — loaded separately (e.g. from TOML); resets to
//!   `Config::default()`. Callers that tuned `config` at init must
//!   re-apply it after `load_snapshot`.
//! - `SimState::spatial` — rebuilt from `hot_pos` + `hot_alive` +
//!   `hot_movement_mode` on load.

use engine_data::entities::CreatureType;
use crate::event::{EventLike, EventRing};
use crate::ids::{AgentId, GroupId};
use crate::state::agent_types::{
    ClassSlot, Creditor, GroupRole, Inventory, Membership, Relationship,
    StatusEffect, StatusEffectKind,
};
use crate::state::{MovementMode, SimState};
use glam::Vec3;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"WSIMSV01";
pub const FORMAT_VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub magic: [u8; 8],
    pub schema_hash: [u8; 32],
    pub tick: u32,
    pub seed: u64,
    pub format_version: u16,
    pub reserved: [u8; 10],
}

impl SnapshotHeader {
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..8].copy_from_slice(&self.magic);
        out[8..40].copy_from_slice(&self.schema_hash);
        out[40..44].copy_from_slice(&self.tick.to_le_bytes());
        out[44..52].copy_from_slice(&self.seed.to_le_bytes());
        out[52..54].copy_from_slice(&self.format_version.to_le_bytes());
        out[54..64].copy_from_slice(&self.reserved);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < HEADER_BYTES {
            return Err(SnapshotError::ShortHeader);
        }
        if &bytes[0..8] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        let mut schema_hash = [0u8; 32];
        schema_hash.copy_from_slice(&bytes[8..40]);
        let tick = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let seed = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let format_version = u16::from_le_bytes(bytes[52..54].try_into().unwrap());
        let mut reserved = [0u8; 10];
        reserved.copy_from_slice(&bytes[54..64]);
        Ok(Self {
            magic,
            schema_hash,
            tick,
            seed,
            format_version,
            reserved,
        })
    }
}

#[derive(Debug)]
pub enum SnapshotError {
    BadMagic,
    ShortHeader,
    SchemaMismatch {
        expected: [u8; 32],
        found: [u8; 32],
    },
    UnsupportedFormatVersion {
        found: u16,
        expected: u16,
    },
    Truncated(&'static str),
    InvalidDiscriminant {
        what: &'static str,
        value: u8,
    },
    Io(std::io::Error),
    MigrationFailed(&'static str),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic bytes — not a snapshot file"),
            Self::ShortHeader => write!(f, "file shorter than 64-byte header"),
            Self::SchemaMismatch { expected, found } => write!(
                f,
                "schema hash mismatch — expected {:02x?}…, found {:02x?}…",
                &expected[..4],
                &found[..4],
            ),
            Self::UnsupportedFormatVersion { found, expected } => write!(
                f,
                "unsupported snapshot format version {} (this engine writes v{})",
                found, expected,
            ),
            Self::Truncated(what) => write!(f, "snapshot truncated while reading {}", what),
            Self::InvalidDiscriminant { what, value } => write!(
                f,
                "invalid {} discriminant: {}",
                what, value,
            ),
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::MigrationFailed(msg) => write!(f, "migration failed: {}", msg),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        SnapshotError::Io(e)
    }
}

// ---------- byte writer / reader helpers (private) ----------

struct W {
    buf: Vec<u8>,
}

impl W {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }
    fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn opt_agent(&mut self, v: Option<AgentId>) {
        match v {
            Some(id) => {
                self.u8(1);
                self.u32(id.raw());
            }
            None => {
                self.u8(0);
                self.u32(0);
            }
        }
    }
    fn opt_u32(&mut self, v: Option<u32>) {
        match v {
            Some(x) => {
                self.u8(1);
                self.u32(x);
            }
            None => {
                self.u8(0);
                self.u32(0);
            }
        }
    }
    fn opt_vec3(&mut self, v: Option<Vec3>) {
        match v {
            Some(p) => {
                self.u8(1);
                self.vec3(p);
            }
            None => {
                self.u8(0);
                self.vec3(Vec3::ZERO);
            }
        }
    }
    fn opt_creature(&mut self, v: Option<CreatureType>) {
        match v {
            Some(c) => {
                self.u8(1);
                self.u8(c as u8);
            }
            None => {
                self.u8(0);
                self.u8(0);
            }
        }
    }
}

struct R<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> R<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], SnapshotError> {
        if self.pos.checked_add(n).map_or(true, |end| end > self.buf.len()) {
            return Err(SnapshotError::Truncated(what));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self, what: &'static str) -> Result<u8, SnapshotError> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &'static str) -> Result<u16, SnapshotError> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn i16(&mut self, what: &'static str) -> Result<i16, SnapshotError> {
        Ok(i16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn i32(&mut self, what: &'static str) -> Result<i32, SnapshotError> {
        Ok(i32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn f32(&mut self, what: &'static str) -> Result<f32, SnapshotError> {
        Ok(f32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn bool(&mut self, what: &'static str) -> Result<bool, SnapshotError> {
        Ok(self.u8(what)? != 0)
    }
    fn vec3(&mut self, what: &'static str) -> Result<Vec3, SnapshotError> {
        let x = self.f32(what)?;
        let y = self.f32(what)?;
        let z = self.f32(what)?;
        Ok(Vec3::new(x, y, z))
    }
    fn opt_agent(&mut self, what: &'static str) -> Result<Option<AgentId>, SnapshotError> {
        let present = self.u8(what)?;
        let raw = self.u32(what)?;
        if present == 0 {
            Ok(None)
        } else {
            Ok(AgentId::new(raw))
        }
    }
    fn opt_u32(&mut self, what: &'static str) -> Result<Option<u32>, SnapshotError> {
        let present = self.u8(what)?;
        let v = self.u32(what)?;
        if present == 0 {
            Ok(None)
        } else {
            Ok(Some(v))
        }
    }
    fn opt_vec3(&mut self, what: &'static str) -> Result<Option<Vec3>, SnapshotError> {
        let present = self.u8(what)?;
        let p = self.vec3(what)?;
        if present == 0 {
            Ok(None)
        } else {
            Ok(Some(p))
        }
    }
    fn opt_creature(&mut self, what: &'static str) -> Result<Option<CreatureType>, SnapshotError> {
        let present = self.u8(what)?;
        let disc = self.u8(what)?;
        if present == 0 {
            return Ok(None);
        }
        Ok(Some(creature_from_disc(disc)?))
    }
}

fn movement_mode_disc(m: MovementMode) -> u8 {
    match m {
        MovementMode::Walk => 0,
        MovementMode::Climb => 1,
        MovementMode::Fly => 2,
        MovementMode::Swim => 3,
        MovementMode::Fall => 4,
    }
}

fn movement_mode_from_disc(v: u8) -> Result<MovementMode, SnapshotError> {
    match v {
        0 => Ok(MovementMode::Walk),
        1 => Ok(MovementMode::Climb),
        2 => Ok(MovementMode::Fly),
        3 => Ok(MovementMode::Swim),
        4 => Ok(MovementMode::Fall),
        v => Err(SnapshotError::InvalidDiscriminant {
            what: "MovementMode",
            value: v,
        }),
    }
}

fn creature_from_disc(v: u8) -> Result<CreatureType, SnapshotError> {
    match v {
        0 => Ok(CreatureType::Human),
        1 => Ok(CreatureType::Wolf),
        2 => Ok(CreatureType::Deer),
        3 => Ok(CreatureType::Dragon),
        v => Err(SnapshotError::InvalidDiscriminant {
            what: "CreatureType",
            value: v,
        }),
    }
}

fn status_kind_from_disc(v: u8) -> Result<StatusEffectKind, SnapshotError> {
    match v {
        0 => Ok(StatusEffectKind::Stun),
        1 => Ok(StatusEffectKind::Slow),
        2 => Ok(StatusEffectKind::Root),
        3 => Ok(StatusEffectKind::Silence),
        4 => Ok(StatusEffectKind::Dot),
        5 => Ok(StatusEffectKind::Hot),
        6 => Ok(StatusEffectKind::Buff),
        7 => Ok(StatusEffectKind::Debuff),
        v => Err(SnapshotError::InvalidDiscriminant {
            what: "StatusEffectKind",
            value: v,
        }),
    }
}

fn group_role_from_disc(v: u8) -> Result<GroupRole, SnapshotError> {
    match v {
        0 => Ok(GroupRole::Member),
        1 => Ok(GroupRole::Officer),
        2 => Ok(GroupRole::Leader),
        3 => Ok(GroupRole::Founder),
        4 => Ok(GroupRole::Apprentice),
        5 => Ok(GroupRole::Outcast),
        v => Err(SnapshotError::InvalidDiscriminant {
            what: "GroupRole",
            value: v,
        }),
    }
}

// ---------- write state ----------

fn write_state(w: &mut W, state: &SimState) {
    let cap = state.agent_cap();
    let cap_usize = cap as usize;

    // Pool state.
    w.u32(cap);
    w.u32(state.pool_next_raw());
    let alive = state.hot_alive();
    debug_assert_eq!(alive.len(), cap_usize);
    for &a in alive {
        w.bool(a);
    }
    let freelist: Vec<u32> = state.pool_freelist_iter().collect();
    w.u32(freelist.len() as u32);
    for raw in freelist {
        w.u32(raw);
    }

    // Hot scalars (cap-length).
    for &p in state.hot_pos() {
        w.vec3(p);
    }
    for &v in state.hot_hp() {
        w.f32(v);
    }
    for &v in state.hot_max_hp() {
        w.f32(v);
    }
    for &m in state.hot_movement_mode() {
        w.u8(movement_mode_disc(m));
    }
    for &v in state.hot_level() {
        w.u32(v);
    }
    for &v in state.hot_move_speed() {
        w.f32(v);
    }
    for &v in state.hot_move_speed_mult() {
        w.f32(v);
    }
    for &v in state.hot_shield_hp() {
        w.f32(v);
    }
    for &v in state.hot_armor() {
        w.f32(v);
    }
    for &v in state.hot_magic_resist() {
        w.f32(v);
    }
    for &v in state.hot_attack_damage() {
        w.f32(v);
    }
    for &v in state.hot_attack_range() {
        w.f32(v);
    }
    for &v in state.hot_mana() {
        w.f32(v);
    }
    for &v in state.hot_max_mana() {
        w.f32(v);
    }
    for &v in state.hot_hunger() {
        w.f32(v);
    }
    for &v in state.hot_thirst() {
        w.f32(v);
    }
    for &v in state.hot_rest_timer() {
        w.f32(v);
    }
    for &v in state.hot_safety() {
        w.f32(v);
    }
    for &v in state.hot_shelter() {
        w.f32(v);
    }
    for &v in state.hot_social() {
        w.f32(v);
    }
    for &v in state.hot_purpose() {
        w.f32(v);
    }
    for &v in state.hot_esteem() {
        w.f32(v);
    }
    for &v in state.hot_risk_tolerance() {
        w.f32(v);
    }
    for &v in state.hot_social_drive() {
        w.f32(v);
    }
    for &v in state.hot_ambition() {
        w.f32(v);
    }
    for &v in state.hot_altruism() {
        w.f32(v);
    }
    for &v in state.hot_curiosity() {
        w.f32(v);
    }
    for &v in state.hot_engaged_with() {
        w.opt_agent(v);
    }
    for &v in state.hot_stun_expires_at_tick() {
        w.u32(v);
    }
    for &v in state.hot_slow_expires_at_tick() {
        w.u32(v);
    }
    for &v in state.hot_slow_factor_q8() {
        w.i16(v);
    }
    for &v in state.hot_cooldown_next_ready_tick() {
        w.u32(v);
    }

    // Cold scalars (cap-length, one tuple per slot).
    for slot in 0..cap_usize {
        let id = AgentId::new((slot + 1) as u32).unwrap();
        w.opt_creature(state.agent_creature_type(id));
        w.opt_u32(state.agent_spawn_tick(id));
        w.opt_u32(state.agent_grid_id(id));
        w.opt_vec3(state.agent_local_pos(id));
        w.opt_vec3(state.agent_move_target(id));
    }

    // Cold collections (pod-shaped SmallVec / array payloads).
    for v in state.cold_status_effects() {
        w.u32(v.len() as u32);
        for fx in v.iter() {
            w.u8(fx.kind as u8);
            w.u32(fx.source.raw());
            w.u32(fx.remaining_ticks);
            w.i16(fx.payload_q8);
        }
    }
    for v in state.cold_memberships() {
        w.u32(v.len() as u32);
        for m in v.iter() {
            w.u32(m.group.raw());
            w.u8(m.role as u8);
            w.u32(m.joined_tick);
            w.i16(m.standing_q8);
        }
    }
    for inv in state.cold_inventory() {
        w.i32(inv.gold);
        for c in inv.commodities.iter() {
            w.u16(*c);
        }
    }
    for v in state.cold_relationships() {
        w.u32(v.len() as u32);
        for r in v.iter() {
            w.u32(r.other.raw());
            w.i16(r.valence_q8);
            w.u32(r.tenure_ticks);
        }
    }
    for slots in state.cold_class_definitions() {
        for s in slots {
            w.u32(s.class_tag);
            w.u8(s.level);
        }
    }
    for v in state.cold_creditor_ledger() {
        w.u32(v.len() as u32);
        for c in v.iter() {
            w.u32(c.creditor.raw());
            w.u32(c.amount);
        }
    }
    for lineage in state.cold_mentor_lineage() {
        for link in lineage {
            w.opt_agent(*link);
        }
    }
    for row in &state.ability_cooldowns {
        for cd in row.iter() {
            w.u32(*cd);
        }
    }
}

// ---------- read state ----------

fn read_state(
    r: &mut R<'_>,
    header: SnapshotHeader,
) -> Result<SimState, SnapshotError> {
    let cap = r.u32("agent_cap")?;
    let next_raw = r.u32("pool_next_raw")?;
    let cap_usize = cap as usize;

    let mut alive = Vec::with_capacity(cap_usize);
    for _ in 0..cap_usize {
        alive.push(r.bool("pool_alive")?);
    }
    let freelist_len = r.u32("pool_freelist_len")? as usize;
    let mut freelist = Vec::with_capacity(freelist_len);
    for _ in 0..freelist_len {
        freelist.push(r.u32("pool_freelist")?);
    }

    // Allocate state with matching cap + seed; we'll overwrite hot/cold fields.
    let mut state = SimState::new(cap, header.seed);
    state.tick = header.tick;

    // Restore pool + sync hot_alive.
    state.restore_pool_from_parts(next_raw, alive.clone(), freelist);
    state.hot_alive_mut_slice().copy_from_slice(&alive);

    // Hot scalars.
    for slot in 0..cap_usize {
        state.hot_pos_mut_slice()[slot] = r.vec3("hot_pos")?;
    }
    read_f32_slice(r, cap_usize, "hot_hp", state.hot_hp_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_max_hp", state.hot_max_hp_mut_slice())?;
    for slot in 0..cap_usize {
        state.hot_movement_mode_mut_slice()[slot] = movement_mode_from_disc(r.u8("hot_movement_mode")?)?;
    }
    read_u32_slice(r, cap_usize, "hot_level", state.hot_level_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_move_speed", state.hot_move_speed_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_move_speed_mult", state.hot_move_speed_mult_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_shield_hp", state.hot_shield_hp_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_armor", state.hot_armor_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_magic_resist", state.hot_magic_resist_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_attack_damage", state.hot_attack_damage_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_attack_range", state.hot_attack_range_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_mana", state.hot_mana_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_max_mana", state.hot_max_mana_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_hunger", state.hot_hunger_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_thirst", state.hot_thirst_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_rest_timer", state.hot_rest_timer_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_safety", state.hot_safety_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_shelter", state.hot_shelter_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_social", state.hot_social_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_purpose", state.hot_purpose_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_esteem", state.hot_esteem_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_risk_tolerance", state.hot_risk_tolerance_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_social_drive", state.hot_social_drive_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_ambition", state.hot_ambition_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_altruism", state.hot_altruism_mut_slice())?;
    read_f32_slice(r, cap_usize, "hot_curiosity", state.hot_curiosity_mut_slice())?;
    for slot in 0..cap_usize {
        state.hot_engaged_with_mut_slice()[slot] = r.opt_agent("hot_engaged_with")?;
    }
    read_u32_slice(r, cap_usize, "hot_stun_expires", state.hot_stun_expires_at_tick_mut_slice())?;
    read_u32_slice(r, cap_usize, "hot_slow_expires", state.hot_slow_expires_at_tick_mut_slice())?;
    for slot in 0..cap_usize {
        state.hot_slow_factor_q8_mut_slice()[slot] = r.i16("hot_slow_factor_q8")?;
    }
    read_u32_slice(r, cap_usize, "hot_cooldown", state.hot_cooldown_next_ready_tick_mut_slice())?;

    // Cold scalars.
    for slot in 0..cap_usize {
        state.cold_creature_type_mut_slice()[slot] = r.opt_creature("cold_creature_type")?;
        state.cold_spawn_tick_mut_slice()[slot] = r.opt_u32("cold_spawn_tick")?;
        state.cold_grid_id_mut_slice()[slot] = r.opt_u32("cold_grid_id")?;
        state.cold_local_pos_mut_slice()[slot] = r.opt_vec3("cold_local_pos")?;
        state.cold_move_target_mut_slice()[slot] = r.opt_vec3("cold_move_target")?;
    }

    // Cold collections.
    for slot in 0..cap_usize {
        let n = r.u32("cold_status_effects_len")? as usize;
        let v = &mut state.cold_status_effects_mut_slice()[slot];
        v.clear();
        for _ in 0..n {
            let kind = status_kind_from_disc(r.u8("status_kind")?)?;
            let source = agent_id_nonzero(r.u32("status_source")?, "status_source")?;
            let remaining_ticks = r.u32("status_remaining")?;
            let payload_q8 = r.i16("status_payload")?;
            v.push(StatusEffect {
                kind,
                source,
                remaining_ticks,
                payload_q8,
            });
        }
    }
    for slot in 0..cap_usize {
        let n = r.u32("cold_memberships_len")? as usize;
        let v = &mut state.cold_memberships_mut_slice()[slot];
        v.clear();
        for _ in 0..n {
            let group = GroupId::new(r.u32("membership_group")?)
                .ok_or(SnapshotError::Truncated("membership_group"))?;
            let role = group_role_from_disc(r.u8("membership_role")?)?;
            let joined_tick = r.u32("membership_joined")?;
            let standing_q8 = r.i16("membership_standing")?;
            v.push(Membership {
                group,
                role,
                joined_tick,
                standing_q8,
            });
        }
    }
    for slot in 0..cap_usize {
        let gold = r.i32("inv_gold")?;
        let mut commodities = [0u16; 8];
        for c in commodities.iter_mut() {
            *c = r.u16("inv_commodity")?;
        }
        state.cold_inventory_mut()[slot] = Inventory { gold, commodities };
    }
    for slot in 0..cap_usize {
        let n = r.u32("cold_relationships_len")? as usize;
        let v = &mut state.cold_relationships_mut_slice()[slot];
        v.clear();
        for _ in 0..n {
            let other = agent_id_nonzero(r.u32("rel_other")?, "rel_other")?;
            let valence_q8 = r.i16("rel_valence")?;
            let tenure_ticks = r.u32("rel_tenure")?;
            v.push(Relationship {
                other,
                valence_q8,
                tenure_ticks,
            });
        }
    }
    for slot in 0..cap_usize {
        let row = &mut state.cold_class_definitions_mut_slice()[slot];
        for s in row.iter_mut() {
            let class_tag = r.u32("class_tag")?;
            let level = r.u8("class_level")?;
            *s = ClassSlot { class_tag, level };
        }
    }
    for slot in 0..cap_usize {
        let n = r.u32("cold_creditor_len")? as usize;
        let v = &mut state.cold_creditor_ledger_mut_slice()[slot];
        v.clear();
        for _ in 0..n {
            let creditor = agent_id_nonzero(r.u32("creditor_id")?, "creditor_id")?;
            let amount = r.u32("creditor_amount")?;
            v.push(Creditor { creditor, amount });
        }
    }
    for slot in 0..cap_usize {
        let row = &mut state.cold_mentor_lineage_mut_slice()[slot];
        for link in row.iter_mut() {
            *link = r.opt_agent("mentor_link")?;
        }
    }
    for slot in 0..cap_usize {
        let row = &mut state.ability_cooldowns[slot];
        for cd in row.iter_mut() {
            *cd = r.u32("ability_cd")?;
        }
    }

    // Rebuild the spatial hash from hot_pos + hot_alive + hot_movement_mode.
    state.rebuild_spatial_from_hot();

    Ok(state)
}

fn read_f32_slice(
    r: &mut R<'_>,
    cap: usize,
    what: &'static str,
    dst: &mut [f32],
) -> Result<(), SnapshotError> {
    debug_assert_eq!(dst.len(), cap);
    for slot in 0..cap {
        dst[slot] = r.f32(what)?;
    }
    Ok(())
}

fn read_u32_slice(
    r: &mut R<'_>,
    cap: usize,
    what: &'static str,
    dst: &mut [u32],
) -> Result<(), SnapshotError> {
    debug_assert_eq!(dst.len(), cap);
    for slot in 0..cap {
        dst[slot] = r.u32(what)?;
    }
    Ok(())
}

fn agent_id_nonzero(raw: u32, what: &'static str) -> Result<AgentId, SnapshotError> {
    AgentId::new(raw).ok_or(SnapshotError::Truncated(what))
}

// ---------- event ring metadata ----------

fn write_event_ring_meta<E: EventLike>(w: &mut W, ring: &EventRing<E>) {
    w.u32(ring.cap_for_snapshot() as u32);
    w.u32(ring.current_tick_for_snapshot());
    w.u32(ring.next_seq_for_snapshot());
    w.u64(ring.total_pushed() as u64);
    w.u64(ring.dispatched() as u64);
}

fn read_event_ring_meta<E: EventLike>(r: &mut R<'_>) -> Result<EventRing<E>, SnapshotError> {
    let cap = r.u32("ring_cap")? as usize;
    let current_tick = r.u32("ring_current_tick")?;
    let next_seq = r.u32("ring_next_seq")?;
    let total_pushed = r.u64("ring_total_pushed")? as usize;
    let dispatched = r.u64("ring_dispatched")? as usize;
    let cap = cap.max(1); // EventRing::with_cap requires cap > 0
    let mut ring = EventRing::with_cap(cap);
    ring.restore_cursors_from_parts(current_tick, next_seq, total_pushed, dispatched);
    Ok(ring)
}

// ---------- public save / load ----------

pub fn save_snapshot<E: EventLike>(
    state: &SimState,
    events: &EventRing<E>,
    path: &Path,
) -> Result<(), SnapshotError> {
    let hash = crate::schema_hash::schema_hash();
    let header = SnapshotHeader {
        magic: *MAGIC,
        schema_hash: hash,
        tick: state.tick,
        seed: state.seed,
        format_version: FORMAT_VERSION,
        reserved: [0; 10],
    };
    let mut w = W::new();
    w.buf.extend_from_slice(&header.to_bytes());
    write_state(&mut w, state);
    write_event_ring_meta(&mut w, events);
    std::fs::write(path, &w.buf)?;
    Ok(())
}

pub fn load_snapshot<E: EventLike>(path: &Path) -> Result<(SimState, EventRing<E>), SnapshotError> {
    let bytes = std::fs::read(path)?;
    load_from_bytes(&bytes)
}

pub fn load_from_bytes<E: EventLike>(bytes: &[u8]) -> Result<(SimState, EventRing<E>), SnapshotError> {
    let header = SnapshotHeader::from_bytes(bytes)?;
    let current = crate::schema_hash::schema_hash();
    if header.schema_hash != current {
        return Err(SnapshotError::SchemaMismatch {
            expected: current,
            found: header.schema_hash,
        });
    }
    if header.format_version != FORMAT_VERSION {
        return Err(SnapshotError::UnsupportedFormatVersion {
            found: header.format_version,
            expected: FORMAT_VERSION,
        });
    }
    let mut r = R::new(&bytes[HEADER_BYTES..]);
    let state = read_state(&mut r, header)?;
    let events = read_event_ring_meta(&mut r)?;
    Ok((state, events))
}
