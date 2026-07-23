//! webband_bridge — THE SHARED HOST<->FIXTURE BRIDGE for the Webband port.
//!
//! Extracted VERBATIM at S9 from `webband_campaign.rs` (S6/S6b), which owned
//! it alone until the spine test needed the same machinery. Nothing here
//! changed in behaviour: the day loop, the snapshot seam, the raid staging /
//! resolution, the trope injections and the determinism signature are the
//! same code the 60-day soak pinned — only the visibility (`pub`) and two
//! ADDITIVE bookkeeping counters (`raid_gold`, `raid_log`) are new, and
//! neither is read by any pre-existing assertion nor written into `log`
//! (the soak's cross-process digest hashes `log`, so a stray line there
//! would have moved a pin).
//!
//! **S12 PROMOTED IT TO A LIBRARY.** It lived at
//! `crates/sims/tests/webband_bridge/mod.rs` (reachable only by that crate's
//! own test binaries), which is precisely why S9 had to report "the campaign
//! loop is not yet reachable from a shipped binary". It is now a real crate
//! so `crates/webband_play` — the playable campaign binary — drives THE SAME
//! code the tests pin. The move was again behaviour-neutral: the file is
//! byte-identical apart from this paragraph and ONE refactor,
//! [`Bridge::run_day`] split into [`Bridge::step_one`] + [`Bridge::dawn`]
//! with the statements in their original order, so a real-time host can
//! spend the 600 ticks across frames and still fold at exactly the same
//! boundary. Proof is the tests' own cross-process digests, unchanged:
//! `webband_campaign::soak_60_day_campaign` fixture=0xb08e48d772417c54 and
//! `webband_spine` fixture=0xd3da575730baf403 campaign=0xcd156f088116f8a0
//! log=0x297194e3e0cbd6bb.
//!
//! `webband_campaign.rs` and `webband_spine.rs` now `use webband_bridge::*;`
//! (the crate) where they used to `mod webband_bridge;` (the file).
//!
//! THE BRIDGE ARCHITECTURE (the decision this file embodies): Webband's own
//! shape is that the storyteller is CAMPAIGN-side and the colony is the sim —
//! so the campaign brain stays in `crates/webband_app` (pure host logic, no
//! engine dependency), the colony stays in the `webband_colony` fixture (no
//! director state, no gold, no roster ids), and the two meet ONLY here, in an
//! integration test that is a dev-dependency of `sims`.
//!
//! The loop, once per campaign day:
//!   1. step the fixture 600 ticks (1 tick = 1 Webband minute);
//!   2. settle the caravan's fixture-side hide sales (purse delta -> gold);
//!   3. read the colony back (holder inv_* -> InventorySnapshot, per-roster
//!      mood/starving_days -> MemberViews, sown plots) — the snapshot seam;
//!   4. resolve a staged raid if the fixture reports it settled (victory =
//!      cohort dead/withdrawn with no plunder; defeat = the fixture's own
//!      plunder) through webband_app's resolve_raid, then RESET the pool;
//!   5. run webband_app's `dawn_fold` (provisioner, exodus, trade income,
//!      caravan sweep, the STORYTELLER with its committed-plan draw);
//!   6. write the resolved tropes back through the fixture's own seams.
//!
//! ROSTER = SLOT POOL: campaign roster members map 1:1 onto colonist slots in
//! roster order; slots beyond the roster are deactivated at founding (the
//! recruit pool). Every activation inherits the slot's own UNIQUE stagger
//! residue, so the fixture's %20 phase-exclusivity determinism construction
//! survives roster churn by construction.

#![allow(dead_code)]
#![allow(non_snake_case)]
// Re-exports serve two test binaries; each uses a different subset.
#![allow(unused_imports)]

use std::collections::VecDeque;

/// S12 D2 — fixture-state persistence (see the module docs).
pub mod persist;

pub use sims::webband_colony::GeneratedRuntime;
pub use webband_app::afield::{
    dispatch_cost, dispatch_party, is_afield, AfieldParty, AfieldReport, DispatchContext,
    DispatchOpts, ErrandFight,
};
pub use webband_app::ambition::{ambition_progress, current_stage, describe_stage, AmbitionStep};
pub use webband_app::campaign::{
    dawn_fold, dawn_fold_political, petition_capacity, resolve_raid, Campaign, CampaignOutcome,
    ColonySnapshot, DawnOutcome, MemberView, RaidResultView,
};
pub use webband_app::factions::{faction_by_id, petitioners, Faction};
pub use webband_app::petitions::{
    answer_petition, describe_petition, petition_choices, petitioner_name, standing_with,
    standing_word, Petition, PetitionCapacity, PetitionChoice, PetitionChoiceKind,
};
pub use webband_app::defs::{BuildingView, InventorySnapshot, StackView};
pub use webband_app::director::{CampaignEvent, Caravan, ItemDrop};
pub use webband_app::founding::new_founding;
pub use webband_app::raids::{troop, ActiveRaid};
pub use webband_app::scenario::ScenarioId;
pub use webband_app::worldgen::BandStatus;

/// The fixture seed — MUST stay the spacing-verified S3 seed (ring angles
/// are (seed, slot)-hashed; the reservation law's >0.9u same-kind spacing
/// precondition was scanned for this seed).
pub const SEED: u64 = 0xC0_10_11_5E_ED;
pub const AGENTS: u32 = 512;
pub const DAY_TICKS: u32 = 600;
/// Muster at dawn+12 (never %600==0, never the %20==10 brawl storm) — the
/// S5 convention.
pub const RAID_DAWN_OFFSET: u32 = 12;

/// The campaign (founding) seed. Deterministic; CHOSEN so the 60-day soak's
/// organic trope mix exercises every wired seam (the windfall trope carries
/// weight 1 of ~11, so not every seed draws one in 60 days — seed 20260722
/// ran a full, otherwise-green campaign with zero windfalls under the
/// pre-stock-scaling staging).
pub const CAMPAIGN_SEED: u32 = 20260722;

// Creature-type ordinals (declaration order in webband_colony.sim).
pub const CT_COLONIST: u32 = 0;
pub const CT_TREE: u32 = 1;
pub const CT_BUSH: u32 = 2;
pub const CT_GAME: u32 = 3;
pub const CT_PLOT: u32 = 4;
pub const CT_STORE: u32 = 5;
pub const CT_CACHE: u32 = 6;
pub const CT_HEARTH: u32 = 7;
pub const CT_BENCH: u32 = 8;
pub const CT_BED: u32 = 9;
pub const CT_SHED: u32 = 10;
pub const CT_WALL: u32 = 11;
pub const CT_MESS: u32 = 12;
pub const CT_RAIDER: u32 = 13;
pub const CT_WARLORD: u32 = 14;
pub const CT_TRADER: u32 = 15;

pub const HIDE_SELL: f64 = 2.0; // floor(hide value 4 * 0.6) — config.wb.hide_sell
pub const MEAL_BUY: i64 = 5; // ceil(meal value 3 * 1.5) — trade.ts BUY_MULT

// ---------------------------------------------------------------------------
// GPU readback / write helpers (the webband_colony.rs idiom, duplicated here
// because integration tests cannot share a module without a common crate).

pub fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_campaign::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

pub fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_campaign::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

pub fn read_vec4(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<[f32; 4]> {
    let bytes = (count as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_campaign::vec4_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[[f32; 4]] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

pub fn write_f32(state: &GeneratedRuntime, buf: &wgpu::Buffer, idx: usize, v: f32) {
    state
        .gpu
        .queue
        .write_buffer(buf, (idx as u64) * 4, bytemuck::cast_slice(&[v]));
}

pub fn write_u32(state: &GeneratedRuntime, buf: &wgpu::Buffer, idx: usize, v: u32) {
    state
        .gpu
        .queue
        .write_buffer(buf, (idx as u64) * 4, bytemuck::cast_slice(&[v]));
}

pub fn write_vec3(state: &GeneratedRuntime, buf: &wgpu::Buffer, idx: usize, v: [f32; 3]) {
    let padded = [v[0], v[1], v[2], 0.0f32];
    state
        .gpu
        .queue
        .write_buffer(buf, (idx as u64) * 16, bytemuck::cast_slice(&padded));
}

pub fn on_big_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn big-stack test thread")
        .join()
        .expect("test thread panicked");
}

/// Alive-filtered (LOAD-BEARING: slot 0 is the AgentId SENTINEL — zero
/// creature_type, zero alive. An unfiltered scan seats a roster member on
/// the sentinel body, who then never eats, never starves, and never walks:
/// found as the famine test's immortal ghost founder.)
pub fn slots_of(types: &[u32], alive: &[u32], ct: u32) -> Vec<usize> {
    (0..types.len())
        .filter(|&i| alive[i] != 0 && types[i] == ct)
        .collect()
}

// ---------------------------------------------------------------------------
// The fixture world map (static slots, read once at founding).

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct World {
    pub n: usize,
    pub colonists: Vec<usize>, // all 20 body slots, ascending
    pub bushes: Vec<usize>,
    pub game: Vec<usize>,
    pub plots: Vec<usize>,
    pub store: usize,
    pub yard_cache: usize,
    pub rim_cache: usize,
    pub hearth: usize,
    pub bench: usize,
    pub bed: usize,
    pub shed: usize,
    pub walls: Vec<usize>,
    pub mess: usize,
    pub looters: Vec<usize>,
    pub bandits: Vec<usize>,
    pub ranks: Vec<usize>,
    pub warlords: Vec<usize>,
    pub trader: usize,
    /// (slot, founding hp) for the whole raid pool — the reset table.
    pub pool_hp: Vec<(usize, f32)>,
    /// Founding positions of the raid pool (parked rim rings) — restored on
    /// every post-raid reset.
    pub pool_pos: Vec<(usize, [f32; 3])>,
}

pub fn read_world(state: &mut GeneratedRuntime) -> World {
    let n = state.agent_count as usize;
    let types = { let b = state.agent_creature_type_buf.clone(); read_u32(state, &b, n) };
    let alive = { let b = state.agent_alive_buf.clone(); read_u32(state, &b, n) };
    let hp = { let b = state.agent_hp_buf.clone(); read_f32(state, &b, n) };
    let pos = { let b = state.agent_pos_buf.clone(); read_vec4(state, &b, n) };
    let slots_of = |ct: u32| slots_of(&types, &alive, ct);
    let caches = slots_of(CT_CACHE);
    assert_eq!(caches.len(), 2, "yard cache + S6 rim cache");
    // The yard cache stands near the origin (ring 2.5); the rim one at 28.
    let r_of = |i: usize| (pos[i][0] * pos[i][0] + pos[i][1] * pos[i][1]).sqrt();
    let (yard_cache, rim_cache) = if r_of(caches[0]) < r_of(caches[1]) {
        (caches[0], caches[1])
    } else {
        (caches[1], caches[0])
    };
    let raiders = slots_of(CT_RAIDER);
    let by_hp = |v: f32| -> Vec<usize> {
        raiders.iter().copied().filter(|&r| (hp[r] - v).abs() < 0.5).collect()
    };
    let warlords = slots_of(CT_WARLORD);
    let mut pool_hp: Vec<(usize, f32)> = Vec::new();
    let mut pool_pos: Vec<(usize, [f32; 3])> = Vec::new();
    for &r in raiders.iter().chain(warlords.iter()) {
        pool_hp.push((r, hp[r]));
        pool_pos.push((r, [pos[r][0], pos[r][1], pos[r][2]]));
    }
    World {
        n,
        colonists: slots_of(CT_COLONIST),
        bushes: slots_of(CT_BUSH),
        game: slots_of(CT_GAME),
        plots: slots_of(CT_PLOT),
        store: slots_of(CT_STORE)[0],
        yard_cache,
        rim_cache,
        hearth: slots_of(CT_HEARTH)[0],
        bench: slots_of(CT_BENCH)[0],
        bed: slots_of(CT_BED)[0],
        shed: slots_of(CT_SHED)[0],
        walls: slots_of(CT_WALL),
        mess: slots_of(CT_MESS)[0],
        looters: by_hp(42.0),
        bandits: by_hp(72.0),
        ranks: by_hp(105.0),
        warlords,
        trader: slots_of(CT_TRADER)[0],
        pool_hp,
        pool_pos,
    }
}

// ---------------------------------------------------------------------------
// The bridge.

/// S9: one resolved raid, as the spine's coherence pins read it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RaidRecord {
    pub day: i64,
    pub victory: bool,
    pub tier: i64,
    /// Loot paid by the raiders the FIXTURE actually slew (TROOPS loot per
    /// body) — nonzero is proof the combat ran, not just the host math.
    pub gold_looted: i64,
    /// Colonists left downed when the storm settled (KO, never death).
    pub downed: usize,
    /// Units the fixture's own plunder sweep stripped, on a defeat.
    pub plunder_taken: Option<i64>,
}

pub struct StagedRaid {
    pub raid: ActiveRaid,
    pub cohort: Vec<usize>,
    pub muster_tick: u32,
}

pub struct Bridge {
    pub state: GeneratedRuntime,
    pub w: World,
    pub campaign: Campaign,
    /// Roster-order mapping member id -> colonist slot.
    pub slot_map: Vec<(String, usize)>,
    /// Deactivated colonist slots (ascending) — the recruit pool.
    pub free_slots: VecDeque<usize>,
    pub tick: u32,
    pub staged: Option<StagedRaid>,
    pub purse_shadow: f64,
    /// The host's own trade policy switch (the famine variant turns it off).
    pub auto_trade: bool,
    // Metrics + the determinism log.
    pub log: Vec<String>,
    pub hungry_member_dawns: u64,
    pub member_dawns: u64,
    pub raids_staged: u64,
    pub raids_won: u64,
    pub raids_lost: u64,
    pub windfalls: u64,
    pub caravans: u64,
    pub trade_gold: i64,
    pub meals_bought: i64,
    pub joins: u64,
    pub departures: Vec<(i64, String)>,
    pub prev_starving: Vec<f32>,
    /// Every storyteller trope that fired, in order (the campaign-shape
    /// evidence the soak asserts on).
    pub event_kinds: Vec<String>,
    // -- S9 additions: ledger bookkeeping for the spine's coherence pins.
    // ADDITIVE ONLY — nothing below is read by the campaign suite, and
    // nothing below writes to `log` (the soak digest hashes `log`).
    /// Cumulative gold credited by won raids (resolve_raid's `gold_looted`).
    pub raid_gold: i64,
    /// One record per resolved raid.
    pub raid_log: Vec<RaidRecord>,
    /// The last snapshot `run_day` handed the dawn fold — the spine
    /// recomputes the storyteller's accrual from it to pin the fold's
    /// step ORDER (the accrual must see post-provisioner gold).
    pub last_snap: Option<ColonySnapshot>,

    // -- S13: THE GUILD LAYER, LIVE (politics campaigns only) --------------
    // Every field below is inert while `campaign.politics_enabled` is false,
    // and `dawn()` takes the pre-S13 branch in that case — which is what
    // keeps the apolitical soak/spine digests valid.
    /// Roster ids currently on the road with their body slot parked. A
    /// dispatched hand is REMOVED FROM COLONY WORK by deactivating its body
    /// (the departure idiom), so the job masks budget them nothing and a raid
    /// musters without them — both by construction, no penalty code.
    pub away: Vec<(String, usize)>,
    /// Petitions opened / answered / lapsed (the political soak's shape).
    pub petitions_opened: u64,
    pub petitions_answered: u64,
    pub petitions_lapsed: u64,
    /// Every answer taken, in order: (day, faction, kind, answer).
    pub petition_log: Vec<(i64, String, String, String)>,
    /// Ambition stages closed, in order.
    pub stages: Vec<String>,
    /// Set when the founders' arc completed — the campaign is over and won.
    pub achieved: Option<String>,
    /// Bands that gave notice / rode out under the guild layer's own clocks.
    pub band_notices: Vec<(i64, String)>,
    /// Dispatch refusals, verbatim from the sim seam (never recomputed here).
    pub refusals: Vec<String>,
}

impl Bridge {
    /// Found a campaign and seat it on the fixture. `sign_bands` = how many
    /// non-founder bands sign on at founding (staging: the fixture's economy
    /// is scaled for ~a dozen hands; the TS colony grows into that size
    /// through the bands machinery, which is a later slice — signing at
    /// founding is the deterministic stand-in, documented).
    pub fn new(scenario: ScenarioId, campaign_seed: u32, sign_bands: usize, auto_trade: bool) -> Option<Bridge> {
        Bridge::new_with(scenario, campaign_seed, sign_bands, auto_trade, false)
    }

    /// S13 — the same founding, with the GUILD LAYER either live or not.
    ///
    /// `politics = true` founds through [`Campaign::new_political`], which
    /// APPENDS the factions/ambition rolls to the seeded stream (S11: the
    /// founding record stays byte-identical, but every post-founding draw
    /// resumes from a moved counter). A political campaign is therefore a
    /// DIFFERENT campaign from the same seed — deliberately, and with its own
    /// recorded digest. `politics = false` is the pre-S13 path, byte for byte:
    /// [`Bridge::new`] delegates here with `false` and the soak/spine pins hold.
    pub fn new_with(
        scenario: ScenarioId,
        campaign_seed: u32,
        sign_bands: usize,
        auto_trade: bool,
        politics: bool,
    ) -> Option<Bridge> {
        let mut state = GeneratedRuntime::try_new(SEED, AGENTS)?;
        let mut w = read_world(&mut state);
        w.colonists.sort_unstable();
        assert_eq!(w.colonists.len(), 20, "founding seeds 20 colonist bodies");

        let founding = new_founding(campaign_seed, 0, scenario).expect("founding generates");
        let mut campaign =
            if politics { Campaign::new_political(founding) } else { Campaign::new(founding) };

        // Sign the first `sign_bands` non-founder bands (cast band order),
        // capped so the roster never exceeds the 20 colonist bodies.
        let bands: Vec<(String, Vec<String>)> = campaign
            .founding
            .cast
            .bands
            .iter()
            .filter(|b| !b.founders)
            .map(|b| {
                let members: Vec<String> = campaign
                    .founding
                    .cast
                    .companions
                    .iter()
                    .filter(|c| c.band.as_deref() == Some(b.id.as_str()))
                    .map(|c| c.id.clone())
                    .collect();
                (b.id.clone(), members)
            })
            .collect();
        for (band_id, members) in bands.iter().take(sign_bands) {
            if campaign.roster.len() + members.len() > w.colonists.len() {
                break;
            }
            campaign.roster.extend(members.iter().cloned());
            if let Some((_, live)) = campaign.band_states.iter_mut().find(|(id, _)| id == band_id) {
                live.state.status = BandStatus::Signed;
            }
        }

        // Seat the roster on the slot pool; deactivate the rest.
        let mut slot_map: Vec<(String, usize)> = Vec::new();
        for (i, id) in campaign.roster.iter().enumerate() {
            slot_map.push((id.clone(), w.colonists[i]));
        }
        let mut free_slots: VecDeque<usize> = VecDeque::new();
        for &slot in w.colonists.iter().skip(campaign.roster.len()) {
            write_u32(&state, &state.agent_alive_buf, slot, 0);
            free_slots.push_back(slot);
        }

        // Replace the fixture's hardcoded founding cache with the SCENARIO's
        // stock (the comparability law's stock lands here — the fixture spawn
        // predates the host layer and carries a teaching-cache superset).
        for buf in [
            &state.agent_inv_meal_buf,
            &state.agent_inv_berries_buf,
            &state.agent_inv_venison_buf,
            &state.agent_inv_grain_buf,
            &state.agent_inv_timber_buf,
            &state.agent_inv_plank_buf,
            &state.agent_inv_hide_buf,
        ] {
            write_f32(&state, buf, w.yard_cache, 0.0);
        }
        // Staging: the scenario's stock provisions a FOUNDERS-sized colony
        // (the TS applyScenario drops it for 4-5 hands). Signing extra bands
        // at founding is this bridge's stand-in for the bands slice, so each
        // signed band arrives carrying its share of provisions — the stock
        // scales with the staged roster (deterministic, zero draws). Without
        // this the day-2 famine is a staging artifact, not the economy.
        let founders_n = campaign.founding.roster.len().max(1);
        let stock_mult = campaign.roster.len() as f32 / founders_n as f32;
        for s in &campaign.founding.stacks {
            let buf = match s.item.as_str() {
                "meal" => Some(state.agent_inv_meal_buf.clone()),
                "berries" => Some(state.agent_inv_berries_buf.clone()),
                "venison" => Some(state.agent_inv_venison_buf.clone()),
                "grain" => Some(state.agent_inv_grain_buf.clone()),
                "timber" => Some(state.agent_inv_timber_buf.clone()),
                "plank" => Some(state.agent_inv_plank_buf.clone()),
                "hide" => Some(state.agent_inv_hide_buf.clone()),
                _ => None, // herbs etc. have no fixture item — campaign-only
            };
            if let Some(buf) = buf {
                let scaled = (s.count as f32 * stock_mult).floor();
                let cur = read_f32(&mut state, &buf, w.n)[w.yard_cache];
                write_f32(&state, &buf, w.yard_cache, cur + scaled);
            }
        }

        let prev_starving = {
            let b = state.agent_starving_days_buf.clone();
            read_f32(&mut state, &b, w.n)
        };

        Some(Bridge {
            state,
            w,
            campaign,
            slot_map,
            free_slots,
            tick: 0,
            staged: None,
            purse_shadow: 0.0,
            auto_trade,
            log: Vec::new(),
            hungry_member_dawns: 0,
            member_dawns: 0,
            raids_staged: 0,
            raids_won: 0,
            raids_lost: 0,
            windfalls: 0,
            caravans: 0,
            trade_gold: 0,
            meals_bought: 0,
            joins: 0,
            departures: Vec::new(),
            prev_starving,
            event_kinds: Vec::new(),
            raid_gold: 0,
            raid_log: Vec::new(),
            last_snap: None,
            away: Vec::new(),
            petitions_opened: 0,
            petitions_answered: 0,
            petitions_lapsed: 0,
            petition_log: Vec::new(),
            stages: Vec::new(),
            achieved: None,
            band_notices: Vec::new(),
            refusals: Vec::new(),
        })
    }

    pub fn slot_of(&self, id: &str) -> Option<usize> {
        self.slot_map.iter().find(|(m, _)| m == id).map(|(_, s)| *s)
    }

    // -- snapshot (fixture -> host) -----------------------------------------

    pub fn holder_stacks(&mut self, out: &mut Vec<StackView>, slot: usize) {
        let n = self.w.n;
        let items: [(&str, wgpu::Buffer); 7] = [
            ("berries", self.state.agent_inv_berries_buf.clone()),
            ("grain", self.state.agent_inv_grain_buf.clone()),
            ("hide", self.state.agent_inv_hide_buf.clone()),
            ("meal", self.state.agent_inv_meal_buf.clone()),
            ("plank", self.state.agent_inv_plank_buf.clone()),
            ("timber", self.state.agent_inv_timber_buf.clone()),
            ("venison", self.state.agent_inv_venison_buf.clone()),
        ];
        for (item, buf) in &items {
            let v = read_f32(&mut self.state, buf, n)[slot];
            let count = v.floor() as i64;
            if count >= 1 {
                out.push(StackView {
                    id: format!("s{slot:03}:{item}"),
                    item: (*item).to_string(),
                    count: count as u32,
                });
            }
        }
    }

    pub fn snapshot(&mut self) -> ColonySnapshot {
        let n = self.w.n;
        let mut stacks: Vec<StackView> = Vec::new();
        for slot in [
            self.w.store,
            self.w.yard_cache,
            self.w.rim_cache,
            self.w.hearth,
            self.w.bench,
        ] {
            self.holder_stacks(&mut stacks, slot);
        }
        let built = { let b = self.state.agent_built_buf.clone(); read_u32(&mut self.state, &b, n) };
        let pos = { let b = self.state.agent_pos_buf.clone(); read_vec4(&mut self.state, &b, n) };
        let mut buildings: Vec<BuildingView> = Vec::new();
        let mut push_b = |slot: usize, kind: &str| {
            buildings.push(BuildingView {
                id: format!("b{slot:03}"),
                kind: kind.to_string(),
                q: pos[slot][0].round() as i32,
                r: pos[slot][1].round() as i32,
                built: built[slot] == 1,
            });
        };
        push_b(self.w.hearth, "hearth");
        push_b(self.w.bench, "workbench");
        push_b(self.w.bed, "bed");
        push_b(self.w.shed, "store_shed");
        push_b(self.w.mess, "mess_table");
        for &wall in &self.w.walls.clone() {
            push_b(wall, "wall");
        }

        let sown = { let b = self.state.agent_sown_buf.clone(); read_u32(&mut self.state, &b, n) };
        let sown_cells: Vec<String> = self
            .w
            .plots
            .iter()
            .filter(|&&p| sown[p] == 1)
            .map(|&p| format!("p{p}"))
            .collect();

        let mood = { let b = self.state.agent_mood_buf.clone(); read_f32(&mut self.state, &b, n) };
        let starving = {
            let b = self.state.agent_starving_days_buf.clone();
            read_f32(&mut self.state, &b, n)
        };
        let mut members: Vec<MemberView> = Vec::new();
        let mut starved_today: Vec<String> = Vec::new();
        for (id, slot) in &self.slot_map {
            members.push(MemberView {
                id: id.clone(),
                mood: f64::from(mood[*slot]),
                starving_days: starving[*slot].round() as i64,
                // S13: a hand on the road is not ready. (Identical to `true`
                // for an apolitical campaign — `afield` is always empty there
                // — so no pinned digest can move.)
                ready: !is_afield(&self.campaign, id),
            });
            if starving[*slot] > self.prev_starving[*slot] + 0.5 {
                starved_today.push(id.clone());
            }
        }
        self.prev_starving = starving;

        ColonySnapshot {
            inventory: InventorySnapshot { stacks, buildings },
            sown_cells,
            members,
            rotted_food_units: 0.0, // spoilage is fixture truth; the host line is cosmetic
            starved_today,
        }
    }

    // -- raid staging / resolution ------------------------------------------

    /// Map a rolled comp onto the dormant pool: rank counts fill their own
    /// rank's slots first, overflow spills to the next rank (up, then down).
    /// A named elite fields a warlord.
    ///
    /// THE CAP IS LOUD (S6b): the pool is 40 bodies and a comp that does not
    /// fit PANICS. It used to log "n bodies dropped (pool cap 20)" and field
    /// a quietly-weakened warband — the escalation clock (day*0.25 into the
    /// budget) made that bind at day 56 of the 60-day soak. A silent cap on
    /// the enemy is exactly what the port's discipline forbids; if this ever
    /// fires again the fixture pool is what must grow (40 CONSECUTIVE slots
    /// carry the strike-phase determinism construction — see the fixture
    /// header), not this mapping.
    pub fn stage_raid(&mut self, raid: &ActiveRaid, muster_tick: u32) {
        let mut want = [0usize; 3]; // looter, bandit, raider
        for (id, n) in &raid.comp {
            match id.as_str() {
                "looter" => want[0] += *n as usize,
                "bandit" => want[1] += *n as usize,
                "raider" => want[2] += *n as usize,
                _ => {}
            }
        }
        let ranks: [Vec<usize>; 3] = [
            self.w.looters.clone(),
            self.w.bandits.clone(),
            self.w.ranks.clone(),
        ];
        let mut cohort: Vec<usize> = Vec::new();
        let mut spill = 0usize;
        // First pass: own rank; spill accumulates upward.
        let mut leftover = [0usize; 3];
        for r in 0..3 {
            let take = want[r].min(ranks[r].len());
            cohort.extend(&ranks[r][..take]);
            leftover[r] = ranks[r].len() - take;
            spill += want[r] - take;
        }
        // Spill: fill any remaining rank slots, heaviest first.
        for r in (0..3).rev() {
            while spill > 0 && leftover[r] > 0 {
                let idx = ranks[r].len() - leftover[r];
                cohort.push(ranks[r][idx]);
                leftover[r] -= 1;
                spill -= 1;
            }
        }
        if raid.elite_name.is_some() {
            cohort.push(self.w.warlords[0]);
        }
        assert_eq!(
            spill, 0,
            "day {}: the rolled comp {:?} (tier {}) wants {} more bodies than the \
             fixture's {}-body raid pool can field — the enemy would be SILENTLY \
             weakened. Grow the pool in assets/sim/webband_colony.sim (40 consecutive \
             slots today; any resize must keep the strike-phase construction).",
            self.campaign.day,
            raid.comp,
            raid.tier,
            spill,
            self.w.looters.len() + self.w.bandits.len() + self.w.ranks.len(),
        );
        // Entry arc: the rolled hex-edge direction picks the approach.
        let angle = (raid.entry_dir as f32) * std::f32::consts::FRAC_PI_3;
        let (cx, cy) = (angle.sin() * 26.0, angle.cos() * 26.0);
        let (px, py) = (angle.cos(), -angle.sin()); // perpendicular spread
        let k = cohort.len() as f32;
        for (i, &slot) in cohort.iter().enumerate() {
            let off = (i as f32) - (k - 1.0) / 2.0;
            write_u32(&self.state, &self.state.agent_musters_at_buf, slot, muster_tick);
            write_u32(&self.state, &self.state.agent_warned_buf, slot, 0);
            write_vec3(
                &self.state,
                &self.state.agent_pos_buf,
                slot,
                [cx + px * off * 2.2, cy + py * off * 2.2, 0.0],
            );
        }
        self.raids_staged += 1;
        self.log.push(format!(
            "day {} raid staged: {} bodies, tier {}, dir {}, elite {:?}",
            self.campaign.day,
            cohort.len(),
            raid.tier,
            raid.entry_dir,
            raid.elite_name
        ));
        self.staged = Some(StagedRaid { raid: raid.clone(), cohort, muster_tick });
    }

    /// If the staged raid has settled (cohort dead or withdrawn), fold it
    /// campaign-side and reset the pool + scars for the next storm.
    pub fn maybe_resolve_raid(&mut self, snap: &ColonySnapshot) {
        let Some(st) = &self.staged else { return };
        if self.tick <= st.muster_tick {
            return;
        }
        let n = self.w.n;
        let alive = { let b = self.state.agent_alive_buf.clone(); read_u32(&mut self.state, &b, n) };
        let ra = { let b = self.state.agent_raid_active_buf.clone(); read_u32(&mut self.state, &b, n) };
        let settled = st
            .cohort
            .iter()
            .all(|&r| alive[r] == 0 || ra[r] >= 2);
        if !settled {
            return;
        }
        let st = self.staged.take().expect("checked");
        let pl = {
            let b = self.state.agent_plundered_at_buf.clone();
            read_u32(&mut self.state, &b, n)
        };
        let plundered = [self.w.store, self.w.yard_cache, self.w.rim_cache]
            .iter()
            .any(|&h| pl[h] != 0);
        let victory = !plundered;

        // Loot: the slain pay in TROOPS loot values (the TS BattleResult).
        let hp = { let b = self.state.agent_hp_buf.clone(); read_f32(&mut self.state, &b, n) };
        let mut gold_looted = 0i64;
        for &r in &st.cohort {
            if alive[r] == 0 {
                let rank = if self.w.warlords.contains(&r) {
                    "warlord"
                } else if self.w.looters.contains(&r) {
                    "looter"
                } else if self.w.bandits.contains(&r) {
                    "bandit"
                } else {
                    "raider"
                };
                gold_looted += troop(rank).expect("troop table").loot;
            }
        }
        if !victory {
            gold_looted = 0;
        }

        let downed = { let b = self.state.agent_downed_buf.clone(); read_u32(&mut self.state, &b, n) };
        let member_hp: Vec<(String, f64)> = self
            .slot_map
            .iter()
            .map(|(id, slot)| (id.clone(), f64::from(hp[*slot] / 100.0)))
            .collect();
        let result = RaidResultView { victory, gold_looted, member_hp };
        let outcome = resolve_raid(&mut self.campaign, &st.raid.clone(), &result, &snap.inventory);

        // Homecoming injections. On defeat (or an all-downed "victory" —
        // timeout with no tender left standing) everyone mends to walking
        // shape (TS resolveRaid: defeat costs stock, never the roster); a
        // won raid leaves the few downed to the organic tend/dawn recovery
        // the S5 win test proved.
        let active_downed: Vec<usize> = self
            .slot_map
            .iter()
            .map(|(_, s)| *s)
            .filter(|&s| downed[s] == 1)
            .collect();
        let all_downed = active_downed.len() == self.slot_map.len() && !self.slot_map.is_empty();
        if !victory || all_downed {
            for &slot in &active_downed {
                write_u32(&self.state, &self.state.agent_downed_buf, slot, 0);
                write_f32(&self.state, &self.state.agent_hp_buf, slot, 30.0);
            }
        }
        // Thoughts land on the mood blend (no event seam from the host; the
        // per-thought views stay flat for host-injected thoughts, documented).
        let tsum = { let b = self.state.agent_thought_sum_buf.clone(); read_f32(&mut self.state, &b, n) };
        for (id, key) in &outcome.thoughts {
            let delta = match key.as_str() {
                "victory" => 8.0,
                "defeat" => -10.0,
                "came_home_to_ashes" => -14.0,
                _ => 0.0,
            };
            if let Some(slot) = self.slot_of(id) {
                write_f32(&self.state, &self.state.agent_thought_sum_buf, slot, tsum[slot] + delta);
            }
        }

        // Reset the pool + the scars so the NEXT storm finds a live seam
        // (plundered_at must clear or RaiderBrain instant-withdraws it).
        for &(slot, hp0) in &self.w.pool_hp {
            write_u32(&self.state, &self.state.agent_alive_buf, slot, 1);
            write_f32(&self.state, &self.state.agent_hp_buf, slot, hp0);
            write_u32(&self.state, &self.state.agent_raid_active_buf, slot, 0);
            write_u32(&self.state, &self.state.agent_musters_at_buf, slot, 0);
            write_u32(&self.state, &self.state.agent_warned_buf, slot, 0);
            write_u32(&self.state, &self.state.agent_strike_cd_until_buf, slot, 0);
            write_f32(&self.state, &self.state.agent_raid_total_buf, slot, 0.0);
            write_f32(&self.state, &self.state.agent_defenders_buf, slot, 0.0);
        }
        for &(slot, p) in &self.w.pool_pos {
            write_vec3(&self.state, &self.state.agent_pos_buf, slot, p);
        }
        for h in [self.w.store, self.w.yard_cache, self.w.rim_cache] {
            write_u32(&self.state, &self.state.agent_plundered_at_buf, h, 0);
        }
        if victory {
            self.raids_won += 1;
        } else {
            self.raids_lost += 1;
        }
        // S9 bookkeeping (additive; no log line, the digest hashes `log`).
        self.raid_gold += gold_looted;
        self.raid_log.push(RaidRecord {
            day: self.campaign.day,
            victory,
            tier: st.raid.tier,
            gold_looted,
            downed: active_downed.len(),
            plunder_taken: outcome.plunder.as_ref().map(|p| i64::from(p.taken_total)),
        });
        self.log.push(format!(
            "day {} raid resolved: victory={victory} loot={gold_looted} downed={} plunder={:?}",
            self.campaign.day,
            active_downed.len(),
            outcome.plunder.as_ref().map(|p| p.taken_total)
        ));
    }

    // -- caravan ------------------------------------------------------------

    /// Credit fixture-side hide sales (the purse only moves under the
    /// SoldHide consumer between the host's own writes).
    pub fn settle_caravan_sales(&mut self) {
        if self.campaign.caravan.is_none() {
            return;
        }
        let purse = {
            let b = self.state.agent_purse_buf.clone();
            f64::from(read_f32(&mut self.state, &b, self.w.n)[self.w.trader])
        };
        let delta = self.purse_shadow - purse;
        if delta > 0.5 {
            let earned = delta.round() as i64;
            self.campaign.gold += earned;
            self.trade_gold += earned;
            if let Some(cv) = self.campaign.caravan.as_mut() {
                cv.traded = true;
            }
            self.log.push(format!(
                "day {} caravan: hunters sold {} gold of hides at the camp",
                self.campaign.day, earned
            ));
        }
        self.purse_shadow = purse;
    }

    /// The host's trade round — the TS trade-panel seam (verdicts in
    /// trade.ts): sell stored hides at floor(value*0.6) capped by the purse,
    /// buy meals off the wares at ceil(value*1.5); purchases land on the RIM
    /// cache (caravanSpot — even good luck makes haul work).
    pub fn host_trade_round(&mut self) {
        if !self.auto_trade || self.campaign.caravan.is_none() {
            return;
        }
        let n = self.w.n;
        // SELL stored hides.
        let store_hides = {
            let b = self.state.agent_inv_hide_buf.clone();
            read_f32(&mut self.state, &b, n)[self.w.store]
        }
        .floor() as i64;
        let afford = (self.purse_shadow / HIDE_SELL).floor() as i64;
        let sell = store_hides.min(afford).max(0);
        if sell > 0 {
            let cur_store = {
                let b = self.state.agent_inv_hide_buf.clone();
                read_f32(&mut self.state, &b, n)[self.w.store]
            };
            let cur_trader = {
                let b = self.state.agent_inv_hide_buf.clone();
                read_f32(&mut self.state, &b, n)[self.w.trader]
            };
            write_f32(&self.state, &self.state.agent_inv_hide_buf, self.w.store, cur_store - sell as f32);
            write_f32(&self.state, &self.state.agent_inv_hide_buf, self.w.trader, cur_trader + sell as f32);
            self.purse_shadow -= sell as f64 * HIDE_SELL;
            write_f32(&self.state, &self.state.agent_purse_buf, self.w.trader, self.purse_shadow as f32);
            let earned = sell * HIDE_SELL as i64;
            self.campaign.gold += earned;
            self.trade_gold += earned;
            if let Some(cv) = self.campaign.caravan.as_mut() {
                cv.traded = true;
            }
            self.log.push(format!(
                "day {} caravan: sold {sell} stored hides for {earned} gold",
                self.campaign.day
            ));
        }
        // BUY meals from the wares while the guild purse holds.
        let mut bought = 0i64;
        if let Some(cv) = self.campaign.caravan.as_mut() {
            if let Some((_, count)) = cv.goods.iter_mut().find(|(item, _)| item == "meal") {
                let can = (self.campaign.gold / MEAL_BUY).min(i64::from(*count)).max(0);
                if can > 0 {
                    *count -= can as u32;
                    bought = can;
                    cv.traded = true;
                }
            }
        }
        if bought > 0 {
            self.campaign.gold -= bought * MEAL_BUY;
            self.meals_bought += bought;
            let cur = {
                let b = self.state.agent_inv_meal_buf.clone();
                read_f32(&mut self.state, &b, n)[self.w.rim_cache]
            };
            write_f32(&self.state, &self.state.agent_inv_meal_buf, self.w.rim_cache, cur + bought as f32);
            self.log.push(format!(
                "day {} caravan: bought {bought} meals ({} gold) — landed at the rim camp",
                self.campaign.day,
                bought * MEAL_BUY
            ));
        }
    }

    // -- dawn injections ----------------------------------------------------

    pub fn apply_dawn(&mut self, out: &DawnOutcome) {
        let n = self.w.n;
        // The provisioner's meals land at the yard cache (CACHE_GROUND).
        if let Some(p) = &out.provision {
            let cur = {
                let b = self.state.agent_inv_meal_buf.clone();
                read_f32(&mut self.state, &b, n)[self.w.yard_cache]
            };
            write_f32(
                &self.state,
                &self.state.agent_inv_meal_buf,
                self.w.yard_cache,
                cur + p.count as f32,
            );
            self.log.push(format!(
                "day {} provisioner: {} meals for {} gold",
                self.campaign.day, p.count, p.gold_spent
            ));
        }

        // The exodus: walkers leave their bodies' slots to the pool.
        for id in &out.departed {
            if let Some(pos) = self.slot_map.iter().position(|(m, _)| m == id) {
                let (_, slot) = self.slot_map.remove(pos);
                write_u32(&self.state, &self.state.agent_alive_buf, slot, 0);
                write_u32(&self.state, &self.state.agent_claimed_job_buf, slot, 0);
                let mut v: Vec<usize> = self.free_slots.iter().copied().collect();
                v.push(slot);
                v.sort_unstable();
                self.free_slots = v.into();
                self.departures.push((self.campaign.day, id.clone()));
                self.log.push(format!("day {} departed: {id}", self.campaign.day));
            }
        }

        // The storyteller's word, if any.
        let events: Vec<CampaignEvent> = out
            .event
            .iter()
            .map(|e| e.payload.clone())
            .chain(out.arrivals.iter().map(|e| e.payload.clone()))
            .collect();
        if let Some(e) = &out.event {
            self.log.push(format!("day {} event: {:?} — {}", self.campaign.day, e.kind, e.title));
            self.event_kinds.push(format!("{:?}", e.kind));
        }
        for a in &out.arrivals {
            self.log.push(format!("day {} arrival: {}", self.campaign.day, a.title));
            self.event_kinds.push(format!("{:?}", a.kind));
        }
        for payload in &events {
            self.apply_event(payload);
        }

        // Tomorrow's raid musters at next-dawn+12 — the S5 injection seam.
        if let Some(raid) = &out.raid_tomorrow {
            let muster = self.tick + DAY_TICKS + RAID_DAWN_OFFSET;
            self.stage_raid(&raid.clone(), muster);
        }

        // A camped caravan breaks camp: the fixture side goes dormant.
        if out.caravan_departed {
            write_u32(&self.state, &self.state.agent_trader_active_buf, self.w.trader, 0);
            write_f32(&self.state, &self.state.agent_purse_buf, self.w.trader, 0.0);
            write_f32(&self.state, &self.state.agent_inv_hide_buf, self.w.trader, 0.0);
            self.purse_shadow = 0.0;
            self.log.push(format!("day {} caravan departed", self.campaign.day));
        }
    }

    /// THE TROPE → FIXTURE SEAM, in one place (S6b: extracted from
    /// `apply_dawn` so the focused injection tests drive the SAME code the
    /// storyteller drives — the soak's dice skip weight-1 tropes on any
    /// given seed, and an unproven seam is an unproven seam).
    pub fn apply_event(&mut self, payload: &CampaignEvent) {
        let n = self.w.n;
        {
            match payload {
                CampaignEvent::Windfall { drops, gold: _ } => {
                    for d in drops {
                        let buf = match d.item.as_str() {
                            "meal" => Some(self.state.agent_inv_meal_buf.clone()),
                            "timber" => Some(self.state.agent_inv_timber_buf.clone()),
                            _ => None,
                        };
                        if let Some(buf) = buf {
                            let cur = read_f32(&mut self.state, &buf, n)[self.w.rim_cache];
                            write_f32(&self.state, &buf, self.w.rim_cache, cur + d.count as f32);
                        }
                    }
                    self.windfalls += 1;
                }
                CampaignEvent::Festival => {
                    // No event seam from the host: cheer + the mood blend
                    // move directly; the thought_festival VIEW stays flat
                    // for host-injected festivals (documented).
                    let tsum = {
                        let b = self.state.agent_thought_sum_buf.clone();
                        read_f32(&mut self.state, &b, n)
                    };
                    for (_, slot) in self.slot_map.clone() {
                        write_f32(&self.state, &self.state.agent_need_cheer_buf, slot, 1.0);
                        write_f32(
                            &self.state,
                            &self.state.agent_thought_sum_buf,
                            slot,
                            tsum[slot] + 15.0,
                        );
                    }
                }
                CampaignEvent::CaravanArrives { caravan } => {
                    self.caravans += 1;
                    self.purse_shadow = caravan.gold as f64;
                    write_u32(&self.state, &self.state.agent_trader_active_buf, self.w.trader, 1);
                    write_f32(&self.state, &self.state.agent_purse_buf, self.w.trader, caravan.gold as f32);
                    self.host_trade_round();
                }
                CampaignEvent::WandererArrives { id, .. } => {
                    // Bridge policy: the guild takes them in (signing terms
                    // are the bands slice; S6 proves the roster seam).
                    if let Some(slot) = self.free_slots.pop_front() {
                        self.campaign.roster.push(id.clone());
                        self.slot_map.push((id.clone(), slot));
                        self.campaign.director.guest = None;
                        self.activate_slot(slot);
                        self.joins += 1;
                        self.log.push(format!("day {} joined: {id} (slot {slot})", self.campaign.day));
                    } else {
                        self.log.push(format!(
                            "day {} wanderer {id} turned away (no bed slot free)",
                            self.campaign.day
                        ));
                    }
                }
                CampaignEvent::Blight { killed_cells } => {
                    for key in killed_cells {
                        if let Some(slot) = key.strip_prefix('p').and_then(|s| s.parse::<usize>().ok()) {
                            write_u32(&self.state, &self.state.agent_sown_buf, slot, 0);
                            write_f32(&self.state, &self.state.agent_growth_buf, slot, 0.0);
                        }
                    }
                }
                // Campaign-side only: the warband walks the world map (its
                // arrival converts through out.raid_tomorrow), the refugee
                // window and the raid announcement carry no fixture write of
                // their own.
                CampaignEvent::WarbandGathers { .. }
                | CampaignEvent::RefugeeBand { .. }
                | CampaignEvent::RaidIncoming { .. } => {}
            }
        }
    }

    pub fn activate_slot(&mut self, slot: usize) {
        let s = &self.state;
        write_u32(s, &s.agent_alive_buf, slot, 1);
        write_f32(s, &s.agent_hp_buf, slot, 100.0);
        write_u32(s, &s.agent_downed_buf, slot, 0);
        write_u32(s, &s.agent_claimed_job_buf, slot, 0);
        write_u32(s, &s.agent_claim_until_buf, slot, 0);
        write_u32(s, &s.agent_carry_kind_buf, slot, 0);
        write_f32(s, &s.agent_carry_n_buf, slot, 0.0);
        write_f32(s, &s.agent_carry_hide_buf, slot, 0.0);
        write_f32(s, &s.agent_need_food_buf, slot, 1.0);
        write_f32(s, &s.agent_need_rest_buf, slot, 1.0);
        write_f32(s, &s.agent_need_comfort_buf, slot, 0.5);
        write_f32(s, &s.agent_need_cheer_buf, slot, 0.5);
        write_f32(s, &s.agent_mood_buf, slot, 60.0);
        write_f32(s, &s.agent_thought_sum_buf, slot, 0.0);
        write_f32(s, &s.agent_starving_days_buf, slot, 0.0);
        write_f32(s, &s.agent_standing_sum_buf, slot, 0.0);
        write_u32(s, &s.agent_strike_cd_until_buf, slot, 0);
        write_u32(s, &s.agent_directive_kind_buf, slot, 0);
        write_vec3(s, &s.agent_pos_buf, slot, [1.5, 1.5, 0.0]);
        self.prev_starving[slot] = 0.0;
    }

    // -- the day ------------------------------------------------------------

    pub fn run_day(&mut self) -> DawnOutcome {
        for _ in 0..DAY_TICKS {
            self.step_one();
        }
        self.dawn()
    }

    /// ONE fixture tick, with the bridge's own clock kept in lockstep. S12
    /// split this out of `run_day` so a real-time host (`webband_play`) can
    /// spend a day's 600 ticks across rendered frames; the batch path calls
    /// it 600 times and is arithmetically identical to the old
    /// `for .. { state.step() }; tick += DAY_TICKS`.
    pub fn step_one(&mut self) {
        self.state.step();
        self.tick += 1;
    }

    /// EVERYTHING `run_day` did after its 600 steps, in the original order:
    /// settle fixture-side caravan sales -> the host trade round -> snapshot
    /// -> resolve a settled raid -> hunger bookkeeping -> `dawn_fold` ->
    /// write the tropes back. Split out at S12 (see `step_one`); no statement
    /// moved.
    pub fn dawn(&mut self) -> DawnOutcome {
        self.settle_caravan_sales();
        self.host_trade_round(); // a camped caravan trades each morning
        let snap = self.snapshot();
        self.maybe_resolve_raid(&snap);
        self.hungry_member_dawns += snap.starved_today.len() as u64;
        self.member_dawns += snap.members.len() as u64;
        // S13: a political campaign folds through `dawn_fold_political`, which
        // adds step 2 (the road) to the SAME 24-step order. An apolitical one
        // takes the identical pre-S13 call.
        let (out, road) = if self.campaign.politics_enabled {
            let mut fight = errand_fight;
            let (out, road) = dawn_fold_political(&mut self.campaign, &snap, &[], &mut fight);
            (out, Some(road))
        } else {
            (dawn_fold(&mut self.campaign, &snap, &[]), None)
        };
        self.last_snap = Some(snap); // S9: the spine's fold-order pin reads it
        if let Some(road) = road {
            self.apply_road(&road);
        }
        self.apply_dawn(&out);
        if self.campaign.politics_enabled {
            self.apply_politics(&out);
        }
        out
    }

    // -- S13: the guild layer's own injections ------------------------------

    /// The ROAD, applied to the colony: hands that rode out stop working,
    /// hands that came home start again (with their wounds), and the rations
    /// they did not eat go back on the pile by the door.
    pub fn apply_road(&mut self, road: &AfieldReport) {
        for id in &road.home_ids {
            if let Some(pos) = self.away.iter().position(|(m, _)| m == id) {
                let (_, slot) = self.away.remove(pos);
                write_u32(&self.state, &self.state.agent_alive_buf, slot, 1);
                write_u32(&self.state, &self.state.agent_claimed_job_buf, slot, 0);
                write_u32(&self.state, &self.state.agent_claim_until_buf, slot, 0);
                self.log.push(format!("day {} home from the road: {id}", self.campaign.day));
            }
        }
        // Homecoming wounds are ABSOLUTE fractions; a hungry road SUBTRACTS.
        for (id, frac) in &road.set_hp {
            if let Some(slot) = self.slot_of(id) {
                write_f32(&self.state, &self.state.agent_hp_buf, slot, (*frac as f32) * 100.0);
            }
        }
        for (id, d) in &road.hp_drain {
            if let Some(slot) = self.slot_of(id) {
                let cur = self.inv_hp(slot);
                write_f32(
                    &self.state,
                    &self.state.agent_hp_buf,
                    slot,
                    (cur - (*d as f32) * 100.0).max(10.0),
                );
            }
        }
        if road.rations_returned > 0 {
            let n = road.rations_returned as f32;
            let cur = self.inv_at(self.w.yard_cache, "meal");
            let cache = self.w.yard_cache;
            write_f32(&self.state, &self.state.agent_inv_meal_buf, cache, cur + n);
        }
    }

    fn inv_hp(&mut self, slot: usize) -> f32 {
        let n = self.w.n;
        let b = self.state.agent_hp_buf.clone();
        read_f32(&mut self.state, &b, n)[slot]
    }

    /// Everything the guild layer decided this dawn, made VISIBLE (the port's
    /// standing complaint about the politics layer was that its effects were
    /// typed values nobody applied or printed).
    pub fn apply_politics(&mut self, out: &DawnOutcome) {
        if let Some(p) = &out.petition_opened {
            self.petitions_opened += 1;
            let who = petitioner_name(&self.campaign, p);
            self.log.push(format!(
                "day {} petition: {who} — {} (by day {})",
                self.campaign.day,
                describe_petition(&self.campaign, p),
                p.expires_day
            ));
        }
        if let Some(l) = &out.petition_lapsed {
            self.petitions_lapsed += 1;
            self.log.push(format!(
                "day {} petition LAPSED: {} lost {:.2} standing{}",
                self.campaign.day,
                l.faction_id,
                l.standing_cost,
                if l.turned_hostile { " — AND THEY ARE NOW HOSTILE" } else { "" }
            ));
            self.petition_log.push((
                self.campaign.day,
                l.faction_id.clone(),
                l.kind.as_str().to_string(),
                "lapse".to_string(),
            ));
        }
        for b in &out.bands.gave_notice {
            self.band_notices.push((self.campaign.day, b.clone()));
            self.log.push(format!("day {} band gave notice: {b}", self.campaign.day));
        }
        for b in &out.bands.signed {
            self.log.push(format!("day {} band signed on: {b}", self.campaign.day));
        }
        match &out.ambition {
            Some(AmbitionStep::Stage { done, total, line }) => {
                self.stages.push(line.clone());
                self.log.push(format!(
                    "day {} AMBITION {done}/{total}: {line}",
                    self.campaign.day
                ));
            }
            Some(AmbitionStep::Achieved { title }) => {
                self.achieved = Some(title.clone());
                self.log
                    .push(format!("day {} AMBITION ACHIEVED: {title}", self.campaign.day));
            }
            None => {}
        }
        // Thought injections (politics + the road) land on the mood blend, the
        // same channel `resolve_raid`'s already use.
        let n = self.w.n;
        let tsum = {
            let b = self.state.agent_thought_sum_buf.clone();
            read_f32(&mut self.state, &b, n)
        };
        for (id, key) in &out.thoughts {
            let delta = match key.as_str() {
                "home_served" => 6.0,
                "home_refused" => -6.0,
                "hungry_road" => -8.0,
                "victory" => 8.0,
                "defeat" => -10.0,
                _ => 0.0,
            };
            if delta == 0.0 {
                continue;
            }
            if let Some(slot) = self.slot_of(id) {
                write_f32(
                    &self.state,
                    &self.state.agent_thought_sum_buf,
                    slot,
                    tsum[slot] + delta,
                );
            }
        }
    }

    /// What the colony can put behind an answer today — read from the LAST
    /// dawn snapshot, never recomputed here.
    pub fn capacity(&self) -> PetitionCapacity {
        match &self.last_snap {
            Some(s) => petition_capacity(&self.campaign, s),
            None => PetitionCapacity::default(),
        }
    }

    /// The three answers, with their costs and their blocked reasons, straight
    /// from the sim seam. A front-end RENDERS these; it never works out
    /// affordability for itself (the `canPlace` law).
    pub fn choices(&self) -> Vec<PetitionChoice> {
        match &self.campaign.petition {
            Some(p) => petition_choices(&self.campaign, p, self.capacity()),
            None => Vec::new(),
        }
    }

    /// TAKE AN ANSWER. `Ok(sentence)` = it landed; `Err(reason)` = the sim
    /// refused it, in the sim's own words.
    pub fn answer(&mut self, choice: PetitionChoiceKind) -> Result<String, String> {
        let Some(p) = self.campaign.petition.clone() else {
            return Err("no one is asking anything of the guild today".to_string());
        };
        if p.chosen.is_some() {
            return Err("that ask is already answered — the company is on the road".to_string());
        }
        let cap = self.capacity();
        if let Some(o) = self.choices().into_iter().find(|o| o.choice == choice) {
            if let Some(why) = o.blocked {
                return Err(why);
            }
        }
        let who = petitioner_name(&self.campaign, &p);
        let kind = p.kind.as_str().to_string();
        let word = match choice {
            PetitionChoiceKind::Send => "send",
            PetitionChoiceKind::Pay => "pay",
            PetitionChoiceKind::Refuse => "refuse",
        };
        // SEND is the only answer that costs PEOPLE: it is completed by a
        // dispatch, and the hands leave the colony's work until they are home.
        if choice == PetitionChoiceKind::Send {
            let sentence = self.send_company(&p)?;
            self.petitions_answered += 1;
            self.petition_log.push((self.campaign.day, p.faction_id.clone(), kind, word.into()));
            return Ok(sentence);
        }
        match answer_petition(&mut self.campaign, choice, cap) {
            Some(thoughts) => {
                self.apply_thoughts(&thoughts);
                self.petitions_answered += 1;
                self.petition_log.push((
                    self.campaign.day,
                    p.faction_id.clone(),
                    kind,
                    word.to_string(),
                ));
                let line = match choice {
                    PetitionChoiceKind::Pay => {
                        format!("{who} was paid {} gold — remembered, by half.", p.need_gold)
                    }
                    _ => format!("{who} was refused. Their rivals will hear of it."),
                };
                self.log.push(format!("day {} answered ({word}): {line}", self.campaign.day));
                Ok(line)
            }
            None => Err("the sim refused that answer".to_string()),
        }
    }

    /// The SEND: pick the hands, price the road, pack the rations, and take
    /// the bodies off colony work. Every refusal sentence comes from
    /// `dispatch_cost` — the seam, not this file.
    fn send_company(&mut self, p: &Petition) -> Result<String, String> {
        let stacks = self.last_snap.as_ref().map(|s| s.inventory.stacks.clone()).unwrap_or_default();
        let unavailable: Vec<String> = self
            .last_snap
            .as_ref()
            .map(|s| s.members.iter().filter(|m| !m.ready).map(|m| m.id.clone()).collect())
            .unwrap_or_default();
        // Hands, in roster order, that can actually ride.
        let hands: Vec<String> = self
            .campaign
            .roster
            .iter()
            .filter(|id| !is_afield(&self.campaign, id) && !unavailable.contains(id))
            .take(p.need_hands as usize)
            .cloned()
            .collect();
        let opts = DispatchOpts {
            target_landmark_id: p.landmark_id.clone(),
            petition_id: Some(p.id.clone()),
            ..Default::default()
        };
        let ctx = DispatchContext { stacks: &stacks, unavailable: &unavailable };
        let cost = dispatch_cost(&self.campaign, &hands, &opts, ctx);
        if let Some(why) = cost.blocked {
            self.refusals.push(why.clone());
            return Err(why);
        }
        let Some(d) = dispatch_party(&mut self.campaign, &hands, opts, ctx) else {
            return Err("the sim refused the dispatch".to_string());
        };
        // FIXTURE INJECTIONS: the rations leave the larder, the hands leave
        // the work.
        for (item, n) in &d.take_items {
            self.remove_stock(item, *n);
        }
        for id in &d.away_ids {
            if let Some(slot) = self.slot_of(id) {
                write_u32(&self.state, &self.state.agent_alive_buf, slot, 0);
                write_u32(&self.state, &self.state.agent_claimed_job_buf, slot, 0);
                self.away.push((id.clone(), slot));
            }
        }
        let names = self.campaign.member_names(&d.away_ids);
        let line = format!(
            "{names} rode out — {} hands, {} days of road, {} rations packed.",
            d.away_ids.len(),
            d.party.travel_days,
            d.party.provisions.round() as i64
        );
        self.log.push(format!("day {} answered (send): {line}", self.campaign.day));
        Ok(line)
    }

    fn apply_thoughts(&mut self, thoughts: &[(String, String)]) {
        let n = self.w.n;
        let tsum = {
            let b = self.state.agent_thought_sum_buf.clone();
            read_f32(&mut self.state, &b, n)
        };
        for (id, key) in thoughts {
            let delta = match key.as_str() {
                "home_served" => 6.0,
                "home_refused" => -6.0,
                _ => 0.0,
            };
            if delta == 0.0 {
                continue;
            }
            if let Some(slot) = self.slot_of(id) {
                write_f32(&self.state, &self.state.agent_thought_sum_buf, slot, tsum[slot] + delta);
            }
        }
    }

    /// Take `count` units of `item` out of the colony's holders, store first.
    fn remove_stock(&mut self, item: &str, count: u32) {
        let buf = match item {
            "meal" => self.state.agent_inv_meal_buf.clone(),
            "berries" => self.state.agent_inv_berries_buf.clone(),
            "venison" => self.state.agent_inv_venison_buf.clone(),
            "grain" => self.state.agent_inv_grain_buf.clone(),
            "timber" => self.state.agent_inv_timber_buf.clone(),
            "plank" => self.state.agent_inv_plank_buf.clone(),
            "hide" => self.state.agent_inv_hide_buf.clone(),
            _ => return,
        };
        let mut left = count as f32;
        for holder in [self.w.store, self.w.yard_cache, self.w.rim_cache, self.w.hearth] {
            if left <= 0.0 {
                break;
            }
            let n = self.w.n;
            let have = read_f32(&mut self.state, &buf, n)[holder];
            let take = have.min(left);
            if take > 0.0 {
                write_f32(&self.state, &buf, holder, have - take);
                left -= take;
            }
        }
    }

    /// A day of COLONY time with no host fold — the fixture's own dawn
    /// systems (eat/heal/spoil/growth) still fire at `%600==0`; the
    /// storyteller, the provisioner and the exodus do not. The focused
    /// injection tests use this so the seam under test is the only thing
    /// moving (an organic raid mid-test would plunder the very rim cache
    /// the windfall test is watching).
    pub fn run_day_quiet(&mut self) {
        for _ in 0..DAY_TICKS {
            self.step_one();
        }
    }

    pub fn inv_at(&mut self, slot: usize, item: &str) -> f32 {
        let n = self.w.n;
        let buf = match item {
            "meal" => self.state.agent_inv_meal_buf.clone(),
            "berries" => self.state.agent_inv_berries_buf.clone(),
            "venison" => self.state.agent_inv_venison_buf.clone(),
            "grain" => self.state.agent_inv_grain_buf.clone(),
            "timber" => self.state.agent_inv_timber_buf.clone(),
            "plank" => self.state.agent_inv_plank_buf.clone(),
            "hide" => self.state.agent_inv_hide_buf.clone(),
            other => panic!("no such fixture item: {other}"),
        };
        read_f32(&mut self.state, &buf, n)[slot]
    }

    pub fn u32_at(&mut self, buf: &wgpu::Buffer, slot: usize) -> u32 {
        let n = self.w.n;
        read_u32(&mut self.state, buf, n)[slot]
    }

    pub fn pos_at(&mut self, slot: usize) -> [f32; 3] {
        let n = self.w.n;
        let b = self.state.agent_pos_buf.clone();
        let p = read_vec4(&mut self.state, &b, n)[slot];
        [p[0], p[1], p[2]]
    }

    pub fn outcome(&self, out: &DawnOutcome) -> CampaignOutcome {
        if out.fell {
            CampaignOutcome::Fell { day: self.campaign.day }
        } else {
            CampaignOutcome::Ongoing
        }
    }

    // -- the determinism digest ---------------------------------------------

    /// SIM-STATE buffers only (S5 finding 2: tally/count views ride the
    /// lossy fold window and are NOT run-to-run stable under volume).
    pub fn fixture_signature(&mut self) -> Vec<(&'static str, Vec<u32>)> {
        let n = self.w.n;
        let mut sig: Vec<(&'static str, Vec<u32>)> = Vec::new();
        let f32_bufs: [(&'static str, wgpu::Buffer); 16] = [
            ("mood", self.state.agent_mood_buf.clone()),
            ("hp", self.state.agent_hp_buf.clone()),
            ("need_food", self.state.agent_need_food_buf.clone()),
            ("starving_days", self.state.agent_starving_days_buf.clone()),
            ("thought_sum", self.state.agent_thought_sum_buf.clone()),
            ("standing_sum", self.state.agent_standing_sum_buf.clone()),
            ("inv_meal", self.state.agent_inv_meal_buf.clone()),
            ("inv_berries", self.state.agent_inv_berries_buf.clone()),
            ("inv_venison", self.state.agent_inv_venison_buf.clone()),
            ("inv_grain", self.state.agent_inv_grain_buf.clone()),
            ("inv_timber", self.state.agent_inv_timber_buf.clone()),
            ("inv_plank", self.state.agent_inv_plank_buf.clone()),
            ("inv_hide", self.state.agent_inv_hide_buf.clone()),
            ("work_left", self.state.agent_work_left_buf.clone()),
            ("purse", self.state.agent_purse_buf.clone()),
            ("carry_n", self.state.agent_carry_n_buf.clone()),
        ];
        for (name, b) in &f32_bufs {
            sig.push((name, read_f32(&mut self.state, b, n).iter().map(|v| v.to_bits()).collect()));
        }
        let u32_bufs: [(&'static str, wgpu::Buffer); 10] = [
            ("alive", self.state.agent_alive_buf.clone()),
            ("built", self.state.agent_built_buf.clone()),
            ("claimed_job", self.state.agent_claimed_job_buf.clone()),
            ("downed", self.state.agent_downed_buf.clone()),
            ("raid_active", self.state.agent_raid_active_buf.clone()),
            ("plundered_at", self.state.agent_plundered_at_buf.clone()),
            ("burnt", self.state.agent_burnt_buf.clone()),
            ("warned", self.state.agent_warned_buf.clone()),
            ("trader_active", self.state.agent_trader_active_buf.clone()),
            ("sown", self.state.agent_sown_buf.clone()),
        ];
        for (name, b) in &u32_bufs {
            sig.push((name, read_u32(&mut self.state, b, n)));
        }
        let posb = self.state.agent_pos_buf.clone();
        let posv = read_vec4(&mut self.state, &posb, n);
        sig.push((
            "pos_xyz",
            posv.iter()
                .flat_map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                .collect(),
        ));
        sig
    }
}

/// The resolver `dawn_fold_political` calls when a FIGHTING errand arrives.
/// The host layer deliberately owns no battle runtime, and the fixture's arena
/// is the colony itself — a second detached battle instance is S7's deferred
/// item and still is. Nothing this slice wires produces a fighting errand (a
/// petition send is peaceful: `DispatchOpts::comp` is `None`, and
/// `resolve_errand` short-circuits on that before ever calling here), so this
/// is a documented, deterministic placeholder rather than a fake battle:
/// the company wins when it outnumbers what it was sent against.
fn errand_fight(c: &Campaign, p: &AfieldParty) -> ErrandFight {
    let foes: i64 = p.comp.as_ref().map_or(0, |comp| comp.iter().map(|(_, n)| i64::from(*n)).sum());
    let victory = p.member_ids.len() as i64 * 3 >= foes;
    ErrandFight {
        victory,
        gold: if victory { foes * 4 } else { 0 },
        member_hp: p
            .member_ids
            .iter()
            .filter(|id| c.roster.iter().any(|r| r == *id))
            .map(|id| (id.clone(), if victory { 0.7 } else { 0.35 }))
            .collect(),
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn sig_hash(sig: &[(&'static str, Vec<u32>)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (name, bits) in sig {
        h ^= fnv1a(name.as_bytes());
        h = h.wrapping_mul(0x100000001b3);
        h ^= fnv1a(bytemuck::cast_slice(bits));
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
