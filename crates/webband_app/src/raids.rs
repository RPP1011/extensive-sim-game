//! Raid math — the port of `F:\MB\src\colony\raids.ts` (budget/tier/comp
//! formulas verbatim, incl. the escalation-clock day terms) and the
//! plunder-not-death resolution as PURE functions over inventory snapshots.
//! The battle itself is fixture business (S5); this module is what feeds it
//! (composition in) and what lands its consequences (gold/renown/plunder out).

use serde::{Deserialize, Serialize};

use crate::defs::{item_nutrition, item_value, js_round, InventorySnapshot};
use crate::rng::{rng_float, rng_int, rng_pick, RngState};
use crate::scenario::ScenarioId;
use crate::worldgen::{ELITE_EPITHETS, ELITE_NAMES};

// ---------- Troops ----------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TroopType {
    pub id: &'static str,
    pub name: &'static str,
    pub hp: f64,
    pub damage: f64,
    pub speed: f64,
    pub reach: f64,
    pub block_chance: f64,
    pub loot: i64,
    /// Budget cost when generating enemy warbands.
    pub power: f64,
}

/// `F:\MB\src\campaign\data.ts:26-46`, verbatim (colors are render-only).
pub const TROOPS: [TroopType; 4] = [
    TroopType { id: "looter", name: "Looter", hp: 42.0, damage: 11.0, speed: 5.0, reach: 1.9, block_chance: 0.06, loot: 9, power: 1.0 },
    TroopType { id: "bandit", name: "Bandit", hp: 72.0, damage: 17.0, speed: 5.5, reach: 2.1, block_chance: 0.22, loot: 20, power: 2.0 },
    TroopType { id: "raider", name: "Raider", hp: 105.0, damage: 24.0, speed: 5.7, reach: 2.2, block_chance: 0.38, loot: 38, power: 3.5 },
    TroopType { id: "warlord", name: "Warlord", hp: 340.0, damage: 34.0, speed: 5.9, reach: 2.7, block_chance: 0.5, loot: 0, power: 12.0 },
];

pub fn troop(id: &str) -> Option<&'static TroopType> {
    TROOPS.iter().find(|t| t.id == id)
}

/// Warband composition, first-seen key order (the JS `Record` insertion
/// order — comp equality in saves and tests depends on it).
pub type PartyComp = Vec<(String, u32)>;

pub fn comp_power(comp: &PartyComp) -> f64 {
    comp.iter()
        .map(|(id, n)| troop(id).map_or(1.0, |t| t.power) * f64::from(*n))
        .sum()
}

/// `raids.ts rollComp`: greedy spend of a power budget — uniform seeded pick
/// among the affordable of looter/bandit/raider until the budget is below a
/// looter (guard 200 iterations, the TS's own runaway brake).
pub fn roll_comp(g: &mut RngState, budget: f64) -> PartyComp {
    let mut comp: PartyComp = Vec::new();
    let mut left = budget;
    let mut guard = 200;
    let looter_power = troop("looter").expect("troop table").power;
    while left >= looter_power && guard > 0 {
        guard -= 1;
        let affordable: Vec<&str> = ["looter", "bandit", "raider"]
            .into_iter()
            .filter(|id| troop(id).is_some_and(|t| t.power <= left))
            .collect();
        let id = *rng_pick(g, &affordable);
        match comp.iter_mut().find(|(k, _)| k == id) {
            Some((_, n)) => *n += 1,
            None => comp.push((id.to_string(), 1)),
        }
        left -= troop(id).expect("troop table").power;
    }
    comp
}

// ---------- Budget / tier (the escalation clock) ----------

/// `raids.ts:64-66`, verbatim — port the MEASURED formula, not the comment.
/// The playtest table (three unwalled founders): budget 6/8/10/12/14/16/20 →
/// win% 100/100/100/75/33/13/0; a fresh colony lands near 12. The `day` term
/// is the escalation clock: enemies scale with time, members never decay.
pub fn raid_budget(wealth: i64, colonists: usize, day: i64) -> i64 {
    js_round(2.0 + colonists as f64 * 2.0 + wealth as f64 * 0.005 + day as f64 * 0.25)
}

/// `raids.ts:68-70`, verbatim.
pub fn raid_tier(wealth: i64, day: i64) -> i64 {
    (1 + (wealth as f64 + day as f64 * 12.0).div_euclid(600.0) as i64).clamp(1, 7)
}

// ---------- Spawn ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveRaid {
    pub id: String,
    pub comp: PartyComp,
    pub tier: i64,
    pub elite_name: Option<String>,
    /// The morning it hits (spawned the evening before — one night to prepare).
    pub arrives_day: i64,
    /// Hex edge direction 0-5 they enter from.
    pub entry_dir: i64,
    /// A world-map warband that arrived: victory breaks that threat.
    pub threat_ref: Option<String>,
    /// A vendetta raid: the fold can settle that saga.
    pub saga_ref: Option<String>,
    /// A band's cause: victory signs (or re-satisfies) that band.
    pub band_ref: Option<String>,
    /// The power that sent them, when one did (factions — later slice; the
    /// field is carried so saves stay forward-compatible).
    pub faction_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SpawnOpts {
    pub elite_name: Option<String>,
    pub threat_ref: Option<String>,
    pub saga_ref: Option<String>,
    pub band_ref: Option<String>,
    pub faction_id: Option<String>,
    pub tier_bump: i64,
}

/// `raids.ts spawnRaid` with the TS's exact draw order: id → comp →
/// elite-name (float always draws when no name is supplied; the two picks
/// only on the coin) → entry dir. The city remap happens AFTER the roll so
/// the seeded stream stays byte-identical across scenarios.
pub fn spawn_raid(
    g: &mut RngState,
    day: i64,
    wealth: i64,
    roster_len: usize,
    scenario: ScenarioId,
    opts: SpawnOpts,
) -> ActiveRaid {
    let tier = (raid_tier(wealth, day) + opts.tier_bump).min(7);
    let id = format!("raid{}", rng_int(g, 1, 1 << 30));
    let comp = roll_comp(
        g,
        raid_budget(wealth, roster_len, day) as f64 * (1.0 + tier as f64 * 0.15),
    );
    let elite_name = match opts.elite_name {
        Some(n) => Some(n),
        None => {
            if rng_float(g) < 0.5 {
                Some(format!("{} {}", rng_pick(g, &ELITE_NAMES), rng_pick(g, &ELITE_EPITHETS)))
            } else {
                None
            }
        }
    };
    let entry_dir = {
        let d = rng_int(g, 0, 5);
        if scenario != ScenarioId::City {
            d
        } else if d == 2 {
            1
        } else if d == 3 {
            4
        } else {
            d
        }
    };
    ActiveRaid {
        id,
        comp,
        tier,
        elite_name,
        arrives_day: day + 1,
        entry_dir,
        threat_ref: opts.threat_ref,
        saga_ref: opts.saga_ref,
        band_ref: opts.band_ref,
        faction_id: opts.faction_id,
    }
}

// ---------- Plunder (defeat's math, pure) ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlunderSeverity {
    /// The line broke (a lost defense).
    Beaten,
    /// Nobody home at all — worse: no line to break means a longer sack.
    Undefended,
}

/// What defeat takes, computed but NOT applied — the campaign loop translates
/// `taken`/`burnt` into fixture injections and applies the scalar deltas to
/// campaign state (see `campaign::resolve_raid`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlunderOutcome {
    /// Per-stack units taken, by stack id (only stacks that lost anything).
    pub taken: Vec<(String, u32)>,
    pub taken_total: u32,
    /// BUILT buildings to mark `burnt`, outermost first.
    pub burnt_building_ids: Vec<String>,
    /// `renown -= this` (floored at 0 by the applier).
    pub renown_loss: i64,
    /// `director.points -= 20` (floored at 0 by the applier).
    pub director_points_loss: i64,
    /// `director.reliefUntil = day + this` — the relief window.
    pub relief_days: i64,
}

/// `raids.ts plunder`, verbatim math over a snapshot:
/// - `take = (10 + tier*8) * (undefended ? 2 : 1)` stock units, by item value
///   descending (tie: stack id order — the TS `localeCompare`);
/// - edible stacks are FLOORED at `ceil(roster*7 / nutrition / stacks.len())`
///   so a company on the road never comes home to a dead colony;
/// - burns the `undefended ? 3 : 2` outermost BUILT buildings (|q|+|r| desc,
///   stable order — JS sort is stable and so is Rust's);
/// - renown −6/−4, storyteller −20 points, relief window 4/3 days.
pub fn plunder(
    inv: &InventorySnapshot,
    roster_len: usize,
    tier: i64,
    severity: PlunderSeverity,
) -> PlunderOutcome {
    let heavy = severity == PlunderSeverity::Undefended;
    let take = (10 + tier * 8) as u32 * if heavy { 2 } else { 1 };
    // Leave a week of food standing whatever else they carry off.
    let spare = roster_len as f64 * 7.0;

    let mut by_value: Vec<&crate::defs::StackView> = inv.stacks.iter().collect();
    by_value.sort_by(|a, b| {
        item_value(&b.item)
            .cmp(&item_value(&a.item))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut taken_total = 0u32;
    let mut taken: Vec<(String, u32)> = Vec::new();
    let n_stacks = by_value.len().max(1) as f64;
    for s in &by_value {
        if taken_total >= take {
            break;
        }
        let nutrition = item_nutrition(&s.item);
        let floor = if nutrition > 0.0 {
            (spare / nutrition / n_stacks).ceil() as u32
        } else {
            0
        };
        let can_take = s.count.saturating_sub(floor);
        let n = can_take.min(take - taken_total);
        if n > 0 {
            taken.push((s.id.clone(), n));
            taken_total += n;
        }
    }

    let mut burnable: Vec<&crate::defs::BuildingView> =
        inv.buildings.iter().filter(|b| b.built).collect();
    burnable.sort_by_key(|b| -(i64::from(b.q.abs()) + i64::from(b.r.abs())));
    let burn_n = burnable.len().min(if heavy { 3 } else { 2 });
    let burnt_building_ids = burnable[..burn_n].iter().map(|b| b.id.clone()).collect();

    PlunderOutcome {
        taken,
        taken_total,
        burnt_building_ids,
        renown_loss: if heavy { 6 } else { 4 },
        director_points_loss: 20,
        relief_days: if heavy { 4 } else { 3 },
    }
}

/// Victory's renown dividend (`raids.ts:139`).
pub fn victory_renown(tier: i64) -> i64 {
    4 + tier * 2
}
