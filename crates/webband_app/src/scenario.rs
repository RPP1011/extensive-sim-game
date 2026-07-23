//! Starting conditions — port of `F:\MB\src\guild\scenario.ts`. Four opening
//! problems out of the same systems, varying WHICH scarcity bites. All
//! numbers verbatim from the TS.
//!
//! THE COMPARABILITY LAW: a scenario is applied AFTER every founding roll and
//! applies only deterministic mutations (stock, gold, standing, flags). It
//! takes NO draws from the seeded stream — `apply_scenario` doesn't even
//! receive the rng, so the law is enforced by signature. Same seed → same
//! cast, country and goals in all four.

use serde::{Deserialize, Serialize};

use crate::founding::{Founding, Stack, CACHE_GROUND};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioId {
    Village,
    Town,
    Wilderness,
    City,
}

pub const ALL_SCENARIOS: [ScenarioId; 4] = [
    ScenarioId::Village,
    ScenarioId::Town,
    ScenarioId::Wilderness,
    ScenarioId::City,
];

pub const DEFAULT_SCENARIO: ScenarioId = ScenarioId::Village;

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSpec {
    pub id: ScenarioId,
    pub name: &'static str,
    /// The pitch, and honestly the difficulty warning.
    pub blurb: &'static str,
    /// Can the guild sign bands and take in wanderers at all?
    pub recruiting: bool,
    /// Coin in the chest on day one.
    pub gold: i64,
    /// What is already in the stockpile (dropped as stray stacks at the
    /// cache ground, in this order).
    pub stock: &'static [(&'static str, u32)],
    /// How the country regards a newcomer here (standing offset applied to
    /// every petitioner power — powers land in a later slice; the offset is
    /// recorded on the founding).
    pub standing: i64,
    /// A standing local trade: gold per day, no work required.
    pub trade_per_day: i64,
    /// The local FOOD MARKET: what a bought meal costs, 0 = no market.
    pub meal_price: i64,
    /// Discount on the signing machinery (the village's recruit pool).
    pub sign_discount: f64,
    /// The city's squeeze: gold owed every dawn. Unpaid rent bleeds STANDING,
    /// never a scripted eviction.
    pub rent_per_day: i64,
}

pub fn scenario_spec(id: ScenarioId) -> &'static ScenarioSpec {
    match id {
        ScenarioId::Village => &VILLAGE,
        ScenarioId::Town => &TOWN,
        ScenarioId::Wilderness => &WILDERNESS,
        ScenarioId::City => &CITY,
    }
}

static VILLAGE: ScenarioSpec = ScenarioSpec {
    id: ScenarioId::Village,
    name: "Beside a village",
    blurb: "A thin living and good neighbours. There are hands here to be won \
            over, if you can feed them — but no coin worth speaking of.",
    recruiting: true,
    gold: 90,
    stock: &[("meal", 14), ("timber", 10), ("plank", 4)],
    standing: 10,
    trade_per_day: 2,
    meal_price: 3,
    sign_discount: 0.55,
    rent_per_day: 0,
};

static TOWN: ScenarioSpec = ScenarioSpec {
    id: ScenarioId::Town,
    name: "Outside a resentful town",
    blurb: "A rich place that did not ask you to come. Its markets are open to \
            you and its people are not: nobody here will take your colours.",
    recruiting: false,
    gold: 420,
    stock: &[("meal", 26), ("timber", 18), ("plank", 12), ("herbs", 6)],
    standing: -30,
    trade_per_day: 14,
    meal_price: 3,
    sign_discount: 1.0,
    rent_per_day: 0,
};

static WILDERNESS: ScenarioSpec = ScenarioSpec {
    id: ScenarioId::Wilderness,
    name: "Alone in the wild",
    blurb: "No neighbours, no market, no help. Whatever this company becomes, \
            it will become by its own hands first.",
    recruiting: false,
    gold: 25,
    stock: &[("meal", 6), ("timber", 4)],
    standing: 0,
    trade_per_day: 0,
    meal_price: 0,
    sign_discount: 1.0,
    rent_per_day: 0,
};

static CITY: ScenarioSpec = ScenarioSpec {
    id: ScenarioId::City,
    name: "Within the city walls",
    blurb: "A charter, a yard inside the walls, and a landlord. People will \
            sign and the market is wide — but everything here has a price, \
            starting with the roof over your head.",
    recruiting: true,
    gold: 200,
    stock: &[("meal", 12), ("timber", 8)],
    standing: 0,
    trade_per_day: 0,
    meal_price: 3,
    sign_discount: 0.8,
    rent_per_day: 8,
};

/// Spoil windows for the items scenarios stock (`defs.ts` ITEMS). `None` =
/// never spoils. Stamped as an absolute day at drop time, the TS idiom.
pub fn spoil_days(item: &str) -> Option<i64> {
    match item {
        "berries" => Some(5),
        "grain" => Some(40),
        "meal" => Some(3),
        "herbs" => Some(10),
        "venison" => Some(3),
        _ => None,
    }
}

/// Stack caps for the same items — kept so the simplified drop below can
/// assert it never needs the full `dropItems` merge/spill machinery.
pub fn stack_max(item: &str) -> u32 {
    match item {
        "meal" => 40,
        "poultice" => 25,
        "herbs" | "hide" | "venison" => 50,
        _ => 75,
    }
}

/// Stamp a founding with its starting condition. Runs LAST, after every roll,
/// and takes no draws (no rng parameter — the law is structural).
///
/// Simplification vs the TS: `dropItems`' merge-≤-stackMax / earlier-spoilDay
/// / ring-spill logic is NOT ported (it belongs to the colony jobs slice).
/// Scenario stocks always fit one fresh stack per item — every stock count is
/// below its item's stackMax, which `scenario_tables_are_sane` asserts.
pub fn apply_scenario(f: &mut Founding, id: ScenarioId) {
    let s = scenario_spec(id);
    f.scenario = id;
    f.gold = s.gold;

    // THE RESOURCE CACHE: everything the wagon left, dropped as stray stacks
    // on the founding ground itself.
    f.stacks.clear();
    for &(item, n) in s.stock {
        f.stacks.push(Stack {
            item: item.to_string(),
            count: n,
            q: CACHE_GROUND.0,
            r: CACHE_GROUND.1,
            spoil_day: spoil_days(item).map(|d| f.day + d),
        });
    }

    // Standing offset toward the country's powers (factions land in a later
    // slice; the founding records the offset the powers will inherit).
    f.standing = s.standing;

    // The provisioner's standing order: where a market exists, the guild
    // buys its bread by default.
    f.provisioning = if s.meal_price > 0 { 4 } else { 0 };
}
