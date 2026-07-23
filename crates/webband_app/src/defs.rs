//! The colony data catalog the CAMPAIGN BRAIN needs — a host-side mirror of
//! `F:\MB\src\colony\defs.ts` (items + building costs), plus the wealth and
//! larder math built on it. The FIXTURE (S3, `assets/sim/webband_colony.sim`)
//! owns the live economy; this table exists so campaign math (raid budgets,
//! plunder value sorting, the provisioner's food-days gauge) computes over
//! plain snapshots without asking the sim. Numbers verbatim from the TS
//! (webband-port-data.md §1 carries the citations).

use serde::{Deserialize, Serialize};

/// JS `Math.round` — half-up toward +∞ (Rust's `round` is half-away-from-zero,
/// which differs for negative halves). Every TS `Math.round` ports through
/// this so a negative-wealth edge case can never skew a pin.
pub fn js_round(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItemDef {
    pub id: &'static str,
    pub name: &'static str,
    pub stack_max: u32,
    pub value: i64,
    pub nutrition: f64,
    pub meal: bool,
    /// `None` = never spoils. Spoil is an ABSOLUTE day stamped at drop time.
    pub spoil_days: Option<i64>,
    pub medicinal: bool,
}

/// `defs.ts:80-90`, verbatim (colors/icons are render-only and stay behind).
pub const ITEMS: [ItemDef; 9] = [
    ItemDef { id: "berries", name: "Berries", stack_max: 75, value: 1, nutrition: 0.5, meal: false, spoil_days: Some(5), medicinal: false },
    ItemDef { id: "grain", name: "Grain", stack_max: 75, value: 1, nutrition: 0.25, meal: false, spoil_days: Some(40), medicinal: false },
    ItemDef { id: "meal", name: "Meals", stack_max: 40, value: 3, nutrition: 1.0, meal: true, spoil_days: Some(3), medicinal: false },
    ItemDef { id: "timber", name: "Timber", stack_max: 75, value: 2, nutrition: 0.0, meal: false, spoil_days: None, medicinal: false },
    ItemDef { id: "plank", name: "Planks", stack_max: 75, value: 3, nutrition: 0.0, meal: false, spoil_days: None, medicinal: false },
    ItemDef { id: "herbs", name: "Herbs", stack_max: 50, value: 2, nutrition: 0.0, meal: false, spoil_days: Some(10), medicinal: false },
    ItemDef { id: "poultice", name: "Poultices", stack_max: 25, value: 8, nutrition: 0.0, meal: false, spoil_days: None, medicinal: true },
    ItemDef { id: "hide", name: "Hides", stack_max: 50, value: 4, nutrition: 0.0, meal: false, spoil_days: None, medicinal: false },
    ItemDef { id: "venison", name: "Venison", stack_max: 50, value: 2, nutrition: 0.5, meal: false, spoil_days: Some(3), medicinal: false },
];

pub fn item_def(id: &str) -> Option<&'static ItemDef> {
    ITEMS.iter().find(|d| d.id == id)
}

pub fn item_value(id: &str) -> i64 {
    item_def(id).map_or(0, |d| d.value)
}

pub fn item_nutrition(id: &str) -> f64 {
    item_def(id).map_or(0.0, |d| d.nutrition)
}

/// Building material costs (`defs.ts:151-170` BUILDINGS `cost` column) — the
/// slice the wealth formula reads. Work minutes / footprints / effects are
/// fixture business (S3) and are deliberately not mirrored here.
pub fn building_cost(kind: &str) -> &'static [(&'static str, u32)] {
    match kind {
        "hearth" => &[("timber", 4)],
        "workbench" => &[("timber", 6)],
        "bed" => &[("plank", 2)],
        "cot" => &[("plank", 3)],
        "mess_table" => &[("plank", 6)],
        "store_shed" => &[("plank", 9)],
        "muster_ground" => &[("timber", 4), ("plank", 2)],
        "wall" => &[("timber", 1)],
        "door" => &[("plank", 2)],
        _ => &[],
    }
}

// ---------- Snapshot views (fixture → host, plain data) ----------

/// One item stack as the fixture reports it. `id` carries the TS stack id —
/// plunder's value-sort tie-break is `a.id.localeCompare(b.id)`, which plain
/// byte order reproduces for the ASCII ids both sides generate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StackView {
    pub id: String,
    pub item: String,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildingView {
    pub id: String,
    pub kind: String,
    pub q: i32,
    pub r: i32,
    /// `state === 'built'` — blueprints and burnt shells are false.
    pub built: bool,
}

/// What the campaign math needs to know about the colony's THINGS.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InventorySnapshot {
    pub stacks: Vec<StackView>,
    pub buildings: Vec<BuildingView>,
}

/// `defs.ts colonyWealth`, verbatim: gold + 40 per head + stack values +
/// twice the material value of every BUILT structure, rounded. The
/// storyteller's difficulty axis — prosperity draws wolves.
pub fn colony_wealth(gold: i64, roster_len: usize, inv: &InventorySnapshot) -> i64 {
    let mut w = gold as f64 + roster_len as f64 * 40.0;
    for s in &inv.stacks {
        w += (item_value(&s.item) * i64::from(s.count)) as f64;
    }
    for b in &inv.buildings {
        if !b.built {
            continue;
        }
        for &(item, n) in building_cost(&b.kind) {
            w += (item_value(item) * i64::from(n) * 2) as f64;
        }
    }
    js_round(w)
}

/// `colony-day.ts foodDays` — the larder in nutrition-days, rounded to one
/// decimal (the HUD/e2e gauge; the provisioner's loop condition).
pub fn food_days(stacks: &[StackView]) -> f64 {
    let mut units = 0.0;
    for s in stacks {
        units += item_nutrition(&s.item) * f64::from(s.count);
    }
    (units * 10.0).round() / 10.0
}
