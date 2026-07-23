//! LANDMARK MARKETS (`F:\MB\src\guild\markets.ts`) — where the country sells
//! food, and the reason "go see if the village sells cheaper meat" is a real
//! order.
//!
//! LAWS ENCODED HERE:
//! 1. **Prices DERIVE** — kind table + id-hash jitter, ZERO draws, the same
//!    market every founding, nothing persisted but what the guild has LEARNED.
//! 2. **KNOWLEDGE IS THE RESOURCE**: a market's prices are unknown until
//!    someone rides there and asks ([`crate::afield::Errand::Inquire`] writes
//!    `market_intel[id] = day`), and only a KNOWN market can be named as the
//!    provisioner's source ([`provision_choice`]).
//! 3. **Prices drift by SEASON** ([`MARKET_SEASON_DAYS`]), so knowledge AGES:
//!    what the guild believes is the prices of the epoch somebody last saw
//!    ([`believed_market`]).
//! 4. **Distance never disappears, it becomes a number** — [`haulage_from`] is
//!    a per-unit surcharge priced off the road.
//!
//! SIMPLIFIED, documented: settlement life (`guild/life.ts` — per-landmark
//! populations and stores) is NOT ported, so the scarcity shift that reads a
//! settlement's real stores is always 0 here and a landmark is never sacked
//! unless something else pushes it onto [`crate::campaign::Campaign::sacked`].
//! The base table, the epoch jitter, the haulage and the intel machinery are
//! whole.

use crate::campaign::Campaign;
use crate::defs::item_nutrition;
use crate::noise::{hash, id_seed};
use crate::worldgen::LandmarkKind;

/// What each settled kind sells, at base prices (jittered ±1 per landmark).
/// `markets.ts:21-29`, verbatim.
pub fn market_kind_table(kind: LandmarkKind) -> Option<&'static [(&'static str, i64)]> {
    match kind {
        LandmarkKind::City => Some(&[
            ("meal", 3), ("grain", 1), ("venison", 2), ("timber", 2),
            ("plank", 3), ("herbs", 2), ("poultice", 5),
        ]),
        LandmarkKind::Village => Some(&[("meal", 3), ("grain", 1), ("venison", 2)]),
        LandmarkKind::Mill => Some(&[("grain", 1), ("meal", 3)]),
        LandmarkKind::Port => Some(&[("meal", 3), ("venison", 2), ("timber", 2)]),
        LandmarkKind::Crossroads => Some(&[("meal", 3), ("plank", 3)]),
        LandmarkKind::Abbey => Some(&[("meal", 3), ("poultice", 5), ("herbs", 2)]),
        _ => None,
    }
}

pub const MARKET_SEASON_DAYS: i64 = 12;

pub fn market_epoch(day: i64) -> i64 {
    day.max(0) / MARKET_SEASON_DAYS
}

/// The market at a landmark (item → price), or `None` where none stands.
/// Pass the INTEL day's epoch to see what the guild believes rather than what
/// is true.
pub fn market_at(c: &Campaign, landmark_id: &str, epoch: i64) -> Option<Vec<(String, i64)>> {
    let lm = c.founding.world.landmark_by_id(landmark_id)?;
    // A place in ashes keeps no market at all.
    if c.sacked.iter().any(|s| s == landmark_id) {
        return None;
    }
    let table = market_kind_table(lm.kind)?;
    let seed = f64::from(id_seed(&lm.id));
    Some(
        table
            .iter()
            .map(|&(item, base)| {
                // Scarcity is settlement life's term (not ported) — 0 here.
                let jitter =
                    (hash(seed, f64::from(id_seed(item)) + epoch as f64 * 7.31) * 3.0).floor() as i64;
                (item.to_string(), (base + jitter - 1).max(1))
            })
            .collect(),
    )
}

/// Has anyone gone and asked? Only known markets can be bought from.
pub fn market_known(c: &Campaign, landmark_id: &str) -> bool {
    c.market_intel.iter().any(|(id, _)| id == landmark_id)
}

/// Carting cost per unit: free nearby, +1g per three days of road.
pub fn haulage_from(c: &Campaign, landmark_id: &str) -> i64 {
    let Some(lm) = c.founding.world.landmark_by_id(landmark_id) else { return 0 };
    crate::afield::travel_days_to(lm.x, lm.z) / 3
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvisionChoice {
    pub landmark_id: String,
    pub name: String,
    pub item: String,
    /// Price per unit, haulage included.
    pub unit: i64,
}

/// What the provisioner would buy from the CURRENT source: the best
/// gold-per-nutrition edible the market sells, haulage included — or `None`
/// when no source is set, the source is unknown, or it sells nothing to eat
/// (the caller falls back to the home market).
pub fn provision_choice(c: &Campaign) -> Option<ProvisionChoice> {
    let src = c.provision_source.clone()?;
    if !market_known(c, &src) {
        return None;
    }
    let lm = c.founding.world.landmark_by_id(&src)?;
    let market = market_at(c, &src, market_epoch(c.day))?;
    let surcharge = haulage_from(c, &src);
    let mut best: Option<(String, i64, f64)> = None;
    for (item, price) in market {
        let nutrition = item_nutrition(&item);
        if nutrition <= 0.0 {
            continue;
        }
        let unit = price + surcharge;
        let per = unit as f64 / nutrition;
        if best.as_ref().is_none_or(|(_, _, bp)| per < *bp) {
            best = Some((item, unit, per));
        }
    }
    best.map(|(item, unit, _)| ProvisionChoice {
        landmark_id: src,
        name: lm.name.clone(),
        item,
        unit,
    })
}

/// A market in a sentence — GENERATED from the derived numbers.
pub fn describe_market(c: &Campaign, landmark_id: &str, epoch: i64) -> String {
    let Some(market) = market_at(c, landmark_id, epoch) else {
        return "no market worth the name".to_string();
    };
    let surcharge = haulage_from(c, landmark_id);
    let body = market
        .iter()
        .map(|(item, price)| {
            let name = crate::defs::item_def(item).map_or(item.clone(), |d| d.name.to_lowercase());
            format!("{name} {price}g")
        })
        .collect::<Vec<_>>()
        .join(", ");
    if surcharge > 0 {
        format!("{body} (+{surcharge}g haulage a unit)")
    } else {
        body
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BelievedMarket {
    pub text: String,
    pub age: i64,
    /// The season has turned since anyone looked.
    pub stale: bool,
}

/// What the guild BELIEVES a known market charges — the intel day's epoch,
/// with its age stated when the word has gone stale. `None` if never seen.
pub fn believed_market(c: &Campaign, landmark_id: &str) -> Option<BelievedMarket> {
    let seen = c.market_intel.iter().find(|(id, _)| id == landmark_id).map(|(_, d)| *d)?;
    let age = (c.day - seen).max(0);
    let stale = market_epoch(seen) < market_epoch(c.day);
    let mut text = describe_market(c, landmark_id, market_epoch(seen));
    if stale {
        text.push_str(&format!(" — as of {age} days back; the season has turned since"));
    }
    Some(BelievedMarket { text, age, stale })
}

/// The nearest market landmark — optionally of one kind ("a nearby village"),
/// falling back to any settlement when that kind never rolled.
pub fn nearest_market(c: &Campaign, kind: Option<LandmarkKind>) -> Option<String> {
    let pool: Vec<&crate::worldgen::Landmark> = c
        .founding
        .world
        .landmarks
        .iter()
        .filter(|lm| market_kind_table(lm.kind).is_some())
        .collect();
    if pool.is_empty() {
        return None;
    }
    let near = |list: &[&crate::worldgen::Landmark]| -> Option<String> {
        let mut best: Option<(&str, f64)> = None;
        for lm in list {
            let d = (lm.x * lm.x + lm.z * lm.z).sqrt();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((lm.id.as_str(), d));
            }
        }
        best.map(|(id, _)| id.to_string())
    };
    if let Some(k) = kind {
        let of_kind: Vec<&crate::worldgen::Landmark> =
            pool.iter().copied().filter(|lm| lm.kind == k).collect();
        if !of_kind.is_empty() {
            return near(&of_kind);
        }
    }
    near(&pool)
}
