//! S7b in-crate tests: the campaign brain (storyteller, raid math, the dawn
//! fold skeleton, save/load). Numeric pins are DERIVED from the TS formulas
//! (webband-port-data.md §4/§7 carries citations), not guessed.

use crate::campaign::{
    dawn_fold, founding_snapshot, load_campaign, resolve_raid, save_campaign, Campaign,
    CampaignError, ColonySnapshot, MemberView, RaidResultView, SAVE_VERSION,
};
use crate::defs::{colony_wealth, food_days, js_round, BuildingView, InventorySnapshot, StackView};
use crate::director::{
    tick_director, CampaignEvent, DirectorView, ThreatState, TropeKind,
};
use crate::founding::new_founding;
use crate::raids::{
    comp_power, plunder, raid_budget, raid_tier, roll_comp, spawn_raid, PlunderSeverity, SpawnOpts,
};
use crate::rng::RngState;
use crate::scenario::ScenarioId;
use crate::worldgen::BandStatus;

fn campaign(seed: u32, scenario: ScenarioId) -> Campaign {
    Campaign::new(new_founding(seed, 0, scenario).expect("founding"))
}

/// A dummy far-off threat: blocks the warband trope without ever arriving.
fn parked_threat() -> ThreatState {
    ThreatState {
        id: "th_test".to_string(),
        name: "Test's war-band".to_string(),
        comp: vec![("looter".to_string(), 3)],
        tier: 1,
        target_landmark_id: "hall".to_string(),
        path: (0..1000).map(|i| (85.0 - i as f64 * 0.01, 0.0)).collect(),
        step_idx: 0,
        pillaged: Vec::new(),
    }
}

// ---------- Raid math pins ----------

#[test]
fn raid_budget_curve_pins() {
    // The measured formula: round(2 + colonists*2 + wealth*0.005 + day*0.25).
    // A fresh village colony: gold 90 + 4 heads * 40 + cache value
    // (meal 14*3 + timber 10*2 + plank 4*3 = 74) = 324 wealth →
    // 2 + 8 + 1.62 + 0.25 = 11.87 → 12: "a fresh colony lands near the 12
    // mark" (raids.ts doc comment, measured 75% win rate at 12).
    assert_eq!(raid_budget(324, 4, 1), 12);
    assert_eq!(raid_budget(0, 3, 0), 8);
    assert_eq!(raid_budget(2000, 6, 40), 34); // 2+12+10+10
    // The escalation clock: +0.25 budget per day, nothing else moving.
    assert_eq!(raid_budget(324, 4, 41) - raid_budget(324, 4, 1), 10);

    assert_eq!(raid_tier(324, 1), 1); // (324+12)/600 -> +0
    assert_eq!(raid_tier(1000, 20), 3); // (1000+240)/600 -> +2
    assert_eq!(raid_tier(10_000, 100), 7); // clamped
    assert_eq!(raid_tier(0, 0), 1);

    // And the wealth feeding it, over a hand-built inventory.
    let inv = InventorySnapshot {
        stacks: vec![
            StackView { id: "s0".into(), item: "meal".into(), count: 14 },
            StackView { id: "s1".into(), item: "timber".into(), count: 10 },
            StackView { id: "s2".into(), item: "plank".into(), count: 4 },
        ],
        buildings: vec![
            // A built hearth adds 2 * (timber 4 * value 2) = 16.
            BuildingView { id: "b0".into(), kind: "hearth".into(), q: 0, r: 1, built: true },
            // A blueprint adds nothing.
            BuildingView { id: "b1".into(), kind: "bed".into(), q: 0, r: 2, built: false },
        ],
    };
    assert_eq!(colony_wealth(90, 4, &inv), 324 + 16);
    assert_eq!(food_days(&inv.stacks), 14.0);
}

#[test]
fn js_round_matches_math_round() {
    assert_eq!(js_round(11.87), 12);
    assert_eq!(js_round(0.5), 1);
    assert_eq!(js_round(-0.5), 0); // JS Math.round(-0.5) === -0
    assert_eq!(js_round(-1.5), -1); // half toward +inf, unlike Rust's round()
}

#[test]
fn roll_comp_spends_the_budget() {
    for seed in [1u32, 7, 42, 999] {
        for budget in [6.0, 12.0, 13.8, 20.0, 34.0] {
            let mut g = RngState::new(seed);
            let comp = roll_comp(&mut g, budget);
            let power = comp_power(&comp);
            // Greedy spend: ends only when the remainder is under a looter.
            assert!(power <= budget, "seed {seed} budget {budget}: overspent {power}");
            assert!(power > budget - 1.0, "seed {seed} budget {budget}: left change {power}");
            for (id, n) in &comp {
                assert!(["looter", "bandit", "raider"].contains(&id.as_str()));
                assert!(*n > 0);
            }
            // Determinism: same seed, same comp.
            let mut g2 = RngState::new(seed);
            assert_eq!(roll_comp(&mut g2, budget), comp);
            assert_eq!(g.rng_counter, g2.rng_counter);
        }
    }
}

#[test]
fn spawn_raid_city_remap_keeps_stream_byte_identical() {
    // The comparability law at the raid seam: the entry-dir roll always
    // happens; only the drawn VALUE is remapped for the city start.
    for seed in 0..40u32 {
        let mut a = RngState::new(seed);
        let mut b = RngState::new(seed);
        let village = spawn_raid(&mut a, 5, 324, 4, ScenarioId::Village, SpawnOpts::default());
        let city = spawn_raid(&mut b, 5, 324, 4, ScenarioId::City, SpawnOpts::default());
        assert_eq!(a.rng_counter, b.rng_counter, "seed {seed}: stream diverged");
        assert_eq!(village.comp, city.comp);
        assert_eq!(village.elite_name, city.elite_name);
        assert_eq!(village.arrives_day, 6);
        assert!(city.entry_dir != 2 && city.entry_dir != 3, "city wall breached");
        match village.entry_dir {
            2 => assert_eq!(city.entry_dir, 1),
            3 => assert_eq!(city.entry_dir, 4),
            d => assert_eq!(city.entry_dir, d),
        }
        // A supplied elite name suppresses ALL elite draws.
        let mut c1 = RngState::new(seed);
        let named = spawn_raid(
            &mut c1,
            5,
            324,
            4,
            ScenarioId::Village,
            SpawnOpts { elite_name: Some("Foe the Test".into()), ..Default::default() },
        );
        assert_eq!(named.elite_name.as_deref(), Some("Foe the Test"));
    }
}

#[test]
fn plunder_pins() {
    // Hand-derived from raids.ts:179-228. roster 3 → spare = 21 food units.
    // tier 1 beaten → take = 18. Stacks by value: meal(3) → timber(2) →
    // berries(1); 3 stacks total.
    //   meal: nutrition 1 → floor ceil(21/1/3) = 7 → takeable 20-7 = 13.
    //   timber: nutrition 0 → floor 0 → takes the remaining 5.
    //   berries: untouched (take satisfied).
    let inv = InventorySnapshot {
        stacks: vec![
            StackView { id: "sb".into(), item: "berries".into(), count: 40 },
            StackView { id: "sm".into(), item: "meal".into(), count: 20 },
            StackView { id: "st".into(), item: "timber".into(), count: 30 },
        ],
        buildings: vec![
            BuildingView { id: "hearth".into(), kind: "hearth".into(), q: 0, r: 1, built: true },
            BuildingView { id: "far_shed".into(), kind: "store_shed".into(), q: 5, r: -4, built: true },
            BuildingView { id: "mid_bed".into(), kind: "bed".into(), q: 2, r: 1, built: true },
            BuildingView { id: "plan".into(), kind: "cot".into(), q: 9, r: 9, built: false },
        ],
    };
    let p = plunder(&inv, 3, 1, PlunderSeverity::Beaten);
    assert_eq!(p.taken, vec![("sm".to_string(), 13), ("st".to_string(), 5)]);
    assert_eq!(p.taken_total, 18);
    // Burns the 2 outermost BUILT buildings: far_shed (9), mid_bed (3);
    // blueprints never burn.
    assert_eq!(p.burnt_building_ids, vec!["far_shed".to_string(), "mid_bed".to_string()]);
    assert_eq!((p.renown_loss, p.director_points_loss, p.relief_days), (4, 20, 3));

    // Undefended doubles the take (36), burns 3, relief 4, renown -6.
    let p2 = plunder(&inv, 3, 1, PlunderSeverity::Undefended);
    assert_eq!(p2.taken_total, 36);
    assert_eq!(p2.taken, vec![("sm".to_string(), 13), ("st".to_string(), 23)]);
    assert_eq!(p2.burnt_building_ids.len(), 3);
    assert_eq!((p2.renown_loss, p2.relief_days), (6, 4));

    // The food floor holds even for a huge take: a tier-7 heavy sack
    // (take 132) leaves the floored share of every edible stack standing.
    let p3 = plunder(&inv, 3, 7, PlunderSeverity::Undefended);
    let meal_taken = p3.taken.iter().find(|(id, _)| id == "sm").map_or(0, |(_, n)| *n);
    let berry_taken = p3.taken.iter().find(|(id, _)| id == "sb").map_or(0, |(_, n)| *n);
    assert_eq!(meal_taken, 13, "meal floor breached");
    // berries: nutrition 0.5 → floor ceil(21/0.5/3) = 14 → takeable 26.
    assert_eq!(berry_taken, 26, "berry floor breached");
}

// ---------- The storyteller ----------

/// A quiet view: nothing eligible but raid/festival/windfall (Town closes
/// recruiting; a parked threat blocks warband; a camped caravan blocks
/// caravan; no feuds, no sown cells, no cause).
fn quiet_setup(seed: u32) -> (Campaign, DirectorView) {
    let mut c = campaign(seed, ScenarioId::Town);
    c.threats.push(parked_threat());
    c.caravan = Some(crate::director::Caravan {
        id: "cv".into(),
        faction_id: None,
        arrived_day: 1,
        leaves_day: i64::MAX,
        gold: 1,
        goods: Vec::new(),
        traded: false,
    });
    let view = DirectorView {
        wealth: 300,
        avg_mood: 70.0,
        ..Default::default()
    };
    (c, view)
}

#[test]
fn plan_saves_toward_expensive_tropes() {
    // Accrual per tick: 2 + ceil(roster/2) + floor(300/800) + mood bonus
    //                 = 2 + 2 + 0 + 2 = 6 (roster is 3 or 4 founders — the
    // ceil makes both 2). Cooldown (3 days from lastEventDay 0) blocks days
    // 2 [6 pts] and nothing fires before the plan commits on day 3 [12 pts].
    // Pool under the quiet view = raid(w3) / festival(w1) / windfall(w1).
    //   plan = raid (60):    fires when 6*(day-1) >= 60 → day 11 exactly.
    //   plan = festival (30): 6*(day-1) >= 30 → day 6.
    //   plan = windfall (25): 6*(day-1) >= 25 → day 6 (30 pts).
    // The saving-up law is the pin: NO event can fire on days 3-5 (cheapest
    // trope costs 25 > 12/18/24 pts), and a raid-planned run fires NOTHING
    // until day 11 despite affording festival from day 6 on.
    let mut saw_raid_first = false;
    let mut saw_cheap_first = false;
    for seed in 0..40u32 {
        let (mut c, view) = quiet_setup(seed);
        let roster_len = c.roster.len();
        assert!((3..=4).contains(&roster_len));
        let mut first: Option<(i64, TropeKind)> = None;
        for _ in 0..30 {
            c.day += 1;
            if let Some(ev) = tick_director(&mut c, &view) {
                first = Some((c.day, ev.kind));
                break;
            }
        }
        let (day, kind) = first.expect("something must fire within 30 days");
        match kind {
            TropeKind::Raid => {
                assert_eq!(day, 11, "seed {seed}: raid must wait for 60 points");
                assert_eq!(c.director.points, 0, "seed {seed}: 60 accrued, 60 spent");
                assert!(c.raid.is_some());
                saw_raid_first = true;
            }
            TropeKind::Festival | TropeKind::Windfall => {
                assert_eq!(day, 6, "seed {seed}: cheap trope fires at 30 points");
                saw_cheap_first = true;
            }
            other => panic!("seed {seed}: impossible first trope {other:?}"),
        }
    }
    // Both behaviors must exist across the seed range or the test is vacuous.
    assert!(saw_raid_first, "no seed committed to a raid first");
    assert!(saw_cheap_first, "no seed committed to a cheap trope first");
}

#[test]
fn plan_points_cap_note() {
    // 66 accrued by day 11 (6 × 10 ticks + 6) exceeds nothing: cap is 120.
    // Sanity that accrual matches the derivation used above.
    let (mut c, view) = quiet_setup(3);
    c.day += 1;
    assert!(tick_director(&mut c, &view).is_none());
    assert_eq!(c.director.points, 6, "one tick of the derived accrual");
}

#[test]
fn mercy_gate_holds() {
    // (roster <= 2 OR mood < 30) AND wealth <= 2000 suppresses raid, warband
    // and feud — but NOT festivals/windfalls/wanderers.
    let run = |roster_cut: usize, mood: f64, wealth: i64, seed: u32| -> Vec<TropeKind> {
        let mut c = campaign(seed, ScenarioId::Village);
        c.roster.truncate(roster_cut);
        let view = DirectorView { wealth, avg_mood: mood, ..Default::default() };
        let mut kinds = Vec::new();
        for _ in 0..150 {
            c.day += 1;
            if let Some(ev) = tick_director(&mut c, &view) {
                kinds.push(ev.kind);
                // Resolve any raid instantly so "one storm at a time" never
                // stalls the run (mercied runs spawn none anyway).
                c.raid = None;
                c.threats.clear();
            }
        }
        kinds
    };

    // Two hands, poor: no violence, ever.
    let kinds = run(2, 80.0, 500, 11);
    assert!(!kinds.is_empty(), "mercy must not silence the whole storyteller");
    assert!(
        !kinds.iter().any(|k| matches!(k, TropeKind::Raid | TropeKind::Warband | TropeKind::Feud)),
        "mercied colony was raided: {kinds:?}"
    );

    // Broken spirits, poor: same shelter.
    let kinds = run(4, 20.0, 500, 11);
    assert!(
        !kinds.iter().any(|k| matches!(k, TropeKind::Raid | TropeKind::Warband | TropeKind::Feud)),
        "low-mood colony was raided: {kinds:?}"
    );

    // Two hands but RICH (wealth > 2000): worth the trouble anyway.
    let kinds = run(2, 80.0, 2500, 11);
    assert!(
        kinds.iter().any(|k| matches!(k, TropeKind::Raid | TropeKind::Warband)),
        "rich colony was never raided: {kinds:?}"
    );
}

#[test]
fn storyteller_determinism_100_days() {
    let run = |seed: u32| -> (Vec<(i64, String)>, Campaign) {
        let mut c = campaign(seed, ScenarioId::Village);
        let snap = founding_snapshot(&c, 70.0);
        let mut log = Vec::new();
        for _ in 0..100 {
            // Fight any raid the morning it arrives (no rng in resolution;
            // alternate outcomes by day parity for branch coverage).
            if let Some(raid) = c.raid.clone() {
                if raid.arrives_day <= c.day {
                    let result = RaidResultView {
                        victory: c.day % 2 == 0,
                        gold_looted: 30,
                        member_hp: c.roster.iter().map(|id| (id.clone(), 0.8)).collect(),
                    };
                    let out = resolve_raid(&mut c, &raid, &result, &snap.inventory);
                    log.push((c.day, format!("resolve v={} plunder={}", result.victory, out.plunder.is_some())));
                }
            }
            let out = dawn_fold(&mut c, &snap, &[]);
            if let Some(ev) = &out.event {
                log.push((c.day, format!("{:?}: {}", ev.kind, ev.text)));
            }
            for a in &out.arrivals {
                log.push((c.day, format!("arrival: {}", a.title)));
            }
            assert!(!out.fell, "nobody starves in this snapshot");
        }
        (log, c)
    };
    for seed in [5u32, 77, 20_260_721] {
        let (log_a, c_a) = run(seed);
        let (log_b, c_b) = run(seed);
        assert_eq!(log_a, log_b, "seed {seed}: event log diverged");
        assert_eq!(c_a, c_b, "seed {seed}: final campaign state diverged");
        assert!(!log_a.is_empty(), "seed {seed}: a 100-day run must produce events");
        // The soak must contain at least one organic raid (the plan
        // mechanic's whole point) — the S6 acceptance bar at host scale.
        // A warband arrival counts: it converts to a raid at the fences.
        assert!(
            log_a.iter().any(|(_, t)| t.starts_with("Raid") || t.starts_with("arrival:")),
            "seed {seed}: 100 days with no raid: {log_a:?}"
        );
    }
}

#[test]
fn forced_trope_payload_shapes() {
    // Force-commit plans with a full purse to pin each payload's shape.
    let force = |seed: u32, plan: TropeKind| -> (Campaign, Option<crate::director::DirectorEvent>) {
        let mut c = campaign(seed, ScenarioId::Village);
        c.day = 10;
        c.director.plan = Some(plan);
        c.director.points = 120;
        let view = DirectorView {
            wealth: 300,
            avg_mood: 70.0,
            sown_cells: (0..10).map(|i| format!("c{i}")).collect(),
            ..Default::default()
        };
        let ev = tick_director(&mut c, &view);
        (c, ev)
    };

    // Caravan: 3 distinct wares within table ranges, purse 60..=140, 2 days.
    let (c, ev) = force(9, TropeKind::Caravan);
    let ev = ev.expect("caravan fires");
    match &ev.payload {
        CampaignEvent::CaravanArrives { caravan } => {
            assert_eq!(caravan.goods.len(), 3);
            let mut items: Vec<&str> = caravan.goods.iter().map(|(i, _)| i.as_str()).collect();
            items.sort_unstable();
            items.dedup();
            assert_eq!(items.len(), 3, "wares drawn without replacement");
            for (item, n) in &caravan.goods {
                let (lo, hi) = match item.as_str() {
                    "grain" => (18, 30),
                    "timber" => (10, 20),
                    "meal" => (6, 10),
                    "herbs" => (5, 10),
                    "poultice" => (2, 4),
                    "plank" => (8, 14),
                    other => panic!("unknown ware {other}"),
                };
                assert!((lo..=hi).contains(n), "{item} x{n} out of range");
            }
            assert!((60..=140).contains(&caravan.gold));
            assert_eq!(caravan.leaves_day, 12);
            assert!(!caravan.traded);
            assert_eq!(c.caravan.as_ref(), Some(caravan));
        }
        other => panic!("wrong payload {other:?}"),
    }

    // Windfall: the exact bundles and the purse, gold applied campaign-side.
    let (c, ev) = force(9, TropeKind::Windfall);
    match &ev.expect("windfall fires").payload {
        CampaignEvent::Windfall { drops, gold } => {
            assert_eq!(*gold, 40);
            assert_eq!(c.gold, 90 + 40); // village purse + the windfall
            let d: Vec<(&str, u32, i32, i32)> =
                drops.iter().map(|d| (d.item.as_str(), d.count, d.q, d.r)).collect();
            assert_eq!(d, vec![("meal", 6, 10, -10), ("timber", 8, 11, -10)]);
        }
        other => panic!("wrong payload {other:?}"),
    }

    // Blight: 40% toll of 10 sown cells = 4, distinct, from the pool.
    let (_, ev) = force(9, TropeKind::Blight);
    match &ev.expect("blight fires").payload {
        CampaignEvent::Blight { killed_cells } => {
            assert_eq!(killed_cells.len(), 4);
            let mut k = killed_cells.clone();
            k.sort();
            k.dedup();
            assert_eq!(k.len(), 4, "cells killed without replacement");
            for cell in killed_cells {
                assert!(cell.starts_with('c'));
            }
        }
        other => panic!("wrong payload {other:?}"),
    }

    // Warband: threat at the rim (~85 units out), rough report filed.
    let (c, ev) = force(9, TropeKind::Warband);
    match &ev.expect("warband fires").payload {
        CampaignEvent::WarbandGathers { threat_id, reported_power } => {
            assert_eq!(c.threats.len(), 1);
            let t = &c.threats[0];
            assert_eq!(&t.id, threat_id);
            let (x, z) = t.path[0];
            assert!(((x * x + z * z).sqrt() - 85.0).abs() < 1e-9);
            assert_eq!(*t.path.last().expect("path"), (0.0, 0.0));
            assert!(t.path.len() >= 3, "the march must take days, not a jump");
            assert!(*reported_power >= 2);
            assert!(c.threat_intel.iter().any(|(id, r)| id == threat_id && !r.exact));
            // The drums mis-count: truth 20+300*0.02+10*0.3 = 29 power spent
            // greedily, report is hash-scaled 0.75-1.35 of the real power.
            let truth = comp_power(&t.comp);
            let ratio = *reported_power as f64 / truth;
            assert!((0.7..=1.4).contains(&ratio), "report {reported_power} vs truth {truth}");
        }
        other => panic!("wrong payload {other:?}"),
    }

    // Wanderer: a freelancer guest with a 2-day clock.
    let (c, ev) = force(9, TropeKind::Wanderer);
    match &ev.expect("wanderer fires").payload {
        CampaignEvent::WandererArrives { id, leaves_day } => {
            assert_eq!(*leaves_day, 12);
            let def = c.founding.cast.companions.iter().find(|x| &x.id == id).expect("cast");
            assert!(def.band.is_none(), "guests are freelancers");
            assert_eq!(c.director.guest.as_ref().map(|g| g.id.as_str()), Some(id.as_str()));
        }
        other => panic!("wrong payload {other:?}"),
    }

    // Refugee band: a camped band gets a 3-day desperation window.
    let (c, ev) = force(9, TropeKind::RefugeeBand);
    match &ev.expect("refugee fires").payload {
        CampaignEvent::RefugeeBand { band_id, desperate_until } => {
            assert_eq!(*desperate_until, 13);
            let (_, live) = c.band_states.iter().find(|(id, _)| id == band_id).expect("band");
            assert_eq!(live.state.status, BandStatus::Camped);
            assert_eq!(live.desperate_until, Some(13));
        }
        other => panic!("wrong payload {other:?}"),
    }
}

#[test]
fn warband_marches_and_converts_to_a_raid() {
    let mut c = campaign(21, ScenarioId::Village);
    c.day = 10;
    c.director.plan = Some(TropeKind::Warband);
    c.director.points = 120;
    let view = DirectorView { wealth: 300, avg_mood: 70.0, ..Default::default() };
    tick_director(&mut c, &view).expect("warband fires");
    // Drain the leftover purse so no OTHER trope fires while the band
    // marches (accrual ~6/day stays under every cost before arrival).
    c.director.points = 0;
    let threat_id = c.threats[0].id.clone();
    let path_len = c.threats[0].path.len();

    // Walk it in via dawn folds until it stands at the gates.
    let snap = founding_snapshot(&c, 70.0);
    let mut arrived_day = None;
    for _ in 0..12 {
        let out = dawn_fold(&mut c, &snap, &[]);
        if !out.arrivals.is_empty() {
            arrived_day = Some(c.day);
            break;
        }
    }
    let arrived = arrived_day.expect("the warband must arrive");
    // ~2 steps/day over a ~9-segment path ≈ 4-5 days of visible warning.
    let expected_days = (path_len as i64 - 1).div_euclid(2) + i64::from((path_len - 1) % 2 != 0);
    assert_eq!(arrived - 10, expected_days, "march pacing");
    let raid = c.raid.clone().expect("arrival converts");
    assert_eq!(raid.threat_ref.as_deref(), Some(threat_id.as_str()));
    assert!(raid.elite_name.as_deref().is_some_and(|n| n.ends_with("the Grim")));
    // tierBump 1 over the base tier.
    assert_eq!(raid.tier, (raid_tier(view.wealth, arrived) + 1).min(7));

    // Engaged: the raid's reference holds the threat at the gates, and
    // victory breaks it.
    let step = c.threats[0].step_idx;
    dawn_fold(&mut c, &snap, &[]);
    assert_eq!(c.threats[0].step_idx, step, "an engaged warband holds its ground");
    let result = RaidResultView {
        victory: true,
        gold_looted: 25,
        member_hp: c.roster.iter().map(|id| (id.clone(), 1.0)).collect(),
    };
    let raid = c.raid.clone().expect("still staged");
    let gold_before = c.gold; // dawn folds moved coin (trade, provisioner)
    resolve_raid(&mut c, &raid, &result, &snap.inventory);
    assert!(c.threats.is_empty(), "victory breaks the threat");
    assert_eq!(c.director.raids_won, 1);
    assert_eq!(c.gold, gold_before + 25);
    assert_eq!(c.renown, crate::raids::victory_renown(raid.tier));
}

// ---------- The dawn fold ----------

#[test]
fn provisioner_buys_the_shortfall() {
    // Village: meal price 3, default provisioning 4 days/mouth.
    let mut c = campaign(31, ScenarioId::Village);
    let n = c.roster.len() as i64;
    let snap = ColonySnapshot {
        inventory: InventorySnapshot {
            stacks: vec![StackView { id: "s0".into(), item: "meal".into(), count: 2 }],
            buildings: Vec::new(),
        },
        members: c
            .roster
            .iter()
            .map(|id| MemberView { id: id.clone(), mood: 70.0, starving_days: 0, ready: true })
            .collect(),
        ..Default::default()
    };
    let out = dawn_fold(&mut c, &snap, &[]);
    let p = out.provision.expect("a market means bread");
    // Target 4*n food-days, larder holds 2 → buy 4n-2 (n=3 → 10, n=4 → 14),
    // well under the 30/day cap and the purse.
    assert_eq!(i64::from(p.count), 4 * n - 2);
    assert_eq!(p.gold_spent, (4 * n - 2) * 3);
    assert_eq!(p.item, "meal");
    // Purse: 90 − the bread + the village's 2/day trade.
    assert_eq!(c.gold, 90 - p.gold_spent + 2);
    assert_eq!(out.trade_income, 2);

    // The wilderness has no market: no provisioner, no trade.
    let mut w = campaign(31, ScenarioId::Wilderness);
    let snap_w = founding_snapshot(&w, 70.0);
    let out = dawn_fold(&mut w, &snap_w, &[]);
    assert!(out.provision.is_none());
    assert_eq!(out.trade_income, 0);

    // The purse caps the buying: 4 gold buys one meal at 3.
    let mut poor = campaign(31, ScenarioId::Village);
    poor.gold = 4;
    let snap_p = ColonySnapshot {
        inventory: InventorySnapshot::default(),
        members: snap.members.clone(),
        ..Default::default()
    };
    let out = dawn_fold(&mut poor, &snap_p, &[]);
    assert_eq!(out.provision.expect("one meal").count, 1);
    assert_eq!(poor.gold, 4 - 3 + 2);
}

#[test]
fn city_rent_bleeds_standing_when_unpaid() {
    let mut c = campaign(13, ScenarioId::City);
    let snap = founding_snapshot(&c, 70.0);
    let gold_before = c.gold;
    let out = dawn_fold(&mut c, &snap, &[]);
    // Paid silently: 8/day out, provisioner may also have bought bread.
    assert!(!out.rent_unpaid);
    let spent = out.provision.as_ref().map_or(0, |p| p.gold_spent);
    assert_eq!(c.gold, gold_before - spent - 8);
    let standing_before = c.standing;

    // A dry purse: standing -3, chronicled, never an eviction.
    c.gold = 0;
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.rent_unpaid);
    assert_eq!(c.standing, standing_before - 3);
    assert!(c
        .chronicle
        .iter()
        .any(|e| e.headline.as_deref() == Some("Rent unpaid")));
}

#[test]
fn starvation_exodus_thresholds() {
    // Non-founders walk at 3 hungry days; founders hold to 6; a signed
    // band walks TOGETHER.
    let mut c = campaign(17, ScenarioId::Village);
    let founders: Vec<String> = c.roster.clone();
    // Roster a freelancer and a full non-founder band, signed.
    let freelancer = c
        .founding
        .cast
        .companions
        .iter()
        .find(|x| x.band.is_none())
        .expect("freelancers exist")
        .id
        .clone();
    let band = c
        .founding
        .cast
        .bands
        .iter()
        .find(|b| !b.founders)
        .expect("non-founder band")
        .clone();
    let band_members: Vec<String> = c
        .founding
        .cast
        .companions
        .iter()
        .filter(|x| x.band.as_deref() == Some(band.id.as_str()))
        .map(|x| x.id.clone())
        .collect();
    c.roster.push(freelancer.clone());
    c.roster.extend(band_members.clone());
    if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| *id == band.id) {
        live.state.status = BandStatus::Signed;
    }

    let starving = |c: &Campaign, days: &dyn Fn(&str) -> i64| -> ColonySnapshot {
        ColonySnapshot {
            members: c
                .roster
                .iter()
                .map(|id| MemberView {
                    id: id.clone(),
                    mood: 50.0,
                    starving_days: days(id),
                    ready: true,
                })
                .collect(),
            ..Default::default()
        }
    };

    // Day 1 of hunger at bar 2: nobody walks.
    let snap = starving(&c, &|_| 2);
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.departed.is_empty());

    // At 3: the freelancer AND the signed band walk (together); founders
    // hold. ONE band member at the bar takes the whole band out.
    let fl = freelancer.clone();
    let first_band_member = band_members[0].clone();
    let snap = starving(&c, &|id| if id == fl || id == first_band_member { 3 } else { 0 });
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.departed.contains(&freelancer));
    for m in &band_members {
        assert!(out.departed.contains(m), "{m} should ride out with the band");
        assert!(!c.roster.contains(m));
    }
    let (_, live) = c.band_states.iter().find(|(id, _)| *id == band.id).expect("band");
    assert_eq!(live.state.status, BandStatus::Camped);
    assert_eq!(live.state.times_departed, 1);
    assert_eq!(live.state.patience, 50);
    for f in &founders {
        assert!(c.roster.contains(f), "founders hold at 3 hungry days");
    }

    // Founders at 5: hold. At 6: walk — and the roster empties: the fall.
    let snap = starving(&c, &|_| 5);
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.departed.is_empty() && !out.fell);
    let snap = starving(&c, &|_| 6);
    let out = dawn_fold(&mut c, &snap, &[]);
    assert_eq!(out.departed.len(), founders.len());
    assert!(out.fell, "an empty roster is the colony's end");
}

#[test]
fn caravan_departure_sweep() {
    let mut c = campaign(23, ScenarioId::Village);
    c.caravan = Some(crate::director::Caravan {
        id: "cv1".into(),
        faction_id: None,
        arrived_day: 1,
        leaves_day: 2,
        gold: 80,
        goods: vec![("grain".into(), 20)],
        traded: true,
    });
    let standing = c.standing;
    let snap = founding_snapshot(&c, 70.0);
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.caravan_departed);
    assert!(c.caravan.is_none());
    assert_eq!(c.standing, standing + 4, "business done is remembered kindly");
    assert!(c
        .chronicle
        .iter()
        .any(|e| e.headline.as_deref() == Some("The caravan moves on")));
}

// ---------- Save / load ----------

#[test]
fn campaign_save_file_roundtrip() {
    // 12 folds: even a raid-committed plan (60 points at ~6/day) fires by
    // day 11, so the chronicle is provably non-empty for every seed.
    let mut c = campaign(4242, ScenarioId::City);
    let snap = founding_snapshot(&c, 70.0);
    for _ in 0..12 {
        if let Some(raid) = c.raid.clone() {
            if raid.arrives_day <= c.day {
                let result = RaidResultView {
                    victory: false,
                    gold_looted: 0,
                    member_hp: c.roster.iter().map(|id| (id.clone(), 0.3)).collect(),
                };
                resolve_raid(&mut c, &raid, &result, &snap.inventory);
            }
        }
        dawn_fold(&mut c, &snap, &[]);
    }
    assert!(!c.chronicle.is_empty(), "twelve days must leave a written trace");

    let path = std::env::temp_dir().join(format!("webband_s7b_save_{}.json", std::process::id()));
    save_campaign(&c, &path).expect("save");
    let back = load_campaign(&path).expect("load");
    assert_eq!(c, back, "save file round-trip must be exact");
    // The resumed stream continues identically on both sides.
    let mut a = c.rng;
    let mut b = back.rng;
    assert_eq!(crate::rng::rng_float(&mut a), crate::rng::rng_float(&mut b));

    // Version discard rule: a foreign version is an explicit error.
    let mut v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("read"),
    )
    .expect("json");
    v["version"] = serde_json::Value::from(SAVE_VERSION + 1);
    std::fs::write(&path, serde_json::to_string(&v).expect("ser")).expect("write");
    match load_campaign(&path) {
        Err(CampaignError::Version { found, want }) => {
            assert_eq!(found, SAVE_VERSION + 1);
            assert_eq!(want, SAVE_VERSION);
        }
        other => panic!("expected version error, got {other:?}"),
    }
    let _ = std::fs::remove_file(&path);
}
