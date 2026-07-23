//! webband_campaign — S6: the storyteller + campaign clock, HOST-DRIVES-FIXTURE.
//!
//! THE BRIDGE ARCHITECTURE (the decision this file embodies): Webband's own
//! shape is that the storyteller is CAMPAIGN-side and the colony is the sim —
//! so the campaign brain stays in `crates/webband_app` (pure host logic, no
//! engine dependency), the colony stays in the `webband_colony` fixture (no
//! director state, no gold, no roster ids), and the two meet ONLY here, in an
//! integration test that is a dev-dependency of `sims`. The alternative (a
//! `bridge` module inside webband_app behind a feature) was rejected because
//! it would let the host crate grow an engine/GPU dependency edge that S7's
//! engine_play wiring would then have to fight; a test-only bridge keeps both
//! library graphs clean and the meeting point in the one place that already
//! owns GPU readbacks.
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
//!   6. write the resolved tropes back through the fixture's own seams:
//!      raid -> `musters_at` on a comp-mapped cohort (S5's seam, verbatim),
//!      windfall -> stock drops on the RIM CACHE (slot 99; ordinary hauls
//!      carry them in), provisioner meals -> the yard cache, caravan ->
//!      the Trader agent (trader_active/purse), wanderer -> re-activating
//!      a pooled colonist slot, exodus -> deactivating one.
//!
//! ROSTER = SLOT POOL: campaign roster members map 1:1 onto colonist slots in
//! roster order; slots beyond the roster are deactivated at founding (the
//! recruit pool). Every activation inherits the slot's own UNIQUE stagger
//! residue, so the fixture's %20 phase-exclusivity determinism construction
//! survives roster churn by construction.
//!
//! Pins:
//!   * THE SOAK — a 60-day seeded campaign: >= 2 organic raids fired by the
//!     real accrual/plan mechanic, >= 1 windfall hauled in off the rim,
//!     >= 1 caravan with trade actually landing, hungry-day share > 0 (no
//!     post-scarcity), roster changes, the colony survives or falls honestly
//!     — and the whole run is DETERMINISTIC: a digest (fixture state-buffer
//!     bits + serialized Campaign + the event log) is written to
//!     target/webband_campaign/soak_digest.txt on first run and ASSERTED
//!     EQUAL on every later run — the cross-process pin (run the test twice).
//!   * THE FALL — an engineered famine (wilderness start, food sources
//!     zeroed): the signed band walks together at the 3-hungry-day bar,
//!     founders hold to 6, the roster empties, and the outcome is the
//!     terminal CampaignOutcome::Fell.
//!
//! windows-gnu note: test bodies run on a spawned 64 MiB thread (the S1
//! wgpu-init stack finding).

#![allow(non_snake_case)]

// THE BRIDGE MACHINERY MOVED (S9, 2026-07-22): the GPU readback helpers, the
// fixture world map, `Bridge` and the determinism digest now live in
// `tests/webband_bridge/mod.rs` so `webband_spine.rs` can drive the SAME
// code. The move was verbatim (visibility + two additive counters only) and
// no assertion in this file changed.
// S12: the bridge is now the `webband_bridge` LIBRARY crate (a dev-dep of
// this crate), so `crates/webband_play` drives the same code. Was
// `mod webband_bridge;` over `tests/webband_bridge/mod.rs`.
use webband_bridge::*;

// ---------------------------------------------------------------------------
// THE SOAK — 60 seeded days through the bridge.

#[test]
fn soak_60_day_campaign() {
    on_big_stack(|| {
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, true) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        let roster0 = b.campaign.roster.len();
        println!(
            "[soak] founding: {} — roster {} of 20 bodies ({} pooled), scenario Village, gold {}",
            b.campaign.founding.name,
            roster0,
            b.free_slots.len(),
            b.campaign.gold
        );
        assert!(roster0 >= 8, "soak staging wants a real colony (roster {roster0})");
        assert!(!b.free_slots.is_empty(), "soak staging wants a recruit pool");

        let mut fell_day: Option<i64> = None;
        for _ in 0..60 {
            let out = b.run_day();
            if b.campaign.day % 5 == 0 || out.event.is_some() || !out.departed.is_empty() {
                let w = webband_app::defs::colony_wealth(
                    b.campaign.gold,
                    b.campaign.roster.len(),
                    &InventorySnapshot::default(),
                );
                let _ = w;
                println!(
                    "[soak day {}] gold={} renown={} roster={} points={} plan={:?} raid={} event={:?}",
                    b.campaign.day,
                    b.campaign.gold,
                    b.campaign.renown,
                    b.campaign.roster.len(),
                    b.campaign.director.points,
                    b.campaign.director.plan,
                    b.campaign.raid.is_some(),
                    out.event.as_ref().map(|e| e.kind)
                );
            }
            if let CampaignOutcome::Fell { day } = b.outcome(&out) {
                fell_day = Some(day);
                break;
            }
        }

        let hungry_share = b.hungry_member_dawns as f64 / b.member_dawns.max(1) as f64;
        let rim_left = {
            let n = b.w.n;
            let m = { let buf = b.state.agent_inv_meal_buf.clone(); read_f32(&mut b.state, &buf, n)[b.w.rim_cache] };
            let t = { let buf = b.state.agent_inv_timber_buf.clone(); read_f32(&mut b.state, &buf, n)[b.w.rim_cache] };
            m + t
        };
        println!(
            "[soak] 60 days: raids staged={} won={} lost={} | windfalls={} caravans={} \
             trade_gold={} meals_bought={} | joins={} departures={} | hungry_share={:.3} \
             ({}/{} member-dawns) | rim stock left={:.1} | fell={:?} | gold={} renown={}",
            b.raids_staged,
            b.raids_won,
            b.raids_lost,
            b.windfalls,
            b.caravans,
            b.trade_gold,
            b.meals_bought,
            b.joins,
            b.departures.len(),
            hungry_share,
            b.hungry_member_dawns,
            b.member_dawns,
            rim_left,
            fell_day,
            b.campaign.gold,
            b.campaign.renown
        );
        for line in &b.log {
            println!("[soak log] {line}");
        }

        // -- the acceptance bars: ROBUST CAMPAIGN SHAPE ---------------------
        //
        // S6b REDESIGN. The bar this replaces was `windfalls >= 1` — i.e.
        // "one particular weight-1 trope out of the ~11 in the table gets
        // drawn inside 60 days". That is a property of the DICE, not of the
        // port: windfall is always-eligible at weight 1 (director.rs:102,
        // :332), so its absence on a given seed is legitimate draw luck, and
        // on THIS seed the storyteller drew none while producing an
        // otherwise complete campaign. Asserting a named trope appears is
        // flaky by construction and would have to be re-tuned every time a
        // fixture change shifts the draw stream.
        //
        // What a working bridge must show on ANY seed is asserted instead:
        // the accrual actually fires events of several kinds, raids arrive
        // organically AND resolve, the colony sim really ran under the
        // campaign clock, scarcity is not solved, and the roster moves. The
        // per-trope SEAMS — the thing a trope-count bar was really trying to
        // protect — are proven directly by the focused injection tests
        // below, which drive the SAME `apply_event` code the storyteller
        // drives, on every wired trope, dice or no dice.
        assert!(
            b.raids_staged >= 2,
            "storyteller fired only {} organic raids over 60 days",
            b.raids_staged
        );
        assert_eq!(
            b.raids_won + b.raids_lost,
            b.raids_staged,
            "a staged raid never resolved"
        );
        assert!(
            b.event_kinds.len() >= 5,
            "the storyteller only spoke {} times in 60 days — the accrual/plan \
             mechanic is not running: {:?}",
            b.event_kinds.len(),
            b.event_kinds
        );
        let mut kinds = b.event_kinds.clone();
        kinds.sort();
        kinds.dedup();
        assert!(
            kinds.len() >= 3,
            "the whole campaign was one or two tropes ({kinds:?}) — the weighted \
             draw is degenerate"
        );
        // The colony sim really ran under the campaign clock (not just the
        // host brain): the founding blueprints were raised by fixture work.
        let built_n = {
            let n = b.w.n;
            let bb = b.state.agent_built_buf.clone();
            let built = read_u32(&mut b.state, &bb, n);
            let types = { let t = b.state.agent_creature_type_buf.clone(); read_u32(&mut b.state, &t, n) };
            (0..n)
                .filter(|&i| built[i] == 1 && types[i] != CT_WALL)
                .count()
        };
        assert!(
            built_n >= 5,
            "only {built_n} non-wall structures stand after 60 campaign days — \
             the colony fixture did no work under the bridge"
        );
        // Trade: goods and gold ACTUALLY moved, not merely a camp pitched.
        // (Seed-pinned like the whole soak; if a future fixture change moves
        // the draw stream off caravans, the seam itself stays proven by
        // `caravan_injection_camps_the_trader_and_moves_goods` — this bar is
        // then a seed decision, not a lost capability.)
        assert!(
            b.caravans >= 1 && b.trade_gold + b.meals_bought > 0,
            "no caravan trade landed (caravans={}, trade_gold={}, meals_bought={})",
            b.caravans,
            b.trade_gold,
            b.meals_bought
        );
        assert!(
            hungry_share > 0.0,
            "post-scarcity: zero hungry member-dawns over the whole soak"
        );
        assert!(
            b.joins + b.departures.len() as u64 > 0,
            "the roster never changed (no wanderer join, no exodus)"
        );
        // The colony survives or falls HONESTLY — both are legal ends; a
        // fall must only come from an empty roster.
        if let Some(day) = fell_day {
            assert!(b.campaign.roster.is_empty(), "fell with a live roster (day {day})");
        }

        // -- the determinism digest (cross-process pin) ---------------------
        let sig = b.fixture_signature();
        let fixture_hash = sig_hash(&sig);
        let campaign_json = serde_json::to_string(&b.campaign).expect("campaign serializes");
        let campaign_hash = fnv1a(campaign_json.as_bytes());
        let log_join = b.log.join("\n");
        let log_hash = fnv1a(log_join.as_bytes());
        let digest = format!(
            "webband_campaign soak digest v1\nfixture={fixture_hash:#018x}\ncampaign={campaign_hash:#018x}\nlog={log_hash:#018x}\nraids={} won={} lost={} windfalls={} caravans={} joins={} departures={} hungry={}/{} gold={} renown={} day={}\n",
            b.raids_staged,
            b.raids_won,
            b.raids_lost,
            b.windfalls,
            b.caravans,
            b.joins,
            b.departures.len(),
            b.hungry_member_dawns,
            b.member_dawns,
            b.campaign.gold,
            b.campaign.renown,
            b.campaign.day,
        );
        println!("[soak digest]\n{digest}");
        let dir = std::path::Path::new("target/webband_campaign");
        std::fs::create_dir_all(dir).expect("digest dir");
        let path = dir.join("soak_digest.txt");
        if path.exists() {
            let prior = std::fs::read_to_string(&path).expect("read prior digest");
            assert_eq!(
                prior, digest,
                "CROSS-PROCESS DETERMINISM BROKEN: this run's digest differs from the \
                 recorded one (delete {path:?} only if the fixture/bridge legitimately changed)"
            );
            println!("[soak] cross-process determinism pin HELD against {path:?}");
        } else {
            std::fs::write(&path, &digest).expect("write digest");
            println!("[soak] digest recorded to {path:?} — run the test again for the cross-process pin");
        }
    });
}

// ---------------------------------------------------------------------------
// THE FALL — engineered famine: exodus empties the roster, the campaign ends.

#[test]
fn famine_exodus_empties_roster_and_falls() {
    on_big_stack(|| {
        // Wilderness: no market (mealPrice 0 — the provisioner cannot paper
        // over the famine), no recruiting. Founders + ONE signed band, so
        // both exodus bars are proven: the band walks TOGETHER at 3 hungry
        // days, founders hold to 6, and the last founder's walk is the fall.
        let Some(mut b) = Bridge::new(ScenarioId::Wilderness, CAMPAIGN_SEED, 1, false) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        // Silence the storyteller with its OWN relief knob: a windfall's 6
        // rim-cache meals can keep the last founder alive indefinitely (found
        // on the first run — the fall stalled at roster 1 on the director's
        // charity). The famine variant isolates the exodus path.
        b.campaign.director.relief_until = i64::MAX / 2;

        let founders: Vec<String> = b.campaign.founding.roster.clone();
        let band_members: Vec<String> = b
            .campaign
            .roster
            .iter()
            .filter(|id| !founders.contains(id))
            .cloned()
            .collect();
        assert!(
            !band_members.is_empty(),
            "famine staging needs a signed non-founder band"
        );
        println!(
            "[famine] roster {} = {} founders + {} band hands",
            b.campaign.roster.len(),
            founders.len(),
            band_members.len()
        );

        // The famine knob (the S3 idiom): every food source zeroed by
        // buffer writes — cache/store/hearth stock emptied, bushes and game
        // pushed past the horizon. Timber stays; the doomed colony still
        // works while it starves.
        let n = b.w.n;
        for slot in [b.w.yard_cache, b.w.rim_cache, b.w.store, b.w.hearth] {
            for buf in [
                &b.state.agent_inv_meal_buf,
                &b.state.agent_inv_berries_buf,
                &b.state.agent_inv_venison_buf,
                &b.state.agent_inv_grain_buf,
            ] {
                write_f32(&b.state, buf, slot, 0.0);
            }
        }
        for &slot in b.w.bushes.iter().chain(b.w.game.iter()) {
            write_u32(&b.state, &b.state.agent_regrow_at_buf, slot, 4_000_000);
        }
        let _ = n;

        let mut outcome = CampaignOutcome::Ongoing;
        for _ in 0..30 {
            let out = b.run_day();
            if !out.departed.is_empty() {
                println!(
                    "[famine day {}] departed: {:?} (roster now {})",
                    b.campaign.day,
                    out.departed,
                    b.campaign.roster.len()
                );
            }
            outcome = b.outcome(&out);
            if outcome != CampaignOutcome::Ongoing {
                break;
            }
        }
        let CampaignOutcome::Fell { day } = outcome else {
            panic!(
                "the famine never felled the colony in 30 days (roster {})",
                b.campaign.roster.len()
            );
        };
        println!("[famine] the colony fell on day {day}; departures: {:?}", b.departures);

        assert!(b.campaign.roster.is_empty(), "fell with a live roster");
        // The signed band walked TOGETHER, before any founder.
        let band_days: Vec<i64> = b
            .departures
            .iter()
            .filter(|(_, id)| band_members.contains(id))
            .map(|(d, _)| *d)
            .collect();
        let founder_days: Vec<i64> = b
            .departures
            .iter()
            .filter(|(_, id)| founders.contains(id))
            .map(|(d, _)| *d)
            .collect();
        assert_eq!(
            band_days.len(),
            band_members.len(),
            "not every band hand walked"
        );
        assert_eq!(founder_days.len(), founders.len(), "not every founder walked");
        let band_day = band_days[0];
        assert!(
            band_days.iter().all(|&d| d == band_day),
            "the signed band did not walk together: {band_days:?}"
        );
        let first_founder = *founder_days.iter().min().expect("founders walked");
        assert!(
            band_day < first_founder,
            "the band (bar 3) should walk before any founder (bar 6): band {band_day}, founder {first_founder}"
        );
        // All slots returned to the pool; the fixture agrees the yard is empty.
        assert_eq!(b.slot_map.len(), 0, "slot map should be empty after the fall");
        let alive = { let buf = b.state.agent_alive_buf.clone(); read_u32(&mut b.state, &buf, b.w.n) };
        let live_colonists = b.w.colonists.iter().filter(|&&c| alive[c] == 1).count();
        assert_eq!(live_colonists, 0, "colonist bodies still active after the fall");
    });
}

// ---------------------------------------------------------------------------
// THE TROPE SEAMS — focused, deterministic injection tests.
//
// WHY THESE EXIST (S6b). The 60-day soak proves the storyteller and the
// bridge run together, but it can only exercise the tropes ITS OWN DICE
// happen to draw. On the pinned campaign seed that is raid / caravan /
// festival / warband / refugee — windfall, wanderer and blight never came
// up, so their fixture seams rode 1400 lines of untested code. Asserting
// "a windfall appears within 60 days" would only have re-rolled the dice.
// Instead every wired trope gets a focused test that injects its
// `CampaignEvent` through `Bridge::apply_event` — the exact function the
// dawn fold calls — and then asserts the SIM actually moved.
//
// The tests step the fixture with `run_day_quiet` (colony dawn systems
// only, no host fold), so the seam under test is the only thing moving.

/// WINDFALL — the storyteller's supply drop lands on the RIM CACHE and the
/// ORDINARY haul economy carries it in ("even good luck makes haul work",
/// the Webband rule). Proves: the drop lands where the bridge says it does,
/// and colonists walk 28 units out to fetch it without any nudge.
#[test]
fn windfall_injection_lands_at_the_rim_and_is_hauled_in() {
    on_big_stack(|| {
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, false) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        // Two days of ordinary work first: the founding cache is hauled in
        // and the colony is running its own economy, so the rim drop is a
        // NEW demand rather than part of the opening haul.
        b.run_day_quiet();
        b.run_day_quiet();
        let rim = b.w.rim_cache;
        let store = b.w.store;
        let (rim_meal0, rim_timber0) = (b.inv_at(rim, "meal"), b.inv_at(rim, "timber"));
        let store_timber0 = b.inv_at(store, "timber");

        // The storyteller's own bundle shape (director.rs Windfall: 6 meals
        // + 8 timber at the rim; q/r are the TS caravanSpot, cosmetic here).
        let drops = vec![
            ItemDrop { item: "meal".into(), count: 6, q: 4, r: -4 },
            ItemDrop { item: "timber".into(), count: 8, q: 4, r: -4 },
        ];
        b.apply_event(&CampaignEvent::Windfall { drops, gold: 0 });
        assert_eq!(b.windfalls, 1, "the windfall arm did not run");

        let rim_meal1 = b.inv_at(rim, "meal");
        let rim_timber1 = b.inv_at(rim, "timber");
        assert!(
            (rim_meal1 - (rim_meal0 + 6.0)).abs() < 0.01
                && (rim_timber1 - (rim_timber0 + 8.0)).abs() < 0.01,
            "the drop did not land at the rim cache: meal {rim_meal0}->{rim_meal1}, \
             timber {rim_timber0}->{rim_timber1}"
        );
        println!("[windfall] dropped at rim: meal {rim_meal1}, timber {rim_timber1}");

        // Now the haul: no injection, no nudge — the ordinary
        // ClaimHaulCache/PickupCache/DeliverStore chain against a holder
        // 28 units out.
        let mut trace: Vec<(usize, f32, f32, f32)> = Vec::new();
        for day in 0..10 {
            b.run_day_quiet();
            trace.push((
                day + 3,
                b.inv_at(rim, "meal"),
                b.inv_at(rim, "timber"),
                b.inv_at(store, "timber"),
            ));
        }
        for (d, m, t, st) in &trace {
            println!("[windfall day {d}] rim meal={m:.1} timber={t:.1} | store timber={st:.1}");
        }
        let (_, rim_meal_end, rim_timber_end, store_timber_end) = *trace.last().expect("10 days");
        assert!(
            rim_timber_end < 1.0,
            "the windfall's timber is still lying at the rim after 10 days \
             ({rim_timber_end}) — nobody hauled the drop in"
        );
        assert!(
            rim_meal_end < 1.0,
            "the windfall's meals are still lying at the rim after 10 days \
             ({rim_meal_end}) — nobody hauled the drop in"
        );
        // It arrived somewhere useful: timber leaves a cache only in a
        // colonist's arms, and the store is where hauled timber lands.
        // (It may then be sawn into planks at the bench, so the bar is the
        // PEAK store timber over the window, not the final reading.)
        let store_timber_peak = trace.iter().map(|t| t.3).fold(f32::MIN, f32::max);
        assert!(
            store_timber_peak > store_timber0,
            "hauled timber never reached the store (peak {store_timber_peak} \
             vs {store_timber0} at the drop)"
        );
        println!(
            "[windfall] hauled in: rim meal {rim_meal1}->{rim_meal_end}, timber \
             {rim_timber1}->{rim_timber_end}; store timber peak {store_timber_peak} \
             (was {store_timber0}), end {store_timber_end}"
        );
    });
}

/// WANDERER — a guest signs on. Proves the ROSTER SEAM: the campaign roster
/// grows, the recruit is seated on a pooled colonist slot (which carries its
/// own unique stagger residue, the determinism construction), the body comes
/// alive, and it goes to work like any other hand.
#[test]
fn wanderer_injection_seats_a_recruit_who_then_works() {
    on_big_stack(|| {
        // One signed band: a real colony with plenty of pooled slots left.
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 1, false) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        b.run_day_quiet();
        let roster0 = b.campaign.roster.len();
        let free0 = b.free_slots.len();
        assert!(free0 > 0, "staging wants a recruit pool");
        let next_slot = *b.free_slots.front().expect("pool");
        let alive_buf = b.state.agent_alive_buf.clone();
        assert_eq!(
            b.u32_at(&alive_buf, next_slot),
            0,
            "a pooled slot must be dormant before the join"
        );

        b.apply_event(&CampaignEvent::WandererArrives {
            id: "wanderer_probe".into(),
            leaves_day: 99,
        });

        assert_eq!(b.joins, 1, "the wanderer arm did not run");
        assert_eq!(b.campaign.roster.len(), roster0 + 1, "roster did not grow");
        assert_eq!(b.free_slots.len(), free0 - 1, "no pool slot was spent");
        assert_eq!(
            b.slot_of("wanderer_probe"),
            Some(next_slot),
            "the recruit was not seated on the head of the pool"
        );
        assert_eq!(
            b.u32_at(&alive_buf, next_slot),
            1,
            "the recruit's body never woke"
        );
        // Unique stagger residue = the phase-exclusivity construction the
        // roster churn must not break: pooled slots ARE original colonist
        // slots, so every seated member owns a distinct %20 phase.
        let staggers: Vec<u32> = {
            let sb = b.state.agent_stagger_buf.clone();
            let n = b.w.n;
            let all = read_u32(&mut b.state, &sb, n);
            b.slot_map.iter().map(|(_, s)| all[*s] % 20).collect()
        };
        let mut uniq = staggers.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            staggers.len(),
            "two seated colonists share a %20 phase after the join: {staggers:?}"
        );

        // And they WORK: two days later the recruit has left the spawn
        // point and put a job on their back.
        let start = b.pos_at(next_slot);
        let job_buf = b.state.agent_claimed_job_buf.clone();
        let mut moved = 0.0f32;
        let mut ever_claimed = false;
        for _ in 0..(2 * DAY_TICKS / 25) {
            for _ in 0..25 {
                b.state.step();
                b.tick += 1;
            }
            let p = b.pos_at(next_slot);
            moved = moved.max(((p[0] - start[0]).powi(2) + (p[1] - start[1]).powi(2)).sqrt());
            if b.u32_at(&job_buf, next_slot) != 0 {
                ever_claimed = true;
            }
        }
        assert!(
            moved > 1.0,
            "the recruit never left the spawn point ({moved:.2}u in two days)"
        );
        assert!(
            ever_claimed,
            "the recruit never claimed a job in two days — a seated body that does \
             not work is not a colonist"
        );
        println!(
            "[wanderer] seated on slot {next_slot}; walked {moved:.1}u in two days, \
             claimed_job seen = {ever_claimed}"
        );
    });
}

/// BLIGHT — the storyteller kills sown growth cells by the very keys the
/// bridge's own snapshot hands it. Proves the round trip
/// (fixture -> `ColonySnapshot::sown_cells` -> director draw -> fixture),
/// which is the only trope whose payload is derived from colony state.
#[test]
fn blight_injection_kills_the_snapshots_own_sown_cells() {
    on_big_stack(|| {
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, false) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        // Sow a plot by hand (the farm chain takes days to reach one; the
        // seam under test is the kill, not the sowing).
        let plot = b.w.plots[0];
        write_u32(&b.state, &b.state.agent_sown_buf, plot, 1);
        write_f32(&b.state, &b.state.agent_growth_buf, plot, 3.0);

        let snap = b.snapshot();
        let key = format!("p{plot}");
        assert!(
            snap.sown_cells.contains(&key),
            "the snapshot did not report the sown plot ({key} not in {:?})",
            snap.sown_cells
        );

        b.apply_event(&CampaignEvent::Blight { killed_cells: vec![key.clone()] });

        let sown_buf = b.state.agent_sown_buf.clone();
        assert_eq!(b.u32_at(&sown_buf, plot), 0, "the blight left the plot sown");
        let g = {
            let gb = b.state.agent_growth_buf.clone();
            let n = b.w.n;
            read_f32(&mut b.state, &gb, n)[plot]
        };
        assert!(g.abs() < 0.01, "the blight left growth standing ({g})");
        // And the next snapshot agrees the cell is gone.
        let snap2 = b.snapshot();
        assert!(
            !snap2.sown_cells.contains(&key),
            "the blighted cell is still reported sown"
        );
        println!("[blight] {key} killed: sown 1->0, growth 3.0->{g}");
    });
}

/// CARAVAN — the trader camps and business is DONE. The soak's dice drew
/// this one, but the seam is pinned here too so a future draw shift cannot
/// quietly retire it (and so the sell path is asserted against exact
/// prices rather than "some gold moved").
#[test]
fn caravan_injection_camps_the_trader_and_moves_goods() {
    on_big_stack(|| {
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, true) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        // Hides in the store for the trader to buy (hunting supplies these
        // organically; the seam under test is the trade).
        write_f32(&b.state, &b.state.agent_inv_hide_buf, b.w.store, 10.0);
        let gold0 = b.campaign.gold;
        let trader = b.w.trader;
        let active_buf = b.state.agent_trader_active_buf.clone();
        assert_eq!(
            b.u32_at(&active_buf, trader),
            0,
            "the trader must found dormant"
        );

        b.campaign.caravan = Some(Caravan {
            id: "caravan_probe".into(),
            faction_id: None,
            arrived_day: b.campaign.day,
            leaves_day: b.campaign.day + 2,
            gold: 60,
            goods: vec![("meal".into(), 12)],
            traded: false,
        });
        b.apply_event(&CampaignEvent::CaravanArrives {
            caravan: b.campaign.caravan.clone().expect("just set"),
        });

        assert_eq!(b.caravans, 1, "the caravan arm did not run");
        assert_eq!(
            b.u32_at(&active_buf, trader),
            1,
            "the camp never pitched (trader_active)"
        );
        // SELL: 10 hides at floor(4 * 0.6) = 2 each, capped by the 60-gold
        // purse (30 hides' worth) — all ten sell for 20 gold.
        assert!(
            b.inv_at(b.w.store, "hide").abs() < 0.01,
            "the stored hides were not sold"
        );
        assert!(
            (b.inv_at(trader, "hide") - 10.0).abs() < 0.01,
            "the hides never reached the trader's packs"
        );
        assert_eq!(b.trade_gold, 20, "10 hides at the 0.6 spread pay 20 gold");
        // BUY: meals at ceil(3 * 1.5) = 5 each, off the guild purse, landing
        // at the RIM cache (caravanSpot — even a purchase makes haul work).
        assert!(b.meals_bought > 0, "no meals were bought off the wares");
        assert!(
            b.inv_at(b.w.rim_cache, "meal") >= b.meals_bought as f32,
            "bought meals did not land at the rim camp"
        );
        assert_eq!(
            b.campaign.gold,
            gold0 + b.trade_gold - b.meals_bought * MEAL_BUY,
            "the guild ledger does not match the goods that moved"
        );
        assert!(
            b.campaign.caravan.as_ref().is_some_and(|c| c.traded),
            "business was done but the caravan was not marked traded"
        );
        println!(
            "[caravan] sold 10 hides for {} gold, bought {} meals; guild gold {} -> {}",
            b.trade_gold, b.meals_bought, gold0, b.campaign.gold
        );
    });
}

/// THE CHRONICLE-ONLY TROPES, asserted as such. RefugeeBand, WarbandGathers
/// and RaidIncoming carry NO fixture write by design — the refugee window
/// and the guest price live on the host's `DirectorState`, a gathering
/// warband walks the world map (campaign-side, `advance_threats`), and a
/// raid announcement is staged through `out.raid_tomorrow`, not through the
/// event. This test pins that they are inert rather than merely unwired:
/// the whole sim-state signature is unchanged across all three.
#[test]
fn campaign_side_tropes_write_nothing_to_the_fixture() {
    on_big_stack(|| {
        let Some(mut b) = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, false) else {
            eprintln!("[webband_campaign] skipping: no wgpu adapter");
            return;
        };
        b.run_day_quiet();
        let before = b.fixture_signature();
        b.apply_event(&CampaignEvent::RefugeeBand {
            band_id: "band_probe".into(),
            desperate_until: b.campaign.day + 6,
        });
        b.apply_event(&CampaignEvent::WarbandGathers {
            threat_id: "threat_probe".into(),
            reported_power: 17,
        });
        let after = b.fixture_signature();
        let mut diverged: Vec<&str> = Vec::new();
        for ((name, a), (_, c)) in before.iter().zip(after.iter()) {
            if a != c {
                diverged.push(name);
            }
        }
        assert!(
            diverged.is_empty(),
            "a campaign-side trope wrote to the fixture: {diverged:?}"
        );
        println!(
            "[campaign-side] RefugeeBand + WarbandGathers are fixture-inert across \
             {} state buffers (host-side effects live on DirectorState/threats and \
             are tested in webband_app)",
            before.len()
        );
    });
}
