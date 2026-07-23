//! webband_spine — S9: THE SPINE. One end-to-end test that declares the
//! Webband port hangs together.
//!
//! The plan's S9 line, executed literally:
//!   FOUND (seeded) -> DAYS WORKED -> RAID FOUGHT -> OUTCOME FOLDED ->
//!   CHRONICLE SANE -> THE SAME SEED REPLAYS BYTE-EQUAL,
//! plus a MID-CAMPAIGN SAVE/LOAD that the replay proves transparent.
//!
//! WHAT EACH PHASE PROVES (and why it is here rather than in a slice test):
//!
//!  1. FOUND — `webband_app::founding::new_founding(seed, 0, Village)` rolls a
//!     REAL cast/world/scenario (S7a's frozen draw order) and the bridge seats
//!     it on the `webband_colony` fixture: roster members take colonist bodies
//!     1:1, the rest of the pool goes dormant. Proves the host's generation and
//!     the fixture's bodies are the same colony, not two parallel fictions.
//!  2. DAYS WORKED — 16 campaign days through the SAME `Bridge` the S6 soak
//!     uses (600 ticks + snapshot + `dawn_fold` + trope write-back). Every
//!     layer of the port must show a measurable mark inside the window:
//!       * ECONOMY: founding blueprints raised by fixture work, chop/forage/
//!         cook/craft/haul minutes tallied, food cooked at the hearth.
//!       * MINDS (S1/S4): pair standing beliefs driven negative by organic
//!         brawls, and `repute` — the one gossiped attr — reaching observers
//!         far beyond the spatial neighbourhood (supper broadcast).
//!       * RAIDS (S5): a raid the STORYTELLER decided on (never injected here)
//!         musters on the fixture's own warning schedule, is fought in
//!         real time, and settles.
//!       * THE FOLD (S7b): the 24-step dawn order, pinned where it is
//!         observable — day increment first, the storyteller's accrual reading
//!         POST-provisioner/post-trade gold and POST-exodus roster, the fall
//!         check last.
//!  3. OUTCOME FOLDED — `resolve_raid` pays victory in the loot of the bodies
//!     the FIXTURE actually slew (gold_looted > 0 is combat evidence, not host
//!     math), or plunders on defeat; and the KO-not-death law holds across it
//!     (roster intact, every colonist body still alive).
//!  4. LEDGER + CHRONICLE COHERENCE — a per-day gold identity that must close
//!     EXACTLY every morning (provisioner spend / trade income / rent / raid
//!     loot / caravan trade), renown moving only on raid days and by exactly
//!     the tier formula, and a chronicle whose entries are day-stamped,
//!     ordered, capped and non-empty. Plus the state-coherence sweep: roster ==
//!     seated bodies, pool slots dormant, no orphaned raid.
//!  5. SAVE/LOAD MID-CAMPAIGN — at day 9 the Campaign is serialized to disk,
//!     read back, asserted FULLY EQUAL (`PartialEq` over every f64 — this is
//!     what serde_json's `float_roundtrip` feature buys), a doctored version is
//!     REJECTED (the found-anew discard rule), and the reloaded value replaces
//!     the live one for the rest of the run.
//!  6. REPLAY BYTE-EQUAL — the whole campaign runs a second time from the same
//!     seeds WITHOUT the save/load, and the digest (fixture state buffers +
//!     serialized Campaign + host event log) must be IDENTICAL. One comparison
//!     therefore proves both determinism and that the round-trip changed
//!     nothing. A digest file under `target/webband_spine/` extends the pin
//!     ACROSS PROCESSES (the S6b idiom).
//!
//! DETERMINISM DISCIPLINE (S5 finding 2): the digest pins SIM STATE — agent
//! fields, pair-belief domains, positions — never the `tally_*`/`count_*`
//! views, which ride the engine's delayed lossy fold window and are not
//! run-to-run stable under combat-era event volume. Views are used only as
//! ">0" evidence, never as pinned numbers.
//!
//! RUNTIME (measured on this host, debug build, one wgpu adapter):
//! ~4-6 minutes for the whole binary — two 16-day campaigns at ~4.5 s of GPU
//! time per 600-tick colony day, plus two runtime constructions. It is meant
//! to be usable as the port's regression gate.
//!
//! windows-gnu note: test bodies run on a spawned 64 MiB thread (the S1
//! wgpu-init stack finding).

#![allow(non_snake_case)]

// S12: the bridge is now the `webband_bridge` LIBRARY crate (a dev-dep of
// this crate), so `crates/webband_play` drives the same code. Was
// `mod webband_bridge;` over `tests/webband_bridge/mod.rs`.
use webband_bridge::*;

use std::path::PathBuf;

use webband_app::campaign::{load_campaign, save_campaign, CampaignError};
use webband_app::defs::colony_wealth;
use webband_app::director::trope_cost;
use webband_app::scenario::scenario_spec;

/// Days of campaign. 16 is the smallest window on this seed that carries the
/// whole spine: the storyteller's first organic raid (staged day 7, fought
/// day 8) plus a caravan and a festival, i.e. three distinct tropes, while
/// keeping the double run inside the regression-gate budget.
const SPINE_DAYS: usize = 16;
/// The campaign day after which the save/load round trip happens (run A only).
const SAVE_AFTER_DAY: usize = 9;

// ---------------------------------------------------------------------------
// Evidence gathered by one full run.

#[derive(Debug, Clone, PartialEq)]
struct SpineEvidence {
    digest: String,
    fixture_hash: u64,
    campaign_hash: u64,
    log_hash: u64,
    day: i64,
    gold: i64,
    renown: i64,
    roster: usize,
    built_non_wall: usize,
    raids: Vec<RaidRecord>,
    standing_sum: f64,
    repute_cells: usize,
    repute_max_observers: usize,
    tallies: Vec<(&'static str, f32)>,
    chronicle_len: usize,
    accrual_days_checked: usize,
    ledger_days_checked: usize,
}

fn sum_view_f32(b: &mut Bridge, buf: wgpu::Buffer) -> f32 {
    let n = b.w.n;
    read_f32(&mut b.state, &buf, n).iter().sum()
}

/// Sum a pair-domain (n x n) f32 belief buffer in f64 — magnitudes reach 1e7
/// and an f32 accumulator would swallow the sign evidence.
fn pair_sum(b: &mut Bridge, buf: wgpu::Buffer) -> f64 {
    let n = b.w.n;
    read_f32(&mut b.state, &buf, n * n)
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

// ---------------------------------------------------------------------------
// THE RUN.

fn spine_run(label: &str, save_load_after: Option<usize>) -> Option<SpineEvidence> {
    // ---- 1. FOUND -------------------------------------------------------
    // Village start, the same staging as the S6 soak (sign every band that
    // fits the 20 colonist bodies) so the spine's first 16 days ARE the
    // soak's first 16 days — the two tests cross-validate.
    let mut b = Bridge::new(ScenarioId::Village, CAMPAIGN_SEED, 99, true)?;
    let founding_name = b.campaign.founding.name.clone();
    let roster0 = b.campaign.roster.len();
    let gold0 = b.campaign.gold;
    let renown0 = b.campaign.renown;
    println!(
        "[spine/{label}] FOUND: \"{founding_name}\" — cast {} companions in {} bands, \
         world {} landmarks, scenario Village; roster {roster0} seated on colonist \
         bodies ({} pooled), gold {gold0}, renown {renown0}",
        b.campaign.founding.cast.companions.len(),
        b.campaign.founding.cast.bands.len(),
        b.campaign.founding.world.landmarks.len(),
        b.free_slots.len(),
    );
    assert!(roster0 >= 8, "spine staging wants a real colony (roster {roster0})");
    assert!(!b.free_slots.is_empty(), "spine staging wants a recruit pool");
    assert_eq!(
        b.slot_map.len(),
        roster0,
        "every roster member must own a colonist body at founding"
    );
    // The generated cast's own constraints reach the bridge intact (S7a
    // asserts them at generation; here they must survive seating).
    {
        let mut prefixes: Vec<String> = b
            .campaign
            .founding
            .cast
            .companions
            .iter()
            .map(|c| c.name.chars().take(4).collect::<String>().to_lowercase())
            .collect();
        let n_all = prefixes.len();
        prefixes.sort();
        prefixes.dedup();
        assert_eq!(
            prefixes.len(),
            n_all,
            "generated cast lost its distinct-4-char-prefix constraint"
        );
    }
    let scenario = scenario_spec(b.campaign.founding.scenario);
    assert_eq!(scenario.rent_per_day, 0, "village start pays no rent");

    // ---- 2. THE DAYS ----------------------------------------------------
    let mut ledger_days = 0usize;
    let mut accrual_days = 0usize;
    let mut provisioned_days = 0usize;
    let mut save_loaded = false;
    // (day, delta) — reported; the per-day assertion is inside the loop.
    let mut renown_moves: Vec<(i64, i64)> = Vec::new();

    for d in 0..SPINE_DAYS {
        let gold_before = b.campaign.gold;
        let renown_before = b.campaign.renown;
        let points_before = b.campaign.director.points;
        let raid_gold_before = b.raid_gold;
        let trade_gold_before = b.trade_gold;
        let meals_before = b.meals_bought;
        let chronicle_before = b.campaign.chronicle.len();
        let raid_log_before = b.raid_log.len();

        let out = b.run_day();

        // -- 4a. THE GOLD LEDGER CLOSES, EVERY MORNING --------------------
        // Every coin that moved has a named cause: a raid the fixture won,
        // caravan business, the dawn provisioner's bread, the local trade.
        let d_raid = b.raid_gold - raid_gold_before;
        let d_trade = b.trade_gold - trade_gold_before;
        let d_meals = b.meals_bought - meals_before;
        let spent = out.provision.as_ref().map_or(0, |p| p.gold_spent);
        let expect_gold =
            gold_before + d_raid + d_trade - d_meals * MEAL_BUY - spent + out.trade_income;
        assert_eq!(
            b.campaign.gold, expect_gold,
            "day {}: the gold ledger does not close — before {gold_before}, raid \
             +{d_raid}, caravan +{d_trade}, meals -{}, provisioner -{spent}, trade \
             +{}; expected {expect_gold}, got {}",
            b.campaign.day,
            d_meals * MEAL_BUY,
            out.trade_income,
            b.campaign.gold
        );
        ledger_days += 1;
        if spent > 0 {
            provisioned_days += 1;
        }
        // -- 4c. RENOWN MOVES ONLY FOR REASONS ----------------------------
        // The only renown mover wired today is a resolved raid: victory pays
        // 4 + 2*tier, defeat costs 4 (6 when undefended) floored at 0
        // (`apply_plunder`'s own `(renown - loss).max(0)`). Asserted BOTH
        // ways — an unexplained move fails just as loudly as a wrong one.
        let resolved_today: Vec<RaidRecord> = b.raid_log[raid_log_before..].to_vec();
        let expect_renown: i64 = resolved_today
            .iter()
            .map(|r| {
                if r.victory {
                    4 + 2 * r.tier
                } else {
                    -renown_before.min(4)
                }
            })
            .sum();
        assert_eq!(
            b.campaign.renown - renown_before,
            expect_renown,
            "day {}: renown moved {} but the raids that resolved this morning \
             ({resolved_today:?}) pay {expect_renown}",
            b.campaign.day,
            b.campaign.renown - renown_before
        );
        if b.campaign.renown != renown_before {
            renown_moves.push((b.campaign.day, b.campaign.renown - renown_before));
        }

        // -- 2d/4b. THE DAWN FOLD'S ORDER, where it is observable ---------
        // The storyteller (step 22) reads wealth built from the gold the
        // provisioner (5), trade income (16) and rent (17) already moved,
        // and the roster the exodus (13) already thinned. Recomputing its
        // accrual from POST-fold gold/roster and the snapshot the fold was
        // handed must reproduce `director.points` exactly.
        let snap = b.last_snap.clone().expect("run_day stashes the snapshot");
        let wealth = colony_wealth(b.campaign.gold, b.campaign.roster.len(), &snap.inventory);
        let mood = if snap.members.is_empty() {
            0.0
        } else {
            snap.members.iter().map(|m| m.mood).sum::<f64>() / snap.members.len() as f64
        };
        let accrual = 2
            + (b.campaign.roster.len() as i64 + 1) / 2
            + wealth.div_euclid(800)
            + i64::from(mood > 60.0) * 2;
        let cost = out.event.as_ref().map_or(0, |e| trope_cost(e.kind));
        let expect_points = (points_before + accrual).min(120) - cost;
        assert_eq!(
            b.campaign.director.points, expect_points,
            "day {}: the storyteller's accrual does not match the POST-fold colony \
             (points {points_before} + accrual {accrual} - cost {cost}); the dawn \
             fold's step order is wrong or the accrual formula drifted",
            b.campaign.day
        );
        accrual_days += 1;

        // Step 1 (day += 1) really is first: everything the FOLD chronicled
        // this morning is stamped with the NEW day. The one legitimate
        // exception is the raid the bridge resolves BEFORE the fold — a
        // storm that broke yesterday is written into yesterday, which is
        // itself the order being asserted.
        let raid_resolved_today = b.raid_log.len() > raid_log_before;
        for e in &b.campaign.chronicle[chronicle_before..] {
            if e.day == b.campaign.day {
                continue;
            }
            assert!(
                raid_resolved_today && e.day == b.campaign.day - 1,
                "a chronicle entry written during the day-{} fold is stamped day {} \
                 (no raid resolved before the fold this morning)",
                b.campaign.day,
                e.day
            );
        }
        // Step 24 (the fall check) really is last.
        assert_eq!(
            out.fell,
            b.campaign.roster.is_empty(),
            "day {}: the fall check disagrees with the roster",
            b.campaign.day
        );

        if let Some(e) = &out.event {
            println!(
                "[spine/{label} day {}] STORYTELLER: {:?} — {} (points {} -> {})",
                b.campaign.day, e.kind, e.title, points_before, b.campaign.director.points
            );
        }

        // -- 5. SAVE / LOAD MID-CAMPAIGN ----------------------------------
        if save_load_after == Some(d) {
            save_loaded = true;
            let dir = std::path::Path::new("target/webband_spine");
            std::fs::create_dir_all(dir).expect("save dir");
            let path: PathBuf = dir.join("spine_midcampaign.json");
            save_campaign(&b.campaign, &path).expect("campaign saves");
            let loaded = load_campaign(&path).expect("campaign loads");
            assert_eq!(
                loaded, b.campaign,
                "the mid-campaign save/load round trip is LOSSY — every field, \
                 including the founding's f64 world coordinates and the live rng \
                 state, must survive (serde_json's float_roundtrip feature)"
            );
            // The discard rule: a save from another shape founds anew.
            let doctored = dir.join("spine_midcampaign_v999.json");
            let raw = std::fs::read_to_string(&path).expect("read save");
            let mut v: serde_json::Value = serde_json::from_str(&raw).expect("save is json");
            v["version"] = serde_json::json!(999);
            std::fs::write(&doctored, v.to_string()).expect("write doctored save");
            match load_campaign(&doctored) {
                Err(CampaignError::Version { found, want }) => {
                    assert_eq!((found, want), (999, 1), "version probe reported wrong ids");
                }
                other => panic!("a foreign save version must be REFUSED, got {other:?}"),
            }
            let bytes = raw.len();
            println!(
                "[spine/{label} day {}] SAVE/LOAD: {bytes} bytes round-tripped equal; \
                 a version-999 save is refused; the RELOADED campaign now drives the \
                 rest of the run",
                b.campaign.day
            );
            // Continue on the value that came off disk.
            b.campaign = loaded;
        }

        if out.fell {
            panic!(
                "the colony fell on day {} — the spine's 16-day window is meant to \
                 be survivable staging",
                b.campaign.day
            );
        }
    }

    // ---- 3. THE RAID, FOUGHT AND FOLDED ---------------------------------
    assert!(
        !b.raid_log.is_empty(),
        "the storyteller never brought a raid to the fences in {SPINE_DAYS} days — \
         the spine cannot prove the combat layer (raids staged: {})",
        b.raids_staged
    );
    assert_eq!(
        b.raids_won + b.raids_lost,
        b.raids_staged,
        "a staged raid never resolved"
    );
    assert!(
        b.event_kinds.iter().any(|k| k == "Raid" || k == "Warband" || k == "Feud"
            || k == "CauseRaid"),
        "the raid was not authored by the storyteller — the spine must never \
         inject one: {:?}",
        b.event_kinds
    );
    for r in &b.raid_log {
        if r.victory {
            assert!(
                r.gold_looted > 0,
                "day {}: a won raid paid no loot — nothing was SLAIN, so no combat \
                 actually ran in the fixture",
                r.day
            );
        } else {
            assert!(
                r.plunder_taken.unwrap_or(0) > 0,
                "day {}: a lost raid stripped nothing — plunder-not-death means the \
                 warband takes STOCK",
                r.day
            );
        }
    }
    // KO, NEVER DEATH: a raid costs the colony stock and standing, never a
    // companion. Every seated body is still alive and still on the roster.
    {
        let n = b.w.n;
        let alive = {
            let buf = b.state.agent_alive_buf.clone();
            read_u32(&mut b.state, &buf, n)
        };
        for (id, slot) in b.slot_map.clone() {
            assert_eq!(
                alive[slot], 1,
                "colonist {id} (slot {slot}) is DEAD — companions are KO'd, never killed"
            );
        }
    }

    // ---- 2a. ECONOMY ------------------------------------------------------
    let n = b.w.n;
    let built_non_wall = {
        let built = {
            let buf = b.state.agent_built_buf.clone();
            read_u32(&mut b.state, &buf, n)
        };
        let types = {
            let buf = b.state.agent_creature_type_buf.clone();
            read_u32(&mut b.state, &buf, n)
        };
        (0..n).filter(|&i| built[i] == 1 && types[i] != CT_WALL).count()
    };
    assert!(
        built_non_wall >= 3,
        "only {built_non_wall} non-wall structures stand after {SPINE_DAYS} campaign \
         days — the colony fixture did no building work under the campaign clock"
    );
    // Work tallies are the engine's LOSSY fold views (S5 finding 2): used
    // here as ">0 happened" evidence only, never as pinned numbers.
    let tally_bufs: [(&'static str, wgpu::Buffer); 7] = [
        ("chop", b.state.view_storage_tally_chop_primary_buf.clone()),
        ("forage", b.state.view_storage_tally_forage_primary_buf.clone()),
        ("hunt", b.state.view_storage_tally_hunt_primary_buf.clone()),
        ("build", b.state.view_storage_tally_build_primary_buf.clone()),
        ("cook", b.state.view_storage_tally_cook_primary_buf.clone()),
        ("craft", b.state.view_storage_tally_craft_primary_buf.clone()),
        ("eat", b.state.view_storage_tally_eat_primary_buf.clone()),
    ];
    let mut tallies: Vec<(&'static str, f32)> = Vec::new();
    for (name, buf) in tally_bufs {
        let v = sum_view_f32(&mut b, buf);
        tallies.push((name, v));
    }
    for (name, v) in &tallies {
        assert!(
            *v > 0.0,
            "no {name} work happened in {SPINE_DAYS} days — an entire arm of the \
             economy is dead: {tallies:?}"
        );
    }

    // ---- 2b. MINDS --------------------------------------------------------
    // Pair standing: organic jostling sours pairs (S4: SOUR_VICTIM x 0.6 per
    // brawl, decayed at the converted tuning.ts rate).
    let standing_buf = b.state.view_storage_standing_brawl_primary_buf.clone();
    let standing_sum = pair_sum(&mut b, standing_buf);
    assert!(
        standing_sum < 0.0,
        "no pair belief moved in {SPINE_DAYS} days (standing_brawl domain sum \
         {standing_sum}) — the minds layer is dead under the campaign clock"
    );
    // Gossip: `repute` is the ONE attr belief that spreads (standing and
    // grudges never gossip — S4's law). A first-hand deed reaches only the
    // spatial neighbourhood (radius 1 cell); the supper broadcast reaches
    // the colony. So "many observers hold a belief about one subject" is
    // gossip evidence, not proximity.
    let (repute_cells, repute_max_observers) = {
        let rep = {
            let buf = b.state.view_storage_repute_primary_buf.clone();
            read_f32(&mut b.state, &buf, n * n)
        };
        let mut cells = 0usize;
        let mut per_subject = vec![0usize; n];
        for observer in 0..n {
            for subject in 0..n {
                if rep[observer * n + subject] != 0.0 {
                    cells += 1;
                    per_subject[subject] += 1;
                }
            }
        }
        (cells, per_subject.into_iter().max().unwrap_or(0))
    };
    assert!(
        repute_cells > 0,
        "no reputation belief was ever written — prowess is not being witnessed"
    );
    assert!(
        repute_max_observers >= 5,
        "the most-known colonist is known to only {repute_max_observers} observers — \
         that is first-hand witnessing, not SUPPER GOSSIP (the merge broadcast)"
    );

    // (4c's renown pin is per-day, inside the loop above.)
    assert!(
        !renown_moves.is_empty(),
        "renown never moved in {SPINE_DAYS} days despite {} resolved raids",
        b.raid_log.len()
    );

    // ---- 4d. CHRONICLE SANITY -------------------------------------------
    let chron = b.campaign.chronicle.clone();
    assert!(!chron.is_empty(), "the campaign wrote no chronicle at all");
    assert!(chron.len() <= 200, "the chronicle is uncapped ({} entries)", chron.len());
    let mut last_day = 0i64;
    for e in &chron {
        assert!(!e.text.trim().is_empty(), "an empty chronicle entry on day {}", e.day);
        assert!(
            e.day >= 1 && e.day <= b.campaign.day,
            "chronicle entry dated day {} outside the campaign (1..={})",
            e.day,
            b.campaign.day
        );
        assert!(e.day >= last_day, "the chronicle is out of order at day {}", e.day);
        last_day = e.day;
    }
    let raid_lines = chron
        .iter()
        .filter(|e| matches!(e.kind, webband_app::campaign::ChronicleKind::Raid))
        .count();
    assert!(
        raid_lines >= b.raid_log.len(),
        "{} raids resolved but only {raid_lines} raid chronicle lines",
        b.raid_log.len()
    );

    // ---- 4e. NO ORPHANED STATE ------------------------------------------
    assert_eq!(
        b.campaign.roster.len(),
        b.slot_map.len(),
        "the campaign roster and the seated bodies disagree"
    );
    for id in &b.campaign.roster {
        assert!(
            b.slot_of(id).is_some(),
            "roster member {id} holds no colonist body"
        );
    }
    {
        let mut slots: Vec<usize> = b.slot_map.iter().map(|(_, s)| *s).collect();
        let seated = slots.len();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), seated, "two roster members share one body");
        let mut all: Vec<usize> = slots.clone();
        all.extend(b.free_slots.iter().copied());
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            b.w.colonists.len(),
            "seated + pooled slots do not account for every colonist body"
        );
        let alive = {
            let buf = b.state.agent_alive_buf.clone();
            read_u32(&mut b.state, &buf, n)
        };
        for &s in &b.free_slots {
            assert_eq!(alive[s], 0, "a pooled body (slot {s}) is walking around");
        }
    }
    assert_eq!(
        b.campaign.raid.is_some(),
        b.staged.is_some(),
        "an announced raid has no staged cohort (or a cohort stands with no raid)"
    );
    if save_load_after.is_some() {
        assert!(save_loaded, "the save/load phase never ran");
    }

    // ---- 6. THE DIGEST ---------------------------------------------------
    let sig = b.fixture_signature();
    let fixture_hash = sig_hash(&sig);
    let campaign_json = serde_json::to_string(&b.campaign).expect("campaign serializes");
    let campaign_hash = fnv1a(campaign_json.as_bytes());
    let log_hash = fnv1a(b.log.join("\n").as_bytes());
    let digest = format!(
        "webband_spine digest v1\nfixture={fixture_hash:#018x}\ncampaign={campaign_hash:#018x}\nlog={log_hash:#018x}\ndays={SPINE_DAYS} day={} gold={} renown={} roster={} built={built_non_wall} raids={}/{} chronicle={}\n",
        b.campaign.day,
        b.campaign.gold,
        b.campaign.renown,
        b.campaign.roster.len(),
        b.raids_won,
        b.raids_staged,
        chron.len(),
    );

    println!(
        "[spine/{label}] {SPINE_DAYS} days: gold {gold0}->{} renown {renown0}->{} roster \
         {roster0}->{} | built(non-wall)={built_non_wall} | raids staged={} won={} lost={} \
         | caravans={} trade_gold={} meals_bought={} | provisioned on {provisioned_days} \
         mornings | joins={} departures={} | ledger closed {ledger_days}/{SPINE_DAYS} days, \
         accrual pinned {accrual_days}/{SPINE_DAYS} | chronicle={} entries",
        b.campaign.gold,
        b.campaign.renown,
        b.campaign.roster.len(),
        b.raids_staged,
        b.raids_won,
        b.raids_lost,
        b.caravans,
        b.trade_gold,
        b.meals_bought,
        b.joins,
        b.departures.len(),
        chron.len(),
    );
    println!(
        "[spine/{label}] raids: {:?}\n[spine/{label}] tallies (lossy views, evidence \
         only): {tallies:?}\n[spine/{label}] minds: standing_brawl domain sum \
         {standing_sum:.1}, repute cells {repute_cells}, best-known colonist has \
         {repute_max_observers} observers",
        b.raid_log
    );
    for line in &b.log {
        println!("[spine/{label} log] {line}");
    }

    Some(SpineEvidence {
        digest,
        fixture_hash,
        campaign_hash,
        log_hash,
        day: b.campaign.day,
        gold: b.campaign.gold,
        renown: b.campaign.renown,
        roster: b.campaign.roster.len(),
        built_non_wall,
        raids: b.raid_log.clone(),
        standing_sum,
        repute_cells,
        repute_max_observers,
        tallies,
        chronicle_len: chron.len(),
        accrual_days_checked: accrual_days,
        ledger_days_checked: ledger_days,
    })
}

// ---------------------------------------------------------------------------
// THE SPINE.

#[test]
fn spine_found_work_raid_fold_save_replay() {
    on_big_stack(|| {
        // RUN A — the campaign WITH the mid-campaign save/load.
        let Some(a) = spine_run("A", Some(SAVE_AFTER_DAY)) else {
            eprintln!("[webband_spine] skipping: no wgpu adapter");
            return;
        };
        // RUN B — the same seeds, no save/load, a fresh runtime.
        let b = spine_run("B", None).expect("run A constructed a runtime, so must run B");

        // ---- 6. REPLAY BYTE-EQUAL ---------------------------------------
        // One comparison, two properties: the port replays bit-for-bit from
        // the same seed, AND the mid-campaign save/load round trip (run A's
        // only difference) changed nothing downstream of it.
        assert_eq!(
            a.fixture_hash, b.fixture_hash,
            "REPLAY BROKEN: the fixture's sim-state buffers differ between two \
             identical-seed campaigns (A did a mid-campaign save/load, B did not)"
        );
        assert_eq!(
            a.campaign_hash, b.campaign_hash,
            "REPLAY BROKEN: the serialized Campaign differs between runs"
        );
        assert_eq!(
            a.log_hash, b.log_hash,
            "REPLAY BROKEN: the host event log differs between runs"
        );
        assert_eq!(a, b, "REPLAY BROKEN: some measured evidence differs between runs");
        println!(
            "[spine] REPLAY: bit-equal across two runs — fixture={:#018x} \
             campaign={:#018x} log={:#018x} (run A round-tripped its Campaign \
             through disk at day {SAVE_AFTER_DAY}; run B did not)",
            a.fixture_hash, a.campaign_hash, a.log_hash
        );

        // ---- the CROSS-PROCESS pin (the S6b idiom) ----------------------
        println!("[spine digest]\n{}", a.digest);
        let dir = std::path::Path::new("target/webband_spine");
        std::fs::create_dir_all(dir).expect("digest dir");
        let path = dir.join("spine_digest.txt");
        if path.exists() {
            let prior = std::fs::read_to_string(&path).expect("read prior digest");
            assert_eq!(
                prior, a.digest,
                "CROSS-PROCESS DETERMINISM BROKEN: this run's spine digest differs \
                 from the recorded one (delete {path:?} only if the fixture, the \
                 bridge or webband_app legitimately changed)"
            );
            println!("[spine] cross-process determinism pin HELD against {path:?}");
        } else {
            std::fs::write(&path, &a.digest).expect("write digest");
            println!(
                "[spine] digest recorded to {path:?} — run the test again for the \
                 cross-process pin"
            );
        }
    });
}
